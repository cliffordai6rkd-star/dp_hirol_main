"""Extend the pinned70750 cache, audit it, then recalibrate only the existing MLP."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import time

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from student_dual_path_common import load_inputs, datasets_from_reference, to_device, tensor_hash, write_json, model_for
from paired_response_full_common import file_hash, episode_ids, PHASES
from new_teacher_response_common import split_details, summarize_zero, balanced_batches
from diffusion_policy.model.vision.render_net_attention import zero_wrench_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['root', 'study', 'experiment']:
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).parent/'config/new_teacher_shared_response.yaml')
    p.add_argument('--resume-cache', action='store_true', help='Complete an unconsumed cache after a failed precision audit')
    a = p.parse_args()
    config = OmegaConf.load(a.config)
    a.experiment.mkdir(parents=True, exist_ok=a.resume_cache)
    if not a.resume_cache:
        (a.experiment/'inputs').symlink_to((a.study/'inputs').resolve(), target_is_directory=True)
    OmegaConf.save(config, a.experiment/'config.yaml')
    torch.set_num_threads(4)
    torch.manual_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    inputs = a.experiment/'inputs'
    assert file_hash(inputs/'pinned_teacher_ema.ckpt') == config.teacher_sha256
    teacher, teacher_cfg, initial = load_inputs(inputs, device)
    assert tensor_hash(teacher.state_dict()) == config.teacher_state_sha256
    compact = json.loads((a.study/'manifest.json').read_text())
    assert compact['teacher_state_sha256'] == config.teacher_state_sha256
    train, val, identities = datasets_from_reference(inputs, a.root/'decoded_cuda')
    datasets = dict(train=train, validation=val)
    split = split_details(datasets, compact['teacher_validation_episodes'])
    assert set(split['validation']['episodes']) == set(config.external_episodes) | set(config.selection_episodes)
    write_json(a.experiment/'episode_split.json', split)
    plan = balanced_batches(train.phase_labels, episode_ids(train), config.rgb.maximum_steps, config.rgb.batch_size, config.seed)
    np.save(a.experiment/'train_batches.npy', plan)
    write_json(a.experiment/'sampling.json', dict(plan_sha256=file_hash(a.experiment/'train_batches.npy'),
        batch_size=config.rgb.batch_size, phase_mix='Equal Contact/Free; cycle episodes uniformly within each phase',
        budgets={str(steps): dict(unique_windows=len(np.unique(plan[:steps])), episodes=sorted(set(episode_ids(train)[plan[:steps].flatten()].tolist())),
            phase_counts=np.bincount(train.phase_labels[plan[:steps].flatten()], minlength=3).tolist()) for steps in [1000, 3000]}))
    provenance = json.loads((a.study/'teacher_provenance.json').read_text())
    assert provenance['optimizer_step'] == config.teacher_step
    write_json(a.experiment/'teacher_provenance.json', provenance)
    enc = teacher.obs_encoder
    started = time.monotonic()
    cache = a.experiment/'paired_cache'
    if a.resume_cache:
        assert not (a.experiment/'adapter').exists(), 'Never mutate a cache consumed by training'
        repair_compact_precision(a, config, teacher, datasets, cache)
        calibrate(a, config, teacher, initial, cache, split, started)
        return
    cache.mkdir()
    manifest = dict(teacher_state_sha256=config.teacher_state_sha256, teacher_checkpoint_sha256=config.teacher_sha256,
        splits=identities, preprocessing='Unchanged historical CUDA-decoded RGB; clear images; physical zero before original normalizer',
        targets='FP32 teacher final context then FP16 storage; same readout for real/zero',
        visual='BF16 DINO + FP32 positional embeddings; original pre-adapter path; FP32 storage',
        reused_from=str(a.study), audit=[], arrays={}, precision=dict(matmul_tf32=False, cudnn_tf32=False))
    historical = torch.load(inputs/'reference_samples.pt', map_location='cpu', weights_only=False)
    for ref in historical:
        obs = datasets[ref['split']][ref['dataset_index']]['obs']
        for key, digest in ref['image_sha256'].items():
            assert hashlib.sha256(obs[key].unsqueeze(0).numpy().tobytes()).hexdigest() == digest
    for name, ds in datasets.items():
        dest = cache/name
        dest.mkdir()
        compact_rows = json.loads((a.study/name/'samples.json').read_text())
        mapping = {r['index']: i for i, r in enumerate(compact_rows)}
        # Checkpoint equality alone does not guarantee backend precision equality.
        # Unannotated historical contexts are not mixed into a canonical cache.
        if compact.get('precision') != manifest['precision']:
            mapping = {}
        old = {k: np.load(a.study/name/(k+'.npy'), mmap_mode='r') for k in ['target', 'zero']}
        arrays, offset, reused = {}, 0, 0
        episodes = episode_ids(ds)
        np.save(dest/'episodes.npy', episodes)
        np.save(dest/'phases.npy', ds.phase_labels)
        loader = DataLoader(ds, batch_size=config.cache_batch_size, num_workers=config.workers, pin_memory=True)
        for raw in loader:
            obs = to_device(raw['obs'], device)
            count = len(next(iter(obs.values())))
            ids = list(range(offset, offset+count))
            missing = [i for i, index in enumerate(ids) if index not in mapping]
            values = {}
            if missing:
                subset = {k: v[missing] for k, v in obs.items()}
                captured = []
                original = enc._encode_dino
                def capture(images, **kw):
                    result = original(images, **kw)
                    captured.append(result)
                    return result
                with torch.no_grad(), torch.autocast('cuda', enabled=False):
                    enc._encode_dino = capture
                    try:
                        target = teacher._encode_context(subset, apply_curriculum_mask=False)
                    finally:
                        enc._encode_dino = original
                    assert len(captured) == 1
                    enc._encode_dino = lambda images, **kw: captured[0]
                    try:
                        zero = teacher._encode_context(zero_wrench_reference(teacher, subset), apply_curriculum_mask=False)
                    finally:
                        enc._encode_dino = original
                for key, tensor in [('target', target), ('zero', zero)]:
                    values[key] = np.empty((count, *tensor.shape[1:]), dtype=np.float16)
                    values[key][missing] = tensor.float().cpu().numpy().astype(np.float16)
            else:
                for key in old:
                    values[key] = np.empty((count, *old[key].shape[1:]), dtype=np.float16)
            for position, index in enumerate(ids):
                if index in mapping:
                    for key in old:
                        values[key][position] = old[key][mapping[index]]
                    reused += 1
            images, _, _, _ = teacher.prepare_context_inputs_with_future_contact(obs)
            seen = []
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                enc.encode_image_context(images, context_adapter=lambda x: (seen.append(x), x)[1])
            values['visual'] = seen[0].float().cpu().numpy()
            for key, array in values.items():
                assert np.isfinite(array).all()
                if key not in arrays:
                    arrays[key] = np.lib.format.open_memmap(dest/(key+'.npy'), mode='w+', dtype=array.dtype,
                        shape=(len(ds), *array.shape[1:]))
                arrays[key][offset:offset+count] = array
            offset += count
            if offset % 1024 < count or offset == len(ds):
                print(json.dumps(dict(event='cache', split=name, done=offset, total=len(ds), reused=reused,
                    elapsed_seconds=time.monotonic()-started)), flush=True)
        for array in arrays.values():
            array.flush()
        # Recompute distributed real/physical-zero probes, including every validation episode.
        probes = []
        for ep in np.unique(episodes):
            for phase in range(3):
                candidates = np.flatnonzero((episodes == ep) & (ds.phase_labels == phase))
                if len(candidates) and (name == 'validation' or ep in [0, 20, 40, 61]):
                    probes.append(int(candidates[len(candidates)//2]))
        for index in probes:
            obs = to_device(next(iter(DataLoader(torch.utils.data.Subset(ds, [index]), batch_size=1)))['obs'], device)
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                t = teacher._encode_context(obs, apply_curriculum_mask=False).float()
                z = teacher._encode_context(zero_wrench_reference(teacher, obs), apply_curriculum_mask=False).float()
            errors = {key: float(np.mean((value.cpu().numpy()[0]-arrays[key][index].astype(np.float32))**2))
                      for key, value in [('target', t), ('zero', z)]}
            assert max(errors.values()) < 1e-5, (name, index, errors)
            manifest['audit'].append(dict(split=name, index=index, episode=int(episodes[index]), phase=int(ds.phase_labels[index]),
                target_errors=errors, image_sha256={k: hashlib.sha256(obs[k].cpu().numpy().tobytes()).hexdigest() for k in enc.image_keys}))
        manifest['arrays'][name] = dict(reused_targets=reused, computed_targets=len(ds)-reused,
            target_sha256=file_hash(dest/'target.npy'), zero_sha256=file_hash(dest/'zero.npy'))
        write_json(dest/'metadata.json', dict(complete=True, **identities[name]))
        write_json(cache/'manifest.json', manifest)
    assert tensor_hash(teacher.state_dict()) == config.teacher_state_sha256
    print('CACHE AUDIT COMPLETE', flush=True)
    calibrate(a, config, teacher, initial, cache, split, started)


def repair_compact_precision(a, config, teacher, datasets, cache):
    """Replace only reused TF32-context rows in this unfinished NEW cache.

    The historical compact cache remains untouched. Its tensors are numerically
    different from our FP32/no-TF32 path, so retaining them would mix targets.
    All other full-cache targets and all pre-adapter features are retained.
    """
    manifest = json.loads((cache/'manifest.json').read_text())
    assert manifest['teacher_checkpoint_sha256'] == config.teacher_sha256
    manifest['precision_repair'] = dict(reason='Historical compact contexts used a different TF32 backend; direct probe524 exceeded1e-5 MSE. Recompute ONLY those927 reused rows in FP32 with TF32 disabled.',
        historical_cache_unchanged=True, full_cache_features_reused=True, tf32_matmul=False, tf32_cudnn=False)
    manifest['precision'] = dict(matmul_tf32=False, cudnn_tf32=False)
    for split, ds in datasets.items():
        rows = json.loads((a.study/split/'samples.json').read_text())
        indices = [r['index'] for r in rows]
        arrays = {k: np.load(cache/split/(k+'.npy'), mmap_mode='r+') for k in ['target', 'zero']}
        assert all(v.shape[0] == len(ds) for v in arrays.values())
        enc = teacher.obs_encoder
        for start in range(0, len(indices), 32):
            ids = indices[start:start+32]
            obs = to_device(next(iter(DataLoader(torch.utils.data.Subset(ds, ids), batch_size=len(ids))))['obs'], next(teacher.parameters()).device)
            captured, original = [], enc._encode_dino
            def capture(images, **kw):
                out = original(images, **kw)
                captured.append(out)
                return out
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                enc._encode_dino = capture
                try:
                    target = teacher._encode_context(obs, apply_curriculum_mask=False)
                finally:
                    enc._encode_dino = original
                enc._encode_dino = lambda images, **kw: captured[0]
                try:
                    zero = teacher._encode_context(zero_wrench_reference(teacher, obs), apply_curriculum_mask=False)
                finally:
                    enc._encode_dino = original
            for key, tensor in [('target', target), ('zero', zero)]:
                arrays[key][ids] = tensor.float().cpu().numpy().astype(np.float16)
        for array in arrays.values():
            array.flush()
        eps = episode_ids(ds)
        # Probe every validation episode/phase and four distributed train episodes.
        for ep in np.unique(eps):
            if split == 'train' and ep not in [0, 20, 40, 61]:
                continue
            for phase in range(3):
                candidates = np.flatnonzero((eps == ep) & (ds.phase_labels == phase))
                if not len(candidates):
                    continue
                index = int(candidates[len(candidates)//2])
                obs = to_device(next(iter(DataLoader(torch.utils.data.Subset(ds, [index]), batch_size=1)))['obs'], next(teacher.parameters()).device)
                with torch.no_grad(), torch.autocast('cuda', enabled=False):
                    actual = dict(target=teacher._encode_context(obs, apply_curriculum_mask=False).float(),
                        zero=teacher._encode_context(zero_wrench_reference(teacher, obs), apply_curriculum_mask=False).float())
                errors = {k: float(np.mean((v.cpu().numpy()[0]-arrays[k][index].astype(np.float32))**2)) for k, v in actual.items()}
                assert max(errors.values()) < 1e-5, (split, index, errors)
                manifest['audit'].append(dict(split=split, index=index, episode=int(ep), phase=phase,
                    precision_repaired=True, target_errors=errors,
                    image_sha256={k: hashlib.sha256(obs[k].cpu().numpy().tobytes()).hexdigest() for k in teacher.obs_encoder.image_keys}))
        manifest['arrays'][split] = dict(reused_initially=len(indices), recomputed_for_precision=len(indices),
            computed_targets=len(ds), target_sha256=file_hash(cache/split/'target.npy'), zero_sha256=file_hash(cache/split/'zero.npy'))
        write_json(cache/split/'metadata.json', dict(complete=True, **manifest['splits'][split]))
        print(json.dumps(dict(event='precision_repair', split=split, rows=len(indices), audit_passed=True)), flush=True)
    assert tensor_hash(teacher.state_dict()) == config.teacher_state_sha256
    write_json(cache/'manifest.json', manifest)
    print('CACHE AUDIT COMPLETE', flush=True)


def calibrate(a, config, teacher, initial, cache, split, started):
    output = a.experiment/'adapter'
    output.mkdir()
    device = next(teacher.parameters()).device
    model, _, _ = model_for(initial, 'c_film_mlp')
    adapter = model.context_adapter.to(device)
    initial_path = a.root/config.adapter_initial_checkpoint
    payload = torch.load(initial_path, map_location='cpu', weights_only=False)
    adapter.load_state_dict(payload['state'], strict=True)
    adapter.requires_grad_(True).eval()
    initial_hash = tensor_hash(adapter.state_dict())
    norm = teacher.obs_encoder.context_norm
    arrays = {s: {k: np.load(cache/s/(k+'.npy'), mmap_mode='r') for k in ['visual', 'zero', 'episodes', 'phases']}
              for s in ['train', 'validation']}
    def get(s, ids):
        return [torch.from_numpy(np.array(arrays[s][k][ids], copy=True)).to(device) for k in ['visual', 'zero']]
    @torch.no_grad()
    def evaluate(step, full_train=False):
        metrics = {}
        for s in arrays:
            rows = []
            if s == 'validation' or full_train:
                ids = np.arange(len(arrays[s]['zero']))
            else:
                ids = []
                for ep in split[s]['episodes']:
                    for phase in range(3):
                        candidates = np.flatnonzero((arrays[s]['episodes'] == ep) & (arrays[s]['phases'] == phase))
                        if len(candidates):
                            ids.append(int(candidates[len(candidates)//2]))
                ids = np.asarray(ids)
            for start in range(0, len(ids), 64):
                selected = ids[start:start+64]
                x, y = get(s, selected)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z = norm(adapter(x))
                errors = (z.float()-y.float()).square().mean((1, 2)).cpu().tolist()
                rows.extend(dict(index=int(i), episode=int(arrays[s]['episodes'][i]), phase=int(arrays[s]['phases'][i]),
                    zero_token_mse=error) for i, error in zip(selected, errors))
            metrics[s] = summarize_zero(rows)
            if s == 'validation':
                selection = [r for r in rows if r['episode'] in config.selection_episodes]
                external = [r for r in rows if r['episode'] in config.external_episodes]
                metrics['selection'] = summarize_zero(selection)
                metrics['external'] = summarize_zero(external)
            if full_train or s == 'validation':
                write_json(output/f'{s}_samples_{step}.json', rows)
        row = dict(step=step, metrics=metrics, full_train=full_train, elapsed_seconds=time.monotonic()-started)
        history.append(row)
        write_json(output/'log.json', history)
        print(json.dumps(dict(event='adapter_validation', step=step, selection=metrics['selection']['all'],
            external=metrics['external']['all'], phases=metrics['validation']['phases'])), flush=True)
        return row
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config.adapter.lr, weight_decay=1e-4, betas=(.9, .999))
    x, y = get('train', [0, 1])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        smoke = norm(adapter(x))
    (smoke.float()-y.float()).square().mean().backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in adapter.parameters())
    assert not any(p.requires_grad or p.grad is not None for p in teacher.parameters())
    optimizer.zero_grad(set_to_none=True)
    write_json(output/'smoke.json', dict(finite_forward=bool(torch.isfinite(smoke).all()),
        adapter_gradient=True, teacher_dino_norm_frozen=True, optimizer_steps=0))
    provenance = dict(initial_checkpoint=str(initial_path), initial_checkpoint_sha256=file_hash(initial_path),
        initial_state_sha256=initial_hash, initialization='Warm-start the previously calibrated128-hidden GELU MLP unchanged',
        teacher_checkpoint_sha256=config.teacher_sha256, config=OmegaConf.to_container(config, resolve=True),
        selection_episodes=list(config.selection_episodes), external_episodes=list(config.external_episodes),
        selection='Only internal validation38/44/53/60 selects checkpoints and early stopping;5/26 evaluation only')
    write_json(output/'manifest.json', provenance)
    history = []
    first = evaluate(0, full_train=True)
    best = first['metrics']['selection']['all']['zero_token_mse']
    best_step, stale = 0, 0
    best_state = copy.deepcopy(adapter.state_dict())
    def save_best():
        torch.save(dict(state={k: v.detach().cpu() for k, v in best_state.items()}, kind='mlp', hidden_dim=128,
            step=best_step, teacher_checkpoint_sha256=config.teacher_sha256, manifest=provenance), output/'best_adapter.pt')
    save_best()
    rng = np.random.default_rng(config.seed)
    order, position = rng.permutation(len(arrays['train']['zero'])), 0
    for step in range(1, config.adapter.max_steps+1):
        if position+config.adapter.batch_size > len(order):
            order, position = rng.permutation(len(order)), 0
        ids = order[position:position+config.adapter.batch_size]
        position += config.adapter.batch_size
        x, y = get('train', ids)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = norm(adapter(x))
        loss = (z.float()-y.float()).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.)
        assert not any(p.grad is not None or p.requires_grad for p in teacher.parameters())
        optimizer.step()
        if step % config.adapter.validate_every == 0 or step == config.adapter.max_steps:
            row = evaluate(step)
            score = row['metrics']['selection']['all']['zero_token_mse']
            if score < best-config.adapter.min_delta:
                best, best_step, stale = score, step, 0
                best_state = copy.deepcopy(adapter.state_dict())
                save_best()
            else:
                stale += 1
            if stale >= config.adapter.patience:
                break
    adapter.load_state_dict(best_state, strict=True)
    adapter.requires_grad_(False)
    final = evaluate(best_step, full_train=True)
    summary = dict(initial=first, best=final, optimizer_steps=step, best_step=best_step,
        stopped_early=step < config.adapter.max_steps, state_sha256=tensor_hash(adapter.state_dict()),
        checkpoint_sha256=file_hash(output/'best_adapter.pt'), teacher_checkpoint_sha256=config.teacher_sha256,
        teacher_frozen_verified=tensor_hash(teacher.state_dict()) == config.teacher_state_sha256,
        frozen_adapter=True, elapsed_seconds=time.monotonic()-started)
    assert summary['teacher_frozen_verified']
    write_json(output/'summary.json', summary)
    print(json.dumps(dict(event='adapter_complete', best_step=best_step, best=best)), flush=True)


if __name__ == '__main__':
    main()
