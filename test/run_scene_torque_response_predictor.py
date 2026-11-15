"""Quick scene/torque -> full patch response diagnostic, without RGB or teacher training."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf

from student_dual_path_common import write_json, tensor_hash
from paired_response_full_common import file_hash, PHASES


class SceneTorquePredictor(nn.Module):
    """Shared rank64 response head; no episode ID, contact label or target as input."""
    def __init__(self, visual_scale, torque_scale, hidden=64, rank=64):
        super().__init__()
        self.register_buffer('visual_scale', torch.as_tensor(visual_scale).float())
        self.register_buffer('torque_scale', torch.as_tensor(torque_scale).float())
        self.scene = nn.Sequential(nn.Linear(256, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.torque = nn.LSTM(7, hidden, num_layers=2, batch_first=True, dropout=0.)
        self.head = nn.Sequential(nn.Linear(3*hidden+6, 128), nn.GELU(),
                                  nn.Linear(128, rank), nn.GELU(), nn.Linear(rank, 256))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, local, pooled, wrench, patch_ids):
        b, obs, history, dims = wrench.shape
        tau = wrench / self.torque_scale
        # Physical-unit zeros pass through the identical scale/LSTM/head.
        both = torch.cat([tau, torch.zeros_like(tau)], 0).reshape(2*b*obs, history, dims)
        _, (h, _) = self.torque(both)
        h = h[-1].reshape(2*b, obs, -1)
        real, zero = h[:b], h[b:]
        observation = patch_ids // 392
        camera = (patch_ids // 196) % 2
        spatial = patch_ids % 196
        xy = torch.stack([(spatial % 14)/13, (spatial // 14)/13], -1)
        position = torch.cat([F.one_hot(observation, 2), F.one_hot(camera, 2), xy], -1).to(local)
        scene = self.scene(local/self.visual_scale)
        global_scene = self.scene(pooled/self.visual_scale)[:, observation]
        prefix = torch.cat([scene, global_scene, position[None].expand(b, -1, -1)], -1)
        # Subtraction makes the physical-zero response zero up to FP rounding.
        return self.head(torch.cat([prefix, real[:, observation]], -1)) - self.head(torch.cat([prefix, zero[:, observation]], -1))


def phase_plan(phases, episodes, config):
    rng = np.random.default_rng(config.seed)
    def stream(phase):
        eps = np.unique(episodes[phases == phase])
        queues = {}
        while True:
            for ep in rng.permutation(eps):
                if not queues.get(int(ep)):
                    queues[int(ep)] = rng.permutation(np.flatnonzero((phases == phase) & (episodes == ep))).tolist()
                yield queues[int(ep)].pop()
    streams = [stream(p) for p in range(3)]
    return np.asarray([[next(streams[p]) for p, count in enumerate(config.phase_mix) for _ in range(count)]
                       for _ in range(config.steps)])


def extract_wrench(base, arrays):
    """Use exactly Dataset.__getitem__'s index/retiming/window history, skip RGB decoding."""
    import hydra
    meta = json.loads((base/'inputs/reference_train_cache.json').read_text())
    cfg = OmegaConf.create(meta['identity']['dataset'])
    cfg.preload_images = False
    cfg.video_decode_device = 'cpu'
    cfg.video_backend = 'pyav'
    train = hydra.utils.instantiate(cfg)
    result = {}
    compact = Path(json.loads((base/'paired_cache/manifest.json').read_text())['reused_from'])
    for split, ds in [('train', train), ('validation', train.get_validation_dataset())]:
        identity = json.loads((base/f'inputs/reference_{split}_cache.json').read_text())['identity'][split]
        assert hashlib.sha256(np.ascontiguousarray(ds.indices).tobytes()).hexdigest() == identity['indices_sha256']
        assert np.array_equal(ds.phase_labels, arrays[split]['phases'])
        values = ds.windowed_lowdim_data['tau_ext']
        gathered = []
        for i in range(len(ds)):
            sequence = ds._sample_indices_to_sequence(i)
            if ds.window_sampling_strategy == 'timestamp':
                sequence = ds._retime_sequence_indices(sequence)
            gathered.append(values[sequence[:ds.n_obs_steps]])
        result[split] = np.asarray(gathered, dtype=np.float32)
        # Historical raw histories verify the extraction at independently cached indices.
        rows = json.loads((compact/split/'samples.json').read_text())
        old = np.load(compact/split/'raw_wrench.npy', mmap_mode='r')
        np.testing.assert_array_equal(result[split][[r['index'] for r in rows]], old)
    train.lerobot_dataset.close()
    return result


def response_rows(prediction, target, indices, phases, episodes, threshold):
    delta = prediction.float()-target.float()
    active = target.float().square().mean(-1).sqrt() > threshold
    cosine = F.cosine_similarity(prediction.float(), target.float(), dim=-1, eps=1e-6)
    rows = []
    for j, index in enumerate(indices):
        mask = active[j]
        rows.append(dict(index=int(index), phase=int(phases[index]), episode=int(episodes[index]),
            mse=float(delta[j].square().mean()), energy=float(target[j].float().square().mean()),
            valid=int(mask.sum()), cosine_sum=float(cosine[j][mask].sum()),
            ss=float(prediction[j][mask].double().square().sum()),
            tt=float(target[j][mask].double().square().sum())))
    return rows


def aggregate(rows):
    if not rows:
        return dict(samples=0)
    mse = float(np.mean([r['mse'] for r in rows]))
    energy = float(np.mean([r['energy'] for r in rows]))
    count, tt = sum(r['valid'] for r in rows), sum(r['tt'] for r in rows)
    return dict(samples=len(rows), response_mse=mse, zero_predictor_mse=energy,
        explained_energy=1-mse/energy if energy else None,
        cosine=sum(r['cosine_sum'] for r in rows)/count if count else None,
        rms_ratio=(sum(r['ss'] for r in rows)/tt)**.5 if tt else None)


def grouped(rows, config):
    select = lambda eps: [r for r in rows if r['episode'] in eps and r['phase'] == 2]
    return dict(phases={name: aggregate([r for r in rows if r['phase'] == p]) for p, name in enumerate(PHASES)},
        episodes={str(ep): {name: aggregate([r for r in rows if r['episode'] == ep and r['phase'] == p])
            for p, name in enumerate(PHASES)} for ep in sorted({r['episode'] for r in rows})},
        selection_contact=aggregate(select(config.selection_episodes)), external_contact=aggregate(select(config.external_episodes)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'config/scene_torque_response_predictor.yaml')
    parser.add_argument('--smoke', action='store_true')
    a = parser.parse_args()
    config = OmegaConf.load(a.config)
    a.output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(config, a.output/'config.yaml')
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda:0')
    cache = a.base/'paired_cache'
    manifest = json.loads((cache/'manifest.json').read_text())
    assert file_hash(a.base/'inputs/pinned_teacher_ema.ckpt') == config.teacher_sha256 == manifest['teacher_checkpoint_sha256']
    arrays = {s: {k: np.load(cache/s/(k+'.npy'), mmap_mode='r') for k in ['visual', 'target', 'zero', 'phases', 'episodes']}
              for s in ['train', 'validation']}
    assert not set(arrays['train']['episodes']) & set(arrays['validation']['episodes'])
    wrench = extract_wrench(a.base, arrays)
    pooled, mean, contact_count = {}, np.zeros((784, 256), dtype=np.float64), 0
    for split, x in arrays.items():
        pooled[split] = np.empty((len(x['visual']), 2, 256), np.float32)
        for start in range(0, len(x['visual']), 128):
            ids = np.arange(start, min(start+128, len(x['visual'])))
            pooled[split][ids] = x['visual'][ids].reshape(len(ids), 2, 392, 256).mean(2)
            if split == 'train':
                ids = ids[x['phases'][ids] == 2]
                mean += (x['target'][ids].astype(np.float32)-x['zero'][ids].astype(np.float32)).sum(0, dtype=np.float64)
                contact_count += len(ids)
    mean = torch.from_numpy((mean/contact_count).astype(np.float32)).to(device)
    torque_scale = np.maximum(wrench['train'].std((0, 1, 2)), .1)
    visual_scale = np.maximum(pooled['train'].std((0, 1)), .1)
    net = SceneTorquePredictor(visual_scale, torque_scale, config.hidden_dim, config.response_rank).to(device).train()
    initial_hash = tensor_hash(net.state_dict())
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    threshold = json.loads((a.base/'cnn/manifest.json').read_text())['threshold']
    plan = phase_plan(arrays['train']['phases'], arrays['train']['episodes'], config)
    np.save(a.output/'train_batches.npy', plan)
    rng = np.random.default_rng(config.seed)
    all_patches = torch.arange(784, device=device)
    def batch(split, ids, patch_ids=all_patches, scene_ids=None, torque_ids=None):
        p = patch_ids.cpu().numpy()
        scene_ids = ids if scene_ids is None else scene_ids
        torque_ids = ids if torque_ids is None else torque_ids
        x = arrays[split]
        local = torch.from_numpy(np.array(x['visual'][np.asarray(scene_ids)[:, None], p[None]], copy=True)).to(device)
        global_scene = torch.from_numpy(pooled[split][scene_ids]).to(device)
        tau = torch.from_numpy(wrench[split][torque_ids]).to(device)
        target = torch.from_numpy(x['target'][np.asarray(ids)[:, None], p[None]].astype(np.float32)-
                                  x['zero'][np.asarray(ids)[:, None], p[None]].astype(np.float32)).to(device)
        return local, global_scene, tau, target
    @torch.no_grad()
    def evaluate(split, model, control=None, reduced=False):
        x = arrays[split]
        ids = np.arange(len(x['phases']))
        if reduced:
            ids = np.asarray([int(j) for ep in np.unique(x['episodes']) for p in range(3)
                for j in np.flatnonzero((x['episodes'] == ep) & (x['phases'] == p))[::8]])
        donors = np.arange(len(x['phases']))
        if control in ['shuffled_torque', 'shuffled_scene']:
            generator = np.random.default_rng(config.seed+99)
            for i in ids:
                candidates = np.flatnonzero((x['phases'] == x['phases'][i]) & (x['episodes'] != x['episodes'][i]))
                donors[i] = generator.choice(candidates)
        rows = []
        for start in range(0, len(ids), config.validation_batch):
            selected = ids[start:start+config.validation_batch]
            local, pool, tau, target = batch(split, selected,
                scene_ids=donors[selected] if control == 'shuffled_scene' else None,
                torque_ids=donors[selected] if control == 'shuffled_torque' else None)
            if control == 'zero': prediction = torch.zeros_like(target)
            elif control == 'train_mean': prediction = mean[None].expand(len(selected), -1, -1)
            else:
                with torch.autocast('cuda', enabled=False):
                    prediction = model(local, pool, tau, all_patches)
            rows.extend(response_rows(prediction, target, selected, x['phases'], x['episodes'], threshold))
        return grouped(rows, config), rows
    history, best, best_step = [], float('inf'), 0
    if not a.smoke:
        scores, _ = evaluate('validation', net)
        best = scores['selection_contact']['response_mse']
        history.append(dict(step=0, validation=scores, gradient_norms={}, training_batch_loss=None,
                            elapsed_seconds=time.monotonic()-started))
        torch.save(dict(model=net.state_dict(), step=0, optimizer=optimizer.state_dict()), a.output/'best.pt')
    # Smoke lets the zero output head open before checking upstream gradients.
    for step in range(1, (10 if a.smoke else config.steps)+1):
        ids = plan[step-1]
        patch_ids = torch.from_numpy(rng.choice(784, config.patches_per_step, replace=False)).to(device)
        local, pool, tau, target = batch('train', ids, patch_ids)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', enabled=False):
            prediction = net(local, pool, tau, patch_ids)
            loss = F.mse_loss(prediction.float(), target)
        assert torch.isfinite(loss)
        loss.backward()
        gradient = {name: float(sum(p.grad.float().square().sum() for p in module.parameters() if p.grad is not None).sqrt())
                    for name, module in [('scene', net.scene), ('torque', net.torque), ('head', net.head)]}
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        optimizer.step()
        if step == 10:
            assert all(v > 0 for v in gradient.values()), gradient
            with torch.no_grad():
                assert net(local, pool, torch.zeros_like(tau), patch_ids).abs().max() < 1e-6
        if a.smoke:
            print(json.dumps(dict(event='smoke_train', step=step, loss=float(loss), gradient_norms=gradient,
                torque_rms=float(tau.square().mean().sqrt()), torque_scale=torque_scale.tolist())), flush=True)
        if step % config.validate_every == 0 or (a.smoke and step == 10):
            scores, _ = evaluate('validation', net, reduced=a.smoke)
            history.append(dict(step=step, validation=scores, gradient_norms=gradient, training_batch_loss=float(loss),
                                elapsed_seconds=time.monotonic()-started))
            score = scores['selection_contact']['response_mse']
            if score < best:
                best, best_step = score, step
                torch.save(dict(model=net.state_dict(), step=step, optimizer=optimizer.state_dict()), a.output/'best.pt')
            write_json(a.output/'log.json', history)
            print(json.dumps(dict(event='validation', **history[-1])), flush=True)
    if a.smoke:
        write_json(a.output/'smoke.json', dict(gradient_norms=gradient, zero_identity=True, elapsed_seconds=time.monotonic()-started))
        return
    torch.save(dict(model=net.state_dict(), step=config.steps, optimizer=optimizer.state_dict()), a.output/'last.pt')
    net.load_state_dict(torch.load(a.output/'best.pt', weights_only=False)['model'])
    results = {}
    for split in ['train', 'validation']:
        results[split], rows = evaluate(split, net)
        write_json(a.output/f'{split}_samples.json', rows)
    controls = {}
    for control in ['zero', 'train_mean', 'shuffled_torque', 'shuffled_scene']:
        controls[control], rows = evaluate('validation', net, control)
        write_json(a.output/f'validation_{control}_samples.json', rows)
    result = dict(metrics=results, controls=controls, best_step=best_step, steps=config.steps,
        parameters=sum(p.numel() for p in net.parameters()), initial_state_sha256=initial_hash,
        best_checkpoint_sha256=file_hash(a.output/'best.pt'), teacher_checkpoint_sha256=config.teacher_sha256,
        teacher_state_sha256=manifest['teacher_state_sha256'], cache_targets=manifest['arrays'], threshold=threshold,
        data_split=json.loads((a.base/'episode_split.json').read_text()),
        training_plan_sha256=file_hash(a.output/'train_batches.npy'),
        training_coverage={str(p): dict(windows=len(np.unique(plan[:, sum(config.phase_mix[:p]):sum(config.phase_mix[:p+1])])),
            episodes=sorted(set(arrays['train']['episodes'][plan[:, sum(config.phase_mix[:p]):sum(config.phase_mix[:p+1])].flatten()].tolist()))) for p in range(3)},
        preprocessing='Existing audited frozen DINO pre-adapter patch features; train-only RMS scales; exact raw-unit torque histories, all256 teacher final-context response dimensions.',
        frozen='Teacher and DINO served only by immutable cached targets/features; no Adapter/Condition/RGB/DiT loaded or trained.',
        selection='Only38/44/53/60 select best checkpoint;5/26 evaluation only. Teacher checkpoint selection previously included5/26.',
        interpretation='A low-rank shared predictor feasibility test, not a mathematical upper bound. No RGB generation or deployment gate imposed.',
        elapsed_seconds=time.monotonic()-started)
    write_json(a.output/'summary.json', result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key in zip(axes, ['response_mse', 'cosine', 'rms_ratio']):
        for group in ['selection_contact', 'external_contact']:
            ax.plot([r['step'] for r in history], [r['validation'][group][key] for r in history], '-o', label=group)
        ax.set(xlabel='Optimizer steps', title=key);ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(a.output/'learning_curves.png', dpi=160);plt.close(fig)
    print(json.dumps(dict(event='complete', steps=config.steps, best_step=best_step, metrics=results,
                         controls=controls, elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
