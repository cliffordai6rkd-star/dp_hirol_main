"""Matched fresh CNN/LSTM vs unchanged D RGB training on the pinned new teacher."""
from pathlib import Path
import argparse
import copy
import json
import time

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Subset

from student_dual_path_common import load_inputs, datasets_from_reference, to_device, tensor_hash, write_json
from paired_response_full_common import (file_hash, active_response_weights, deterministic_mode,
    explicit_zero_alpha, sample_metrics, grouped_metrics, PHASES)
from new_teacher_response_common import (fresh_reference, SharedCNNLSTM, FullPairedDataset,
    module_norms, continuation_decision)
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.vision.render_net_attention import zero_wrench_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['root', 'experiment']:
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--group', choices=['cnn', 'd'], required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--resume', type=Path)
    p.add_argument('--config', type=Path, help='Independent continuation-policy snapshot; training hyperparameters stay fixed')
    a = p.parse_args()
    config = OmegaConf.load(a.config or a.experiment/'config.yaml')
    assert a.steps <= config.rgb.maximum_steps
    output = a.experiment/(('smoke_' if a.smoke else '') + a.group)
    output.mkdir(exist_ok=a.resume is not None)
    device = torch.device('cuda:0')
    torch.set_num_threads(4)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    teacher, teacher_cfg, initial = load_inputs(a.experiment/'inputs', device)
    assert file_hash(a.experiment/'inputs/pinned_teacher_ema.ckpt') == config.teacher_sha256
    assert tensor_hash(teacher.state_dict()) == config.teacher_state_sha256
    adapter_path = a.experiment/'adapter/best_adapter.pt'
    adapter = torch.load(adapter_path, map_location='cpu', weights_only=False)
    assert adapter['teacher_checkpoint_sha256'] == config.teacher_sha256
    owner = fresh_reference(initial, teacher, a.root/config.condition_checkpoint, config.seed)
    owner.context_adapter.load_state_dict(adapter['state'], strict=True)
    owner.to(device).set_training_stage('rgb')
    if a.group == 'cnn':
        torch.manual_seed(config.seed)
        network = SharedCNNLSTM(owner).to(device)
        owner.requires_grad_(False)
        network.requires_grad_(True)
    else:
        network = owner
    deterministic_mode(owner)
    deterministic_mode(network)
    parameters = [p for p in network.parameters() if p.requires_grad]
    assert parameters
    if a.group == 'd':
        assert {id(p) for p in parameters} == {id(p) for p in owner.rgb_parameters()}
        assert len(owner.backbone.image_self_attention) == 4 and len(owner.backbone.cross_attention) == 2
        assert owner.backbone.decoder_film is not None
    assert owner.context_adapter.mapping[0].out_features == 128
    frozen = {k: tensor_hash(m.state_dict()) for k, m in [('adapter', owner.context_adapter), ('condition', owner.condition_network)]}
    cache = a.experiment/'paired_cache'
    cache_manifest = json.loads((cache/'manifest.json').read_text())
    assert cache_manifest['teacher_state_sha256'] == config.teacher_state_sha256
    train, val, identities = datasets_from_reference(a.experiment/'inputs', a.root/'decoded_cuda')
    for s in ['train', 'validation']:
        assert identities[s] == cache_manifest['splits'][s]
    datasets = {s: FullPairedDataset(ds, cache/s) for s, ds in [('train', train), ('validation', val)]}
    plan = np.load(a.experiment/'train_batches.npy')
    assert plan.shape == (config.rgb.maximum_steps, config.rgb.batch_size)
    threshold_values = []
    free_ids = np.flatnonzero(train.phase_labels == 0)
    for start in range(0, len(free_ids), 128):
        ids = free_ids[start:start+128]
        dt = datasets['train'].arrays['target'][ids, ::8].astype(np.float32)-datasets['train'].arrays['zero'][ids, ::8].astype(np.float32)
        threshold_values.append(np.sqrt(np.mean(dt**2, -1)).flatten())
    threshold = max(float(config.rgb.response_minimum_rms), float(np.quantile(np.concatenate(threshold_values), config.rgb.response_free_quantile)))
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    def inputs(batch):
        images, wrench, _, _ = teacher.prepare_context_inputs_with_future_contact(batch['obs'])
        # Run physical-unit zero through the entire original preprocessing path,
        # including any gating before/after normalization, rather than assuming
        # normalized zeros or bypassing the policy preprocessor.
        _, zero, _, _ = teacher.prepare_context_inputs_with_future_contact(zero_wrench_reference(teacher, batch['obs']))
        return images, wrench, zero
    def render(net, images, wrench):
        condition_owner = net if a.group == 'd' else owner
        with torch.no_grad(), amp():
            masks, condition = condition_owner.condition_network(images, wrench)
            gates = {k: condition_owner._mask_gate(masks[k].flatten(0, 1), images[k]).reshape_as(masks[k]).float()
                     * condition['contact_alpha'][..., None, None, None] for k in images}
        with torch.backends.cudnn.flags(enabled=False), amp():
            if a.group == 'd':
                rendered = net(images, wrench)
                residual = {k: rendered[k].float()-images[k].float() for k in images}
            else:
                raw = net(images, wrench)
                residual = {k: owner.residual_scale*raw[k].float()*gates[k] for k in images}
        # Both heads start at zero, use identical pre-cap scale and deployed
        # support, and receive the same0.1 gate*alpha pixel modification bound.
        result = {k: (images[k]+torch.maximum(-config.rgb.epsilon*gates[k],
                    torch.minimum(config.rgb.epsilon*gates[k], residual[k]))).clamp(0, 1) for k in images}
        return result, condition['contact_alpha']
    def encode(images, grad=False):
        with amp():
            return teacher.obs_encoder.encode_image_context(images, context_adapter=owner.context_adapter, input_grad=grad).float()
    def batch_for(split, ids):
        return to_device(next(iter(DataLoader(Subset(datasets[split], list(map(int, ids))), batch_size=len(ids)))), device)
    def denominator(batch):
        dt = batch['target'].float()-batch['zero'].float()
        weights = active_response_weights(dt, threshold)*(batch['phase'] == 2)[:, None]
        return dict(delta=weights.sum(), contact=(batch['phase'] == 2).sum()*dt.shape[1],
                    free=(batch['phase'] == 0).sum()*dt.shape[1])
    def losses(batch, den):
        images, wrench, zero_wrench = inputs(batch)
        rendered, _ = render(network, images, wrench)
        rendered0, _ = render(network, images, zero_wrench)
        z, z0 = encode(rendered, True), encode(rendered0, True)
        target, zero = batch['target'].float(), batch['zero'].float()
        delta = target-zero
        weights = active_response_weights(delta, threshold)*(batch['phase'] == 2)[:, None]
        contact = (z-target).square().mean(-1)
        free = .5*((z-zero).square().mean(-1)+(z0-zero).square().mean(-1))
        return dict(delta=((z-z0-delta).square().mean(-1)*weights).sum()/den['delta'].clamp_min(1e-12),
            contact=contact[batch['phase'] == 2].sum()/den['contact'].clamp_min(1),
            free=free[batch['phase'] == 0].sum()/den['free'].clamp_min(1))
    coefficients = dict(delta=float(config.rgb.lambda_delta), contact=float(config.rgb.lambda_contact), free=float(config.rgb.lambda_free))
    def parts(batch):
        def subset(value, section):
            return {k: subset(v, section) for k, v in value.items()} if isinstance(value, dict) else value[section]
        return [subset(batch, slice(i, i+config.rgb.microbatch)) for i in range(0, len(batch['index']), config.rgb.microbatch)]
    def diagnostics(batch):
        vectors = {k: [torch.zeros_like(p) for p in parameters] for k in coefficients}
        for part in parts(batch):
            terms = losses(part, denominator(batch))
            for key, loss in terms.items():
                gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                for dest, gradient in zip(vectors[key], gradients):
                    if gradient is not None:
                        dest.add_(gradient.detach())
        return {k: sum(float(g.float().square().sum()) for g in gs)**.5 for k, gs in vectors.items()}
    def check_frozen():
        for label, module in [('adapter', owner.context_adapter), ('condition', owner.condition_network)]:
            assert not any(p.requires_grad or p.grad is not None for p in module.parameters())
            assert tensor_hash(module.state_dict()) == frozen[label]
        assert not any(p.requires_grad or p.grad is not None for p in teacher.parameters())
    probe = batch_for('train', plan[0])
    image, wrench, zero_wrench = inputs(probe)
    zero_images, _, _, _ = teacher.prepare_context_inputs_with_future_contact(zero_wrench_reference(teacher, probe['obs']))
    assert all(torch.equal(image[k], zero_images[k]) for k in image)
    # Physical zero must be raw-unit zero followed by the teacher's original normalizer.
    denormalized = teacher.normalizer[teacher.wrench_key].unnormalize(zero_wrench)
    assert denormalized.abs().max() < 1e-5
    with torch.no_grad():
        rendered, alpha = render(network, image, wrench)
        repeat, _ = render(network, image, wrench)
        rendered0, alpha0 = render(network, image, zero_wrench)
        assert all(torch.equal(rendered[k], repeat[k]) for k in image)
        assert all(torch.equal(rendered[k], image[k]) for k in image), 'Fresh generator must start at exact identity'
        # Stage1 cache must reproduce the ONLINE deployment feature path.
        ids = probe['index'].cpu().numpy()
        visual = torch.from_numpy(np.array(np.load(cache/'train/visual.npy', mmap_mode='r')[ids], copy=True)).to(device)
        with amp():
            cached_z = teacher.obs_encoder.context_norm(owner.context_adapter(visual)).float()
        online_z = encode(image)
        cached_error = float((online_z-cached_z).square().mean())
        # BF16 GEMM/attention kernels change with batch shape. Verify the cache
        # at its ORIGINAL encode batch size, then separately report the online
        # microbatch drift instead of confusing it with an index mismatch.
        repeats = (config.cache_batch_size+len(ids)-1)//len(ids)
        audit_images = {k: v.repeat(repeats, 1, 1, 1, 1)[:config.cache_batch_size] for k, v in image.items()}
        audit_z = encode(audit_images)[:len(ids)]
        matched_batch_cache_error = float((audit_z-cached_z).square().mean())
        assert matched_batch_cache_error < 1e-5, matched_batch_cache_error
        with explicit_zero_alpha(owner):
            closed, _ = render(network, image, zero_wrench)
        assert all(torch.equal(closed[k], image[k]) for k in image)
    gradients0 = diagnostics(probe)
    assert gradients0['delta'] > 0 and gradients0['contact'] > 0
    check_frozen()
    checkpoint_provenance = dict(teacher_sha256=file_hash(a.experiment/'inputs/pinned_teacher_ema.ckpt'),
        adapter_sha256=file_hash(adapter_path), condition_source_sha256=file_hash(a.root/config.condition_checkpoint),
        student_initial_config_sha256=file_hash(a.experiment/'inputs/student_rgb_initial.ckpt'))
    manifest = dict(group=a.group, smoke=a.smoke, config=OmegaConf.to_container(config, resolve=True),
        checkpoints=checkpoint_provenance, initial_state_sha256=tensor_hash(network.state_dict()), frozen=frozen,
        shared_torque_state_sha256=tensor_hash(network.wrench_encoder.state_dict()),
        initialization='Fresh seed42 image networks; exact zero RGB heads; SAME pinned-teacher torque projection/LSTM/positions; no pretrained RGB weights',
        pre_cap_residual_scale=owner.residual_scale, pixel_bound=config.rgb.epsilon,
        trainable_names=[n for n, p in network.named_parameters() if p.requires_grad],
        parameters=sum(p.numel() for p in parameters), sample_plan_sha256=file_hash(a.experiment/'train_batches.npy'),
        paired_cache=str(cache), cache_targets=cache_manifest['arrays'], threshold=threshold, coefficients=coefficients,
        initial_gradient_norms=gradients0, physical_zero_verified=True, deterministic_forward_verified=True,
        explicit_alpha0_identity=True, initial_identity=True, matched_batch_calibration_cache_mse=matched_batch_cache_error,
        online_calibration_batch_drift_mse=cached_error,
        actual_zero_alpha_mean=float(alpha0.mean()), actual_zero_alpha_max=float(alpha0.max()))
    write_json(output/'manifest.json', manifest)
    OmegaConf.save(config, output/'config.yaml')
    print(json.dumps(dict(event='preflight', group=a.group, gradients=gradients0, zero_alpha=float(alpha0.mean()), online_cache_mse=cached_error)), flush=True)
    optimizer = torch.optim.AdamW(parameters, lr=config.rgb.lr, weight_decay=config.rgb.weight_decay, betas=tuple(config.rgb.betas))
    ema_network = copy.deepcopy(network).eval().requires_grad_(False)
    ema = EMAModel(ema_network, **dict(initial['cfg'].ema))
    history, training, gradient_history = [], [], [dict(step=0, norms=gradients0)]
    start_step, best = 0, float('inf')
    if a.resume:
        saved = torch.load(a.resume, map_location='cpu', weights_only=False)
        for key in ['checkpoints', 'initial_state_sha256', 'sample_plan_sha256', 'shared_torque_state_sha256']:
            assert saved['manifest'][key] == manifest[key], key
        network.load_state_dict(saved['model'], strict=True)
        ema_network.load_state_dict(saved['ema_model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        ema.load_state_dict(saved['ema'])
        start_step, best = saved['step'], saved['best']
        history, training, gradient_history = saved['history'], saved['training'], saved['gradient_history']
    started = time.monotonic()
    def save(name, step):
        torch.save(dict(model={k: v.detach().cpu() for k, v in network.state_dict().items()},
            ema_model={k: v.detach().cpu() for k, v in ema_network.state_dict().items()},
            optimizer=optimizer.state_dict(), ema=ema.state_dict(), step=step, best=best,
            manifest=manifest, history=history, training=training, gradient_history=gradient_history), output/name)
    @torch.no_grad()
    def evaluate(step, full_train=False):
        metrics, visuals = {}, []
        for split, ds in datasets.items():
            ids = list(range(len(ds)))
            if split == 'train' and not full_train:
                ids = []
                for ep in np.unique(ds.episodes):
                    for phase in range(3):
                        candidates = np.flatnonzero((ds.episodes == ep) & (ds.dataset.phase_labels == phase))
                        if len(candidates):
                            ids.append(int(candidates[len(candidates)//2]))
            if a.smoke:
                ids = [int(np.flatnonzero(ds.dataset.phase_labels == p)[0]) for p in range(3)]
            records = []
            for raw in DataLoader(Subset(ds, ids), batch_size=config.rgb.validation_batch,
                                  num_workers=config.workers, pin_memory=True):
                batch = to_device(raw, device)
                images, wrench, zero_wrench = inputs(batch)
                rendered, alpha = render(ema_network, images, wrench)
                rendered0, alpha0 = render(ema_network, images, zero_wrench)
                z, z0 = encode(rendered), encode(rendered0)
                rows = sample_metrics(z, z0, batch['target'], batch['zero'], rendered, rendered0, images, alpha, alpha0, threshold)
                for i, row in enumerate(rows):
                    ep, phase, index = int(batch['episode'][i]), int(batch['phase'][i]), int(batch['index'][i])
                    records.append(dict(index=index, episode=ep, phase=phase, **row))
                    if split == 'validation' and phase == 2 and not any(r['episode'] == ep for r in visuals):
                        visuals.append(dict(episode=ep, index=index, original={k: v[i].cpu() for k, v in images.items()},
                            rendered={k: v[i].cpu() for k, v in rendered.items()},
                            teacher_response=(batch['target'][i].float()-batch['zero'][i].float()).cpu(),
                            student_response=(z[i]-z0[i]).cpu()))
            metrics[split] = grouped_metrics(records)
            if split == 'validation':
                selection = grouped_metrics([r for r in records if r['episode'] in config.selection_episodes])
                external = grouped_metrics([r for r in records if r['episode'] in config.external_episodes])
            write_json(output/f'{split}_samples_{step}.json', records)
        row = dict(step=step, metrics=metrics, selection=selection['phases'], external=external['phases'],
            full_train=full_train, weights='EMA', elapsed_seconds=time.monotonic()-started)
        # Smoke selects just one episode and is only an execution check.
        if not a.smoke:
            row['generalization_gap'] = {phase: {k: metrics['validation']['phases'][phase][k]-metrics['train']['phases'][phase][k]
                for k in ['token_mse', 'zero_token_mse', 'response_difference_mse']}
                for phase in PHASES}
        history.append(row)
        write_json(output/'log.json', history)
        torch.save(visuals, output/f'visuals_{step}.pt')
        print(json.dumps(dict(event='validation', group=a.group, step=step,
            contact=metrics['validation']['phases']['contact'], selection=row['selection']['contact'], external=row['external']['contact'])), flush=True)
        return row
    if not start_step:
        first = evaluate(0, full_train=not a.smoke)
        if not a.smoke:
            best = first['selection']['contact']['response_difference_mse']
        save('initial.pt', 0)
        save('best.pt', 0)
    iterator = iter(DataLoader(datasets['train'], batch_sampler=plan[start_step:a.steps].tolist(),
        num_workers=config.workers, pin_memory=True))
    for step in range(start_step+1, a.steps+1):
        batch = to_device(next(iterator), device)
        optimizer.zero_grad(set_to_none=True)
        den, totals = denominator(batch), {}
        for part in parts(batch):
            terms = losses(part, den)
            loss = sum(coefficients[k]*value for k, value in terms.items())
            assert torch.isfinite(loss)
            loss.backward()
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.)+float(value.detach())
        norms = module_norms(network)
        torch.nn.utils.clip_grad_norm_(parameters, config.rgb.gradient_clip)
        optimizer.step()
        ema.step(network)
        if step == start_step+1 or step % config.rgb.log_every == 0 or a.smoke:
            row = dict(step=step, losses=totals, gradient_norms=norms, indices=batch['index'].cpu().tolist(),
                episodes=batch['episode'].cpu().tolist(), lr=config.rgb.lr, elapsed_seconds=time.monotonic()-started)
            training.append(row)
            write_json(output/'training.json', training)
            print(json.dumps(dict(event='train', group=a.group, **row)), flush=True)
        if step % config.rgb.validate_every == 0 or step == a.steps:
            check_frozen()
            gradient_history.append(dict(step=step, norms=diagnostics(probe)))
            write_json(output/'loss_gradients.json', gradient_history)
            final = evaluate(step, full_train=step == a.steps and not a.smoke)
            if a.smoke:
                assert step >= 2
                assert norms['cnn'] > 0 and norms['lstm'] > 0
            elif final['selection']['contact']['response_difference_mse'] < best:
                best = final['selection']['contact']['response_difference_mse']
                save('best.pt', step)
            save('latest.pt', step)
    check_frozen()
    with torch.no_grad(), explicit_zero_alpha(owner):
        closed, _ = render(network, image, zero_wrench)
    assert all(torch.equal(closed[k], image[k]) for k in image), 'Alpha0 identity must also hold AFTER RGB learning'
    assert tensor_hash(teacher.state_dict()) == config.teacher_state_sha256
    summary = dict(manifest=manifest, initial=history[0], final=history[-1], steps=a.steps,
        frozen_verified=True, trained_alpha0_identity_verified=True, peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,
        eligible_for_extension=continuation_decision(history, config.rgb) if not a.smoke else dict(extend=False, reason='Smoke only'),
        checkpoint_sha256=file_hash(output/'latest.pt'), best_checkpoint_sha256=file_hash(output/'best.pt'),
        initial_checkpoint_sha256=file_hash(output/'initial.pt'),
        elapsed_seconds=time.monotonic()-started)
    write_json(output/'summary.json', summary)
    print(json.dumps(dict(event='complete', group=a.group, steps=a.steps, eligible=summary['eligible_for_extension'])), flush=True)


if __name__ == '__main__':
    main()
