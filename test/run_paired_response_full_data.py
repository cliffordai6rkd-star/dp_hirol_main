"""Full-episode paired-response A/B experiment; frozen deployed student paths.

Only experiment outputs are written. RenderNet, production losses/configs and
teacher code are imported unchanged from V. The zero branch ALWAYS renders
through the real frozen Condition Network, without an identity shortcut.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import time

import dill
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Subset

from student_dual_path_common import (load_inputs, model_for, datasets_from_reference,
    PreparedDataset, to_device, tensor_hash, write_json)
from paired_response_full_common import (PHASES, file_hash, episode_ids, spread_order,
    active_response_weights, deterministic_mode, explicit_zero_alpha,
    gradient_norms, sample_metrics, grouped_metrics, extension_allowed)
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.vision.render_net_attention import zero_wrench_reference
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor


def event(kind, **values):
    print(json.dumps(dict(kind=kind, **values)), flush=True)


def load_student(root, initial, config, device):
    checkpoint = root / config.starting_checkpoint
    payload = torch.load(checkpoint, map_location='cpu', pickle_module=dill, weights_only=False)
    model, model_cfg, _ = model_for(initial, config.variant)
    model.load_state_dict(payload['state_dicts']['ema_model'], strict=True)
    calibration = torch.load(root / config.adapter_checkpoint, map_location='cpu', weights_only=False)
    assert calibration['kind'] == 'mlp' and calibration['hidden_dim'] == 128
    assert all(torch.equal(v, model.context_adapter.state_dict()[k]) for k, v in calibration['state'].items())
    model.to(device).set_training_stage('rgb')
    deterministic_mode(model)
    assert model.dual_path_enabled and model.backbone.decoder_film is not None
    assert len(model.backbone.image_self_attention) == 4 and len(model.backbone.cross_attention) == 2
    assert model.context_adapter.dim == 256 and model.context_adapter.mapping[0].out_features == 128
    return model, model_cfg, checkpoint


def train_probe_ids(dataset, episodes):
    """Fixed progress probe across ALL train episodes; full train metrics at endpoints."""
    ids = []
    for e in np.unique(episodes):
        for phase in range(3):
            rows = np.flatnonzero((episodes == e) & (dataset.phase_labels == phase))
            if len(rows):
                ids.append(int(rows[len(rows) // 2]))
    return sorted(ids)


def estimate_threshold(dataset, prepared, config):
    """Teacher TRAIN-Free noise floor; sample uniformly over Free windows and patches."""
    array = np.load(prepared / 'train/response.npy', mmap_mode='r')
    ids = np.flatnonzero(dataset.phase_labels == 0)
    # This selects data values only, without student or validation feedback.
    values = []
    dim = np.load(prepared / 'train/target.npy', mmap_mode='r').shape[-1]
    for start in range(0, len(ids), 256):
        values.append(np.asarray(array[ids[start:start+256], ::8], dtype=np.float32).reshape(-1) / dim**.5)
    threshold = max(float(config.response_threshold.minimum_rms),
                    float(np.quantile(np.concatenate(values), config.response_threshold.free_quantile)))
    return threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Existing D checkpoint/image cache root')
    parser.add_argument('--experiment', type=Path, required=True, help='Pinned teacher inputs + prepared_cuda')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'config/civi_paired_response_full_data.yaml')
    parser.add_argument('--group', choices=['a', 'b'], required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Only a small split probe; never formal results')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--calibration-manifest', type=Path)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    if args.steps is not None:
        config.steps = args.steps
    assert 0 < config.steps <= config.maximum_steps
    args.root, args.experiment = args.root.resolve(), args.experiment.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'manifest.json').exists() and args.resume is None:
        raise RuntimeError('Existing experiment preserved: choose a fresh output directory or --resume')
    torch.set_num_threads(4)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0')
    teacher, teacher_cfg, initial = load_inputs(args.experiment/'inputs', device)
    model, model_cfg, student_path = load_student(args.root, initial, config, device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert {id(p) for p in parameters} == {id(p) for p in model.rgb_parameters()}
    assert not any(p.requires_grad for p in teacher.parameters())
    assert not any(p.requires_grad for p in model.context_adapter.parameters())
    assert not any(p.requires_grad for p in model.condition_network.parameters())
    frozen_hash = {name: tensor_hash(module.state_dict()) for name, module in
                   [('teacher', teacher), ('adapter', model.context_adapter), ('condition', model.condition_network)]}
    starting_hash = tensor_hash(model.state_dict())
    prepared = args.experiment/'prepared_cuda'
    cache_manifest = json.loads((prepared/'manifest.json').read_text())
    assert cache_manifest['teacher_state_sha256'] == frozen_hash['teacher'], 'Stale paired teacher cache'
    train, validation, splits = datasets_from_reference(args.experiment/'inputs', args.root/'decoded_cuda')
    datasets = dict(train=train, validation=validation)
    episodes = {s: episode_ids(ds) for s, ds in datasets.items()}
    assert not set(episodes['train']) & set(episodes['validation'])
    for s in datasets:
        assert cache_manifest[s] == splits[s], 'Cache indices/phase split mismatch'
    caches = {s: PreparedDataset(ds, prepared/s) for s, ds in datasets.items()}
    split_details = {s: dict(**splits[s], episodes=sorted(set(episodes[s].tolist())),
        per_episode={str(e): {name: int(np.sum((episodes[s] == e) & (ds.phase_labels == phase)))
                            for phase, name in enumerate(PHASES)} for e in np.unique(episodes[s])})
        for s, ds in datasets.items()}
    threshold = estimate_threshold(train, prepared, config)
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)

    def encode(images, owner=model, grad=False):
        with amp():
            return teacher.obs_encoder.encode_image_context(images,
                context_adapter=owner.context_adapter, input_grad=grad).float()

    def inputs(batch):
        images, wrench, _, _ = teacher.prepare_context_inputs_with_future_contact(batch['obs'])
        zero = teacher.normalizer[teacher.wrench_key].normalize(torch.zeros_like(wrench))
        return images, wrench, zero

    def render(owner, images, wrench):
        # cuDNN eval LSTM cannot backpropagate; use the native deterministic path.
        with torch.backends.cudnn.flags(enabled=False), amp():
            rendered, _, condition = owner(images, wrench, return_condition=True)
        return rendered, condition['contact_alpha']

    def batch_from_indices(split, indices):
        return to_device(next(iter(DataLoader(Subset(caches[split], list(map(int, indices))),
                                             batch_size=len(indices)))), device)

    # Audit actual pixels against historical cache identities and re-read ONLY probes from the teacher.
    audit = []
    for s, ds in datasets.items():
        probe = train_probe_ids(ds, episodes[s])
        probe = [probe[i] for i in np.linspace(0, len(probe)-1, min(12, len(probe))).round().astype(int)]
        for start in range(0, len(probe), config.microbatch):
            ids = probe[start:start+config.microbatch]
            batch = batch_from_indices(s, ids)
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                target = teacher._encode_context(batch['obs'], apply_curriculum_mask=False).float()
                zero = teacher._encode_context(zero_wrench_reference(teacher, batch['obs']),
                                               apply_curriculum_mask=False).float()
            for i, index in enumerate(ids):
                errors = {k: float((v[i]-batch[k][i].float()).square().mean())
                          for k, v in [('target', target), ('zero', zero)]}
                expected = (target[i]-zero[i]).norm(dim=-1)
                response_errors = (expected-batch['response'][i].float()).abs()
                response_error = float(response_errors.max())
                # Targets are stored as FP16, while response was computed before
                # quantization. Bound norm differences using the triangle
                # inequality plus the target storage rounding error.
                cached_t, cached_0 = batch['target'][i].float(), batch['zero'][i].float()
                response_tolerance = ((target[i]-cached_t).norm(dim=-1)+(zero[i]-cached_0).norm(dim=-1)
                    + torch.finfo(torch.float16).eps*(cached_t.norm(dim=-1)+cached_0.norm(dim=-1))+1e-4)
                assert max(errors.values()) < 1e-5 and torch.all(response_errors <= response_tolerance), (s, index, errors, response_error)
                audit.append(dict(split=s, index=index, episode=int(episodes[s][index]), **errors,
                    response_max_error=response_error, response_max_tolerance=float(response_tolerance.max()),
                    image_hash={k: hashlib.sha256(batch['obs'][k][i].cpu().numpy().tobytes()).hexdigest()
                                for k in teacher.obs_encoder.image_keys}))
    historical = torch.load(args.experiment/'inputs/reference_samples.pt', map_location='cpu', weights_only=False)
    for ref in historical:
        obs = datasets[ref['split']][ref['dataset_index']]['obs']
        for key, digest in ref['image_sha256'].items():
            assert hashlib.sha256(obs[key].unsqueeze(0).numpy().tobytes()).hexdigest() == digest

    order = spread_order(train.phase_labels, episodes['train'], config.seed, 0)
    calibration_ids = list(map(int, order[:config.batch_size]))
    # Calibrate on a few distributed train rows per phase when the first batch omits a rare phase.
    for phase in range(3):
        calibration_ids.extend(int(i) for i in order[train.phase_labels[order] == phase][:4])
    calibration_ids = list(dict.fromkeys(calibration_ids))
    calibration = batch_from_indices('train', calibration_ids)

    def denominators(batch):
        dt = batch['target'].float()-batch['zero'].float()
        rw = (batch['response'].detach().float()-config.response_noise_floor).clamp_min(0)
        dw = active_response_weights(dt, threshold)
        return dict(global_mse=torch.tensor(dt.shape[0]*dt.shape[1], device=device),
                    response=rw.sum(), delta=dw.sum(),
                    zero=(batch['phase'] == 2).sum()*dt.shape[1],
                    free=(batch['phase'] == 0).sum()*dt.shape[1])

    def terms(batch, denominator, paired):
        im, wr, wr0 = inputs(batch)
        rendered, _ = render(model, im, wr)
        z = encode(rendered, grad=True)
        target, zero = batch['target'].float(), batch['zero'].float()
        errors = (z-target).square().mean(-1)
        rw = (batch['response'].detach().float()-config.response_noise_floor).clamp_min(0)
        result = dict(global_mse=errors.sum()/denominator['global_mse'],
                      response=(errors*rw).sum()/denominator['response'].clamp_min(torch.finfo(torch.float32).eps))
        if paired:
            rendered0, _ = render(model, im, wr0)
            z0 = encode(rendered0, grad=True)
            dt = target-zero
            dw = active_response_weights(dt, threshold)
            result['delta'] = (((z-z0-dt).square().mean(-1))*dw).sum()/denominator['delta'].clamp_min(1e-12)
            zero_errors = (z0-zero).square().mean(-1)
            for key, phase in [('zero', 2), ('free', 0)]:
                result[key] = zero_errors[batch['phase'] == phase].sum()/denominator[key].clamp_min(1)
        return result

    def slices(batch):
        n = len(batch['phase'])
        def sub(value, sl):
            return {k: sub(v, sl) for k, v in value.items()} if isinstance(value, dict) else value[sl]
        return [sub(batch, slice(i, min(i+config.microbatch, n))) for i in range(0, n, config.microbatch)]

    def gradient_diagnostics(batch):
        vectors = {k: [torch.zeros_like(p) for p in parameters]
                   for k in ['global_mse', 'response', 'delta', 'zero', 'free']}
        denominator = denominators(batch)
        for part in slices(batch):
            losses = terms(part, denominator, True)
            for key, loss in losses.items():
                grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                for dest, g in zip(vectors[key], grads):
                    if g is not None:
                        dest.add_(g.detach())
        vectors['base'] = [a+b for a, b in zip(vectors['global_mse'], vectors['response'])]
        result = {k: gradient_norms(v, names) for k, v in vectors.items()}
        dot = sum(float((a*b).sum()) for a, b in zip(vectors['base'], vectors['delta']))
        result['base_delta_cosine'] = dot/max(result['base']['all']*result['delta']['all'], 1e-20)
        return result

    gradient0 = gradient_diagnostics(calibration)
    assert gradient0['base']['all'] > 0 and gradient0['delta']['all'] > 0
    coefficients = dict(global_mse=1., response=1.,
        delta=config.delta_gradient_fraction*gradient0['base']['all']/gradient0['delta']['all'],
        zero=float(config.lambda_zero), free=float(config.lambda_free))
    if args.calibration_manifest:
        pinned = json.loads(args.calibration_manifest.read_text())
        assert pinned['starting_state_sha256'] == starting_hash
        assert pinned['teacher_checkpoint_sha256'] == file_hash(args.experiment/'inputs/pinned_teacher_ema.ckpt')
        assert np.isclose(coefficients['delta'], pinned['coefficients']['delta'], rtol=1e-3)
        coefficients['delta'] = pinned['coefficients']['delta']
    if args.group == 'a':
        coefficients.update(delta=0., zero=0., free=0.)
    probe = batch_from_indices('train', calibration_ids[:config.microbatch])
    im, wr, wr0 = inputs(probe)
    with torch.no_grad():
        normal, alpha = render(model, im, wr)
        repeated, _ = render(model, im, wr)
        assert all(torch.equal(normal[k], repeated[k]) for k in im)
        actual_zero, alpha0 = render(model, im, wr0)
        with explicit_zero_alpha(model):
            closed, closed_alpha = render(model, im, wr0)
        assert torch.count_nonzero(closed_alpha) == 0 and all(torch.equal(closed[k], im[k]) for k in im)
    zero_diagnostic = dict(predicted_alpha_min=float(alpha0.min()), predicted_alpha_max=float(alpha0.max()),
        predicted_alpha_mean=float(alpha0.mean()),
        actual_zero_rgb_max=max(float((actual_zero[k]-im[k]).abs().max()) for k in im),
        contact_L0_gradient_norm=gradient0['zero']['all'], free_gradient_norm=gradient0['free']['all'],
        contact_L0_trainable=gradient0['zero']['all'] > 0,
        free_trainable=gradient0['free']['all'] > 0,
        explicit_alpha_zero_rgb_exact=True, deterministic_pair_verified=True)
    write_json(args.output/'cache_audit.json', audit)
    write_json(args.output/'episode_split.json', split_details)
    optimizer_options = OmegaConf.to_container(model_cfg.optimizer, resolve=True)
    optimizer_options['lr'] *= float(config.rgb_lr_multiplier)
    optimizer = torch.optim.AdamW(parameters, **optimizer_options)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    ema = EMAModel(ema_model, **dict(initial['cfg'].ema))
    manifest = dict(group=args.group, smoke=args.smoke, config=OmegaConf.to_container(config, resolve=True),
        starting_state_sha256=starting_hash, checkpoint_sha256=file_hash(student_path),
        teacher_checkpoint_sha256=file_hash(args.experiment/'inputs/pinned_teacher_ema.ckpt'),
        adapter_checkpoint_sha256=file_hash(args.root/config.adapter_checkpoint), frozen_hashes=frozen_hash,
        splits=split_details, optimizer=optimizer_options, ema=dict(initial['cfg'].ema), coefficients=coefficients,
        response_rms_threshold=threshold, threshold_source='TRAIN Free responses only',
        zero_branch=zero_diagnostic, initial_gradients=gradient0, calibration_indices=calibration_ids,
        trainable_names=names, cache_directory=str(prepared), teacher_weights='ema_model',
        checkpoint_source=str(student_path), adapter_calibration='existing train-only MLP checkpoint; not refit',
        pair_path='same RGB; physical zero -> original normalizer; actual predicted gate/alpha in both branches')
    write_json(args.output/'manifest.json', manifest)
    OmegaConf.save(config, args.output/'experiment_config.yaml')
    model_cfg.teacher_checkpoint = str(args.experiment/'inputs/pinned_teacher_ema.ckpt')
    model_cfg.teacher_config = None
    preprocessor = WrenchPreprocessor.from_policy(teacher).eval()
    model_cfg.wrench_preprocessing = preprocessor.get_config()
    OmegaConf.save(model_cfg, args.output/'student_config.yaml')
    OmegaConf.save(teacher_cfg, args.output/'teacher_config.yaml')
    event('preflight', group=args.group, coefficients=coefficients, zero_branch=zero_diagnostic,
          threshold=threshold, gradients=gradient0)
    if args.check_only:
        return

    started = time.monotonic()
    history, gradient_log, start_step, best = [], [], 0, float('inf')
    initial_metrics = None

    def check_frozen():
        for name, module in [('adapter', model.context_adapter), ('condition', model.condition_network)]:
            assert tensor_hash(module.state_dict()) == frozen_hash[name]
            assert not any(p.requires_grad or p.grad is not None for p in module.parameters())
        assert not any(p.requires_grad or p.grad is not None for p in teacher.parameters())

    def save(name, step, metrics):
        states = {label: {k: v.detach().cpu() for k, v in owner.state_dict().items()}
                  for label, owner in [('model', model), ('ema_model', ema_model), ('wrench_preprocessor', preprocessor)]}
        payload = dict(cfg=model_cfg, teacher_cfg=teacher_cfg, state_dicts=states,
            optimizer=optimizer.state_dict(), ema=ema.state_dict(), manifest=manifest,
            step=step, metrics=metrics, initial_metrics=initial_metrics, best=best,
            history=history, gradients=gradient_log,
            pickles={k: dill.dumps(v) for k, v in dict(global_step=step, training_stage='rgb', stage_step=step).items()})
        temporary = args.output/(name+'.tmp')
        torch.save(payload, temporary, pickle_module=dill)
        temporary.replace(args.output/name)

    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', pickle_module=dill, weights_only=False)
        for k in ['starting_state_sha256', 'teacher_checkpoint_sha256', 'frozen_hashes']:
            assert saved['manifest'][k] == manifest[k], k
        for key, value in coefficients.items():
            assert np.isclose(value, saved['manifest']['coefficients'][key], rtol=1e-3), key
        # Retain the exact original lambdas across extension/resume, even when
        # GPU reduction rounding slightly changes a repeated gradient probe.
        coefficients = dict(saved['manifest']['coefficients'])
        manifest['coefficients'] = coefficients
        write_json(args.output/'manifest.json', manifest)
        model.load_state_dict(saved['state_dicts']['model'], strict=True)
        ema_model.load_state_dict(saved['state_dicts']['ema_model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        ema.load_state_dict(saved['ema'])
        start_step, best = saved['step'], saved['best']
        history, gradient_log, initial_metrics = saved['history'], saved['gradients'], saved['initial_metrics']

    def log(row):
        history.append(row)
        write_json(args.output/'log.json', history)
        visible = {k: v for k, v in row.items() if k not in ['kind', 'metrics']}
        if 'metrics' in row:
            visible['metrics'] = {s: m['phases'] for s, m in row['metrics'].items()}
        event(row['kind'], group=args.group, **visible)

    @torch.no_grad()
    def evaluate(step, full_train=False):
        metrics, visuals = {}, []
        for split, ds in datasets.items():
            ids = list(range(len(ds))) if split == 'validation' or full_train else train_probe_ids(ds, episodes[split])
            if args.smoke:
                ids = train_probe_ids(ds, episodes[split])[:6]
            records = []
            loader = DataLoader(Subset(caches[split], ids), batch_size=config.validation_batch,
                                num_workers=min(4, config.workers), pin_memory=True)
            captured = set()
            for raw in loader:
                batch = to_device(raw, device)
                im, wr, wr0 = inputs(batch)
                rendered, alpha = render(ema_model, im, wr)
                rendered0, alpha0 = render(ema_model, im, wr0)
                z, z0 = encode(rendered, ema_model), encode(rendered0, ema_model)
                rows = sample_metrics(z, z0, batch['target'], batch['zero'], rendered, rendered0,
                                      im, alpha, alpha0, threshold)
                for i, row in enumerate(rows):
                    index = int(batch['index'][i])
                    phase, ep = int(batch['phase'][i]), int(episodes[split][index])
                    records.append(dict(index=index, episode=ep, phase=phase, **row))
                    if split == 'validation' and phase == 2 and ep not in captured:
                        captured.add(ep)
                        visuals.append(dict(index=index, episode=ep, original={k: v[i].cpu() for k, v in im.items()},
                            rendered={k: v[i].cpu() for k, v in rendered.items()},
                            teacher_response=(batch['target'][i].float()-batch['zero'][i].float()).cpu(),
                            student_response=(z[i]-z0[i]).cpu()))
            metrics[split if full_train else 'train_probe' if split == 'train' else split] = grouped_metrics(records)
            write_json(args.output/f'{split}_samples_{step}.json', records)
        train_key = 'train' if full_train else 'train_probe'
        gaps = {phase: {k: metrics['validation']['phases'][phase][k]-metrics[train_key]['phases'][phase][k]
                         for k in ['token_mse', 'zero_token_mse', 'response_difference_mse']}
                for phase in PHASES if metrics[train_key]['phases'][phase]['samples'] and
                metrics['validation']['phases'][phase]['samples']}
        row = dict(kind='validation', step=step, weights='EMA', full_train=full_train,
                   metrics=metrics, generalization_gap=gaps, elapsed_seconds=time.monotonic()-started)
        log(row)
        torch.save(visuals, args.output/f'visuals_{step}.pt')
        return row

    if start_step == 0:
        initial_metrics = evaluate(0, full_train=True)
        best = initial_metrics['metrics']['validation']['phases']['contact']['token_mse']
        save('best.ckpt', 0, initial_metrics)
    # Persist a deterministic sampling plan; epoch replay makes resume independent of validation timing.
    num_batches = (len(train)+config.batch_size-1)//config.batch_size
    sample_hashes = []
    gradient_log.append(dict(step=start_step, gradients=gradient0, source='initial calibration'))
    final_metrics = initial_metrics
    current_epoch, iterator = None, None
    for step in range(start_step+1, int(config.steps)+1):
        epoch, position = divmod(step-1, num_batches)
        if epoch != current_epoch:
            current_epoch = epoch
            order = spread_order(train.phase_labels, episodes['train'], config.seed, epoch)
            sample_hashes.append(dict(epoch=epoch, sha256=hashlib.sha256(order.tobytes()).hexdigest(),
                                     rows=len(order), all_unique=len(set(order.tolist())) == len(train)))
            loader = DataLoader(Subset(caches['train'], order[position*config.batch_size:].tolist()),
                batch_size=config.batch_size, num_workers=config.workers, pin_memory=True,
                persistent_workers=False)
            iterator = iter(loader)
        batch = to_device(next(iterator), device)
        optimizer.zero_grad(set_to_none=True)
        denominator = denominators(batch)
        losses = {}
        for part in slices(batch):
            values = terms(part, denominator, bool(config.groups[args.group].paired_response))
            loss = sum(coefficients[k]*v for k, v in values.items())
            assert torch.isfinite(loss)
            loss.backward()
            for k, value in values.items():
                losses[k] = losses.get(k, 0.)+float(value.detach())
        norms = gradient_norms([p.grad for p in parameters], names)
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, config.max_gradient_norm))
        optimizer.step()
        ema.step(model)
        if step == 1 or step % config.log_every == 0:
            log(dict(kind='train', step=step, losses=losses,
                objective=sum(coefficients[k]*v for k, v in losses.items()), gradient_norms=norms,
                gradient_norm_before_clip=norm, lr=optimizer.param_groups[0]['lr'],
                indices=batch['index'].cpu().tolist(), episodes=[int(episodes['train'][i]) for i in batch['index'].cpu()],
                phase_counts=torch.bincount(batch['phase'], minlength=3).cpu().tolist(),
                elapsed_seconds=time.monotonic()-started))
        if step % config.gradient_every == 0 or step == config.steps:
            diagnostics = gradient_diagnostics(calibration)
            gradient_log.append(dict(step=step, gradients=diagnostics))
            write_json(args.output/'gradient_norms.json', gradient_log)
        if step % config.validate_every == 0 or step == config.steps:
            check_frozen()
            final_metrics = evaluate(step, full_train=step == config.steps)
            score = final_metrics['metrics']['validation']['phases']['contact']['token_mse']
            improved = score < best
            best = min(best, score)
            save('latest.ckpt', step, final_metrics)
            if improved:
                save('best.ckpt', step, final_metrics)
    check_frozen()
    assert tensor_hash(teacher.state_dict()) == frozen_hash['teacher']
    write_json(args.output/'sampling_epochs.json', sample_hashes)
    initial_phases = initial_metrics['metrics']['validation']['phases']
    current_phases = final_metrics['metrics']['validation']['phases']
    write_json(args.output/'summary.json', dict(manifest=manifest, steps=int(config.steps),
        initial=initial_metrics, final=final_metrics, best_contact_token_mse=best,
        frozen_teacher_dino_dit_adapter_condition_verified=True,
        eligible_for_extension=extension_allowed(initial_phases, current_phases, config.extension_relative_tolerance),
        elapsed_seconds=time.monotonic()-started, peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20))
    event('complete', group=args.group, steps=int(config.steps), best=best)


if __name__ == '__main__':
    main()
