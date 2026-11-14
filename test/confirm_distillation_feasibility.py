"""Longer feasibility probes and unseen-noise behavior checks, in isolation.

Consumes the fixed snapshot/samples produced by diagnose_response_distillation.
No production model, optimizer, configuration or running job is changed.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from diagnose_response_distillation import (
    TEACHER, BehaviorProbes, amp, common, event, fingerprints, fit_pixels,
    gates_for, images_of, json_write, load_render_net, renderer_test,
    response_weighted_token_mse,
)


def gradient_checks(samples, teacher, device, output):
    rows = []
    for sample in samples:
        if sample['split'] != 'validation':
            continue
        images = {k: v.detach().requires_grad_(True)
                  for k, v in images_of(sample, device).items()}
        with amp():
            tokens = teacher.obs_encoder.encode_image_context(images, input_grad=True)
            global_loss, weighted, _, _ = response_weighted_token_mse(
                tokens, sample['target'].to(device), sample['response'].to(device))
        parameters = tuple(images.values())
        first = torch.autograd.grad(global_loss, parameters, retain_graph=True)
        second = torch.autograd.grad(weighted, parameters)
        gates = gates_for(sample, 'predicted_gate', device)
        first = torch.cat([(g.float() * gates[k]).flatten() for k, g in zip(images, first)])
        second = torch.cat([(g.float() * gates[k]).flatten() for k, g in zip(images, second)])
        norms = first.norm().item(), second.norm().item()
        rows.append({'sample': sample['id'], 'phase': sample['phase'],
                     'global_pixel_gradient_norm': norms[0],
                     'response_pixel_gradient_norm': norms[1],
                     'gradient_cosine': F.cosine_similarity(first, second, dim=0).item()
                     if norms[0] and norms[1] else None})
    json_write(output / 'gradient_checks.json', rows)
    event('gradient_checks_complete', rows=rows)


def heldout_checks(samples, teacher, device, source, output):
    """Evaluate unseen diffusion noise/timesteps and three independent rollouts.

    Pixel candidates have been fitted to their images; unseen noise is NOT an
    unseen image/episode. Only the scratch renderer's validation is held-out
    across episodes. We measure teacher agreement, not task success.
    """
    probes = [BehaviorProbes(teacher, samples, device, timesteps=(9, 33, 58, 88),
                            noise_seed=seed) for seed in (7000, 8000, 9000)]
    candidates = []
    for sample in samples:
        if sample['split'] == 'validation':
            for label, key in (('raw_image', 'raw_tokens'), ('trained_student', 'student_tokens'),
                               ('teacher_zero_force', 'zero_target'),
                               ('frozen_ffn_control', 'ffn_tokens')):
                candidates.append((sample, label, sample[key]))
    by_id = {s['id']: s for s in samples}
    for folder, label in ((source, 'pixel100'), (output, 'pixel1000')):
        path = folder / 'optimized_pixels.pt'
        if path.is_file():
            fitted = torch.load(path, map_location='cpu', weights_only=False)
            for key, values in fitted.items():
                gate, sample_id = key.split('/', 1)
                if by_id[sample_id]['split'] == 'validation':
                    candidates.append((by_id[sample_id], f'{label}_{gate}', values['tokens']))
    for phase in (1, 2):
        sample = next(s for s in samples if s['split'] == 'validation' and s['phase'] == phase)
        fitted = torch.load(source / f'behavior_phase{phase}.pt', map_location='cpu', weights_only=False)
        candidates.append((sample, 'behavior100_predicted_gate', fitted['tokens']))
    renderer_path = output / 'renderer_tokens.pt'
    if renderer_path.is_file():
        for sample_id, tokens in torch.load(renderer_path, map_location='cpu', weights_only=False).items():
            candidates.append((by_id[sample_id], 'scratch_renderer1000', tokens))
    rows, teacher_actions = [], {}
    with torch.no_grad(), amp():
        for sample, label, tokens in candidates:
            tokens = tokens.to(device)
            rollout_errors, executed_errors = [], []
            per_noise = [probe.loss(tokens, sample).item() for probe in probes]
            for seed in (20261114, 20261115, 20261116):
                key = (sample['id'], seed)
                shape = (1, teacher.horizon, teacher.action_dim)
                if key not in teacher_actions:
                    generator = torch.Generator(device=device).manual_seed(seed)
                    action = teacher.conditional_sample(shape, context=sample['target'].to(device), generator=generator)
                    teacher_actions[key] = teacher.normalizer['action'].unnormalize(action.float())
                generator = torch.Generator(device=device).manual_seed(seed)
                prediction = teacher.conditional_sample(shape, context=tokens, generator=generator)
                prediction = teacher.normalizer['action'].unnormalize(prediction.float())
                error = (prediction - teacher_actions[key]).square()
                rollout_errors.append(error.mean().sqrt().item())
                start = teacher.action_start_index
                executed_errors.append(error[:, start:start + teacher.n_action_steps].mean().sqrt().item())
            rows.append({'sample': sample['id'], 'split': sample['split'], 'phase': sample['phase'],
                         'variant': label, 'heldout_denoiser_mse': statistics.mean(per_noise),
                         'denoiser_mse_by_noise': per_noise,
                         'rollout_joint_teacher_rmse': statistics.mean(rollout_errors),
                         'rollout_joint_teacher_rmse_by_seed': rollout_errors,
                         'executed_chunk_joint_teacher_rmse': statistics.mean(executed_errors)})
            event('heldout_done', **rows[-1])
    json_write(output / 'heldout_behavior.json', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--pixel-steps', type=int, default=1000)
    parser.add_argument('--renderer-steps', type=int, default=1000)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    output = args.source / 'confirmation'
    output.mkdir(exist_ok=True)
    before = fingerprints()
    model, _, _, _, device, info = load_render_net(args.source / 'student_snapshot.ckpt', 'cuda:0')
    payload, cfg = common.load_checkpoint_payload(TEACHER)
    teacher = common.instantiate_policy(payload, cfg, device, capture_attention=False)
    teacher.eval().requires_grad_(False)
    model.eval().requires_grad_(False)
    del payload
    gc.collect()
    samples = torch.load(args.source / 'samples.pt', map_location='cpu', weights_only=False)
    probes = BehaviorProbes(teacher, samples, device)
    path = output / 'pixel_results.json'
    rows = json.loads(path.read_text()) if path.is_file() else []
    artifact = output / 'optimized_pixels.pt'
    fitted = torch.load(artifact, map_location='cpu', weights_only=False) if artifact.is_file() else {}
    for phase in range(3):
        sample = next(s for s in samples if s['split'] == 'validation' and s['phase'] == phase)
        for gate in ('predicted_gate', 'full_image'):
            if any(row['sample'] == sample['id'] and row['gate'] == gate for row in rows):
                continue
            row, images, tokens = fit_pixels(sample, teacher, probes, device, name=gate,
                                             steps=args.pixel_steps, lr=.03)
            rows.append(row)
            fitted[f'{gate}/{sample["id"]}'] = {'images': images, 'tokens': tokens}
            torch.save(fitted, artifact)
            json_write(path, rows)
    renderer_test(SimpleNamespace(renderer_steps=args.renderer_steps), samples,
                  model, teacher, probes, device, output)
    gradient_checks(samples, teacher, device, output)
    heldout_checks(samples, teacher, device, args.source, output)
    if fingerprints() != before:
        raise RuntimeError('Production files changed during independent confirmation')
    json_write(output / 'metadata.json', {'student_info': info, 'teacher': str(TEACHER),
               'pixel_steps': args.pixel_steps, 'renderer_steps': args.renderer_steps,
               'production_files_unchanged': True, 'production_hashes': before,
               'heldout_timesteps': [9, 33, 58, 88], 'heldout_noise_seeds': [7000, 8000, 9000],
               'rollout_seeds': [20261114, 20261115, 20261116],
               'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20})
    event('confirmation_complete', output=str(output), production_files_unchanged=True)


if __name__ == '__main__':
    main()
