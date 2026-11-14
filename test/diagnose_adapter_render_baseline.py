"""Read-only comparison of identity rendering and trained RGB distillation.

Replays the exact validation split/cache in training order, uses EMA weights
and BF16 as validation did, and writes diagnostics outside production runs.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from diffusion_policy.dataset.render_net_teacher_cache import CachedTeacherDataset, TeacherTargetCache
from diffusion_policy.scripts.visualize_cross_attention_episode import load_checkpoint_payload, instantiate_policy
from diffusion_policy.scripts.visualize_render_net_episode import load_render_net


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda:0')
    models, saved = {}, {}
    for name in ('best_condition', 'best', 'latest'):
        path = args.run_dir / 'checkpoints' / f'{name}.ckpt'
        models[name], _, cfg, teacher_cfg, _, _ = load_render_net(path, str(device))
        p = torch.load(path, map_location='cpu', pickle_module=dill, weights_only=False)
        saved[name] = dict(epoch=dill.loads(p['pickles']['epoch']),
                           step=dill.loads(p['pickles']['global_step']))
        del p
    base = models['best_condition']
    assert base.output.weight.count_nonzero() == base.output.bias.count_nonzero() == 0
    for name, model in models.items():
        for branch in ('context_adapter', 'condition_network'):
            ref = getattr(base, branch).state_dict()
            assert all(torch.equal(v, ref[k]) for k, v in getattr(model, branch).state_dict().items()), (name, branch)
    payload, teacher_cfg = load_checkpoint_payload(Path(cfg.teacher_checkpoint))
    teacher = instantiate_policy(payload, teacher_cfg, device, capture_attention=False)
    teacher.eval().requires_grad_(False)
    del payload
    cache_path = Path(json.loads((args.run_dir / 'teacher_cache.json').read_text())['splits']['validation'])
    meta = json.loads((cache_path / 'metadata.json').read_text())
    dataset = hydra.utils.instantiate(OmegaConf.create(meta['identity']['dataset']))
    validation = dataset.get_validation_dataset()
    assert len(validation) == meta['length']
    indices = np.ascontiguousarray(validation.indices)
    assert hashlib.sha256(indices.tobytes()).hexdigest() == meta['identity']['validation']['indices_sha256']
    cache = TeacherTargetCache(cache_path, meta['identity'], len(validation), token_dtype=meta['dtype'])
    loader = DataLoader(CachedTeacherDataset(validation, cache), batch_size=16,
                        shuffle=False, num_workers=4, pin_memory=True)
    totals, per_sample = defaultdict(lambda: defaultdict(float)), defaultdict(list)
    pixel_stats = defaultdict(lambda: defaultdict(float))
    started = time.monotonic()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            obs = {k: v.to(device, non_blocking=True) for k, v in batch['obs'].items()}
            target = batch['teacher_targets']['tokens'].to(device).float()
            zero = batch['teacher_targets']['zero_tokens'].to(device).float()
            patches = batch['teacher_targets']['patch_attention']
            response = torch.cat([patches[k].to(device) for k in teacher.obs_encoder.image_keys], -1).flatten(1)
            weights = (response.float() - float(cfg.distillation.response_noise_floor)).clamp_min(0)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                images, wrench, _, _ = teacher.prepare_context_inputs_with_future_contact(obs)
                if batch_index == 0:
                    unchanged = base(images, wrench)
                    assert all(torch.equal(unchanged[k], images[k]) for k in images)
                    print(json.dumps({'event': 'identity_initialization_verified',
                                      'adapter_and_condition_weights_frozen_across_checkpoints': True}), flush=True)
                variants = {
                    'original_without_adapter': teacher.obs_encoder.encode_image_context(images),
                    'original_with_adapter': teacher.obs_encoder.encode_image_context(images, context_adapter=base.context_adapter),
                    'teacher_zero_context': zero,
                }
                for name in ('best', 'latest'):
                    model = models[name]
                    rendered, region, condition = model(images, wrench, return_condition=True)
                    variants[name] = teacher.obs_encoder.encode_image_context(rendered, context_adapter=model.context_adapter)
                    for key in images:
                        delta = (rendered[key].float() - images[key].float()).abs()
                        pixel_stats[name]['absolute_sum'] += delta.double().sum().item()
                        pixel_stats[name]['elements'] += delta.numel()
                        pixel_stats[name]['max_abs'] = max(pixel_stats[name]['max_abs'], delta.max().item())
                    pixel_stats[name]['mask_open_sum'] += sum((v > model.region_threshold).sum().item() for v in region.values())
                    pixel_stats[name]['mask_elements'] += sum(v.numel() for v in region.values())
                    pixel_stats[name]['alpha_sum'] += condition['contact_alpha'].sum().item()
                    pixel_stats[name]['alpha_elements'] += condition['contact_alpha'].numel()
            for name, tokens in variants.items():
                errors = (tokens.float() - target).square().mean(-1)
                zero_errors = (tokens.float() - zero).square().mean(-1)
                t = totals[name]
                t['error_sum'] += errors.double().sum().item()
                t['zero_error_sum'] += zero_errors.double().sum().item()
                t['tokens'] += errors.numel()
                t['weighted_sum'] += (errors * weights).double().sum().item()
                t['weight_sum'] += weights.double().sum().item()
                per_sample[name].extend(errors.mean(-1).cpu().tolist())
            if batch_index % 30 == 0 or batch_index == len(loader) - 1:
                print(json.dumps({'batch': batch_index + 1, 'batches': len(loader),
                                  'elapsed_seconds': time.monotonic() - started}), flush=True)
    metrics = {}
    for name, t in totals.items():
        global_mse = t['error_sum'] / t['tokens']
        response_mse = t['weighted_sum'] / t['weight_sum']
        metrics[name] = dict(global_mse=global_mse, response_mse=response_mse,
                             student_loss=global_mse + response_mse,
                             zero_context_mse=t['zero_error_sum'] / t['tokens'])
    for name, t in pixel_stats.items():
        metrics[name].update(render_mean_absolute_delta=t['absolute_sum'] / t['elements'],
                             render_max_absolute_delta=t['max_abs'],
                             allowed_area_fraction=t['mask_open_sum'] / t['mask_elements'],
                             predicted_alpha_mean=t['alpha_sum'] / t['alpha_elements'])
    np.savez_compressed(args.output_dir / 'per_sample_global_mse.npz', **per_sample)
    summary = json.loads((args.run_dir / 'training_summary.json').read_text())
    assert abs(metrics['best']['student_loss'] - summary['best_val_mse']) < 2e-5
    assert abs(metrics['latest']['student_loss'] - summary['latest_val_mse']) < 2e-5
    result = dict(run_dir=str(args.run_dir.resolve()), samples=len(validation),
                  weights='ema_model', precision='BF16 with FP32 losses', checkpoints=saved,
                  rgb_initial_output_exact_identity=True, adapter_and_conditions_frozen=True,
                  validation_reproduced=True, metrics=metrics,
                  rgb_best_gain_relative_to_identity=1 - metrics['best']['student_loss'] / metrics['original_with_adapter']['student_loss'])
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    cache.close()
    dataset.lerobot_dataset.close()


if __name__ == '__main__':
    main()
