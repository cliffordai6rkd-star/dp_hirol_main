"""Independent diagnostics; never changes production networks or checkpoints.

Run from the repository root with the fdp Python environment. All fitted pixel
parameters, feature regressions and renderer updates belong to this process
only. The command snapshots the input student checkpoint in its output folder.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

from diffusion_policy.model.vision.render_net_attention import context_effect_regions
from diffusion_policy.model.vision.response_distillation import response_weighted_token_mse
from diffusion_policy.scripts import visualize_cross_attention_episode as common
from diffusion_policy.scripts.visualize_render_net_episode import load_render_net


TEACHER = ROOT / 'data/outputs/2026.11.12/20.05.59_876448_fdp_dis_board/teacher/checkpoints/optimizer_step=00180000-action_mse=0.00470914.ckpt'
PRODUCTION = (
    'diffusion_policy/model/vision/render_net.py',
    'diffusion_policy/model/vision/force_aware_obs_encoder.py',
    'diffusion_policy/policy/force_aware_diffusion_transformer_policy.py',
    'diffusion_policy/workspace/train_distill_render_net_workspace.py',
    'diffusion_policy/config/train_distill_render_net_student_response.yaml',
)


def fingerprints():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in PRODUCTION}


def json_write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def event(name, **values):
    print(json.dumps({'event': name, **values}, ensure_ascii=False), flush=True)


def amp():
    return torch.autocast('cuda', dtype=torch.bfloat16)


def cpu_dict(values):
    return {key: value.detach().cpu() for key, value in values.items()}


def images_of(sample, device):
    return {key: value.to(device) for key, value in sample['images'].items()}


def token_metrics(prediction, target, response):
    prediction, target, response = prediction.float(), target.float(), response.float()
    global_loss, weighted, numerator, denominator = response_weighted_token_mse(prediction, target, response)
    result = {
        'global_mse': global_loss.item(), 'response_mse': weighted.item(),
        'combined_loss': (global_loss + weighted).item(),
        'response_numerator': numerator.item(), 'response_weight_sum': denominator.item(),
        'cosine': F.cosine_similarity(prediction, target, dim=-1).mean().item(),
        'relative_mse': ((prediction - target).square().mean() / target.square().mean().clamp_min(1e-8)).item(),
        'student_rms': prediction.square().mean().sqrt().item(),
        'teacher_rms': target.square().mean().sqrt().item(),
    }
    if response.max() > 1e-6:
        indices = response.argsort(dim=-1, descending=True)
        error = (prediction - target).square().mean(dim=-1)
        count = max(1, error.shape[-1] // 10)
        result['top10_response_token_mse'] = error.gather(-1, indices[:, :count]).mean().item()
    return result


def select_samples(dataset, split, per_phase=2):
    result = []
    for phase in range(3):
        indices = np.flatnonzero(dataset.phase_labels == phase)
        grouped = {}
        for index in indices:
            anchor = dataset._anchor_index_from_sampler_row(dataset.indices[index])
            grouped.setdefault(int(dataset.episode_index[anchor]), []).append(int(index))
        episodes = sorted(grouped)
        positions = np.linspace(0, len(episodes) - 1, per_phase).round().astype(int)
        for position in positions:
            episode = episodes[int(position)]
            candidates = grouped[episode]
            index = candidates[len(candidates) // 2]
            # Avoid phase boundaries so both observation times have this phase.
            for offset in np.argsort(np.abs(np.arange(len(candidates)) - len(candidates) // 2)):
                candidate = candidates[int(offset)]
                sequence = dataset._sample_indices_to_sequence(candidate)
                if dataset.window_sampling_strategy == 'timestamp':
                    sequence = dataset._retime_sequence_indices(sequence)
                if np.all(dataset._frame_contact_phase[sequence[:dataset.n_obs_steps]] == phase):
                    index = candidate
                    break
            item = dataset[index]
            result.append({
                'id': f'{split}_phase{phase}_episode{episode}_index{index}',
                'split': split, 'phase': phase, 'episode': episode, 'dataset_index': index,
                'obs': {key: value.unsqueeze(0) for key, value in item['obs'].items()},
                'action': item['action'].unsqueeze(0),
            })
    return result


def prepare_samples(args, teacher, model, cfg, output, device):
    prepared = output / 'samples.pt'
    if prepared.is_file():
        return torch.load(prepared, map_location='cpu', weights_only=False)
    metadata = json.loads((args.cache / 'train/metadata.json').read_text())
    dataset_cfg = OmegaConf.create(metadata['identity']['dataset'])
    dataset = hydra.utils.instantiate(dataset_cfg)
    validation = dataset.get_validation_dataset()
    selected = select_samples(dataset, 'train', args.per_phase)
    selected += select_samples(validation, 'validation', args.per_phase)
    result = []
    for sample in selected:
        obs = {key: value.to(device) for key, value in sample.pop('obs').items()}
        images, wrench, _, _ = teacher.prepare_context_inputs_with_future_contact(obs)
        zero_obs = {key: value for key, value in obs.items() if key != 'contact_alpha'}
        for key in dict.fromkeys((teacher.wrench_key, teacher.phase_wrench_key, teacher.future_phase_wrench_key)):
            if key is not None:
                zero_obs[key] = torch.zeros_like(obs[key])
        with torch.no_grad():
            target = teacher._encode_context(obs, apply_curriculum_mask=False).float()
            zero_target = teacher._encode_context(zero_obs, apply_curriculum_mask=False).float()
            delta = target - zero_target
            response = delta.norm(dim=-1)
            patch_maps, offset = {}, 0
            shaped = response.reshape(1, teacher.n_obs_steps, -1)
            for key in teacher.obs_encoder.image_keys:
                count = teacher.obs_encoder.patch_counts[key]
                patch_maps[key] = shaped[..., offset:offset + count]
                offset += count
            oracle_masks = context_effect_regions(
                patch_maps, images, teacher.obs_encoder.backbone,
                selection='soft', normalization='image_minmax',
            )
            with amp():
                raw_tokens = teacher.obs_encoder.encode_image_context(images).float()
                ffn_tokens = teacher.obs_encoder.encode_image_context(images, apply_fusion_ffn=True).float()
                rendered, masks, condition = model(images, wrench, return_condition=True)
                student_tokens = teacher.obs_encoder.encode_image_context(rendered).float()
            predicted_alpha = condition['contact_alpha']
            true_alpha = obs['contact_alpha'][:, :teacher.n_obs_steps]
            sample.update({
                'images': cpu_dict(images), 'wrench': wrench.detach().cpu(),
                'target': target.cpu(), 'zero_target': zero_target.cpu(),
                'response': response.cpu(), 'masks': cpu_dict(masks), 'oracle_masks': cpu_dict(oracle_masks),
                'predicted_alpha': predicted_alpha.cpu(), 'true_alpha': true_alpha.cpu(),
                'raw_tokens': raw_tokens.cpu(), 'ffn_tokens': ffn_tokens.cpu(),
                'student_tokens': student_tokens.cpu(), 'rendered': cpu_dict(rendered),
                'teacher_force_mse': delta.square().mean().item(),
                'raw_zero_mse': F.mse_loss(raw_tokens, zero_target).item(),
                'ffn_zero_mse': F.mse_loss(ffn_tokens, zero_target).item(),
            })
            # Confirm this diagnostic uses the exact cached target contract.
            split_metadata = json.loads((args.cache / sample['split'] / 'metadata.json').read_text())
            array = np.load(args.cache / sample['split'] / split_metadata['arrays']['tokens']['file'], mmap_mode='r')
            cached = torch.from_numpy(np.array(array[sample['dataset_index']], copy=True)).float()
            sample['cache_target_mse'] = F.mse_loss(target.cpu().squeeze(0), cached).item()
        result.append(sample)
        event('sample_prepared', sample=sample['id'], alpha_true=true_alpha.tolist(),
              alpha_pred=predicted_alpha.tolist(), raw_zero_mse=sample['raw_zero_mse'],
              ffn_zero_mse=sample['ffn_zero_mse'], teacher_force_mse=sample['teacher_force_mse'])
    torch.save(result, prepared)
    del dataset, validation
    gc.collect()
    torch.cuda.empty_cache()
    return result


class BehaviorProbes:
    def __init__(self, teacher, samples, device, *, timesteps=(0, 25, 50, 75), noise_seed=1000):
        self.teacher, self.device, self.probes = teacher, device, {}
        for index, sample in enumerate(samples):
            generator = torch.Generator(device=device).manual_seed(noise_seed + index)
            action = teacher.normalizer['action'].normalize(sample['action'].to(device))
            noise = torch.randn(action.shape, device=device, generator=generator)
            values = []
            for time_step in timesteps:
                t = torch.tensor([time_step], device=device)
                noisy = teacher.noise_scheduler.add_noise(action, noise, t)
                with torch.no_grad(), amp():
                    output = teacher.model(noisy, t, sample['target'].to(device)).float()
                values.append((noisy, t, output))
            self.probes[sample['id']] = values

    def loss(self, tokens, sample, *, training=False, iteration=0):
        if training:
            values = self.probes[sample['id']]
            noisy, timestep, target = values[iteration % len(values)]
            prediction = self.teacher.model(noisy, timestep, tokens).float()
            return F.mse_loss(prediction, target)
        losses = []
        with torch.no_grad(), amp():
            for noisy, timestep, target in self.probes[sample['id']]:
                prediction = self.teacher.model(noisy, timestep, tokens).float()
                losses.append(F.mse_loss(prediction, target))
        return torch.stack(losses).mean()

    def evaluate(self, tokens, sample):
        with torch.no_grad(), amp():
            probe = self.loss(tokens, sample).item()
            # Matched initial diffusion noise isolates the context effect.
            predictions = []
            for context in (sample['target'].to(self.device), tokens):
                generator = torch.Generator(device=self.device).manual_seed(20261114)
                predictions.append(self.teacher.conditional_sample(
                    (1, self.teacher.horizon, self.teacher.action_dim),
                    context=context, generator=generator,
                ).float())
            teacher_action = self.teacher.normalizer['action'].unnormalize(predictions[0])
            student_action = self.teacher.normalizer['action'].unnormalize(predictions[1])
            return {
                'denoiser_mse': probe,
                'normalized_action_teacher_mse': F.mse_loss(predictions[1], predictions[0]).item(),
                'joint_action_teacher_rmse': F.mse_loss(student_action, teacher_action).sqrt().item(),
                'teacher_demo_joint_rmse': F.mse_loss(teacher_action, sample['action'].to(self.device)).sqrt().item(),
                'student_demo_joint_rmse': F.mse_loss(student_action, sample['action'].to(self.device)).sqrt().item(),
            }


def baseline_tests(samples, teacher, probes, output, device):
    rows = []
    for sample in samples:
        for name, key in (('raw_image', 'raw_tokens'), ('trained_student', 'student_tokens'),
                          ('teacher_zero_force', 'zero_target'), ('image_plus_frozen_ffn_diagnostic', 'ffn_tokens')):
            tokens = sample[key].to(device)
            row = {'sample': sample['id'], 'phase': sample['phase'], 'split': sample['split'], 'variant': name}
            row.update(token_metrics(tokens, sample['target'].to(device), sample['response'].to(device)))
            row.update(probes.evaluate(tokens, sample))
            row.update({key: sample[key] for key in ('teacher_force_mse', 'raw_zero_mse', 'ffn_zero_mse', 'cache_target_mse')})
            rows.append(row)
        event('baseline_done', sample=sample['id'], raw=rows[-4]['global_mse'], student=rows[-3]['global_mse'],
              force_only=rows[-2]['global_mse'], ffn_control=rows[-1]['global_mse'])
    # Offline linear regression is a diagnostic of feature coordinates only.
    # Train and validation episodes remain separate; this is not a model adapter.
    training = [sample for sample in samples if sample['split'] == 'train']
    x = torch.cat([sample['raw_tokens'].reshape(-1, 256) for sample in training]).double().to(device)
    y = torch.cat([sample['zero_target'].reshape(-1, 256) for sample in training]).double().to(device)
    x = torch.cat([x, torch.ones(x.shape[0], 1, device=device)], dim=1)
    penalty = torch.eye(x.shape[1], device=device, dtype=x.dtype) * 1e-2
    penalty[-1, -1] = 0
    mapping = torch.linalg.solve(x.T @ x + penalty, x.T @ y)
    regression = []
    for sample in samples:
        features = sample['raw_tokens'].to(device).double()
        features = torch.cat([features, torch.ones_like(features[..., :1])], dim=-1)
        predicted = (features @ mapping).float()
        regression.append({'sample': sample['id'], 'split': sample['split'], 'phase': sample['phase'],
                           'zero_target_mse': F.mse_loss(predicted, sample['zero_target'].to(device)).item(),
                           'original_zero_target_mse': sample['raw_zero_mse']})
    json_write(output / 'baseline.json', rows)
    json_write(output / 'offline_feature_regression.json', regression)
    event('baseline_complete', rows=len(rows))
    return rows


def gates_for(sample, name, device):
    masks = sample['masks']
    alpha = sample['predicted_alpha']
    if name == 'predicted_gate':
        pass
    elif name == 'predicted_region_alpha_one':
        alpha = torch.ones_like(alpha)
    elif name == 'teacher_region_true_alpha':
        masks, alpha = sample['oracle_masks'], sample['true_alpha']
    elif name == 'full_image':
        masks = {key: torch.ones_like(value) for key, value in masks.items()}
        alpha = torch.ones_like(alpha)
    else:
        raise ValueError(name)
    return {key: mask.to(device) * alpha.to(device)[..., None, None, None] for key, mask in masks.items()}


def fit_pixels(sample, teacher, probes, device, *, name, steps, lr, objective='token', seed=42):
    images = images_of(sample, device)
    gates = gates_for(sample, name, device)
    target, response = sample['target'].to(device), sample['response'].to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    parameters = torch.nn.ParameterDict({
        key: torch.nn.Parameter(torch.zeros_like(image) if seed == 42 else
                                .02 * torch.randn(image.shape, device=device, generator=generator))
        for key, image in images.items()
    })
    optimizer = torch.optim.Adam(parameters.parameters(), lr=lr)
    history, best, best_step, best_images = [], float('inf'), 0, None
    started = time.monotonic()
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        rendered = {key: (image + gates[key] * parameters[key].tanh()).clamp(0, 1)
                    for key, image in images.items()}
        with amp():
            tokens = teacher.obs_encoder.encode_image_context(rendered, input_grad=True)
            global_loss, response_loss, _, _ = response_weighted_token_mse(tokens, target, response)
            token_loss = global_loss + response_loss
            if objective == 'behavior':
                loss = probes.loss(tokens, sample, training=True, iteration=step)
            else:
                loss = token_loss
        # Best selection for behavior uses all four fixed probes, not the
        # currently sampled diffusion time alone.
        score = probes.loss(tokens.detach(), sample).item() if objective == 'behavior' else token_loss.item()
        if score < best:
            best, best_step = score, step
            best_images = cpu_dict(rendered)
        if step % 25 == 0 or step == steps:
            record = {'step': step, 'objective': loss.item(), 'global_mse': global_loss.item(),
                      'response_mse': response_loss.item(), 'selection': score}
            history.append(record)
            event('pixel_progress', sample=sample['id'], gate=name, objective_type=objective, **record,
                  seconds=round(time.monotonic() - started, 1))
        if step == steps:
            break
        loss.backward()
        optimizer.step()
    with torch.no_grad(), amp():
        rendered = {key: value.to(device) for key, value in best_images.items()}
        tokens = teacher.obs_encoder.encode_image_context(rendered).float()
    result = {'sample': sample['id'], 'split': sample['split'], 'phase': sample['phase'], 'gate': name,
              'objective': objective, 'steps': steps, 'lr': lr, 'seed': seed, 'best_step': best_step,
              'seconds': time.monotonic() - started, 'history': history,
              **token_metrics(tokens, target, response), **probes.evaluate(tokens, sample)}
    changes = torch.cat([(rendered[key] - images[key]).flatten() for key in images])
    result.update(pixel_abs_mean=changes.abs().mean().item(), pixel_abs_max=changes.abs().max().item(),
                  clip_boundary_fraction=torch.cat([value.flatten() for value in rendered.values()]).le(.0001).float().mean().item(),
                  effective_gate_mean=torch.cat([value.flatten() for value in gates.values()]).mean().item())
    return result, best_images, tokens.detach().cpu()


def pixel_tests(args, samples, teacher, probes, device, output):
    path = output / 'pixel_results.json'
    rows = json.loads(path.read_text()) if path.is_file() else []
    artifact = output / 'optimized_pixels.pt'
    fitted = torch.load(artifact, map_location='cpu', weights_only=False) if artifact.is_file() else {}
    # One example of every phase from each split: six independent inversions.
    selected = []
    for split in ('train', 'validation'):
        for phase in range(3):
            selected.append(next(s for s in samples if s['split'] == split and s['phase'] == phase))
    for name in ('predicted_gate', 'predicted_region_alpha_one', 'teacher_region_true_alpha', 'full_image'):
        for sample in selected:
            if any(row['sample'] == sample['id'] and row['gate'] == name and row['objective'] == 'token' for row in rows):
                continue
            row, images, tokens = fit_pixels(sample, teacher, probes, device, name=name,
                                             steps=args.pixel_steps, lr=args.pixel_lr)
            rows.append(row)
            fitted[f'{name}/{sample["id"]}'] = {'images': images, 'tokens': tokens}
            json_write(path, rows)
            torch.save(fitted, output / 'optimized_pixels.pt')
            event('pixel_done', **{key: value for key, value in row.items() if key != 'history'})
    return rows


def behavior_tests(args, samples, teacher, probes, device, output):
    rows = []
    for phase in (1, 2):
        sample = next(s for s in samples if s['split'] == 'validation' and s['phase'] == phase)
        row, images, tokens = fit_pixels(sample, teacher, probes, device, name='predicted_gate',
                                         steps=args.pixel_steps, lr=.03, objective='behavior')
        rows.append(row)
        torch.save({'images': images, 'tokens': tokens}, output / f'behavior_phase{phase}.pt')
        json_write(output / 'behavior_results.json', rows)
        event('behavior_done', **{key: value for key, value in row.items() if key != 'history'})
    return rows


def renderer_test(args, samples, model, teacher, probes, device, output):
    """Fit a scratch copy of the unchanged RGB renderer on three examples."""
    scratch = copy.deepcopy(model)
    scratch.set_training_stage('rgb')
    scratch.set_stage_mode(True)
    training = [next(s for s in samples if s['split'] == 'train' and s['phase'] == phase) for phase in range(3)]
    validation = [next(s for s in samples if s['split'] == 'validation' and s['phase'] == phase) for phase in range(3)]
    optimizer = torch.optim.AdamW(scratch.rgb_parameters(), lr=2e-4, weight_decay=1e-4)
    history = []
    for step in range(args.renderer_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for sample in training:
            with amp():
                rendered = scratch(images_of(sample, device), sample['wrench'].to(device))
                tokens = teacher.obs_encoder.encode_image_context(rendered, input_grad=True)
                global_loss, weighted, _, _ = response_weighted_token_mse(
                    tokens, sample['target'].to(device), sample['response'].to(device))
                loss = global_loss + weighted
            total += loss.item() / len(training)
            if step < args.renderer_steps:
                (loss / len(training)).backward()
        if step % 25 == 0 or step == args.renderer_steps:
            history.append({'step': step, 'train_loss': total})
            event('renderer_progress', step=step, train_loss=total)
        if step < args.renderer_steps:
            torch.nn.utils.clip_grad_norm_(scratch.rgb_parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
    scratch.eval()
    rows, fitted_tokens = [], {}
    with torch.no_grad():
        for sample in training + validation:
            with amp():
                rendered = scratch(images_of(sample, device), sample['wrench'].to(device))
                tokens = teacher.obs_encoder.encode_image_context(rendered).float()
            rows.append({'sample': sample['id'], 'phase': sample['phase'], 'split': sample['split'],
                         **token_metrics(tokens, sample['target'].to(device), sample['response'].to(device)),
                         **probes.evaluate(tokens, sample)})
            fitted_tokens[sample['id']] = tokens.cpu()
    json_write(output / 'renderer_results.json', {'history': history, 'samples': rows})
    torch.save(fitted_tokens, output / 'renderer_tokens.pt')
    del scratch
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student-checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=ROOT / 'data/cache/render_net_teacher/1db0fc5ebd31be95197cf981')
    parser.add_argument('--stages', nargs='+', default=['baseline', 'pixels', 'renderer', 'behavior'],
                        choices=['baseline', 'pixels', 'renderer', 'behavior'])
    parser.add_argument('--pixel-steps', type=int, default=100)
    parser.add_argument('--pixel-lr', type=float, default=.03)
    parser.add_argument('--renderer-steps', type=int, default=100)
    parser.add_argument('--per-phase', type=int, default=2)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    original_hashes = fingerprints()
    snapshot = output / 'student_snapshot.ckpt'
    if not snapshot.is_file():
        shutil.copyfile(args.student_checkpoint, snapshot)
    # The visualization loader needs the resolved teacher config beside the
    # snapshot; it otherwise reads the exact pinned teacher checkpoint.
    model, preprocessor, cfg, teacher_cfg, device, info = load_render_net(snapshot, 'cuda:0')
    teacher_payload, teacher_cfg = common.load_checkpoint_payload(TEACHER)
    teacher = common.instantiate_policy(teacher_payload, teacher_cfg, device, capture_attention=False)
    teacher.eval().requires_grad_(False)
    del teacher_payload
    model.eval().requires_grad_(False)
    gc.collect()
    metadata = {'student_source': str(args.student_checkpoint.resolve()), 'student_snapshot': str(snapshot),
                'student_info': info, 'teacher': str(TEACHER), 'device': str(device),
                'pixel_steps': args.pixel_steps, 'renderer_steps': args.renderer_steps,
                'production_hashes': original_hashes, 'created_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    json_write(output / 'metadata.json', metadata)
    samples = prepare_samples(args, teacher, model, cfg, output, device)
    probes = BehaviorProbes(teacher, samples, device)
    if 'baseline' in args.stages:
        baseline_tests(samples, teacher, probes, output, device)
    if 'pixels' in args.stages:
        pixel_tests(args, samples, teacher, probes, device, output)
    if 'renderer' in args.stages:
        renderer_test(args, samples, model, teacher, probes, device, output)
    if 'behavior' in args.stages:
        behavior_tests(args, samples, teacher, probes, device, output)
    if fingerprints() != original_hashes:
        raise RuntimeError('Production files changed during independent diagnostics')
    metadata.update(completed_at=time.strftime('%Y-%m-%d %H:%M:%S'), production_files_unchanged=True,
                    gpu_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
    json_write(output / 'metadata.json', metadata)
    event('complete', output=str(output), production_files_unchanged=True,
          gpu_peak_allocated_mib=metadata['gpu_peak_allocated_mib'])


if __name__ == '__main__':
    main()
