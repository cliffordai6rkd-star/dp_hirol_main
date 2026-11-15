"""Read-only FP32 evaluation for CIVI RGB -> frozen Teacher DiT distillation."""
from pathlib import Path
import csv
import json
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from civi_budget_extension_eval import fp32_evaluation
from reevaluate_civi_convergence import pad_batch, tensors
from student_dual_path_common import to_device, tensor_hash, write_json
from run_scene_torque_response_predictor import SceneTorquePredictor
from diffusion_policy.model.vision.render_net_attention import zero_wrench_reference

PHASES = ('free', 'precontact', 'contact')


def energy(value):
    return value.double().flatten(1).square().mean(1)


class ActionImages(Dataset):
    def __init__(self, paired):
        self.paired, self.dataset = paired, paired.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        return dict(obs=item['obs'], action=item['action'], index=index,
                    episode=int(self.paired.episodes[index]),
                    phase=int(self.dataset.phase_labels[index]))


def ratio(a, b):
    return a / b if b > 0 else None


def aggregate(rows):
    result = dict(samples=len(rows), episodes=len({r['episode'] for r in rows}))
    if not rows:
        return result
    for key in ('dit_mse', 'dit_identity_mse', 'dit_initial_mse', 'dit_teacher_energy',
                'predictor_mse', 'teacher_mse', 'predictor_energy', 'teacher_energy',
                'response_energy', 'rgb_energy', 'rgb_update_energy',
                'hard_clip_fraction', 'range_clip_fraction', 'alpha_mean'):
        result[key] = float(np.mean([r[key] for r in rows]))
    for name in ('response', 'rgb', 'rgb_update'):
        result[name + '_rms'] = result[name + '_energy'] ** .5
    result['rgb_linf'] = max(r['rgb_linf'] for r in rows)
    result['dit_error_identity_ratio'] = ratio(result['dit_mse'], result['dit_identity_mse'])
    result['dit_error_initial_ratio'] = ratio(result['dit_mse'], result['dit_initial_mse'])
    active = sum(r['active_patch_count'] for r in rows)
    for target in ('predictor', 'teacher'):
        result[target + '_error_zero_ratio'] = ratio(result[target + '_mse'], result[target + '_energy'])
        result['cosine_' + target] = ratio(sum(r['cosine_' + target + '_sum'] for r in rows), active)
        amplitude = ratio(sum(r['active_response_ss'] for r in rows), sum(r['active_' + target + '_ss'] for r in rows))
        result['amplitude_ratio_' + target] = amplitude ** .5 if amplitude is not None else None
    return result


def summarize(rows):
    summary, episodes = {}, []
    scopes = dict(train=lambda r: r['split'] == 'train',
                  validation_internal=lambda r: r['split'] == 'validation' and r['episode'] not in (5, 26),
                  validation_all=lambda r: r['split'] == 'validation',
                  retained_5_26=lambda r: r['split'] == 'validation' and r['episode'] in (5, 26))
    keys = ('dit_error_identity_ratio', 'dit_error_initial_ratio',
            'predictor_error_zero_ratio', 'teacher_error_zero_ratio')
    for scope, choose in scopes.items():
        summary[scope] = {}
        for phase, name in enumerate(PHASES):
            selected = [r for r in rows if choose(r) and r['phase'] == phase]
            per_episode = [dict(scope=scope, phase=name, episode=ep,
                                **aggregate([r for r in selected if r['episode'] == ep]))
                           for ep in sorted({r['episode'] for r in selected})]
            episodes.extend(per_episode)
            value = aggregate(selected)
            if selected:
                for key in keys:
                    defined = [r[key] for r in per_episode if r[key] is not None]
                    value[key + '_episode_macro'] = float(np.mean(defined)) if defined else None
                    value[key + '_defined_episodes'] = len(defined)
            summary[scope][name] = value
    return summary, episodes


class Evaluator:
    def __init__(self, experiment):
        self.x, self.r = experiment, experiment.r
        r = self.r
        state = torch.load(experiment.a.predictor / 'best.pt', map_location='cpu', weights_only=False)['model']
        self.predictor = SceneTorquePredictor(state['visual_scale'], state['torque_scale']).to(r.device).eval().requires_grad_(False)
        self.predictor.load_state_dict(state, strict=True)
        self.predictor_hash = tensor_hash(self.predictor.state_dict())
        self.mode, self.precision, self.hooks = 'prepare', {}, []
        modules = dict(dino=r.teacher.obs_encoder.backbone,
                       projection=r.teacher.obs_encoder.image_projection,
                       adapter=r.owner.context_adapter, context_norm=r.teacher.obs_encoder.context_norm,
                       condition=r.owner.condition_network, teacher_context=r.teacher.obs_encoder,
                       dit=r.teacher.model, predictor=self.predictor,
                       renderer_online=experiment.network, renderer_ema=experiment.ema)
        for name, module in modules.items():
            def hook(_module, args, kwargs, output, name=name):
                record = self.precision.setdefault(self.mode, {}).setdefault(name, dict(inputs=set(), outputs=set()))
                for key, values in (('inputs', (args, kwargs)), ('outputs', output)):
                    for value in tensors(values):
                        if value.is_floating_point():
                            record[key].add(str(value.dtype))
                            if self.mode == 'fp32_evaluation':
                                assert value.dtype == torch.float32, (name, value.dtype)
            self.hooks.append(module.register_forward_hook(hook, with_kwargs=True))
        self.visual_plan = {}
        for split in ('train', 'validation'):
            paired = r.datasets[split]
            phases = (2,) if split == 'train' else (0, 1, 2)
            ids = []
            for phase in phases:
                seen = set()
                for index in np.flatnonzero(paired.dataset.phase_labels == phase):
                    ep = int(paired.episodes[index])
                    if split == 'validation' and ep in (5, 26):
                        continue
                    if ep not in seen:
                        ids.append(int(index)); seen.add(ep)
                    if len(seen) == 2:
                        break
            self.visual_plan[split] = ids
        write_json(experiment.out / 'visual_indices.json', self.visual_plan)

    def encode(self, images, adapter=None):
        return self.r.teacher.obs_encoder.encode_image_context(
            images, input_grad=False,
            context_adapter=adapter if adapter is not None else self.r.owner.context_adapter)

    def evaluate(self, step, train_contact=False):
        x, r = self.x, self.r
        self.mode = 'fp32_evaluation'
        rows, visuals = [], []
        started = time.monotonic()
        with fp32_evaluation():
            for split in (('train', 'validation') if train_contact else ('validation',)):
                dataset = ActionImages(r.datasets[split]); visited = []
                loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4,
                                    pin_memory=True, generator=torch.Generator().manual_seed(0))
                for ordinal, raw in enumerate(loader):
                    size = len(raw['index'])
                    if split == 'train' and not (raw['phase'] == 2).any():
                        continue
                    selected = [j for j in range(size) if split != 'train' or int(raw['phase'][j]) == 2]
                    batch = to_device(pad_batch(raw, 8-size), r.device)
                    images, wrench = r.inputs(batch)
                    images = {key: value.float() for key, value in images.items()}
                    captured = []
                    def adapter(visual):
                        captured.append(visual)
                        return r.owner.context_adapter(visual)
                    z0 = self.encode(images, adapter)
                    visual = captured[0]
                    pred = self.predictor(visual, visual.reshape(8, 2, 392, 256).mean(2),
                                          batch['obs'][r.teacher.wrench_key][:, :2].float(),
                                          torch.arange(784, device=r.device))
                    teacher = r.teacher._encode_context(batch['obs'], apply_curriculum_mask=False)
                    zero = r.teacher._encode_context(zero_wrench_reference(r.teacher, batch['obs']), apply_curriculum_mask=False)
                    response_target = teacher-zero
                    masks, condition = r.owner.condition_network(images, wrench)
                    spatial = {key:r.owner._mask_gate(masks[key].flatten(0,1), images[key]).reshape_as(masks[key]) for key in images}
                    alpha = condition['contact_alpha']
                    rgb, stats = x.ema(images, pred, spatial, alpha, trace=True)
                    z = self.encode(rgb)
                    initial_rgb = rgb if step == 0 else x.initial(images, pred, spatial, alpha)
                    initial_z = z if step == 0 else self.encode(initial_rgb)
                    action = r.teacher.normalizer['action'].normalize(batch['action'].float())
                    losses = dict(dit_mse=[], dit_identity_mse=[], dit_initial_mse=[], dit_teacher_energy=[])
                    for probe in range(x.a.eval_probes):
                        seed = x.a.validation_seed + (0 if split == 'train' else 10_000_000) + ordinal*1009 + probe*100_000_019
                        generator = torch.Generator(device=r.device).manual_seed(seed)
                        noise = torch.randn(action.shape, device=r.device, dtype=action.dtype, generator=generator)
                        times = torch.randint(0, r.teacher.noise_scheduler.config.num_train_timesteps,
                                              (8,), device=r.device, generator=generator).long()
                        noisy = r.teacher.noise_scheduler.add_noise(action, noise, times)
                        expected = r.teacher.model(noisy, times, teacher)
                        current = r.teacher.model(noisy, times, z)
                        identity = r.teacher.model(noisy, times, z0)
                        initial = current if step == 0 else r.teacher.model(noisy, times, initial_z)
                        for key, value in dict(dit_mse=energy(current-expected),
                                               dit_identity_mse=energy(identity-expected),
                                               dit_initial_mse=energy(initial-expected),
                                               dit_teacher_energy=energy(expected)).items():
                            losses[key].append(value)
                    metrics = {key:torch.stack(values).mean(0) for key,values in losses.items()}
                    response = z-z0; active = response_target.square().mean(-1).sqrt() > r.threshold
                    metrics.update(predictor_mse=energy(response-pred), teacher_mse=energy(response-response_target),
                                   predictor_energy=energy(pred), teacher_energy=energy(response_target),
                                   response_energy=energy(response),
                                   rgb_energy=torch.stack([energy(rgb[key]-images[key]) for key in images]).mean(0),
                                   rgb_update_energy=torch.stack([energy(rgb[key]-initial_rgb[key]) for key in images]).mean(0),
                                   rgb_linf=torch.stack([(rgb[key]-images[key]).abs().flatten(1).max(1).values for key in images]).max(0).values,
                                   hard_clip_fraction=stats['hard_clipped_channels']/stats['allowed_channels'].clamp_min(1),
                                   range_clip_fraction=stats['range_clipped_channels']/stats['allowed_channels'].clamp_min(1),
                                   alpha_mean=alpha.double().flatten(1).mean(1), active_patch_count=active.sum(1),
                                   active_response_ss=(response.double().square().sum(-1)*active).sum(1))
                    for name, target in (('predictor', pred), ('teacher', response_target)):
                        metrics['cosine_'+name+'_sum'] = (F.cosine_similarity(response, target, dim=-1, eps=1e-6)*active).double().sum(1)
                        metrics['active_'+name+'_ss'] = (target.double().square().sum(-1)*active).sum(1)
                    for key in images:
                        support = spatial[key]*alpha[...,None,None,None]
                        assert ((rgb[key]-images[key]).abs()-.1*support).max() <= 2e-7
                        closed = (support == 0).expand_as(rgb[key])
                        assert torch.equal(rgb[key][closed], images[key][closed])
                    assert all(torch.isfinite(v).all() for v in metrics.values()), 'Nonfinite FP32 evaluation'
                    values = {key:value.cpu().tolist() for key,value in metrics.items()}
                    for j in selected:
                        index = int(raw['index'][j])
                        rows.append(dict(step=step, split=split, index=index,
                                         episode=int(raw['episode'][j]), phase=int(raw['phase'][j]),
                                         **{key:value[j] for key,value in values.items()}))
                        visited.append(index)
                        if index in self.visual_plan[split]:
                            visuals.append(dict(split=split, index=index, episode=int(raw['episode'][j]),
                                                phase=int(raw['phase'][j]),
                                                original={key:images[key][j,-1].cpu() for key in images},
                                                initial={key:initial_rgb[key][j,-1].cpu() for key in images},
                                                rendered={key:rgb[key][j,-1].cpu() for key in images},
                                                support={key:(spatial[key]*alpha[...,None,None,None])[j,-1].cpu() for key in images}))
                    if ordinal % 50 == 0:
                        print(json.dumps(dict(event='fp32_eval', step=step, split=split,
                                              windows=len(visited), elapsed=time.monotonic()-started)), flush=True)
                expected = np.flatnonzero(dataset.dataset.phase_labels == 2).tolist() if split == 'train' else list(range(len(dataset)))
                assert visited == expected, (split, len(visited), len(expected))
        summary, episodes = summarize(rows)
        with (x.out/f'fp32_windows_{step:06d}.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        write_json(x.out/f'metrics_{step:06d}.json', summary)
        write_json(x.out/f'episodes_{step:06d}.json', episodes)
        torch.save(visuals, x.out/f'visuals_{step:06d}.pt')
        assert tensor_hash(self.predictor.state_dict()) == self.predictor_hash
        self.write_precision(); self.mode = 'between_evaluations'
        return summary

    def write_precision(self):
        write_json(self.x.out/'precision.json', {mode:{name:{key:sorted(values) for key,values in record.items()}
                                                      for name,record in names.items()}
                                               for mode,names in self.precision.items()})
