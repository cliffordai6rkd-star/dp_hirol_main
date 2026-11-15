"""Tiny train-only pixel / individual renderer / shared renderer diagnostic.

No production code or checkpoint is written. Nine fresh optimizers, <=20 steps
each. Endpoint displacement scales are fixed before any losses are measured.
"""
from pathlib import Path
import argparse
import json
import math
import os
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf

from run_response_decoder_ablation import Runtime
from run_scene_torque_response_predictor import SceneTorquePredictor
from student_dual_path_common import tensor_hash, write_json
from paired_response_full_common import file_hash, deterministic_mode
from diffusion_policy.model.vision.render_net_attention import zero_wrench_reference


SCALES = (0.01, 0.03, 0.1, 0.3, 1.0)
OBSERVED_STEPS = (0, 1, 5, 10, 20)


def event(name, **values):
    print(json.dumps(dict(event=name, **values)), flush=True)


def select_windows(dataset):
    """Episode quantiles then phase midpoint; uses indices/labels only."""
    phases, episodes = dataset.dataset.phase_labels, dataset.episodes
    result = []
    used = set()
    for phase, count, label in ((2, 4, 'C'), (0, 2, 'F')):
        eligible = sorted(set(episodes[phases == phase]) - used)
        chosen = [eligible[i] for i in np.linspace(0, len(eligible)-1, count).round().astype(int)]
        for ordinal, episode in enumerate(chosen, 1):
            indices = np.flatnonzero((episodes == episode) & (phases == phase))
            result.append(dict(label=f'{label}{ordinal}', index=int(indices[len(indices)//2]),
                               episode=int(episode), phase=phase, split='train'))
        used.update(chosen)
    assert len(result) == 6 and len({x['episode'] for x in result}) == 6
    return result


def cosine(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    denominator = float(a.norm()*b.norm())
    return float(a.dot(b))/denominator if denominator > 1e-30 else None


def mean_metrics(rows):
    keys = ('predicted_mse', 'teacher_mse', 'cosine_predicted', 'cosine_teacher',
            'response_rms', 'response_change_rms', 'update_rms', 'total_rms',
            'hard_clip_fraction', 'range_clip_fraction', 'bound_saturated_fraction')
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    for k in ('response_rms', 'response_change_rms', 'update_rms', 'total_rms'):
        out[k] = math.sqrt(np.mean([r[k]**2 for r in rows]))
    return out


class Study:
    def __init__(self, args):
        self.a = args
        self.started = time.monotonic()
        self.updates = []
        self.dependencies = json.loads(args.dependencies.read_text())
        assert self.dependencies['teacher_step'] == 150000
        config = OmegaConf.load(Path(__file__).parent/'config/response_decoder_gate_ablation.yaml')
        for key in ('teacher_sha256', 'predictor_sha256', 'condition_sha256', 'decoder_initial_sha256'):
            config[key] = self.dependencies[key]
        config.strict_reference = False
        args.variant, args.smoke = 'A', False
        self.r = Runtime(args, config)
        r = self.r
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        assert file_hash(args.checkpoint) == 'fec1fc56443602efc5be1ff84af00afd787f9363f30dcb3e572e64e84e3683d5'
        for key in ('teacher_sha256', 'predictor_sha256', 'adapter_sha256', 'decoder_initial_sha256'):
            assert saved['manifest'][key] == self.dependencies[key]
        r.network.load_state_dict(saved['ema_model'], strict=True)
        r.network.float().eval()
        r.teacher.float().eval().requires_grad_(False)
        deterministic_mode(r.network)
        self.names = [n for n, p in r.network.named_parameters() if p.requires_grad]
        assert self.names == saved['manifest']['trainable_names']
        self.params = [p for p in r.network.parameters() if p.requires_grad]
        self.initial = {k: v.detach().cpu().clone() for k, v in r.network.state_dict().items()}
        self.initial_hash = tensor_hash(self.initial)
        r.check_frozen()
        state = torch.load(args.predictor/'best.pt', map_location='cpu', weights_only=False)['model']
        self.predictor = SceneTorquePredictor(state['visual_scale'], state['torque_scale']).to(r.device).eval()
        self.predictor.load_state_dict(state, strict=True)
        self.predictor.requires_grad_(False)
        self.predictor_hash = tensor_hash(self.predictor.state_dict())
        self.selection = select_windows(r.datasets['train'])
        # Persist the index-only selection and all analysis choices before a forward/loss.
        self.protocol = dict(selection=self.selection, steps=args.steps, seed=42,
            epsilon=0.1, endpoint_displacement_scales=SCALES, observed_steps=OBSERVED_STEPS,
            network_optimizer=dict(type='AdamW', lr=args.network_lr, betas=[0.9, 0.999],
                eps=1e-8, weight_decay=1e-4, gradient_clip=1.0, fresh_state_each_instance=True),
            pixel_optimizer=dict(type='Adam', lr=args.pixel_lr, betas=[0.9, 0.999], eps=1e-8),
            objective='mean over ALL response tokens and channels of (E(RGB)-E(original)-dP)^2',
            precision='FP32 renderer, gates, DINO, projection, Adapter, context_norm, Predictor, Teacher and losses; TF32 disabled',
            predictor_target='Recomputed once from FP32 original visual tokens and frozen matched Predictor; fixed/detached thereafter',
            teacher_response='Recomputed FP32 Teacher(nonzero wrench)-Teacher(physical-zero wrench)',
            free='Monitor only; same predicted-response MSE used ONLY for the initial gradient comparison',
            shared='microbatch=1; sum of 4 losses divided by 4; exactly one optimizer.step per iteration',
            matching='Choose one fixed endpoint scale per group solely by pooled Contact update RMS nearest the minimum unscaled group RMS in log space; report mismatch >25%',
            scales_semantics='A: RGB displacement from common initial render; B/C: parameter displacement from initial EMA, then normal renderer forward',
            checkpoint=str(args.checkpoint), checkpoint_sha256=file_hash(args.checkpoint), checkpoint_step=saved['step'],
            weights='ema_model', initial_state_sha256=self.initial_hash, trainable_names=self.names,
            trainable_parameters=sum(p.numel() for p in self.params), dependencies=self.dependencies,
            source_sha256=file_hash(Path(__file__)), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            gpu=torch.cuda.get_device_name(), start_gpu_query=subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv'], text=True))
        write_json(args.output/'protocol.json', self.protocol)
        self.samples = []
        for meta in self.selection:
            b = r.batch('train', [meta['index']])
            with torch.no_grad():
                im, wr = r.inputs(b)
                im = {k: v.float().detach() for k, v in im.items()}
                seen = []
                def capture(z):
                    seen.append(z.detach())
                    return r.owner.context_adapter(z)
                z0 = r.teacher.obs_encoder.encode_image_context(im, context_adapter=capture).float()
                assert z0.dtype == torch.float32 and len(seen) == 1
                visual = seen[0]
                pred = self.predictor(visual, visual.reshape(1, 2, 392, 256).mean(2),
                    b['obs'][r.teacher.wrench_key][:, :2].float(), torch.arange(784, device=r.device)).detach()
                masks, condition = r.owner.condition_network(im, wr)
                spatial = {k: r.owner._mask_gate(masks[k].flatten(0, 1), im[k]).reshape_as(masks[k]).float().detach() for k in im}
                alpha = condition['contact_alpha'].float().detach()
                support = {k: spatial[k]*alpha[..., None, None, None] for k in im}
                target = r.teacher._encode_context(b['obs'], apply_curriculum_mask=False).float()
                zero = r.teacher._encode_context(zero_wrench_reference(r.teacher, b['obs']), apply_curriculum_mask=False).float()
                rendered, stats = r.network(im, pred, spatial, alpha, trace=True)
                rendered = {k: v.detach() for k, v in rendered.items()}
                base_response = self.encode(rendered)-z0
                cached_pred = r.predicted(b, 'train')
                s = dict(**meta, images=im, spatial=spatial, alpha=alpha, support=support,
                    lower={k: (v-0.1*support[k]).clamp(0, 1) for k, v in im.items()},
                    upper={k: (v+0.1*support[k]).clamp(0, 1) for k, v in im.items()},
                    pred=pred, teacher=(target-zero).detach(), z0=z0.detach(),
                    initial_rgb=rendered, initial_response=base_response.detach(), initial_stats=stats,
                    cached_prediction_mse=float((pred-cached_pred).square().mean()),
                    teacher_cache_mse=float(((target-zero)-(b['target'].float()-b['zero'].float())).square().mean()))
                self.samples.append(s)
        self.baseline = [self.measure(s, s['initial_rgb'], s['initial_stats']) for s in self.samples]
        write_json(args.output/'baseline.json', self.baseline)
        # Verify deterministic FP32 readout in both grad-enabled and no-grad paths.
        for s in self.samples:
            rgb = self.render(s)
            assert max(float((rgb[k]-s['initial_rgb'][k]).abs().max()) for k in rgb) == 0
            response = self.encode(rgb, True)-s['z0']
            assert float((response-s['initial_response']).abs().max()) == 0
        self.restore()
        event('prepared', selection=self.selection, initial_contact=mean_metrics(self.baseline[:4]))

    def restore(self):
        self.r.network.zero_grad(set_to_none=True)
        self.r.network.load_state_dict(self.initial, strict=True)

    def encode(self, images, grad=False):
        return self.r.teacher.obs_encoder.encode_image_context(images,
            context_adapter=self.r.owner.context_adapter, input_grad=grad).float()

    def render(self, sample, trace=False):
        return self.r.network(sample['images'], sample['pred'], sample['spatial'], sample['alpha'], trace=trace)

    def loss(self, sample, rgb):
        return (self.encode(rgb, True)-sample['z0']-sample['pred']).square().mean()

    @torch.no_grad()
    def measure(self, s, rgb, stats=None, pixel_clip=None):
        response = self.encode(rgb)-s['z0']
        update = torch.cat([(rgb[k]-s['initial_rgb'][k]).flatten() for k in rgb])
        total = torch.cat([(rgb[k]-s['images'][k]).flatten() for k in rgb])
        allowed = saturated = 0
        for k, v in rgb.items():
            mask = (s['support'][k] > 0).expand_as(v)
            allowed += int(mask.sum())
            saturated += int((((v-s['lower'][k]).abs() < 2e-7) | ((v-s['upper'][k]).abs() < 2e-7))[mask].sum())
            assert float((s['lower'][k]-v).max()) <= 2e-7
            assert float((v-s['upper'][k]).max()) <= 2e-7
            assert torch.equal(v[~mask], s['images'][k][~mask])
        if stats is not None:
            den = max(float(stats['allowed_channels'][0]), 1)
            hard = float(stats['hard_clipped_channels'][0])/den
            range_clip = float(stats['range_clipped_channels'][0])/den
        else:
            hard, range_clip = pixel_clip or (0.0, 0.0)
        return dict(label=s['label'], index=s['index'], episode=s['episode'], phase=s['phase'],
            predicted_mse=float((response-s['pred']).square().mean()),
            teacher_mse=float((response-s['teacher']).square().mean()),
            cosine_predicted=float(F.cosine_similarity(response.flatten(), s['pred'].flatten(), dim=0, eps=1e-12)),
            cosine_teacher=float(F.cosine_similarity(response.flatten(), s['teacher'].flatten(), dim=0, eps=1e-12)),
            response_rms=float(response.square().mean().sqrt()),
            response_change_rms=float((response-s['initial_response']).square().mean().sqrt()),
            update_rms=float(update.square().mean().sqrt()), total_rms=float(total.square().mean().sqrt()),
            rgb_linf=float(total.abs().max()), hard_clip_fraction=hard, range_clip_fraction=range_clip,
            bound_saturated_fraction=saturated/max(allowed, 1))

    def gradient(self, s):
        loss = self.loss(s, self.render(s))
        grads = torch.autograd.grad(loss, self.params, allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten().cpu()
            for p, g in zip(self.params, grads)])

    def gradients(self):
        self.restore()
        gs = [self.gradient(s) for s in self.samples]
        contact = torch.stack(gs[:4])
        free = torch.stack(gs[4:])
        mean_contact, mean_free = contact.mean(0), free.mean(0)
        norms = contact.double().norm(dim=1)
        result = dict(labels=[s['label'] for s in self.samples],
            contact_norms=norms.tolist(), free_norms=free.double().norm(dim=1).tolist(),
            contact_cosine=[[cosine(a, b) for b in contact] for a in contact],
            contact_gram=(contact.double()@contact.double().T).tolist(),
            mean_contact_norm=float(mean_contact.double().norm()), mean_free_norm=float(mean_free.double().norm()),
            mean_contact_free_cosine=cosine(mean_contact, mean_free),
            alignment_ratio=float(mean_contact.double().norm()/norms.mean()),
            free_definition='Gradient of full predicted-response MSE at initial parameters; diagnostic only, never optimized')
        # Independently check C's actual microbatch=1 accumulation against these gradients.
        self.r.network.zero_grad(set_to_none=True)
        for s in self.samples[:4]:
            (self.loss(s, self.render(s))/4).backward()
        actual = torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten().detach().cpu() for p in self.params])
        result['accumulated_mean_gradient_relative_error'] = float((actual-mean_contact).norm()/mean_contact.norm())
        assert result['accumulated_mean_gradient_relative_error'] < 1e-6
        self.restore()
        write_json(self.a.output/'gradients.json', result)
        event('gradients', **result)
        return result

    def pixels(self, index):
        s = self.samples[index]
        self.r.network.requires_grad_(False)
        variables = {k: torch.nn.Parameter(v.clone()) for k, v in s['initial_rgb'].items()}
        optimizer = torch.optim.Adam(list(variables.values()), lr=self.a.pixel_lr)
        history = [dict(step=0, **self.baseline[index])]
        last_clip = (0.0, 0.0)
        for step in range(1, self.a.steps+1):
            optimizer.zero_grad(set_to_none=True)
            loss = self.loss(s, variables)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                hard = ranges = allowed = 0
                for k, v in variables.items():
                    bound = 0.1*s['support'][k]
                    mask = (bound > 0).expand_as(v)
                    hard += int((((v-s['images'][k]).abs() > bound) & mask).sum())
                    ranges += int((((v < 0) | (v > 1)) & mask).sum())
                    allowed += int(mask.sum())
                    v.copy_(torch.maximum(s['lower'][k], torch.minimum(s['upper'][k], v)))
                last_clip = (hard/max(allowed, 1), ranges/max(allowed, 1))
            if step in OBSERVED_STEPS or step == self.a.steps:
                history.append(dict(step=step, **self.measure(s, variables, pixel_clip=last_clip)))
        endpoint = {k: v.detach().clone() for k, v in variables.items()}
        scaled = {}
        for scale in SCALES:
            out = {k: s['initial_rgb'][k]+scale*(v-s['initial_rgb'][k]) for k, v in endpoint.items()}
            row = self.measure(s, out, pixel_clip=last_clip if scale == 1 else (0.0, 0.0))
            # A has no shared parameters; the other 5 windows remain at their baseline.
            scaled[str(scale)] = [row if j == index else base.copy() for j, base in enumerate(self.baseline)]
        for n, p in self.r.network.named_parameters():
            p.requires_grad_(n in self.names)
        self.updates.append(dict(instance=f'A_{s["label"]}', optimizer_steps=self.a.steps, sample_backwards=self.a.steps))
        result = dict(kind='A', fit=[s['label']], history=history, scaled=scaled)
        event('instance_done', kind='A', fit=s['label'], endpoint=scaled['1.0'][index])
        return result

    def network(self, indices, name):
        self.restore()
        assert tensor_hash(self.r.network.state_dict()) == self.initial_hash
        optimizer = torch.optim.AdamW(self.params, lr=self.a.network_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
        history = [dict(step=0, rows=self.baseline)]
        for step in range(1, self.a.steps+1):
            optimizer.zero_grad(set_to_none=True)
            for i in indices:
                s = self.samples[i]
                loss = self.loss(s, self.render(s))/len(indices)
                assert torch.isfinite(loss)
                loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            optimizer.step()  # Exactly once, AFTER every Contact microbatch.
            if step in OBSERVED_STEPS or step == self.a.steps:
                rows = []
                with torch.no_grad():
                    for s in self.samples:
                        out, stats = self.render(s, True)
                        rows.append(self.measure(s, out, stats))
                history.append(dict(step=step, rows=rows, gradient_norm=float(norm)))
                event('network_progress', instance=name, step=step,
                    fit_mse=float(np.mean([rows[i]['predicted_mse'] for i in indices])))
        endpoint = {n: p.detach().cpu().clone() for n, p in self.r.network.named_parameters() if n in self.names}
        scaled = {}
        for scale in SCALES:
            self.restore()
            with torch.no_grad():
                for n, p in self.r.network.named_parameters():
                    if n in endpoint:
                        p.copy_(self.initial[n].to(p)+scale*(endpoint[n].to(p)-self.initial[n].to(p)))
                rows = []
                for s in self.samples:
                    out, stats = self.render(s, True)
                    rows.append(self.measure(s, out, stats))
                scaled[str(scale)] = rows
        self.updates.append(dict(instance=name, optimizer_steps=self.a.steps, sample_backwards=self.a.steps*len(indices)))
        self.restore()
        self.r.check_frozen()
        return dict(kind='B' if len(indices) == 1 else 'C', fit=[self.samples[i]['label'] for i in indices],
                    history=history, scaled=scaled)

    def run(self):
        gradients = self.gradients()
        runs = []
        for i in range(4):
            runs.append(self.pixels(i))
            write_json(self.a.output/'runs.json', runs)
        for i in range(4):
            runs.append(self.network([i], f'B_C{i+1}'))
            write_json(self.a.output/'runs.json', runs)
        runs.append(self.network(list(range(4)), 'C_shared'))
        write_json(self.a.output/'runs.json', runs)
        group_rows = {}
        for group in ('A', 'B', 'C'):
            selected = [x for x in runs if x['kind'] == group]
            group_rows[group] = {str(scale): mean_metrics(
                selected[0]['scaled'][str(scale)][:4] if group == 'C' else
                [run['scaled'][str(scale)][i] for i, run in enumerate(selected)]) for scale in SCALES}
        target = min(group_rows[g]['1.0']['update_rms'] for g in group_rows)
        matching = {g: min(SCALES, key=lambda x: abs(math.log(max(group_rows[g][str(x)]['update_rms'], 1e-30)/max(target, 1e-30)))) for g in group_rows}
        rms = [group_rows[g][str(matching[g])]['update_rms'] for g in group_rows]
        self.restore()
        self.r.check_frozen()
        assert self.predictor_hash == tensor_hash(self.predictor.state_dict())
        assert all(p.grad is None and not p.requires_grad for p in self.predictor.parameters())
        assert file_hash(self.a.checkpoint) == self.protocol['checkpoint_sha256']
        result = dict(baseline=self.baseline, gradients=gradients, group_scales=group_rows,
            matching=dict(target_update_rms=target, scales=matching, max_min_rms_ratio=max(rms)/max(min(rms), 1e-30),
                within_25_percent=max(rms)/max(min(rms), 1e-30) <= 1.25),
            updates=self.updates, total_optimizer_steps=sum(x['optimizer_steps'] for x in self.updates),
            total_optimizer_sample_backwards=sum(x['sample_backwards'] for x in self.updates),
            frozen_verified=True, checkpoint_unchanged=True, network_restored=True,
            elapsed_seconds=time.monotonic()-self.started,
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
            limitations='Only this initialization, six fixed train windows and <=20 steps. No global expressivity conclusion. Gradient cosines are supporting evidence only. No validation/generalization claim.')
        write_json(self.a.output/'summary.json', result)
        event('complete', updates=result['total_optimizer_steps'], matching=result['matching'],
              elapsed_seconds=result['elapsed_seconds'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path('data/diagnostics/civi_student_iteration')
    match = root/'final150k_match'
    for key, default in dict(base=match/'matched', predictor=match/'predictor', cache=match/'predictor_cache',
        old=Path('data/diagnostics/2026.11.15/student_dual_path_ablation'),
        dependencies=match/'dependencies.json', checkpoint=root/'final150k_baseline_s42_3000/best.pt',
        output=root/'final150k_local_updates_20').items():
        parser.add_argument('--'+key, type=Path, default=default)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--network-lr', type=float, default=2e-4)
    parser.add_argument('--pixel-lr', type=float, default=1e-3)
    args = parser.parse_args()
    assert 1 <= args.steps <= 20
    assert torch.cuda.device_count() == 1, 'Expose exactly one confirmed idle GPU'
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    with torch.autocast('cuda', enabled=False):
        Study(args).run()


if __name__ == '__main__':
    main()
