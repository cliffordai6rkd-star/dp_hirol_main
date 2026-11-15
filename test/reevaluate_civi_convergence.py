"""Evaluation only: complete matched-150k EMA trajectory, no optimizer/backward.

All original/modified image encodes use the same FP32 batch shape (8 windows).
Historical BF16 predictions and FP16 teacher targets are never used as metrics.
"""
from pathlib import Path
import argparse
import copy
import csv
import json
import os
import subprocess
import time
from collections.abc import Mapping

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf

from run_response_decoder_ablation import Runtime
from run_scene_torque_response_predictor import SceneTorquePredictor
from student_dual_path_common import tensor_hash, write_json, to_device
from paired_response_full_common import file_hash, deterministic_mode
from diffusion_policy.model.vision.render_net_attention import zero_wrench_reference


PHASES = ('free', 'precontact', 'contact')
NODES = ((0, '1000/initial.pt'), (1000, '1000/best.pt'),
         (2800, '3000/best.pt'), (3000, '3000/latest.pt'))
BATCH = 8


def event(name, **values):
    print(json.dumps(dict(event=name, **values)), flush=True)


class IndexedImages(Dataset):
    """Reuse the verified decoded dataset, without copying old target caches."""
    def __init__(self, paired):
        self.dataset, self.episodes = paired.dataset, paired.episodes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return dict(obs=self.dataset[index]['obs'], index=index,
                    episode=int(self.episodes[index]),
                    phase=int(self.dataset.phase_labels[index]))


def pad_batch(value, count):
    if isinstance(value, dict):
        return {k: pad_batch(v, count) for k, v in value.items()}
    return torch.cat([value, value[-1:].expand(count, *value.shape[1:])]) if count else value


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for v in value.values():
            yield from tensors(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from tensors(v)


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def ratio(a, b):
    return a / b if b > 0 else None


def aggregate(rows):
    out = dict(samples=len(rows), episodes=len({r['episode'] for r in rows}))
    if not rows:
        return out
    keys = ('predictor_mse', 'teacher_mse', 'predictor_energy', 'teacher_energy',
            'response_energy', 'rgb_energy', 'rgb_from_step0_energy',
            'rgb_from_step2800_energy', 'response_from_step0_energy',
            'response_from_step2800_energy', 'hard_clip_fraction', 'range_clip_fraction',
            'alpha_mean', 'active_patch_count')
    out.update({k: mean(rows, k) for k in keys})
    for key in ('response', 'rgb', 'rgb_from_step0', 'rgb_from_step2800',
                'response_from_step0', 'response_from_step2800'):
        out[key + '_rms'] = out[key + '_energy'] ** .5
    for target in ('predictor', 'teacher'):
        out[target + '_rms'] = out[target + '_energy'] ** .5
        out[target + '_error_zero_ratio'] = ratio(out[target + '_mse'], out[target + '_energy'])
        out[target + '_improvement_vs_identity'] = (
            1 - out[target + '_error_zero_ratio']
            if out[target + '_error_zero_ratio'] is not None else None)
        count = sum(r['active_patch_count'] for r in rows)
        out['cosine_' + target] = ratio(sum(r['cosine_' + target + '_sum'] for r in rows), count)
        out['amplitude_ratio_' + target] = (
            ratio(sum(r['active_response_ss'] for r in rows),
                  sum(r['active_' + target + '_ss'] for r in rows)))
        if out['amplitude_ratio_' + target] is not None:
            out['amplitude_ratio_' + target] **= .5
    return out


def summarize(rows):
    summary, episode_rows = {}, []
    scopes = dict(train=lambda r: r['split'] == 'train',
                  validation_all=lambda r: r['split'] == 'validation',
                  validation_internal=lambda r: r['split'] == 'validation' and r['episode'] not in (5, 26),
                  retained_5_26=lambda r: r['split'] == 'validation' and r['episode'] in (5, 26))
    for scope, choose in scopes.items():
        summary[scope] = {}
        for phase, label in enumerate(PHASES):
            selected = [r for r in rows if choose(r) and r['phase'] == phase]
            episodes = sorted({r['episode'] for r in selected})
            per_episode = [dict(scope=scope, phase=label, episode=ep,
                                **aggregate([r for r in selected if r['episode'] == ep])) for ep in episodes]
            episode_rows.extend(per_episode)
            result = aggregate(selected)
            for target in ('predictor', 'teacher'):
                values = [r[target + '_error_zero_ratio'] for r in per_episode
                          if r[target + '_error_zero_ratio'] is not None]
                result[target + '_episode_macro_error_zero_ratio'] = float(np.mean(values)) if values else None
                result[target + '_valid_ratio_episodes'] = len(values)
            summary[scope][label] = result
    return summary, episode_rows


class Evaluation:
    def __init__(self, args):
        self.a, self.started = args, time.monotonic()
        dependencies = json.loads(args.dependencies.read_text())
        assert dependencies['teacher_step'] == 150000
        config = OmegaConf.load(Path(__file__).parent / 'config/response_decoder_gate_ablation.yaml')
        for key in ('teacher_sha256', 'predictor_sha256', 'condition_sha256', 'decoder_initial_sha256'):
            config[key] = dependencies[key]
        config.strict_reference = False
        args.variant, args.smoke = 'A', False
        self.r = r = Runtime(args, config)
        r.teacher.float().eval().requires_grad_(False)
        self.models, inventory = {}, []
        for step, suffix in NODES:
            path = args.root / ('final150k_baseline_s42_' + suffix)
            saved = torch.load(path, map_location='cpu', weights_only=False)
            assert saved['step'] == step
            for key in ('teacher_sha256', 'predictor_sha256', 'adapter_sha256', 'decoder_initial_sha256'):
                assert saved['manifest'][key] == dependencies[key]
            model = r.network if step == 0 else copy.deepcopy(r.network)
            model.load_state_dict(saved['ema_model'], strict=True)
            model.float().eval().requires_grad_(False)
            deterministic_mode(model)
            self.models[step] = model
            inventory.append(dict(step=step, path=str(path), file_sha256=file_hash(path),
                                  ema_state_sha256=tensor_hash(model.state_dict())))
        for model in self.models.values():
            for name in ('context_adapter', 'condition_network', 'wrench_encoder'):
                assert tensor_hash(getattr(model.owner, name).state_dict()) == tensor_hash(getattr(r.owner, name).state_dict())
        r.check_frozen()
        state = torch.load(args.predictor / 'best.pt', map_location='cpu', weights_only=False)['model']
        self.predictor = SceneTorquePredictor(state['visual_scale'], state['torque_scale']).to(r.device).float().eval()
        self.predictor.load_state_dict(state, strict=True)
        self.predictor.requires_grad_(False)
        deterministic_mode(r.teacher)
        deterministic_mode(self.predictor)
        self.predictor_hash = tensor_hash(self.predictor.state_dict())
        plan1000 = np.load(args.root / 'final150k_baseline_s42_1000/train_batches.npy')
        plan = np.load(args.root / 'final150k_baseline_s42_3000/train_batches.npy')
        assert plan.shape == (3000, 8) and np.array_equal(plan1000, plan[:1000])
        exposure = []
        for step, _ in NODES:
            draws = r.train.phase_labels[plan[:step].reshape(-1)]
            contact = plan[:step].reshape(-1)[draws == 2]
            exposure.append(dict(step=step, phase_draws=np.bincount(draws, minlength=3).tolist(),
                                 contact_draws=len(contact), contact_unique=len(np.unique(contact)),
                                 contact_exposure=len(contact) / 3581))
        self.precision = {}
        self.hooks = []
        modules = dict(dino=r.teacher.obs_encoder.backbone,
                       projection=r.teacher.obs_encoder.image_projection,
                       adapter=r.owner.context_adapter, context_norm=r.teacher.obs_encoder.context_norm,
                       teacher_context=r.teacher.obs_encoder, predictor=self.predictor,
                       condition_network=r.owner.condition_network,
                       **{'renderer_' + str(s): m for s, m in self.models.items()})
        for name, module in modules.items():
            self.precision[name] = dict(input_float_dtypes=set(), output_float_dtypes=set(),
                                       parameter_dtypes=sorted({str(p.dtype) for p in module.parameters()}),
                                       trainable_parameters=sum(p.numel() for p in module.parameters() if p.requires_grad),
                                       training=module.training)
            def hook(_module, inputs, output, name=name):
                for field, value in (('input_float_dtypes', inputs), ('output_float_dtypes', output)):
                    for v in tensors(value):
                        if v.is_floating_point():
                            self.precision[name][field].add(str(v.dtype))
                            assert v.dtype == torch.float32, (name, v.dtype)
            self.hooks.append(module.register_forward_hook(hook))
        self.protocol = dict(weights='ema_model ONLY', nodes=inventory, dependencies=dependencies,
            source_sha256=file_hash(Path(__file__)), batch_size=BATCH, tail='repeat final window, exclude repeats',
            effective_patch_threshold=r.threshold, active_mask='FP32 Teacher patch RMS > threshold; identical for both cosines and every node',
            mse='Full tokens/channels; no active-patch filtering in MSE',
            targets='Recomputed FP32 dP and Teacher(wrench)-Teacher(physical-zero wrench); no historical target metrics',
            free_metrics='Measure actual dS MSE/RMS directly; do not assume all Free Teacher responses are exactly zero',
            readout='DINO + projection + calibrated Adapter + context_norm; FP32 throughout',
            teacher_response='Exact native _encode_context twice, apply_curriculum_mask=False',
            reference_definitions=dict(identity='dS=0 / original RGB', experiment_initial='EMA step0',
                                       local_probe_initial='EMA step2800'),
            exposure=exposure, sampling_plans=dict(prefix_equal=True,
                step1000_sha256=file_hash(args.root/'final150k_baseline_s42_1000/train_batches.npy'),
                step3000_sha256=file_hash(args.root/'final150k_baseline_s42_3000/train_batches.npy')),
            excluded_exposure='Old-teacher warm start, Oracle, calibration, other structures/seeds, all temporary probes',
            missing_nodes='No parameter snapshots at 200/400/.../800 or1200/1400/.../2600. No step2000 snapshot. No補訓.',
            duplicate_node='1000/latest is same EMA as1000/best; selected_eval copies step2800, no additional node',
            internal_validation_episodes=[38,44,53,60], retained_evaluation_episodes=[5,26],
            optimizer_updates=0, backward_calls=0, gradients_enabled=False,
            precision_policy='FP32 modules/inputs/outputs, no autocast/TF32; deterministic algorithms; math SDP',
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu=torch.cuda.get_device_name(),
            gpu_query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv'],text=True))
        write_json(args.output/'protocol.json', self.protocol)
        event('ready', nodes=inventory, exposure=exposure, threshold=r.threshold)

    def encode(self, images, adapter=None):
        return self.r.teacher.obs_encoder.encode_image_context(
            images, context_adapter=adapter or self.r.owner.context_adapter, input_grad=False)

    @torch.no_grad()
    def run(self):
        r, records = self.r, {step: [] for step in self.models}
        coverage = {}
        for split, paired in r.datasets.items():
            ds = IndexedImages(paired)
            assert len(ds) == (17740 if split == 'train' else 1893)
            assert np.bincount(ds.dataset.phase_labels,minlength=3).tolist() == (
                [13419,740,3581] if split == 'train' else [1317,108,468])
            loader = DataLoader(ds,batch_size=BATCH,num_workers=4,pin_memory=True,shuffle=False)
            visited = []
            for ordinal, raw in enumerate(loader):
                size = len(raw['index'])
                batch = to_device(pad_batch(raw, BATCH-size), r.device)
                images, wrench = r.inputs(batch)
                images = {k:v.float() for k,v in images.items()}
                seen = []
                def capture(z):
                    seen.append(z)
                    return r.owner.context_adapter(z)
                z0 = self.encode(images, capture)
                assert len(seen) == 1 and z0.shape == (BATCH,784,256)
                visual = seen[0]
                pred = self.predictor(visual,visual.reshape(BATCH,2,392,256).mean(2),
                    batch['obs'][r.teacher.wrench_key][:,:2].float(),torch.arange(784,device=r.device))
                target = r.teacher._encode_context(batch['obs'],apply_curriculum_mask=False)
                zero = r.teacher._encode_context(zero_wrench_reference(r.teacher,batch['obs']),apply_curriculum_mask=False)
                teacher = target-zero
                active = teacher.square().mean(-1).sqrt() > r.threshold
                masks,condition = r.owner.condition_network(images,wrench)
                spatial = {k:r.owner._mask_gate(masks[k].flatten(0,1),images[k]).reshape_as(masks[k]) for k in images}
                alpha = condition['contact_alpha']
                renders, responses, traces = {}, {}, {}
                for step, model in self.models.items():
                    rgb,stats = model(images,pred,spatial,alpha,trace=True)
                    renders[step], responses[step], traces[step] = rgb,self.encode(rgb)-z0,stats
                for step,response in responses.items():
                    stats,rgb = traces[step],renders[step]
                    energy = lambda v:v.double().flatten(1).square().mean(1)
                    metrics = dict(predictor_mse=energy(response-pred),teacher_mse=energy(response-teacher),
                        predictor_energy=energy(pred),teacher_energy=energy(teacher),response_energy=energy(response),
                        response_from_step0_energy=energy(response-responses[0]),
                        response_from_step2800_energy=energy(response-responses[2800]),
                        alpha_mean=alpha.double().flatten(1).mean(1),active_patch_count=active.sum(1),
                        active_response_ss=(response.double().square().sum(-1)*active).sum(1))
                    for name,other in (('predictor',pred),('teacher',teacher)):
                        metrics['cosine_'+name+'_sum']=(F.cosine_similarity(response,other,dim=-1,eps=1e-6)*active).double().sum(1)
                        metrics['active_'+name+'_ss']=(other.double().square().sum(-1)*active).sum(1)
                    for name,reference in (('rgb',images),('rgb_from_step0',renders[0]),('rgb_from_step2800',renders[2800])):
                        metrics[name+'_energy']=torch.stack([energy(rgb[k]-reference[k]) for k in rgb]).mean(0)
                    den=stats['allowed_channels'].clamp_min(1)
                    metrics['hard_clip_fraction']=stats['hard_clipped_channels']/den
                    metrics['range_clip_fraction']=stats['range_clipped_channels']/den
                    for key,v in rgb.items():
                        support=spatial[key]*alpha[...,None,None,None]
                        assert (v-images[key]).abs().max() <= .1000002
                        assert torch.equal(v[(support==0).expand_as(v)],images[key][(support==0).expand_as(v)])
                    cpu={k:v[:size].cpu().tolist() for k,v in metrics.items()}
                    for j in range(size):
                        records[step].append(dict(step=step,split=split,index=int(raw['index'][j]),
                            episode=int(raw['episode'][j]),phase=int(raw['phase'][j]),
                            **{k:v[j] for k,v in cpu.items()}))
                visited.extend(raw['index'].tolist())
                if ordinal % 25 == 0 or len(visited)==len(ds):
                    event('progress',split=split,done=len(visited),total=len(ds),
                          elapsed_seconds=time.monotonic()-self.started)
            assert visited == list(range(len(ds)))
            coverage[split]=dict(samples=len(visited),phase_counts=np.bincount(ds.dataset.phase_labels,minlength=3).tolist(),
                                 episodes=sorted(map(int,np.unique(ds.episodes))))
        summary, episode_rows = {}, []
        for step,rows in records.items():
            with (self.a.output/f'windows_ema_{step}.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            summary[str(step)],episodes=summarize(rows)
            episode_rows.extend(dict(step=step,**row) for row in episodes)
        # Pair every episode against its OWN checkpoint reference before macro averaging.
        lookup={(row['step'],row['scope'],row['phase'],row['episode']):row for row in episode_rows}
        for step in self.models:
            for scope,phases in summary[str(step)].items():
                for label,values in phases.items():
                    for target in ('predictor','teacher'):
                        for anchor in (0,2800):
                            paired=[]
                            for row in episode_rows:
                                if row['step']==step and row['scope']==scope and row['phase']==label:
                                    initial=lookup[(anchor,scope,label,row['episode'])][target+'_mse']
                                    if initial>0:paired.append(1-row[target+'_mse']/initial)
                            values[f'{target}_episode_macro_improvement_vs_step{anchor}']=float(np.mean(paired)) if paired else None
                            initial=summary[str(anchor)][scope][label][target+'_mse']
                            values[f'{target}_improvement_vs_step{anchor}']=1-values[target+'_mse']/initial if initial>0 else None
        write_json(self.a.output/'episode_metrics.json',episode_rows)
        r.check_frozen()
        assert tensor_hash(self.predictor.state_dict()) == self.predictor_hash
        for meta in self.protocol['nodes']:
            assert file_hash(Path(meta['path'])) == meta['file_sha256']
            model=self.models[meta['step']]
            assert tensor_hash(model.state_dict()) == meta['ema_state_sha256']
            assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
        precision={name:{k:sorted(v) if isinstance(v,set) else v for k,v in values.items()}
                   for name,values in self.precision.items()}
        result=dict(summary=summary,coverage=coverage,precision=precision,optimizer_updates=0,
                    backward_calls=0,checkpoint_and_frozen_hashes_unchanged=True,
                    elapsed_seconds=time.monotonic()-self.started,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated())
        write_json(self.a.output/'summary.json',result)
        event('complete',coverage=coverage,elapsed_seconds=result['elapsed_seconds'],optimizer_updates=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path('data/diagnostics/civi_student_iteration')
    match=root/'final150k_match'
    for key,default in dict(root=root,base=match/'matched',predictor=match/'predictor',cache=match/'predictor_cache',
            old=Path('data/diagnostics/2026.11.15/student_dual_path_ablation'),dependencies=match/'dependencies.json',
            output=root/'final150k_convergence_fp32').items():
        parser.add_argument('--'+key,type=Path,default=default)
    args=parser.parse_args()
    assert torch.cuda.device_count()==1
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    with torch.no_grad(),torch.autocast('cuda',enabled=False):
        Evaluation(args).run()


if __name__=='__main__':
    main()
