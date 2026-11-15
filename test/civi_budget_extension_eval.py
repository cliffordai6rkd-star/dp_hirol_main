"""FP32 evaluation for the bounded continuation; never changes training precision."""
from contextlib import contextmanager
from pathlib import Path
import copy
import csv
import json
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from reevaluate_civi_convergence import IndexedImages, pad_batch, tensors, PHASES
from run_scene_torque_response_predictor import SceneTorquePredictor
from student_dual_path_common import to_device, tensor_hash, write_json
from diffusion_policy.model.vision.render_net_attention import zero_wrench_reference


@contextmanager
def fp32_evaluation():
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), random.getstate(), np.random.get_state())
    flags = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
             torch.backends.mha.get_fastpath_enabled(), torch.backends.cuda.flash_sdp_enabled(),
             torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.math_sdp_enabled())
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    try:
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            yield
    finally:
        torch.use_deterministic_algorithms(flags[0])
        torch.backends.cudnn.deterministic = flags[1]
        torch.backends.mha.set_fastpath_enabled(flags[2])
        torch.backends.cuda.enable_flash_sdp(flags[3])
        torch.backends.cuda.enable_mem_efficient_sdp(flags[4])
        torch.backends.cuda.enable_math_sdp(flags[5])
        torch.set_rng_state(rng[0]); torch.cuda.set_rng_state_all(rng[1])
        random.setstate(rng[2]); np.random.set_state(rng[3])


def ratio(a, b):
    return a / b if b > 0 else None


def aggregate(rows):
    out = dict(samples=len(rows), episodes=len({r['episode'] for r in rows}))
    if not rows:
        return out
    keys = ('predictor_mse', 'teacher_mse', 'predictor_energy', 'teacher_energy',
            'response_energy', 'rgb_energy', 'rgb_from_step0_energy', 'rgb_from_step3000_energy',
            'hard_clip_fraction', 'range_clip_fraction', 'alpha_mean', 'active_patch_count')
    out.update({k:float(np.mean([r[k] for r in rows])) for k in keys})
    for k in ('response', 'rgb', 'rgb_from_step0', 'rgb_from_step3000'):
        out[k+'_rms'] = out[k+'_energy']**.5
    out['rgb_linf'] = max(r['rgb_linf'] for r in rows)
    for target in ('predictor', 'teacher'):
        out[target+'_rms'] = out[target+'_energy']**.5
        out[target+'_error_zero_ratio'] = ratio(out[target+'_mse'],out[target+'_energy'])
        value = out[target+'_error_zero_ratio']
        out[target+'_improvement_vs_identity'] = 1-value if value is not None else None
        out['cosine_'+target] = ratio(sum(r['cosine_'+target+'_sum'] for r in rows),sum(r['active_patch_count'] for r in rows))
        value = ratio(sum(r['active_response_ss'] for r in rows),sum(r['active_'+target+'_ss'] for r in rows))
        out['amplitude_ratio_'+target] = value**.5 if value is not None else None
    return out


def summarize(rows, anchors):
    result, episodes = {}, []
    scopes = dict(train=lambda r:r['split']=='train',
                  validation_all=lambda r:r['split']=='validation',
                  validation_internal=lambda r:r['split']=='validation' and r['episode'] not in (5,26),
                  retained_5_26=lambda r:r['split']=='validation' and r['episode'] in (5,26))
    for scope, choose in scopes.items():
        result[scope] = {}
        for phase,label in enumerate(PHASES):
            selected = [r for r in rows if choose(r) and r['phase']==phase]
            eps = sorted({r['episode'] for r in selected})
            per_episode = [dict(scope=scope,phase=label,episode=ep,**aggregate([r for r in selected if r['episode']==ep])) for ep in eps]
            episodes.extend(per_episode)
            summary = aggregate(selected)
            if selected:
                for target in ('predictor','teacher'):
                    values=[r[target+'_error_zero_ratio'] for r in per_episode if r[target+'_error_zero_ratio'] is not None]
                    summary[target+'_episode_macro_error_zero_ratio']=float(np.mean(values)) if values else None
                    summary[target+'_episode_macro_improvement_vs_identity']=1-float(np.mean(values)) if values else None
                    for anchor, lookup in anchors.items():
                        values=[1-r[target+'_mse']/lookup[(scope,label,r['episode'])][target+'_mse']
                                for r in per_episode if lookup[(scope,label,r['episode'])][target+'_mse']>0]
                        summary[f'{target}_episode_macro_improvement_vs_step{anchor}']=float(np.mean(values)) if values else None
            result[scope][label]=summary
    return result,episodes


class Evaluator:
    def __init__(self, experiment, root, reference):
        self.x,self.r,self.root,self.reference=experiment,experiment.r,root,reference
        r=self.r
        self.initial0=copy.deepcopy(experiment.ema).eval().requires_grad_(False)
        self.initial0.load_state_dict(torch.load(root/'final150k_baseline_s42_1000/initial.pt',map_location='cpu',weights_only=False)['ema_model'])
        self.initial3000=copy.deepcopy(experiment.ema).eval().requires_grad_(False)
        # On an interrupted continuation retain the original step3000 RGB anchor.
        source=torch.load(root/'final150k_baseline_s42_3000/latest.pt',map_location='cpu',weights_only=False)
        self.initial3000.load_state_dict(source['ema_model'])
        state=torch.load(experiment.a.predictor/'best.pt',map_location='cpu',weights_only=False)['model']
        self.predictor=SceneTorquePredictor(state['visual_scale'],state['torque_scale']).to(r.device).eval().requires_grad_(False)
        self.predictor.load_state_dict(state)
        self.predictor_hash=tensor_hash(self.predictor.state_dict())
        old=json.loads((reference/'episode_metrics.json').read_text())
        self.anchors={s:{(a['scope'],a['phase'],a['episode']):a for a in old if a['step']==s} for s in (0,3000)}
        self.mode='prepare';self.precision={};self.hooks=[]
        modules=dict(dino=r.teacher.obs_encoder.backbone,projection=r.teacher.obs_encoder.image_projection,
                     adapter=r.owner.context_adapter,context_norm=r.teacher.obs_encoder.context_norm,
                     condition=r.owner.condition_network,teacher_context=r.teacher.obs_encoder,
                     predictor=self.predictor,renderer_online=experiment.network,renderer_ema=experiment.ema)
        for name,module in modules.items():
            def hook(_module, inputs, kwargs, output, name=name):
                record=self.precision.setdefault(self.mode,{}).setdefault(name,dict(input_float_dtypes=set(),output_float_dtypes=set()))
                for field,items in (('input_float_dtypes',(inputs,kwargs)),('output_float_dtypes',output)):
                    for value in tensors(items):
                        if value.is_floating_point():
                            record[field].add(str(value.dtype))
                            if self.mode=='fp32_evaluation':assert value.dtype==torch.float32,(name,value.dtype)
            self.hooks.append(module.register_forward_hook(hook,with_kwargs=True))

    def encode(self,images,adapter=None):
        return self.r.teacher.obs_encoder.encode_image_context(images,context_adapter=adapter or self.r.owner.context_adapter,input_grad=False)

    def evaluate(self,step,train_contact=False):
        r=self.r;self.mode='fp32_evaluation';rows=[];started=time.monotonic()
        with fp32_evaluation():
            for split in (['train','validation'] if train_contact else ['validation']):
                dataset=IndexedImages(r.datasets[split]);visited=[]
                loader=DataLoader(dataset,batch_size=8,num_workers=4,pin_memory=True,shuffle=False,
                                  generator=torch.Generator().manual_seed(0))
                for ordinal,raw in enumerate(loader):
                    size=len(raw['index'])
                    # Preserve the preceding reevaluation's ORIGINAL batch membership/shape.
                    # Only batches with no Contact at all may be skipped in train evaluation.
                    if split=='train' and not (raw['phase']==2).any():continue
                    selected=[j for j in range(size) if split!='train' or int(raw['phase'][j])==2]
                    b=to_device(pad_batch(raw,8-size),r.device);images,wrench=r.inputs(b)
                    images={k:v.float() for k,v in images.items()}
                    seen=[]
                    def capture(z):seen.append(z);return r.owner.context_adapter(z)
                    z0=self.encode(images,capture);visual=seen[0]
                    pred=self.predictor(visual,visual.reshape(8,2,392,256).mean(2),
                        b['obs'][r.teacher.wrench_key][:,:2].float(),torch.arange(784,device=r.device))
                    target=r.teacher._encode_context(b['obs'],apply_curriculum_mask=False)
                    zero=r.teacher._encode_context(zero_wrench_reference(r.teacher,b['obs']),apply_curriculum_mask=False)
                    teacher=target-zero;active=teacher.square().mean(-1).sqrt()>r.threshold
                    masks,condition=r.owner.condition_network(images,wrench)
                    spatial={k:r.owner._mask_gate(masks[k].flatten(0,1),images[k]).reshape_as(masks[k]) for k in images}
                    alpha=condition['contact_alpha']
                    rgb,stats=self.x.ema(images,pred,spatial,alpha,trace=True)
                    response=self.encode(rgb)-z0
                    references={0:self.initial0(images,pred,spatial,alpha),3000:self.initial3000(images,pred,spatial,alpha)}
                    energy=lambda value:value.double().flatten(1).square().mean(1)
                    metrics=dict(predictor_mse=energy(response-pred),teacher_mse=energy(response-teacher),
                        predictor_energy=energy(pred),teacher_energy=energy(teacher),response_energy=energy(response),
                        alpha_mean=alpha.double().flatten(1).mean(1),active_patch_count=active.sum(1),
                        active_response_ss=(response.double().square().sum(-1)*active).sum(1))
                    for name,other in (('predictor',pred),('teacher',teacher)):
                        metrics['cosine_'+name+'_sum']=(F.cosine_similarity(response,other,dim=-1,eps=1e-6)*active).double().sum(1)
                        metrics['active_'+name+'_ss']=(other.double().square().sum(-1)*active).sum(1)
                    for name,other in (('rgb',images),('rgb_from_step0',references[0]),('rgb_from_step3000',references[3000])):
                        metrics[name+'_energy']=torch.stack([energy(rgb[k]-other[k]) for k in rgb]).mean(0)
                    metrics['rgb_linf']=torch.stack([(rgb[k]-images[k]).abs().flatten(1).max(1).values for k in rgb]).max(0).values
                    den=stats['allowed_channels'].clamp_min(1)
                    metrics['hard_clip_fraction']=stats['hard_clipped_channels']/den
                    metrics['range_clip_fraction']=stats['range_clipped_channels']/den
                    for key,value in rgb.items():
                        support=spatial[key]*alpha[...,None,None,None]
                        assert ((value-images[key]).abs()-.1*support).max()<=2e-7
                        assert torch.equal(value[(support==0).expand_as(value)],images[key][(support==0).expand_as(value)])
                    assert all(torch.isfinite(v).all() for v in metrics.values())
                    cpu={k:v.cpu().tolist() for k,v in metrics.items()}
                    for j in selected:
                        rows.append(dict(step=step,split=split,index=int(raw['index'][j]),episode=int(raw['episode'][j]),
                            phase=int(raw['phase'][j]),**{k:v[j] for k,v in cpu.items()}))
                        visited.append(int(raw['index'][j]))
                    if ordinal%50==0:
                        print(json.dumps(dict(event='fp32_eval',step=step,split=split,windows=len(visited),elapsed=time.monotonic()-started)),flush=True)
                expected=np.flatnonzero(dataset.dataset.phase_labels==2).tolist() if split=='train' else list(range(1893))
                assert visited==expected,(split,len(visited),len(expected))
        summary,episodes=summarize(rows,self.anchors)
        out=self.x.out
        with (out/f'fp32_windows_{step}.csv').open('w') as stream:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        write_json(out/f'fp32_episodes_{step}.json',episodes)
        write_json(out/f'fp32_metrics_{step}.json',summary)
        assert tensor_hash(self.predictor.state_dict())==self.predictor_hash
        if step==3000:self.verify_reference(rows)
        self.mode='between_evaluations'
        return summary

    def verify_reference(self,rows):
        with (self.reference/'windows_ema_3000.csv').open() as stream:
            reference={(r['split'],int(r['index'])):r for r in csv.DictReader(stream)}
        fields=('predictor_mse','teacher_mse','predictor_energy','teacher_energy','response_energy','rgb_energy',
                'cosine_predictor_sum','cosine_teacher_sum','active_patch_count')
        actual=np.array([[r[k] for k in fields] for r in rows]);expected=np.array([[float(reference[(r['split'],r['index'])][k]) for k in fields] for r in rows])
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-12)
        write_json(self.x.out/'zero_update_reproduction.json',dict(windows=len(rows),fields=fields,
            max_absolute_difference=float(np.max(np.abs(actual-expected))),bit_exact=bool(np.array_equal(actual,expected)),
            optimizer_updates=0,reference=str(self.reference)))

    def write_precision(self):
        write_json(self.x.out/'precision.json',{mode:{name:{key:sorted(values) for key,values in record.items()}
            for name,record in records.items()} for mode,records in self.precision.items()})
