"""Measure the existing RGB path before spending controlled-iteration updates."""
from pathlib import Path
import argparse
import json
import hashlib

import numpy as np
import torch
from omegaconf import OmegaConf

from run_response_decoder_ablation import Runtime, gradient_norms
from student_dual_path_common import write_json
from diffusion_policy.model.vision.render_net_attention import zero_wrench_reference


def panels(r):
    ds=r.datasets['train']
    eps=np.unique(ds.episodes)
    chosen=eps[np.linspace(0,len(eps)-1,24).round().astype(int)]
    contact=[int(x[len(x)//2]) for ep in chosen
             if len(x:=np.flatnonzero((ds.episodes==ep)&(r.train.phase_labels==2)))]
    free=[int(x[len(x)//2]) for ep in chosen[::3]
          if len(x:=np.flatnonzero((ds.episodes==ep)&(r.train.phase_labels==0)))]
    internal=[]
    val=r.datasets['validation']
    for ep in r.config.selection_episodes:
        for phase,n in [(0,4),(1,4),(2,8)]:
            x=np.flatnonzero((val.episodes==ep)&(r.val.phase_labels==phase))
            internal.extend(x[np.linspace(0,len(x)-1,min(n,len(x))).round().astype(int)].tolist())
    return dict(train_contact=contact,train_free=free,internal_validation=internal,
                train_episodes=chosen.tolist(),selection_episodes=list(r.config.selection_episodes))


def summarize(rows):
    result={}
    for phase,name in enumerate(['free','precontact','contact']):
        group=[v for v in rows if v['phase']==phase]
        if not group:continue
        summed=lambda k:sum(float(v[k]) for v in group)
        allowed=summed('allowed_channels')
        result[name]=dict(samples=len(group),
            support_fraction=allowed/max(summed('total_channels'),1),
            hard_clip_fraction=summed('hard_clipped_channels')/max(allowed,1),
            range_clip_fraction=summed('range_clipped_channels')/max(allowed,1),
            effective_slope_fraction=summed('effective_channels')/max(allowed,1),
            relative_effective_slope_fraction=summed('relative_effective_channels')/max(allowed,1),
            observed_logit_gradient_fraction=summed('observed_logit_gradient_channels')/max(allowed,1),
            alpha_mean=float(np.mean([np.mean(v['alpha']) for v in group])),
            alpha_min=min(min(v['alpha']) for v in group),alpha_max=max(max(v['alpha']) for v in group),
            zero_torque_alpha_mean=float(np.mean([np.mean(v['zero_alpha']) for v in group])),
            zero_deployment_rgb_rms=float(np.mean([v['zero_rgb_mse'] for v in group]))**.5,
            zero_deployment_response_mse=float(np.mean([v['zero_response_mse'] for v in group])))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['base','predictor','old','cache','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).parent/'config/response_decoder_gate_ablation.yaml')
    a=p.parse_args();a.variant='A';a.smoke=False
    config=OmegaConf.load(a.config)
    torch.set_num_threads(4);torch.manual_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a.output.mkdir(parents=True,exist_ok=False)
    r=Runtime(a,config);panel=panels(r);write_json(a.output/'panels.json',panel)
    r.distribution(a.output)
    checks={};all_results={}
    for label,path in [('initial',a.cache/'decoder/initial.pt'),('existing1000_ema',a.cache/'decoder/best.pt')]:
        saved=torch.load(path,map_location='cpu',weights_only=False)
        r.network.load_state_dict(saved.get('ema_model',saved['model']),strict=True)
        r.network.capture_pixels=True;records=[]
        ids=panel['train_contact']+panel['train_free']
        for start in range(0,len(ids),4):
            batch=r.batch('train',ids[start:start+4]);im,wr=r.inputs(batch)
            pred=r.predicted(batch,'train');sp,al=r.gates(im,wr)
            z0=r.encode(im).detach()
            r.network.zero_grad(set_to_none=True)
            with r.amp():rendered,stats=r.network(im,pred,sp,al,trace=True)
            z=r.encode(rendered,True)
            target=pred*(batch['phase']!=0)[:,None,None]
            loss=(z-z0-target).square().mean();loss.backward()
            grads=gradient_norms(r.network)
            observed=torch.zeros(len(pred),device=r.device)
            for t in r.network.pixel_tensors.values():
                allowed=(t['support']>0).expand_as(t['logits'])
                observed+=((t['logits'].grad.abs()>0)&allowed).flatten(1).sum(-1)/len(im)
            stats['observed_logit_gradient_channels']=observed
            r.network.capture_pixels=False;r.network.pixel_tensors={}
            zim,zwr,_,_=r.teacher.prepare_context_inputs_with_future_contact(zero_wrench_reference(r.teacher,batch['obs']))
            assert all(torch.equal(zim[k],im[k]) for k in im)
            assert r.teacher.normalizer[r.teacher.wrench_key].unnormalize(zwr).abs().max()<1e-5
            zsp,za=r.gates(zim,zwr)
            with torch.no_grad(),r.amp():
                zero_rgb=r.network(zim,torch.zeros_like(pred),zsp,za)
                closed=r.network(im,pred,sp,torch.zeros_like(al))
            assert all(torch.equal(closed[k],im[k]) for k in im)
            stats['zero_alpha']=za
            stats['zero_rgb_mse']=torch.cat([(zero_rgb[k]-im[k]).flatten(1) for k in im],1).square().mean(-1)
            stats['zero_response_mse']=(r.encode(zero_rgb)-z0).square().mean((1,2))
            cpu={k:v.detach().cpu().numpy() for k,v in stats.items()}
            for i in range(len(pred)):
                records.append(dict(index=int(batch['index'][i]),episode=int(batch['episode'][i]),phase=int(batch['phase'][i]),
                    **{k:v[i].tolist() for k,v in cpu.items()}))
            if start==0:
                encoder=r.teacher.obs_encoder
                assert tuple(r.owner.image_keys)==tuple(encoder.image_keys)
                assert all(encoder.patch_counts[k]==196 for k in encoder.image_keys)
                with torch.no_grad(),r.amp():
                    _,tokens=encoder._encode_image_streams(im,encoder.image_keys,len(pred),2,input_grad=False)
                    reconstructed=encoder.context_norm(r.owner.context_adapter(torch.cat(tokens,dim=2).reshape(len(pred),784,256))).float()
                layout_error=float((reconstructed-z0).abs().max())
                assert layout_error==0,layout_error
                checks[label]=dict(alpha0_exact_identity=True,physical_zero_unnormalized=True,
                    parameter_gradients=grads,layout_max_error=layout_error,
                    layout=dict(time=2,cameras=list(encoder.image_keys),patch_grid=[14,14],channels=256,student_grid=[28,28],interpolation='bilinear align_corners=False'),
                    l0_renderer_gradient='none: frozen Adapter and original RGB reference; monitoring only',
                    image_sha256={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in im.items()})
            r.network.capture_pixels=True
        r.network.capture_pixels=False;r.network.pixel_tensors={};r.network.zero_grad(set_to_none=True)
        r.check_frozen()
        write_json(a.output/(label+'_samples.json'),records)
        all_results[label]=summarize(records)
        print(json.dumps(dict(event='audit',model=label,result=all_results[label])),flush=True)
    write_json(a.output/'audit.json',dict(manifest=r.manifest,checks=checks,results=all_results,
        interpretation='Pixel gradient counts are measured in supported RGB channels; analytic slopes are output-logit derivatives. Task loss backprop additionally measures actual nonzero logit gradients.',
        training_updates=0))


if __name__=='__main__':main()
