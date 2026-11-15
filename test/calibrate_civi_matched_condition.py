"""Match the existing condition network to a newly pinned teacher; no RGB updates."""
from pathlib import Path
import argparse
import copy
import json
import time

import numpy as np
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader,Subset

from student_dual_path_common import load_inputs,datasets_from_reference,model_for,to_device,write_json,tensor_hash
from paired_response_full_common import file_hash,deterministic_mode
from new_teacher_response_common import FullPairedDataset
from run_scene_torque_response_predictor import phase_plan
from diffusion_policy.model.vision.render_net_attention import context_effect_patch_weights


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['base','old','source']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--steps',type=int,default=1000)
    a=p.parse_args();out=a.base/'condition';out.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();dev=torch.device('cuda:0')
    teacher,_,initial=load_inputs(a.base/'inputs',dev);teacher_hash=tensor_hash(teacher.state_dict())
    manifest=json.loads((a.base/'paired_cache/manifest.json').read_text())
    assert file_hash(a.base/'inputs/pinned_teacher_ema.ckpt')==manifest['teacher_checkpoint_sha256']
    assert teacher_hash==manifest['teacher_state_sha256']
    owner,cfg,_=model_for(initial,'c_film_mlp')
    owner.load_state_dict(torch.load(a.source/'d/best.pt',map_location='cpu',weights_only=False)['ema_model'],strict=True)
    owner.to(dev).eval().requires_grad_(False);deterministic_mode(owner)
    net=owner.condition_network.requires_grad_(True)
    # CuDNN RNNs must allocate training reserve space for backward. Dropout
    # was already disabled above; this mode enables training without noise.
    net.train()
    ema=copy.deepcopy(net).requires_grad_(False).eval()
    attention=cfg.render_net.high_attention_render
    train,val,identity=datasets_from_reference(a.base/'inputs',a.old/'decoded_cuda');assert identity==manifest['splits']
    datasets={s:FullPairedDataset(ds,a.base/'paired_cache'/s) for s,ds in [('train',train),('validation',val)]}
    options=OmegaConf.create(dict(seed=42,steps=a.steps,phase_mix=[2,2,4]));plan=phase_plan(train.phase_labels,datasets['train'].episodes,options)
    np.save(out/'train_batches.npy',plan)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.0002,weight_decay=.0001)
    def forward(module,b):
        im,wr,_,_=teacher.prepare_context_inputs_with_future_contact(b['obs'])
        d=(b['target'].float()-b['zero'].float()).reshape(len(b['phase']),2,2,196,256)
        patches={k:d[:,:,i].norm(dim=-1) for i,k in enumerate(owner.image_keys)}
        targets=context_effect_patch_weights(patches,selection=attention.get('selection','soft'),
            percentile=attention.get('percentile',90.),percentiles=attention.get('percentiles',[80.,99.]),
            normalization=attention.get('normalization'),strength_scale=attention.get('strength_scale'))
        with torch.autocast('cuda',dtype=torch.bfloat16):masks,prediction=module(im,wr)
        numerator=torch.zeros(len(d),device=dev);denominator=torch.zeros_like(numerator)
        smooth=[]
        for k,mask in masks.items():
            predicted=F.adaptive_avg_pool2d(mask.flatten(0,1),(14,14)).reshape(len(d),2,196).float()
            valid=(patches[k].amax(-1)>1e-6).float()
            errors=F.smooth_l1_loss(predicted,targets[k],reduction='none').mean(-1)
            numerator+=(valid*errors).sum(-1);denominator+=valid.sum(-1)
            smooth.extend([mask[...,1:,:].sub(mask[...,:-1,:]).abs().flatten(1).mean(-1),mask[...,:,1:].sub(mask[...,:,:-1]).abs().flatten(1).mean(-1)])
        region=numerator/denominator.clamp_min(1)
        alpha=(prediction['contact_alpha'].float()-b['obs']['contact_alpha'][:,:2].float()).square().mean(-1)
        smoothness=torch.stack(smooth).float().mean(0)
        loss=attention.get('context_patch_loss_weight',1.)*region+alpha+attention.get('smoothness_loss_weight',.02)*smoothness
        return loss,region,alpha,prediction['contact_alpha'].float()
    @torch.no_grad()
    def evaluate(step):
        rows=[]
        for raw in DataLoader(datasets['validation'],batch_size=16,num_workers=4,pin_memory=True):
            b=to_device(raw,dev);loss,region,alpha,pred=forward(ema,b)
            rows.extend(dict(index=int(b['index'][i]),episode=int(b['episode'][i]),phase=int(b['phase'][i]),
                loss=float(loss[i]),region_loss=float(region[i]),alpha_mse=float(alpha[i]),alpha=float(pred[i].mean())) for i in range(len(loss)))
        result={str(ep):{name:dict(samples=len(x),loss=float(np.mean([r['loss'] for r in x])),
            alpha_mse=float(np.mean([r['alpha_mse'] for r in x])),region_loss=float(np.mean([r['region_loss'] for r in x])))
            for phase,name in enumerate(['free','precontact','contact']) if len(x:=[r for r in rows if r['episode']==ep and r['phase']==phase])}
            for ep in sorted({r['episode'] for r in rows})}
        score=float(np.mean([r['loss'] for r in rows if r['episode'] in [38,44,53,60]]))
        write_json(out/f'validation_{step}.json',dict(rows=rows,episodes=result,selection_loss=score))
        history.append(dict(step=step,selection_loss=score,episodes=result,elapsed_seconds=time.monotonic()-started));write_json(out/'log.json',history)
        print(json.dumps(dict(event='matched_condition_validation',step=step,selection_loss=score)),flush=True)
        return score
    history=[];best=evaluate(0);stale=0;best_step=0
    def save(step):return dict(state=ema.state_dict(),model=net.state_dict(),optimizer=optimizer.state_dict(),step=step,
        teacher_sha256=manifest['teacher_checkpoint_sha256'],initial_condition_sha256=file_hash(a.source/'d/best.pt'))
    torch.save(save(0),out/'best.pt')
    for step,raw in enumerate(DataLoader(datasets['train'],batch_sampler=plan.tolist(),num_workers=4,pin_memory=True),1):
        b=to_device(raw,dev);optimizer.zero_grad(set_to_none=True)
        loss,_,_,_=forward(net,b);loss.mean().backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
        with torch.no_grad():
            for dest,src in zip(ema.parameters(),net.parameters()):dest.lerp_(src,.01)
        if step%200==0 or step==a.steps:
            score=evaluate(step)
            if score<best-1e-6:best,best_step,stale=score,step,0;torch.save(save(step),out/'best.pt')
            else:stale+=1
            torch.save(save(step),out/'latest.pt')
            if stale>=10:break
    assert not any(p.requires_grad or p.grad is not None for p in teacher.parameters())
    assert tensor_hash(teacher.state_dict())==teacher_hash
    write_json(out/'summary.json',dict(steps=step,best_step=best_step,selection_loss=best,
        teacher_sha256=manifest['teacher_checkpoint_sha256'],condition_sha256=file_hash(out/'best.pt'),
        validation_selection_episodes=[38,44,53,60],retained_evaluation_episodes=[5,26],rgb_updates=0))


if __name__=='__main__':main()
