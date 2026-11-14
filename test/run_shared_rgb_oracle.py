"""Shared CNN oracle, matched D renderer, and per-image oracle on a pinned teacher.

This is a diagnostic generator, not a new RenderNet. All use the same frozen
Adapter, deployed Condition Network, pixel budget and paired feature objective.
"""
from pathlib import Path
import argparse
import copy
import json
import time

import dill
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from student_dual_path_common import load_inputs, model_for, datasets_from_reference, tensor_hash, write_json, to_device
from paired_response_full_common import (active_response_weights, deterministic_mode, explicit_zero_alpha,
    grouped_metrics, sample_metrics, file_hash)


class TinySharedGenerator(nn.Module):
    """Spatial CNN + full-history torque MLP; no sample IDs/embedding table."""
    def __init__(self, history):
        super().__init__()
        self.torque=nn.Sequential(nn.Linear(history*7,64),nn.GELU(),nn.Linear(64,32),nn.GELU())
        self.encoder=nn.Sequential(nn.Conv2d(5,16,5,2,2),nn.GELU(),nn.Conv2d(16,32,3,2,1),nn.GELU(),
                                   nn.Conv2d(32,32,3,1,1),nn.GELU())
        self.decoder=nn.Sequential(nn.Conv2d(64,32,3,1,1),nn.GELU(),nn.Conv2d(32,3,3,1,1))
        nn.init.zeros_(self.decoder[-1].weight);nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, images, wrench):
        b,t=wrench.shape[:2];condition=self.torque(wrench.flatten(0,1).flatten(1));out={}
        for camera,(key,image) in enumerate(images.items()):
            x=image.flatten(0,1);labels=x.new_zeros(len(x),2,*x.shape[-2:]);labels[:,camera]=1
            h=self.encoder(torch.cat([x,labels],1));c=condition[...,None,None].expand(-1,-1,*h.shape[-2:])
            delta=F.interpolate(self.decoder(torch.cat([h,c],1)),size=x.shape[-2:],mode='bilinear',align_corners=False)
            out[key]=delta.tanh().reshape_as(image)
        return out


class StudyDataset(torch.utils.data.Dataset):
    def __init__(self,dataset,study,split):
        self.dataset=dataset;self.rows=json.loads((study/split/'samples.json').read_text())
        self.arrays={k:np.load(study/split/(k+'.npy'),mmap_mode='r') for k in ['target','zero']}
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];x=self.dataset[row['index']]
        x.update({k:torch.from_numpy(np.array(v[i],copy=True)) for k,v in self.arrays.items()})
        x.update(phase=row['phase'],episode=row['episode'],index=i,original_index=row['index'],teacher_held_out=row['teacher_held_out'])
        return x


def load_context(root,study,dev):
    teacher,cfg,initial=load_inputs(study/'inputs',dev)
    model,_,_=model_for(initial,'c_film_mlp')
    checkpoint=root/'review_1000/c_mlp_checkpoint_1000.ckpt'
    payload=torch.load(checkpoint,map_location='cpu',pickle_module=dill,weights_only=False)
    model.load_state_dict(payload['state_dicts']['ema_model'],strict=True)
    adapter=torch.load(root/'calibration_mlp_cuda/adapter.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(v,model.context_adapter.state_dict()[k]) for k,v in adapter['state'].items())
    model.to(dev).eval().requires_grad_(False);deterministic_mode(model)
    train,val,_=datasets_from_reference(study/'inputs',root/'decoded_cuda')
    datasets={s:StudyDataset(ds,study,s) for s,ds in [('train',train),('validation',val)]}
    return teacher,model,datasets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--kind',choices=['shared','rendernet','individual'],required=True)
    p.add_argument('--steps',type=int,default=200);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--epsilon',type=float,default=.1);p.add_argument('--check-only',action='store_true')
    p.add_argument('--response-only',action='store_true',help='Isolate response reachability from old-adapter absolute mismatch')
    p.add_argument('--individual-fixed-zero',action='store_true',help='Keep the individual oracle zero-torque RGB exactly at the original image')
    p.add_argument('--individual-split',choices=['both','train','validation'],default='both')
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    assert not a.individual_fixed_zero or a.kind=='individual'
    output=a.study/('oracle_'+a.kind+('_response_only' if a.response_only else '')+('_fixed_zero' if a.individual_fixed_zero else ''));output.mkdir(exist_ok=False)
    dev=torch.device('cuda:0');teacher,owner,datasets=load_context(a.root,a.study,dev)
    baseline=tensor_hash(owner.state_dict());frozen=dict(teacher=tensor_hash(teacher.state_dict()),adapter=tensor_hash(owner.context_adapter.state_dict()),condition=tensor_hash(owner.condition_network.state_dict()))
    amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16)
    trainrows=datasets['train'].rows;valrows=datasets['validation'].rows
    train_contact=[i for i,r in enumerate(trainrows) if r['phase']==2]
    train_free=[i for i,r in enumerate(trainrows) if r['phase']==0]
    quiet=np.asarray(datasets['train'].arrays['target'][train_free],dtype=np.float32)-np.asarray(datasets['train'].arrays['zero'][train_free],dtype=np.float32)
    threshold=max(.001,float(np.quantile(np.sqrt(np.mean(quiet**2,-1)),.99)))
    if a.kind=='shared':network=TinySharedGenerator(8).to(dev);parameters=list(network.parameters())
    elif a.kind=='rendernet':owner.set_training_stage('rgb');deterministic_mode(owner);network=owner;parameters=owner.rgb_parameters()
    else:network=None;parameters=[]
    optim=torch.optim.AdamW(parameters,lr=5e-5 if a.kind=='rendernet' else 1e-3,weight_decay=1e-4) if parameters else None
    original_network=copy.deepcopy(network.state_dict()) if network is not None else None
    manifest=dict(kind=a.kind,steps=a.steps,epsilon=a.epsilon,seed=42,
        teacher_checkpoint_sha256=file_hash(a.study/'inputs/pinned_teacher_ema.ckpt'),adapter_sha256=frozen['adapter'],
        baseline_checkpoint_state_sha256=baseline,parameters=sum(p.numel() for p in parameters),
        objective='paired full delta MSE + 10 RGB MSE' if a.response_only else 'global token MSE + response-weighted token MSE + paired full delta MSE + 0.25 zero-token MSE + 10 RGB MSE',
        zero_branch='physical zero normalized by new teacher; frozen condition network recomputes gate/alpha',
        bound='abs(delta RGB) <= epsilon * deployed predicted gate * alpha, then clip [0,1]',
        threshold=threshold,teacher_old_adapter='Existing MLP calibrated on old teacher; frozen and identical in all cases',
        initialization='Tiny CNN starts at identity; D starts at existing step1000; architecture comparison includes this difference',
        individual_fixed_zero=a.individual_fixed_zero,individual_split=a.individual_split,
        optimizer='Tiny CNN AdamW lr1e-3; D uses requested reduced lr5e-5; no claim of matched optimization difficulty')
    write_json(output/'manifest.json',manifest)

    def batch_for(split,ids):
        return to_device(next(iter(DataLoader(torch.utils.data.Subset(datasets[split],ids),batch_size=len(ids)))),dev)

    def inputs(batch):
        im,wr,_,_=teacher.prepare_context_inputs_with_future_contact(batch['obs'])
        wr0=teacher.normalizer[teacher.wrench_key].normalize(torch.zeros_like(wr))
        return im,wr,wr0

    def encode(im,grad=False):
        with amp():return teacher.obs_encoder.encode_image_context(im,context_adapter=owner.context_adapter,input_grad=grad).float()

    @torch.no_grad()
    def gates(im,wr):
        with amp():masks,condition=owner.condition_network(im,wr)
        return {k:owner._mask_gate(masks[k].flatten(0,1),im[k]).reshape_as(masks[k]).float()*condition['contact_alpha'][...,None,None,None] for k in im},condition['contact_alpha']

    def rendered(im,wr,gate,pixels=None):
        if pixels is not None:raw=pixels
        elif a.kind=='shared':
            with amp():raw=network(im,wr)
            raw={k:a.epsilon*v.float()*gate[k] for k,v in raw.items()}
        else:
            with torch.backends.cudnn.flags(enabled=False),amp():images=owner(im,wr)
            raw={k:images[k].float()-im[k].float() for k in im}
        return {k:(im[k]+torch.maximum(-a.epsilon*gate[k],torch.minimum(a.epsilon*gate[k],raw[k]))).clamp(0,1) for k in im}

    def objective(batch,pixel_pair=None,grad=True):
        im,wr,wr0=inputs(batch);gate,alpha=gates(im,wr);gate0,alpha0=gates(im,wr0)
        image=rendered(im,wr,gate,None if pixel_pair is None else pixel_pair[0])
        image0=rendered(im,wr0,gate0,None if pixel_pair is None else pixel_pair[1])
        z,z0=encode(image,grad),encode(image0,grad)
        target,zero=batch['target'].float(),batch['zero'].float();dt=target-zero;ds=z-z0
        errors=(z-target).square().mean(-1);w=(dt.detach().norm(dim=-1)-1e-6).clamp_min(0)
        dw=active_response_weights(dt,threshold)
        losses=dict(global_mse=errors.mean(),response=(w*errors).sum()/w.sum().clamp_min(1e-7),
            delta=((ds-dt).square().mean(-1)*dw).sum()/dw.sum().clamp_min(1e-12),
            zero=(z0-zero).square().mean(),rgb=torch.cat([(image[k]-im[k]).flatten(1) for k in im],1).square().mean())
        loss=losses['delta']+10*losses['rgb']
        if not a.response_only:loss=loss+losses['global_mse']+losses['response']+.25*losses['zero']
        return loss,losses,(z,z0,image,image0,im,alpha,alpha0)

    history=[];started=time.monotonic()
    @torch.no_grad()
    def evaluate(step):
        metrics={};visuals=[]
        for split in ['train','validation']:
            records=[]
            for raw in DataLoader(datasets[split],batch_size=16,num_workers=4,pin_memory=True):
                batch=to_device(raw,dev);_,_,out=objective(batch,grad=False)
                z,z0,image,image0,im,alpha,alpha0=out
                rows=sample_metrics(z,z0,batch['target'],batch['zero'],image,image0,im,alpha,alpha0,threshold)
                for i,row in enumerate(rows):
                    index=int(batch['index'][i]);metadata=datasets[split].rows[index]
                    records.append(dict(index=index,episode=int(batch['episode'][i]),phase=int(batch['phase'][i]),teacher_held_out=metadata['teacher_held_out'],**row))
                    if split=='validation' and row['rgb_max']>0 and int(batch['phase'][i])==2 and not any(v['episode']==int(batch['episode'][i]) for v in visuals):
                        visuals.append(dict(episode=int(batch['episode'][i]),index=index,original={k:v[i].cpu() for k,v in im.items()},rendered={k:v[i].cpu() for k,v in image.items()},teacher_response=(batch['target'][i].float()-batch['zero'][i].float()).cpu(),student_response=(z[i]-z0[i]).cpu()))
            metrics[split]=grouped_metrics(records)
            metrics[split]['teacher_held_out_contact']=grouped_metrics([r for r in records if r['teacher_held_out']])['phases']['contact']
            write_json(output/f'{split}_samples_{step}.json',records)
        row=dict(step=step,metrics=metrics,elapsed_seconds=time.monotonic()-started);history.append(row);write_json(output/'log.json',history)
        torch.save(visuals,output/f'visuals_{step}.pt')
        print(json.dumps(dict(event='evaluation',kind=a.kind,step=step,train_contact=metrics['train']['phases']['contact'],validation_contact=metrics['validation']['phases']['contact'],teacher_held_out=metrics['validation']['teacher_held_out_contact'])),flush=True)
        return metrics

    # Alpha closure is a standalone check; it is never substituted for deployed zero torque.
    check=batch_for('train',train_contact[:2]);im,wr,wr0=inputs(check)
    with torch.no_grad(),explicit_zero_alpha(owner),amp():
        exact=owner(im,wr0)
    assert all(torch.equal(exact[k],im[k]) for k in im)
    if a.check_only:
        if parameters:
            loss,_,_=objective(check);loss.backward();assert any(p.grad is not None and p.grad.norm()>0 for p in parameters)
        write_json(output/'preflight.json',dict(alpha_zero_identity=True,gradient=True));return
    if a.kind=='individual':
        results={};rows_out=[];visuals=[]
        for split in (['train','validation'] if a.individual_split=='both' else [a.individual_split]):
            rows=datasets[split].rows;selected=[]
            for ep in sorted({r['episode'] for r in rows}):
                ids=[i for i,r in enumerate(rows) if r['phase']==2 and r['episode']==ep]
                if ids:selected.extend(ids[int((len(ids)-1)*f)] for f in [1/3,2/3])
            if split=='train':
                all_eps=sorted({rows[i]['episode'] for i in selected})
                chosen={all_eps[i] for i in np.linspace(0,len(all_eps)-1,6).round().astype(int)}
                selected=[i for i in selected if rows[i]['episode'] in chosen]
            for start in range(0,len(selected),a.batch):
                ids=selected[start:start+a.batch];batch=batch_for(split,ids);im,wr,wr0=inputs(batch)
                variables=[{k:nn.Parameter(torch.zeros_like(v),requires_grad=(branch==0 or not a.individual_fixed_zero)) for k,v in im.items()} for branch in range(2)]
                op=torch.optim.Adam([p for d in variables for p in d.values() if p.requires_grad],lr=.003)
                gate,_=gates(im,wr);gate0,_=gates(im,wr0)
                for step in range(a.steps):
                    op.zero_grad(set_to_none=True);loss,losses,out=objective(batch,variables);loss.backward();op.step()
                    with torch.no_grad():
                        for d,g in zip(variables,[gate,gate0]):
                            for k in d:d[k].clamp_(-a.epsilon*g[k],a.epsilon*g[k])
                    if step%50==0:print(json.dumps(dict(event='individual',split=split,chunk=start,step=step,loss=float(loss))),flush=True)
                with torch.no_grad():
                    _,_,out=objective(batch,variables,False);z,z0,image,image0,im,alpha,alpha0=out
                    metrics=sample_metrics(z,z0,batch['target'],batch['zero'],image,image0,im,alpha,alpha0,threshold)
                for i,row in enumerate(metrics):rows_out.append(dict(split=split,index=ids[i],episode=int(batch['episode'][i]),phase=2,teacher_held_out=rows[ids[i]]['teacher_held_out'],**row))
                for i in range(len(ids)):
                    visuals.append(dict(split=split,episode=int(batch['episode'][i]),index=ids[i],
                        original={k:v[i].cpu() for k,v in im.items()},rendered={k:v[i].cpu() for k,v in image.items()},
                        zero_rendered={k:v[i].cpu() for k,v in image0.items()},
                        teacher_response=(batch['target'][i].float()-batch['zero'][i].float()).cpu(),
                        student_response=(z[i]-z0[i]).cpu()))
            results[split]=grouped_metrics([r for r in rows_out if r['split']==split])
            write_json(output/'per_sample.json',rows_out)
            torch.save(visuals,output/'visuals.pt')
        final=results
    else:
        initial=evaluate(0);rng=np.random.default_rng(42);trainlog=[]
        # Identical precomputed batch IDs for shared CNN and D; contact samples spread across all episodes.
        def cycle(ids):
            episodes=sorted({trainrows[i]['episode'] for i in ids});queues={}
            while True:
                for ep in rng.permutation(episodes):
                    if not queues.get(ep):queues[ep]=list(rng.permutation([i for i in ids if trainrows[i]['episode']==ep]))
                    yield int(queues[ep].pop())
        contact_stream,free_stream=cycle(train_contact),cycle(train_free)
        for step in range(1,a.steps+1):
            ids=[next(contact_stream) for _ in range(a.batch-1)]+[next(free_stream)]
            batch=batch_for('train',ids);optim.zero_grad(set_to_none=True)
            loss,losses,out=objective(batch);loss.backward();norm=float(nn.utils.clip_grad_norm_(parameters,1.));optim.step()
            if step==1 or step%25==0:
                row=dict(step=step,indices=ids,losses={k:float(v.detach()) for k,v in losses.items()},gradient_norm=norm,elapsed_seconds=time.monotonic()-started)
                trainlog.append(row);write_json(output/'training.json',trainlog);print(json.dumps(row),flush=True)
            if step%100==0 or step==a.steps:final=evaluate(step)
        torch.save(network.state_dict(),output/'network.pt')
    assert tensor_hash(teacher.state_dict())==frozen['teacher']
    assert tensor_hash(owner.context_adapter.state_dict())==frozen['adapter']
    assert tensor_hash(owner.condition_network.state_dict())==frozen['condition']
    write_json(output/'summary.json',dict(manifest=manifest,final=final,initial=None if a.kind=='individual' else initial,
        frozen_verified=True,elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(event='complete',kind=a.kind)),flush=True)


if __name__=='__main__':main()
