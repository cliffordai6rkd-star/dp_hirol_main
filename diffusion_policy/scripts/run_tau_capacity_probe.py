"""Controlled contact-only token regression versus a small flow-matching DiT.

Uses paired FP32 targets, disjoint episodes and target-defined fixed metrics.
This diagnoses direct-token capacity, not deployment-equivalent RGB rendering.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from diffusion_policy.scripts.tau_probe_models import TokenMLP, ResidualTransformer, sample_flow


def weights(target):
    energy = target.square().mean(-1)
    weight = .5 + .5 * (energy / energy.mean(-1, keepdim=True).clamp_min(1e-8)).clamp(max=10)
    return weight / weight.mean(-1, keepdim=True)


def metric_arrays(pred, target):
    error = (pred.float()-target.float()).square().mean(-1)
    energy = target.float().square().mean(-1)
    k = max(1, math.ceil(energy.shape[-1]*.1))
    top = energy.topk(k, dim=-1).indices
    cos = torch.nn.functional.cosine_similarity(pred.float(), target.float(), dim=-1, eps=1e-8)
    return {'mse': error.mean(-1), 'energy': energy.mean(-1),
            'top10_mse': error.gather(-1, top).mean(-1),
            'top10_energy': energy.gather(-1, top).mean(-1),
            'top10_cosine': cos.gather(-1, top).mean(-1),
            'weighted_mse': (weights(target)*error).mean(-1)}


def summary(metrics):
    v = {key: float(value.double().mean()) for key,value in metrics.items()}
    v['relative_mse'] = v['mse']/max(v['energy'],1e-12)
    v['top10_relative_mse'] = v['top10_mse']/max(v['top10_energy'],1e-12)
    return v


@torch.no_grad()
def evaluate(model, data, scale, kind, seed=777, chunk=8):
    all_metrics = {}
    for offset in range(0,len(data['base']),chunk):
        base,tau = data['base'][offset:offset+chunk], data['tau'][offset:offset+chunk]
        pred = (sample_flow(model, base, tau, steps=16, seed=seed+offset)
                if kind == 'dit_flow' else model(base,tau)) * scale
        scores = metric_arrays(pred, data['delta'][offset:offset+chunk])
        for key,value in scores.items():
            all_metrics.setdefault(key,[]).append(value.cpu())
    arrays = {key: torch.cat(parts) for key,parts in all_metrics.items()}
    return summary(arrays), arrays


def select(data, indices):
    return {key: value[indices] for key,value in data.items()}


def train_run(kind, seed, data, output, steps, lr, eval_every=250):
    torch.manual_seed(seed)
    if kind == 'mlp' or kind == 'mlp_no_tau':
        model = TokenMLP(dim=data['train']['base'].shape[-1], tau_dim=56, width=256).cuda()
    else:
        # Full-width noise needs an injective projection/output. Hidden<256
        # leaves Gaussian components outside the velocity head's column space.
        model = ResidualTransformer(dim=256,hidden=256,depth=3,heads=4).cuda()
    if kind == 'mlp_no_tau':
        data = {split: {**items,'tau': torch.zeros_like(items['tau'])} for split,items in data.items()}
    scale = data['train']['delta'].square().mean().sqrt()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train = data['train']
    curves=[]
    best=float('inf')
    best_state=None
    best_step=0
    start=time.monotonic()
    for step in range(steps+1):
        if step % eval_every == 0 or step == steps:
            model.eval()
            tr,_=evaluate(model,data['train'],scale,kind)
            va,_=evaluate(model,data['validation'],scale,kind)
            curves.append({'step':step,'train':tr,'validation':va,'seconds':time.monotonic()-start})
            score=.5*(va['relative_mse']+va['top10_relative_mse'])
            if score < best:
                best=score;best_step=step
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            (output/f'{kind}_{seed}_progress.json').write_text(json.dumps(curves,indent=2)+'\n')
            print(f'{kind} seed={seed} step={step} train_ratio={tr["relative_mse"]:.4f} '
                  f'val_ratio={va["relative_mse"]:.4f} top10={va["top10_relative_mse"]:.4f}',flush=True)
            model.train()
        if step == steps:
            break
        idx=torch.randint(len(train['base']),(4,),device='cuda')
        base,tau,y=train['base'][idx],train['tau'][idx],train['delta'][idx]/scale
        if kind == 'dit_flow':
            t=torch.rand(len(idx),device='cuda')
            noise=torch.randn_like(y)
            xt=(1-t[:,None,None])*noise+t[:,None,None]*y
            pred=model(base,tau,noisy=xt,t=t)
            target=y-noise
        else:
            pred=model(base,tau)
            target=y
        loss=(weights(y)*(pred-target).square().mean(-1)).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f'nonfinite {kind} step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    model.load_state_dict(best_state)
    model.eval()
    report={'kind':kind,'seed':seed,'params':sum(p.numel() for p in model.parameters()),
            'steps':steps,'best_validation_step':best_step,'residual_scale':float(scale),
            'seconds':time.monotonic()-start,'curve':curves}
    for split in ('train','validation','test'):
        scores,arrays=evaluate(model,data[split],scale,kind)
        report[split]=scores
        report[split+'_per_observation']={k:v.tolist() for k,v in arrays.items()}
    if kind != 'mlp_no_tau':
        # A single fixed derangement within each split tests whether learned
        # predictions actually use tau. Images and targets remain unchanged.
        test=data['test']
        shuffled={**test,'tau':test['tau'].roll(len(test['tau'])//2,0)}
        report['test_shuffled_tau'],_=evaluate(model,shuffled,scale,kind)
    if kind == 'dit_flow':
        report['test_other_sampling_seeds']=[evaluate(model,data['test'],scale,kind,seed=s)[0] for s in (778,779)]
    torch.save({'model':best_state,'kind':kind,'seed':seed,'scale':float(scale)},output/f'{kind}_{seed}_best.pt')
    (output/f'{kind}_{seed}_result.json').write_text(json.dumps(report,indent=2)+'\n')
    del model,optimizer
    torch.cuda.empty_cache()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--models',nargs='+',default=['mlp','transformer','dit_flow','mlp_no_tau'])
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43])
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--eval-every',type=int,default=250)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.25)
    raw=torch.load(args.data,map_location='cpu',weights_only=False)
    data={}
    for split in ('train','validation','test'):
        rows=[i for i,r in enumerate(raw['records']) if r['split']==split]
        data[split]={key:raw[key][rows].flatten(0,1).cuda() for key in ('base','delta','tau')}
        assert (data[split]['delta'].square().mean((1,2))>1e-4).all()
    args.output.mkdir(parents=True,exist_ok=True)
    baseline={split:summary(metric_arrays(torch.zeros_like(x['delta']),x['delta'])) for split,x in data.items()}
    (args.output/'baselines.json').write_text(json.dumps(baseline,indent=2)+'\n')
    results=[]
    for kind in args.models:
        for seed in args.seeds:
            result_path=args.output/f'{kind}_{seed}_result.json'
            if result_path.exists():
                results.append(json.loads(result_path.read_text()))
            else:
                results.append(train_run(kind,seed,data,args.output,args.steps,args.lr,args.eval_every))
    (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('PROBES_DONE',flush=True)


if __name__=='__main__':
    main()
