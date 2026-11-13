"""Nonzero five-window memorization: direct-token diagnostics only."""
import argparse
import json
from pathlib import Path
import time

import torch

from diffusion_policy.scripts.probe_rgb_capacity import select_contact_windows
from diffusion_policy.scripts.run_tau_capacity_probe import weights, metric_arrays, summary
from diffusion_policy.scripts.tau_probe_models import TokenMLP, ResidualTransformer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=2000)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.25)
    raw=torch.load(args.data,map_location='cpu',weights_only=False)
    indices=select_contact_windows(raw['records'],raw['delta'])
    data={key:raw[key][indices].flatten(0,1).cuda() for key in ('base','delta','tau')}
    base,tau,y=data['base'],data['tau'],data['delta']
    assert (y.square().mean((1,2))>1e-4).all()
    scale=y.square().mean().sqrt()
    args.output.mkdir(parents=True,exist_ok=True)
    report={'indices':indices,'records':[raw['records'][i] for i in indices],
            'purpose':'memorization only; no heldout claim','results':{}}
    for kind in ('mlp','transformer'):
        torch.manual_seed(42)
        model=(TokenMLP() if kind=='mlp' else ResidualTransformer(hidden=256)).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        history=[]
        started=time.monotonic()
        for step in range(args.steps+1):
            if step%250==0 or step==args.steps:
                with torch.no_grad():
                    scores=summary(metric_arrays(model(base,tau)*scale,y))
                history.append({'step':step,'seconds':time.monotonic()-started,**scores})
                print(f'{kind} five contact step={step} ratio={scores["relative_mse"]:.4g} '
                      f'top10={scores["top10_relative_mse"]:.4g}',flush=True)
                report['results'][kind]={'history':history,'params':sum(p.numel() for p in model.parameters())}
                (args.output/'five_contact_results.json').write_text(json.dumps(report,indent=2)+'\n')
            if step==args.steps: break
            idx=torch.randperm(len(base),device='cuda')[:4]
            pred=model(base[idx],tau[idx])
            loss=(weights(y[idx])*(pred-y[idx]/scale).square().mean(-1)).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
        del model,optimizer
        torch.cuda.empty_cache()


if __name__=='__main__':main()
