"""Equal-budget Linear/MLP calibration, with no renderer or torque input."""
from pathlib import Path
import argparse,copy,json,time
import numpy as np
import torch
from student_dual_path_common import load_inputs,model_for,write_json,tensor_hash
from diffusion_policy.model.diffusion.ema_model import EMAModel

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['inputs','prepared','output-dir']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--kind',choices=['linear','mlp'],required=True)
    p.add_argument('--steps',type=int,default=2000)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=True
    dev=torch.device('cuda:0');teacher,_,initial=load_inputs(a.inputs,dev)
    model,_,_=model_for(initial,'c_film_mlp' if a.kind=='mlp' else 'c_film_linear')
    adapter=model.context_adapter.to(dev)
    for module in adapter.modules():
        if isinstance(module,torch.nn.Linear):output=module
    torch.nn.init.zeros_(output.weight);torch.nn.init.zeros_(output.bias)
    ema_adapter=copy.deepcopy(adapter).eval().requires_grad_(False)
    ema=EMAModel(ema_adapter,**dict(initial['cfg'].ema))
    optim=torch.optim.AdamW(adapter.parameters(),lr=2e-4,weight_decay=1e-4,betas=(.9,.999))
    arrays={s:{k:np.load(a.prepared/s/f'{k}.npy',mmap_mode='r') for k in ['visual','zero']}
            for s in ['train','validation']}
    norm=teacher.obs_encoder.context_norm
    def get(split,ids):
        return [torch.from_numpy(np.array(arrays[split][k][ids],copy=True)).to(dev)
                for k in ['visual','zero']]
    @torch.no_grad()
    def evaluate():
        totals={}
        for s in arrays:
            # Full held-out validation; fixed 1,024 training rows for progress.
            limit=len(arrays[s]['visual']) if s=='validation' else min(1024,len(arrays[s]['visual']))
            total=0.
            for start in range(0,limit,64):
                x,y=get(s,slice(start,min(start+64,limit)))
                with torch.autocast('cuda',dtype=torch.bfloat16):z=norm(ema_adapter(x))
                total+=(z.float()-y.float()).square().mean().item()*len(x)
            totals[s+'_zero_mse']=total/limit
        return totals
    history=[];start=time.monotonic()
    def log(step):
        row=dict(step=step,**evaluate(),elapsed_seconds=time.monotonic()-start)
        history.append(row);write_json(a.output_dir/'log.json',history);print(json.dumps(row),flush=True)
    log(0);rng=np.random.default_rng(42);order=rng.permutation(len(arrays['train']['visual']));pos=0
    for step in range(1,a.steps+1):
        if pos+64>len(order):order=rng.permutation(len(order));pos=0
        ids=order[pos:pos+64];pos+=64;x,y=get('train',ids)
        optim.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):z=norm(adapter(x))
        loss=(z.float()-y.float()).square().mean();loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.)
        optim.step();ema.step(adapter)
        if step%200==0 or step==a.steps:log(step)
    state={k:v.detach().cpu() for k,v in ema_adapter.state_dict().items()}
    torch.save(dict(state=state,kind=a.kind,steps=a.steps,hidden_dim=128,history=history),a.output_dir/'adapter.pt')
    write_json(a.output_dir/'summary.json',dict(kind=a.kind,optimizer_steps=a.steps,batch_size=64,
        initial_validation_mse=history[0]['validation_zero_mse'],final_validation_mse=history[-1]['validation_zero_mse'],
        best_validation_mse=min(r['validation_zero_mse'] for r in history),
        state_sha256=tensor_hash(state),parameters=sum(x.numel() for x in adapter.parameters()),
        cached_inputs='BF16 pre-adapter visual tokens; frozen final context_norm applied after adapter',
        target='same physical-zero teacher final context as production calibration',elapsed_seconds=time.monotonic()-start))

if __name__=='__main__':main()
