"""Train-only per-image reachability reference; never a deployable shared student."""
from pathlib import Path
import argparse
import json
import time
import shutil

import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf

from run_response_decoder_ablation import Runtime
from run_civi_rgb_iteration import fixed_rows,summarize
from audit_civi_rgb_iteration import panels
from student_dual_path_common import write_json
from paired_response_full_common import file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['base','predictor','old','cache','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--samples',type=int,default=24)
    p.add_argument('--dependencies',type=Path)
    p.add_argument('--renderer-checkpoint',type=Path,help='Frozen shared EMA reference, evaluated on the exact Oracle subset')
    a=p.parse_args();a.variant='A';a.smoke=False
    c=OmegaConf.load(Path(__file__).parent/'config/response_decoder_gate_ablation.yaml')
    if a.dependencies:
        d=json.loads(a.dependencies.read_text())
        for k in ['teacher_sha256','predictor_sha256','condition_sha256','decoder_initial_sha256']:c[k]=d[k]
        c.strict_reference=False
        assert file_hash(a.base/'adapter/best_adapter.pt')==d['adapter_sha256']
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a.output.mkdir(parents=True,exist_ok=False)
    r=Runtime(a,c);panel=panels(r);ids=panel['train_contact']
    r.manifest['threshold']=r.threshold
    r.manifest['executed_source_sha256']={str(Path(__file__)):file_hash(Path(__file__))}
    shutil.copy2(Path(__file__),a.output/Path(__file__).name)
    assert 1<=a.samples<=len(ids)
    ids=[ids[i] for i in np.linspace(0,len(ids)-1,a.samples).round().astype(int)]
    panel['oracle_train_ids']=ids
    shared_rows=[]
    if a.renderer_checkpoint:
        saved=torch.load(a.renderer_checkpoint,map_location='cpu',weights_only=False)
        assert saved['manifest']['frozen']==r.manifest['frozen']
        assert saved['manifest']['teacher_sha256']==c.teacher_sha256
        r.network.load_state_dict(saved['ema_model'],strict=True)
        r.network.eval().requires_grad_(False);r.check_frozen()
        r.manifest['shared_reference_checkpoint']=dict(path=str(a.renderer_checkpoint),sha256=file_hash(a.renderer_checkpoint),step=saved['step'])
    write_json(a.output/'panels.json',panel)
    rows=[];history=[];visuals=[];started=time.monotonic()
    for offset in range(0,len(ids),2):
        b=r.batch('train',ids[offset:offset+2]);im,wr=r.inputs(b);pred=r.predicted(b,'train');sp,al=r.gates(im,wr)
        z0=r.encode(im).detach();support={k:sp[k]*al[...,None,None,None] for k in im}
        if a.renderer_checkpoint:
            with torch.no_grad():
                with r.amp():shared,stats=r.network(im,pred,sp,al,trace=True)
                shared_z=r.encode(shared)
                shared_rows.extend(fixed_rows(r,b,pred,im,shared,shared_z,z0,stats))
        variables={k:nn.Parameter(torch.zeros_like(v)) for k,v in im.items()}
        opt=torch.optim.Adam(list(variables.values()),lr=.003)
        def render():
            return {k:torch.where(support[k]>0,(im[k]+v).clamp(0,1),im[k]) for k,v in variables.items()}
        for step in range(1,a.steps+1):
            opt.zero_grad(set_to_none=True);out=render();z=r.encode(out,True)
            loss=(z-z0-pred).square().mean();assert torch.isfinite(loss)
            loss.backward();assert sum(float(v.grad.square().sum()) for v in variables.values())>0
            opt.step()
            with torch.no_grad():
                for k,v in variables.items():v.copy_(v.clamp(-.1*support[k],.1*support[k]))
            if step==1 or step%50==0:
                history.append(dict(chunk=offset,step=step,loss=float(loss.detach()),elapsed_seconds=time.monotonic()-started))
                write_json(a.output/'log.json',history)
        with torch.no_grad():
            out=render();z=r.encode(out)
            counts={name:torch.zeros(len(pred),device=r.device) for name in ['allowed_channels','total_channels','hard_clipped_channels','range_clipped_channels']}
            for k in im:
                allowed=(support[k]>0).expand_as(im[k]);v=variables[k]
                counts['allowed_channels']+=allowed.flatten(1).sum(-1)/len(im)
                counts['total_channels']+=im[k][0].numel()/len(im)
                counts['hard_clipped_channels']+=((v.abs()>=.1*support[k])&allowed).flatten(1).sum(-1)/len(im)
                counts['range_clipped_channels']+=(((im[k]+v<0)|(im[k]+v>1))&allowed).flatten(1).sum(-1)/len(im)
            rows.extend(fixed_rows(r,b,pred,im,out,z,z0,counts))
            for j,index in enumerate(ids[offset:offset+2]):
                visuals.append(dict(index=index,episode=int(b['episode'][j]),original={k:v[j].cpu() for k,v in im.items()},
                    rendered={k:v[j].cpu() for k,v in out.items()},predicted=pred[j].cpu(),student=(z[j]-z0[j]).cpu(),teacher=(b['target'][j]-b['zero'][j]).float().cpu(),
                    **(dict(shared_rendered={k:v[j].cpu() for k,v in shared.items()},shared_student=(shared_z[j]-z0[j]).cpu()) if a.renderer_checkpoint else {})))
        print(json.dumps(dict(event='train_oracle',finished=offset+len(pred),total=len(ids),last_loss=float(loss))),flush=True)
    r.check_frozen();write_json(a.output/'samples.json',rows);torch.save(visuals,a.output/'visuals.pt')
    result=dict(manifest=r.manifest,final=summarize(rows,c),per_image_steps=a.steps,
        rgb_shared_optimizer_updates=0,pixel_optimizer_updates=a.steps*((len(ids)+1)//2),
        shared_reference=summarize(shared_rows,c) if shared_rows else None,
        purpose='TRAINING sample reachability only; no validation Oracle used as supervision; not a shared model/global upper bound',
        elapsed_seconds=time.monotonic()-started)
    if shared_rows:write_json(a.output/'shared_reference_samples.json',shared_rows)
    write_json(a.output/'summary.json',result);print(json.dumps(dict(event='complete',contact=result['final']['phases']['contact'])),flush=True)


if __name__=='__main__':main()
