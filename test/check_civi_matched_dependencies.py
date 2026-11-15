"""Actual new-teacher cache, physical-zero, frozen-path and RGB-gradient smoke."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import torch
from omegaconf import OmegaConf

from run_response_decoder_ablation import Runtime,gradient_norms
from run_scene_torque_response_predictor import SceneTorquePredictor
from student_dual_path_common import write_json
from paired_response_full_common import file_hash
from diffusion_policy.model.vision.render_net_attention import zero_wrench_reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['base','predictor','old','cache','dependencies','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.variant='A';a.smoke=False;a.output.mkdir(parents=True,exist_ok=False)
    d=json.loads(a.dependencies.read_text());c=OmegaConf.load(Path(__file__).parent/'config/response_decoder_gate_ablation.yaml')
    for k in ['teacher_sha256','predictor_sha256','condition_sha256','decoder_initial_sha256']:c[k]=d[k]
    c.strict_reference=False
    torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    r=Runtime(a,c)
    assert r.teacher.future_phase_wrench_key is None,'Deployment must not preprocess future force'
    assert file_hash(a.base/'adapter/best_adapter.pt')==d['adapter_sha256']
    saved=torch.load(a.predictor/'best.pt',map_location='cpu',weights_only=False)['model']
    net=SceneTorquePredictor(saved['visual_scale'],saved['torque_scale']).to(r.device).eval().requires_grad_(False);net.load_state_dict(saved,strict=True)
    rows=[]
    for split in ['train','validation']:
        ds=r.datasets[split]
        phases=ds.dataset.phase_labels
        visual=np.load(a.base/'paired_cache'/split/'visual.npy',mmap_mode='r')
        for phase in range(3):
            ids=np.flatnonzero(phases==phase);i=int(ids[len(ids)//2]);b=r.batch(split,[i]);im,wr=r.inputs(b)
            zero=zero_wrench_reference(r.teacher,b['obs']);zim,zwr,_,_=r.teacher.prepare_context_inputs_with_future_contact(zero)
            assert r.teacher.normalizer[r.teacher.wrench_key].unnormalize(zwr).abs().max()<1e-5
            assert all(torch.equal(zim[k],im[k]) for k in im)
            with torch.no_grad(),torch.autocast('cuda',enabled=False):
                actual=r.teacher._encode_context(b['obs'],apply_curriculum_mask=False).float()
                actual0=r.teacher._encode_context(zero,apply_curriculum_mask=False).float()
            errors=dict(target=float((actual-b['target'].float()).square().mean()),zero=float((actual0-b['zero'].float()).square().mean()))
            assert max(errors.values())<1e-5,(split,i,errors)
            local=torch.from_numpy(np.array(visual[[i]],copy=True)).to(r.device)
            pooled=local.reshape(1,2,392,256).mean(2);tau=b['obs'][r.teacher.wrench_key][:,:2].float()
            with torch.no_grad(),torch.autocast('cuda',enabled=False):
                predicted=net(local,pooled,tau,torch.arange(784,device=r.device))
                p0=net(local,pooled,torch.zeros_like(tau),torch.arange(784,device=r.device))
            prediction_error=float((predicted-r.predicted(b,split)).square().mean())
            assert prediction_error<1e-6 and p0.abs().max()<1e-7
            sp,al=r.gates(im,wr);z0=r.encode(im).detach()
            with r.amp():
                rendered=r.network(im,predicted,sp,al)
                closed=r.network(im,predicted,sp,torch.zeros_like(al))
            assert all(torch.equal(closed[k],im[k]) for k in im)
            loss=(r.encode(rendered,True)-z0-predicted).square().mean()
            r.network.zero_grad(set_to_none=True);loss.backward();gradients=gradient_norms(r.network)
            assert gradients['head']>0 and np.isfinite(gradients['head'])
            r.network.zero_grad(set_to_none=True);r.check_frozen()
            rows.append(dict(split=split,index=i,episode=int(b['episode'][0]),phase=phase,teacher_cache_error=errors,predictor_cache_error=prediction_error,
                gradients=gradients,alpha=al.detach().cpu().tolist(),image_hash={k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in im.items()}))
    write_json(a.output/'smoke.json',dict(dependencies=d,checks=rows,alpha0_exact_identity=True,physical_zero_before_normalizer=True,
        future_input_required=False,all_frozen_verified=True,rgb_optimizer_updates=0,cache_consistency_verified=True))
    print(json.dumps(dict(event='matched_smoke_passed',teacher_sha256=d['teacher_sha256'],samples=len(rows),optimizer_updates=0)),flush=True)


if __name__=='__main__':main()
