"""Separate the renderer's gain from the calibrated visual adapter's gain."""
from pathlib import Path
import argparse,json
import dill,numpy as np
import torch
from student_dual_path_common import write_json
from diffusion_policy.model.vision.student_context_adapter import StudentContextAdapter

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();torch.set_num_threads(4)
    dev=torch.device('cuda:0');torch.backends.cuda.matmul.allow_tf32=True
    teacher=torch.load(root/'inputs/pinned_teacher_ema.ckpt',map_location='cpu',pickle_module=dill,weights_only=False)
    state=teacher['state_dicts']['ema_model'];norm=torch.nn.LayerNorm(256).to(dev).eval().requires_grad_(False)
    norm.load_state_dict({k:state['obs_encoder.context_norm.'+k] for k in ['weight','bias']})
    del teacher,state
    arrays={k:np.load(root/'prepared_cuda/validation'/f'{k}.npy',mmap_mode='r')
            for k in ['visual','target','zero','response']}
    phases=np.array([r['phase'] for r in json.loads((root/'experiments_cuda/a_baseline/per_sample_step0.json').read_text())])
    result={}
    for kind in ['linear','mlp']:
        cal=torch.load(root/f'calibration_{kind}_cuda/adapter.pt',map_location='cpu',weights_only=False)
        adapter=StudentContextAdapter(256,kind=kind,hidden_dim=128).to(dev).eval().requires_grad_(False)
        adapter.load_state_dict(cal['state'],strict=True)
        errors=[];numerators=[];denominators=[];zero_errors=[];teacher_response=[]
        with torch.no_grad():
            for start in range(0,len(phases),64):
                x={k:torch.from_numpy(np.array(v[start:start+64],copy=True)).to(dev) for k,v in arrays.items()}
                with torch.autocast('cuda',dtype=torch.bfloat16):z=norm(adapter(x['visual']))
                e=(z.float()-x['target'].float()).square().mean(-1)
                w=(x['response'].float()-1e-6).clamp_min(0)
                errors.extend(e.mean(-1).cpu().tolist())
                zero_errors.extend((z.float()-x['zero'].float()).square().mean((1,2)).cpu().tolist())
                teacher_response.extend((x['target'].float()-x['zero'].float()).square().mean((1,2)).cpu().tolist())
                numerators.extend((e*w).sum(-1).cpu().tolist());denominators.extend(w.sum(-1).cpu().tolist())
        result[kind]={}
        for phase,name in [(-1,'all'),(0,'free'),(1,'precontact'),(2,'contact')]:
            select=np.ones(len(phases),dtype=bool) if phase==-1 else phases==phase
            glob=float(np.mean(np.array(errors)[select]));den=float(np.sum(np.array(denominators)[select]))
            weighted=float(np.sum(np.array(numerators)[select]))/den if den else 0.
            result[kind][name]=dict(samples=int(select.sum()),token_mse=glob,response_weighted_mse=weighted,
                objective=glob+weighted,zero_calibration_mse=float(np.mean(np.array(zero_errors)[select])),
                teacher_response_mse=float(np.mean(np.array(teacher_response)[select])))
    write_json(root/'identity_baseline.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
