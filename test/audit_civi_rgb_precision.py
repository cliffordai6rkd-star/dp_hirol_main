"""Compare RGB response/readout sensitivity under FP32 and BF16, without updates."""
from pathlib import Path
import argparse
import json

import torch
from torch.nn import functional as F
from omegaconf import OmegaConf

from run_response_decoder_ablation import Runtime
from audit_civi_rgb_iteration import panels
from student_dual_path_common import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['base','predictor','old','cache','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.variant='A';a.smoke=False
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    c=OmegaConf.load(Path(__file__).parent/'config/response_decoder_gate_ablation.yaml')
    a.output.mkdir(parents=True,exist_ok=False);r=Runtime(a,c)
    ids=panels(r)['train_contact'][:4];b=r.batch('train',ids);im,wr=r.inputs(b);sp,al=r.gates(im,wr);pred=r.predicted(b,'train')
    def encode(images,mode,grad=False):
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=mode=='bf16'):
            return r.teacher.obs_encoder.encode_image_context(images,context_adapter=r.owner.context_adapter,input_grad=grad).float()
    noise={k:torch.randn_like(v) for k,v in im.items()};records={};gradients={}
    for mode in ['bf16','fp32']:
        z0=encode(im,mode).detach();records[mode]=[]
        for scale in [0.,1e-6,1e-5,1e-4,1e-3,1e-2]:
            out={k:(v+noise[k]*sp[k]*al[...,None,None,None]*scale).clamp(0,1) for k,v in im.items()}
            with torch.no_grad():z=encode(out,mode)
            rgb=torch.cat([(out[k]-im[k]).flatten(1) for k in im],1)
            records[mode].append(dict(scale=scale,rgb_rms=float(rgb.square().mean().sqrt()),response_mse=float((z-z0).square().mean()),
                response_rms=float((z-z0).square().mean().sqrt()),identity_max_error=float((encode(im,mode)-z0).abs().max())))
        pixel={k:v.clone().requires_grad_(True) for k,v in im.items()};z=encode(pixel,mode,True)
        loss=(z-z0-pred).square().mean()
        grad=torch.autograd.grad(loss,list(pixel.values()));gradients[mode]=torch.cat([v.flatten() for v in grad])
    result=dict(indices=ids,sensitivity=records,pixel_gradient_cosine=float(F.cosine_similarity(gradients['bf16'],gradients['fp32'],dim=0)),
        gradient_norm={k:float(v.norm()) for k,v in gradients.items()},training_updates=0)
    write_json(a.output/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
