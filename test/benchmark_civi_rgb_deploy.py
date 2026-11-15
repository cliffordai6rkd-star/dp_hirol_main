"""Verify online deployable RGB matches cached-predictor evaluation and time it."""
from pathlib import Path
import argparse
import json
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from run_response_decoder_ablation import Runtime
from run_scene_torque_response_predictor import SceneTorquePredictor
from response_decoder_ablation import DiagnosticResponseDecoder
from civi_patch_response_renderer import PatchResponseRenderer
from civi_semantic_patch_renderer import SemanticPatchRenderer
from civi_rgb_deploy import CIVISharedRGB
from student_dual_path_common import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['base','predictor','old','cache','checkpoint','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--dependencies',type=Path)
    a=p.parse_args();a.variant='A';a.smoke=False
    torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    c=OmegaConf.load(Path(__file__).parent/'config/response_decoder_gate_ablation.yaml')
    if a.dependencies:
        deps=json.loads(a.dependencies.read_text())
        for key in ['teacher_sha256','predictor_sha256','decoder_initial_sha256','condition_sha256']:c[key]=deps[key]
        c.strict_reference=False
    r=Runtime(a,c)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    options=saved.get('manifest',{}).get('parameterization',dict(input_alpha=True,film_alpha=True,projection=True,ca_response=True,parameterization='hard'))
    if options.get('architecture')=='shared_semantic_patch_rgb':model=SemanticPatchRenderer(r.owner)
    elif options.get('architecture')=='shared_patch_rgb':model=PatchResponseRenderer(r.owner)
    else:model=DiagnosticResponseDecoder(r.owner,.1,**options)
    model.to(r.device).eval().requires_grad_(False).load_state_dict(saved['ema_model'],strict=True)
    state=torch.load(a.predictor/'best.pt',map_location='cpu',weights_only=False)['model']
    predictor=SceneTorquePredictor(state['visual_scale'],state['torque_scale']).to(r.device).eval().requires_grad_(False);predictor.load_state_dict(state,strict=True)
    deployed=CIVISharedRGB(r.teacher,predictor,model).to(r.device).eval()
    ids=[int(np.flatnonzero((r.datasets['validation'].episodes==38)&(r.val.phase_labels==2))[0])]
    b=r.batch('validation',ids);im,wr=r.inputs(b);sp,al=r.gates(im,wr);pred=r.predicted(b,'validation')
    obs={k:v for k,v in b['obs'].items() if k not in ['contact_alpha','contact_phase']}
    with torch.no_grad(),r.amp():
        scene=deployed.scene(im).float()
        if isinstance(model,SemanticPatchRenderer):model.visual_tokens=scene
        reference=model(im,pred,sp,al)
    # Cache prediction was generated from BF16 DINO batch32, whereas online batch1
    # uses its actual current scene features. Quantify, never claim exact equality.
    for _ in range(5):deployed(obs)
    torch.cuda.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    timings=[]
    for _ in range(30):
        torch.cuda.synchronize();started=time.perf_counter();actual=deployed(obs);torch.cuda.synchronize();timings.append((time.perf_counter()-started)*1000)
    allowed=('condition_network',) if isinstance(model,PatchResponseRenderer) else ('backbone','camera_embedding','spatial_modulations','condition_network')
    active_modules=[deployed.scene,deployed.preprocess,deployed.predictor]
    params={}
    for module in active_modules:
        for p in module.parameters():params[id(p)]=p.numel()
    for name,p in model.named_parameters():
        if not name.startswith('owner.') or name.split('.')[1] in allowed:
            params[id(p)]=p.numel()
    result=dict(checkpoint=str(a.checkpoint),input='Two most recent causal RGB observations + raw-unit causal torque histories; no future/offline phases/teacher targets',
        per_stream_batch=1,output='Enhanced RGB only; downstream policy has no Adapter or token target dependency',
        online_vs_fp16_cached_prediction_rgb_mse=float(torch.cat([(actual[k]-reference[k]).flatten() for k in actual]).square().mean()),
        latency_ms=dict(median=float(np.median(timings)),p90=float(np.quantile(timings,.9)),mean=float(np.mean(timings))),
        active_parameters=sum(params.values()),parameter_bytes_fp32=4*sum(params.values()),
        additional_peak_activation_bytes=torch.cuda.max_memory_allocated()-before,
        process_allocated_bytes=before,note='Benchmark process also holds frozen teacher for audit/reference; process allocation is NOT minimal inference footprint. Active weights + extra peak are reported separately.',
        gpu=torch.cuda.get_device_name())
    a.output.parent.mkdir(parents=True,exist_ok=True);write_json(a.output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
