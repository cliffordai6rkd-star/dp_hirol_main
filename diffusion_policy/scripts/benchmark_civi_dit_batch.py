"""Measure exact KD64 microbatch alternatives without any optimizer updates."""
import argparse
import copy
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import Subset

from diffusion_policy.distillation.dit_rgb import MixedPlan, move, slice_batch, fp32_reference, write_json
from diffusion_policy.workspace.train_civi_student_dit_workspace import TrainCiviStudentDiTWorkspace


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    a=p.parse_args();cfg=OmegaConf.load('diffusion_policy/config/train_civi_student_dit_minimal.yaml')
    cfg.output_dir=a.output;cfg.data.phase_counts=[16,16,32];cfg.training.global_batch=64
    ws=TrainCiviStudentDiTWorkspace(cfg,a.output);ws.prepare_data()
    indices=ws.plan.next();batch=move(next(iter(ws.loader(Subset(ws.datasets['train'],indices),batch_size=64))),ws.device)
    # Read cached Condition/teacher context; visual encode stays B2 in every arm.
    rows=[]
    for micro in [2,4,8,16]:
        cfg.training.microbatch=micro
        ws.model.set_stage_mode(True)
        def one():
            ws.optimizer.zero_grad(set_to_none=True)
            total=torch.zeros((),device=ws.device)
            for start in range(0,64,micro):
                result=ws.kd(slice_batch(batch,start,start+micro),probe=0)
                loss=result['kd'].sum()/64;loss.backward();total+=loss.detach()
            return total
        try:
            torch.cuda.reset_peak_memory_stats(ws.device)
            one();torch.cuda.synchronize()
            durations=[]
            for _ in range(3):
                started=time.monotonic();loss=one();torch.cuda.synchronize();durations.append(time.monotonic()-started)
            row=dict(microbatch=micro,global_batch=64,fixed_visual_encode_batch=2,seconds=durations,
                median_seconds=float(torch.tensor(durations).median()),loss=float(loss),
                peak_memory_bytes=torch.cuda.max_memory_allocated(ws.device),optimizer_updates=0)
        except torch.OutOfMemoryError:
            ws.optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
            row=dict(microbatch=micro,error='CUDA OOM; no optimizer update',optimizer_updates=0)
        rows.append(row);print(json.dumps(row),flush=True)
        write_json(Path(a.output)/'batch_benchmark.json',rows)
    ws.check_frozen()


if __name__=='__main__':main()
