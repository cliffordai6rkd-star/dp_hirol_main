"""Real-checkpoint correctness checks, including DDP gradient equivalence.

Run single GPU for strict resume/export/physical-zero checks (four accounted
optimizer calls); run torchrun --nproc_per_node=2 ... --ddp for gradient-only
single/DDP comparison. No other process is signalled or paused.
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusion_policy.distillation.dit_rgb import (
    MixedPlan, diffusion_pair, element_mse, fp32_reference, move, rng_state,
    restore_rng, slice_batch, state_sha, write_json,
)
from diffusion_policy.distillation.budget import UpdateBudget
from diffusion_policy.workspace.train_force_aware_diffusion_workspace import initialize_distributed
from diffusion_policy.workspace.train_civi_student_dit_workspace import TrainCiviStudentDiTWorkspace
from diffusion_policy.scripts.benchmark_civi_dit import prepared, backward, difference, grad_vector
from diffusion_policy.policy.civi_rgb import export_rgb, CiviRGB


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--ddp',action='store_true')
    parser.add_argument('--weights',required=True)
    args=parser.parse_args()
    cfg=OmegaConf.load('diffusion_policy/config/train_civi_student_dit_minimal.yaml');cfg.output_dir=args.output
    distributed=initialize_distributed(cfg.training)
    ws=TrainCiviStudentDiTWorkspace(cfg,args.output,distributed)
    batch,indices,datasets=prepared(ws)
    ws.model.load_state_dict(torch.load(args.weights,map_location=ws.device,weights_only=False),strict=True)
    ws.plan=MixedPlan(datasets['train'].phases,datasets['train'].episodes,cfg.data.phase_counts)
    ws.data_identity={'verification':'real fixed eight-sample batch; exact full-data manifests are checked in formal smoke'}
    result={}
    if args.ddp:
        loss,_,reference=backward(ws,batch,True)
        ws.train_model=DDP(ws.model,device_ids=[distributed.local_rank],broadcast_buffers=False)
        ws.optimizer.zero_grad(set_to_none=True)
        # Recursive rank sharding; includes cached condition and sample IDs.
        def shard(x):
            if isinstance(x,dict):return {k:shard(v) for k,v in x.items()}
            return x[distributed.rank::distributed.world_size]
        local=shard(batch)
        accumulation=len(local['action'])//cfg.training.microbatch
        losses=[]
        with fp32_reference():
            for part in range(accumulation):
                import contextlib
                sync=ws.train_model.no_sync() if part+1<accumulation else contextlib.nullcontext()
                with sync:
                    output=ws.kd(slice_batch(local,part*cfg.training.microbatch,(part+1)*cfg.training.microbatch),reference=True,probe=0)
                    value=output['kd'].sum()/len(local['action']);value.backward();losses.append(value.detach())
        current=grad_vector(ws)
        value=torch.stack(losses).sum();dist.all_reduce(value);value/=distributed.world_size
        result=dict(world_size=distributed.world_size,loss_max_abs=float((value-loss).abs()),
                    gradient=difference(reference,current),optimizer_updates=0)
        assert result['loss_max_abs']<1e-6
        assert result['gradient']['relative_l2']<1e-4,result
        ws.check_frozen()
        if distributed.is_main:write_json(Path(args.output)/'ddp_equivalence.json',result)
        dist.destroy_process_group();return
    budget=UpdateBudget(cfg.budget.ledger_root,cfg.budget.task_id,cfg.budget.maximum_new_updates,args.output)
    def update(probe_step):
        ws.model.set_stage_mode(True)
        backward(ws,batch,True)
        torch.nn.utils.clip_grad_norm_(ws.parameters,cfg.training.max_grad_norm,error_if_nonfinite=True)
        budget.charge(probe_step,indices);ws.optimizer.step();ws.ema.step(ws.model)
        ws.step+=1;ws.plan.next()
    update(1);update(2)
    ws.check_frozen()
    part=slice_batch(batch,0,cfg.training.microbatch)
    images,wrench=ws.inputs(part)
    masks=part['static']['patch_attention']
    with torch.no_grad(),fp32_reference():
        zeros=ws.model(images,wrench,cached_condition=({k:masks[k] for k in images},{'contact_alpha':torch.zeros_like(masks['alpha'])}))
        assert all(torch.equal(zeros[k],images[k]) for k in images)
        physical={k:v.clone() for k,v in part['obs'].items()};physical[ws.teacher.wrench_key].zero_()
        zero_wrench=ws.preprocessor(physical)[0]
        zero_masks,zero_condition=ws.model.condition_network(images,zero_wrench)
        actual=ws.model(images,zero_wrench)
        result['physical_zero']=dict(alpha_mean=float(zero_condition['contact_alpha'].mean()),
            rgb_rms=float(torch.stack([(actual[k]-images[k]).square().mean() for k in images]).mean().sqrt()))
        result['alpha0_exact_identity']=True
        # Frozen visual path must retain nonzero gradients to rendered RGB.
    rgb={k:v.detach().requires_grad_() for k,v in images.items()}
    z=ws.encode(rgb,True,True)
    noisy,rho,_=diffusion_pair(ws.teacher,part['action'],sample_ids=part['sample_id'].tolist())
    with torch.no_grad():target=ws.teacher.model(noisy,rho,part['static']['tokens'])
    loss=element_mse(ws.teacher.model(noisy,rho,z),target).mean()
    grad=torch.autograd.grad(loss,list(rgb.values()))
    result['rgb_gradient_norm']=float(torch.cat([v.flatten() for v in grad]).norm())
    assert result['rgb_gradient_norm']>0
    ws.save()
    saved_step=ws.step
    saved_rng=rng_state()
    update(3)
    expected={k:v.detach().clone() for k,v in ws.model.state_dict().items()}
    expected_optimizer=copy.deepcopy(ws.optimizer.state_dict())
    expected_next=ws.plan.next()
    expected_draw=torch.randn(4,device=ws.device)
    # Reload the same online state with its *own* AdamW, never EMA+online state.
    ws.restore(Path(args.output)/'latest.pt')
    assert ws.step==saved_step
    update(3)
    changes={k:float((v-expected[k]).abs().max()) for k,v in ws.model.state_dict().items() if not torch.equal(v,expected[k])}
    result['resume_parameter_differences']=changes
    write_json(Path(args.output)/'resume_difference.json',result)
    assert not changes, changes
    assert ws.plan.next()==expected_next
    assert torch.equal(torch.randn(4,device=ws.device),expected_draw)
    result['strict_resume_same_next_update']=True
    result['actual_optimizer_calls']=4
    ws.check_frozen()
    # Export the saved EMA; verify exact online deployment against that state.
    export_rgb(Path(args.output)/'latest.pt',Path(args.output)/'civi_rgb.pt',ws.preprocessor)
    deployed=CiviRGB.load(Path(args.output)/'civi_rgb.pt',ws.device)
    expected_model=copy.deepcopy(ws.model).eval().requires_grad_(False)
    expected_model.load_state_dict(torch.load(Path(args.output)/'latest.pt',map_location=ws.device,weights_only=False)['ema_model'],strict=True)
    with torch.no_grad(),fp32_reference():
        expected_rgb=expected_model(images,wrench)
        actual_rgb=deployed(images,part['obs'][ws.teacher.wrench_key])
        for k in images:torch.testing.assert_close(actual_rgb[k],expected_rgb[k],rtol=0,atol=0)
    keys=list(torch.load(Path(args.output)/'civi_rgb.pt',map_location='cpu',weights_only=False)['renderer'])
    assert not any(any(term in key for term in ('dino','adapter','predictor','oracle')) for key in keys)
    result['export_exact']=True;result['deployment_parameter_count']=sum(p.numel() for p in deployed.parameters())
    write_json(Path(args.output)/'correctness.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
