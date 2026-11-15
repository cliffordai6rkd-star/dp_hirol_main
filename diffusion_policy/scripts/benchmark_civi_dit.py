"""Actual-model KD correctness/throughput checks. Never suspends other jobs.

Only --prepare-updates 2 mutates a temporary renderer, with shared ledger charge.
All timing/precision/compile comparisons use the same restored weights, samples,
fixed probes and global batch. No cached DiT output or condition memory.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from diffusion_policy.dataset.civi_dit_dataset import historical_datasets
from diffusion_policy.distillation.budget import UpdateBudget
from diffusion_policy.distillation.dit_rgb import move, slice_batch, fp32_reference, write_json, state_sha
from diffusion_policy.workspace.train_civi_student_dit_workspace import TrainCiviStudentDiTWorkspace


def grad_vector(ws):
    return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten() for p in ws.parameters])


def difference(a,b):
    a,b = a.float().flatten(),b.float().flatten()
    return dict(max_abs=float((a-b).abs().max()),rms=float((a-b).square().mean().sqrt()),
                relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-20)),
                cosine=float(torch.nn.functional.cosine_similarity(a,b,dim=0)))


def prepared(ws):
    datasets,identity = historical_datasets(ws.cfg.data,ws.preprocessor.get_config())
    ds = datasets['train']
    from diffusion_policy.distillation.dit_rgb import MixedPlan
    plan = MixedPlan(ds.phases,ds.episodes,ws.cfg.data.phase_counts)
    indices = plan.next()
    raw = next(iter(ws.loader(Subset(ds,indices),batch_size=len(indices))))
    batch = move(raw,ws.device)
    tokens, masks, alphas = [], {k:[] for k in ws.options['image_keys']}, []
    b = ws.cfg.validation.encode_batch
    with torch.no_grad(),fp32_reference():
        for start in range(0,len(indices),b):
            part = slice_batch(batch,start,start+b)
            images,wrench = ws.inputs(part)
            tokens.append(ws.teacher._encode_context(part['obs'],apply_curriculum_mask=False))
            m,c = ws.model.condition_network(images,wrench)
            for k in masks:
                masks[k].append(m[k])
            alphas.append(c['contact_alpha'])
    batch['static'] = dict(tokens=torch.cat(tokens),patch_attention={**{k:torch.cat(v) for k,v in masks.items()},'alpha':torch.cat(alphas)})
    # Native teacher entrypoint and this DiT regression entrypoint use the same
    # full context. Assert native context reuse is exactly equal.
    part = slice_batch(batch,0,b)
    with torch.no_grad(),fp32_reference():
        images,wrench = ws.inputs(part)
        native = ws.teacher._encode_context(part['obs'],apply_curriculum_mask=False)
        via = ws.teacher.obs_encoder(images=images,wrench_history=wrench,
            future_contact=ws.teacher.prepare_context_inputs_with_future_contact(part['obs'])[3],
            contact_alpha=part['obs']['contact_alpha'][:,:ws.teacher.n_obs_steps])
        torch.testing.assert_close(via,native,rtol=0,atol=0)
    return batch,indices,datasets


def backward(ws,batch,reference=False,record=None):
    ws.optimizer.zero_grad(set_to_none=True)
    losses,contexts = [],[]
    context = fp32_reference() if reference else torch.autocast('cuda',enabled=False)
    with context:
        for start in range(0,len(batch['action']),ws.cfg.training.microbatch):
            result = ws.kd(slice_batch(batch,start,start+ws.cfg.training.microbatch),reference=reference,probe=0,record=record)
            loss = result['kd'].sum()/len(batch['action'])
            loss.backward()
            losses.append(loss.detach());contexts.append(result['context'].detach())
    return torch.stack(losses).sum(),torch.cat(contexts),grad_vector(ws).detach().clone()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--iterations',type=int,default=3)
    parser.add_argument('--compile',action='store_true');parser.add_argument('--prepare-updates',type=int,choices=(0,2),default=2)
    parser.add_argument('--reuse',help='Use a previously saved benchmark tensor state without more updates')
    args = parser.parse_args()
    cfg = OmegaConf.load('diffusion_policy/config/train_civi_student_dit_minimal.yaml');cfg.output_dir=args.output
    ws = TrainCiviStudentDiTWorkspace(cfg,output_dir=args.output)
    batch,indices,datasets = prepared(ws)
    if args.reuse:
        ws.model.load_state_dict(torch.load(args.reuse,map_location=ws.device,weights_only=False),strict=True)
    budget = UpdateBudget(cfg.budget.ledger_root,cfg.budget.task_id,cfg.budget.maximum_new_updates,args.output)
    for step in range(args.prepare_updates):
        backward(ws,batch,reference=True)
        if step==1:
            assert ws.model.wrench_encoder.weight_ih_l0.grad.norm()>0, 'Direct torque LSTM has no gradient after second backward'
            assert ws.model.backbone.stem.body[0].weight.grad.norm()>0
        torch.nn.utils.clip_grad_norm_(ws.parameters,cfg.training.max_grad_norm,error_if_nonfinite=True)
        budget.charge(step+1,indices)
        ws.optimizer.step()
    ws.check_frozen()
    torch.save(ws.model.state_dict(),Path(args.output)/'benchmark_weights.pt')
    fp_loss,fp_context,fp_grad = backward(ws,batch,True)
    precision = {}
    for name,flags in [('renderer_bf16',dict(renderer=True,visual=False,dit=False)),
                       ('all_bf16',dict(renderer=True,visual=True,dit=True)),
                       ('visual_bf16',dict(renderer=False,visual=True,dit=False)),
                       ('dit_bf16',dict(renderer=False,visual=False,dit=True))]:
        cfg.training.amp.update(flags)
        loss,context,gradient = backward(ws,batch)
        precision[name] = dict(loss=float(loss),fp32_loss=float(fp_loss),context=difference(fp_context,context),
                               parameter_gradient=difference(fp_grad,gradient))
        print(json.dumps(dict(precision=name,**precision[name])),flush=True)
    cfg.training.amp.update(renderer=True,visual=False,dit=False)
    # Direct visual input gradients, including a small constrained RGB change.
    part = slice_batch(batch,0,ws.cfg.training.microbatch); images,_ = ws.inputs(part)
    rgb = {k:(v+1e-4).clamp(0,1).detach().requires_grad_() for k,v in images.items()}
    rgb_grads = {}
    for name,ref in [('fp32',True),('bf16',False)]:
        cfg.training.amp.visual = not ref
        z = ws.encode(rgb,True,ref)
        result = ws.kd(part,reference=True,probe=0)
        predicted = ws.teacher.model(result['noisy'],result['rho'],z)
        loss = (predicted-result['expected'].detach()).square().mean()
        rgb_grads[name] = torch.cat([g.flatten() for g in torch.autograd.grad(loss,list(rgb.values()))])
    cfg.training.amp.visual = False
    timing = []
    eager_values = {}
    def measure(name):
        torch.cuda.reset_peak_memory_stats(ws.device)
        torch.cuda.synchronize();start=time.monotonic()
        loss,context,gradient = backward(ws,batch)
        torch.cuda.synchronize();first=time.monotonic()-start
        durations=[]
        for _ in range(args.iterations):
            torch.cuda.synchronize();start=time.monotonic();backward(ws,batch);torch.cuda.synchronize()
            durations.append(time.monotonic()-start)
        if name=='eager':
            eager_values.update(gradient=gradient.clone(),context=context.clone(),loss=loss.clone())
        row = dict(name=name,first_step_seconds=first,stable_seconds=durations,
                   stable_median_seconds=float(torch.tensor(durations).median()),
                   samples_per_second=len(indices)/float(torch.tensor(durations).median()),
                   peak_memory_bytes=torch.cuda.max_memory_allocated(ws.device),
                   gradient_vs_fp32=difference(fp_grad,gradient),gradient_vs_eager=difference(eager_values['gradient'],gradient),
                   context_vs_eager=difference(eager_values['context'],context),loss=float(loss))
        timing.append(row);print(json.dumps(row),flush=True)
        write_json(Path(args.output)/'benchmark.json',dict(indices=indices,precision=precision,
                   rgb_gradient=difference(rgb_grads['fp32'],rgb_grads['bf16']),timing=timing))
    measure('eager')
    if args.compile:
        original_visual,original_teacher,original_student = ws.visual,ws.dit_target,ws.dit_student
        for kind in ['renderer','visual','dit_teacher','dit_student']:
            try:
                if kind=='renderer':
                    ws.model.compile_backbone(backend='inductor',mode='default')
                elif kind=='visual':
                    ws.visual = torch.compile(original_visual,backend='inductor')
                elif kind=='dit_teacher':
                    ws.dit_target = torch.compile(original_teacher,backend='inductor')
                else:
                    ws.dit_student = torch.compile(original_student,backend='inductor')
                measure('compile_'+kind)
            except Exception as exc:
                timing.append(dict(name='compile_'+kind,error=str(exc),fallback='eager'))
                print(json.dumps(timing[-1]),flush=True)
            finally:
                ws.model._compiled_backbone_forward=None
                ws.visual,ws.dit_target,ws.dit_student=original_visual,original_teacher,original_student
                torch._dynamo.reset();torch.cuda.empty_cache()
    # Include actual DataLoader/H2D throughput, same batch exposure and compute.
    data=Subset(datasets['train'],indices*args.iterations)
    loader_times={}
    for workers in [0,2]:
        cfg.data.num_workers=workers
        started=time.monotonic();samples=0
        for raw in ws.loader(data,batch_size=len(indices)):
            current=move(raw,ws.device);current['static']=batch['static']
            backward(ws,current);samples+=len(current['action'])
        torch.cuda.synchronize()
        loader_times[str(workers)]=dict(seconds=time.monotonic()-started,samples=samples,
                                      samples_per_second=samples/(time.monotonic()-started),includes_worker_startup=True)
    ws.check_frozen()
    write_json(Path(args.output)/'benchmark.json',dict(indices=indices,precision=precision,
        rgb_gradient=difference(rgb_grads['fp32'],rgb_grads['bf16']),timing=timing,dataloader=loader_times,
        dtype_by_module=dict(master_weights='FP32',renderer='BF16',condition='FP32',dino='FP32',projection='FP32',
            adapter='FP32',context_norm='FP32',teacher_dit='FP32',student_dit='FP32',loss_reduction='FP32'),
        real_optimizer_updates=args.prepare_updates))


if __name__=='__main__':
    main()
