"""Equal-budget RGB distillation; production global + response-weighted MSE only."""
from pathlib import Path
import argparse,copy,json,time
import dill
import numpy as np
import torch
from torch.utils.data import DataLoader
from student_dual_path_common import (load_inputs,model_for,datasets_from_reference,
    PreparedDataset,to_device,write_json,tensor_hash,VARIANTS)
from diffusion_policy.model.vision.response_distillation import response_weighted_token_mse
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['inputs','prepared','image-cache','adapter','output-dir']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--steps',type=int,default=5000);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--validate-every',type=int,default=1000);p.add_argument('--no-compile',action='store_true')
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    dev=torch.device('cuda:0');teacher,teacher_cfg,initial=load_inputs(a.inputs,dev)
    model,cfg,transfer=model_for(initial,a.variant)
    calibration=torch.load(a.adapter,map_location='cpu',weights_only=False)
    model.context_adapter.load_state_dict(calibration['state'],strict=True)
    model.to(dev);model.set_training_stage('rgb');model.set_stage_mode(True)
    ema_model=copy.deepcopy(model).eval().requires_grad_(False)
    ema=EMAModel(ema_model,**dict(cfg.ema))
    train,val,splits=datasets_from_reference(a.inputs,a.image_cache)
    train,val=PreparedDataset(train,a.prepared/'train'),PreparedDataset(val,a.prepared/'validation')
    loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=8,pin_memory=True,
                      persistent_workers=True,drop_last=True,generator=torch.Generator().manual_seed(42))
    vloader=DataLoader(val,batch_size=a.batch_size,num_workers=4,pin_memory=True,persistent_workers=True)
    optim=torch.optim.AdamW(model.rgb_parameters(),**dict(cfg.optimizer))
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optim,mode='min',factor=.5,patience=8,
                                                        threshold=1e-4,min_lr=1e-5)
    frozen={k:v.detach().clone() for k,v in model.state_dict().items()
            if k.startswith(('condition_network.','context_adapter.'))}
    common={k:v for k,v in model.state_dict().items()
            if not k.startswith(('backbone.image_self_attention.4.','backbone.image_self_attention.5.',
                                  'backbone.decoder_film.','context_adapter.'))}
    manifest=dict(variant=a.variant,seed=42,optimizer_steps=a.steps,batch_size=a.batch_size,
        splits=splits,adapter_kind=calibration['kind'],adapter_steps=calibration['steps'],
        shared_initialization_sha256=tensor_hash(common),condition_sha256=tensor_hash(model.condition_network.state_dict()),
        adapter_sha256=tensor_hash(model.context_adapter.state_dict()),transfer=transfer,
        rgb_parameters=sum(p.numel() for p in model.rgb_parameters()),
        objective='global token MSE + 1.0 response-weighted MSE; noise_floor=1e-6',
        response_difference='diagnostic only: (S(normal)-S(physical-zero)) - (T(normal)-T(physical-zero)); '
                            'student condition network recomputes gate/alpha in both interventions',
        reference_initialization=str(a.inputs/'student_rgb_initial.ckpt'))
    write_json(a.output_dir/'manifest.json',manifest)
    cfg.teacher_checkpoint=str((a.inputs/'pinned_teacher_ema.ckpt').resolve())
    cfg.teacher_config=None;cfg.output_dir=str(a.output_dir.resolve());cfg.training.max_steps=a.steps
    from omegaconf import OmegaConf
    OmegaConf.save(cfg,a.output_dir/'config.yaml')
    preproc=WrenchPreprocessor.from_policy(teacher).eval()
    cfg.wrench_preprocessing=preproc.get_config()
    OmegaConf.save(cfg,a.output_dir/'config.yaml')
    OmegaConf.save(teacher_cfg,a.output_dir.parent/'teacher_config.yaml')
    history=[];start=time.monotonic();best=float('inf')
    amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16)
    def tokens_for(rendered,adapter,input_grad=False):
        return teacher.obs_encoder.encode_image_context(rendered,input_grad=input_grad,context_adapter=adapter)
    def prepared(batch):
        images,wrench,_,_=teacher.prepare_context_inputs_with_future_contact(batch['obs'])
        return images,wrench
    def encode_train(images):
        return tokens_for(images,model.context_adapter,input_grad=True)
    encode=encode_train
    if not a.no_compile:
        encode=torch.compile(encode_train,fullgraph=True,dynamic=False)
    def save(step,metrics):
        payload=dict(cfg=cfg,state_dicts={
            'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
            'ema_model':{k:v.detach().cpu() for k,v in ema_model.state_dict().items()},
            'wrench_preprocessor':preproc.state_dict()},
            pickles={k:dill.dumps(v) for k,v in dict(global_step=step,epoch=step//len(loader),
                training_stage='rgb',stage_step=step).items()},teacher_cfg=teacher_cfg,
            optimizer=optim.state_dict(),metrics=metrics)
        torch.save(payload,a.output_dir/'latest.ckpt',pickle_module=dill)
    @torch.no_grad()
    def evaluate(step):
        nonlocal best
        ema_model.eval();totals={s:dict(count=0,token_sum=0.,weighted_sum=0.,weight_sum=0.,
            diff_sum=0.,rgb_sum=0.,rgb_square_sum=0.,pixels=0) for s in ['all','free','precontact','contact']}
        rows=[];visuals=[]
        for raw in vloader:
            batch=to_device(raw,dev);images,wrench=prepared(batch)
            # Zero is in physical torque units, then passed through the pinned normalizer.
            zero_wrench=teacher.normalizer[teacher.wrench_key].normalize(torch.zeros_like(wrench))
            with amp():
                rendered,masks,condition=ema_model(images,wrench,return_condition=True)
                z=tokens_for(rendered,ema_model.context_adapter)
                rendered_zero=ema_model(images,zero_wrench)
                z0=tokens_for(rendered_zero,ema_model.context_adapter)
            errors=(z.float()-batch['target'].float()).square().mean(-1)
            diff=((z.float()-z0.float())-(batch['target'].float()-batch['zero'].float())).square().mean((1,2))
            weights=(batch['response'].float()-1e-6).clamp_min(0)
            pixel=torch.cat([(rendered[k].float()-images[k].float()).flatten(1) for k in sorted(images)],dim=1)
            for i in range(len(z)):
                phase=int(batch['phase'][i]);row=dict(index=int(batch['index'][i]),phase=phase,
                    token_mse=float(errors[i].mean()),response_difference_mse=float(diff[i]),
                    rgb_residual_mae=float(pixel[i].abs().mean()),rgb_residual_rms=float(pixel[i].square().mean().sqrt()))
                rows.append(row)
                for scope in ['all',['free','precontact','contact'][phase]]:
                    t=totals[scope];t['count']+=1;t['token_sum']+=float(errors[i].mean())
                    t['weighted_sum']+=float((errors[i]*weights[i]).sum());t['weight_sum']+=float(weights[i].sum())
                    t['diff_sum']+=float(diff[i]);t['rgb_sum']+=float(pixel[i].abs().sum())
                    t['rgb_square_sum']+=float(pixel[i].square().sum());t['pixels']+=pixel.shape[1]
                if step in [0,a.steps] and sum(v['phase']==phase for v in visuals)<2:
                    visuals.append(dict(index=int(batch['index'][i]),phase=phase,
                        images={k:v[i].cpu() for k,v in images.items()},
                        rendered={k:v[i].cpu() for k,v in rendered.items()},
                        rendered_zero={k:v[i].cpu() for k,v in rendered_zero.items()},
                        masks={k:v[i].cpu() for k,v in masks.items()},alpha=condition['contact_alpha'][i].cpu()))
        metrics={}
        for scope,t in totals.items():
            n=t['count'];weighted=t['weighted_sum']/t['weight_sum'] if t['weight_sum'] else 0.
            metrics[scope]=dict(samples=n,token_mse=t['token_sum']/n,response_weighted_mse=weighted,
                objective=t['token_sum']/n+weighted,response_difference_mse=t['diff_sum']/n,
                rgb_residual_mae=t['rgb_sum']/t['pixels'],rgb_residual_rms=(t['rgb_square_sum']/t['pixels'])**.5)
        row=dict(kind='validation',step=step,metrics=metrics,elapsed_seconds=time.monotonic()-start)
        history.append(row);write_json(a.output_dir/'log.json',history)
        with (a.output_dir/'log.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        write_json(a.output_dir/'latest_validation.json',row)
        print(json.dumps(row),flush=True)
        if step in [0,a.steps]:
            torch.save(visuals,a.output_dir/f'renders_step{step}.pt')
            write_json(a.output_dir/f'per_sample_step{step}.json',rows)
        best=min(best,metrics['all']['objective'])
        return metrics
    # Real-image preflight runs before the first optimizer update.
    first=to_device(next(iter(vloader)),dev);images,wrench=prepared(first)
    checkimages={k:v[:1] for k,v in images.items()};checkwrench=wrench[:1]
    with torch.no_grad(),amp():
        shape=ema_model(checkimages,checkwrench)
        for k in images:assert shape[k].shape==checkimages[k].shape
    # The controlled condition output, not a label overridden by prediction,
    # closes alpha here; all existing region computation remains active.
    original_forward=model.condition_network.forward
    def closed(*args,**kw):
        masks,cond=original_forward(*args,**kw);cond['contact_alpha']=torch.zeros_like(cond['contact_alpha'])
        return masks,cond
    model.condition_network.forward=closed
    try:
        with torch.no_grad(),amp():
            unchanged=model(checkimages,checkwrench)
            assert all(torch.equal(unchanged[k],checkimages[k]) for k in unchanged)
    finally:model.condition_network.forward=original_forward
    write_json(a.output_dir/'preflight.json',dict(real_rgb_shape_correct=True,alpha_zero_rgb_exact=True,
        condition_and_adapter_frozen=True,teacher_frozen=True))
    metrics=evaluate(0)
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    iterator=iter(loader)
    for step in range(1,a.steps+1):
        try:raw=next(iterator)
        except StopIteration:iterator=iter(loader);raw=next(iterator)
        batch=to_device(raw,dev);images,wrench=prepared(batch)
        model.set_stage_mode(True);optim.zero_grad(set_to_none=True)
        with amp():
            rendered=model(images,wrench);z=encode(rendered)
            global_loss,response_loss,_,_=response_weighted_token_mse(z,batch['target'],batch['response'],noise_floor=1e-6)
            loss=global_loss+response_loss
        if not torch.isfinite(loss):raise RuntimeError('nonfinite distillation loss')
        loss.backward()
        diagnostics={}
        if step==1 or step%50==0:
            for name,selector in [('cnn','backbone.stem.'),('sa','backbone.image_self_attention.'),
                ('ca','backbone.cross_attention.'),('film','backbone.decoder_film.'),('lstm','wrench_encoder.'),
                ('spatial','spatial_modulations.')]:
                grads=[p.grad.detach().float() for k,p in model.named_parameters() if k.startswith(selector) and p.grad is not None]
                diagnostics[name+'_gradient_norm']=float(sum(g.square().sum() for g in grads).sqrt()) if grads else 0.
            if step==1:
                assert diagnostics['cnn_gradient_norm']>0 and diagnostics['lstm_gradient_norm']>0
                if a.variant.startswith('c_'):assert diagnostics['film_gradient_norm']>0
            row=dict(kind='train',step=step,loss=float(loss.detach()),token_mse=float(global_loss.detach()),
                response_mse=float(response_loss.detach()),diagnostics=diagnostics,lr=optim.param_groups[0]['lr'],
                elapsed_seconds=time.monotonic()-start)
            history.append(row)
            with (a.output_dir/'log.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        torch.nn.utils.clip_grad_norm_(model.rgb_parameters(),1.)
        optim.step();ema.step(model)
        if step%a.validate_every==0 or step==a.steps:
            for k,v in frozen.items():assert torch.equal(v,model.state_dict()[k]),k
            assert all(p.grad is None for p in teacher.parameters())
            metrics=evaluate(step);scheduler.step(metrics['all']['objective']);save(step,metrics)
    write_json(a.output_dir/'summary.json',dict(variant=a.variant,optimizer_steps=a.steps,final=metrics,
        best_validation_objective=best,manifest=manifest,elapsed_seconds=time.monotonic()-start,
        teacher_condition_adapter_frozen=True,peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20))

if __name__=='__main__':main()
