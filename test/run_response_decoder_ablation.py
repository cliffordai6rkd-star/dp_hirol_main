"""Matched ABCD route/gate ablation, reusing the existing immutable response cache."""
from pathlib import Path
import argparse
import copy
import json
import time

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from student_dual_path_common import load_inputs, datasets_from_reference, model_for, to_device, tensor_hash, write_json
from paired_response_full_common import file_hash, deterministic_mode
from new_teacher_response_common import FullPairedDataset
from run_scene_torque_response_predictor import phase_plan
from response_conditioned_rgb_common import metric_rows, grouped
from response_decoder_ablation import DiagnosticResponseDecoder, summarize_traces
from diffusion_policy.model.vision.render_net_attention import zero_wrench_reference


class Runtime:
    def __init__(self,a,config):
        self.a,self.config,self.device=a,config,torch.device('cuda:0')
        self.started=time.monotonic()
        assert file_hash(a.base/'inputs/pinned_teacher_ema.ckpt')==config.teacher_sha256
        assert file_hash(a.predictor/'best.pt')==config.predictor_sha256
        self.teacher,_,initial=load_inputs(a.base/'inputs',self.device)
        self.owner,_,_=model_for(initial,'c_film_mlp')
        source=a.base/'d/best.pt'
        assert file_hash(source)==config.decoder_initial_sha256
        saved=torch.load(source,map_location='cpu',weights_only=False)
        self.owner.load_state_dict(saved['ema_model'],strict=True)
        self.owner.context_adapter.load_state_dict(torch.load(a.base/'adapter/best_adapter.pt',map_location='cpu',weights_only=False)['state'],strict=True)
        if config.get('condition_sha256'):
            condition=a.base/'condition/best.pt'
            assert file_hash(condition)==config.condition_sha256
            self.owner.condition_network.load_state_dict(torch.load(condition,map_location='cpu',weights_only=False)['state'],strict=True)
        self.owner.to(self.device).eval().requires_grad_(False)
        deterministic_mode(self.owner)
        self.network=DiagnosticResponseDecoder(self.owner,config.epsilon,**dict(config.variants[a.variant])).to(self.device).eval()
        if config.get('strict_reference',True):
            reference=torch.load(a.cache/'decoder/initial.pt',map_location='cpu',weights_only=False)
            assert tensor_hash(self.network.state_dict())==tensor_hash(reference['model']), 'All arms use the original SAME initial tensor state'
        self.frozen={name:tensor_hash(m.state_dict()) for name,m in self.frozen_modules()}
        train,val,identities=datasets_from_reference(a.base/'inputs',a.old/'decoded_cuda')
        self.train,self.val=train,val
        manifest=json.loads((a.base/'paired_cache/manifest.json').read_text())
        assert manifest['teacher_checkpoint_sha256']==config.teacher_sha256
        prediction_manifest=json.loads((a.cache/'prediction_manifest.json').read_text())
        assert prediction_manifest['predictor_checkpoint_sha256']==config.predictor_sha256
        assert prediction_manifest['teacher_cache']['teacher_checkpoint_sha256']==config.teacher_sha256
        assert identities==manifest['splits']
        self.datasets={s:FullPairedDataset(ds,a.base/'paired_cache'/s) for s,ds in [('train',train),('validation',val)]}
        self.predictions={s:np.load(a.cache/(s+'_predicted.npy'),mmap_mode='r') for s in self.datasets}
        self.threshold=json.loads((a.base/'cnn/manifest.json').read_text())['threshold']
        self.plan=phase_plan(train.phase_labels,self.datasets['train'].episodes,config)
        if config.get('strict_reference',True):assert np.array_equal(self.plan,np.load(a.cache/'decoder/train_batches.npy'))
        self.parameters=[p for p in self.network.parameters() if p.requires_grad]
        self.manifest=dict(variant=a.variant,structure=OmegaConf.to_container(config.variants[a.variant]),config=OmegaConf.to_container(config),
            teacher_sha256=config.teacher_sha256,predictor_sha256=config.predictor_sha256,
            decoder_initial_sha256=config.decoder_initial_sha256,adapter_sha256=file_hash(a.base/'adapter/best_adapter.pt'),
            initial_state_sha256=tensor_hash(self.network.state_dict()),frozen=self.frozen,
            target='RAW predictor response for Contact/Precontact, ZERO for Free; never multiply target by alpha',
            zero_reference='Original image, identical deterministic DINO/Adapter/ContextNorm readout',
            ca_disabled_semantics='C removes CA response update but preserves its pure visual FFN; common decoder FiLM stays active in every arm',
            spatial_gates='Original spatial gates remain in Spatial Modulation and FiLM; Alpha appears only at RGB output in B/C/D',
            trainable_names=[n for n,p in self.network.named_parameters() if p.requires_grad],
            trainable_parameters=sum(p.numel() for p in self.parameters),
            cache=json.loads((a.cache/'prediction_manifest.json').read_text()),
            trace_scope='Fixed one midpoint per episode/phase, separately on train and validation; RMS is sqrt of mean energy, not mean per-image RMS')

    def frozen_modules(self):
        return [('teacher',self.teacher),('adapter',self.owner.context_adapter),('condition',self.owner.condition_network),('torque',self.owner.wrench_encoder)]

    def check_frozen(self):
        for name,module in self.frozen_modules():
            assert not any(p.requires_grad or p.grad is not None for p in module.parameters()),name
            assert tensor_hash(module.state_dict())==self.frozen[name],name

    def amp(self):return torch.autocast('cuda',dtype=torch.bfloat16)

    def inputs(self,batch):
        images,wrench,_,_=self.teacher.prepare_context_inputs_with_future_contact(batch['obs'])
        return images,wrench

    @torch.no_grad()
    def gates(self,images,wrench):
        with self.amp():
            masks,condition=self.owner.condition_network(images,wrench)
            spatial={k:self.owner._mask_gate(masks[k].flatten(0,1),images[k]).reshape_as(masks[k]).float() for k in images}
        return spatial,condition['contact_alpha'].float()

    def encode(self,images,grad=False):
        with self.amp():return self.teacher.obs_encoder.encode_image_context(images,context_adapter=self.owner.context_adapter,input_grad=grad).float()

    def predicted(self,batch,split):
        return torch.from_numpy(np.array(self.predictions[split][batch['index'].cpu().numpy()],copy=True)).to(self.device).float()

    def batch(self,split,ids):
        return to_device(next(iter(DataLoader(Subset(self.datasets[split],list(map(int,ids))),batch_size=len(ids)))),self.device)

    def gradient_diagnostic(self,model,output,step):
        diagnostic=copy.deepcopy(model).eval().requires_grad_(False)
        active=set(self.manifest['trainable_names'])
        for name,parameter in diagnostic.named_parameters():parameter.requires_grad_(name in active)
        batch=self.batch('train',self.plan[0]);pred=self.predicted(batch,'train');total=0.
        for start in range(0,len(pred),self.config.microbatch):
            sl=slice(start,start+self.config.microbatch)
            part={**batch,'obs':{k:v[sl] for k,v in batch['obs'].items()}}
            images,wrench=self.inputs(part);spatial,alpha=self.gates(images,wrench)
            z0=self.encode(images).detach()
            with self.amp():rendered=diagnostic(images,pred[sl],spatial,alpha)
            z=self.encode(rendered,True);target=pred[sl]*(batch['phase'][sl]!=0)[:,None,None]
            weights=torch.where(batch['phase'][sl]==0,self.config.free_weight,1.)
            loss=((z-z0-target).square().mean((1,2))*weights).sum()/len(pred)
            loss.backward();total+=float(loss.detach())
        norms=gradient_norms(diagnostic)
        assert norms['head']>0
        assert (norms['projection']>0)==self.network.use_projection
        assert (norms['ca_attention']>0)==self.network.use_ca_response
        write_json(output/f'probe_gradients_{step}.json',dict(loss=total,norms=norms,indices=self.plan[0].tolist(),
            weights='EMA/exact saved model supplied to diagnostic; fixed initial train batch, no optimizer step'))
        del diagnostic

    def rows(self,batch,pred,images,rendered,z,z0):
        return metric_rows(z-z0,pred,batch['target'].float()-batch['zero'].float(),images,rendered,
            batch['index'].cpu().tolist(),batch['episode'].cpu().tolist(),batch['phase'].cpu().tolist(),self.threshold)

    def probe_ids(self,split):
        ds=self.datasets[split]
        return [int(c[len(c)//2]) for ep in np.unique(ds.episodes) for phase in range(3)
            if len(c:=np.flatnonzero((ds.episodes==ep)&(ds.dataset.phase_labels==phase)))]

    @torch.no_grad()
    def trace(self,model,output,step):
        summary={}
        for split in self.datasets:
            records=[]
            ids=self.probe_ids(split)
            if self.a.smoke:ids=ids[:8]
            for start in range(0,len(ids),8):
                batch=self.batch(split,ids[start:start+8]);images,wrench=self.inputs(batch)
                pred=self.predicted(batch,split);spatial,alpha=self.gates(images,wrench)
                with self.amp():rendered,stats=model(images,pred,spatial,alpha,trace=True)
                stats['target_mse']=(pred*(batch['phase']!=0)[:,None,None]).square().mean((1,2))
                stats['student_response_mse']=(self.encode(rendered)-self.encode(images)).square().mean((1,2))
                cpu={key:value.detach().cpu().numpy() for key,value in stats.items()}
                for i in range(len(pred)):
                    records.append(dict(index=int(batch['index'][i]),episode=int(batch['episode'][i]),phase=int(batch['phase'][i]),
                        **{key:value[i].tolist() for key,value in cpu.items()}))
            write_json(output/f'{split}_trace_samples_{step}.json',records)
            summary[split]=summarize_traces(records)
        write_json(output/f'trace_{step}.json',summary)
        return summary

    @torch.no_grad()
    def evaluate(self,model,output,step,full_train=False):
        result,visuals={},[]
        for split,ds in self.datasets.items():
            ids=list(range(len(ds))) if split=='validation' or full_train else self.probe_ids(split)
            if self.a.smoke:ids=self.probe_ids(split)[:8]
            records=[]
            for raw in DataLoader(Subset(ds,ids),batch_size=self.config.validation_batch,num_workers=self.config.workers,pin_memory=True):
                batch=to_device(raw,self.device);images,wrench=self.inputs(batch);pred=self.predicted(batch,split)
                spatial,alpha=self.gates(images,wrench);z0=self.encode(images)
                with self.amp():rendered=model(images,pred,spatial,alpha)
                z=self.encode(rendered);records.extend(self.rows(batch,pred,images,rendered,z,z0))
                if split=='validation':
                    for i in range(len(pred)):
                        ep=int(batch['episode'][i])
                        if int(batch['phase'][i])!=2 or any(v['episode']==ep for v in visuals):continue
                        visuals.append(dict(index=int(batch['index'][i]),episode=ep,
                            original={k:v[i].cpu() for k,v in images.items()},rendered={k:v[i].cpu() for k,v in rendered.items()},
                            predicted=pred[i].cpu(),teacher=(batch['target'][i].float()-batch['zero'][i].float()).cpu(),student=(z[i]-z0[i]).cpu()))
            write_json(output/f'{split}_samples_{step}.json',records)
            result[split]=grouped(records,self.config)
        torch.save(visuals,output/f'visuals_{step}.pt')
        return result

    @torch.no_grad()
    def distribution(self,output):
        contact=np.flatnonzero(self.train.phase_labels==2)
        ids=contact[np.linspace(0,len(contact)-1,8).round().astype(int)]
        batch=self.batch('train',ids);images,wrench=self.inputs(batch);pred=self.predicted(batch,'train')
        _,alpha=self.gates(images,wrench)
        backbone=self.owner.backbone
        rows=[]
        with torch.backends.cudnn.flags(enabled=False),self.amp():old=self.owner.encode_wrench_history(wrench)
        for camera,key in enumerate(self.owner.image_keys):
            flat=images[key].flatten(0,1)
            with self.amp():
                feat=backbone.stem(flat)
                for encoder in backbone.encoders:feat=encoder(F.avg_pool2d(feat,2))
                gh,gw=feat.shape[-2:]
                tokens=feat.flatten(2).transpose(1,2)
                tokens=tokens+backbone._position(gh,gw,tokens)+self.owner.camera_embedding.weight[camera].reshape(1,1,-1).to(tokens)
                for block in backbone.image_self_attention[:2]:tokens=block(tokens)
            patch=pred.reshape(len(ids),2,2,196,256)[:,:,camera].reshape(2*len(ids),196,256)
            candidates={'original_lstm8':old,'raw_response196':patch,'alpha_response196':patch*alpha.reshape(-1,1,1)}
            for block_id,block in enumerate(backbone.cross_attention):
                for label,value in candidates.items():
                    with self.amp():
                        q=block.image_norm(tokens);k=block.wrench_norm(value)
                        update,weights=block.cross_attention(query=q,key=k,value=value,need_weights=True,average_attn_weights=True)
                    x=value.float();centered=x-x.mean(1,keepdim=True)
                    sv=torch.linalg.svdvals(centered).square()
                    rank=(sv.sum(-1).square()/sv.square().sum(-1).clamp_min(1e-12)).mean()
                    entropy=-(weights.float().clamp_min(1e-12).log()*weights.float()).sum(-1)
                    normalized=F.normalize(x,dim=-1);sim=normalized@normalized.transpose(1,2)
                    n=x.shape[1];off=(sim.sum((1,2))-sim.diagonal(dim1=1,dim2=2).sum(-1))/(n*(n-1))
                    rows.append(dict(camera=key,block=block_id,source=label,tokens=n,value_rms=float(x.square().mean().sqrt()),
                        key_normalized_rms=float(k.float().square().mean().sqrt()),
                        across_token_std_rms=float(centered.square().mean().sqrt()),centered_effective_rank=float(rank),
                        mean_token_cosine=float(off.mean()),normalized_attention_entropy=float((entropy/np.log(n)).mean()),
                        mean_max_attention=float(weights.max(-1).values.float().mean()),attention_update_rms=float(update.float().square().mean().sqrt())))
        write_json(output/'ca_distribution_initial.json',dict(rows=rows,indices=ids.tolist(),
            control='Identical image queries and frozen initial CA weights for each candidate; both CA blocks tested on same first-stage image queries. Explicit attention weights are diagnostic only; training uses need_weights=False.'))


def gradient_norms(network):
    groups=[('projection','response_projection.'),('cnn','owner.backbone.stem.'),
        ('ca_attention','owner.backbone.cross_attention.'),('film','owner.backbone.decoder_film.'),('head','owner.backbone.output.')]
    return {label:sum(float(p.grad.float().square().sum()) for n,p in network.named_parameters()
        if p.grad is not None and n.startswith(prefix) and (label!='ca_attention' or '.cross_attention.' in n.removeprefix(prefix)))**.5
        for label,prefix in groups}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['base','predictor','old','cache','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--variant',choices=list('ABCD'),required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).parent/'config/response_decoder_gate_ablation.yaml')
    p.add_argument('--reuse-baseline',action='store_true')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();config=OmegaConf.load(a.config)
    torch.set_num_threads(4);torch.manual_seed(config.seed);torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    output=a.output/(a.variant+('_smoke' if a.smoke else ''));output.mkdir(parents=True,exist_ok=False)
    OmegaConf.save(config,output/'config.yaml')
    r=Runtime(a,config);network=r.network
    np.save(output/'train_batches.npy',r.plan)
    r.manifest['sample_plan_sha256']=file_hash(output/'train_batches.npy')
    write_json(output/'structure.json',r.manifest)
    probe=r.batch('train',r.plan[0]);im,wr=r.inputs(probe);pr=r.predicted(probe,'train');sp,al=r.gates(im,wr)
    zero_im,zero_wr,_,_=r.teacher.prepare_context_inputs_with_future_contact(zero_wrench_reference(r.teacher,probe['obs']))
    assert all(torch.equal(zero_im[k],im[k]) for k in im)
    assert r.teacher.normalizer[r.teacher.wrench_key].unnormalize(zero_wr).abs().max()<1e-5
    with torch.no_grad(),r.amp():
        closed=network(im,pr,sp,torch.zeros_like(al));plain=network(im,pr,sp,al);traced,_=network(im,pr,sp,al,trace=True)
    assert all(torch.equal(closed[k],im[k]) for k in im)
    assert all(torch.equal(plain[k],traced[k]) for k in im),'Trace instrumentation must preserve actual RGB exactly'
    r.distribution(output)
    trace0=r.trace(network,output,0)
    r.gradient_diagnostic(network,output,0)
    first=r.evaluate(network,output,0)
    if a.reuse_baseline:
        assert a.variant=='A' and not a.smoke
        old=json.loads((a.cache/'decoder/summary.json').read_text())
        for key in ['teacher_sha256','predictor_sha256','decoder_initial_sha256','adapter_sha256']:
            assert old['manifest'][key]==r.manifest[key],key
        for key in ['batch_size','microbatch','validation_batch','validate_every','lr','weight_decay','free_weight','gradient_clip']:
            assert old['manifest']['config']['decoder'][key]==config[key],key
        assert old['steps']==config.steps
        def same_metrics(actual,expected):
            for phase in ['free','precontact','contact']:
                for key in ['realization_mse','realization_cosine','response_mse','end_to_end_mse','rgb_rms']:
                    x,y=actual['validation']['phases'][phase][key],expected['validation']['phases'][phase][key]
                    assert x==y or abs(x-y)<1e-7,(phase,key,x,y)
        same_metrics(first,old['initial'])
        saved=torch.load(a.cache/'decoder/best.pt',map_location='cpu',weights_only=False)
        network.load_state_dict(saved['ema_model'],strict=True)
        audit=r.evaluate(network,output,'reuse_audit');same_metrics(audit,old['final'])
        r.trace(network,output,config.steps)
        r.gradient_diagnostic(network,output,config.steps)
        r.check_frozen()
        with torch.no_grad(),r.amp():closed=network(im,pr,sp,torch.zeros_like(al))
        assert all(torch.equal(closed[k],im[k]) for k in im)
        # Reuse original full-train statistics, not the smaller audit probe.
        write_json(output/'log.json',json.loads((a.cache/'decoder/log.json').read_text()))
        write_json(output/'training.json',json.loads((a.cache/'decoder/training.json').read_text()))
        final=old['final'];final.pop('oracle_matched_contact',None)
        write_json(output/'summary.json',dict(manifest=r.manifest,initial=old['initial'],final=final,
            steps=old['steps'],best_step=old['best_step'],frozen_verified=True,alpha0_identity=True,
            baseline_reused=True,baseline_checkpoint_sha256=file_hash(a.cache/'decoder/best.pt'),
            reproduction='Exact initial tensors, batches, optimizer settings, and reproduced full validation metrics verified; old1000-step EMA/full-train metrics reused without further training',
            elapsed_seconds=time.monotonic()-r.started))
        print(json.dumps(dict(event='complete',variant='A',baseline_reused=True)),flush=True);return
    optimizer=torch.optim.AdamW(r.parameters,lr=config.lr,weight_decay=config.weight_decay)
    ema=copy.deepcopy(network).eval().requires_grad_(False)
    history=[dict(step=0,metrics=first,elapsed_seconds=time.monotonic()-r.started)]
    best=first['validation']['selection_contact'].get('realization_mse',float('inf'))
    torch.save(dict(model=network.state_dict(),ema_model=ema.state_dict(),step=0),output/'initial.pt')
    torch.save(dict(model=network.state_dict(),ema_model=ema.state_dict(),step=0),output/'best.pt')
    iterations=2 if a.smoke else config.steps
    iterator=iter(DataLoader(r.datasets['train'],batch_sampler=r.plan[:iterations].tolist(),num_workers=config.workers,pin_memory=True))
    training=[]
    for step in range(1,iterations+1):
        batch=to_device(next(iterator),r.device);pred=r.predicted(batch,'train');optimizer.zero_grad(set_to_none=True);total=0.
        for start in range(0,len(pred),config.microbatch):
            sl=slice(start,start+config.microbatch)
            part={**batch,'obs':{k:v[sl] for k,v in batch['obs'].items()}}
            images,wrench=r.inputs(part);spatial,alpha=r.gates(images,wrench);z0=r.encode(images).detach()
            with r.amp():rendered=network(images,pred[sl],spatial,alpha)
            z=r.encode(rendered,True)
            target=pred[sl]*(batch['phase'][sl]!=0)[:,None,None]
            weights=torch.where(batch['phase'][sl]==0,config.free_weight,1.)
            loss=((z-z0-target).square().mean((1,2))*weights).sum()/len(pred)
            assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
        gradients=gradient_norms(network)
        assert gradients['head']>0
        if network.use_projection:assert gradients['projection']>0
        else:assert gradients['projection']==0
        if network.use_ca_response:assert gradients['ca_attention']>0
        else:assert gradients['ca_attention']==0
        torch.nn.utils.clip_grad_norm_(r.parameters,config.gradient_clip);optimizer.step()
        with torch.no_grad():
            for dest,src in zip(ema.parameters(),network.parameters()):dest.lerp_(src,1-config.ema_decay)
        if step==1 or step%25==0 or a.smoke:
            training.append(dict(step=step,loss=total,gradient_norms=gradients,indices=batch['index'].cpu().tolist(),elapsed_seconds=time.monotonic()-r.started))
            write_json(output/'training.json',training);print(json.dumps(dict(event='train',variant=a.variant,**training[-1])),flush=True)
        if step%config.validate_every==0 or step==iterations:
            r.check_frozen();metrics=r.evaluate(ema,output,step)
            r.trace(ema,output,step)
            history.append(dict(step=step,metrics=metrics,elapsed_seconds=time.monotonic()-r.started))
            write_json(output/'log.json',history)
            state=dict(model=network.state_dict(),ema_model=ema.state_dict(),optimizer=optimizer.state_dict(),step=step,manifest=r.manifest)
            score=metrics['validation']['selection_contact'].get('realization_mse',total)
            if score<best:best=score;torch.save(state,output/'best.pt')
            torch.save(state,output/'latest.pt')
            print(json.dumps(dict(event='validation',variant=a.variant,step=step,contact=metrics['validation']['phases']['contact'],external=metrics['validation']['external_contact'])),flush=True)
    saved=torch.load(output/'best.pt',map_location='cpu',weights_only=False)
    # Primary comparison is the equal-budget endpoint, even if an arm peaked earlier.
    final=r.evaluate(ema,output,'endpoint',full_train=not a.smoke)
    r.trace(ema,output,'endpoint')
    r.gradient_diagnostic(ema,output,'endpoint')
    with torch.no_grad(),r.amp():closed=ema(im,pr,sp,torch.zeros_like(al))
    assert all(torch.equal(closed[k],im[k]) for k in im)
    r.check_frozen()
    write_json(output/'summary.json',dict(manifest=r.manifest,initial=first,final=final,steps=iterations,best_step=saved['step'],
        frozen_verified=True,alpha0_identity=True,zero_reference_original=True,
        best_checkpoint_sha256=file_hash(output/'best.pt'),endpoint_checkpoint_sha256=file_hash(output/'latest.pt'),
        evaluation_weights='EMA at the fixed training-budget endpoint; best checkpoint retained separately',
        elapsed_seconds=time.monotonic()-r.started))
    print(json.dumps(dict(event='complete',variant=a.variant,steps=iterations)),flush=True)


if __name__=='__main__':main()
