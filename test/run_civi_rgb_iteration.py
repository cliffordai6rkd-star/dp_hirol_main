"""Controlled shared-RGB fitting with immutable paired targets and acceptance metrics."""
from pathlib import Path
import argparse
import copy
import json
import time
import shutil

import numpy as np
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from run_response_decoder_ablation import Runtime, gradient_norms
from response_decoder_ablation import DiagnosticResponseDecoder
from response_conditioned_rgb_common import metric_rows, aggregate
from audit_civi_rgb_iteration import panels
from student_dual_path_common import to_device, write_json, tensor_hash
from paired_response_full_common import file_hash
from run_scene_torque_response_predictor import phase_plan
from diffusion_policy.model.vision.render_net_attention import zero_wrench_reference
from civi_patch_response_renderer import PatchResponseRenderer
from civi_semantic_patch_renderer import SemanticPatchRenderer


def fixed_rows(r,batch,pred,images,rendered,z,z0,stats):
    dS=z-z0;dT=batch['target'].float()-batch['zero'].float()
    rows=metric_rows(dS,pred,dT,images,rendered,batch['index'].cpu().tolist(),
        batch['episode'].cpu().tolist(),batch['phase'].cpu().tolist(),r.threshold)
    active=dT.square().mean(-1).sqrt()>r.threshold
    cos=F.cosine_similarity(dS,pred,dim=-1,eps=1e-6)
    for i,row in enumerate(rows):
        mask=active[i]
        row.update(predictor_active=int(mask.sum()),realization_cosine_sum=float(cos[i][mask].sum()),
            s_predictor_ss=float(dS[i][mask].double().square().sum()),
            predictor_ss=float(pred[i][mask].double().square().sum()),
            absolute_token_mse=float((z[i]-batch['target'][i].float()).square().mean()),
            zero_token_mse=float((z0[i]-batch['zero'][i].float()).square().mean()),
            **{k:float(stats[k][i]) for k in ['allowed_channels','total_channels','hard_clipped_channels','range_clipped_channels']})
    return rows


def summarize(rows,config):
    def agg(selected):
        result=aggregate(selected)
        if not selected:return result
        allowed=sum(x['allowed_channels'] for x in selected)
        result.update(absolute_token_mse=float(np.mean([x['absolute_token_mse'] for x in selected])),
            zero_token_mse=float(np.mean([x['zero_token_mse'] for x in selected])),
            hard_clip_fraction=sum(x['hard_clipped_channels'] for x in selected)/max(1,allowed),
            range_clip_fraction=sum(x['range_clipped_channels'] for x in selected)/max(1,allowed))
        return result
    episodes={str(ep):{name:agg([x for x in rows if x['episode']==ep and x['phase']==phase])
        for phase,name in enumerate(['free','precontact','contact'])} for ep in sorted({x['episode'] for x in rows})}
    def macro(eps,phase):
        values=[episodes[str(ep)][phase] for ep in eps if str(ep) in episodes and episodes[str(ep)][phase]['samples']]
        if not values:return dict(episodes=0)
        keys=set.intersection(*(set(x) for x in values)) - {'samples'}
        return dict(episodes=len(values),**{k:float(np.mean([x[k] for x in values])) for k in keys if all(x[k] is not None for x in values)})
    return dict(phases={name:agg([x for x in rows if x['phase']==phase]) for phase,name in enumerate(['free','precontact','contact'])},
        episodes=episodes,macro={name:macro([int(x) for x in episodes],name) for name in ['free','precontact','contact']},
        selection_contact=agg([x for x in rows if x['phase']==2 and x['episode'] in config.selection_episodes]),
        selection_macro_contact=macro(config.selection_episodes,'contact'),
        retained_contact=agg([x for x in rows if x['phase']==2 and x['episode'] in config.external_episodes]))


def gradient_groups(network):
    out=gradient_norms(network)
    if isinstance(network,PatchResponseRenderer):
        out['head']=sum(float(p.grad.float().square().sum()) for p in network.head.parameters() if p.grad is not None)**.5
        out['response_projection']=sum(float(p.grad.float().square().sum()) for p in network.response.parameters() if p.grad is not None)**.5
        out['scene']=sum(float(p.grad.float().square().sum()) for p in network.scene.parameters() if p.grad is not None)**.5
    for name,match in [('cnn_all',lambda n:'.stem.' in n or '.encoders.' in n),
                       ('decoder',lambda n:'.decoders.' in n),('spatial',lambda n:'.spatial_modulations.' in n)]:
        out[name]=sum(float(p.grad.float().square().sum()) for n,p in network.named_parameters() if p.grad is not None and match(n))**.5
    out['rgb_lstm']=sum(float(p.grad.float().square().sum()) for p in network.owner.wrench_encoder.parameters() if p.grad is not None)**.5
    return out


class Experiment:
    def __init__(self,a,c):
        self.a,self.c,self.out=a,c,a.output
        legacy=OmegaConf.load(Path(__file__).parent/'config/response_decoder_gate_ablation.yaml')
        if a.dependencies:
            dependencies=json.loads(a.dependencies.read_text())
            for key in ['teacher_sha256','predictor_sha256','decoder_initial_sha256','condition_sha256']:legacy[key]=dependencies[key]
            legacy.strict_reference=False
            assert file_hash(a.base/'adapter/best_adapter.pt')==dependencies['adapter_sha256']
        a.variant_original=a.variant;requested_smoke=a.smoke;a.variant='A';a.smoke=False
        self.r=Runtime(a,legacy);a.variant=a.variant_original;a.smoke=requested_smoke
        r=self.r
        self.panel=panels(r)
        initial=copy.deepcopy(r.network.state_dict())
        options=dict(c.variants[a.variant])
        if options.get('architecture') in ['shared_patch_rgb','shared_semantic_patch_rgb']:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                cls=SemanticPatchRenderer if options['architecture']=='shared_semantic_patch_rgb' else PatchResponseRenderer
                self.network=cls(r.owner,c.epsilon).to(r.device).eval()
        else:
            self.network=DiagnosticResponseDecoder(r.owner,c.epsilon,**options).to(r.device).eval()
            self.network.load_state_dict(initial,strict=True)
        if a.rgb_init:
            assert a.dependencies and not isinstance(self.network,PatchResponseRenderer)
            saved=torch.load(a.rgb_init,map_location='cpu',weights_only=False)
            source=saved['ema_model'];state=self.network.state_dict()
            assert set(source)==set(state)
            # A new teacher keeps the newly matched frozen snapshots. Transfer
            # RGB parameters only; no old Adapter/Condition target may leak in.
            protected=('owner.context_adapter.','owner.condition_network.')
            for name in state:
                if not name.startswith(protected):state[name]=source[name]
            self.network.load_state_dict(state,strict=True)
            r.check_frozen()
        r.network=self.network
        self.visual_arrays={s:np.load(a.base/'paired_cache'/s/'visual.npy',mmap_mode='r') for s in ['train','validation']}
        self.params=[p for p in self.network.parameters() if p.requires_grad]
        torch.manual_seed(c.seed);torch.cuda.manual_seed_all(c.seed)
        self.ema=copy.deepcopy(self.network).requires_grad_(False).eval()
        self.optimizer=torch.optim.AdamW(self.params,lr=c.lr,weight_decay=c.weight_decay)
        self.started=time.monotonic();self.log=[];self.training=[];self.best=float('inf')
        self.start_step=0;self.stale=0
        self.initial_train_ids=self.panel['train_contact']+self.panel['train_free']
        if a.scope=='fit':
            rng=np.random.default_rng(c.seed)
            pools=[self.panel['train_free'],self.panel['train_contact']]
            queues=[[],[]];plan=[]
            for _ in range(c.steps):
                row=[]
                for p,count in [(0,2),(1,6)]:
                    for _ in range(count):
                        if not queues[p]:queues[p]=rng.permutation(pools[p]).tolist()
                        row.append(queues[p].pop())
                plan.append(row)
            self.plan=np.asarray(plan)
        else:
            options=OmegaConf.create(dict(seed=c.seed,steps=c.steps,phase_mix=[2,2,4]))
            self.plan=phase_plan(r.train.phase_labels,r.datasets['train'].episodes,options)
        self.manifest=dict(**r.manifest,iteration_variant=a.variant,iteration_config=OmegaConf.to_container(c),
            scope=a.scope,readout='E=frozen DINO+projection/embeddings+calibrated Adapter+ContextNorm; identical online encode batch shape',
            active_mask='fixed Teacher RMS > threshold; shared by ALL cosines and RMS ratios',threshold=r.threshold,
            phases='offline labels only for sampling/loss/reporting, never generator input',
            retained_episodes='5/26 repeatedly viewed and used for teacher selection; never candidate checkpoint selection',
            parameterization=dict(c.variants[a.variant]),initial_tensor_state_sha256=tensor_hash(self.network.state_dict()))
        self.manifest['trainable_names']=[n for n,p in self.network.named_parameters() if p.requires_grad]
        self.manifest['trainable_parameters']=sum(p.numel() for p in self.params)
        self.manifest['visual_training_precision']='fp32' if a.variant in c.fp32_training_variants else 'bf16'
        self.manifest['visual_evaluation_precision']='bf16 fixed for all candidates'
        if a.rgb_init:self.manifest['rgb_initialization']=dict(checkpoint=str(a.rgb_init),sha256=file_hash(a.rgb_init),weights='ema_model',protected='Retain newly matched Adapter and Condition; transfer RGB only')
        self.manifest['response_training_target']='teacher' if a.variant in c.teacher_supervised_variants else 'predictor'
        if a.final:
            accepted=json.loads(a.acceptance.read_text())
            assert accepted['initial_tensor_state_sha256']==self.manifest['initial_tensor_state_sha256']
        self.oracle={}
        if a.oracle:
            assert a.variant=='oracle_init' and a.scope=='fit'
            source=torch.load(a.oracle/'visuals.pt',map_location='cpu',weights_only=False)
            expected=set(self.panel['train_contact'])
            assert set(x['index'] for x in source)==expected
            assert all(x['episode'] not in c.selection_episodes and x['episode'] not in c.external_episodes for x in source)
            self.oracle={x['index']:x for x in source}
            self.manifest['train_only_oracle']=dict(path=str(a.oracle),sha256=file_hash(a.oracle/'visuals.pt'),
                indices=sorted(expected),warm_updates=c.oracle_initialization_steps,validation_oracle_used=False)
        if a.start_from:
            assert not a.resume and a.scope=='full' and a.variant=='baseline'
            saved=torch.load(a.start_from,map_location='cpu',weights_only=False)
            for key in ['teacher_sha256','predictor_sha256','adapter_sha256','decoder_initial_sha256']:
                assert saved['manifest'][key]==self.manifest[key]
            self.network.load_state_dict(saved['model'],strict=True);self.ema.load_state_dict(saved['ema_model'],strict=True)
            self.optimizer.load_state_dict(saved['optimizer'])
            previous=int(saved['step'])
            options=OmegaConf.create(dict(seed=c.seed,steps=previous+c.steps,phase_mix=[2,2,4]))
            full=phase_plan(r.train.phase_labels,r.datasets['train'].episodes,options)
            assert np.array_equal(full[:previous],np.load(a.cache/'decoder/train_batches.npy')[:previous])
            self.plan=full[previous:]
            self.manifest['continued_from']=dict(checkpoint=str(a.start_from),sha256=file_hash(a.start_from),previous_updates=previous,
                optimizer_restored=True,new_batch_plan_starts_after_previous_updates=True)
            r.check_frozen()
        if a.resume:
            saved=torch.load(a.resume,map_location='cpu',weights_only=False)
            assert saved['manifest']['initial_tensor_state_sha256']==self.manifest['initial_tensor_state_sha256']
            assert saved['manifest']['iteration_variant']==a.variant
            assert saved['manifest']['iteration_config']['seed']==c.seed
            for key in ['teacher_sha256','predictor_sha256','adapter_sha256']:
                assert saved['manifest'][key]==self.manifest[key]
            assert saved['manifest']['frozen']==self.manifest['frozen']
            self.network.load_state_dict(saved['model'],strict=True);self.ema.load_state_dict(saved['ema_model'],strict=True)
            self.optimizer.load_state_dict(saved['optimizer']);self.start_step=saved['step'];self.best=saved['best'];self.stale=saved.get('stale',0)
            assert self.start_step<=c.steps
            old_plan=np.load(a.resume.parent/'train_batches.npy')
            assert np.array_equal(self.plan[:self.start_step],old_plan[:self.start_step]),'Resume must continue the same batches without replay'
            self.log=[row for row in json.loads((a.resume.parent/'log.json').read_text()) if row['step']<=self.start_step]
            self.training=[row for row in json.loads((a.resume.parent/'training.json').read_text()) if row['step']<=self.start_step]
            for name in ['initial_full.json','initial_full_ema.json','best.pt']:
                previous=a.resume.parent/name
                if previous.exists():shutil.copy2(previous,self.out/name)
            shutil.copy2(a.resume,self.out/'latest.pt')
            torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
            self.manifest['resumed_from']=dict(checkpoint=str(a.resume),sha256=file_hash(a.resume),
                previous_updates=self.start_step,optimizer_restored=True,rng_restored=True,
                sampler_prefix_verified=True,initial_full_reference_preserved=True)
            r.check_frozen()
        np.save(self.out/'train_batches.npy',self.plan)
        self.manifest['sample_plan_sha256']=file_hash(self.out/'train_batches.npy')
        source=self.out/'source_snapshot';source.mkdir()
        paths=[Path(__file__),Path(__file__).with_name('response_decoder_ablation.py'),
            Path(__file__).with_name('run_response_decoder_ablation.py'),Path(__file__).with_name('response_conditioned_rgb_common.py'),
            Path(__file__).with_name('civi_patch_response_renderer.py'),Path(__file__).with_name('civi_semantic_patch_renderer.py'),a.config]
        hashes={}
        for path in paths:
            name=path.name;shutil.copy2(path,source/name);hashes[str(path)]=file_hash(path)
        self.manifest['executed_source_sha256']=hashes
        write_json(source/'sha256.json',hashes)
        write_json(self.out/'manifest.json',self.manifest);write_json(self.out/'panels.json',self.panel)
        OmegaConf.save(c,self.out/'config.yaml')

    @torch.no_grad()
    def evaluate(self,model,label,step,full=False,controls=False):
        r,c=self.r,self.c;result={};visuals=[]
        ids_by_split=dict(train=self.initial_train_ids,validation=list(range(len(r.datasets['validation']))) if full else self.panel['internal_validation'])
        if self.a.scope=='full':
            ids_by_split['train']=r.probe_ids('train')
        for split,ids in ids_by_split.items():
            rows=[]
            loader=DataLoader(Subset(r.datasets[split],ids),batch_size=c.validation_batch,num_workers=c.workers,pin_memory=True)
            for raw in loader:
                batch=to_device(raw,r.device);im,wr=r.inputs(batch);pred=r.predicted(batch,split);sp,al=r.gates(im,wr)
                z0=r.encode(im)
                self.scene_features(model,batch,split)
                with r.amp():rendered,stats=model(im,pred,sp,al,trace=True)
                z=r.encode(rendered)
                rows.extend(fixed_rows(r,batch,pred,im,rendered,z,z0,stats))
                if full and split=='validation':
                    for j in range(len(pred)):
                        ep=int(batch['episode'][j])
                        if int(batch['phase'][j])!=2 or any(x['episode']==ep for x in visuals):continue
                        visuals.append(dict(index=int(batch['index'][j]),episode=ep,original={k:v[j].cpu() for k,v in im.items()},
                            rendered={k:v[j].cpu() for k,v in rendered.items()},student=(z[j]-z0[j]).cpu(),predicted=pred[j].cpu(),
                            teacher=(batch['target'][j]-batch['zero'][j]).float().cpu()))
            write_json(self.out/f'{split}_{label}_{step}_samples.json',rows)
            result[split]=summarize(rows,c)
        if full:torch.save(visuals,self.out/f'visuals_{label}_{step}.pt')
        return result

    @torch.no_grad()
    def controls(self,model,label):
        r=self.r;c=self.c
        ids=self.panel['internal_validation'];records={k:[] for k in ['correct','shuffled','zero','physical_zero']}
        for start in range(0,len(ids),c.validation_batch):
            b=r.batch('validation',ids[start:start+c.validation_batch]);im,wr=r.inputs(b);sp,al=r.gates(im,wr)
            pred=r.predicted(b,'validation');z0=r.encode(im)
            self.scene_features(model,b,'validation')
            # Deterministic across-episode rotation of conditions, no target-driven selection.
            shifted=ids[len(ids)//2:]+ids[:len(ids)//2]
            other=r.batch('validation',shifted[start:start+c.validation_batch]);shuffle=r.predicted(other,'validation')
            zi,zw,_,_=r.teacher.prepare_context_inputs_with_future_contact(zero_wrench_reference(r.teacher,b['obs']))
            zsp,za=r.gates(zi,zw)
            for name,p,s,a in [('correct',pred,sp,al),('shuffled',shuffle,sp,al),('zero',torch.zeros_like(pred),sp,al),('physical_zero',torch.zeros_like(pred),zsp,za)]:
                with r.amp():out,stats=model(im,p,s,a,trace=True)
                z=r.encode(out);records[name].extend(fixed_rows(r,b,pred,im,out,z,z0,stats))
            with r.amp():closed=model(im,pred,sp,torch.zeros_like(al))
            assert all(torch.equal(closed[k],im[k]) for k in im)
        result={k:summarize(v,c) for k,v in records.items()}
        write_json(self.out/f'controls_{label}.json',result)
        return result

    def save(self,step):
        r=self.r
        return dict(step=step,best=self.best,stale=self.stale,model=self.network.state_dict(),ema_model=self.ema.state_dict(),optimizer=self.optimizer.state_dict(),
            manifest=self.manifest,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())

    def scene_features(self,model,batch,split,sl=None):
        if isinstance(model,SemanticPatchRenderer):
            ids=batch['index'].cpu().numpy()
            if sl is not None:ids=ids[sl]
            array=self.visual_arrays[split]
            model.visual_tokens=torch.from_numpy(np.array(array[ids],copy=True)).to(self.r.device)

    def run(self):
        r,c=self.r,self.c
        if self.start_step==0:
            self.initial_full=self.evaluate(self.network,'initial_reference',0,full=True) if not self.a.smoke else None
            if not self.a.smoke and self.a.start_from:
                self.initial_full_ema=self.evaluate(self.ema,'initial_reference_ema',0,full=True)
            else:self.initial_full_ema=self.initial_full
            write_json(self.out/'initial_full.json',self.initial_full)
            write_json(self.out/'initial_full_ema.json',self.initial_full_ema)
            for label,model in [('online',self.network),('ema',self.ema)]:
                metrics=self.evaluate(model,label,0,full=self.a.final)
                self.log.append(dict(step=0,weights=label,metrics=metrics))
                if label=='ema':self.best=metrics['validation']['selection_macro_contact']['realization_mse']
            torch.save(self.save(0),self.out/'initial.pt')
            torch.save(self.save(0),self.out/'best.pt')
            write_json(self.out/'log.json',self.log)
        loader=DataLoader(r.datasets['train'],batch_sampler=self.plan[self.start_step:].tolist(),num_workers=c.workers,pin_memory=True)
        step=self.start_step
        for step,raw in enumerate(loader,start=self.start_step+1):
            batch=to_device(raw,r.device);pred=r.predicted(batch,'train');self.optimizer.zero_grad(set_to_none=True)
            total=0.;components=dict(response=0.,free=0.,rgb=0.)
            for start in range(0,len(pred),c.microbatch):
                sl=slice(start,start+c.microbatch);part={**batch,'obs':{k:v[sl] for k,v in batch['obs'].items()}}
                im,wr=r.inputs(part);sp,al=r.gates(im,wr)
                if self.a.variant in c.fp32_training_variants:
                    def train_encode(images,grad=False):
                        with torch.autocast('cuda',enabled=False):
                            return r.teacher.obs_encoder.encode_image_context(images,context_adapter=r.owner.context_adapter,input_grad=grad).float()
                else:train_encode=r.encode
                oracle_stage=bool(self.oracle) and step<=c.oracle_initialization_steps
                z0=train_encode(im).detach() if not oracle_stage else None
                self.scene_features(self.network,batch,'train',sl)
                with r.amp():rendered=self.network(im,pred[sl],sp,al)
                if oracle_stage:
                    expected={k:[] for k in im}
                    for j,index in enumerate(batch['index'][sl].cpu().tolist()):
                        if int(batch['phase'][sl][j])==0:
                            for k in im:expected[k].append(im[k][j])
                        else:
                            record=self.oracle[index]
                            for k in im:
                                original=record['original'][k].to(r.device)
                                assert torch.equal(original,im[k][j]),'Oracle RGB must match the exact training image'
                                expected[k].append(record['rendered'][k].to(r.device))
                    per=torch.cat([(rendered[k]-torch.stack(expected[k])).flatten(1) for k in im],1).square().mean(-1)
                    loss=per.sum()/len(pred)
                    assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
                    components.setdefault('oracle_pixel',0.);components['oracle_pixel']+=float(loss.detach())
                    continue
                z=train_encode(rendered,True);free=batch['phase'][sl]==0
                target=pred[sl]*(~free)[:,None,None]
                if self.a.variant in c.teacher_supervised_variants:
                    target=(batch['target'][sl].float()-batch['zero'][sl].float())*(~free)[:,None,None]
                mse=(z-z0-target).square().mean((1,2))
                rgb=torch.cat([(rendered[k]-im[k]).flatten(1) for k in im],1).square().mean(-1)
                weights=torch.where(free,c.free_weight,1.)
                loss=(weights*mse+c.rgb_weight*rgb).sum()/len(pred)
                assert torch.isfinite(loss)
                loss.backward();total+=float(loss.detach())
                components['response']+=float(mse[~free].sum().detach())/len(pred)
                components['free']+=float(mse[free].sum().detach())/len(pred)
                components['rgb']+=float(rgb.sum().detach())/len(pred)
            norms=gradient_groups(self.network)
            assert norms['head']>0
            torch.nn.utils.clip_grad_norm_(self.params,c.gradient_clip);self.optimizer.step()
            with torch.no_grad():
                for dest,src in zip(self.ema.parameters(),self.network.parameters()):dest.lerp_(src,1-c.ema_decay)
            if step==1 or step%25==0:
                self.training.append(dict(step=step,loss=total,components=components,gradient_norms=norms,elapsed_seconds=time.monotonic()-self.started))
                write_json(self.out/'training.json',self.training)
                print(json.dumps(dict(event='train',variant=self.a.variant,scope=self.a.scope,**self.training[-1])),flush=True)
            if step%c.validate_every==0 or step==c.steps:
                for label,model in [('online',self.network),('ema',self.ema)]:
                    metrics=self.evaluate(model,label,step,full=self.a.final)
                    self.log.append(dict(step=step,weights=label,metrics=metrics,elapsed_seconds=time.monotonic()-self.started))
                    if label=='ema':
                        score=metrics['validation']['selection_macro_contact']['realization_mse']
                        if score<self.best-(c.early_stopping_min_delta if self.a.final else 0.):
                            self.best=score;self.stale=0;torch.save(self.save(step),self.out/'best.pt')
                        else:self.stale+=1
                    print(json.dumps(dict(event='validation',variant=self.a.variant,step=step,weights=label,
                        train_contact=metrics['train']['phases']['contact'],internal_contact=metrics['validation']['selection_macro_contact'])),flush=True)
                write_json(self.out/'log.json',self.log);torch.save(self.save(step),self.out/'latest.pt')
                r.check_frozen()
                if self.a.final and self.stale>=c.early_stopping_patience:break
        final={}
        for label,model in [('online',self.network),('ema',self.ema)]:
            final[label]=self.evaluate(model,label,'endpoint',full=not self.a.smoke)
        controls=self.controls(self.ema,'ema_endpoint')
        r.check_frozen()
        write_json(self.out/'summary.json',dict(manifest=self.manifest,steps=step,new_rgb_updates=step-self.start_step,
            maximum_steps=c.steps,early_stopped=step<c.steps,formal_training=self.a.final,
            initial={x['weights']:x['metrics'] for x in self.log if x['step']==0},final=final,controls=controls,
            initial_full=json.loads((self.out/'initial_full.json').read_text()) if (self.out/'initial_full.json').exists() else None,
            initial_full_ema=json.loads((self.out/'initial_full_ema.json').read_text()) if (self.out/'initial_full_ema.json').exists() else None,
            frozen_verified=True,alpha0_identity=True,elapsed_seconds=time.monotonic()-self.started,
            checkpoint_sha256=file_hash(self.out/'latest.pt'),peak_memory_bytes=torch.cuda.max_memory_allocated()))
        print(json.dumps(dict(event='complete',variant=self.a.variant,steps=c.steps,output=str(self.out))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['base','predictor','old','cache','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).parent/'config/civi_rgb_iteration.yaml')
    p.add_argument('--variant',default='baseline');p.add_argument('--scope',choices=['fit','full'],default='fit')
    p.add_argument('--steps',type=int);p.add_argument('--seed',type=int);p.add_argument('--resume',type=Path)
    p.add_argument('--start-from',type=Path,help='Continue immutable historical full-data run with optimizer, without repeating its updates')
    p.add_argument('--oracle',type=Path,help='Training-only Oracle reference; only oracle_init variant')
    p.add_argument('--dependencies',type=Path,help='Explicit matched dependency hashes for a separate teacher experiment')
    p.add_argument('--rgb-init',type=Path,help='Warm-start previous shared RGB EMA while retaining NEW matched frozen Adapter/Condition')
    p.add_argument('--final',action='store_true',help='Only after fixed gates and two seeds pass; max100k with early stopping')
    p.add_argument('--acceptance',type=Path,help='Machine-readable passing two-seed acceptance for these exact dependencies/variant')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();c=OmegaConf.load(a.config)
    if a.steps is not None:c.steps=a.steps
    if a.seed is not None:c.seed=a.seed
    if a.final:
        assert a.scope=='full' and a.acceptance and a.dependencies
        acceptance=json.loads(a.acceptance.read_text());deps=json.loads(a.dependencies.read_text())
        assert acceptance['promote'] and acceptance['same_dependencies'] and acceptance['same_variant'] and len(acceptance['seeds'])>=2
        assert acceptance['same_budget'] and acceptance['same_initialization']
        assert acceptance['variant']==a.variant and acceptance['dependencies']=={k:deps[k] for k in ['teacher_sha256','predictor_sha256','adapter_sha256','condition_sha256','decoder_initial_sha256']}
        assert deps['teacher_step']==150000,'Final distillation pins the explicitly requested final150k teacher'
        assert c.steps<=c.final_maximum_steps
    else:assert c.steps<=5000
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    Experiment(a,c).run()


if __name__=='__main__':main()
