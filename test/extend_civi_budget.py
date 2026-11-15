"""One bounded continuation of the existing Mixed/BF16 trajectory, never formal100k.

Resume the ONLINE weights with their AdamW state; evaluate only EMA using the
preceding full-dataset FP32 protocol. All outputs/checkpoints are independent.
"""
from pathlib import Path
import argparse
import copy
import csv
import hashlib
import json
import os
import random
import shutil
import time
import traceback
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from run_civi_rgb_iteration import Experiment
from run_scene_torque_response_predictor import phase_plan
from student_dual_path_common import to_device,tensor_hash,write_json
from paired_response_full_common import file_hash
from civi_budget_extension_eval import Evaluator


def event(name,**values):
    print(json.dumps(dict(event=name,**values)),flush=True)


class Ledger:
    def __init__(self,root,out,requested):
        self.root,self.out,self.id=root,out,out.name
        self.csv=root/'experiments.csv';self.state_path=root/'state.json'
        self.rows=list(csv.DictReader(self.csv.open()));self.fields=list(self.rows[0])
        ids=[r['id'] for r in self.rows];assert len(ids)==len(set(ids))
        self.state=json.loads(self.state_path.read_text())
        assert int(self.state['exploration_budget'])==20000
        assert sum(int(r['budget_counted']) for r in self.rows)==self.state['exploration_rgb_updates']
        # Reconcile the two authorized temporary probes once, including pixels.
        for name,shared,pixel in [('final150k_local_updates_20',100,80),('final150k_mixed_updates_20',40,0)]:
            path=root/name/'summary.json';data=json.loads(path.read_text())
            assert data['total_optimizer_steps']==shared+pixel
            if name not in ids:
                row={k:'' for k in self.fields};row.update(id=name,status='complete',rgb_updates=str(shared),
                    budget_counted=str(shared+pixel),teacher_step='150000',seed='42',
                    reason=f'Temporary local probe; {shared} shared + {pixel} pixel updates. Count once; no trajectory checkpoint.')
                self.rows.append(row);ids.append(name)
        assert self.id not in ids,'Use a fresh continuation directory; never double-register a run'
        self.before=sum(int(r['budget_counted']) for r in self.rows)
        # Count any uncommitted journal entries from an interrupted extension.
        for row in self.rows:
            path=root/row['id']/'updates.jsonl'
            if row['id'].startswith('final150k_budget_extend') and path.exists():
                actual=sum(bool(line.strip()) for line in path.open())
                if actual>int(row['budget_counted']):
                    self.before+=actual-int(row['budget_counted'])
                    row['rgb_updates']=row['budget_counted']=str(actual)
                    row['status']='interrupted_reconciled'
        self.limit=min(requested,20000-self.before)
        assert self.limit>0,(self.before,'No exploration budget remains')
        if self.state.get('reserved_rgb_updates',0):
            for process in self.state.get('running_processes',[]):
                command=Path(f"/proc/{process.get('pid',-1)}/cmdline")
                assert not command.exists() or 'extend_civi_budget.py' not in command.read_bytes().decode(errors='replace'),'Another extension is still running'
            for row in self.rows:
                if row['status'] in ('reserved','running') and row['id'].startswith('final150k_budget_extend'):
                    row['status']='interrupted_reconciled'
        row={k:'' for k in self.fields};row.update(id=self.id,status='reserved',seed='42',teacher_step='150000',
            rgb_updates='0',budget_counted='0',source_head='2443e63e047aff9ba178bc11e0834e5f8205fe60',
            hypothesis='Bounded continuation of fixed Mixed/BF16 trajectory; primary true Teacher transfer',
            data='All train windows; balanced2F2P4C; complete internal FP32 validation; retained5/26 evaluation only',
            checkpoint=str(out/'latest.pt'),promotion='False',reason='User-authorized bounded extension, at most total6000 and global20000')
        self.rows.append(row);self.row=row;self.used=0
        self.journal=(out/'updates.jsonl').open('a',buffering=1)
        self.commit('reserved')
        write_json(out/'budget_audit.json',dict(inclusive_ceiling=20000,spent_before=self.before,
            remaining_before=20000-self.before,reserved=self.limit,original_ledger=16402,
            reconciled_probes=dict(local_shared=100,local_pixel=80,mixed_shared=40),
            shared_before=sum(int(r['rgb_updates'] or 0) for r in self.rows if int(r['budget_counted'] or 0)>0),
            note='Historical reference1000 and dependency calibration excluded exactly as original ledger; evaluation0'))

    def update(self,step,loss):
        assert self.used<self.limit
        self.used+=1
        self.journal.write(json.dumps(dict(update=self.used,model_step=step,loss=loss))+'\n')

    def commit(self,status,reason=''):
        self.row.update(status=status,rgb_updates=str(self.used),budget_counted=str(self.used))
        if reason:self.row['reason']=reason
        with self.csv.open('w') as stream:
            w=csv.DictWriter(stream,fieldnames=self.fields);w.writeheader();w.writerows(self.rows)
        shared=sum(int(r['rgb_updates'] or 0) for r in self.rows if int(r['budget_counted'] or 0)>0)
        total=sum(int(r['budget_counted'] or 0) for r in self.rows)
        assert total==self.before+self.used<=20000
        self.state.update(updated_at=datetime.now().astimezone().isoformat(),
            stage='bounded_extension_'+status,exploration_rgb_updates=total,exploration_shared_rgb_updates=shared,
            exploration_pixel_rgb_updates=total-shared,reserved_rgb_updates=self.limit-self.used if status in ('reserved','running') else 0,
            running_processes=[dict(id=self.id,pid=os.getpid(),output=str(self.out))] if status in ('reserved','running') else [],
            next_action='Only finish this authorized bounded continuation; no loss/structure search or automatic100k' if status in ('reserved','running') else 'Review bounded-extension report; no automatic further training',
            bounded_extension=dict(id=self.id,used=self.used,reserved_limit=self.limit,total_step_cap=6000,reason=reason))
        if status not in ('reserved','running'):
            self.state['completed_experiments']=list(dict.fromkeys(self.state.get('completed_experiments',[])+[self.id]))
        write_json(self.state_path,self.state)


class Extension(Experiment):
    def __init__(self,args,config,ledger,saved):
        self.ledger=ledger;self.requested_saved=saved
        if 'extension_state' in saved:
            # The legacy constructor reads these BF16 history containers, while
            # this probe's authoritative histories are fp32_history/training_bf16.
            # Older extension checkpoints omitted the unused containers.
            for name in ('log.json','training.json'):
                path=args.resume.parent/name
                if not path.exists():write_json(path,[])
        super().__init__(args,config)
        assert tensor_hash(self.network.state_dict())==tensor_hash(saved['model'])
        assert tensor_hash(self.ema.state_dict())==tensor_hash(saved['ema_model'])
        assert [id(p) for g in self.optimizer.param_groups for p in g['params']]==[id(p) for p in self.params]
        assert all(int(s['step'])==self.start_step for s in self.optimizer.state.values())
        assert self.start_step>=3000 and self.start_step<config.steps<=6000
        assert config.fp32_training_variants==[] and config.rgb_weight==0 and config.free_weight==1
        assert config.batch_size==8 and config.microbatch==4 and config.lr==2e-4
        assert config.weight_decay==1e-4 and config.gradient_clip==1 and config.ema_decay==.99
        assert torch.equal(torch.get_rng_state(),saved['torch_rng'])
        assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),saved['cuda_rng']))
        if 'python_rng' in saved:random.setstate(saved['python_rng'])
        if 'numpy_rng' in saved:np.random.set_state(saved['numpy_rng'])
        rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
        self.evaluator=Evaluator(self,args.root,args.reference)
        torch.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1])
        # Only parameters outside this exact allowlist may ever be optimized.
        self.frozen_versions=[(n,p,p._version) for n,m in self.r.frozen_modules() for p in m.parameters()]
        self.training=[];self.fp32_history=[];self.best_p=float('inf');self.best_t=float('inf')
        self.no_new=0;self.teacher_worse=0;self.free_worse=0;self.pre_worse=0
        self.step=self.start_step
        self.cache_paths=[args.cache/'prediction_manifest.json',args.base/'paired_cache/manifest.json',
                          args.base/'inputs/reference_train_cache.json',args.base/'inputs/reference_validation_cache.json']
        self.cache_hashes={str(p):file_hash(p) for p in self.cache_paths}
        self.array_stats={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in
                          [args.cache/'train_predicted.npy',args.cache/'validation_predicted.npy']}
        self.manifest['bounded_extension']=dict(start_step=self.start_step,total_step_cap=config.steps,
            new_update_cap=ledger.limit,budget_before=ledger.before,global_budget=20000,
            authorization_budget_before=args.authorization_before,authorization_new_update_cap=args.authorization_cap,
            optimizer='Original AdamW state paired with ONLINE model; EMA restored separately',scheduler='NONE in source/checkpoint',
            training_precision='Original BF16 autocast Renderer/gates/visual E; FP32 parameters and original FP16 predictor cache cast to float',
            evaluation_precision='Separate FP32 deterministic context; restores all RNG/backend flags afterwards',
            evaluation='EMA ONLY; fixed B8 original batch membership, Teacher patchRMS>.001, equal episode normalized ratios',
            stop='3 consecutive checks neither P nor T sets new best; OR Teacher increases2 consecutive intervals; OR Free/precontactTeacher deteriorates3 consecutive intervals',
            references=['identity/original RGB','experiment step0','continuation step3000'],retained='5/26 never used in stopping/selection',
            legacy_rng='Source persists Torch/CUDA. NumPy sampler is local seeded generator with verified full plan; no random augmentation or Python random training use.',
            source_checkpoint_sha256=file_hash(args.resume),sampler_prefix_verified=True,first_new_batch=self.plan[self.start_step].tolist(),
            cache_hashes=self.cache_hashes,source_original_training_sha256=file_hash(Path(__file__).with_name('run_civi_rgb_iteration.py')))
        write_json(self.out/'manifest.json',self.manifest)
        # Keep the native resume constructor usable without mixing its legacy
        # BF16 selection history into the separate FP32 stopping history.
        write_json(self.out/'log.json',[])
        write_json(self.out/'training.json',[])
        write_json(self.out/'resume_verification.json',dict(source_step=self.start_step,
            online_state_sha256=tensor_hash(self.network.state_dict()),ema_state_sha256=tensor_hash(self.ema.state_dict()),
            adamw_step_min=min(int(s['step']) for s in self.optimizer.state.values()),
            adamw_step_max=max(int(s['step']) for s in self.optimizer.state.values()),
            optimizer_group={k:v for k,v in self.optimizer.param_groups[0].items() if k!='params'},
            torch_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            cuda_rng_sha256=[hashlib.sha256(s.cpu().numpy().tobytes()).hexdigest() for s in torch.cuda.get_rng_state_all()],
            online_optimizer_pair_verified=True,ema_separate=True,scheduler=None,
            executed_prefix_verified=True,first_new_step=self.start_step+1,first_new_batch=self.plan[self.start_step].tolist()))
        for path in (Path(__file__),Path(__file__).with_name('civi_budget_extension_eval.py')):
            shutil.copy2(path,self.out/'source_snapshot'/path.name)
        self.check_integrity()

    def check_integrity(self):
        self.r.check_frozen()
        for name,p,version in self.frozen_versions:
            assert p._version==version and p.grad is None,(name,'Frozen tensor changed')
        assert all(file_hash(Path(p))==h for p,h in self.cache_hashes.items()),'Cache manifest mismatch'
        assert all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==s for p,s in self.array_stats.items()),'Prediction cache changed'
        assert file_hash(self.a.resume)==self.manifest['bounded_extension']['source_checkpoint_sha256']

    def save(self,step):
        result=super().save(step)
        result.update(python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            sampler_next_step=step+1,sample_plan_sha256=file_hash(self.out/'train_batches.npy'),
            extension_state=dict(best_predictor=self.best_p,best_teacher=self.best_t,no_new=self.no_new,
                teacher_worse=self.teacher_worse,free_worse=self.free_worse,pre_worse=self.pre_worse,
                history=self.fp32_history,global_budget_used=self.ledger.before+self.ledger.used),
            training_backend=dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,mha_fastpath=torch.backends.mha.get_fastpath_enabled(),
                flash_sdp=torch.backends.cuda.flash_sdp_enabled(),mem_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
                math_sdp=torch.backends.cuda.math_sdp_enabled()))
        return result

    def checkpoint(self,step,numbered=True):
        value=self.save(step)
        torch.save(value,self.out/'latest.pt')
        if numbered:torch.save(value,self.out/f'step_{step:06d}.pt')

    def evaluate_point(self,train=False):
        self.check_integrity();metrics=self.evaluator.evaluate(self.step,train)
        c=metrics['validation_internal']['contact'];p=c['predictor_episode_macro_error_zero_ratio'];t=c['teacher_episode_macro_error_zero_ratio']
        f=metrics['validation_internal']['free']['response_energy'];pre=metrics['validation_internal']['precontact']['teacher_episode_macro_error_zero_ratio']
        previous=self.fp32_history[-1] if self.fp32_history else None
        new_p=p<self.best_p;new_t=t<self.best_t
        self.no_new=0 if new_p or new_t else self.no_new+1
        if previous:
            self.teacher_worse=self.teacher_worse+1 if t>previous['teacher'] else 0
            self.free_worse=self.free_worse+1 if f>previous['free_actual_mse'] else 0
            self.pre_worse=self.pre_worse+1 if pre>previous['pre_teacher'] else 0
        self.best_p=min(self.best_p,p);self.best_t=min(self.best_t,t)
        self.fp32_history.append(dict(step=self.step,predictor=p,teacher=t,free_actual_mse=f,pre_teacher=pre,
            no_new=self.no_new,teacher_worse=self.teacher_worse,free_worse=self.free_worse,pre_worse=self.pre_worse))
        write_json(self.out/'fp32_history.json',self.fp32_history)
        self.evaluator.write_precision();self.checkpoint(self.step)
        if new_t and p<=self.fp32_history[0]['predictor']:
            torch.save(self.save(self.step),self.out/'best_teacher.pt')
        self.ledger.commit('running')
        event('validation',**self.fp32_history[-1])
        if self.no_new>=3:return '3 checks without new Predictor or Teacher best'
        if self.teacher_worse>=2:return 'True Teacher error worsened2 consecutive500-step intervals'
        if self.free_worse>=3:return 'Free actual response MSE worsened3 consecutive500-step intervals'
        if self.pre_worse>=3:return 'Precontact Teacher error worsened3 consecutive500-step intervals'
        return None

    def train_step(self,batch,step):
        # Same operation order, target cache, reduction, autocast scopes, AdamW,
        # clipping and EMA lerp as run_civi_rgb_iteration.py:298-352.
        r,c=self.r,self.c;self.evaluator.mode='bf16_training'
        assert all(file_hash(Path(p))==h for p,h in self.cache_hashes.items()),'Cache manifest mismatch'
        assert all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==s for p,s in self.array_stats.items()),'Prediction cache changed'
        pred=r.predicted(batch,'train');self.optimizer.zero_grad(set_to_none=True)
        total=0.;components=dict(response=0.,free=0.,rgb=0.)
        for start in range(0,len(pred),c.microbatch):
            sl=slice(start,start+c.microbatch);part={**batch,'obs':{k:v[sl] for k,v in batch['obs'].items()}}
            im,wr=r.inputs(part);sp,al=r.gates(im,wr)
            z0=r.encode(im).detach()
            with r.amp():rendered=self.network(im,pred[sl],sp,al)
            z=r.encode(rendered,True);free=batch['phase'][sl]==0
            target=pred[sl]*(~free)[:,None,None]
            mse=(z-z0-target).square().mean((1,2))
            rgb=torch.cat([(rendered[k]-im[k]).flatten(1) for k in im],1).square().mean(-1)
            weights=torch.where(free,c.free_weight,1.)
            loss=(weights*mse+c.rgb_weight*rgb).sum()/len(pred)
            assert torch.isfinite(loss),'Nonfinite loss'
            loss.backward();total+=float(loss.detach())
            components['response']+=float(mse[~free].sum().detach())/len(pred)
            components['free']+=float(mse[free].sum().detach())/len(pred)
            components['rgb']+=float(rgb.sum().detach())/len(pred)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in self.params),'Nonfinite gradient'
        norm=torch.nn.utils.clip_grad_norm_(self.params,c.gradient_clip)
        assert torch.isfinite(norm)
        self.optimizer.step();self.step=step
        self.ledger.update(step,total)
        with torch.no_grad():
            for dest,src in zip(self.ema.parameters(),self.network.parameters()):dest.lerp_(src,1-c.ema_decay)
        assert all(torch.isfinite(p).all() for p in self.params),'Nonfinite updated parameter'
        for name,p,version in self.frozen_versions:
            assert p._version==version and p.grad is None,(name,'Frozen tensor changed')
        if step%25==0:
            row=dict(step=step,loss=total,components=components,gradient_norm=float(norm),elapsed=time.monotonic()-self.started)
            self.training.append(row);write_json(self.out/'training_bf16.json',self.training);event('train',**row)

    def run(self):
        # Zero-update reference is mandatory before the FIRST optimizer step.
        if 'extension_state' in self.requested_saved:
            state=self.requested_saved['extension_state']
            self.best_p=state['best_predictor'];self.best_t=state['best_teacher'];self.no_new=state['no_new']
            self.teacher_worse=state['teacher_worse'];self.free_worse=state['free_worse'];self.pre_worse=state['pre_worse']
            self.fp32_history=state['history']
            reproduced=self.evaluator.evaluate(self.step,train_contact=True)['validation_internal']['contact']
            for target in ('predictor','teacher'):
                assert abs(reproduced[target+'_episode_macro_error_zero_ratio']-self.fp32_history[-1][target])<1e-6
            self.checkpoint(self.step)
        else:
            self.evaluate_point(train=True)
        assert self.ledger.used==0
        event('resume_verified',step=self.step,first_new_step=self.step+1,budget=self.ledger.limit,
              online_hash=tensor_hash(self.network.state_dict()),ema_hash=tensor_hash(self.ema.state_dict()))
        reason=None
        loader=DataLoader(self.r.datasets['train'],batch_sampler=self.plan[self.start_step:self.c.steps].tolist(),
                          num_workers=self.c.workers,pin_memory=True)
        for step,raw in enumerate(loader,start=self.start_step+1):
            assert self.ledger.used<self.ledger.limit
            batch=to_device(raw,self.r.device)
            assert batch['index'].cpu().tolist()==self.plan[step-1].tolist()
            assert self.r.train.phase_labels[self.plan[step-1]].tolist()==[0,0,1,1,2,2,2,2]
            self.train_step(batch,step)
            if step%500==0 or step==self.c.steps:
                reason=self.evaluate_point(train=step==self.c.steps)
                if reason:break
        if self.step<self.c.steps:
            # Same node, but now add the required complete train-Contact endpoint.
            self.evaluator.evaluate(self.step,train_contact=True)
        reason=reason or ('Total6000 reached' if self.step==6000 else 'Remaining exploration budget reached')
        self.check_integrity();self.checkpoint(self.step);self.evaluator.write_precision()
        self.ledger.commit('complete',reason)
        write_json(self.out/'summary.json',dict(start_step=self.start_step,final_step=self.step,new_optimizer_updates=self.ledger.used,
            stop_reason=reason,global_budget_used=self.ledger.before+self.ledger.used,global_budget_remaining=20000-self.ledger.before-self.ledger.used,
            fp32_history=self.fp32_history,original_checkpoint_unchanged=True,frozen_verified=True,cache_verified=True,
            elapsed_seconds=time.monotonic()-self.started,formal_training=False,acceptance_relaxed=False,
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated()))
        event('complete',step=self.step,new_updates=self.ledger.used,reason=reason)


def main():
    p=argparse.ArgumentParser(description=__doc__);root=Path('data/diagnostics/civi_student_iteration');match=root/'final150k_match'
    for key,default in dict(root=root,base=match/'matched',predictor=match/'predictor',cache=match/'predictor_cache',
        old=Path('data/diagnostics/2026.11.15/student_dual_path_ablation'),dependencies=match/'dependencies.json',
        reference=root/'final150k_convergence_fp32',resume=root/'final150k_baseline_s42_3000/latest.pt',
        output=root/'final150k_budget_extend_s42_6000').items():p.add_argument('--'+key,type=Path,default=default)
    p.add_argument('--max-total-step',type=int,default=6000)
    p.add_argument('--max-new-updates',type=int,default=3000)
    a=p.parse_args();assert 3000<a.max_total_step<=6000 and 0<a.max_new_updates<=3000
    assert torch.cuda.device_count()==1,'Expose one confirmed idle GPU'
    saved=torch.load(a.resume,map_location='cpu',weights_only=False)
    if 'extension_state' in saved:
        assert not (a.resume.parent/'summary.json').exists(),'Completed/stopped continuation cannot be resumed as an interruption'
        state=saved['extension_state']
        assert state['no_new']<3 and state['teacher_worse']<2 and state['free_worse']<3 and state['pre_worse']<3,'Saved stopping rule already triggered'
        failure=a.resume.parent/'failure.json'
        if failure.exists():
            error=json.loads(failure.read_text())['error'].lower()
            assert not any(x in error for x in ('nonfinite','frozen','cache')),'Resolve the recorded anomaly before any further authorization to train'
    a.output.mkdir(parents=True,exist_ok=False)
    ledger=Ledger(a.root,a.output,a.max_new_updates)
    experiment=None
    try:
        assert saved['step']>=3000 and not any(k in saved for k in ('scheduler','lr_scheduler'))
        original=saved.get('manifest',{}).get('bounded_extension',{})
        a.authorization_before=original.get('authorization_budget_before',original.get('budget_before',ledger.before))
        a.authorization_cap=original.get('authorization_new_update_cap',original.get('new_update_cap',ledger.limit))
        # Lost/uncheckpointed steps remain real optimizer updates. A restart may
        # neither reset this authorization's3000 cap nor the global20000 cap.
        ledger.limit=min(ledger.limit,a.authorization_cap-(ledger.before-a.authorization_before))
        assert ledger.limit>0,'This continuation authorization has no unspent updates'
        config=OmegaConf.create(saved['manifest']['iteration_config'])
        config.steps=min(a.max_total_step,saved['step']+ledger.limit)
        ledger.limit=config.steps-saved['step'];assert ledger.limit>0
        ledger.commit('reserved')
        a.config=a.resume.parent/'config.yaml'
        a.variant='baseline';a.scope='full';a.smoke=False;a.final=False;a.acceptance=None;a.oracle=None;a.start_from=None
        a.rgb_init=Path(saved['manifest']['rgb_initialization']['checkpoint'])
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        backend=saved.get('training_backend',{})
        if backend:
            torch.use_deterministic_algorithms(backend['deterministic_algorithms'])
            torch.backends.cudnn.deterministic=backend['cudnn_deterministic']
            torch.backends.mha.set_fastpath_enabled(backend['mha_fastpath'])
            torch.backends.cuda.enable_flash_sdp(backend['flash_sdp'])
            torch.backends.cuda.enable_mem_efficient_sdp(backend['mem_efficient_sdp'])
            torch.backends.cuda.enable_math_sdp(backend.get('math_sdp',True))
        experiment=Extension(a,config,ledger,saved)
        experiment.run()
    except BaseException as error:
        if experiment is not None:
            # Preserve the reached online/EMA/optimizer state, including on an anomaly.
            try:experiment.checkpoint(experiment.step);experiment.evaluator.write_precision()
            except Exception:pass
        write_json(a.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),updates=ledger.used))
        ledger.commit('failed',repr(error));raise


if __name__=='__main__':main()
