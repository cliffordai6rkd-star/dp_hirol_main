"""Bounded new CIVI objective: render RGB and distil the unchanged frozen DiT.

Only a new explicitly bounded run is supported. Dependencies and RGB support
remain unchanged; no Teacher/Predictor/Adapter/Condition optimization occurs.
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

import dill
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from run_response_decoder_ablation import Runtime
from response_decoder_ablation import DiagnosticResponseDecoder
from student_dual_path_common import model_for, tensor_hash, to_device, write_json
from diffusion_policy.model.vision.render_net import build_render_net
from paired_response_full_common import file_hash
from civi_frozen_dit_common import Evaluator


def event(name, **values):
    print(json.dumps(dict(event=name, **values)), flush=True)


class Ledger:
    def __init__(self, args):
        self.a, self.id = args, args.output.name
        self.path = args.root/'experiments.csv'
        self.state_path = args.root/'state.json'
        self.rows = list(csv.DictReader(self.path.open())); self.fields = list(self.rows[0])
        self.state = json.loads(self.state_path.read_text())
        assert len({r['id'] for r in self.rows}) == len(self.rows)
        assert sum(int(r['budget_counted']) for r in self.rows) == self.state['exploration_rgb_updates']
        for proc in self.state.get('running_processes', []):
            path = Path(f"/proc/{proc.get('pid', -1)}/cmdline")
            assert not path.exists() or b'train_civi_frozen_dit.py' not in path.read_bytes(), 'A DiT run is already active'
        existing = next((r for r in self.rows if r['id'] == self.id), None)
        if args.resume:
            assert existing is not None
            assert existing['status'] not in ('complete', 'stopped'), 'Completed/stopped run cannot resume automatically'
            self.row = existing
            metadata = json.loads((args.output/'budget_authorization.json').read_text())
            assert args.steps == metadata['maximum_optimizer_updates']
            assert args.additional_budget == metadata['additional_authorized_budget']
            self.used = sum(bool(line.strip()) for line in (args.output/'updates.jsonl').read_text().splitlines())
            old = int(self.row['budget_counted']); assert self.used >= old
            self.state['exploration_rgb_updates'] += self.used-old
            self.row['rgb_updates'] = self.row['budget_counted'] = str(self.used)
            self.before = self.state['exploration_rgb_updates']-self.used
        else:
            assert existing is None and not self.state.get('reserved_rgb_updates', 0)
            self.before, self.used = self.state['exploration_rgb_updates'], 0
            old_ceiling = self.state['exploration_budget']
            self.state['exploration_budget'] += args.additional_budget
            assert args.steps <= self.state['exploration_budget']-self.before, 'Explicit budget is insufficient'
            self.row = {key:'' for key in self.fields}
            self.row.update(id=self.id, status='reserved', seed=str(args.seed), teacher_step='150000',
                            rgb_updates='0', budget_counted='0',
                            source_head='2443e63e047aff9ba178bc11e0834e5f8205fe60',
                            hypothesis='New frozen DiT output objective; identical CIVI renderer/support/dependencies',
                            data='All56 train episodes,2F2P4C; full internal validation; retained5/26 report only',
                            checkpoint=str(args.output/'latest.pt'), promotion='False',
                            reason='Explicit bounded DiT distillation; fresh optimizer; no automatic extension')
            self.rows.append(self.row)
            write_json(args.output/'budget_authorization.json', dict(
                prior_ceiling=old_ceiling, prior_used=self.before,
                additional_authorized_budget=args.additional_budget,
                resulting_ceiling=self.state['exploration_budget'],
                maximum_optimizer_updates=args.steps,
                initialization=args.initialize,
                note='Additional budget must be explicitly selected by user; prior checkpoints are immutable.'))
        assert self.used < args.steps
        self.journal = (args.output/'updates.jsonl').open('a', buffering=1)
        self.commit('reserved')

    def update(self, step, loss):
        assert self.used < self.a.steps
        self.used += 1
        self.journal.write(json.dumps(dict(update=self.used, model_step=step, loss=loss))+'\n')

    def commit(self, status, reason=''):
        self.row.update(status=status, rgb_updates=str(self.used), budget_counted=str(self.used))
        if reason:
            self.row['reason'] = reason
        total = sum(int(r['budget_counted']) for r in self.rows)
        assert total == self.before+self.used <= self.state['exploration_budget']
        with self.path.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields); writer.writeheader(); writer.writerows(self.rows)
        active = status in ('running', 'reserved')
        shared = sum(int(r['rgb_updates'] or 0) for r in self.rows if int(r['budget_counted']) > 0)
        self.state.update(updated_at=datetime.now().astimezone().isoformat(),
                          stage='frozen_dit_'+status, exploration_rgb_updates=total,
                          exploration_shared_rgb_updates=shared, exploration_pixel_rgb_updates=total-shared,
                          reserved_rgb_updates=self.a.steps-self.used if active else 0,
                          running_processes=[dict(id=self.id, pid=os.getpid(), output=str(self.a.output))] if active else [],
                          next_action='Finish only this bounded frozen DiT run' if active else 'Review DiT report/renderings; no automatic further training',
                          frozen_dit_run=dict(id=self.id, used=self.used, cap=self.a.steps, reason=reason))
        if not active:
            self.state['completed_experiments'] = list(dict.fromkeys(self.state.get('completed_experiments', [])+[self.id]))
        write_json(self.state_path, self.state)


class Experiment:
    def __init__(self, args, ledger):
        self.a, self.out, self.ledger = args, args.output, ledger
        source = torch.load(args.source, map_location='cpu', weights_only=False)
        deps = json.loads(args.dependencies.read_text())
        config = OmegaConf.load(Path(__file__).parent/'config/response_decoder_gate_ablation.yaml')
        for key in ('teacher_sha256', 'predictor_sha256', 'decoder_initial_sha256', 'condition_sha256'):
            config[key] = deps[key]
        config.strict_reference = False
        config.steps, config.batch_size, config.microbatch, config.seed = args.steps, 8, 4, args.seed
        args.variant, args.smoke = 'A', False
        self.r = Runtime(args, config)
        r = self.r
        for key in ('teacher_sha256', 'predictor_sha256', 'adapter_sha256'):
            assert source['manifest'][key] == (deps[key] if key in deps else r.manifest[key])
        self.network = r.network
        if args.initialize == 'ema6000':
            assert source['step'] == 6000
            self.network.load_state_dict(source['ema_model'], strict=True)
        else:
            # New random RGB parameters; keep ALL already-matched frozen tensors.
            initial = torch.load(args.base/'inputs/student_rgb_initial.ckpt', map_location='cpu', pickle_module=dill, weights_only=False)
            _, resolved, _ = model_for(initial, 'c_film_mlp')
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(args.seed)
                owner = build_render_net(7, dict(resolved.render_net)).to(r.device).eval()
                self.network = DiagnosticResponseDecoder(owner, .1, input_alpha=True,
                    film_alpha=True, projection=True, ca_response=True, parameterization='hard').to(r.device).eval()
            trainable = {name for name,p in self.network.named_parameters() if p.requires_grad}
            state = self.network.state_dict()
            for key in set(state)-trainable:
                state[key] = source['ema_model'][key]
            self.network.load_state_dict(state, strict=True)
            r.owner, r.network = self.network.owner, self.network
        r.check_frozen()
        self.network.eval()
        self.params = [p for p in self.network.parameters() if p.requires_grad]
        assert sum(p.numel() for p in self.params) == 1834932
        self.ema = copy.deepcopy(self.network).eval().requires_grad_(False)
        self.initial = copy.deepcopy(self.ema).eval().requires_grad_(False)
        self.optimizer = torch.optim.AdamW(self.params, lr=2e-4, weight_decay=1e-4)
        self.step, self.history, self.training = 0, [], []
        self.best, self.no_new, self.free_worse = float('inf'), 0, 0
        self.frozen_versions = [(name,p,p._version) for name,module in r.frozen_modules() for p in module.parameters()]
        self.cache_paths = [args.cache/'prediction_manifest.json', args.base/'paired_cache/manifest.json',
                            args.base/'inputs/reference_train_cache.json', args.base/'inputs/reference_validation_cache.json']
        self.cache_hashes = {str(path):file_hash(path) for path in self.cache_paths}
        self.array_stats = {str(args.cache/(s+'_predicted.npy')):
            ((args.cache/(s+'_predicted.npy')).stat().st_size, (args.cache/(s+'_predicted.npy')).stat().st_mtime_ns)
            for s in ('train', 'validation')}
        self.source_hash = file_hash(args.source)
        self.manifest = dict(**r.manifest, teacher_step=150000, objective='frozen_dit_output_mse',
            initialization=args.initialize, parent_checkpoint=str(args.source), parent_step=int(source['step']),
            parent_checkpoint_sha256=self.source_hash, new_optimizer=True, scheduler=None,
            loss='Mean over batch8 of full Teacher DiT output MSE, including Free; no alpha on target; no token/demo auxiliary loss',
            gradient_path='Renderer -> frozen DINO/projection/Adapter/context_norm -> frozen DiT; no force-fusion shortcut on student branch',
            teacher_target='Native matched Teacher context computed online on original RGB/raw torque; same noisy action/timestep as student',
            training_precision='Original BF16 autocast strategy; FP32 parameters/AdamW. No uniform FP32 training.',
            evaluation_precision='FP32 throughout Renderer/visual/context/DiT; fixed original encode B8; tail pads excluded',
            validation_probes=args.eval_probes, validation_seed=args.validation_seed,
            batch_size=8, microbatch=4, phase_mix=[2,2,4], lr=2e-4, weight_decay=1e-4, gradient_clip=1., ema_decay=.99,
            maximum_optimizer_updates=args.steps, selection='Internal38/44/53/60 Contact DiT error / identity-error, equal episode macro; retained5/26 excluded',
            stopping='Authorized update cap; nonfinite/frozen/cache anomaly;3 checks no DiT best;3 consecutive Free-MSE increases',
            rgb_constraint='Unchanged spatial gate, all three alpha placements, hard epsilon=.1*support and RGB[0,1]',
            acceptance='Original gates retained; this run alone cannot establish downstream policy effectiveness')
        self.evaluator = Evaluator(self)
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        if args.resume:
            saved = torch.load(args.resume, map_location='cpu', weights_only=False)
            assert saved['manifest']['objective'] == self.manifest['objective']
            assert saved['no_new'] < 3 and saved['free_worse'] < 3
            self.network.load_state_dict(saved['model'], strict=True)
            self.ema.load_state_dict(saved['ema_model'], strict=True)
            self.optimizer.load_state_dict(saved['optimizer'])
            initial_state = torch.load(self.out/'initial.pt', map_location='cpu', weights_only=False)['ema_model']
            self.initial.load_state_dict(initial_state, strict=True)
            self.step, self.history, self.training = saved['step'], saved['history'], saved['training']
            self.best, self.no_new, self.free_worse = saved['best'], saved['no_new'], saved['free_worse']
            torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
            random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
            assert np.array_equal(np.load(self.out/'train_batches.npy'), r.plan)
            assert all(int(value['step']) == self.step for value in self.optimizer.state.values())
        else:
            np.save(self.out/'train_batches.npy', r.plan)
        self.check_integrity(full=True)
        write_json(self.out/'manifest.json', self.manifest)
        source_dir = self.out/'source_snapshot'; source_dir.mkdir(exist_ok=True)
        for path in (Path(__file__), Path(__file__).with_name('civi_frozen_dit_common.py'),
                     Path(__file__).with_name('response_decoder_ablation.py'),
                     Path(__file__).with_name('response_conditioned_rgb_common.py')):
            shutil.copy2(path, source_dir/path.name)
        self.started = time.monotonic()

    def check_integrity(self, full=False):
        for name,p,version in self.frozen_versions:
            assert p._version == version and p.grad is None, (name, 'Frozen parameter changed')
        assert all(file_hash(Path(path)) == value for path,value in self.cache_hashes.items()), 'Cache manifest changed'
        assert all((Path(path).stat().st_size, Path(path).stat().st_mtime_ns) == value for path,value in self.array_stats.items()), 'Prediction cache changed'
        if full:
            self.r.check_frozen()
            assert file_hash(self.a.source) == self.source_hash, 'Original checkpoint changed'

    def save(self, label='latest.pt'):
        state = dict(step=self.step, model=self.network.state_dict(), ema_model=self.ema.state_dict(),
                     optimizer=self.optimizer.state_dict(), manifest=self.manifest,
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                     python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                     sampler_next_step=self.step+1, history=self.history, training=self.training,
                     best=self.best, no_new=self.no_new, free_worse=self.free_worse,
                     actual_optimizer_updates=self.ledger.used)
        torch.save(state, self.out/label)

    def objective(self, batch):
        r = self.r
        pred = r.predicted(batch, 'train')
        total = 0.
        for start in (0,4):
            part = {key:({k:v[start:start+4] for k,v in value.items()} if key=='obs' else value[start:start+4])
                    for key,value in batch.items()}
            images,wrench = r.inputs(part); spatial,alpha = r.gates(images,wrench)
            with torch.no_grad(), r.amp():
                teacher_context = r.teacher._encode_context(part['obs'], apply_curriculum_mask=False).detach()
            with r.amp():
                rendered = self.network(images,pred[start:start+4],spatial,alpha)
            student_context = r.encode(rendered, True)
            action = r.teacher.normalizer['action'].normalize(part['action'].float())
            noise = torch.randn_like(action)
            times = torch.randint(0,r.teacher.noise_scheduler.config.num_train_timesteps,
                                  (len(action),),device=r.device).long()
            noisy = r.teacher.noise_scheduler.add_noise(action,noise,times)
            with r.amp():
                with torch.no_grad(): target = r.teacher.model(noisy,times,teacher_context)
                prediction = r.teacher.model(noisy,times,student_context)
                loss = (prediction.float()-target.float()).square().flatten(1).mean(1).sum()/8
            assert torch.isfinite(loss), 'Nonfinite loss'
            loss.backward(); total += float(loss.detach())
        return total

    def evaluate_point(self, train=False):
        self.check_integrity(full=True)
        metrics = self.evaluator.evaluate(self.step,train_contact=train)
        current = metrics['validation_internal']['contact']['dit_error_identity_ratio_episode_macro']
        free = metrics['validation_internal']['free']['response_energy']
        assert current is not None and np.isfinite(current)
        improved = current < self.best
        self.no_new = 0 if improved else self.no_new+1
        if self.history:
            self.free_worse = self.free_worse+1 if free > self.history[-1]['free_actual_mse'] else 0
        self.best = min(self.best,current)
        self.history.append(dict(step=self.step,dit_identity_ratio=current,
            dit_initial_ratio=metrics['validation_internal']['contact']['dit_error_initial_ratio_episode_macro'],
            teacher_response_ratio=metrics['validation_internal']['contact']['teacher_error_zero_ratio_episode_macro'],
            free_actual_mse=free,no_new=self.no_new,free_worse=self.free_worse))
        write_json(self.out/'history.json',self.history)
        self.save(); self.save(f'step_{self.step:06d}.pt')
        if improved: self.save('best_dit.pt')
        self.ledger.commit('running'); event('validation',**self.history[-1])
        if self.no_new >= 3: return '3 internal checks without new DiT best'
        if self.free_worse >= 3: return 'Free actual MSE increased3 consecutive checks'
        return None

    def run(self):
        r = self.r
        if not self.a.resume:
            assert not self.optimizer.state
            self.save('initial.pt')
            self.evaluate_point(train=True)
        available = min(self.a.steps-self.ledger.used,
                        self.ledger.state['exploration_budget']-self.ledger.before-self.ledger.used)
        final_step = self.step+available
        assert available > 0 and final_step <= len(r.plan)
        loader = DataLoader(r.datasets['train'],batch_sampler=r.plan[self.step:final_step].tolist(),
                            num_workers=4,pin_memory=True)
        reason = None
        for step,raw in enumerate(loader,start=self.step+1):
            batch = to_device(raw,r.device)
            assert batch['index'].cpu().tolist() == r.plan[step-1].tolist()
            assert batch['phase'].cpu().tolist() == [0,0,1,1,2,2,2,2]
            self.evaluator.mode = 'bf16_training'
            self.check_integrity()
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.objective(batch)
            gradients = [p.grad for p in self.params if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients), 'Nonfinite gradient'
            norm = torch.nn.utils.clip_grad_norm_(self.params,1.)
            assert torch.isfinite(norm) and norm > 0, 'Nonfinite or zero RGB gradient'
            self.optimizer.step(); self.step = step; self.ledger.update(step,loss)
            with torch.no_grad():
                for dest,src in zip(self.ema.parameters(),self.network.parameters()): dest.lerp_(src,.01)
            assert all(torch.isfinite(p).all() for p in self.params), 'Nonfinite updated parameter'
            self.check_integrity()
            if step % 25 == 0 or step == final_step:
                row = dict(step=step,loss=loss,gradient_norm=float(norm),elapsed=time.monotonic()-self.started)
                self.training.append(row); write_json(self.out/'training_bf16.json',self.training); event('train',**row)
            if step % self.a.validate_every == 0 or step == final_step:
                reason = self.evaluate_point(train=step == final_step)
                if reason: break
        if self.step < final_step:
            self.evaluator.evaluate(self.step,train_contact=True)
        reason = reason or 'Authorized optimizer-update cap reached'
        self.check_integrity(full=True); self.save(); self.evaluator.write_precision()
        self.ledger.commit('complete',reason)
        write_json(self.out/'summary.json',dict(final_step=self.step,new_optimizer_updates=self.ledger.used,
            stop_reason=reason,initialization=self.a.initialize,source_step=self.manifest['parent_step'],
            global_used=self.ledger.before+self.ledger.used,global_ceiling=self.ledger.state['exploration_budget'],
            original_checkpoint_unchanged=True,frozen_verified=True,cache_verified=True,
            elapsed_seconds=time.monotonic()-self.started,history=self.history,
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),formal_training=False,acceptance_relaxed=False))
        event('complete',step=self.step,updates=self.ledger.used,reason=reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path('data/diagnostics/civi_student_iteration'); match = root/'final150k_match'
    for key,default in dict(root=root,base=match/'matched',predictor=match/'predictor',
        cache=match/'predictor_cache',dependencies=match/'dependencies.json',
        old=Path('data/diagnostics/2026.11.15/student_dual_path_ablation'),
        source=root/'final150k_budget_extend_s42_6000/latest.pt').items():
        parser.add_argument('--'+key,type=Path,default=default)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,required=True)
    parser.add_argument('--initialize',choices=('ema6000','scratch'),required=True)
    parser.add_argument('--additional-budget',type=int,default=0)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--validate-every',type=int,default=200)
    parser.add_argument('--validation-seed',type=int,default=4242)
    parser.add_argument('--eval-probes',type=int,default=2)
    args = parser.parse_args()
    assert 0 < args.steps <= 1000 and args.additional_budget in (0,1000)
    assert args.eval_probes == 2 and args.validate_every == 200
    assert torch.cuda.device_count() == 1, 'Expose exactly one confirmed idle GPU'
    if args.resume:
        assert args.output.resolve() == args.resume.parent.resolve()
        assert not (args.output/'summary.json').exists(), 'Completed run must not be extended'
        assert not (args.output/'failure.json').exists(), 'Recorded anomaly requires review before restart'
    else:
        args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    ledger = Ledger(args); experiment = None
    try:
        experiment = Experiment(args,ledger)
        experiment.run()
    except BaseException as error:
        if experiment is not None:
            try: experiment.save(); experiment.evaluator.write_precision()
            except Exception: pass
        write_json(args.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),updates=ledger.used))
        ledger.commit('failed',repr(error)); raise


if __name__ == '__main__':
    main()
