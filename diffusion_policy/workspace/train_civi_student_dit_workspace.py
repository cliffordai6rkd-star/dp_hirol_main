"""Production direct-torque CIVI renderer, trained only against frozen DiT output.

Historical response workspaces remain loadable. This objective deliberately
owns a fresh run, optimizer and renderer EMA; no response wrapper is involved.
"""
import contextlib
import copy
import json
import logging
import random
import time
from pathlib import Path

import dill
import numpy as np
import torch
import torch.distributed as dist
import hydra
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

from diffusion_policy.distillation.dit_rgb import (
    ARCHITECTURE, PHASES, MixedPlan, diffusion_pair, element_mse, file_sha,
    fp32_reference, move, restore_rng, rng_state, seed_for, slice_batch,
    state_sha, summarize, write_json, foreach_ema_step,
)
from diffusion_policy.distillation.budget import UpdateBudget
from diffusion_policy.dataset.civi_dit_dataset import historical_datasets
from diffusion_policy.dataset.render_net_teacher_cache import TeacherTargetCache, cache_key
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.vision.render_net import build_render_net
from diffusion_policy.model.vision.student_context_adapter import StudentContextAdapter
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor
from diffusion_policy.scripts.visualize_cross_attention_episode import load_checkpoint_payload, instantiate_policy
from diffusion_policy.workspace.train_force_aware_diffusion_workspace import DistributedContext

log = logging.getLogger(__name__)


class PlannedBatches:
    """Separate prefetched sampler position from the last *committed* update."""
    def __init__(self, plan, steps, rank, world):
        self.plan, self.steps, self.rank, self.world = copy.deepcopy(plan), steps, rank, world
        self.pending = {}

    def __iter__(self):
        for step in range(self.steps):
            indices = self.plan.next()
            self.pending[step] = (copy.deepcopy(self.plan.state_dict()), indices)
            yield indices[self.rank::self.world]

    def __len__(self):
        return self.steps


class TrainCiviStudentDiTWorkspace:
    def __init__(self, cfg, output_dir=None, distributed=None):
        self.cfg = cfg
        self.out = Path(output_dir or cfg.output_dir)
        self.d = distributed or DistributedContext()
        self.device = torch.device(f'cuda:{self.d.local_rank}' if self.d.enabled else cfg.training.device)
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        self.out.mkdir(parents=True, exist_ok=True)
        if not cfg.training.resume_from and (self.out/'initial.pt').exists():
            raise FileExistsError('Use a new run directory, or explicit resume_from; initial.pt already exists')
        torch.manual_seed(cfg.training.seed); np.random.seed(cfg.training.seed); random.seed(cfg.training.seed)
        torch.backends.cuda.matmul.allow_tf32 = cfg.training.amp.allow_tf32
        torch.backends.cudnn.allow_tf32 = cfg.training.amp.allow_tf32
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        # SDPA is already used by existing MHA need_weights=False; do not install
        # a different attention implementation or cache rho-dependent memory.
        torch.backends.mha.set_fastpath_enabled(False)
        if cfg.distillation.objective != 'denoiser_mse' or cfg.distillation.kd_weight != 1:
            raise ValueError('Minimal path requires KD weight=1')
        if any(float(v) != 0 for k,v in cfg.distillation.items() if k.endswith('_loss_weight')):
            raise ValueError('Auxiliary losses are disabled for this objective')
        if cfg.training.resume_from and cfg.training.warm_start_from:
            raise ValueError('warm_start and resume are mutually exclusive')
        t = cfg.teacher
        for path, expected in ((t.checkpoint,t.file_sha256),(t.adapter,t.adapter_sha256),(t.condition,t.condition_sha256)):
            if file_sha(path) != expected:
                raise ValueError(f'Dependency SHA256 mismatch: {path}')
        payload, self.teacher_cfg = load_checkpoint_payload(Path(t.checkpoint))
        step = dill.loads(payload['pickles']['optimizer_step'])
        if step != t.step or t.state_key != 'ema_model':
            raise ValueError('Teacher must be the pinned 150k EMA')
        if state_sha(payload['state_dicts'][t.state_key]) != t.state_sha256:
            raise ValueError('Teacher parameter state mismatch')
        self.teacher = instantiate_policy(payload, self.teacher_cfg, self.device, capture_attention=False)
        del payload
        self.teacher.eval().requires_grad_(False)
        self.preprocessor = WrenchPreprocessor.from_policy(self.teacher).eval().requires_grad_(False)
        self.options = OmegaConf.to_container(cfg.render_net, resolve=True)
        if self.options['image_keys'] != list(self.teacher.obs_encoder.image_keys):
            raise ValueError('Camera order differs from teacher')
        if self.options['max_obs_steps'] != self.teacher.n_obs_steps:
            raise ValueError('Causal observation count differs from teacher')
        self.model = build_render_net(self.teacher.obs_encoder.wrench_projection.in_features, self.options).to(self.device)
        self.model.initialize_wrench_encoder(self.teacher.obs_encoder)
        self.adapter = StudentContextAdapter(self.teacher.obs_encoder.n_emb, kind='mlp', hidden_dim=128).to(self.device)
        adapter = torch.load(t.adapter, map_location='cpu', weights_only=False)
        condition = torch.load(t.condition, map_location='cpu', weights_only=False)
        if adapter['teacher_checkpoint_sha256'] != t.matched_teacher_file_sha256 or condition['teacher_sha256'] != t.matched_teacher_file_sha256:
            raise ValueError('Adapter/Condition belong to another teacher')
        self.adapter.load_state_dict(adapter['state'], strict=True)
        self.model.condition_network.load_state_dict(condition['state'], strict=True)
        self.adapter.eval().requires_grad_(False)
        self.model.set_training_stage('rgb'); self.model.set_stage_mode(True)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.parameters, **OmegaConf.to_container(cfg.optimizer))
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.ema = EMAModel(self.ema_model, **dict(cfg.ema))
        self.scaler = torch.amp.GradScaler('cuda', enabled=False) # BF16, FP32 master weights
        self.scheduler = None # no hidden schedule in this new, fixed-LR objective
        self.step, self.best, self.stale, self.selection = 0, float('inf'), 0, {}
        self.plan = None
        self.dependency = dict(teacher_file=t.file_sha256, teacher_state=t.state_sha256,
                               adapter=t.adapter_sha256, condition=t.condition_sha256,
                               normalizer=state_sha(self.teacher.normalizer.state_dict()),
                               preprocessing=self.preprocessor.get_config())
        self.frozen = dict(teacher=state_sha(self.teacher.state_dict()), adapter=state_sha(self.adapter.state_dict()),
                           condition=state_sha(self.model.condition_network.state_dict()))
        self.dit_target, self.dit_student = self.teacher.model, self.teacher.model
        self.visual = self.teacher.obs_encoder.encode_image_context
        self.train_model = self.model
        if cfg.training.warm_start_from:
            old = torch.load(cfg.training.warm_start_from, map_location='cpu', weights_only=False)
            if old['architecture'] != ARCHITECTURE or old['options'] != self.options or old['dependency'] != self.dependency:
                raise ValueError('Warm start must have audited architecture and dependencies')
            key = {'online':'model','ema':'ema_model'}[cfg.training.warm_start_state]
            self.model.load_state_dict(old[key], strict=True)
            self.ema_model.load_state_dict(self.model.state_dict(), strict=True)
        self.frozen['condition'] = state_sha(self.model.condition_network.state_dict())
        if self.d.is_main:
            audit = []
            source = self.teacher.obs_encoder.state_dict()
            for name, value in self.model.state_dict().items():
                mapped = name.replace('wrench_encoder.', 'force_encoder.')
                origin = ('matched Condition snapshot' if name.startswith('condition_network.') else
                          '150k EMA teacher direct torque path' if mapped in source and name.split('.')[0] in
                          ('wrench_projection','wrench_encoder','wrench_position','obs_position','force_modality') else
                          'fresh base initialization (seed42); no response-conditioned transfer')
                audit.append(dict(key=name, shape=list(value.shape), source=origin))
            write_json(self.out/'initialization.json', audit)
            write_json(self.out/'dependencies.json', self.dependency)
            OmegaConf.save(cfg, self.out/'config.yaml', resolve=True)
            OmegaConf.save(self.teacher_cfg, self.out/'teacher_config.yaml', resolve=True)
            write_json(self.out/'teacher_semantics.json', dict(
                step=step, state_key=t.state_key, prediction_type=self.teacher.noise_scheduler.config.prediction_type,
                metric='denoiser KD (single noisy-action regression), not sampled action MSE',
                action_representation=self.teacher.action_representation,
                relative_pose_actions=self.teacher.relative_pose_actions, action_reference_key=self.teacher.action_reference_key,
                horizon=self.teacher.horizon, action_dim=self.teacher.action_dim,
                action_start_index=self.teacher.action_start_index, execution_chunk=self.teacher.n_action_steps,
                num_inference_steps=self.teacher.num_inference_steps,
                native_valid_mask='Native policy rejects valid_mask; this dataset produces full valid padded horizons',
                scheduler=OmegaConf.to_container(self.teacher_cfg.policy.noise_scheduler),
                extra_conditions='No past_action; context is the sole DiT auxiliary condition'))

    def amp(self, module, reference=False):
        enabled = not reference and self.cfg.training.amp.enabled and self.cfg.training.amp[module] and self.device.type == 'cuda'
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=enabled)

    def inputs(self, batch):
        return ({k:batch['obs'][k][:,:self.teacher.n_obs_steps].float() for k in self.options['image_keys']},
                self.preprocessor(batch['obs'])[0])

    def encode(self, images, grad, reference=False):
        with self.amp('visual', reference):
            return self.visual(images, input_grad=grad, context_adapter=self.adapter).float()

    def check_frozen(self):
        for name, module in [('teacher',self.teacher),('adapter',self.adapter),('condition',self.model.condition_network)]:
            if any(p.requires_grad or p.grad is not None for p in module.parameters()):
                raise RuntimeError(f'Frozen {name} gained trainable parameters/gradients')
            if state_sha(module.state_dict()) != self.frozen[name]:
                raise RuntimeError(f'Frozen {name} changed')

    def loader(self, dataset, *, batch_sampler=None, batch_size=None):
        workers = int(self.cfg.data.num_workers)
        kwargs = dict(num_workers=workers, pin_memory=self.cfg.data.pin_memory,
                      generator=torch.Generator().manual_seed(0))
        if workers:
            kwargs.update(persistent_workers=True, prefetch_factor=self.cfg.data.prefetch_factor, multiprocessing_context='spawn')
        if batch_sampler is not None:
            return DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)

    def prepare_data(self):
        # Only rank0 performs decoder audits, hashes, cache construction. Other
        # ranks reconstruct read-only mmap datasets after rank0 finishes.
        if self.d.enabled and not self.d.is_main:
            dist.barrier()
        self.datasets, identity = historical_datasets(self.cfg.data, self.preprocessor.get_config())
        self.data_identity = identity
        if self.d.is_main:
            write_json(self.out/'data_manifest.json', identity)
        b = int(self.cfg.validation.encode_batch)
        if b != int(self.cfg.training.microbatch):
            raise ValueError('Training/evaluation visual encode batch must match')
        for split, dataset in self.datasets.items():
            spec = dict(version=1, architecture=ARCHITECTURE, split=split, dependency=self.dependency,
                        indices=identity[split], rgb=identity['rgb'], images='clear uint8/255, no augmentation',
                        camera_time_layout='B,To=2,camera=[side,wrist],patch', encode_batch=b,
                        dtype='float32', precision='FP32 (TF32 disabled)', condition='FP32 frozen online prediction')
            cache = TeacherTargetCache(Path(self.cfg.data.cache_directory)/cache_key(spec), spec, len(dataset), token_dtype='float32')
            started = time.monotonic()
            if not cache.complete:
                if not self.d.is_main:
                    raise RuntimeError('Rank0 static cache is incomplete')
                subset = Subset(dataset, range(cache.completed, len(dataset)))
                pending_tokens, pending_maps = [], {}
                with torch.no_grad(), fp32_reference():
                    for ordinal,raw in enumerate(self.loader(subset, batch_size=b)):
                        size = len(raw['index']); batch = move(raw, self.device)
                        if size < b:
                            batch = self.pad(batch, b-size)
                        images, wrench = self.inputs(batch)
                        tokens = self.teacher._encode_context(batch['obs'], apply_curriculum_mask=False)
                        masks, condition = self.model.condition_network(images, wrench)
                        pending_tokens.append(tokens[:size].cpu())
                        values = {**masks, 'alpha':condition['contact_alpha']}
                        for key,value in values.items():
                            pending_maps.setdefault(key,[]).append(value[:size].cpu())
                        # Preserve fixed encode shape while amortizing mmap flush
                        # and metadata commits. Cache progress remains a prefix.
                        if len(pending_tokens) == 32 or (ordinal+1)*b >= len(subset):
                            cache.append(torch.cat(pending_tokens),{k:torch.cat(v) for k,v in pending_maps.items()})
                            pending_tokens,pending_maps = [],{}
                            if ordinal%512 < 32:
                                log.info('Static cache %s %d/%d',split,cache.completed,len(dataset))
            dataset.cache = cache
            # Audits include shuffled sample reads, full FP32 target and gates.
            errors = []
            with torch.no_grad(), fp32_reference():
                for index in [len(dataset)-1,0,len(dataset)//2]:
                    batch = move(next(iter(self.loader(Subset(dataset,[index]*b),batch_size=b))), self.device)
                    images, wrench = self.inputs(batch)
                    online = self.teacher._encode_context(batch['obs'], apply_curriculum_mask=False)
                    masks, condition = self.model.condition_network(images,wrench)
                    error = (online-batch['static']['tokens']).abs().max().item()
                    if error > 2e-5:
                        raise RuntimeError(f'Cached/online context mismatch {split} {index}: {error}')
                    for k in masks:
                        torch.testing.assert_close(masks[k], batch['static']['patch_attention'][k], rtol=0, atol=2e-5)
                    torch.testing.assert_close(condition['contact_alpha'],batch['static']['patch_attention']['alpha'],rtol=0,atol=2e-5)
                    errors.append(dict(index=index,context_max_abs=error))
            if self.d.is_main:
                write_json(self.out/f'cache_{split}.json', dict(identity=spec, audits=errors, seconds=time.monotonic()-started))
        if self.d.enabled and self.d.is_main:
            dist.barrier()
        self.plan = MixedPlan(self.datasets['train'].phases, self.datasets['train'].episodes,
                              self.cfg.data.phase_counts, self.cfg.training.seed)
        if self.cfg.training.resume_from:
            self.restore(self.cfg.training.resume_from)

    @staticmethod
    def pad(value, count):
        if isinstance(value, dict):
            return {k:TrainCiviStudentDiTWorkspace.pad(v,count) for k,v in value.items()}
        return torch.cat([value,value[-1:].expand(count,*value.shape[1:])])

    def setup_acceleration(self):
        cfg = self.cfg.training.compile
        kwargs = dict(backend=cfg.backend, mode=cfg.mode)
        if cfg.renderer:
            self.model.compile_backbone(**kwargs)
        if cfg.visual:
            self.visual = torch.compile(self.teacher.obs_encoder.encode_image_context, **kwargs)
        if cfg.dit_teacher:
            self.dit_target = torch.compile(self.teacher.model, **kwargs)
        if cfg.dit_student:
            self.dit_student = torch.compile(self.teacher.model, **kwargs)
        if self.d.enabled:
            self.train_model = DDP(self.model, device_ids=[self.d.local_rank], broadcast_buffers=False,
                                   find_unused_parameters=False)

    def kd(self, batch, model=None, reference=False, probe=None, record=None):
        model = self.train_model if model is None else model
        images, wrench = self.inputs(batch)
        static = batch.get('static')
        cached = None if static is None else ({k:static['patch_attention'][k] for k in images},
                                              {'contact_alpha':static['patch_attention']['alpha']})
        with self.timed('renderer',record), self.amp('renderer',reference):
            rgb_result = model(images,wrench,cached_condition=cached,return_rgb_stats=reference)
            rendered, stats = rgb_result if reference else (rgb_result,None)
        with self.timed('student_visual',record):
            context = self.encode(rendered,torch.is_grad_enabled(),reference)
        noisy, rho, demo = diffusion_pair(self.teacher,batch['action'],sample_ids=batch['sample_id'].tolist() if probe is not None else None,
                                         probe=probe or 0,seed=self.cfg.validation.seed)
        with self.timed('teacher_dit',record), torch.no_grad(), self.amp('dit',reference):
            target_context = static['tokens'].float() if static is not None else self.teacher._encode_context(batch['obs'],apply_curriculum_mask=False)
            expected = (self.teacher.model if reference else self.dit_target)(noisy,rho,target_context)
        with self.timed('student_dit',record), self.amp('dit',reference):
            predicted = (self.teacher.model if reference else self.dit_student)(noisy,rho,context)
        return dict(kd=element_mse(predicted,expected,batch.get('valid_mask')),
                    demo=element_mse(predicted,demo,batch.get('valid_mask')),
                    teacher_demo=element_mse(expected,demo,batch.get('valid_mask')),
                    rendered=rendered, images=images, context=context, expected=expected, stats=stats,
                    noisy=noisy,rho=rho,target_context=target_context)

    @contextlib.contextmanager
    def timed(self, name, record):
        if record is None:
            yield; return
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record();record.setdefault(name,[]).append((start,end))

    @torch.no_grad()
    def evaluate(self, *, full_action=False, shuffled=False):
        self.ema_model.eval()
        rows, visuals = [], []
        dataset = self.datasets['validation']
        count = len(dataset) if self.cfg.validation.max_samples is None else min(len(dataset), self.cfg.validation.max_samples)
        ids = list(range(count))[self.d.rank::self.d.world_size]
        b = int(self.cfg.validation.encode_batch)
        order = np.random.default_rng(self.cfg.validation.seed+77).permutation(len(dataset))
        donor_plan = np.empty(len(dataset),dtype=np.int64)
        donor_plan[order] = np.roll(order,1) # fixed derangement, independent of rank/batch
        with fp32_reference():
            for raw in self.loader(Subset(dataset,ids),batch_size=b):
                size = len(raw['index']); batch = move(raw,self.device)
                if size < b:
                    batch = self.pad(batch,b-size)
                # Validation always bypasses compiled training-only callables.
                old_visual = self.visual
                self.visual = self.teacher.obs_encoder.encode_image_context
                try:
                    results = [self.kd(batch,model=self.ema_model,reference=True,probe=probe)
                               for probe in range(self.cfg.validation.probes)]
                finally:
                    self.visual = old_visual
                result = results[0]
                images,rendered = result['images'],result['rendered']
                z0 = self.teacher.obs_encoder.encode_image_context(images,input_grad=False,context_adapter=self.adapter)
                identity = self.teacher.model(result['noisy'],result['rho'],z0)
                zero_response = torch.stack([element_mse(self.teacher.model(r['noisy'],r['rho'],z0),r['expected']) for r in results]).mean(0)
                metrics = dict(kd=torch.stack([r['kd'] for r in results]).mean(0),
                               demo=torch.stack([r['demo'] for r in results]).mean(0),
                               teacher_demo=torch.stack([r['teacher_demo'] for r in results]).mean(0),
                               identity_kd=zero_response,
                               rgb_mse=torch.stack([(rendered[k]-images[k]).square().flatten(1).mean(1) for k in images]).mean(0),
                               rgb_linf=torch.stack([(rendered[k]-images[k]).abs().flatten(1).max(1).values for k in images]).max(0).values,
                               actual_response_mse=element_mse(result['context'],z0))
                masks = batch['static']['patch_attention']
                alpha = masks['alpha']
                for key in images:
                    support = self.ema_model._mask_gate(masks[key].flatten(0,1),images[key]).reshape_as(masks[key])*alpha[...,None,None,None]
                    delta = rendered[key]-images[key]
                    assert (delta.abs()-.1*support).max() <= 2e-7
                    closed = (support == 0).expand_as(delta)
                    assert torch.equal(rendered[key][closed], images[key][closed])
                stats = result['stats']
                denominator = torch.stack([v['allowed'] for v in stats.values()]).sum(0).clamp_min(1)
                metrics.update(hard_clip_fraction=torch.stack([v['hard_clipped'] for v in stats.values()]).sum(0)/denominator,
                               range_clip_fraction=torch.stack([v['range_clipped'] for v in stats.values()]).sum(0)/denominator)
                if shuffled:
                    paired = copy.copy(batch);paired['obs'] = dict(batch['obs'])
                    # Fixed, split-local cyclic donor; independent of rank/B.
                    donor_ids = [int(donor_plan[int(i)]) for i in raw['index']]
                    donors = move(next(iter(self.loader(Subset(dataset,donor_ids),batch_size=len(donor_ids)))),self.device)
                    if size < b:
                        donors = self.pad(donors,b-size)
                    paired['obs'][self.teacher.wrench_key] = donors['obs'][self.teacher.wrench_key]
                    paired.pop('static') # recompute Condition from the donor torque, never reuse original gates
                    p_images,p_wrench = self.inputs(paired)
                    rgb = self.ema_model(p_images,p_wrench)
                    zs = self.teacher.obs_encoder.encode_image_context(rgb,input_grad=False,context_adapter=self.adapter)
                    metrics['shuffled_kd'] = element_mse(self.teacher.model(result['noisy'],result['rho'],zs),result['expected'])
                if full_action:
                    predictions = {}
                    for name,context in [('enhanced',result['context']),('identity',z0),('teacher',result['target_context'])]:
                        generators = [torch.Generator(device=self.device).manual_seed(seed_for(int(i),999,self.cfg.validation.seed)) for i in batch['sample_id']]
                        action = self.teacher.conditional_sample((b,self.teacher.horizon,self.teacher.action_dim),context,generator=generators)
                        predictions[name] = self.teacher.normalizer['action'].unnormalize(action)
                    # Reuse native conversion and execution slicing (also valid
                    # when a future compatible teacher uses relative EE poses).
                    from diffusion_policy.policy.force_aware_diffusion_transformer_policy import relative_pose_to_absolute_pose, normalize_pose_quaternions
                    truth = batch['action']
                    if self.teacher.action_representation == 'ee_pose':
                        predictions = {k:normalize_pose_quaternions(v) for k,v in predictions.items()}
                    if self.teacher.relative_pose_actions:
                        ref = batch['obs'][self.teacher.action_reference_key]
                        predictions = {k:relative_pose_to_absolute_pose(v,ref) for k,v in predictions.items()}
                        truth = relative_pose_to_absolute_pose(truth,ref)
                    start,end = self.teacher.action_start_index,self.teacher.action_start_index+self.teacher.n_action_steps
                    metrics.update({name+'_sampled_action_mse':element_mse(value[:,start:end],truth[:,start:end]) for name,value in predictions.items()})
                values = {k:v.cpu().tolist() for k,v in metrics.items()}
                for j in range(size):
                    row = dict(index=int(raw['index'][j]),episode=int(raw['episode'][j]),phase=int(raw['phase'][j]),
                               **{k:v[j] for k,v in values.items()})
                    row['rgb_rms'] = row['rgb_mse']**.5
                    row['actual_response_rms'] = row['actual_response_mse']**.5
                    row['improvement_vs_original'] = row['identity_kd']-row['kd']
                    rows.append(row)
                    if row['phase']==2 and not any(v['episode']==row['episode'] for v in visuals):
                        visuals.append(dict(episode=row['episode'],index=row['index'],
                            original={k:v[j,-1].cpu() for k,v in images.items()},
                            enhanced={k:v[j,-1].cpu() for k,v in rendered.items()}))
        if self.d.enabled:
            gathered = [None]*self.d.world_size
            dist.all_gather_object(gathered,rows)
            rows = [r for part in gathered for r in part]
        rows.sort(key=lambda r:r['index'])
        if len(rows) != count or len({r['index'] for r in rows}) != count:
            raise RuntimeError('Validation windows duplicated or missing')
        summary,episodes = summarize(rows,self.cfg.validation.retained_episodes)
        if self.d.is_main:
            initial_path = self.out/'validation_0_samples.json'
            if initial_path.exists():
                initial = {r['index']:r for r in json.loads(initial_path.read_text())}
                for row in rows:
                    row['improvement_vs_run_step0'] = initial[row['index']]['kd']-row['kd']
            write_json(self.out/f'validation_{self.step}_samples.json',rows)
            write_json(self.out/f'validation_{self.step}_episodes.json',episodes)
            write_json(self.out/f'validation_{self.step}.json',dict(step=self.step,precision='FP32',weight_type='renderer EMA',
                full_validation=count==len(dataset),sampled_action=full_action,summary=summary))
            torch.save(visuals,self.out/f'visuals_{self.step}.pt')
            self.save_visuals(visuals)
        score = summary.get('internal',{}).get('contact',{}).get('kd')
        if score is None:
            # A truncated smoke has no right to choose a "best" model.
            return False
        if count == len(dataset):
            if score < self.best-self.cfg.training.early_stopping.min_delta:
                self.best,self.stale,self.selection = score,0,dict(step=self.step,internal_contact_kd=score)
                self.save('best.pt')
            else:
                self.stale += 1
        return self.stale >= self.cfg.training.early_stopping.patience

    def save_visuals(self, visuals):
        from PIL import Image
        for v in visuals[:4]:
            strips = []
            for key in self.options['image_keys']:
                strips.append(torch.cat([v['original'][key],v['enhanced'][key],(v['enhanced'][key]-v['original'][key]).abs()*10],dim=-1))
            array = (torch.cat(strips,dim=-2).clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8')
            Image.fromarray(array).save(self.out/f'rgb_step{self.step}_episode{v["episode"]}.png')

    def save(self, name='latest.pt'):
        states = [rng_state()]
        if self.d.enabled:
            states = [None]*self.d.world_size
            dist.all_gather_object(states,rng_state())
        if not self.d.is_main:
            return
        payload = dict(architecture=ARCHITECTURE,options=self.options,dependency=self.dependency,data_identity=self.data_identity,
                       cfg=OmegaConf.to_container(self.cfg,resolve=True),stage='rgb',step=self.step,
                       model=self.model.state_dict(),ema_model=self.ema_model.state_dict(),ema=self.ema.state_dict(),
                       optimizer=self.optimizer.state_dict(),scaler=self.scaler.state_dict(),scheduler=None,
                       rng=states,world_size=self.d.world_size,sampler=self.plan.state_dict(),
                       best=self.best,stale=self.stale,selection=self.selection)
        temporary = self.out/(name+'.tmp')
        torch.save(payload,temporary);temporary.replace(self.out/name)

    def restore(self, path):
        saved = torch.load(path,map_location='cpu',weights_only=False)
        checks = dict(architecture=ARCHITECTURE,options=self.options,dependency=self.dependency,data_identity=self.data_identity)
        for key,value in checks.items():
            if saved[key] != value:
                raise ValueError(f'Refusing incompatible resume: {key}')
        old = saved['cfg']
        # A larger explicit step cap/eval cadence is allowed; optimizer/precision/
        # acceleration/batch semantics must not silently change on resume.
        for key in ('optimizer','ema','distillation'):
            if old[key] != OmegaConf.to_container(self.cfg[key],resolve=True):
                raise ValueError(f'Resume {key} changed')
        for key in ('microbatch','global_batch','amp','compile','seed','max_grad_norm'):
            current = self.cfg.training[key]
            current = OmegaConf.to_container(current,resolve=True) if OmegaConf.is_config(current) else current
            if old['training'][key] != current:
                raise ValueError(f'Resume training.{key} changed')
        if saved['world_size'] != self.d.world_size:
            raise ValueError('Exact RNG resume requires the same world size; use explicit warm_start for repartitioning')
        self.model.load_state_dict(saved['model'],strict=True);self.ema_model.load_state_dict(saved['ema_model'],strict=True)
        self.optimizer.load_state_dict(saved['optimizer']);self.ema.load_state_dict(saved['ema']);self.scaler.load_state_dict(saved['scaler'])
        self.plan.load_state_dict(saved['sampler'])
        self.step,self.best,self.stale,self.selection = saved['step'],saved['best'],saved['stale'],saved['selection']
        restore_rng(saved['rng'][self.d.rank])
        if self.d.is_main and Path(path).parent.resolve() != self.out.resolve():
            # Preserve the original step0 reference when a resumable run is
            # moved to a new output directory; never redefine it at resume.
            import shutil
            for name in ('validation_0_samples.json','validation_0.json','initialization.json'):
                source = Path(path).parent/name
                if source.exists() and not (self.out/name).exists():
                    shutil.copy2(source,self.out/name)

    def run(self):
        self.prepare_data()
        if self.cfg.training.cache_only:
            return
        global_batch = int(self.cfg.training.global_batch)
        micro = int(self.cfg.training.microbatch)
        if global_batch != sum(self.cfg.data.phase_counts) or global_batch%(micro*self.d.world_size):
            raise ValueError('Global Mixed batch must equal phase counts and divide microbatch * world_size')
        accumulation = global_batch//(micro*self.d.world_size)
        if self.d.is_main:
            write_json(self.out/'batch_semantics.json',dict(microbatch=micro,gradient_accumulation=accumulation,
                global_batch=global_batch,world_size=self.d.world_size,phase_counts=list(self.cfg.data.phase_counts)))
        budget = UpdateBudget(self.cfg.budget.ledger_root,self.cfg.budget.task_id,self.cfg.budget.maximum_new_updates,self.out) if self.d.is_main else None
        self.setup_acceleration()
        if not self.cfg.training.resume_from:
            random.seed(self.cfg.training.seed+self.d.rank)
            np.random.seed(self.cfg.training.seed+self.d.rank)
            torch.manual_seed(self.cfg.training.seed+self.d.rank)
        remaining = int(self.cfg.training.max_steps)-self.step
        if remaining < 0:
            raise ValueError('max_steps precedes resumed update')
        if self.step == 0:
            self.save('initial.pt')
            self.evaluate(full_action=True,shuffled=True)
        sampler = PlannedBatches(self.plan,remaining,self.d.rank,self.d.world_size)
        loader = self.loader(self.datasets['train'],batch_sampler=sampler)
        profiles,history = [], []
        previous = time.monotonic()
        for ordinal,raw in enumerate(loader):
            wait = time.monotonic()-previous
            self.model.set_stage_mode(True)
            self.teacher.eval();self.adapter.eval()
            self.optimizer.zero_grad(set_to_none=True)
            profile = {} if ordinal < self.cfg.training.profile_steps else None
            with self.timed('h2d',profile):
                batch = move(raw,self.device)
            total = torch.zeros((),device=self.device)
            started = time.monotonic()
            for part in range(accumulation):
                sync = self.train_model.no_sync() if self.d.enabled and part+1<accumulation else contextlib.nullcontext()
                with sync:
                    result = self.kd(slice_batch(batch,part*micro,(part+1)*micro),record=profile)
                    phases = batch['phase'][part*micro:(part+1)*micro]
                    weight = torch.where(phases==0,self.cfg.data.free_weight,1.)
                    loss = (result['kd']*weight).sum()/global_batch*self.d.world_size
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite KD')
                    total += loss.detach()
                    with self.timed('backward',profile):
                        loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.parameters,self.cfg.training.max_grad_norm,error_if_nonfinite=True)
            pending,indices = sampler.pending.pop(ordinal)
            if self.d.is_main:
                budget.charge(self.step+1,indices)
            if self.d.enabled:
                dist.barrier() # propagate rank0 accounting failure before mutation
            with self.timed('optimizer_ema',profile):
                self.optimizer.step();foreach_ema_step(self.ema,self.model)
            self.plan.load_state_dict(pending);self.step += 1
            if profile is not None:
                torch.cuda.synchronize(self.device)
                profiles.append(dict(step=self.step,data_wait_seconds=wait,
                    **{k:sum(a.elapsed_time(b) for a,b in pairs) for k,pairs in profile.items()}))
            if self.step%self.cfg.training.log_every == 0 or self.step == 1:
                if self.d.enabled:
                    dist.all_reduce(total);total /= self.d.world_size
                entry = dict(step=self.step,denoiser_kd=float(total),gradient_norm=float(norm),
                    step_seconds=time.monotonic()-started,data_wait_seconds=wait,
                    samples_seen=self.plan.position*global_batch,contact_exposures=self.plan.position*self.cfg.data.phase_counts[2],
                    peak_memory_bytes=torch.cuda.max_memory_allocated(self.device))
                if self.d.is_main:
                    with (self.out/'train.jsonl').open('a') as stream:
                        stream.write(json.dumps(entry)+'\n')
                    log.info('step=%d denoiser KD=%.8g grad=%.5g',self.step,entry['denoiser_kd'],entry['gradient_norm'])
                    history.append(entry)
            if self.step%self.cfg.training.save_every == 0:
                self.check_frozen();self.save()
            if self.step%self.cfg.validation.every == 0 or self.step == self.cfg.training.max_steps:
                stop = self.evaluate(full_action=self.step%self.cfg.validation.full_action_every==0 or self.step==self.cfg.training.max_steps,
                                     shuffled=self.step%self.cfg.validation.shuffled_every==0 or self.step==self.cfg.training.max_steps)
                self.check_frozen();self.save()
                if stop:
                    break
            previous = time.monotonic()
        self.check_frozen();self.save()
        if self.d.is_main:
            write_json(self.out/'profile.json',dict(profiles=profiles,units='CUDA components: ms; data wait: seconds',
                history=history,precision=OmegaConf.to_container(self.cfg.training.amp),
                note='Full KD: renderer + input-gradient frozen DINO/Adapter + target/student DiT + backward + optimizer/EMA. Static cache preparation/validation recorded separately.'))
            from diffusion_policy.policy.civi_rgb import export_rgb
            export_rgb(self.out/'latest.pt',self.out/'civi_rgb.pt',self.preprocessor)
        if self.d.enabled:
            dist.barrier()
