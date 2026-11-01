"""Train only RenderNet using a frozen teacher's context or denoising outputs."""

import contextlib
import copy
import fcntl
import hashlib
import itertools
import json
import logging
import math
import pathlib
import random
import shutil
import time

import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf, open_dict
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from diffusion_policy.common.render_net_training_logger import RenderNetTrainingLogger
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.teacher_cache_lifecycle import TeacherCacheLifecycle
from diffusion_policy.dataset.render_net_teacher_cache import (
    CACHE_VERSION, CachedTeacherDataset, TeacherTargetCache, cache_key, dataset_source_signature,
)
from diffusion_policy.dataset.image_result_cache import use_shared_ram_result_cache
from diffusion_policy.model.vision.render_net import RenderNet, build_render_net
from diffusion_policy.model.vision.render_net_attention import (
    attention_regions, teacher_attention_targets, validate_percentiles,
)
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor
from diffusion_policy.scripts.visualize_cross_attention_episode import (
    instantiate_policy, load_checkpoint_payload,
)
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.workspace.train_force_aware_diffusion_workspace import (
    DistributedContext, per_rank_batch_size, resolve_amp_dtype,
)


log = logging.getLogger(__name__)


class _NullTrainingLogger:
    def log(self, data):
        pass

    def save_plot(self):
        pass


def student_context_tokens(teacher, render_net, obs, *, return_aux=False, image_context_encoder=None):
    """Only RGB and normalized wrench enter RenderNet; no teacher force/fusion path."""
    images, wrench, _, _ = teacher.prepare_context_inputs(obs)
    rendered, masks = render_net(images, wrench, return_aux=True)
    encode = teacher.obs_encoder.encode_image_context if image_context_encoder is None else image_context_encoder
    tokens = encode(rendered, input_grad=True)
    return (tokens, masks) if return_aux else tokens


class TrainDistillRenderNetWorkspace(BaseWorkspace):
    exclude_keys = ("teacher_policy", "_training_model")
    include_keys = ("global_step", "epoch", "best_val_loss", "initial_train_mse",
                    "initial_val_mse", "shape_info", "teacher_state_key", "latest_val_loss",
                    "best_epoch", "best_step", "epochs_without_improvement",
                    "early_stopping_best_loss", "stop_reason")

    def __init__(self, cfg, output_dir=None, distributed=None):
        super().__init__(cfg, output_dir)
        self.distributed = distributed or DistributedContext()
        if self.distributed.enabled:
            with open_dict(cfg):
                if str(cfg.training.device).startswith("cuda"):
                    cfg.training.device = f"cuda:{self.distributed.local_rank}"
            if cfg.get("teacher_cache", {}).get("load_into_ram", False):
                raise ValueError("DDP keeps teacher tokens on SSD; set teacher_cache.load_into_ram=false")
        seed = int(cfg.training.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device(str(cfg.training.device))
        self._validate_distillation_config()
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; use training.device=cpu")
            torch.cuda.set_device(self.device)
        self.amp_enabled = bool(cfg.training.amp.enabled) and self.device.type == "cuda"
        self.amp_dtype = resolve_amp_dtype(cfg.training.amp.dtype)
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.training.amp.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.training.amp.allow_tf32)

        checkpoint = pathlib.Path(str(cfg.teacher_checkpoint)).expanduser().resolve()
        payload, self.teacher_cfg = load_checkpoint_payload(checkpoint)
        self.teacher_state_key = "ema_model" if "ema_model" in payload["state_dicts"] else "model"
        self.teacher_policy = instantiate_policy(payload, self.teacher_cfg, self.device,
                                                 capture_attention=False)
        del payload
        if self.teacher_policy.obs_encoder.cross_attn_query != "image":
            raise ValueError("RenderNet distillation requires teacher cross_attn_query: image")
        self.teacher_policy.eval().requires_grad_(False)
        self.normalizer = self.teacher_policy.normalizer
        self.wrench_preprocessor = WrenchPreprocessor.from_policy(self.teacher_policy).eval().requires_grad_(False)
        with open_dict(self.cfg):
            self.cfg.wrench_preprocessing = self.wrench_preprocessor.get_config()
        wrench_dim = int(self.teacher_policy.obs_encoder.wrench_projection.in_features)
        render_config = OmegaConf.to_container(cfg.render_net, resolve=True)
        if int(render_config.get("architecture_version", 1)) == 2:
            encoder = self.teacher_policy.obs_encoder
            render_config.update(
                image_keys=list(encoder.image_keys),
                wrench_history_steps=int(encoder.wrench_history_steps),
                max_obs_steps=int(encoder.max_obs_steps),
            )
            if render_config.get("initialize_wrench_from_teacher", True):
                if not isinstance(encoder.force_encoder, torch.nn.LSTM):
                    raise ValueError("RenderNet teacher wrench initialization requires an LSTM teacher")
                render_config.update(
                    conditioning_dim=int(encoder.n_emb),
                    wrench_layers=int(encoder.force_encoder.num_layers),
                    wrench_dropout=float(encoder.force_encoder.dropout),
                )
            with open_dict(self.cfg):
                self.cfg.render_net = OmegaConf.create(render_config)
        self.model = build_render_net(wrench_dim, render_config).to(self.device)
        if isinstance(self.model, RenderNet) and self.model.initialize_wrench_from_teacher:
            self.model.initialize_wrench_encoder(self.teacher_policy.obs_encoder)
        self.attention_config = cfg.render_net.get("high_attention_render", {})
        self.attention_enabled = bool(self.attention_config.get("enabled", False))
        self.mask_loss_weight = float(self.attention_config.get("mask_loss_weight", 1.0))
        if self.attention_enabled:
            validate_percentiles(self.attention_config["percentiles"])
            if not math.isfinite(self.mask_loss_weight) or self.mask_loss_weight <= 0:
                raise ValueError("high_attention_render.mask_loss_weight must be positive and finite")
        self.loss_metrics = {}
        self._student_image_encoder = None
        self._training_model = None
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **OmegaConf.to_container(cfg.optimizer))
        self.grad_scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_enabled and self.amp_dtype == torch.float16,
        )
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.initial_train_mse = None
        self.initial_val_mse = None
        self.shape_info = None
        self.latest_val_loss = None
        self.validation_enabled = True
        self.best_epoch = 0
        self.best_step = 0
        self.epochs_without_improvement = 0
        self.early_stopping_best_loss = float("inf")
        self.stop_reason = None
        self.started_at = time.monotonic()
        self.resume_from = None
        resume_path = cfg.training.get("resume_from")
        if resume_path:
            resume_path = pathlib.Path(str(resume_path)).expanduser().resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(f"RenderNet resume checkpoint does not exist: {resume_path}")
            # Restore optimizer/model and the loop counters, while keeping the
            # current Hydra output directory so a resumed run can append logs.
            restored = self.load_checkpoint(path=resume_path, include_keys=self.include_keys)
            saved_objective = restored["cfg"].get("distillation", {}).get("objective", "token_mse")
            if saved_objective != self.objective:
                raise ValueError("Cannot resume with a different distillation objective: "
                                 f"checkpoint={saved_objective}, configured={self.objective}. "
                                 "Start a new run so checkpoint selection metrics remain comparable.")
            self.resume_from = str(resume_path)
            # A checkpoint may have been saved immediately before an orderly
            # stop; an explicit resume should always re-enter the epoch loop.
            self.stop_reason = None
            log.info("Resuming RenderNet from %s: epoch=%d step=%d best_val_mse=%.8f",
                     resume_path, self.epoch, self.global_step, self.best_val_loss)
        self.num_epochs = int(cfg.training.num_epochs)
        self.max_steps = cfg.training.max_steps
        self.max_val_batches = cfg.training.max_val_batches
        self.early_stopping_enabled = bool(cfg.training.early_stopping.enabled)
        self.patience = int(cfg.training.early_stopping.patience)
        self.min_delta = float(cfg.training.early_stopping.min_delta)
        if self.num_epochs < 1 or any(value is not None and int(value) < 1
                                      for value in (self.max_steps, self.max_val_batches)):
            raise ValueError("num_epochs must be positive; max_steps/max_val_batches must be positive or null")
        if self.patience < 1 or not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("early_stopping.patience must be positive; min_delta must be finite and nonnegative")
        log.info(f"Teacher: {checkpoint} ({self.teacher_state_key}), cross_attn_query=image")
        log.info(f"Trainable parameters: teacher=0, RenderNet={sum(p.numel() for p in self.model.parameters()):,}")
        log.info(f"Backbone={self.model.backbone_name}; high-attention rendering={self.attention_enabled}")
        if isinstance(self.model, RenderNet):
            log.info("RenderNet v2: full LSTM sequence [B*To,%d,%d], image-query cross-attention, "
                     "conv_block=%s, rendering_mode=%s, cameras=%s; teacher wrench initialization=%s",
                     self.model.wrench_history_steps, self.model.wrench_encoder.hidden_size,
                     self.model.conv_block, getattr(self.model, "rendering_mode", "residual"),
                     self.model.image_keys, self.model.initialize_wrench_from_teacher)
        if self.attention_enabled:
            log.info("Teacher attention: executed-action DiT→fusion→DINO patches, all denoising calls; "
                  f"percentiles={list(self.attention_config['percentiles'])}; "
                  "batched rollout on model device; student predicts region at inference; "
                  f"validation metric={self.metric_name}")
        log.info("Distillation objective=%s; token auxiliary weight=%g; demo weight=%g; "
                 "DINO and DiT stay frozen, student input gradients remain enabled",
                 self.objective, self.distillation_config.get("token_loss_weight", 0.0),
                 self.distillation_config.get("demo_loss_weight", 0.0))
        log.info(f"Device={self.device}; AMP={self.amp_enabled} {self.amp_dtype}; "
              f"GPU={torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else 'CPU'}")
        log.info(f"Max epochs={self.num_epochs}; max steps={self.max_steps}; "
              f"validation={'all batches' if self.max_val_batches is None else self.max_val_batches}; "
              f"early stopping={self.early_stopping_enabled}, patience={self.patience} epochs, "
              f"min_delta={self.min_delta}")

    @property
    def distillation_config(self):
        return getattr(self, "cfg", {}).get("distillation", {})

    @property
    def objective(self):
        # Saved configurations predating behavior distillation retain their loss.
        return str(self.distillation_config.get("objective", "token_mse"))

    @property
    def metric_name(self):
        return "denoiser MSE" if self.objective == "denoiser_mse" else "context token MSE"

    def _validate_distillation_config(self):
        if self.objective not in {"token_mse", "denoiser_mse"}:
            raise ValueError("distillation.objective must be token_mse or denoiser_mse")
        for name in ("token_loss_weight", "demo_loss_weight"):
            value = float(self.distillation_config.get(name, 0.0))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"distillation.{name} must be finite and nonnegative")

    @contextlib.contextmanager
    def _validation_probes(self):
        if self.objective != "denoiser_mse":
            yield
            return
        seed = int(self.distillation_config.get("validation_seed", 42)) + self.distributed_context.rank
        previous = getattr(self, "_validation_generator", None)
        self._validation_generator = torch.Generator(device=self.device).manual_seed(seed)
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] \
            if self.device.type == "cuda" else []
        try:
            # DataLoader iterator creation can consume the default CPU RNG even
            # though noise/timestep probes use a dedicated device generator.
            with torch.random.fork_rng(devices=devices):
                yield
        finally:
            self._validation_generator = previous

    @property
    def distributed_context(self):
        return getattr(self, "distributed", DistributedContext())

    @property
    def is_main_process(self):
        return self.distributed_context.is_main

    def _sum(self, values):
        if not self.distributed_context.enabled:
            return list(values)
        totals = torch.tensor(values, device=self.device, dtype=torch.float64)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        return totals.tolist()

    def _max_seconds(self, seconds):
        if not self.distributed_context.enabled:
            return seconds
        value = torch.tensor(seconds, device=self.device, dtype=torch.float64)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return value.item()

    def _broadcast_status(self, status):
        dist.broadcast_object_list(status, src=0,
                                   device=self.device if self.device.type == "cuda" else None)
        if not status[0]["ok"]:
            raise RuntimeError(f"Preparation failed on rank 0: {status[0]['error']}")

    def _instantiate_dataset(self, config):
        def instantiate():
            dataset = hydra.utils.instantiate(config)
            # The Hirol dataset converts RAM preload into a shared tmpfs cache
            # under torchrun. Do not silently turn this into an SSD image cache.
            wants_ram = config.get("preload_images", False) and (
                config.get("load_result_add", "ram") in (None, "", "ram", "memory", "mem")
                or use_shared_ram_result_cache(config.get("load_result_add"))
            )
            if (self.distributed_context.enabled and wants_ram
                    and hasattr(dataset, "load_result_add")
                    and not use_shared_ram_result_cache(dataset.load_result_add)):
                dataset.lerobot_dataset.close()
                raise RuntimeError("Decoded images could not fit in shared RAM. Increase /dev/shm "
                                   "capacity (Docker: --shm-size) and available host RAM.")
            return dataset

        if not self.distributed_context.enabled:
            return instantiate()
        dataset = None
        status = [None]
        if self.is_main_process:
            try:
                dataset = instantiate()
                status[0] = {"ok": True}
            except Exception as exc:
                status[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        # Decode once before followers check free tmpfs space or open the cache.
        self._broadcast_status(status)
        return dataset if self.is_main_process else instantiate()

    def _setup_distributed_model(self):
        if not self.distributed_context.enabled:
            return
        config = self.cfg.training.get("distributed", {})
        self._training_model = DistributedDataParallel(
            self.model,
            device_ids=[self.distributed_context.local_rank] if self.device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=bool(config.get("find_unused_parameters", False)),
        )

    def _autocast(self):
        if not self.amp_enabled:
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=self.amp_dtype)

    def _setup_acceleration(self):
        config = self.cfg.training.get("compile", {})
        if not config.get("enabled", False):
            return
        if self.device.type != "cuda":
            log.info("Training compilation skipped on CPU")
            return
        options = {"backend": str(config.get("backend", "inductor")),
                   "mode": str(config.get("mode", "default")),
                   "fullgraph": True, "dynamic": False}
        if config.get("render_backbone", True):
            if not isinstance(self.model, RenderNet):
                raise ValueError("Compiling the RenderNet backbone requires architecture_version=2")
            self.model.compile_backbone(**options)
        if config.get("student_dino", True):
            self._student_image_encoder = torch.compile(
                self.teacher_policy.obs_encoder.encode_image_context, **options,
            )
        log.info("torch.compile enabled for training: render_backbone=%s, student_dino=%s, "
                 "mode=%s; first training batch includes compilation; validation uses eager execution",
                 config.get("render_backbone", True), config.get("student_dino", True), options["mode"])

    def _progress(self, iterable, description, total=None, *, initial=0, unit="batch"):
        config = getattr(self, "cfg", {}).get("logging", {})
        return tqdm(iterable, desc=description, total=total, initial=initial, unit=unit, dynamic_ncols=True,
                    mininterval=float(config.get("tqdm_interval_sec", 1.0)),
                    disable=not self.is_main_process or not config.get("progress_bar", True))

    def _attention_progress(self, completed, total, elapsed):
        if completed == 0:
            self.attention_progress_last_seconds = 0.0
            return
        if elapsed - self.attention_progress_last_seconds >= 30:
            log.info(f"Teacher attention: samples={completed}/{total}, elapsed={elapsed:.1f}s, "
                     f"ETA={(total - completed) * elapsed / completed:.1f}s")
            self.attention_progress_last_seconds = elapsed

    def _loss(self, obs, cached_targets=None, *, action=None):
        # The student always uses its own region prediction. Spatial targets
        # are training-only; model selection excludes all auxiliary losses.
        supervise_regions = self.attention_enabled and self.model.training and torch.is_grad_enabled()
        with self._autocast():
            with torch.no_grad():
                if cached_targets is not None:
                    teacher_tokens = cached_targets["tokens"].to(self.device, non_blocking=True)
                    if supervise_regions:
                        target_masks = attention_regions(
                            cached_targets["patch_attention"], obs, self.teacher_policy.obs_encoder.backbone,
                            self.attention_config["percentiles"],
                        )
                elif supervise_regions:
                    teacher_tokens, target_masks = teacher_attention_targets(
                        self.teacher_policy, obs, percentiles=self.attention_config["percentiles"],
                        seed=self.attention_config.get("teacher_seed", 42),
                        progress_callback=self._attention_progress,
                    )
                else:
                    teacher_tokens = self.teacher_policy._encode_context(
                        obs, apply_curriculum_mask=False,
                    ).detach()
            compiled_encoder = (getattr(self, "_student_image_encoder", None)
                                if self.model.training and torch.is_grad_enabled() else None)
            renderer = (getattr(self, "_training_model", None)
                        if self.model.training and torch.is_grad_enabled() else None)
            student_tokens, masks = student_context_tokens(
                self.teacher_policy, self.model if renderer is None else renderer, obs,
                return_aux=True, image_context_encoder=compiled_encoder,
            )
            if student_tokens.shape != teacher_tokens.shape:
                raise RuntimeError(f"Token shapes differ: teacher={teacher_tokens.shape}, student={student_tokens.shape}")
            token_mse = F.mse_loss(student_tokens.float(), teacher_tokens.float())
            mask_mse = (torch.stack([F.mse_loss(masks[key].float(), target_masks[key].float())
                                     for key in masks]).mean() if supervise_regions else
                        token_mse.new_zeros(()))
            denoiser_mse = demo_mse = token_mse.new_zeros(())
            if self.objective == "denoiser_mse":
                denoiser_mse, demo_mse = self._denoiser_losses(
                    action, student_tokens, teacher_tokens.to(student_tokens.dtype),
                )
                loss = (denoiser_mse
                        + float(self.distillation_config.get("token_loss_weight", 0.0)) * token_mse
                        + float(self.distillation_config.get("demo_loss_weight", 0.0)) * demo_mse)
            else:
                loss = token_mse
            if supervise_regions:
                loss = loss + self.mask_loss_weight * mask_mse
            self.loss_metrics = {"token_mse": float(token_mse.detach()),
                                 "denoiser_mse": float(denoiser_mse.detach()),
                                 "demo_mse": float(demo_mse.detach()),
                                 "mask_mse": float(mask_mse.detach()), "loss": float(loss.detach()),
                                 "mask_supervised": supervise_regions}
        if self.shape_info is None:
            self.shape_info = {
                "teacher_token_shape": list(teacher_tokens.shape),
                "student_token_shape": list(student_tokens.shape),
                "camera_keys": list(self.teacher_policy.obs_encoder.image_keys),
                "camera_count": len(self.teacher_policy.image_keys),
                "observation_steps": self.teacher_policy.n_obs_steps,
                "patch_tokens_per_camera": dict(self.teacher_policy.obs_encoder.patch_counts),
                "patch_tokens_per_observation": teacher_tokens.shape[1] // self.teacher_policy.n_obs_steps,
                "wrench_key": self.teacher_policy.wrench_key,
                "wrench_shape": list(obs[self.teacher_policy.wrench_key].shape),
            }
            log.info("Token layout: " + json.dumps(self.shape_info))
        return loss

    def _denoiser_losses(self, action, student_tokens, teacher_tokens):
        if action is None:
            raise ValueError("denoiser_mse distillation requires batch['action']")
        teacher = self.teacher_policy
        action = action.to(self.device, non_blocking=True)
        if action.ndim != 3 or action.shape[0] != student_tokens.shape[0]:
            raise ValueError("Distillation actions must have shape [B, horizon, action_dim]")
        normalized_action = teacher.normalizer["action"].normalize(action)
        generator = None
        if not (self.model.training and torch.is_grad_enabled()):
            generator = getattr(self, "_validation_generator", None)
            if generator is None:
                seed = (int(self.distillation_config.get("validation_seed", 42))
                        + self.distributed_context.rank)
                generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(normalized_action.shape, device=self.device,
                            dtype=normalized_action.dtype, generator=generator)
        timesteps = torch.randint(
            0, teacher.noise_scheduler.config.num_train_timesteps,
            (normalized_action.shape[0],), device=self.device, generator=generator,
        ).long()
        noisy_action = teacher.noise_scheduler.add_noise(normalized_action, noise, timesteps)
        with torch.no_grad():
            target = teacher.model(noisy_action, timesteps, teacher_tokens)
        # Frozen parameters must still propagate derivatives w.r.t. context;
        # no_grad here would sever the RenderNet -> DINO -> DiT gradient path.
        prediction = teacher.model(noisy_action, timesteps, student_tokens)
        denoiser_mse = F.mse_loss(prediction.float(), target.float())
        prediction_type = teacher.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            demo_target = noise
        elif prediction_type == "sample":
            demo_target = normalized_action
        elif prediction_type == "v_prediction":
            demo_target = teacher.noise_scheduler.get_velocity(normalized_action, noise, timesteps)
        else:
            raise ValueError(f"unsupported prediction type {prediction_type!r}")
        return denoiser_mse, F.mse_loss(prediction.float(), demo_target.float())

    def _token_mse(self, loss):
        # Fallback permits lightweight callers that replace _loss in loop tests.
        return getattr(self, "loss_metrics", {}).get("token_mse", float(loss.detach()))

    def _selection_mse(self, loss):
        return getattr(self, "loss_metrics", {}).get(self.objective, float(loss.detach()))

    def _obs(self, batch):
        return dict_apply(batch["obs"], lambda value: value.to(self.device, non_blocking=True))

    def _batch_loss(self, batch):
        obs = self._obs(batch)
        if self.objective == "denoiser_mse":
            return self._loss(obs, cached_targets=batch.get("teacher_targets"), action=batch.get("action"))
        if "teacher_targets" in batch:
            return self._loss(obs, cached_targets=batch["teacher_targets"])
        return self._loss(obs)

    @torch.no_grad()
    def _prepare_cached_datasets(self, dataset, validation_dataset, dataset_cfg):
        if not self.distributed_context.enabled or not self.cfg.get("teacher_cache", {}).get("enabled", False):
            return self._prepare_cached_datasets_local(dataset, validation_dataset, dataset_cfg)
        # All ranks lease the same directory before rank 0 starts building it.
        # The last rank to exit can clean up without a collective in finally,
        # which would deadlock if a peer has already failed.
        status = [None]
        if self.is_main_process:
            try:
                root, identity = self._teacher_cache_spec(dataset, validation_dataset, dataset_cfg)
                status[0] = {"ok": True, "root": str(root), "identity": identity}
            except Exception as exc:
                status[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        self._broadcast_status(status)
        cache_spec = pathlib.Path(status[0]["root"]), status[0]["identity"]
        lifetime = getattr(self, "_teacher_cache_lifetime", None)
        if lifetime is not None:
            lifetime.acquire(cache_spec[0])
        dist.barrier()
        # Only rank 0 may build/resume targets. Other ranks wait without taking
        # the writer lock, then reopen the complete arrays with read-only mmap.
        result = None
        status = [None]
        if self.is_main_process:
            try:
                result = self._prepare_cached_datasets_local(
                    dataset, validation_dataset, dataset_cfg, cache_spec=cache_spec)
                status[0] = {"ok": True}
            except Exception as exc:
                status[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        self._broadcast_status(status)
        if not self.is_main_process:
            result = self._prepare_cached_datasets_local(
                dataset, validation_dataset, dataset_cfg, cache_spec=cache_spec)
        return result

    def _teacher_cache_spec(self, dataset, validation_dataset, dataset_cfg):
        config = self.cfg.get("teacher_cache", {})
        if getattr(dataset, "image_randomer", None) is not None:
            raise ValueError("Teacher caching requires dataset.image_randomer_config=null (fixed RGB inputs)")
        checkpoint = pathlib.Path(str(self.cfg.teacher_checkpoint)).resolve()
        stat = checkpoint.stat()
        token_dtype = str(config.get("dtype", "float32"))
        identity = {
            "version": CACHE_VERSION,
            "checkpoint": {"path": str(checkpoint), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            "state_key": self.teacher_state_key,
            "policy": OmegaConf.to_container(self.teacher_cfg.policy, resolve=True),
            "dataset": OmegaConf.to_container(dataset_cfg, resolve=True),
            "source": dataset_source_signature(dataset.dataset_path),
            "amp_enabled": self.amp_enabled, "amp_dtype": str(self.amp_dtype),
            "allow_tf32": bool(self.cfg.training.amp.allow_tf32),
            "capture_attention": self.attention_enabled,
            "attention_seed": int(self.attention_config.get("teacher_seed", 42)),
            "cache_batch_size": int(config["batch_size"]),
        }
        # Keep the identity of existing float32 caches; FP16 gets a distinct path.
        if token_dtype != "float32":
            identity["token_storage_dtype"] = token_dtype
        for name, source in (("train", dataset), ("validation", validation_dataset)):
            indices = np.ascontiguousarray(source.indices)
            identity[name] = {"samples": len(source), "indices_shape": list(indices.shape),
                              "indices_dtype": str(indices.dtype),
                              "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest()}
        root = pathlib.Path(str(config["directory"])).expanduser().resolve() / cache_key(identity)
        return root, identity

    @torch.no_grad()
    def _prepare_cached_datasets_local(self, dataset, validation_dataset, dataset_cfg, *, cache_spec=None):
        config = self.cfg.get("teacher_cache", {})
        if not config.get("enabled", False):
            return dataset, validation_dataset
        root, identity = cache_spec or self._teacher_cache_spec(dataset, validation_dataset, dataset_cfg)
        lifetime = getattr(self, "_teacher_cache_lifetime", None)
        if lifetime is not None:
            # Register before mkdir/build so a partial cache is also removed.
            lifetime.acquire(root)
        token_dtype = str(config.get("dtype", "float32"))
        root.mkdir(parents=True, exist_ok=True)
        prepared = []
        self.teacher_cache_paths = {}
        with (root / "build.lock").open("a") as lock:
            try:
                lock_mode = fcntl.LOCK_EX if self.is_main_process else fcntl.LOCK_SH
                fcntl.flock(lock.fileno(), lock_mode | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another process is building this teacher cache: {root}") from exc
            for name, source in (("train", dataset), ("validation", validation_dataset)):
                # Full-data teachers have no validation rows or cache targets.
                if len(source) == 0:
                    prepared.append(source)
                    continue
                cache = TeacherTargetCache(root / name, {**identity, "split": name}, len(source),
                                           token_dtype=token_dtype)
                if lifetime is not None:
                    lifetime.track_cache(cache)
                self.teacher_cache_paths[name] = str(cache.directory)
                log.info(f"Teacher cache {name}: {cache.completed}/{len(source)} samples; "
                         f"tokens={token_dtype}, attention=float32; {cache.directory}")
                start_count = cache.completed
                started = last_log = time.monotonic()
                if not cache.complete:
                    if not self.is_main_process:
                        raise RuntimeError(f"Rank 0 did not complete teacher cache: {cache.directory}")
                    worker_options = ({"multiprocessing_context": "spawn"}
                                      if self.distributed_context.enabled and int(config["num_workers"]) > 0
                                      else {})
                    loader = DataLoader(
                        Subset(source, range(cache.completed, len(source))),
                        batch_size=int(config["batch_size"]), num_workers=int(config["num_workers"]),
                        shuffle=False, pin_memory=self.device.type == "cuda",
                        generator=torch.Generator().manual_seed(int(self.cfg.training.get("seed", 42))),
                        **worker_options,
                    )
                    with self._progress(None, f"Decode + teacher cache [{name}]", total=len(source),
                                        initial=cache.completed, unit="sample") as progress:
                        for batch in self._iterate_loader(loader):
                            obs = self._obs(batch)
                            with self._autocast():
                                if self.attention_enabled:
                                    tokens, attention = teacher_attention_targets(
                                        self.teacher_policy, obs, percentiles=self.attention_config["percentiles"],
                                        seed=self.attention_config.get("teacher_seed", 42),
                                        progress_callback=self._attention_progress, return_patch_attention=True,
                                    )
                                else:
                                    tokens = self.teacher_policy._encode_context(obs, apply_curriculum_mask=False).detach()
                                    attention = {}
                            cache.append(tokens, attention)
                            progress.update(len(tokens))
                            now = time.monotonic()
                            if cache.completed == start_count + len(tokens) or cache.complete or now - last_log >= 30:
                                elapsed = now - started
                                rate = (cache.completed - start_count) / max(elapsed, 1e-6)
                                log.info(f"Teacher cache {name}: {cache.completed}/{len(source)}, "
                                         f"{rate:.2f} samples/s, elapsed={elapsed:.1f}s, "
                                         f"ETA={(len(source) - cache.completed) / max(rate, 1e-6):.1f}s; "
                                         f"token_shape={list(tokens.shape)}")
                                last_log = now
                prepared.append(CachedTeacherDataset(source, cache))
        if self.is_main_process:
            manifest_path = pathlib.Path(self.output_dir, "teacher_cache.json")
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps({"status": "complete", "splits": self.teacher_cache_paths}, indent=2) + "\n")
        if config.get("load_into_ram", False) and not config.get("only", False):
            cached = [item for item in prepared if isinstance(item, CachedTeacherDataset)]
            required = sum(item.cache.nbytes for item in cached)
            available = next(int(line.split()[1]) * 1024 for line in
                             pathlib.Path("/proc/meminfo").read_text().splitlines()
                             if line.startswith("MemAvailable:"))
            if required > available:
                raise RuntimeError(f"Teacher cache needs {required / 2**30:.1f} GiB RAM, "
                                   f"available={available / 2**30:.1f} GiB; use teacher_cache.load_into_ram=false")
            for item in cached:
                log.info(f"Loading teacher cache into RAM: {item.cache.directory}, "
                         f"{item.cache.nbytes / 2**30:.2f} GiB")
                item.cache.load_into_ram()
            log.info(f"Teacher targets resident in RAM: {required / 2**30:.2f} GiB; "
                     "DataLoader workers share the read-only arrays")
        log.info("Complete teacher cache ready; training/validation will read cached targets")
        return tuple(prepared)

    def _loader(self, dataset, config, *, validation=False):
        kwargs = OmegaConf.to_container(config, resolve=True)
        context = self.distributed_context
        if context.enabled:
            global_batch = int(kwargs["batch_size"])
            batch_is_global = bool(self.cfg.training.get("distributed", {}).get("batch_size_is_global", True))
            kwargs["batch_size"] = per_rank_batch_size(global_batch, context.world_size, batch_is_global)
            if validation:
                # No padding/repetition: each validation sample contributes once.
                limit = len(dataset)
                if self.max_val_batches is not None:
                    limit = min(limit, int(self.max_val_batches) * kwargs["batch_size"] * context.world_size)
                kwargs["sampler"] = range(context.rank, limit, context.world_size)
            else:
                kwargs["sampler"] = DistributedSampler(
                    dataset, num_replicas=context.world_size, rank=context.rank,
                    shuffle=bool(kwargs.get("shuffle", True)), seed=int(self.cfg.training.seed),
                )
            kwargs.pop("shuffle", None)
        if int(kwargs.get("num_workers", 0)) == 0:
            kwargs.pop("persistent_workers", None)
            kwargs.pop("multiprocessing_context", None)
        elif context.enabled:
            # NCCL state cannot be safely inherited by forked workers. Images
            # stay shared via tmpfs and teacher arrays reopen as read-only mmap.
            kwargs.setdefault("multiprocessing_context", "spawn")
        elif isinstance(dataset, CachedTeacherDataset) and dataset.cache.in_ram:
            kwargs["multiprocessing_context"] = "fork"
        return DataLoader(dataset, **kwargs)

    def _iterate_loader(self, loader):
        iterator = iter(loader)
        lifetime = getattr(self, "_teacher_cache_lifetime", None)
        return iterator if lifetime is None else lifetime.track_iterator(iterator)

    def _gpu_metrics(self):
        if self.device.type != "cuda":
            return {}
        return {
            "gpu_allocated_mib": torch.cuda.memory_allocated(self.device) / 2**20,
            "gpu_reserved_mib": torch.cuda.memory_reserved(self.device) / 2**20,
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(self.device) / 2**20,
        }

    @torch.no_grad()
    def _validate(self, loader):
        with self._validation_probes():
            return self._validate_batches(loader)

    def _validate_batches(self, loader):
        self.model.eval()
        total, count = 0.0, 0
        component_totals = {key: 0.0 for key in ("token_mse", "denoiser_mse", "demo_mse")}
        started = time.monotonic()
        batches = len(loader)
        if self.max_val_batches is not None:
            batches = min(batches, int(self.max_val_batches))
        log.info(f"Validation started: epoch={self.epoch}, batches={batches}; "
                 f"metric={self.metric_name}; auxiliary losses excluded; no attention capture")
        progress = self._progress(itertools.islice(self._iterate_loader(loader), batches),
                                  f"Validation epoch {self.epoch}", batches)
        for index, batch in enumerate(progress):
            loss = self._batch_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite validation MSE at step {self.global_step}")
            size = next(iter(batch["obs"].values())).shape[0]
            total += self._selection_mse(loss) * size
            for key in component_totals:
                component_totals[key] += getattr(self, "loss_metrics", {}).get(key, 0.0) * size
            count += size
            progress.set_postfix(mse=f"{total / count:.6f}", refresh=False)
            if index == 0 or (index + 1) % 10 == 0 or index + 1 == batches:
                elapsed = time.monotonic() - started
                log.info(f"Validating epoch={self.epoch}: batches={index + 1}/{batches}, "
                         f"samples={count}, mse={total / count:.8f}, elapsed={elapsed:.1f}s, "
                         f"ETA={(batches - index - 1) * elapsed / (index + 1):.1f}s")
        total, count = self._sum([total, count])
        count = int(count)
        if count == 0:
            raise RuntimeError("Validation split is empty")
        self.validation_samples = count
        self.validation_seconds = self._max_seconds(time.monotonic() - started)
        self.validation_samples_per_second = count / max(self.validation_seconds, 1e-6)
        totals = self._sum(list(component_totals.values()))
        self.validation_metrics = {key: value / count for key, value in zip(component_totals, totals)}
        return total / count

    def _summary(self, status):
        if not self.is_main_process:
            return
        summary = {
            "status": status, "global_step": self.global_step,
            "epoch": self.epoch, "max_epochs": self.num_epochs,
            "max_steps": self.max_steps, "stop_reason": self.stop_reason,
            "best_epoch": self.best_epoch, "best_step": self.best_step,
            "early_stopping": {"enabled": self.early_stopping_enabled,
                               "patience_epochs": self.patience, "min_delta": self.min_delta,
                               "epochs_without_improvement": self.epochs_without_improvement},
            "teacher_checkpoint": str(pathlib.Path(self.cfg.teacher_checkpoint).resolve()),
            "teacher_state_key": self.teacher_state_key, "cross_attn_query": "image",
            "wrench_preprocessing": OmegaConf.to_container(self.cfg.wrench_preprocessing, resolve=True),
            "render_net": OmegaConf.to_container(self.cfg.render_net, resolve=True),
            "validation_enabled": self.validation_enabled,
            "distillation": OmegaConf.to_container(OmegaConf.create(self.distillation_config), resolve=True),
            "objective": self.objective,
            "validation_metric": f"{self.metric_name} (excludes auxiliary losses)" if self.validation_enabled else None,
            "teacher_cache": getattr(self, "teacher_cache_paths", None),
            "world_size": self.distributed_context.world_size,
            "teacher_trainable_parameters": sum(p.numel() for p in self.teacher_policy.parameters() if p.requires_grad),
            "student_trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "initial_train_mse": self.initial_train_mse, "initial_val_mse": self.initial_val_mse,
            "latest_val_mse": self.latest_val_loss,
            "best_val_mse": self.best_val_loss if self.validation_enabled else None,
            "validation_samples": getattr(self, "validation_samples", 0),
            "best_checkpoint": str(self.get_checkpoint_path("best").resolve()) if self.validation_enabled else None,
            "latest_checkpoint": str(self.get_checkpoint_path("latest").resolve()),
            "metrics_png": str(pathlib.Path(self.output_dir, "training_metrics.png").resolve()),
            "elapsed_seconds": time.monotonic() - self.started_at,
            **(self.shape_info or {}), **self._gpu_metrics(),
        }
        pathlib.Path(self.output_dir, "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def _validate_and_save(self, loader, logger):
        if loader is None:
            return False
        self.latest_val_loss = self._validate(loader)
        improved = self.latest_val_loss < self.best_val_loss
        if self.latest_val_loss < self.early_stopping_best_loss - self.min_delta:
            self.early_stopping_best_loss = self.latest_val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        if improved:
            self.best_val_loss = self.latest_val_loss
            self.best_epoch = self.epoch
            self.best_step = self.global_step
            if self.is_main_process:
                self.save_checkpoint(tag="best", use_thread=False)
            log.info(f"New best checkpoint: {self.get_checkpoint_path('best')} "
                     f"(epoch={self.epoch}, val_mse={self.best_val_loss:.8f})")
        metrics = {"global_step": self.global_step, "epoch": self.epoch,
                   "objective": self.objective,
                   "val_mse": self.latest_val_loss, "best_val_mse": self.best_val_loss,
                   "epochs_without_improvement": self.epochs_without_improvement,
                   "val_seconds": getattr(self, "validation_seconds", 0.0),
                   "val_samples_per_second": getattr(self, "validation_samples_per_second", 0.0),
                   "elapsed_seconds": time.monotonic() - self.started_at, **self._gpu_metrics()}
        metrics.update({f"val_{key}": value for key, value in getattr(self, "validation_metrics", {}).items()})
        logger.log(metrics)
        logger.save_plot()
        log.info(f"epoch={self.epoch}/{self.num_epochs} step={self.global_step} "
              f"val_mse={self.latest_val_loss:.8f} best={self.best_val_loss:.8f} "
              f"samples={self.validation_samples} stale_epochs={self.epochs_without_improvement}/{self.patience}")
        self._summary("running")
        return self.early_stopping_enabled and self.epochs_without_improvement >= self.patience

    def run(self):
        cache_config = self.cfg.get("teacher_cache", {})
        with TeacherCacheLifecycle(cleanup_on_exit=cache_config.get("cleanup_on_exit", True)) as lifetime:
            self._teacher_cache_lifetime = lifetime
            try:
                return self._run()
            finally:
                self._teacher_cache_lifetime = None

    def _run(self):
        output = pathlib.Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if self.is_main_process:
            OmegaConf.save(self.cfg, output / "config.yaml", resolve=True)
            OmegaConf.save(self.teacher_cfg, output / "teacher_config.yaml", resolve=True)
            # Continuing in a new directory preserves the previously best model.
            if self.resume_from:
                previous_best = pathlib.Path(self.resume_from).parent / "best.ckpt"
                destination = self.get_checkpoint_path("best")
                if previous_best.is_file() and not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(previous_best, destination)
        dataset_cfg = copy.deepcopy(self.teacher_cfg.task.dataset)
        # Resolve teacher-root interpolations before applying distillation overrides.
        dataset_cfg = OmegaConf.create(OmegaConf.to_container(dataset_cfg, resolve=True))
        with open_dict(dataset_cfg):
            dataset_cfg.update(OmegaConf.to_container(self.cfg.dataset, resolve=True))
        # Keep the original dataset config as the cache identity. Rank-local
        # decoder devices do not change supervision, so all ranks reuse the
        # existing single-GPU teacher cache rather than creating new copies.
        runtime_dataset_cfg = copy.deepcopy(dataset_cfg)
        if self.distributed_context.enabled and "video_decode_device" in runtime_dataset_cfg:
            runtime_dataset_cfg.video_decode_device = str(self.device) if self.device.type == "cuda" else "auto"
        dataset = self._instantiate_dataset(runtime_dataset_cfg)
        validation_dataset = dataset.get_validation_dataset()
        if not len(dataset):
            dataset.lerobot_dataset.close()
            raise RuntimeError("Distillation needs a nonempty training split")
        self.validation_enabled = len(validation_dataset) > 0
        if not self.validation_enabled:
            self.early_stopping_enabled = False
            log.info("No validation split: train on all episodes until the epoch/step limit; "
                     "validation, early stopping and best checkpoint selection are disabled. "
                     "Save latest.ckpt after each epoch.")
        log.info(f"Dataset: train={len(dataset)}, validation={len(validation_dataset)}; "
              "normalizer=teacher checkpoint (not refitted)")
        try:
            train_dataset, validation_dataset = self._prepare_cached_datasets(dataset, validation_dataset, dataset_cfg)
            cache_config = self.cfg.get("teacher_cache", {})
            if cache_config.get("enabled", False) and cache_config.get("only", False):
                log.info("teacher_cache.only=true: cache prepared, exiting before training")
                if self.distributed_context.enabled:
                    dist.barrier()
                return
            train_loader = self._loader(train_dataset, self.cfg.dataloader)
            val_loader = (self._loader(validation_dataset, self.cfg.val_dataloader, validation=True)
                          if self.validation_enabled else None)
            self._setup_acceleration()
            self._setup_distributed_model()
            runtime_seed = int(self.cfg.training.seed) + self.distributed_context.rank
            random.seed(runtime_seed)
            np.random.seed(runtime_seed)
            torch.manual_seed(runtime_seed)
            log.info("Training replicas=%d; per-rank batch=%d; global batch=%d; "
                     "teacher tokens=%s; decoded image cache=%s",
                     self.distributed_context.world_size, train_loader.batch_size,
                     train_loader.batch_size * self.distributed_context.world_size,
                     "SSD mmap" if not cache_config.get("load_into_ram", False) else "RAM",
                     getattr(dataset, "load_result_cache_path", None) or "process RAM")
            logging_config = self.cfg.get("logging", {})
            logger_context = (RenderNetTrainingLogger(output, logging_config.get("plot_every_steps", 100))
                              if self.is_main_process else contextlib.nullcontext(_NullTrainingLogger()))
            with logger_context as logger:
                if self.is_main_process:
                    log.info(f"Local metrics: {logger.path}; PNG: {output / 'training_metrics.png'}")
                if self.resume_from is None:
                    self.model.eval()
                    log.info(f"Initial evaluation started: one training batch "
                             f"(batch_size={self.cfg.dataloader.batch_size}); validation={self.validation_enabled}")
                    with torch.no_grad(), self._validation_probes():
                        initial_batch = next(self._iterate_loader(train_loader))
                        initial_loss = self._batch_loss(initial_batch)
                        size = next(iter(initial_batch["obs"].values())).shape[0]
                        total, count = self._sum([self._selection_mse(initial_loss) * size, size])
                        self.initial_train_mse = total / count
                    log.info(f"Initial training batch complete: {self.metric_name}={self.initial_train_mse:.8f}")
                    logger.log({"global_step": 0, "epoch": 0, "initial_train_mse": self.initial_train_mse,
                                "objective": self.objective,
                                "lr": self.optimizer.param_groups[0]["lr"],
                                "elapsed_seconds": time.monotonic() - self.started_at, **self._gpu_metrics()})
                    logger.save_plot()
                    if self.validation_enabled:
                        self.initial_val_mse = self._validate(val_loader)
                        self.latest_val_loss = self.initial_val_mse
                        self.best_val_loss = self.initial_val_mse
                        self.early_stopping_best_loss = self.initial_val_mse
                        if self.is_main_process:
                            self.save_checkpoint(tag="best", use_thread=False)
                        log.info(f"Initial train_mse={self.initial_train_mse:.8f}, val_mse={self.initial_val_mse:.8f}")
                        logger.log({"global_step": 0, "epoch": 0, "initial_train_mse": self.initial_train_mse,
                                    "objective": self.objective,
                                    "initial_val_mse": self.initial_val_mse,
                                    "val_seconds": getattr(self, "validation_seconds", 0.0),
                                    "val_samples_per_second": getattr(self, "validation_samples_per_second", 0.0),
                                    "elapsed_seconds": time.monotonic() - self.started_at, **self._gpu_metrics()})
                        logger.save_plot()
                else:
                    log.info("Skipping initial evaluation; restored checkpoint state will be continued")
                self._summary("running")
                for epoch_index in range(self.epoch, self.num_epochs):
                    if self.max_steps is not None and self.global_step >= int(self.max_steps):
                        self.stop_reason = "max_steps"
                        break
                    if isinstance(train_loader.sampler, DistributedSampler):
                        train_loader.sampler.set_epoch(epoch_index)
                    epoch_number = epoch_index + 1
                    epoch_total, epoch_samples = 0.0, 0
                    epoch_loss_total = epoch_mask_total = 0.0
                    epoch_components = {key: 0.0 for key in ("token_mse", "denoiser_mse", "demo_mse")}
                    epoch_started = previous_end = time.monotonic()
                    batches = len(train_loader)
                    if self.max_steps is not None:
                        batches = min(batches, int(self.max_steps) - self.global_step)
                    progress = self._progress(itertools.islice(self._iterate_loader(train_loader), batches),
                                              f"Training epoch {epoch_number}/{self.num_epochs}", batches)
                    log.info(f"Training epoch={epoch_number}/{self.num_epochs} started: batches={batches}, "
                             f"per_rank_batch_size={train_loader.batch_size}, lr={self.optimizer.param_groups[0]['lr']:.3g}")
                    self.model.train()
                    for batch_index, batch in enumerate(progress):
                        step_started = time.monotonic()
                        data_seconds = step_started - previous_end
                        self.model.train()
                        self.optimizer.zero_grad(set_to_none=True)
                        loss = self._batch_loss(batch)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"Non-finite MSE at step {self.global_step}")
                        self.grad_scaler.scale(loss).backward()
                        self.grad_scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), float(self.cfg.training.max_grad_norm), error_if_nonfinite=True,
                        )
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                        grad_norm_value = float(grad_norm)
                        step_seconds = time.monotonic() - step_started
                        self.global_step += 1
                        size = next(iter(batch["obs"].values())).shape[0]
                        primary_total, loss_total, mask_total, global_size, *component_totals = self._sum([
                            self._selection_mse(loss) * size, float(loss.detach()) * size,
                            self.loss_metrics.get("mask_mse", 0.0) * size, size,
                            *[self.loss_metrics.get(key, 0.0) * size for key in epoch_components],
                        ])
                        epoch_total += primary_total
                        epoch_samples += global_size
                        epoch_loss_total += loss_total
                        epoch_mask_total += mask_total
                        for key, value in zip(epoch_components, component_totals):
                            epoch_components[key] += value
                        metrics = {"global_step": self.global_step, "epoch": epoch_number,
                                   "objective": self.objective,
                                   "train_mse": primary_total / global_size, "train_loss": loss_total / global_size,
                                   "train_mask_mse": mask_total / global_size,
                                   "grad_norm": grad_norm_value, "lr": self.optimizer.param_groups[0]["lr"],
                                   "batch_in_epoch": batch_index + 1, "batches_in_epoch": batches,
                                   "data_seconds": data_seconds, "step_seconds": step_seconds,
                                   "samples_per_second": global_size / max(data_seconds + step_seconds, 1e-6),
                                   "elapsed_seconds": time.monotonic() - self.started_at, **self._gpu_metrics()}
                        metrics.update({f"train_{key}": value / global_size
                                        for key, value in zip(epoch_components, component_totals)})
                        logger.log(metrics)
                        progress.set_postfix(mse=f"{metrics['train_mse']:.6f}",
                                             mean=f"{epoch_total / epoch_samples:.6f}",
                                             mask=f"{metrics['train_mask_mse']:.5f}", lr=f"{metrics['lr']:.2g}",
                                             refresh=False)
                        if (self.global_step == 1 or self.global_step % int(self.cfg.training.log_every) == 0
                                or batch_index + 1 == batches):
                            elapsed = time.monotonic() - epoch_started
                            log.info(f"epoch={epoch_number}/{self.num_epochs} step={self.global_step} "
                                  f"batch={batch_index + 1}/{batches} "
                                  f"train_mse={metrics['train_mse']:.8f} "
                                  f"epoch_mean_mse={epoch_total / epoch_samples:.8f} "
                                  f"mask_mse={metrics['train_mask_mse']:.8f} loss={metrics['train_loss']:.8f} "
                                  f"grad_norm={metrics['grad_norm']:.5f} "
                                  f"lr={metrics['lr']:.3g} samples/s={metrics['samples_per_second']:.1f} "
                                  f"data={data_seconds:.3f}s step_time={step_seconds:.3f}s "
                                  f"elapsed={elapsed:.1f}s ETA={(batches - batch_index - 1) * elapsed / (batch_index + 1):.1f}s "
                                  f"GPU_peak_MiB={metrics.get('gpu_peak_allocated_mib', 0):.0f}")
                        previous_end = time.monotonic()
                        if self.max_steps is not None and self.global_step >= int(self.max_steps):
                            break
                    self.epoch = epoch_number
                    epoch_seconds = self._max_seconds(time.monotonic() - epoch_started)
                    logger.log({"global_step": self.global_step, "epoch": self.epoch,
                                "objective": self.objective,
                                "epoch_train_mse": epoch_total / epoch_samples,
                                "epoch_train_loss": epoch_loss_total / epoch_samples,
                                "epoch_train_mask_mse": epoch_mask_total / epoch_samples,
                                "epoch_samples": epoch_samples, "epoch_seconds": epoch_seconds,
                                "epoch_samples_per_second": epoch_samples / max(epoch_seconds, 1e-6),
                                **{f"epoch_train_{key}": value / epoch_samples
                                   for key, value in epoch_components.items()}})
                    log.info(f"Training epoch={self.epoch} complete: samples={epoch_samples}, "
                             f"mean_mse={epoch_total / epoch_samples:.8f}, "
                             f"mean_loss={epoch_loss_total / epoch_samples:.8f}, elapsed={epoch_seconds:.1f}s")
                    early_stop = self._validate_and_save(val_loader, logger)
                    if self.max_steps is not None and self.global_step >= int(self.max_steps):
                        self.stop_reason = "max_steps"
                    elif early_stop:
                        self.stop_reason = "early_stopping"
                    elif self.epoch >= self.num_epochs:
                        self.stop_reason = "max_epochs"
                    if self.is_main_process:
                        self.save_checkpoint(tag="latest", use_thread=False)
                    if self.stop_reason is not None:
                        break
                if any(p.grad is not None or p.requires_grad for p in self.teacher_policy.parameters()):
                    raise RuntimeError("Teacher unexpectedly acquired trainable parameters or gradients")
                self._summary("early_stopped" if self.stop_reason == "early_stopping" else "completed")
                log.info("Training stopped: %s; epoch=%d, step=%d", self.stop_reason, self.epoch, self.global_step)
                if self.validation_enabled:
                    log.info("Best epoch=%d, step=%d, val_mse=%.8f", self.best_epoch, self.best_step, self.best_val_loss)
                log.info("Checkpoint: %s", self.get_checkpoint_path("best" if self.validation_enabled else "latest"))
            if self.distributed_context.enabled:
                # Keep the shared image cache alive until every rank is finished.
                dist.barrier()
        finally:
            lifetime = getattr(self, "_teacher_cache_lifetime", None)
            if lifetime is not None:
                lifetime.close_readers()
            dataset.lerobot_dataset.close()
