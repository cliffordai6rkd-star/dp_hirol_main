"""Train only RenderNet using a frozen teacher's context or denoising outputs."""

import contextlib
import copy
import dill
import fcntl
import hashlib
import itertools
import json
import logging
import math
import pathlib
import random
import shutil
import sys
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
from tqdm.contrib.logging import logging_redirect_tqdm

from diffusion_policy.common.render_net_training_logger import RenderNetTrainingLogger
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.teacher_cache_lifecycle import TeacherCacheLifecycle
from diffusion_policy.dataset.render_net_teacher_cache import (
    CACHE_VERSION, CachedTeacherDataset, TeacherTargetCache, cache_key, dataset_source_signature,
)
from diffusion_policy.dataset.image_result_cache import use_shared_ram_result_cache
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.vision.render_net import RenderNet, build_render_net
from diffusion_policy.model.vision.render_net_attention import (
    attention_regions, context_effect_patch_weights, context_effect_regions,
    percentile_region, teacher_attention_targets, teacher_context_effect_targets,
    validate_percentiles,
)
from diffusion_policy.model.vision.wrench_preprocessing import (
    WRENCH_PREPROCESSING_VERSION, WrenchPreprocessor,
)
from diffusion_policy.scripts.visualize_cross_attention_episode import (
    instantiate_policy, load_checkpoint_payload, patch_grid,
)
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.workspace.train_force_aware_diffusion_workspace import (
    DistributedContext, per_rank_batch_size, resolve_amp_dtype,
)


log = logging.getLogger(__name__)


REGION_SOURCE_ALIASES = {
    # Canonical names exposed in the config.
    "noise_attention": "noise_attention",
    "context": "context",
    # Names used by earlier runs; keep them loadable for old checkpoints.
    "attention": "noise_attention",
    "context_effect": "context",
}


def normalize_region_source(source):
    source = str(source).strip().lower()
    try:
        return REGION_SOURCE_ALIASES[source]
    except KeyError as exc:
        choices = ", ".join(("noise_attention", "context"))
        raise ValueError(
            f"high_attention_render.source must be one of: {choices}"
        ) from exc


class _NullTrainingLogger:
    def log(self, data):
        pass

    def save_plot(self):
        pass


def student_context_tokens(teacher, render_net, obs, *, return_aux=False,
                           return_contact=False, image_context_encoder=None):
    """Only RGB and normalized wrench enter RenderNet; no teacher force/fusion path."""
    images, wrench, contact, future_contact = teacher.prepare_context_inputs(obs)
    # The teacher's preprocessing already applies the configured hysteresis or
    # token-level future-contact rule.  Reuse that exact per-observation gate
    # in the student renderer; a missing future phase falls back to current
    # contact, matching the teacher preprocessing contract.
    contact_gate = future_contact if future_contact is not None else contact
    renderer_impl = getattr(render_net, "module", render_net)
    if isinstance(renderer_impl, RenderNet):
        rendered, masks = render_net(
            images, wrench, future_contact=contact_gate, return_aux=True,
        )
    else:
        # Lightweight test/render adapters from older checkpoints may expose
        # only the original two-input signature.
        rendered, masks = render_net(images, wrench, return_aux=True)
    encode = teacher.obs_encoder.encode_image_context if image_context_encoder is None else image_context_encoder
    tokens = encode(rendered, input_grad=True)
    if return_aux and return_contact:
        return tokens, masks, contact_gate
    if return_aux:
        return tokens, masks
    if return_contact:
        return tokens, contact_gate
    return tokens


class TrainDistillRenderNetWorkspace(BaseWorkspace):
    exclude_keys = ("teacher_policy", "_training_model")
    include_keys = ("global_step", "epoch", "best_val_loss", "initial_train_mse",
                    "initial_val_mse", "shape_info", "teacher_state_key", "latest_val_loss",
                    "best_epoch", "best_step", "epochs_without_improvement",
                    "early_stopping_best_loss", "stop_reason", "training_stage", "stage_step",
                    "stage_epoch", "stage_transition_reason", "render_stages_enabled",
                    "intensity_epochs", "resume_architecture_report")

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
        backbone_config = dict(render_config.pop("backbone", {}))
        encoder = self.teacher_policy.obs_encoder
        render_config.update(
            image_keys=list(encoder.image_keys),
            wrench_history_steps=int(encoder.wrench_history_steps),
            max_obs_steps=int(encoder.max_obs_steps),
        )
        if render_config.get("initialize_wrench_from_teacher", True):
            teacher_force_encoder = getattr(encoder, "force_encoder", None)
            if teacher_force_encoder is None:
                # Lightweight adapters used for cache-only/compatibility runs
                # may expose image encoding without the teacher wrench path.
                # The active force-aware teacher always has this module.
                log.warning("Teacher has no force_encoder; disabling RenderNet wrench initialization")
                render_config["initialize_wrench_from_teacher"] = False
            elif not isinstance(teacher_force_encoder, torch.nn.LSTM):
                raise ValueError("RenderNet teacher wrench initialization requires an LSTM teacher")
            else:
                render_config.update(
                    conditioning_dim=int(encoder.n_emb),
                    wrench_layers=int(teacher_force_encoder.num_layers),
                    wrench_dropout=float(teacher_force_encoder.dropout),
                )
        render_config.update(backbone_config)
        with open_dict(self.cfg):
            self.cfg.render_net = OmegaConf.create(render_config)
        self.model = build_render_net(wrench_dim, render_config).to(self.device)
        if (isinstance(self.model, RenderNet) and self.model.initialize_wrench_from_teacher
                and hasattr(self.teacher_policy.obs_encoder, "force_encoder")):
            self.model.initialize_wrench_encoder(self.teacher_policy.obs_encoder)
        self.ema_model = self.ema = None
        if cfg.training.get("use_ema", False):
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.eval().requires_grad_(False)
            self.ema = EMAModel(self.ema_model, **dict(cfg.get("ema", {})))
        self.attention_config = cfg.render_net.get("high_attention_render", {})
        self.attention_enabled = bool(self.attention_config.get("enabled", False))
        self.intensity_loss_weight = float(
            self.attention_config.get("intensity_loss_weight", 1.0)
        )
        self.context_patch_loss_weight = float(
            self.attention_config.get("context_patch_loss_weight", 1.0)
        )
        self.smoothness_loss_weight = float(
            self.attention_config.get("smoothness_loss_weight", 0.02)
        )
        self.region_source = normalize_region_source(
            self.attention_config.get("source", "noise_attention")
        )
        if self.attention_enabled:
            selection = self.attention_config.get("selection", "soft")
            if self.region_source == "context":
                if self.attention_config.get("baseline", "raw_zero") != "raw_zero":
                    raise ValueError("context supervision requires baseline=raw_zero")
                if selection == "soft":
                    normalization = self.attention_config.get("normalization")
                    if normalization not in (None, "image_minmax"):
                        raise ValueError(f"Unknown context normalization: {normalization!r}")
                    if normalization is None:
                        validate_percentiles(self.attention_config.get("percentiles", [80.0, 99.0]))
                        strength_scale = self.attention_config.get("strength_scale")
                        if strength_scale is not None and (
                                not math.isfinite(float(strength_scale)) or float(strength_scale) <= 0):
                            raise ValueError("context strength_scale must be positive and finite")
                elif selection == "binary_top_percentile":
                    percentile = float(self.attention_config.get("percentile", 90.0))
                    if not 0 <= percentile < 100:
                        raise ValueError("context binary selection requires 0 <= percentile < 100")
                else:
                    raise ValueError(
                        "high_attention_render.selection must be soft or binary_top_percentile"
                    )
            elif self.region_source == "noise_attention":
                validate_percentiles(self.attention_config["percentiles"])
            for name, value in (("intensity_loss_weight", self.intensity_loss_weight),
                                ("context_patch_loss_weight", self.context_patch_loss_weight),
                                ("smoothness_loss_weight", self.smoothness_loss_weight)):
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"high_attention_render.{name} must be finite and nonnegative")
        self.loss_metrics = {}
        self._student_image_encoder = None
        self._training_model = None
        stage_config = cfg.training.get("render_stages", {})
        # A staged run needs a supervised gate.  If high-attention rendering
        # is disabled, there is no intensity objective to warm up, so retain a
        # single joint RGB training stage.
        self.render_stages_enabled = (
            bool(stage_config.get("enabled", True)) and self.attention_enabled
            and hasattr(self.model, "set_training_stage")
        )
        self.intensity_epochs = int(stage_config.get(
            "intensity_epochs", cfg.training.get("intensity_warmup_epochs", 2),
        ))
        if self.intensity_epochs < 0:
            raise ValueError("training.render_stages.intensity_epochs must be nonnegative")
        self.training_stage = (
            "intensity" if self.render_stages_enabled and self.intensity_epochs > 0
            else ("rgb" if self.render_stages_enabled else "joint")
        )
        self.stage_step = 0
        self.stage_epoch = 0
        self.stage_transition_reason = "initial_config"
        self.resume_architecture_report = None
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **OmegaConf.to_container(cfg.optimizer))
        self.lr_scheduler = None
        scheduler_config = cfg.training.get("lr_scheduler", {})
        if scheduler_config.get("enabled", False):
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=float(scheduler_config.get("factor", 0.5)),
                patience=int(scheduler_config.get("patience", 2)),
                threshold=float(scheduler_config.get("threshold", 1e-4)), threshold_mode="abs",
                min_lr=float(scheduler_config.get("min_lr", 1e-5)),
            )
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
            excluded = []
            if self.lr_scheduler is None:
                excluded.append("lr_scheduler")
            if self.ema is None:
                excluded.extend(("ema", "ema_model"))
            try:
                restored = torch.load(resume_path.open("rb"), pickle_module=dill,
                                      map_location="cpu")
                saved_cfg = restored.get("cfg", {})
                saved_render = saved_cfg.get("render_net", {})
                saved_backbone = saved_render.get("backbone", {})
                saved_base = saved_backbone.get("base_channels", saved_render.get("base_channels"))
                current_base = cfg.render_net.get("backbone", {}).get(
                    "base_channels", cfg.render_net.get("base_channels")
                )
                mismatches = []
                saved_state = restored.get("state_dicts", {}).get("model", {})
                current_state = self.model.state_dict()
                for key, value in saved_state.items():
                    target = current_state.get(key)
                    if target is None:
                        mismatches.append(f"{key}: checkpoint-only")
                    elif tuple(value.shape) != tuple(target.shape):
                        mismatches.append(
                            f"{key}: checkpoint{tuple(value.shape)} != current{tuple(target.shape)}"
                        )
                for key in current_state:
                    if key not in saved_state:
                        mismatches.append(f"{key}: current-only")
                self.resume_architecture_report = {
                    "status": "compatible" if not mismatches else "incompatible",
                    "saved_base_channels": saved_base,
                    "current_base_channels": current_base,
                    "mismatched_keys": mismatches[:32],
                    "mismatch_count": len(mismatches),
                }
                if mismatches:
                    raise RuntimeError(
                        "RenderNet checkpoint architecture mismatch; refusing to silently skip "
                        f"{len(mismatches)} parameters (base_channels checkpoint={saved_base}, "
                        f"current={current_base}). Examples: " + "; ".join(mismatches[:8])
                    )
                self.load_payload(restored, include_keys=self.include_keys, exclude_keys=excluded)
            except (RuntimeError, KeyError, ValueError) as exc:
                raise RuntimeError(
                    "RenderNet checkpoint could not be resumed with the current architecture. "
                    "Shape-mismatched or missing parameters are rejected explicitly; "
                    f"Original error: {exc}"
                ) from exc
            saved_objective = restored["cfg"].get("distillation", {}).get("objective", "token_mse")
            if saved_objective != self.objective:
                raise ValueError("Cannot resume with a different distillation objective: "
                                 f"checkpoint={saved_objective}, configured={self.objective}. "
                                 "Start a new run so checkpoint selection metrics remain comparable.")
            if self.ema_model is not None and "ema_model" not in restored["state_dicts"]:
                self.ema_model.load_state_dict(self.model.state_dict())
                log.info("Checkpoint has no student EMA: initialize from restored RenderNet; "
                         "EMA warmup starts at zero")
            if self.lr_scheduler is not None and "lr_scheduler" not in restored["state_dicts"]:
                if math.isfinite(self.best_val_loss):
                    self.lr_scheduler.step(self.best_val_loss)
                log.info("Checkpoint has no LR scheduler: retain optimizer LR, use saved best "
                         "validation as baseline, and start plateau counting at zero")
            if "training_stage" not in restored.get("pickles", {}):
                self.training_stage = (
                    "intensity" if self.render_stages_enabled and self.epoch < self.intensity_epochs
                    else "rgb"
                )
                self.stage_transition_reason = "legacy_checkpoint_epoch_inference"
            if hasattr(self.model, "set_training_stage"):
                self.model.set_training_stage(self.training_stage)
            if getattr(self, "ema", None) is not None:
                # A resumed checkpoint may have been written before the stage
                # transition fix.  Make the frozen gate snapshot identical in
                # online and EMA models before validation or another update.
                self.ema.sync_frozen_parameters(self.model)
            self.resume_from = str(resume_path)
            # A checkpoint may have been saved immediately before an orderly
            # stop; an explicit resume should always re-enter the epoch loop.
            self.stop_reason = None
            log.info("Resuming RenderNet from %s: epoch=%d step=%d best_val_mse=%.8f",
                     resume_path, self.epoch, self.global_step, self.best_val_loss)
        self.num_epochs = int(cfg.training.num_epochs)
        self.intensity_warmup_epochs = int(cfg.training.get("intensity_warmup_epochs", 2))
        self.max_steps = cfg.training.max_steps
        self.max_val_batches = cfg.training.max_val_batches
        self.early_stopping_enabled = bool(cfg.training.early_stopping.enabled)
        self.patience = int(cfg.training.early_stopping.patience)
        self.min_delta = float(cfg.training.early_stopping.min_delta)
        if self.num_epochs < 1 or self.intensity_warmup_epochs < 0 or any(
                value is not None and int(value) < 1
                for value in (self.max_steps, self.max_val_batches)):
            raise ValueError("num_epochs must be positive; intensity_warmup_epochs nonnegative; "
                             "max_steps/max_val_batches must be positive or null")
        if self.render_stages_enabled and self.intensity_epochs > self.num_epochs:
            raise ValueError("training.render_stages.intensity_epochs cannot exceed training.num_epochs")
        if not resume_path:
            if hasattr(self.model, "set_training_stage"):
                self.model.set_training_stage(self.training_stage)
        if getattr(self, "ema", None) is not None:
            self.ema.sync_frozen_parameters(self.model)
        if self.patience < 1 or not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("early_stopping.patience must be positive; min_delta must be finite and nonnegative")
        log.info(f"Teacher: {checkpoint} ({self.teacher_state_key}), cross_attn_query=image")
        log.info(f"Trainable parameters: teacher=0, RenderNet={sum(p.numel() for p in self.model.parameters()):,}")
        log.info("Student validation/inference weights=%s; EMA updates=%d",
                 self.evaluation_state_key, self.ema.optimization_step if self.ema is not None else 0)
        if self.lr_scheduler is not None:
            log.info("LR scheduler=ReduceLROnPlateau: monitor validation %s, factor=%g, "
                     "reduce after %d non-improving epochs, threshold=%g, min_lr=%g",
                     self.metric_name, self.lr_scheduler.factor, self.lr_scheduler.patience + 1,
                     self.lr_scheduler.threshold, self.lr_scheduler.min_lrs[0])
        log.info(f"Image encoder={self.model.image_encoder_name}; wrench encoder={self.model.wrench_encoder_name}; "
                 f"high-attention rendering={self.attention_enabled}")
        if isinstance(self.model, RenderNet):
            log.info("RenderNet: full LSTM sequence [B*To,%d,%d], image self-attn=%d, cross-attn=%d, "
                     "decoder=%d, conv_block=%s, rendering_mode=%s, cameras=%s; "
                     "teacher wrench initialization=%s",
                     self.model.wrench_history_steps, self.model.wrench_encoder.hidden_size,
                     len(self.model.backbone.image_self_attention),
                     len(self.model.backbone.cross_attention),
                     self.model.backbone.decoder_layers,
                     self.model.conv_block, getattr(self.model, "rendering_mode", "residual"),
                     self.model.image_keys, self.model.initialize_wrench_from_teacher)
        if self.attention_enabled:
            if self.region_source == "context":
                if self.attention_config.get("selection", "soft") == "soft":
                    scale = self.attention_config.get("strength_scale")
                    if self.attention_config.get("normalization") == "image_minmax":
                        mapping = "per-image strength=(d-min)/(max-min); constant maps are zero"
                    else:
                        mapping = (f"strength=d/(d+{float(scale):g})" if scale is not None else
                                   f"legacy percentiles={list(self.attention_config.get('percentiles', [80., 99.]))}")
                    log.info("Teacher context: FP32 L2(Z(W)-Z(raw_zero)); %s; zero effect stays zero; "
                             "student predicts the field during training and inference", mapping)
                else:
                    log.info("Teacher region: FP32 L2(Z(W)-Z(raw_zero)), binary top percentile=%g; "
                             "student hard patch gate at inference, prediction threshold=%g; no teacher at inference",
                             self.attention_config["percentile"],
                             self.attention_config.get("prediction_threshold", 0.5))
            else:
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
                 f"render stages={self.render_stages_enabled} "
                 f"(intensity={self.intensity_epochs}, current={self.training_stage}); "
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

    @property
    def evaluation_state_key(self):
        return "ema_model" if getattr(self, "ema_model", None) is not None else "model"

    def _step_lr_scheduler(self, validation_mse):
        if self.lr_scheduler is None:
            return
        previous_lr = self.optimizer.param_groups[0]["lr"]
        self.lr_scheduler.step(validation_mse)
        current_lr = self.optimizer.param_groups[0]["lr"]
        if current_lr < previous_lr and self.is_main_process:
            log.info("Learning rate reduced: %.6g -> %.6g at epoch %d; validation %s=%.8f",
                     previous_lr, current_lr, self.epoch, self.metric_name, validation_mse)

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
                raise ValueError("Compiling requires the active RenderNet backbone")
            self.model.compile_backbone(**options)
        if config.get("student_dino", True):
            self._student_image_encoder = torch.compile(
                self.teacher_policy.obs_encoder.encode_image_context, **options,
            )
        log.info("torch.compile enabled for training: render_backbone=%s, student_dino=%s, "
                 "mode=%s; first training batch includes compilation; validation uses eager execution",
                 config.get("render_backbone", True), config.get("student_dino", True), options["mode"])

    def _progress_enabled(self):
        config = getattr(self, "cfg", {}).get("logging", {})
        return (self.is_main_process and bool(config.get("progress_bar", True))
                and (config.get("progress_bar_non_tty", True) or sys.stderr.isatty()))

    def _progress(self, iterable, description, total=None, *, initial=0, unit="batch"):
        config = getattr(self, "cfg", {}).get("logging", {})
        return tqdm(iterable, desc=description, total=total, initial=initial, unit=unit, dynamic_ncols=True,
                    mininterval=float(config.get("tqdm_interval_sec", 1.0)),
                    leave=bool(config.get("tqdm_leave", False)),
                    disable=not self._progress_enabled())

    def _status_interval(self):
        return max(1.0, float(getattr(self, "cfg", {}).get("logging", {}).get("status_interval_sec", 60.0)))

    def _attention_progress(self, completed, total, elapsed):
        if completed == 0:
            self.attention_progress_last_seconds = 0.0
            return
        if elapsed - self.attention_progress_last_seconds >= self._status_interval():
            if not self._progress_enabled() and self.is_main_process:
                log.info(f"Teacher attention: samples={completed}/{total}, elapsed={elapsed:.1f}s, "
                         f"ETA={(total - completed) * elapsed / completed:.1f}s")
            self.attention_progress_last_seconds = elapsed

    def _region_masks(self, patches, obs):
        source = getattr(
            self, "region_source",
            normalize_region_source(self.attention_config.get("source", "noise_attention")),
        )
        if source == "context":
            return context_effect_regions(patches, obs, self.teacher_policy.obs_encoder.backbone,
                                          percentile=self.attention_config.get("percentile", 90.0),
                                          percentiles=self.attention_config.get(
                                              "percentiles", [80.0, 99.0]),
                                          selection=self.attention_config.get("selection", "soft"),
                                          strength_scale=self.attention_config.get("strength_scale"),
                                          normalization=self.attention_config.get("normalization"))
        return attention_regions(patches, obs, self.teacher_policy.obs_encoder.backbone,
                                 self.attention_config["percentiles"])

    def _region_targets(self, obs, *, return_patch_attention=False):
        source = getattr(
            self, "region_source",
            normalize_region_source(self.attention_config.get("source", "noise_attention")),
        )
        if source == "context":
            tokens, patches = teacher_context_effect_targets(
                self.teacher_policy, obs, baseline=self.attention_config.get("baseline", "raw_zero"))
            return tokens, (patches if return_patch_attention else self._region_masks(patches, obs))
        return teacher_attention_targets(
            self.teacher_policy, obs, percentiles=self.attention_config["percentiles"],
            seed=self.attention_config.get("teacher_seed", 42),
            progress_callback=self._attention_progress, return_patch_attention=return_patch_attention,
        )

    def _region_patch_targets(self, patches):
        source = getattr(
            self, "region_source",
            normalize_region_source(self.attention_config.get("source", "noise_attention")),
        )
        if source == "context":
            return context_effect_patch_weights(
                patches,
                percentile=self.attention_config.get("percentile", 90.0),
                percentiles=self.attention_config.get("percentiles", [80.0, 99.0]),
                selection=self.attention_config.get("selection", "soft"),
                strength_scale=self.attention_config.get("strength_scale"),
                normalization=self.attention_config.get("normalization"),
            )
        return {
            key: percentile_region(value, self.attention_config["percentiles"])
            for key, value in patches.items()
        }

    def _patch_pool(self, masks):
        pooled = {}
        backbone = self.teacher_policy.obs_encoder.backbone
        for key, mask in masks.items():
            batch, steps, _, height, width = mask.shape
            grid = patch_grid(
                self.teacher_policy.obs_encoder.patch_counts[key], height, width, backbone,
            )
            pooled[key] = F.adaptive_avg_pool2d(
                mask.reshape(batch * steps, 1, height, width), grid,
            ).reshape(batch, steps, -1)
        return pooled

    @staticmethod
    def _intensity_smoothness(masks):
        values = []
        for mask in masks.values():
            values.append(mask[..., 1:, :].sub(mask[..., :-1, :]).abs().mean())
            values.append(mask[..., :, 1:].sub(mask[..., :, :-1]).abs().mean())
        return torch.stack(values).mean() if values else torch.zeros(())

    def _joint_training_active(self):
        return (not self.model.training) or getattr(self, "training_stage", "joint") in {"rgb", "joint"}

    @staticmethod
    def _contact_token_metrics(student_tokens, teacher_tokens, contact_gate):
        """Return contact ratio and token MSE split without changing gradients."""
        if contact_gate is None:
            contact_gate = torch.ones(
                student_tokens.shape[0], 1, device=student_tokens.device, dtype=torch.bool,
            )
        if contact_gate.ndim == 3:
            contact_gate = contact_gate.any(dim=-1)
        if contact_gate.ndim != 2:
            raise ValueError("contact gate must have shape [B, To] or [B, To, K]")
        if student_tokens.shape[0] != contact_gate.shape[0]:
            raise ValueError("contact gate batch size does not match context tokens")
        steps = contact_gate.shape[1]
        if steps < 1 or student_tokens.shape[1] % steps:
            raise ValueError("context token layout is not divisible by observation steps")
        tokens_per_step = student_tokens.shape[1] // steps
        token_open = contact_gate.to(torch.bool).unsqueeze(-1).expand(
            -1, -1, tokens_per_step,
        ).reshape(student_tokens.shape[0], -1)
        per_token = (student_tokens.float() - teacher_tokens.float()).square().mean(dim=-1)
        open_values = per_token[token_open]
        closed_values = per_token[~token_open]
        zero = per_token.new_zeros(())
        open_mse = open_values.mean() if open_values.numel() else zero
        closed_mse = closed_values.mean() if closed_values.numel() else zero
        return {
            "contact_open_ratio": contact_gate.float().mean(),
            "open_token_mse": open_mse,
            "closed_token_mse": closed_mse,
            "open_token_count": float(open_values.numel()),
            "closed_token_count": float(closed_values.numel()),
        }

    def _maybe_transition_stage(self, completed_epochs=None):
        """Apply the configured boundary before the next optimizer epoch."""
        if not getattr(self, "render_stages_enabled", False):
            desired = "joint"
        else:
            completed = self.epoch if completed_epochs is None else int(completed_epochs)
            # A saved RGB stage is authoritative on resume.  An intensity
            # checkpoint saved exactly at the boundary transitions before the
            # next optimizer epoch.
            desired = (
                "rgb" if self.training_stage == "intensity"
                and completed >= getattr(self, "intensity_epochs", 0)
                else self.training_stage
            )
        if desired == self.training_stage:
            if hasattr(self.model, "set_training_stage"):
                self.model.set_training_stage(self.training_stage)
            if getattr(self, "ema", None) is not None:
                self.ema.sync_frozen_parameters(self.model)
            return False
        previous = self.training_stage
        self.training_stage = desired
        self.stage_step = 0
        self.stage_epoch = 0
        self.stage_transition_reason = f"epoch_boundary:{previous}->{desired}"
        reset_rgb = False
        if (previous == "intensity" and desired == "rgb"
                and hasattr(self.model, "initialize_rgb_from_gate")):
            # Transfer the completed gate representation once.  The gate
            # branch remains independent and frozen after this snapshot;
            # RGB then owns and updates its full encoder/decoder path.
            self.model.initialize_rgb_from_gate()
            reset_rgb = True
        if hasattr(self.model, "set_training_stage"):
            self.model.set_training_stage(desired)
            gate_count = sum(p.numel() for p in self.model.gate_parameters())
            rgb_count = sum(p.numel() for p in self.model.rgb_parameters())
        else:
            gate_count = rgb_count = 0
        if getattr(self, "ema", None) is not None:
            # The newly frozen gate must use one exact snapshot in both
            # online and EMA models.  If RGB was copied from the gate, also
            # reset its EMA branch to that same boundary state.
            if reset_rgb:
                self.ema.sync_parameters(self.model)
            else:
                self.ema.sync_frozen_parameters(self.model)
        log.info("RenderNet training stage transition: %s -> %s at completed_epoch=%d; "
                 "gate_parameters=%d, rgb_parameters=%d",
                 previous, desired, self.epoch, gate_count, rgb_count)
        return True

    def _loss(self, obs, cached_targets=None, *, action=None):
        # Region targets are used for the warm-up loss and retained as
        # diagnostics during RGB training and validation.  Model selection
        # still uses only the primary objective.
        supervise_regions = (
            self.attention_enabled and self.model.training and torch.is_grad_enabled()
            and (not getattr(self, "render_stages_enabled", False)
                 or getattr(self, "training_stage", "joint") == "intensity")
        )
        measure_regions = bool(self.attention_enabled)
        with self._autocast():
            with torch.no_grad():
                target_patches = None
                target_masks = None
                if cached_targets is not None:
                    teacher_tokens = cached_targets["tokens"].to(self.device, non_blocking=True)
                    if measure_regions:
                        target_patches = {
                            key: value.to(self.device, non_blocking=True)
                            for key, value in cached_targets["patch_attention"].items()
                        }
                        target_masks = self._region_masks(target_patches, obs)
                elif measure_regions:
                    teacher_tokens, target_patches = self._region_targets(
                        obs, return_patch_attention=True,
                    )
                    target_masks = self._region_masks(target_patches, obs)
                else:
                    teacher_tokens = self.teacher_policy._encode_context(
                        obs, apply_curriculum_mask=False,
                    ).detach()
            compiled_encoder = (getattr(self, "_student_image_encoder", None)
                                if self.model.training and torch.is_grad_enabled() else None)
            renderer = (getattr(self, "_training_model", None)
                        if self.model.training and torch.is_grad_enabled() else None)
            if renderer is None:
                renderer = self.model
                if not (self.model.training and torch.is_grad_enabled()):
                    renderer = getattr(self, "ema_model", None)
                    if renderer is None:
                        renderer = self.model
            student_tokens, masks, contact_gate = student_context_tokens(
                self.teacher_policy, renderer, obs,
                return_aux=True, return_contact=True, image_context_encoder=compiled_encoder,
            )
            if student_tokens.shape != teacher_tokens.shape:
                raise RuntimeError(f"Token shapes differ: teacher={teacher_tokens.shape}, student={student_tokens.shape}")
            token_mse = F.mse_loss(student_tokens.float(), teacher_tokens.float())
            contact_metrics = self._contact_token_metrics(
                student_tokens, teacher_tokens, contact_gate,
            )
            mask_mse = token_mse.new_zeros(())
            patch_mse = token_mse.new_zeros(())
            smoothness = token_mse.new_zeros(())
            if measure_regions and target_masks is not None:
                region_context = contextlib.nullcontext() if supervise_regions else torch.no_grad()
                with region_context:
                    mask_mse = torch.stack([
                        F.mse_loss(masks[key].float(), target_masks[key].float())
                        for key in masks
                    ]).mean()
                    smoothness = self._intensity_smoothness(masks)
                    predicted_patches = self._patch_pool(masks)
                    target_patch_weights = self._region_patch_targets(target_patches)
                    patch_losses = []
                    for key in predicted_patches:
                        target = target_patch_weights[key].to(
                            predicted_patches[key].device, predicted_patches[key].dtype,
                        )
                        patch_losses.append(F.smooth_l1_loss(predicted_patches[key], target))
                    patch_mse = torch.stack(patch_losses).mean()
            denoiser_mse = demo_mse = token_mse.new_zeros(())
            if self.objective == "denoiser_mse" and not getattr(self, "render_stages_enabled", False):
                denoiser_mse, demo_mse = self._denoiser_losses(
                    action, student_tokens, teacher_tokens.to(student_tokens.dtype),
                )
                active_denoiser = (denoiser_mse if self._joint_training_active()
                                   else denoiser_mse.detach() * 0)
                loss = (active_denoiser
                        + float(self.distillation_config.get("token_loss_weight", 0.0)) * token_mse
                        + float(self.distillation_config.get("demo_loss_weight", 0.0)) * demo_mse)
            else:
                loss = token_mse
            intensity_loss = (
                float(getattr(self, "context_patch_loss_weight", 1.0)) * patch_mse
                + float(getattr(self, "intensity_loss_weight", 1.0)) * mask_mse * float(
                    self.attention_config.get("pixel_loss_weight", 0.0)
                )
                + float(getattr(self, "smoothness_loss_weight", 0.02)) * smoothness
            ) if measure_regions else token_mse.new_zeros(())
            if not supervise_regions:
                # Diagnostics must not create a second RGB-stage gradient path.
                intensity_loss = intensity_loss.detach()
                mask_mse = mask_mse.detach()
                patch_mse = patch_mse.detach()
                smoothness = smoothness.detach()
            if supervise_regions:
                loss = loss + intensity_loss
            if getattr(self, "render_stages_enabled", False) and self.model.training and torch.is_grad_enabled():
                if getattr(self, "training_stage", "joint") == "intensity":
                    if not self.attention_enabled:
                        raise RuntimeError(
                            "RenderNet intensity stage requires high_attention_render.enabled=true "
                            "so the teacher context-effect field can supervise the gate"
                        )
                    loss = intensity_loss
                elif self.training_stage == "rgb":
                    # RGB stage updates the complete RGB branch; the
                    # independently trained gate path is frozen.
                    loss = token_mse
            self.loss_metrics = {"token_mse": float(token_mse.detach()),
                                 "denoiser_mse": float(denoiser_mse.detach()),
                                 "demo_mse": float(demo_mse.detach()),
                                 "mask_mse": float(mask_mse.detach()),
                                 "context_patch_mse": float(patch_mse.detach()),
                                 "intensity_smoothness": float(smoothness.detach()),
                                 "intensity_loss": float(intensity_loss.detach()),
                                 "contact_open_ratio": float(contact_metrics["contact_open_ratio"].detach()),
                                 "open_token_mse": float(contact_metrics["open_token_mse"].detach()),
                                 "closed_token_mse": float(contact_metrics["closed_token_mse"].detach()),
                                 "open_token_count": contact_metrics["open_token_count"],
                                 "closed_token_count": contact_metrics["closed_token_count"],
                                 "loss": float(loss.detach()),
                                 "mask_supervised": supervise_regions,
                                 "mask_measured": measure_regions,
                                 "joint_training": self._joint_training_active(),
                                 "training_stage": getattr(self, "training_stage", "joint")}
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
            "wrench_preprocessing_version": WRENCH_PREPROCESSING_VERSION,
            "future_contact_mode": getattr(self.teacher_policy, "future_contact_mode", "legacy_rollout"),
            "contact_fusion_gate": bool(getattr(self.teacher_policy, "contact_fusion_gate", False)),
            "dataset": OmegaConf.to_container(dataset_cfg, resolve=True),
            "source": dataset_source_signature(dataset.dataset_path),
            "amp_enabled": self.amp_enabled, "amp_dtype": str(self.amp_dtype),
            "allow_tf32": bool(self.cfg.training.amp.allow_tf32),
            "capture_attention": self.attention_enabled,
            "attention_seed": int(self.attention_config.get("teacher_seed", 42)),
            "cache_batch_size": int(config["batch_size"]),
        }
        # Storage precision is separate from the preprocessing/fusion contract.
        if token_dtype != "float32":
            identity["token_storage_dtype"] = token_dtype
        precision = config.get("precision", "model")
        if precision not in {"model", "float32"}:
            raise ValueError("teacher_cache.precision must be model or float32")
        if precision != "model":
            identity["target_precision"] = precision
        region_source = None
        if self.attention_enabled:
            region_source = getattr(
                self, "region_source",
                normalize_region_source(self.attention_config.get("source", "noise_attention")),
            )
            identity["region_source"] = (
                "context_effect_v1" if region_source == "context" else "noise_attention_v1"
            )
        if self.attention_enabled and region_source == "context":
            identity["context_effect_baseline"] = self.attention_config.get("baseline", "raw_zero")
            identity["context_effect_precision"] = "float32"
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
                            amp = (torch.autocast(device_type=self.device.type, enabled=False)
                                   if config.get("precision", "model") == "float32" else self._autocast())
                            with amp:
                                if self.attention_enabled:
                                    tokens, attention = self._region_targets(obs, return_patch_attention=True)
                                else:
                                    tokens = self.teacher_policy._encode_context(obs, apply_curriculum_mask=False).detach()
                                    attention = {}
                            cache.append(tokens, attention)
                            progress.update(len(tokens))
                            now = time.monotonic()
                            if (not self._progress_enabled()
                                    and now - last_log >= self._status_interval()):
                                elapsed = now - started
                                rate = (cache.completed - start_count) / max(elapsed, 1e-6)
                                log.info(f"Teacher cache {name}: {cache.completed}/{len(source)}, "
                                         f"{rate:.2f} samples/s, elapsed={elapsed:.1f}s, "
                                         f"ETA={(len(source) - cache.completed) / max(rate, 1e-6):.1f}s; "
                                         f"token_shape={list(tokens.shape)}")
                                last_log = now
                    log.info("Teacher cache %s ready: %d samples, %.1fs", name, cache.completed,
                             time.monotonic() - started)
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
        if getattr(self, "ema_model", None) is not None:
            self.ema_model.eval()
        total, count = 0.0, 0
        component_keys = (
            "token_mse", "denoiser_mse", "demo_mse", "mask_mse",
            "context_patch_mse", "intensity_smoothness", "intensity_loss",
            "contact_open_ratio",
        )
        component_totals = {key: 0.0 for key in component_keys}
        open_token_sum = closed_token_sum = open_token_count = closed_token_count = 0.0
        started = last_status = time.monotonic()
        batches = len(loader)
        if self.max_val_batches is not None:
            batches = min(batches, int(self.max_val_batches))
        progress = self._progress(itertools.islice(self._iterate_loader(loader), batches),
                                  f"Val {self.epoch}", batches)
        for index, batch in enumerate(progress):
            loss = self._batch_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite validation MSE at step {self.global_step}")
            size = next(iter(batch["obs"].values())).shape[0]
            total += self._selection_mse(loss) * size
            for key in component_totals:
                component_totals[key] += getattr(self, "loss_metrics", {}).get(key, 0.0) * size
            metrics = getattr(self, "loss_metrics", {})
            open_token_sum += metrics.get("open_token_mse", 0.0) * metrics.get("open_token_count", 0.0)
            closed_token_sum += metrics.get("closed_token_mse", 0.0) * metrics.get("closed_token_count", 0.0)
            open_token_count += metrics.get("open_token_count", 0.0)
            closed_token_count += metrics.get("closed_token_count", 0.0)
            count += size
            progress.set_postfix(mse=f"{total / count:.6f}", refresh=False)
            now = time.monotonic()
            if not self._progress_enabled() and self.is_main_process and now - last_status >= self._status_interval():
                log.info("Val %d | batch=%d/%d mse=%.6f elapsed=%.1fs",
                         self.epoch, index + 1, batches, total / count, now - started)
                last_status = now
        progress.close()
        total, count = self._sum([total, count])
        count = int(count)
        if count == 0:
            raise RuntimeError("Validation split is empty")
        self.validation_samples = count
        self.validation_seconds = self._max_seconds(time.monotonic() - started)
        self.validation_samples_per_second = count / max(self.validation_seconds, 1e-6)
        totals = self._sum(list(component_totals.values()) + [
            open_token_sum, closed_token_sum, open_token_count, closed_token_count,
        ])
        component_count = len(component_totals)
        self.validation_metrics = {
            key: value / count for key, value in zip(component_totals, totals[:component_count])
        }
        open_sum, closed_sum, open_count, closed_count = totals[component_count:]
        self.validation_metrics["open_token_mse"] = open_sum / open_count if open_count else 0.0
        self.validation_metrics["closed_token_mse"] = closed_sum / closed_count if closed_count else 0.0
        self.validation_metrics["open_token_count"] = open_count
        self.validation_metrics["closed_token_count"] = closed_count
        return total / count

    def _summary(self, status):
        if not self.is_main_process:
            return
        summary = {
            "status": status, "global_step": self.global_step,
            "epoch": self.epoch, "max_epochs": self.num_epochs,
            "intensity_warmup_epochs": self.intensity_warmup_epochs,
            "render_stages": {
                "enabled": self.render_stages_enabled,
                "intensity_epochs": self.intensity_epochs,
                "current_stage": self.training_stage,
                "stage_step": self.stage_step,
                "stage_epoch": self.stage_epoch,
                "transition_reason": self.stage_transition_reason,
            },
            "max_steps": self.max_steps, "stop_reason": self.stop_reason,
            "best_epoch": self.best_epoch, "best_step": self.best_step,
            "early_stopping": {"enabled": self.early_stopping_enabled,
                               "patience_epochs": self.patience, "min_delta": self.min_delta,
                               "epochs_without_improvement": self.epochs_without_improvement},
            "teacher_checkpoint": str(pathlib.Path(self.cfg.teacher_checkpoint).resolve()),
            "teacher_state_key": self.teacher_state_key, "cross_attn_query": "image",
            "student_state_key": self.evaluation_state_key,
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "lr": self.optimizer.param_groups[0]["lr"],
            "lr_scheduler": (self.lr_scheduler.state_dict()
                             if self.lr_scheduler is not None and self.validation_enabled else None),
            "wrench_preprocessing": OmegaConf.to_container(self.cfg.wrench_preprocessing, resolve=True),
            "render_net": OmegaConf.to_container(self.cfg.render_net, resolve=True),
            "validation_enabled": self.validation_enabled,
            "distillation": OmegaConf.to_container(OmegaConf.create(self.distillation_config), resolve=True),
            "objective": self.objective,
            "validation_metric": f"{self.metric_name} (excludes auxiliary losses)" if self.validation_enabled else None,
            "teacher_cache": getattr(self, "teacher_cache_paths", None),
            "world_size": self.distributed_context.world_size,
            "batching": {
                "train_per_rank": getattr(self, "train_batch_size", None),
                "train_global": getattr(self, "train_global_batch_size", None),
                "train_batches_per_epoch": getattr(self, "train_batches_per_epoch", None),
                "validation_per_rank": getattr(self, "val_batch_size", None),
                "validation_batches": getattr(self, "val_batches", None),
            },
            "teacher_trainable_parameters": sum(p.numel() for p in self.teacher_policy.parameters() if p.requires_grad),
            "student_trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "initial_train_mse": self.initial_train_mse, "initial_val_mse": self.initial_val_mse,
            "latest_val_mse": self.latest_val_loss,
            "latest_validation_metrics": getattr(self, "validation_metrics", {}),
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
        self._step_lr_scheduler(self.latest_val_loss)
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
        metrics = {"global_step": self.global_step, "epoch": self.epoch,
                   "objective": self.objective,
                   "student_state_key": self.evaluation_state_key,
                   "val_per_rank_batch_size": getattr(self, "val_batch_size", None),
                   "val_global_batch_size": (
                       getattr(self, "val_batch_size", 0) * self.distributed_context.world_size
                       if getattr(self, "val_batch_size", None) is not None else None
                   ),
                   "lr": self.optimizer.param_groups[0]["lr"],
                   "val_mse": self.latest_val_loss, "best_val_mse": self.best_val_loss,
                   "epochs_without_improvement": self.epochs_without_improvement,
                   "val_seconds": getattr(self, "validation_seconds", 0.0),
                   "val_samples_per_second": getattr(self, "validation_samples_per_second", 0.0),
                   "elapsed_seconds": time.monotonic() - self.started_at, **self._gpu_metrics()}
        metrics.update({f"val_{key}": value for key, value in getattr(self, "validation_metrics", {}).items()})
        logger.log(metrics)
        logger.save_plot()
        self._summary("running")
        return self.early_stopping_enabled and self.epochs_without_improvement >= self.patience

    def run(self):
        cache_config = self.cfg.get("teacher_cache", {})
        with logging_redirect_tqdm(), TeacherCacheLifecycle(
                cleanup_on_exit=cache_config.get("cleanup_on_exit", True)) as lifetime:
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
                     "validation, LR scheduling, early stopping and best checkpoint selection are disabled. "
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
            self.train_batch_size = int(train_loader.batch_size)
            self.train_global_batch_size = int(train_loader.batch_size * self.distributed_context.world_size)
            self.train_batches_per_epoch = int(len(train_loader))
            self.val_batch_size = int(val_loader.batch_size) if val_loader is not None else None
            self.val_batches = int(len(val_loader)) if val_loader is not None else 0
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
            log.info("Loader lengths: train=%d batches/epoch; validation=%d batches; "
                     "configured batch=%d/%s",
                     self.train_batches_per_epoch, self.val_batches,
                     self.train_global_batch_size,
                     self.val_batch_size if self.val_batch_size is not None else "disabled")
            logging_config = self.cfg.get("logging", {})
            logger_context = (RenderNetTrainingLogger(output, logging_config.get("plot_every_steps", 100))
                              if self.is_main_process else contextlib.nullcontext(_NullTrainingLogger()))
            with logger_context as logger:
                if self.is_main_process:
                    log.info(f"Local metrics: {logger.path}; PNG: {output / 'training_metrics.png'}")
                if self.resume_from is None:
                    self.model.eval()
                    log.info(f"Initial evaluation started: one training batch "
                             f"(per_rank_batch={train_loader.batch_size}, "
                             f"global_batch={train_loader.batch_size * self.distributed_context.world_size}); "
                             f"validation={self.validation_enabled}")
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
                        self._step_lr_scheduler(self.initial_val_mse)
                        if self.is_main_process:
                            self.save_checkpoint(tag="best", use_thread=False)
                        log.info(f"Initial train_mse={self.initial_train_mse:.8f}, val_mse={self.initial_val_mse:.8f}")
                        logger.log({"global_step": 0, "epoch": 0, "initial_train_mse": self.initial_train_mse,
                                    "objective": self.objective,
                                    "student_state_key": self.evaluation_state_key,
                                    "initial_val_mse": self.initial_val_mse,
                                    "val_seconds": getattr(self, "validation_seconds", 0.0),
                                    "val_samples_per_second": getattr(self, "validation_samples_per_second", 0.0),
                                    "elapsed_seconds": time.monotonic() - self.started_at, **self._gpu_metrics()})
                        logger.save_plot()
                else:
                    log.info("Skipping initial evaluation; restored checkpoint state will be continued")
                self._summary("running")
                for epoch_index in range(self.epoch, self.num_epochs):
                    self._maybe_transition_stage(completed_epochs=self.epoch)
                    if self.max_steps is not None and self.global_step >= int(self.max_steps):
                        self.stop_reason = "max_steps"
                        break
                    if isinstance(train_loader.sampler, DistributedSampler):
                        train_loader.sampler.set_epoch(epoch_index)
                    epoch_number = epoch_index + 1
                    epoch_total, epoch_samples = 0.0, 0
                    epoch_loss_total = epoch_mask_total = 0.0
                    epoch_components = {key: 0.0 for key in (
                        "token_mse", "denoiser_mse", "demo_mse", "context_patch_mse",
                        "intensity_smoothness", "intensity_loss", "contact_open_ratio",
                    )}
                    epoch_open_token_sum = epoch_closed_token_sum = 0.0
                    epoch_open_token_count = epoch_closed_token_count = 0.0
                    epoch_started = previous_end = last_status = time.monotonic()
                    batches = len(train_loader)
                    if self.max_steps is not None:
                        batches = min(batches, int(self.max_steps) - self.global_step)
                    progress = self._progress(itertools.islice(self._iterate_loader(train_loader), batches),
                                              f"Train {epoch_number}/{self.num_epochs}", batches)
                    if hasattr(self.model, "set_stage_mode"):
                        self.model.set_stage_mode(True)
                    else:
                        self.model.train()
                    for batch_index, batch in enumerate(progress):
                        step_started = time.monotonic()
                        data_seconds = step_started - previous_end
                        if hasattr(self.model, "set_stage_mode"):
                            self.model.set_stage_mode(True)
                        else:
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
                        scale_before = self.grad_scaler.get_scale()
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                        if self.ema is not None and self.grad_scaler.get_scale() >= scale_before:
                            self.ema.step(self.model)
                        grad_norm_value = float(grad_norm)
                        step_seconds = time.monotonic() - step_started
                        self.global_step += 1
                        self.stage_step += 1
                        size = next(iter(batch["obs"].values())).shape[0]
                        metrics_for_batch = self.loss_metrics
                        open_count = metrics_for_batch.get("open_token_count", 0.0)
                        closed_count = metrics_for_batch.get("closed_token_count", 0.0)
                        primary_total, loss_total, mask_total, global_size, *component_totals = self._sum([
                            self._selection_mse(loss) * size, float(loss.detach()) * size,
                            self.loss_metrics.get("mask_mse", 0.0) * size, size,
                            *[self.loss_metrics.get(key, 0.0) * size for key in epoch_components],
                            metrics_for_batch.get("open_token_mse", 0.0) * open_count,
                            open_count,
                            metrics_for_batch.get("closed_token_mse", 0.0) * closed_count,
                            closed_count,
                        ])
                        regular_count = len(epoch_components)
                        regular_totals = component_totals[:regular_count]
                        open_total, open_count_total, closed_total, closed_count_total = component_totals[regular_count:]
                        epoch_total += primary_total
                        epoch_samples += global_size
                        epoch_loss_total += loss_total
                        epoch_mask_total += mask_total
                        for key, value in zip(epoch_components, regular_totals):
                            epoch_components[key] += value
                        epoch_open_token_sum += open_total
                        epoch_open_token_count += open_count_total
                        epoch_closed_token_sum += closed_total
                        epoch_closed_token_count += closed_count_total
                        metrics = {"global_step": self.global_step, "epoch": epoch_number,
                                   "objective": self.objective,
                                   "train_mse": primary_total / global_size, "train_loss": loss_total / global_size,
                                   "train_mask_mse": mask_total / global_size,
                                   "train_per_rank_batch_size": size,
                                   "train_global_batch_size": global_size,
                                   "grad_norm": grad_norm_value, "lr": self.optimizer.param_groups[0]["lr"],
                                   "batch_in_epoch": batch_index + 1, "batches_in_epoch": batches,
                                   "data_seconds": data_seconds, "step_seconds": step_seconds,
                                   "samples_per_second": global_size / max(data_seconds + step_seconds, 1e-6),
                                   "elapsed_seconds": time.monotonic() - self.started_at, **self._gpu_metrics()}
                        metrics.update({f"train_{key}": value / global_size
                                        for key, value in zip(epoch_components, regular_totals)})
                        metrics.update({
                            "train_open_token_mse": metrics_for_batch.get("open_token_mse", 0.0),
                            "train_closed_token_mse": metrics_for_batch.get("closed_token_mse", 0.0),
                        })
                        logger.log(metrics)
                        progress.set_postfix(loss=f"{metrics['train_loss']:.4f}",
                                             mse=f"{epoch_total / epoch_samples:.4f}",
                                             field=f"{metrics['train_intensity_loss']:.4f}", lr=f"{metrics['lr']:.2g}",
                                             refresh=False)
                        now = time.monotonic()
                        if (not self._progress_enabled() and self.is_main_process
                                and now - last_status >= self._status_interval()):
                            log.info("Train %d/%d | step=%d batch=%d/%d loss=%.5f mse=%.5f field=%.5f lr=%.2g",
                                     epoch_number, self.num_epochs, self.global_step, batch_index + 1, batches,
                                     metrics['train_loss'], epoch_total / epoch_samples,
                                     metrics['train_intensity_loss'], metrics['lr'])
                            last_status = now
                        previous_end = time.monotonic()
                        if self.max_steps is not None and self.global_step >= int(self.max_steps):
                            break
                    progress.close()
                    self.epoch = epoch_number
                    self.stage_epoch += 1
                    epoch_seconds = self._max_seconds(time.monotonic() - epoch_started)
                    logger.log({"global_step": self.global_step, "epoch": self.epoch,
                                "objective": self.objective,
                                "epoch_train_mse": epoch_total / epoch_samples,
                                "epoch_train_loss": epoch_loss_total / epoch_samples,
                                "epoch_train_mask_mse": epoch_mask_total / epoch_samples,
                                "epoch_samples": epoch_samples, "epoch_seconds": epoch_seconds,
                                "epoch_samples_per_second": epoch_samples / max(epoch_seconds, 1e-6),
                                **{f"epoch_train_{key}": value / epoch_samples
                                   for key, value in epoch_components.items()},
                                "epoch_train_open_token_mse": (
                                    epoch_open_token_sum / epoch_open_token_count
                                    if epoch_open_token_count else 0.0
                                ),
                                "epoch_train_closed_token_mse": (
                                    epoch_closed_token_sum / epoch_closed_token_count
                                    if epoch_closed_token_count else 0.0
                                ),
                                "epoch_open_token_count": epoch_open_token_count,
                                "epoch_closed_token_count": epoch_closed_token_count})
                    early_stop = self._validate_and_save(val_loader, logger)
                    if self.is_main_process:
                        validation = (
                            f" val={self.latest_val_loss:.6f} best={self.best_val_loss:.6f}"
                            f" stale={self.epochs_without_improvement}/{self.patience}"
                            f"{' [best saved]' if self.best_step == self.global_step else ''}"
                            if val_loader is not None else "")
                        log.info("Epoch %d/%d | step=%d loss=%.6f train=%.6f field=%.6f%s elapsed=%.1fs",
                                 self.epoch, self.num_epochs, self.global_step,
                                 epoch_loss_total / epoch_samples, epoch_total / epoch_samples,
                                 epoch_components["intensity_loss"] / epoch_samples, validation,
                                 time.monotonic() - epoch_started)
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
