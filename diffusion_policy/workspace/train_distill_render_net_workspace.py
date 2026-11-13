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
from diffusion_policy.common.checkpoint_action_mse import resolve_student_teacher
from diffusion_policy.common.dataset_config import canonical_dataset_config
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.teacher_cache_lifecycle import TeacherCacheLifecycle
from diffusion_policy.dataset.render_net_teacher_cache import (
    CACHE_VERSION, CachedTeacherDataset, TeacherTargetCache, cache_key, dataset_source_signature,
)
from diffusion_policy.dataset.image_result_cache import use_shared_ram_result_cache
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.vision.render_net import RenderNet, build_render_net
from diffusion_policy.model.vision.force_context_distillation import (
    raw_zero_wrench_observations, teacher_context_pair, compute_force_context_losses,
)
from diffusion_policy.model.vision.render_net_attention import (
    attention_regions, context_effect_patch_weights, context_effect_regions,
    percentile_region, teacher_attention_targets, teacher_context_effect_targets,
    validate_percentiles,
)
from diffusion_policy.model.vision.wrench_preprocessing import (
    WRENCH_PREPROCESSING_VERSION, WrenchPreprocessor, validate_precontact_seconds,
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
                           return_contact=False, image_context_encoder=None,
                           use_teacher_ffn=False, return_phase=False,
                           condition_only=False, observation_mask=None,
                           selected_observation_indices=None, return_rendered=False):
    """Render and encode with teacher supervision; student alpha is predicted causally."""
    prepare_with_future = getattr(teacher, "prepare_context_inputs_with_future_contact", None)
    if prepare_with_future is not None:
        images, wrench, contact, future_contact = prepare_with_future(obs)
    else:
        # Lightweight teacher adapters used by existing render-only callers
        # expose the original four-value preprocessing interface.
        images, wrench, contact, future_contact = teacher.prepare_context_inputs(obs)
        if future_contact is None:
            future_contact = contact
    # Offline labels supervise the condition field; the student predicts the
    # blend weight from the observations that are available at inference.
    renderer_impl = getattr(render_net, "module", render_net)
    if isinstance(renderer_impl, RenderNet):
        # Offline alpha is the teacher target. A phase-enabled student must
        # learn and use its own causal alpha for both training and deployment.
        student_alpha = (
            obs["contact_alpha"][:, :teacher.n_obs_steps]
            if (renderer_impl.use_contact_alpha and "contact_alpha" in obs
                and not renderer_impl.phase_prediction_enabled)
            else None
        )
        result = render_net(
            images, wrench, future_contact=future_contact, return_aux=True,
            contact_alpha=student_alpha, return_phase=return_phase or condition_only,
            condition_only=condition_only, render_observation_mask=observation_mask,
        )
        if return_phase or condition_only:
            rendered, masks, phase_predictions = result
        else:
            rendered, masks = result
            phase_predictions = None
        if condition_only:
            return None, masks, future_contact, phase_predictions
    else:
        # Lightweight test/render adapters from older checkpoints may expose
        # only the original two-input signature.
        rendered, masks = render_net(images, wrench, return_aux=True)
        phase_predictions = None
    encode = teacher.obs_encoder.encode_image_context if image_context_encoder is None else image_context_encoder
    encode_kwargs = {}
    if selected_observation_indices is not None:
        encode_kwargs["selected_observation_indices"] = selected_observation_indices
    if use_teacher_ffn:
        tokens = encode(rendered, input_grad=True, apply_fusion_ffn=True, **encode_kwargs)
    else:
        tokens = encode(rendered, input_grad=True, **encode_kwargs)
    if return_rendered:
        return tokens, rendered
    if return_phase:
        return tokens, masks, future_contact, phase_predictions
    if return_aux and return_contact:
        return tokens, masks, future_contact
    if return_aux:
        return tokens, masks
    if return_contact:
        return tokens, future_contact
    return tokens


class TrainDistillRenderNetWorkspace(BaseWorkspace):
    exclude_keys = ("teacher_policy", "_training_model")
    include_keys = ("global_step", "epoch", "best_val_loss", "initial_train_mse",
                    "initial_val_mse", "shape_info", "teacher_state_key", "latest_val_loss",
                    "best_epoch", "best_step", "epochs_without_improvement",
                    "early_stopping_best_loss", "stop_reason", "training_stage", "stage_step",
                    "stage_epoch", "stage_transition_reason", "render_stages_enabled",
                    "condition_warmup_epochs", "resume_architecture_report",
                    "condition_best_state", "condition_best_metric", "condition_best_epoch",
                    "condition_transition_pending", "force_difference_scale", "force_region_scale",
                    "force_context_statistics")

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

        checkpoint, self.teacher_source_cfg = resolve_student_teacher(cfg)
        if cfg.get("teacher_selection"):
            score = cfg.teacher_selection
            log.info("Selected teacher %s: checkpoint action MSE=%.8f, episodes=%s",
                     checkpoint, score["checkpoint_action_mse"], score["episode_ids"])
        else:
            log.info("Using specified teacher checkpoint: %s", checkpoint)
        payload, self.teacher_cfg = load_checkpoint_payload(checkpoint)
        self.teacher_state_key = "ema_model" if "ema_model" in payload["state_dicts"] else "model"
        self.teacher_policy = instantiate_policy(payload, self.teacher_cfg, self.device,
                                                 capture_attention=False)
        del payload
        if self.teacher_policy.obs_encoder.cross_attn_query != "image":
            raise ValueError("RenderNet distillation requires teacher cross_attn_query: image")
        if int(cfg.render_net.get("cross_camera_attn_layers", 0)) > 0:
            encoder = self.teacher_policy.obs_encoder
            if (getattr(encoder, "fusion_output_mode", "all") != "visual"
                    or not encoder.pre_cross_self_attn_layers or encoder.cross_attn_layers):
                raise ValueError(
                    "Cross-camera RenderNet requires a visual-output joint self-attention "
                    "teacher checkpoint; use the newly trained teacher, not an old all-token checkpoint"
                )
        self.teacher_policy.eval().requires_grad_(False)
        log.info("Student token supervision: frozen image encoder%s; deployment uses RenderNet only",
                 " + teacher fusion FFNs" if self.use_teacher_ffn else "")
        self.normalizer = self.teacher_policy.normalizer
        if cfg.render_net.get("phase_prediction", {}).get("enabled", False):
            source_policy = (self.teacher_source_cfg.policy
                             if self.teacher_source_cfg is not None else self.teacher_cfg.policy)
            duration = validate_precontact_seconds(source_policy.get("precontact_seconds"))
            if duration is None:
                raise ValueError(
                    "Phase-enabled student requires policy.precontact_seconds in the teacher source config; "
                    "set it to 0.5 to construct offline pre-contact ramp targets"
                )
            if self.teacher_policy.future_phase_wrench_key is not None:
                raise ValueError("Pre-contact distillation requires a teacher with causal current wrench history")
            # This changes the frozen teacher's data path and alpha labels,
            # not its learned weights. Persist the effective configuration so
            # cached targets cannot be mistaken for the legacy binary gate.
            self.teacher_policy.precontact_seconds = duration
            with open_dict(self.teacher_cfg.policy):
                self.teacher_cfg.policy.precontact_seconds = duration
            log.info("Frozen teacher pre-contact fusion: %.3gs linear ramp; retain causal measured tau_ext", duration)
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
        if render_config.get("image_encoder", "cnn") == "dino_lora":
            dino_config = dict(render_config.get("dino") or {})
            dino_path = dino_config.get("pretrained_model_name_or_path")
            if dino_path is None:
                dino_path = getattr(encoder, "pretrained_model_name_or_path", None)
            if not dino_path:
                raise ValueError(
                    "DINO RenderNet requires dino.pretrained_model_name_or_path or a teacher "
                    "obs_encoder.pretrained_model_name_or_path"
                )
            dino_config["pretrained_model_name_or_path"] = str(dino_path)
            render_config["dino"] = dino_config
            log.info("Student DINO loads independent pretrained weights from %s; "
                     "frozen base with LoRA=%s", dino_path, dino_config.get("lora", {}))
        with open_dict(self.cfg):
            self.cfg.render_net = OmegaConf.create(render_config)
        self.model = build_render_net(wrench_dim, render_config).to(self.device)
        if isinstance(self.model, RenderNet):
            log.info("RGB residual modulation: spatial region gate=%s, contact alpha=%s",
                     self.model.use_region_gate, self.model.use_contact_alpha)
        if (isinstance(self.model, RenderNet) and self.model.initialize_wrench_from_teacher
                and hasattr(self.teacher_policy.obs_encoder, "force_encoder")):
            self.model.initialize_wrench_encoder(self.teacher_policy.obs_encoder)
        phase_config = cfg.render_net.get("phase_prediction", {})
        if (isinstance(self.model, RenderNet)
                and phase_config.get("freeze_wrench_encoder", False)
                and hasattr(self.model, "freeze_wrench_encoder")):
            self.model.freeze_wrench_encoder()
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
        # joint_after_rgb=false selects condition training followed by a
        # frozen-condition renderer; legacy joint schedules remain supported.
        self.render_stages_enabled = (
            bool(stage_config.get("enabled", True)) and self.attention_enabled
            and bool(getattr(self.model, "phase_prediction_enabled", False))
            and hasattr(self.model, "set_training_stage")
        )
        self.condition_warmup_epochs = int(stage_config.get(
            "condition_warmup_epochs", stage_config.get("warmup_epochs", 1),
        ))
        self.rgb_warmup_epochs = int(stage_config.get("rgb_warmup_epochs", 0))
        # When false, RGB/render training remains the final stage. This keeps
        # the condition branch frozen instead of re-opening the shared
        # encoder in a joint phase after the render stage.
        self.joint_after_rgb = bool(stage_config.get("joint_after_rgb", True))
        self.condition_early_stopping = {"enabled": True, "patience": 5, "min_delta": 1e-4}
        self.condition_early_stopping.update(stage_config.get("condition_early_stopping", {}))
        self.condition_best_state = None
        self.condition_best_metric = float("inf")
        self.condition_best_epoch = None
        self.condition_transition_pending = False
        condition_patience = int(self.condition_early_stopping.get("patience", 5))
        condition_delta = float(self.condition_early_stopping.get("min_delta", 1e-4))
        if condition_patience < 1 or not math.isfinite(condition_delta) or condition_delta < 0:
            raise ValueError("condition_early_stopping needs positive patience and finite nonnegative min_delta")
        if self.condition_warmup_epochs < 0:
            raise ValueError("training.render_stages.condition_warmup_epochs must be nonnegative")
        if self.rgb_warmup_epochs < 0:
            raise ValueError("training.render_stages.rgb_warmup_epochs must be nonnegative")
        self.training_stage = (
            "condition_warmup" if self.render_stages_enabled and self.condition_warmup_epochs > 0
            else "rgb" if self.strict_two_stage else "joint"
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
        self.selection_metric_changed = False
        self.force_difference_scale = self.force_context_config.get("difference_scale")
        self.force_region_scale = self.force_context_config.get("region_scale")
        self.force_context_statistics = None
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
                saved_image_encoder = saved_backbone.get(
                    "image_encoder", saved_render.get("image_encoder", "cnn")
                )
                current_image_encoder = cfg.render_net.get("image_encoder", "cnn")
                current_base = cfg.render_net.get("backbone", {}).get(
                    "base_channels", cfg.render_net.get("base_channels")
                )
                mismatches = []
                if saved_image_encoder != current_image_encoder:
                    mismatches.append(
                        f"image_encoder: checkpoint={saved_image_encoder}, current={current_image_encoder}"
                    )
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
                    "saved_image_encoder": saved_image_encoder,
                    "current_image_encoder": current_image_encoder,
                    "mismatched_keys": mismatches[:32],
                    "mismatch_count": len(mismatches),
                }
                if mismatches:
                    raise RuntimeError(
                        "RenderNet checkpoint architecture mismatch; refusing to silently skip "
                        f"{len(mismatches)} parameters (base_channels checkpoint={saved_base}, "
                        f"current={current_base}). Examples: " + "; ".join(mismatches[:8])
                    )
                saved_stage_config = saved_cfg.get("training", {}).get("render_stages", {})
                saved_strict = (bool(saved_stage_config.get("enabled", True))
                                and self.attention_enabled
                                and bool(getattr(self.model, "phase_prediction_enabled", False))
                                and not bool(saved_stage_config.get("joint_after_rgb", True)))
                if saved_strict != self.strict_two_stage:
                    raise ValueError("Cannot resume across joint and strict two-stage schedules; start a new run")
                self.load_payload(restored, include_keys=self.include_keys, exclude_keys=excluded)
                # DictOfTensorMixin reconstructs normalizer parameters on the
                # checkpoint's CPU device. This module is shared with the
                # frozen teacher and must follow the resumed GPU inputs.
                self.wrench_preprocessor.to(self.device)
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
            saved_force = restored["cfg"].get("distillation", {}).get("force_context", {})
            if bool(saved_force.get("enabled", False)) != self.force_context_enabled:
                raise ValueError("Cannot resume across token-only and force-context objectives; start a new run")
            if self.force_context_enabled:
                for name in ("difference_weight", "region_weight"):
                    if saved_force.get(name, 1.0) != self.force_context_config.get(name, 1.0):
                        raise ValueError(f"Cannot change force_context.{name} on resume; start a new run")
                for name, saved_value in (("difference_scale", self.force_difference_scale),
                                          ("region_scale", self.force_region_scale)):
                    requested = self.force_context_config.get(name)
                    if requested is not None and requested != saved_value:
                        raise ValueError(f"Cannot change force_context.{name} on resume; start a new run")
            saved_phases = restored["cfg"].get("distillation", {}).get("token_region_phases")
            saved_phases = None if saved_phases is None else tuple(sorted(saved_phases))
            self.selection_metric_changed = saved_phases != self.token_region_phases
            if self.selection_metric_changed:
                # Retain learned weights and optimizer momentum, but establish
                # a new baseline for the newly selected observations.
                self.initial_train_mse = self.initial_val_mse = self.latest_val_loss = None
                self.best_val_loss = self.early_stopping_best_loss = float("inf")
                self.best_epoch = self.best_step = self.epochs_without_improvement = 0
                if self.lr_scheduler is not None:
                    self.lr_scheduler.best = float("inf")
                    self.lr_scheduler.num_bad_epochs = self.lr_scheduler.cooldown_counter = 0
                log.info("Token/region supervision phases changed: %s -> %s; "
                         "recompute validation baseline and reset best/plateau counters",
                         saved_phases, self.token_region_phases)
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
                if self.render_stages_enabled and self.epoch < self.condition_warmup_epochs:
                    self.training_stage = "condition_warmup"
                elif (self.render_stages_enabled and self.rgb_warmup_epochs > 0
                      and self.epoch < self.condition_warmup_epochs + self.rgb_warmup_epochs):
                    self.training_stage = "rgb"
                else:
                    self.training_stage = "joint" if self.joint_after_rgb else "rgb"
                self.stage_transition_reason = "legacy_checkpoint_epoch_inference"
            if hasattr(self.model, "set_training_stage"):
                self.model.set_training_stage(self.training_stage)
            if getattr(self, "ema", None) is not None:
                # Keep frozen teacher-initialized encoders identical in the
                # online and EMA models before validation or another update.
                self.ema.sync_frozen_parameters(self.model)
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
        if self.num_epochs < 1 or any(
                value is not None and int(value) < 1
                for value in (self.max_steps, self.max_val_batches)):
            raise ValueError("num_epochs must be positive; "
                             "max_steps/max_val_batches must be positive or null")
        if self.render_stages_enabled and (
                self.condition_warmup_epochs + self.rgb_warmup_epochs > self.num_epochs):
            raise ValueError("training render warmup stages cannot exceed training.num_epochs")
        if (self.strict_two_stage
                and self.condition_warmup_epochs + max(1, self.rgb_warmup_epochs) > self.num_epochs):
            raise ValueError("Strict two-stage training must reserve at least one render epoch")
        if (self.distributed_context.enabled and self.render_stages_enabled
                and self.condition_warmup_epochs > 0
                and not bool(cfg.training.get("distributed", {}).get("find_unused_parameters", False))):
            raise ValueError(
                "Condition-only warmup with DDP requires training.distributed.find_unused_parameters=true"
            )
        if not resume_path:
            if hasattr(self.model, "set_training_stage"):
                self.model.set_training_stage(self.training_stage)
        if getattr(self, "ema", None) is not None:
            self.ema.sync_frozen_parameters(self.model)
        if self.patience < 1 or not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("early_stopping.patience must be positive; min_delta must be finite and nonnegative")
        log.info(f"Teacher: {checkpoint} ({self.teacher_state_key}), cross_attn_query=image")
        log.info("Trainable parameters: teacher=0, RenderNet=%s; RenderNet total=%s",
                 f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}",
                 f"{sum(p.numel() for p in self.model.parameters()):,}")
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
                 "teacher image encoder and action head stay frozen, student input gradients remain enabled",
                 self.objective, self.distillation_config.get("token_loss_weight", 0.0),
                 self.distillation_config.get("demo_loss_weight", 0.0))
        log.info(f"Device={self.device}; AMP={self.amp_enabled} {self.amp_dtype}; "
              f"GPU={torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else 'CPU'}")
        log.info(f"Max epochs={self.num_epochs}; max steps={self.max_steps}; "
                 f"condition warmup={self.render_stages_enabled} "
                 f"(condition_epochs={self.condition_warmup_epochs}, "
                 f"rgb_epochs={self.rgb_warmup_epochs}, joint_after_rgb={self.joint_after_rgb}, "
                 f"current={self.training_stage}); "
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
    def use_teacher_ffn(self):
        # Historical student checkpoints retain their original visual loss path.
        return bool(self.distillation_config.get("use_teacher_ffn", False))

    @property
    def force_context_config(self):
        return self.distillation_config.get("force_context", {})

    @property
    def force_context_enabled(self):
        return bool(self.force_context_config.get("enabled", False))

    @property
    def token_region_phases(self):
        phases = self.distillation_config.get("token_region_phases")
        return None if phases is None else tuple(sorted(set(int(phase) for phase in phases)))

    @property
    def skip_unselected_observations(self):
        return bool(self.distillation_config.get("skip_unselected_observations", False))

    @property
    def strict_two_stage(self):
        return (getattr(self, "render_stages_enabled", False)
                and not getattr(self, "joint_after_rgb", True))

    @property
    def condition_selection(self):
        return self.strict_two_stage and getattr(self, "training_stage", "joint") == "condition_warmup"

    @property
    def metric_name(self):
        if self.force_context_enabled:
            return "force context loss"
        if self.condition_selection:
            return "condition loss (CE + alpha + region)"
        return "denoiser MSE" if self.objective == "denoiser_mse" else "context token MSE"

    @property
    def metric_scope(self):
        phases = self.token_region_phases
        return "all" if phases is None else "phases_" + "_".join(map(str, phases))

    _REGION_TOKEN_METRICS = frozenset({
        "token_mse", "mask_mse", "context_patch_mse", "intensity_smoothness", "intensity_loss",
    })

    def _component_weight(self, key, size):
        if key in self._REGION_TOKEN_METRICS:
            return getattr(self, "loss_metrics", {}).get("token_region_observation_count", size)
        return size

    def _selection_weight(self, size):
        if self.condition_selection or (self.model.training and getattr(self, "training_stage", "joint") == "condition_warmup"):
            return size
        return self._component_weight("token_mse", size)

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
        phases = self.distillation_config.get("token_region_phases")
        if phases is not None:
            if (not isinstance(phases, (list, tuple)) and not OmegaConf.is_list(phases)
                    or not phases or any(isinstance(phase, bool) or not isinstance(phase, int)
                                         or phase not in (0, 1, 2) for phase in phases)):
                raise ValueError("distillation.token_region_phases must be null or a nonempty list of 0, 1, 2")
            if self.objective != "token_mse":
                raise ValueError("distillation.token_region_phases currently requires objective=token_mse")
        if self.skip_unselected_observations and (
                self.objective != "token_mse" or phases is None
                or not self.cfg.get("render_net", {}).get("phase_prediction", {}).get("enabled", False)):
            raise ValueError("skip_unselected_observations requires token_mse, "
                             "token_region_phases and phase prediction")
        for name in ("token_loss_weight", "demo_loss_weight", "phase_ce_weight", "alpha_loss_weight"):
            value = float(self.distillation_config.get(name, 0.0))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"distillation.{name} must be finite and nonnegative")
        if self.force_context_enabled:
            render = self.cfg.get("render_net", {})
            if (self.objective != "token_mse" or self.token_region_phases is not None
                    or self.skip_unselected_observations or self.use_teacher_ffn
                    or not render.get("cross_camera_attn_layers", 0)
                    or render.get("high_attention_render", {}).get("enabled", False)
                    or render.get("phase_prediction", {}).get("enabled", False)):
                raise ValueError("force_context requires all-observation cross-camera RGB token distillation without condition heads")
            for name in ("difference_weight", "region_weight"):
                value = float(self.force_context_config.get(name, 1.0))
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"force_context.{name} must be finite and nonnegative")
            for name in ("difference_scale", "region_scale"):
                value = self.force_context_config.get(name)
                if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                    raise ValueError(f"force_context.{name} must be positive and finite or null")
            cache = self.cfg.get("teacher_cache", {})
            if cache.get("enabled", False) and (
                    cache.get("dtype", "float32") != "float32"
                    or cache.get("precision", "model") != "float32"):
                raise ValueError("force_context teacher pairs require float32 cache dtype and precision")

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
            # The robot dataset converts RAM preload into a shared tmpfs cache
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
        # Register DDP reducer hooks for the joint graph before the condition
        # warmup temporarily freezes RGB-only parameters. They are re-enabled
        # after warmup and need reducer hooks already installed.
        if hasattr(self.model, "set_training_stage"):
            self.model.set_training_stage("joint")
        self._training_model = DistributedDataParallel(
            self.model,
            device_ids=[self.distributed_context.local_rank] if self.device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=bool(config.get("find_unused_parameters", False)),
        )
        if hasattr(self.model, "set_training_stage"):
            self.model.set_training_stage(self.training_stage)

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
        render_dynamic = False
        if config.get("render_backbone", True):
            if not isinstance(self.model, RenderNet):
                raise ValueError("Compiling requires the active RenderNet backbone")
            # The independent RGB path receives only selected observations, so
            # its batch dimension varies. The shared path encodes the full batch
            # and already compiles its selected-observation decoder dynamically.
            render_dynamic = (self.skip_unselected_observations
                              and not self.model.shared_condition_backbone)
            self.model.compile_backbone(**{**options, "dynamic": render_dynamic})
        if config.get("student_dino", True):
            image_options = {**options, "dynamic": self.skip_unselected_observations}
            self._student_image_encoder = torch.compile(
                self.teacher_policy.obs_encoder.encode_image_context, **image_options,
            )
        log.info("torch.compile enabled for training: render_backbone=%s, student_dino=%s, "
                 "mode=%s, render_dynamic=%s; first training batch includes compilation; "
                 "validation uses eager execution",
                 config.get("render_backbone", True), config.get("student_dino", True),
                 options["mode"], render_dynamic)

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
        return (not self.model.training) or getattr(self, "training_stage", "joint") == "joint"

    def _token_region_observation_mask(self, obs, tokens):
        steps = getattr(self.teacher_policy, "n_obs_steps", None)
        if steps is None:
            # Legacy render-only teacher adapters expose temporal observations
            # but do not carry the full policy's window configuration.
            steps = next(iter(obs.values())).shape[1]
        steps = int(steps)
        shape = (tokens.shape[0], steps)
        if self.token_region_phases is None:
            return torch.ones(shape, dtype=torch.bool, device=tokens.device)
        if "contact_phase" not in obs:
            raise KeyError("token_region_phases requires dataset contact_phase labels")
        labels = obs["contact_phase"]
        if labels.ndim != 2 or labels.shape[0] != shape[0] or labels.shape[1] < steps:
            raise ValueError("contact_phase labels must match the [B, To] observations")
        labels = labels[:, :steps].to(tokens.device)
        selected = torch.zeros(shape, dtype=torch.bool, device=tokens.device)
        for phase in self.token_region_phases:
            selected |= labels == phase
        return selected

    @staticmethod
    def _selected_observation_mean(values, selected):
        values = values[selected]
        # Sum over an empty selection is a graph-connected zero, unlike an
        # empty mean. Free-only batches can still train the phase/alpha head.
        return values.sum() / max(values.numel(), 1)

    @classmethod
    def _selected_observation_mse(cls, prediction, target, selected):
        difference = prediction.float() - target.float()
        return cls._selected_observation_mean(difference.square(), selected)

    @classmethod
    def _selected_region_smoothness(cls, masks, selected):
        values = []
        for mask in masks.values():
            values.append(cls._selected_observation_mean(
                mask[..., 1:, :].sub(mask[..., :-1, :]).abs(), selected))
            values.append(cls._selected_observation_mean(
                mask[..., :, 1:].sub(mask[..., :, :-1]).abs(), selected))
        return torch.stack(values).mean()

    @staticmethod
    def _contact_token_metrics(student_tokens, teacher_tokens, contact_mask, measured_observations=None):
        """Return contact ratio and token MSE split without changing gradients."""
        if contact_mask is None:
            contact_mask = torch.ones(
                student_tokens.shape[0], 1, device=student_tokens.device, dtype=torch.bool,
            )
        if contact_mask.ndim == 3:
            contact_mask = contact_mask.any(dim=-1)
        if contact_mask.ndim != 2:
            raise ValueError("contact mask must have shape [B, To] or [B, To, K]")
        if student_tokens.shape[0] != contact_mask.shape[0]:
            raise ValueError("contact mask batch size does not match context tokens")
        steps = contact_mask.shape[1]
        if steps < 1 or student_tokens.shape[1] % steps:
            raise ValueError("context token layout is not divisible by observation steps")
        tokens_per_step = student_tokens.shape[1] // steps
        token_open = contact_mask.to(torch.bool).unsqueeze(-1).expand(
            -1, -1, tokens_per_step,
        ).reshape(student_tokens.shape[0], -1)
        per_token = (student_tokens.float() - teacher_tokens.float()).square().mean(dim=-1)
        token_measured = torch.ones_like(token_open)
        if measured_observations is not None:
            if measured_observations.shape != contact_mask.shape:
                raise ValueError("measured observation mask must match contact mask")
            token_measured = measured_observations.to(torch.bool).unsqueeze(-1).expand(
                -1, -1, tokens_per_step,
            ).reshape_as(token_open)
        open_values = per_token[token_open & token_measured]
        closed_values = per_token[~token_open & token_measured]
        zero = per_token.new_zeros(())
        open_mse = open_values.mean() if open_values.numel() else zero
        closed_mse = closed_values.mean() if closed_values.numel() else zero
        return {
            "contact_open_ratio": contact_mask.float().mean(),
            "open_token_mse": open_mse,
            "closed_token_mse": closed_mse,
            "open_token_count": float(open_values.numel()),
            "closed_token_count": float(closed_values.numel()),
        }

    def _remember_best_condition(self):
        """Embed the chosen evaluation snapshot so a resumed run is self-contained."""
        if not self.condition_selection:
            return
        selected = self.ema_model if getattr(self, "ema_model", None) is not None else self.model
        self.condition_best_state = {key: value.detach().cpu().clone()
                                     for key, value in selected.state_dict().items()}
        self.condition_best_metric = self.best_val_loss
        self.condition_best_epoch = self.epoch

    def _condition_validation_loss(self, metrics):
        return (float(self.distillation_config.get("phase_ce_weight", 1.0)) * metrics["phase_ce"]
                + float(self.distillation_config.get("alpha_loss_weight", 1.0)) * metrics["alpha_mse"]
                + metrics["intensity_loss"])

    def _reset_stage_optimization(self):
        # The renderer has a different objective and new trainable parameters.
        # Do not carry condition plateaus, LR reductions or Adam moments over.
        self.optimizer.state.clear()
        for group in self.optimizer.param_groups:
            group["lr"] = float(self.cfg.optimizer.lr)
        if self.lr_scheduler is not None:
            self.lr_scheduler.best = float("inf")
            self.lr_scheduler.num_bad_epochs = self.lr_scheduler.cooldown_counter = 0
            self.lr_scheduler.last_epoch = 0
            self.lr_scheduler._last_lr = [group["lr"] for group in self.optimizer.param_groups]
        self.best_val_loss = self.early_stopping_best_loss = float("inf")
        self.best_epoch = self.best_step = self.epochs_without_improvement = 0
        self.latest_val_loss = None
        if getattr(self, "ema", None) is not None:
            self.ema.optimization_step = 0
            self.ema.decay = 0.0

    def _maybe_transition_stage(self, completed_epochs=None):
        """Advance strict condition -> RGB, or the legacy condition -> RGB -> joint."""
        if not getattr(self, "render_stages_enabled", False):
            desired = "joint"
        else:
            completed = self.epoch if completed_epochs is None else int(completed_epochs)
            condition_epochs = getattr(self, "condition_warmup_epochs", 0)
            rgb_epochs = getattr(self, "rgb_warmup_epochs", 0)
            if (self.training_stage == "condition_warmup"
                    and (completed >= condition_epochs or getattr(self, "condition_transition_pending", False))):
                desired = "rgb" if self.strict_two_stage or rgb_epochs > 0 else "joint"
            elif self.training_stage == "rgb" and completed >= condition_epochs + rgb_epochs:
                desired = "joint" if self.joint_after_rgb else "rgb"
            else:
                desired = self.training_stage
        if desired == self.training_stage:
            if hasattr(self.model, "set_training_stage"):
                self.model.set_training_stage(self.training_stage)
            if getattr(self, "ema", None) is not None:
                self.ema.sync_frozen_parameters(self.model)
            return False
        previous = self.training_stage
        condition_plateau = getattr(self, "condition_transition_pending", False)
        if self.strict_two_stage and previous == "condition_warmup":
            state = getattr(self, "condition_best_state", None)
            if state is not None:
                # Validation selects EMA when enabled. Freeze that exact state,
                # rather than replacing it with the last online condition net.
                self.model.load_state_dict(state)
                if getattr(self, "ema_model", None) is not None:
                    self.ema_model.load_state_dict(state)
            self._reset_stage_optimization()
        self.training_stage = desired
        self.stage_step = 0
        self.stage_epoch = 0
        reason = "condition_plateau" if condition_plateau else "epoch_boundary"
        self.stage_transition_reason = f"{reason}:{previous}->{desired}"
        self.condition_transition_pending = False
        initialized_rgb = False
        if (previous == "condition_warmup" and desired in {"rgb", "joint"}
                and hasattr(self.model, "initialize_rgb_from_gate")):
            # Warm-start a separate RGB branch when the architecture provides
            # one; shared-backbone RenderNet correctly treats this as a no-op.
            self.model.initialize_rgb_from_gate()
            initialized_rgb = True
        if hasattr(self.model, "set_training_stage"):
            self.model.set_training_stage(desired)
            gate_count = sum(p.numel() for p in self.model.gate_parameters())
            rgb_count = sum(p.numel() for p in self.model.rgb_parameters())
        else:
            gate_count = rgb_count = 0
        if getattr(self, "ema", None) is not None:
            # A copied RGB initialization must also start from the same
            # state in EMA; conditions remain trainable after this boundary.
            if initialized_rgb:
                self.ema.sync_parameters(self.model)
            else:
                self.ema.sync_frozen_parameters(self.model)
        log.info("RenderNet training stage transition: %s -> %s at completed_epoch=%d; "
                 "condition_parameters=%d, rgb_parameters=%d",
                 previous, desired, self.epoch, gate_count, rgb_count)
        return True

    def _force_context_loss(self, obs, cached_targets=None):
        if self.force_difference_scale is None or self.force_region_scale is None:
            raise RuntimeError("Prepare training teacher-pair statistics before force-context training")
        targets = teacher_context_pair(self.teacher_policy, obs) if cached_targets is None else cached_targets
        teacher_tokens = targets["tokens"].to(self.device, non_blocking=True).detach()
        teacher_zero = targets["zero_tokens"].to(self.device, non_blocking=True).detach()
        zero_obs = raw_zero_wrench_observations(self.teacher_policy, obs)
        # One DDP forward includes both branches, so gradients from the
        # difference share one reducer graph. Zero is applied before the
        # teacher's normalizer and the student's frozen force encoder.
        paired_obs = {key: torch.cat((obs[key], value), dim=0) for key, value in zero_obs.items()}
        batch_size = teacher_tokens.shape[0]
        training = self.model.training and torch.is_grad_enabled()
        renderer = getattr(self, "_training_model", None) if training else getattr(self, "ema_model", None)
        if renderer is None:
            renderer = self.model
        compiled_encoder = getattr(self, "_student_image_encoder", None) if training else None
        with self._autocast():
            tokens, paired_rgb = student_context_tokens(
                self.teacher_policy, renderer, paired_obs,
                image_context_encoder=compiled_encoder, return_rendered=True,
            )
        student_tokens, student_zero = tokens.split(batch_size)
        encoder = self.teacher_policy.obs_encoder
        steps = self.teacher_policy.n_obs_steps
        images = {key: obs[key][:, :steps] for key in encoder.image_keys}
        rendered = {key: value[:batch_size] for key, value in paired_rgb.items()}
        components = compute_force_context_losses(
            student_tokens, student_zero, teacher_tokens, teacher_zero,
            rendered, images, encoder, steps,
            difference_scale=self.force_difference_scale, region_scale=self.force_region_scale,
        )
        loss = (components["token_mse"]
                + float(self.force_context_config.get("difference_weight", 1.0)) * components["force_delta_loss"]
                + float(self.force_context_config.get("region_weight", 1.0)) * components["context_region_loss"])
        self.loss_metrics = {name: float(value.detach()) for name, value in components.items()}
        self.loss_metrics.update(
            loss=float(loss.detach()), token_mse_measured=True,
            token_region_observation_count=batch_size * steps,
            training_stage=self.training_stage, joint_training=True,
            mask_supervised=False, mask_measured=False,
        )
        if self.shape_info is None:
            self.shape_info = {
                "teacher_token_shape": list(teacher_tokens.shape),
                "student_token_shape": list(student_tokens.shape),
                "camera_keys": list(encoder.image_keys), "camera_count": len(encoder.image_keys),
                "observation_steps": steps,
                "patch_tokens_per_camera": dict(encoder.context_patch_counts),
                "patch_tokens_per_observation": teacher_tokens.shape[1] // steps,
                "wrench_key": self.teacher_policy.wrench_key,
                "wrench_shape": list(obs[self.teacher_policy.wrench_key].shape),
            }
            log.info("Token layout: " + json.dumps(self.shape_info))
        return loss

    def _loss(self, obs, cached_targets=None, *, action=None):
        if self.force_context_enabled:
            return self._force_context_loss(obs, cached_targets)
        training = self.model.training and torch.is_grad_enabled()
        condition_warmup = (self.condition_selection or
                            training and getattr(self, "training_stage", "joint") == "condition_warmup")
        supervise_regions = self.attention_enabled and training
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
            observation_mask = self._token_region_observation_mask(obs, teacher_tokens)
            observation_count = int(observation_mask.sum())
            sparse_images = training and self.skip_unselected_observations
            condition_only = condition_warmup or (sparse_images and observation_count == 0)
            # Build indices outside the compiled DINO graph. Conditions always
            # see all observations; RGB decoding/DINO need only supervised rows.
            selected_indices = (observation_mask.flatten().nonzero(as_tuple=True)[0]
                                if sparse_images and not condition_only else None)
            token_region_gradient_scale = 1.0
            if training and self.token_region_phases is not None and self.distributed_context.enabled:
                # DDP averages rank gradients. Weight the local mean so ranks
                # with few contact observations cannot outweigh other ranks.
                global_count = self._sum([observation_count])[0]
                token_region_gradient_scale = (observation_count * self.distributed_context.world_size
                                               / max(global_count, 1))
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
            phase_enabled = bool(getattr(self.model, "phase_prediction_enabled", False))
            student_result = student_context_tokens(
                self.teacher_policy, renderer, obs,
                return_aux=True, return_contact=True, image_context_encoder=compiled_encoder,
                use_teacher_ffn=self.use_teacher_ffn,
                return_phase=phase_enabled,
                condition_only=condition_only,
                observation_mask=observation_mask if sparse_images else None,
                selected_observation_indices=selected_indices,
            )
            phase_predictions = None
            if phase_enabled:
                student_tokens, masks, contact_mask, phase_predictions = student_result
            else:
                student_tokens, masks, contact_mask = student_result
            if condition_only:
                token_mse = teacher_tokens.new_zeros(())
            else:
                if student_tokens.shape != teacher_tokens.shape:
                    raise RuntimeError(
                        f"Token shapes differ: teacher={teacher_tokens.shape}, student={student_tokens.shape}"
                    )
                steps = observation_mask.shape[1]
                if student_tokens.shape[1] % steps:
                    raise ValueError("Context token layout must divide evenly into observation steps")
                token_mse = self._selected_observation_mse(
                    student_tokens.reshape(student_tokens.shape[0], steps, -1, student_tokens.shape[-1]),
                    teacher_tokens.reshape(teacher_tokens.shape[0], steps, -1, teacher_tokens.shape[-1]),
                    observation_mask,
                )
            if phase_enabled and "contact_alpha" in obs:
                contact_mask = obs["contact_alpha"][:, :self.teacher_policy.n_obs_steps] > 0
            contact_metrics = self._contact_token_metrics(
                teacher_tokens if condition_only else student_tokens,
                teacher_tokens, contact_mask,
                measured_observations=(torch.zeros_like(observation_mask) if condition_only
                                       else observation_mask if sparse_images else None),
            )
            mask_mse = token_mse.new_zeros(())
            patch_mse = token_mse.new_zeros(())
            smoothness = token_mse.new_zeros(())
            if measure_regions and target_masks is not None:
                region_context = contextlib.nullcontext() if supervise_regions else torch.no_grad()
                with region_context:
                    mask_mse = torch.stack([
                        self._selected_observation_mse(masks[key], target_masks[key], observation_mask)
                        for key in masks
                    ]).mean()
                    smoothness = self._selected_region_smoothness(masks, observation_mask)
                    predicted_patches = self._patch_pool(masks)
                    target_patch_weights = self._region_patch_targets(target_patches)
                    patch_losses = []
                    for key in predicted_patches:
                        target = target_patch_weights[key].to(
                            predicted_patches[key].device, predicted_patches[key].dtype,
                        )
                        patch_losses.append(self._selected_observation_mean(
                            F.smooth_l1_loss(predicted_patches[key], target, reduction="none"),
                            observation_mask,
                        ))
                    patch_mse = torch.stack(patch_losses).mean()
            denoiser_mse = demo_mse = token_mse.new_zeros(())
            if self.objective == "denoiser_mse" and not condition_warmup:
                denoiser_mse, demo_mse = self._denoiser_losses(
                    action, student_tokens, teacher_tokens.to(student_tokens.dtype),
                )
            region_loss = (
                float(getattr(self, "context_patch_loss_weight", 1.0)) * patch_mse
                + float(getattr(self, "intensity_loss_weight", 1.0)) * mask_mse * float(
                    self.attention_config.get("pixel_loss_weight", 0.0)
                )
                + float(getattr(self, "smoothness_loss_weight", 0.02)) * smoothness
            ) if measure_regions else token_mse.new_zeros(())
            phase_ce = alpha_mse = alpha_mae = token_mse.new_zeros(())
            phase_accuracy = token_mse.new_zeros(())
            if phase_enabled:
                if phase_predictions is None:
                    raise RuntimeError("Phase-enabled RenderNet did not return condition predictions")
                if "contact_phase" not in obs or "contact_alpha" not in obs:
                    raise KeyError(
                        "Phase prediction training requires offline contact_phase/contact_alpha targets"
                    )
                phase_logits = phase_predictions["phase_logits"].float()
                alpha_prediction = phase_predictions["contact_alpha"].float()
                phase_target = obs["contact_phase"][:, :phase_logits.shape[1]].long()
                alpha_target = obs["contact_alpha"][:, :alpha_prediction.shape[1]].float()
                if phase_target.shape != phase_logits.shape[:2] or alpha_target.shape != alpha_prediction.shape:
                    raise ValueError("Phase and alpha labels must match [B, To] predictions")
                phase_ce = F.cross_entropy(phase_logits.flatten(0, 1), phase_target.flatten())
                alpha_mse = F.mse_loss(alpha_prediction, alpha_target)
                alpha_mae = F.l1_loss(alpha_prediction, alpha_target)
                phase_accuracy = (phase_logits.argmax(dim=-1) == phase_target).float().mean()
            condition_loss = (
                float(self.distillation_config.get("phase_ce_weight", 1.0)) * phase_ce
                + float(self.distillation_config.get("alpha_loss_weight", 1.0)) * alpha_mse
            )
            if condition_warmup:
                # Warmup trains the condition field; the renderer is bypassed.
                loss = condition_loss + region_loss * token_region_gradient_scale
            elif getattr(self, "training_stage", "joint") == "rgb":
                # RGB warmup isolates the inverse-rendering path after the
                # condition field has converged.  The stage setter freezes the
                # condition modules, so auxiliary condition losses are omitted.
                loss = token_mse * token_region_gradient_scale
            elif self.objective == "denoiser_mse":
                loss = (denoiser_mse
                        + float(self.distillation_config.get("token_loss_weight", 0.0)) * token_mse
                        + float(self.distillation_config.get("demo_loss_weight", 0.0)) * demo_mse
                        + (condition_loss + region_loss if training else 0))
            else:
                # During joint optimization retain condition supervision so
                # the shared feature trunk and all condition outputs adapt
                # alongside the RGB residual network.
                loss = token_mse * token_region_gradient_scale + (
                    condition_loss + region_loss * token_region_gradient_scale if training else 0)
            # A sparse RGB batch can contain no selected contact observations.
            # In that case ``condition_only`` intentionally skips the renderer
            # and ``token_mse`` is a detached scalar zero.  Keep the optimizer
            # step valid by attaching a zero-valued anchor to the parameters
            # that are trainable in the current stage.
            if training and not loss.requires_grad:
                trainable = [parameter for parameter in self.model.parameters()
                             if parameter.requires_grad]
                if trainable:
                    loss = loss + sum(parameter.reshape(-1)[0] * 0 for parameter in trainable)
            self.loss_metrics = {"token_mse": float(token_mse.detach()),
                                 "token_mse_measured": not condition_warmup and observation_count > 0,
                                 "token_region_observation_count": observation_count,
                                 "denoiser_mse": float(denoiser_mse.detach()),
                                 "demo_mse": float(demo_mse.detach()),
                                 "mask_mse": float(mask_mse.detach()),
                                 "context_patch_mse": float(patch_mse.detach()),
                                 "intensity_smoothness": float(smoothness.detach()),
                                 "intensity_loss": float(region_loss.detach()),
                                 "phase_ce": float(phase_ce.detach()),
                                 "alpha_mse": float(alpha_mse.detach()),
                                 "alpha_mae": float(alpha_mae.detach()),
                                 "phase_accuracy": float(phase_accuracy.detach()),
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
                "student_token_shape": list(teacher_tokens.shape if condition_only else student_tokens.shape),
                "camera_keys": list(self.teacher_policy.obs_encoder.image_keys),
                "camera_count": len(self.teacher_policy.image_keys),
                "observation_steps": self.teacher_policy.n_obs_steps,
                "patch_tokens_per_camera": dict(getattr(
                    self.teacher_policy.obs_encoder, "context_patch_counts",
                    self.teacher_policy.obs_encoder.patch_counts,
                )),
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
        metrics = getattr(self, "loss_metrics", {})
        if not metrics.get("token_mse_measured", True):
            return float(loss.detach())
        return metrics.get("token_mse", float(loss.detach()))

    def _selection_mse(self, loss):
        if self.force_context_enabled:
            return float(loss.detach())
        metrics = getattr(self, "loss_metrics", {})
        if self.condition_selection:
            return self._condition_validation_loss(metrics) if "phase_ce" in metrics else float(loss.detach())
        if (not metrics.get("token_mse_measured", True)
                and self.model.training
                and getattr(self, "training_stage", "joint") == "condition_warmup"):
            return float(loss.detach())
        return metrics.get(self.objective, float(loss.detach()))

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
            "dataset": OmegaConf.to_container(dataset_cfg, resolve=True),
            "source": dataset_source_signature(dataset.dataset_path),
            "amp_enabled": self.amp_enabled, "amp_dtype": str(self.amp_dtype),
            "allow_tf32": bool(self.cfg.training.amp.allow_tf32),
            "capture_attention": self.attention_enabled,
            "attention_seed": int(self.attention_config.get("teacher_seed", 42)),
            "cache_batch_size": int(config["batch_size"]),
        }
        if self.force_context_enabled:
            identity["target_contract"] = "visual_context_real_raw_zero_pair_v1"
            identity["target_precision"] = "float32"
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
                "context_effect_v2" if region_source == "context" else "noise_attention_v1"
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
                                zero_tokens = None
                                if self.force_context_enabled:
                                    pair = teacher_context_pair(self.teacher_policy, obs)
                                    tokens, zero_tokens, attention = pair["tokens"], pair["zero_tokens"], {}
                                elif self.attention_enabled:
                                    tokens, attention = self._region_targets(obs, return_patch_attention=True)
                                else:
                                    tokens = self.teacher_policy._encode_context(obs, apply_curriculum_mask=False).detach()
                                    attention = {}
                            if self.force_context_enabled:
                                cache.append(tokens, attention, zero_tokens=zero_tokens)
                            else:
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

    def _prepare_force_context_scales(self, train_dataset):
        """Fix both scales from training targets; validation never fits them."""
        if not self.force_context_enabled:
            return
        status = [None]
        if self.is_main_process:
            try:
                difference = getattr(self, "force_difference_scale", None)
                region = getattr(self, "force_region_scale", None)
                statistics = getattr(self, "force_context_statistics", None)
                if difference is None or region is None:
                    if not isinstance(train_dataset, CachedTeacherDataset):
                        raise ValueError("Automatic force-context scales require the completed training teacher cache")
                    statistics = train_dataset.cache.force_difference_statistics()
                    if difference is None:
                        difference = max(statistics["mean_square"], 1e-8)
                    if region is None:
                        region = max(math.sqrt(statistics["mean_square"]), 1e-4)
                statistics = {**(statistics or {}), "source": "train", "baseline": "raw_zero",
                              "difference_scale": float(difference), "region_scale": float(region)}
                path = pathlib.Path(self.output_dir, "force_context_statistics.json")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(statistics, indent=2, allow_nan=False) + "\n")
                status[0] = {"ok": True, "statistics": statistics}
            except Exception as exc:
                if not self.distributed_context.enabled:
                    raise
                status[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if self.distributed_context.enabled:
            self._broadcast_status(status)
        self.force_context_statistics = status[0]["statistics"]
        self.force_difference_scale = self.force_context_statistics["difference_scale"]
        self.force_region_scale = self.force_context_statistics["region_scale"]
        with open_dict(self.cfg.distillation.force_context):
            self.cfg.distillation.force_context.difference_scale = self.force_difference_scale
            self.cfg.distillation.force_context.region_scale = self.force_region_scale
        if self.is_main_process:
            OmegaConf.save(self.cfg, pathlib.Path(self.output_dir, "config.yaml"), resolve=True)
        log.info("Force-context scales fixed from training targets: difference=%.8g, region=%.8g",
                 self.force_difference_scale, self.force_region_scale)

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

    def _limit_debug_dataset(self, dataset, limit, *, name):
        """Return a deterministic tiny view for smoke/ablation runs."""
        if limit is None:
            return dataset
        limit = int(limit)
        if limit < 1 or len(dataset) <= limit:
            return dataset
        labels = (dataset.get_phase_labels() if hasattr(dataset, "get_phase_labels")
                  else np.zeros(len(dataset), dtype=np.int8))
        phases = self.token_region_phases
        if phases is not None:
            candidates = np.flatnonzero(np.isin(labels, np.asarray(phases, dtype=np.int8)))
        else:
            candidates = np.arange(len(dataset), dtype=np.int64)
        if len(candidates) == 0:
            candidates = np.arange(len(dataset), dtype=np.int64)
        positions = np.linspace(0, len(candidates) - 1,
                                num=min(limit, len(candidates)), dtype=np.int64)
        selected = np.unique(candidates[positions])
        if len(selected) < limit:
            remaining = np.setdiff1d(np.arange(len(dataset), dtype=np.int64), selected,
                                     assume_unique=False)
            selected = np.concatenate([selected, remaining[:limit - len(selected)]])
        selected.sort()
        subset = copy.copy(dataset)
        subset.indices = np.ascontiguousarray(dataset.indices[selected])
        subset.phase_labels = np.asarray(labels[selected], dtype=np.int8).copy()
        phase_values, phase_counts = np.unique(subset.phase_labels, return_counts=True)
        log.info("Debug %s subset: %d/%d samples; phase_counts=%s", name, len(subset),
                 len(dataset), dict(zip(phase_values.tolist(), phase_counts.tolist())))
        return subset

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
        total, count, selection_count = 0.0, 0, 0.0
        region_count = 0.0
        component_keys = (
            "token_mse", "denoiser_mse", "demo_mse", "mask_mse",
            "context_patch_mse", "intensity_smoothness", "intensity_loss",
            "phase_ce", "alpha_mse", "alpha_mae", "phase_accuracy", "contact_open_ratio",
            "force_delta_mse", "force_delta_loss", "context_region_loss", "loss",
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
            selected = self._selection_weight(size)
            total += self._selection_mse(loss) * selected
            selection_count += selected
            region_count += self._component_weight("token_mse", size)
            for key in component_totals:
                component_totals[key] += getattr(self, "loss_metrics", {}).get(key, 0.0) * self._component_weight(key, size)
            metrics = getattr(self, "loss_metrics", {})
            open_token_sum += metrics.get("open_token_mse", 0.0) * metrics.get("open_token_count", 0.0)
            closed_token_sum += metrics.get("closed_token_mse", 0.0) * metrics.get("closed_token_count", 0.0)
            open_token_count += metrics.get("open_token_count", 0.0)
            closed_token_count += metrics.get("closed_token_count", 0.0)
            count += size
            if self.force_context_enabled:
                progress.set_postfix(
                    loss=f"{total / max(selection_count, 1):.6f}",
                    token=f"{component_totals['token_mse'] / max(region_count, 1):.5f}",
                    delta=f"{component_totals['force_delta_loss'] / max(count, 1):.5f}",
                    region=f"{component_totals['context_region_loss'] / max(count, 1):.5f}",
                    refresh=False,
                )
            else:
                progress.set_postfix(mse=f"{total / max(selection_count, 1):.6f}", refresh=False)
            now = time.monotonic()
            if not self._progress_enabled() and self.is_main_process and now - last_status >= self._status_interval():
                if self.force_context_enabled:
                    log.info("Val %d | batch=%d/%d loss=%.6f token=%.6f delta=%.6f region=%.6f elapsed=%.1fs",
                             self.epoch, index + 1, batches, total / max(selection_count, 1),
                             component_totals["token_mse"] / max(region_count, 1),
                             component_totals["force_delta_loss"] / max(count, 1),
                             component_totals["context_region_loss"] / max(count, 1), now - started)
                else:
                    log.info("Val %d | batch=%d/%d mse=%.6f elapsed=%.1fs",
                             self.epoch, index + 1, batches, total / max(selection_count, 1), now - started)
                last_status = now
        progress.close()
        total, count, selection_count, region_count = self._sum([total, count, selection_count, region_count])
        count = int(count)
        if count == 0:
            raise RuntimeError("Validation split is empty")
        if selection_count == 0:
            raise RuntimeError("Validation split has no observations in distillation.token_region_phases")
        self.validation_samples = count
        self.validation_selected_observations = int(region_count)
        self.validation_seconds = self._max_seconds(time.monotonic() - started)
        self.validation_samples_per_second = count / max(self.validation_seconds, 1e-6)
        totals = self._sum(list(component_totals.values()) + [
            open_token_sum, closed_token_sum, open_token_count, closed_token_count,
        ])
        component_count = len(component_totals)
        self.validation_metrics = {
            key: value / max(region_count if key in self._REGION_TOKEN_METRICS else count, 1)
            for key, value in zip(component_totals, totals[:component_count])
        }
        open_sum, closed_sum, open_count, closed_count = totals[component_count:]
        self.validation_metrics["open_token_mse"] = open_sum / open_count if open_count else 0.0
        self.validation_metrics["closed_token_mse"] = closed_sum / closed_count if closed_count else 0.0
        self.validation_metrics["open_token_count"] = open_count
        self.validation_metrics["closed_token_count"] = closed_count
        self.validation_metrics["token_mse_measured"] = not self.condition_selection
        # Region terms are averaged over selected contact observations, while
        # phase/alpha use all windows. Combine their properly reduced means.
        if self.condition_selection:
            return self._condition_validation_loss(self.validation_metrics)
        return total / selection_count

    def _summary(self, status):
        if not self.is_main_process:
            return
        summary = {
            "status": status, "global_step": self.global_step,
            "metric_scope": self.metric_scope,
            "epoch": self.epoch, "max_epochs": self.num_epochs,
            "render_stages": {
                "enabled": self.render_stages_enabled,
                "condition_warmup_epochs": self.condition_warmup_epochs,
                "rgb_warmup_epochs": self.rgb_warmup_epochs,
                "joint_after_rgb": self.joint_after_rgb,
                "current_stage": self.training_stage,
                "stage_step": self.stage_step,
                "stage_epoch": self.stage_epoch,
                "transition_reason": self.stage_transition_reason,
                "condition_best_metric": getattr(self, "condition_best_metric", None),
                "condition_best_epoch": getattr(self, "condition_best_epoch", None),
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
            "validation_metric": self.metric_name if self.validation_enabled else None,
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
            "validation_selected_observations": getattr(self, "validation_selected_observations", 0),
            "best_checkpoint": str(self.get_checkpoint_path("best").resolve()) if self.validation_enabled else None,
            "latest_checkpoint": str(self.get_checkpoint_path("latest").resolve()),
            "metrics_png": (str(pathlib.Path(self.output_dir, "training_metrics.png").resolve())
                            if self.cfg.get("logging", {}).get("enable_plots", False) else None),
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
        condition_stop = getattr(self, "condition_early_stopping", {}) if self.condition_selection else {}
        min_delta = float(condition_stop.get("min_delta", self.min_delta))
        if self.latest_val_loss < self.early_stopping_best_loss - min_delta:
            self.early_stopping_best_loss = self.latest_val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        if improved:
            self.best_val_loss = self.latest_val_loss
            self.best_epoch = self.epoch
            self.best_step = self.global_step
            self._remember_best_condition()
            if self.is_main_process:
                self.save_checkpoint(tag="best", use_thread=False)
                if self.condition_selection:
                    self.save_checkpoint(tag="condition_best", use_thread=False)
        metrics = {"global_step": self.global_step, "epoch": self.epoch,
                   "objective": "condition_warmup" if self.condition_selection else self.objective,
                   "training_stage": self.training_stage,
                   "selection_metric": self.metric_name,
                   "metric_scope": self.metric_scope,
                   "val_token_region_observation_count": getattr(self, "validation_selected_observations", 0),
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
        if self.condition_selection:
            if (bool(condition_stop.get("enabled", True))
                    and self.epochs_without_improvement >= int(condition_stop.get("patience", self.patience))):
                self.condition_transition_pending = True
            return False  # A condition plateau starts render training, not a run-wide stop.
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

    def _dataset_config(self):
        # Use the supplied teacher YAML for data paths/windows; the checkpoint
        # remains the source of the model architecture and fitted normalizer.
        source = self.teacher_source_cfg if self.teacher_source_cfg is not None else self.teacher_cfg
        config = OmegaConf.create(OmegaConf.to_container(source.task.dataset, resolve=True))
        with open_dict(config):
            config.update(OmegaConf.to_container(self.cfg.dataset, resolve=True))
            if int(self.cfg.get("render_net", {}).get("cross_camera_attn_layers", 0)) > 0:
                # The full-image renderer has no phase/alpha/region targets.
                # Keep the teacher's wrench preprocessing, but do not build
                # offline contact labels that neither model nor loss consumes.
                config.contact_label_config = None
            elif hasattr(self, "wrench_preprocessor"):
                config.contact_label_config = self.wrench_preprocessor.get_config()
                with open_dict(self.cfg.dataset):
                    self.cfg.dataset.contact_label_config = config.contact_label_config
        return canonical_dataset_config(config)

    def _run(self):
        dataset_cfg = self._dataset_config()
        output = pathlib.Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if self.is_main_process:
            OmegaConf.save(self.cfg, output / "config.yaml", resolve=True)
            OmegaConf.save(self.teacher_cfg, output / "teacher_config.yaml", resolve=True)
            if self.teacher_source_cfg is not None:
                OmegaConf.save(self.teacher_source_cfg, output / "teacher_source_config.yaml", resolve=True)
            # Continuing in a new directory preserves the previously best model.
            if self.resume_from:
                previous_best = pathlib.Path(self.resume_from).parent / "best.ckpt"
                destination = self.get_checkpoint_path("best")
                if (previous_best.is_file() and not destination.exists()
                        and not self.selection_metric_changed):
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(previous_best, destination)
                if self.selection_metric_changed and destination.exists():
                    # Preserve the former best for review; it was selected
                    # using a different validation denominator.
                    legacy = destination.with_name("best_previous_metric.ckpt")
                    if legacy.exists():
                        raise FileExistsError(legacy)
                    destination.replace(legacy)
        # Keep the original dataset config as the cache identity. Rank-local
        # decoder devices do not change supervision, so all ranks reuse the
        # existing single-GPU teacher cache rather than creating new copies.
        runtime_dataset_cfg = copy.deepcopy(dataset_cfg)
        if self.distributed_context.enabled and "video_decode_device" in runtime_dataset_cfg:
            runtime_dataset_cfg.video_decode_device = str(self.device) if self.device.type == "cuda" else "auto"
        dataset = self._instantiate_dataset(runtime_dataset_cfg)
        validation_dataset = dataset.get_validation_dataset()
        dataset = self._limit_debug_dataset(
            dataset, self.cfg.training.get("debug_train_samples"), name="train",
        )
        validation_dataset = self._limit_debug_dataset(
            validation_dataset, self.cfg.training.get("debug_validation_samples"), name="validation",
        )
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
            self._prepare_force_context_scales(train_dataset)
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
            logger_context = (RenderNetTrainingLogger(
                output, logging_config.get("plot_every_steps", 100),
                enable_plots=bool(logging_config.get("enable_plots", False)),
                async_plots=bool(logging_config.get("async_plots", True)),
                plot_interval_sec=float(logging_config.get("plot_interval_sec", 20.0)))
                              if self.is_main_process else contextlib.nullcontext(_NullTrainingLogger()))
            with logger_context as logger:
                if self.is_main_process:
                    log.info("Local metrics: %s; plotting=%s", logger.path,
                             bool(logging_config.get("enable_plots", False)))
                logger.log({"global_step": self.global_step, "epoch": self.epoch,
                            "objective": "condition_warmup" if self.condition_selection else self.objective,
                            "training_stage": self.training_stage, "selection_metric": self.metric_name,
                            "metric_scope": self.metric_scope})
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
                        selected = self._selection_weight(size)
                        total, count = self._sum([self._selection_mse(initial_loss) * selected, selected])
                        self.initial_train_mse = total / count if count else float("nan")
                    log.info(f"Initial training batch complete: {self.metric_name}={self.initial_train_mse:.8f}")
                    logger.log({"global_step": 0, "epoch": 0, "initial_train_mse": self.initial_train_mse,
                                "objective": "condition_warmup" if self.condition_selection else self.objective,
                                "training_stage": self.training_stage, "selection_metric": self.metric_name,
                                "train_token_mse_measured": not self.condition_selection,
                                "metric_scope": self.metric_scope,
                                "lr": self.optimizer.param_groups[0]["lr"],
                                "elapsed_seconds": time.monotonic() - self.started_at, **self._gpu_metrics()})
                    logger.save_plot()
                    if self.validation_enabled:
                        self.initial_val_mse = self._validate(val_loader)
                        self.latest_val_loss = self.initial_val_mse
                        self.best_val_loss = self.initial_val_mse
                        self.early_stopping_best_loss = self.initial_val_mse
                        self._step_lr_scheduler(self.initial_val_mse)
                        self._remember_best_condition()
                        if self.is_main_process:
                            self.save_checkpoint(tag="best", use_thread=False)
                            if self.condition_selection:
                                self.save_checkpoint(tag="condition_best", use_thread=False)
                        log.info(f"Initial train_mse={self.initial_train_mse:.8f}, val_mse={self.initial_val_mse:.8f}")
                        logger.log({"global_step": 0, "epoch": 0, "initial_train_mse": self.initial_train_mse,
                                    "objective": "condition_warmup" if self.condition_selection else self.objective,
                                    "training_stage": self.training_stage, "selection_metric": self.metric_name,
                                    "train_token_mse_measured": not self.condition_selection,
                                    "val_token_mse_measured": not self.condition_selection,
                                    "metric_scope": self.metric_scope,
                                    "val_token_region_observation_count": getattr(self, "validation_selected_observations", 0),
                                    "student_state_key": self.evaluation_state_key,
                                    "initial_val_mse": self.initial_val_mse,
                                    "val_seconds": getattr(self, "validation_seconds", 0.0),
                                    "val_samples_per_second": getattr(self, "validation_samples_per_second", 0.0),
                                    "elapsed_seconds": time.monotonic() - self.started_at, **self._gpu_metrics()})
                        logger.save_plot()
                else:
                    log.info("Skipping initial evaluation; restored checkpoint state will be continued")
                    if self.selection_metric_changed and self.validation_enabled:
                        log.info("Recomputing validation baseline for %s", self.metric_scope)
                        self._validate_and_save(val_loader, logger)
                self._summary("running")
                for epoch_index in range(self.epoch, self.num_epochs):
                    self._maybe_transition_stage(completed_epochs=self.epoch)
                    if (self.strict_two_stage and self.training_stage == "rgb"
                            and self.rgb_warmup_epochs > 0 and self.stage_epoch >= self.rgb_warmup_epochs):
                        self.stop_reason = "render_max_epochs"
                        break
                    if self.max_steps is not None and self.global_step >= int(self.max_steps):
                        self.stop_reason = "max_steps"
                        break
                    if isinstance(train_loader.sampler, DistributedSampler):
                        train_loader.sampler.set_epoch(epoch_index)
                    epoch_number = epoch_index + 1
                    epoch_total, epoch_samples = 0.0, 0
                    epoch_selection_count = epoch_region_count = 0.0
                    epoch_loss_total = epoch_mask_total = 0.0
                    epoch_components = {key: 0.0 for key in (
                        "token_mse", "denoiser_mse", "demo_mse", "context_patch_mse",
                        "intensity_smoothness", "intensity_loss", "phase_ce", "alpha_mse",
                        "alpha_mae", "phase_accuracy", "contact_open_ratio",
                        "force_delta_mse", "force_delta_loss", "context_region_loss",
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
                        selection_size = self._selection_weight(size)
                        region_size = self._component_weight("token_mse", size)
                        (primary_total, loss_total, mask_total, global_size,
                         selection_size, region_size, *component_totals) = self._sum([
                            self._selection_mse(loss) * selection_size, float(loss.detach()) * size,
                            self.loss_metrics.get("mask_mse", 0.0) * region_size, size,
                            selection_size, region_size,
                            *[self.loss_metrics.get(key, 0.0) * self._component_weight(key, size)
                              for key in epoch_components],
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
                        epoch_selection_count += selection_size
                        epoch_region_count += region_size
                        epoch_loss_total += loss_total
                        epoch_mask_total += mask_total
                        for key, value in zip(epoch_components, regular_totals):
                            epoch_components[key] += value
                        epoch_open_token_sum += open_total
                        epoch_open_token_count += open_count_total
                        epoch_closed_token_sum += closed_total
                        epoch_closed_token_count += closed_count_total
                        condition_warmup = self.training_stage == "condition_warmup"
                        token_measured = not condition_warmup and selection_size > 0
                        metrics = {"global_step": self.global_step, "epoch": epoch_number,
                                   "objective": "condition_warmup" if condition_warmup else self.objective,
                                   "selection_metric": self.metric_name,
                                   "metric_scope": self.metric_scope,
                                   "train_token_region_observation_count": region_size,
                                   "train_mse": primary_total / max(selection_size, 1), "train_loss": loss_total / global_size,
                                   "train_token_mse_measured": token_measured,
                                   "train_mask_mse": mask_total / max(region_size, 1),
                                   "train_per_rank_batch_size": size,
                                   "train_global_batch_size": global_size,
                                   "grad_norm": grad_norm_value, "lr": self.optimizer.param_groups[0]["lr"],
                                   "batch_in_epoch": batch_index + 1, "batches_in_epoch": batches,
                                   "data_seconds": data_seconds, "step_seconds": step_seconds,
                                   "samples_per_second": global_size / max(data_seconds + step_seconds, 1e-6),
                                   "elapsed_seconds": time.monotonic() - self.started_at, **self._gpu_metrics()}
                        metrics.update({f"train_{key}": value / max(
                                            region_size if key in self._REGION_TOKEN_METRICS else global_size, 1)
                                        for key, value in zip(epoch_components, regular_totals)})
                        metrics.update({
                            "train_open_token_mse": metrics_for_batch.get("open_token_mse", 0.0),
                            "train_closed_token_mse": metrics_for_batch.get("closed_token_mse", 0.0),
                            "train_open_token_count": metrics_for_batch.get("open_token_count", 0.0),
                            "train_closed_token_count": metrics_for_batch.get("closed_token_count", 0.0),
                        })
                        logger.log(metrics)
                        if self.force_context_enabled:
                            progress.set_postfix(
                                loss=f"{metrics['train_loss']:.4f}", token=f"{metrics['train_token_mse']:.4f}",
                                delta=f"{metrics['train_force_delta_loss']:.4f}",
                                region=f"{metrics['train_context_region_loss']:.4f}",
                                lr=f"{metrics['lr']:.2g}", refresh=False,
                            )
                        else:
                            progress.set_postfix(loss=f"{metrics['train_loss']:.4f}",
                                                 mse=f"{epoch_total / max(epoch_selection_count, 1):.4f}",
                                                 field=f"{metrics['train_intensity_loss']:.4f}", lr=f"{metrics['lr']:.2g}",
                                                 refresh=False)
                        now = time.monotonic()
                        if (not self._progress_enabled() and self.is_main_process
                                and now - last_status >= self._status_interval()):
                            if self.force_context_enabled:
                                log.info("Train %d/%d | step=%d batch=%d/%d loss=%.5f token=%.5f delta=%.5f region=%.5f lr=%.2g",
                                         epoch_number, self.num_epochs, self.global_step, batch_index + 1, batches,
                                         metrics["train_loss"], metrics["train_token_mse"],
                                         metrics["train_force_delta_loss"], metrics["train_context_region_loss"], metrics["lr"])
                            else:
                                log.info("Train %d/%d | step=%d batch=%d/%d loss=%.5f mse=%.5f field=%.5f lr=%.2g",
                                         epoch_number, self.num_epochs, self.global_step, batch_index + 1, batches,
                                         metrics['train_loss'], epoch_total / max(epoch_selection_count, 1),
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
                                "objective": "condition_warmup" if condition_warmup else self.objective,
                                "selection_metric": self.metric_name,
                                "metric_scope": self.metric_scope,
                                "epoch_token_region_observation_count": epoch_region_count,
                                "epoch_train_token_mse_measured": not condition_warmup and epoch_selection_count > 0,
                                "epoch_train_mse": epoch_total / max(epoch_selection_count, 1),
                                "epoch_train_loss": epoch_loss_total / epoch_samples,
                                "epoch_train_mask_mse": epoch_mask_total / max(epoch_region_count, 1),
                                "epoch_samples": epoch_samples, "epoch_seconds": epoch_seconds,
                                "epoch_samples_per_second": epoch_samples / max(epoch_seconds, 1e-6),
                                **{f"epoch_train_{key}": value / max(
                                       epoch_region_count if key in self._REGION_TOKEN_METRICS else epoch_samples, 1)
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
                        if self.force_context_enabled:
                            log.info("Epoch %d/%d | step=%d loss=%.6f token=%.6f delta=%.6f region=%.6f%s elapsed=%.1fs",
                                     self.epoch, self.num_epochs, self.global_step, epoch_loss_total / epoch_samples,
                                     epoch_components["token_mse"] / max(epoch_region_count, 1),
                                     epoch_components["force_delta_loss"] / epoch_samples,
                                     epoch_components["context_region_loss"] / epoch_samples, validation,
                                     time.monotonic() - epoch_started)
                        else:
                            log.info("Epoch %d/%d | step=%d loss=%.6f train=%.6f field=%.6f%s elapsed=%.1fs",
                                     self.epoch, self.num_epochs, self.global_step,
                                     epoch_loss_total / epoch_samples, epoch_total / max(epoch_selection_count, 1),
                                     epoch_components["intensity_loss"] / max(epoch_region_count, 1), validation,
                                     time.monotonic() - epoch_started)
                    if self.max_steps is not None and self.global_step >= int(self.max_steps):
                        self.stop_reason = "max_steps"
                    elif early_stop:
                        self.stop_reason = "early_stopping"
                    elif (self.strict_two_stage and self.training_stage == "rgb"
                          and self.rgb_warmup_epochs > 0 and self.stage_epoch >= self.rgb_warmup_epochs):
                        self.stop_reason = "render_max_epochs"
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
