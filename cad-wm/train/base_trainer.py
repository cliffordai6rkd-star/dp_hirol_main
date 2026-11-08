import argparse
import copy
import logging
import math
import os
import random
import tempfile
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from tqdm.auto import tqdm
from train.device_backend import (
    autocast as backend_autocast,
    device_count as backend_device_count,
    ensure_npu,
    get_rng_state_all,
    is_npu_device,
    manual_seed_all,
    set_rng_state_all,
    grad_scaler,
)


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ModelEMA:
    """Maintain an inference model with exponentially averaged parameters."""

    def __init__(self, model, decay=0.999, update_after_step=0, update_every=1):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}.")
        if update_after_step < 0:
            raise ValueError("EMA update_after_step must be non-negative.")
        if update_every < 1:
            raise ValueError("EMA update_every must be at least 1.")

        self.decay = float(decay)
        self.update_after_step = int(update_after_step)
        self.update_every = int(update_every)
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        if step % self.update_every != 0:
            return

        source_parameters = dict(model.named_parameters())
        for name, averaged in self.model.named_parameters():
            source = source_parameters[name].detach()
            if step <= self.update_after_step:
                averaged.copy_(source)
            else:
                averaged.lerp_(source, 1.0 - self.decay)

        source_buffers = dict(model.named_buffers())
        for name, averaged in self.model.named_buffers():
            averaged.copy_(source_buffers[name].detach())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("dataset/config/dataset_test_cfg.yaml"),
    )
    return parser.parse_args()

class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.train_config = config.get("train") or {}

        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = self.train_config.get("device", default_device)
        device_type = str(self.device).split(":", 1)[0].lower()
        if device_type == "npu":
            ensure_npu()
        if device_type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "A CUDA device was configured, but torch.cuda.is_available() "
                "returned False. Check the NVIDIA driver or set train.device=cpu "
                "for a smoke test."
            )
        if device_type not in {"cuda", "npu", "cpu"}:
            raise ValueError(
                f"train.device must be cuda:*, npu:*, or cpu, got {self.device!r}."
            )

        self.validation_enabled = bool(self.train_config.get("validation_enabled", True))
        self.val_ratio = float(self.train_config.get("val_ratio", 0.1))
        self.val_every = int(self.train_config.get("val_every", 1))
        if self.validation_enabled and self.val_every <= 0:
            raise ValueError("train.val_every must be a positive integer")
        self.split_mode = self.train_config.get("split_mode", "episode")
        configured_val_episodes = (
            self.train_config.get("val_episode_indices")
            if self.validation_enabled else None
        )
        if configured_val_episodes is None:
            self.val_episode_indices = None
        else:
            if not isinstance(configured_val_episodes, (list, tuple)):
                raise ValueError(
                    "train.val_episode_indices must be a list of episode indices"
                )
            if not configured_val_episodes:
                raise ValueError("train.val_episode_indices must not be empty")
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in configured_val_episodes
            ):
                raise ValueError(
                    "train.val_episode_indices must contain integer episode indices"
                )
            if len(set(configured_val_episodes)) != len(configured_val_episodes):
                raise ValueError(
                    "train.val_episode_indices must not contain duplicates"
                )
            self.val_episode_indices = tuple(configured_val_episodes)
            if self.split_mode != "episode":
                raise ValueError(
                    "train.val_episode_indices requires train.split_mode=episode"
                )
        self.seed = int(self.train_config.get("seed", 42))
        self.deterministic = bool(
            self.train_config.get("deterministic", True)
        )
        self.cudnn_benchmark = bool(
            self.train_config.get("cudnn_benchmark", False)
        )
        self.allow_tf32 = bool(self.train_config.get("allow_tf32", False))
        if self.deterministic and self.cudnn_benchmark:
            raise ValueError(
                "train.deterministic=true is incompatible with "
                "train.cudnn_benchmark=true"
            )
        self.val_loader = None
        purged_kfold_config = self.train_config.get("purged_kfold") or {}
        if not isinstance(purged_kfold_config, dict):
            raise ValueError("train.purged_kfold must be a mapping or null")
        self.purged_kfold_num_folds = int(
            purged_kfold_config.get("num_folds", 3)
        )
        self.purged_kfold_fold_index = int(
            purged_kfold_config.get("fold_index", 0)
        )
        if self.validation_enabled and self.split_mode == "purged_kfold":
            if self.purged_kfold_num_folds < 2:
                raise ValueError("train.purged_kfold.num_folds must be at least 2")
            if not (
                0
                <= self.purged_kfold_fold_index
                < self.purged_kfold_num_folds
            ):
                raise ValueError(
                    "train.purged_kfold.fold_index must be in "
                    f"[0, {self.purged_kfold_num_folds - 1}]"
                )
        self.current_split_metadata = None

        self.batch_size = int(self.train_config.get("batch_size", 64))
        # Keep the decoded dataset tensors on the training device.  CUDA
        # tensors cannot be safely produced by multiprocessing DataLoader
        # workers, so setup() forces a single-process loader when enabled.
        self.dataset_on_device = bool(
            self.train_config.get("dataset_on_device", False)
        )
        self.num_workers = int(self.train_config.get("num_workers", 4))
        if self.num_workers < 0:
            raise ValueError("train.num_workers must be non-negative")
        self.pin_memory = bool(self.train_config.get("pin_memory", False))
        self.non_blocking_transfer = bool(
            self.train_config.get("non_blocking_transfer", self.pin_memory)
        )
        self.persistent_workers = bool(
            self.train_config.get("persistent_workers", False)
        )
        configured_device_batch_keys = self.train_config.get("device_batch_keys")
        if configured_device_batch_keys is None:
            self.device_batch_keys = None
        else:
            if not isinstance(configured_device_batch_keys, (list, tuple, set)):
                raise ValueError("train.device_batch_keys must be a sequence or null")
            self.device_batch_keys = {str(key) for key in configured_device_batch_keys}
        # Keep scalar metric aggregation on-device until epoch end.  Calling
        # .item() for every diagnostic on every batch creates a CUDA sync.
        self.defer_metric_sync = bool(
            self.train_config.get("defer_metric_sync", False)
        )
        self.prefetch_factor = int(self.train_config.get("prefetch_factor", 2))
        if self.prefetch_factor < 1:
            raise ValueError("train.prefetch_factor must be at least 1")
        if self.persistent_workers and self.num_workers <= 0:
            raise ValueError(
                "train.persistent_workers requires train.num_workers > 0"
            )

        # Validation does not need a second pool of persistent workers.  On
        # memory-constrained hosts, sharing the dataset between train and val
        # workers can duplicate Arrow/Python pages through fork COW, so keep
        # validation in the main process by default.  Every setting remains
        # configurable for larger machines.
        self.val_num_workers = int(
            self.train_config.get("val_num_workers", 0)
        )
        self.val_prefetch_factor = int(
            self.train_config.get("val_prefetch_factor", 1)
        )
        self.val_pin_memory = bool(
            self.train_config.get("val_pin_memory", self.pin_memory)
        )
        self.val_persistent_workers = bool(
            self.train_config.get("val_persistent_workers", False)
        )
        if self.val_num_workers < 0:
            raise ValueError("train.val_num_workers must be non-negative")
        if self.val_prefetch_factor < 1:
            raise ValueError("train.val_prefetch_factor must be at least 1")
        if self.val_persistent_workers and self.val_num_workers <= 0:
            raise ValueError(
                "train.val_persistent_workers requires train.val_num_workers > 0"
            )
        self.lr = float(self.train_config.get("lr", 1e-4))
        self.weight_decay = float(self.train_config.get("weight_decay", 1e-4))
        self.num_epochs = int(self.train_config.get("num_epochs", 20))
        configured_max_steps = self.train_config.get(
            "max_optimizer_steps", self.train_config.get("max_train_steps")
        )
        self.max_train_steps = (
            None if configured_max_steps is None else int(configured_max_steps)
        )
        self.checkpoint_every_steps = int(
            self.train_config.get("checkpoint_every_steps", 0)
        )
        self.checkpoint_every_epochs = int(
            self.train_config.get("checkpoint_every_epochs", 0)
        )
        self.save_latest_checkpoint = bool(
            self.train_config.get("save_latest_checkpoint", True)
        )
        self.step_based_training = (
            self.max_train_steps is not None or self.checkpoint_every_steps > 0
        )
        self.max_optimizer_steps = None
        if self.max_train_steps is not None and self.max_train_steps <= 0:
            raise ValueError("train.max_optimizer_steps must be positive")
        if self.checkpoint_every_steps < 0:
            raise ValueError("train.checkpoint_every_steps must be non-negative")
        if self.checkpoint_every_epochs < 0:
            raise ValueError("train.checkpoint_every_epochs must be non-negative")
        gradient_clip_norm = self.train_config.get(
            "gradient_clip_norm",
            self.train_config.get("max_grad_norm"),
        )
        self.gradient_clip_norm = (
            float(gradient_clip_norm) if gradient_clip_norm is not None else None
        )
        self.gradient_every = int(self.train_config.get("gradient_every", 1))
        if self.gradient_every < 1:
            raise ValueError("train.gradient_every must be a positive integer")
        self.monitor_key = self.train_config.get("monitor_key", "val_loss")
        self.scheduler_monitor_key = self.train_config.get(
            "scheduler_monitor_key",
            self.monitor_key,
        )
        self.early_stopping_monitor_key = self.train_config.get(
            "early_stopping_monitor_key",
            self.monitor_key,
        )
        configured_top_k = self.train_config.get("top_k", 3)
        self.top_k = 3 if configured_top_k is None else int(configured_top_k)
        if self.top_k < 1:
            raise ValueError("train.top_k must be at least 1")
        self.best_checkpoints = []
        # Subclasses may evaluate inference quality at each scheduled save.
        self.checkpoint_monitor_key = None

        early_stopping = self.train_config.get("early_stopping") or {}
        self.early_stopping_enabled = bool(early_stopping.get("enabled", False))
        if not self.validation_enabled and self.early_stopping_monitor_key.startswith("val_"):
            self.early_stopping_enabled = False
        self.early_stopping_patience = int(early_stopping.get("patience", 20))
        self.early_stopping_warmup = int(early_stopping.get("warmup_epochs", 0))
        self.early_stopping_min_delta = float(early_stopping.get("min_delta", 0.0))
        self.early_stopping_best = float("inf")
        self.early_stopping_bad_epochs = 0

        self.dataset = None
        self.loader = None
        self.model = None
        self.optimizer = None

        self.output_dir = Path(
            self.train_config.get("output_dir", "outputs/cad_wm")
        )
        self.ckpt_dir = self.output_dir / "checkpoints"

        # ``resume_from`` may point at a checkpoint file, a checkpoint
        # directory, or the output directory containing ``checkpoints/``.
        # Keep this in the trainer (rather than only in shell wrappers) so a
        # resumed run restores the complete optimization state consistently.
        configured_resume = self.train_config.get("resume_from")
        if configured_resume is None:
            configured_resume = self.train_config.get("resume_checkpoint")
        if configured_resume is None:
            configured_resume = self.train_config.get("resume")
        if isinstance(configured_resume, bool):
            self.resume_from = self.output_dir if configured_resume else None
        elif configured_resume:
            self.resume_from = Path(str(configured_resume)).expanduser()
        else:
            self.resume_from = None
        self.resume_checkpoint_path = None
        self.resume_epoch = 0
        self._resume_loaded = False
        self.current_epoch = 0

        self.global_step = 0
        self._last_step_checkpoint = None
        self.loss_history = []
        self.train_step_history = []
        self.inference_history = []
        self.plot_every_steps = int(self.train_config.get("plot_every_steps", 0))
        if self.plot_every_steps < 0:
            raise ValueError("train.plot_every_steps must be non-negative")
        self.last_train_epoch_metrics = {}
        self.last_train_eval_epoch_metrics = {}
        self.last_val_epoch_metrics = {}

        train_eval_config = self.train_config.get("train_eval") or {}
        self.train_eval_enabled = bool(train_eval_config.get("enabled", False))

        self.last_summary = None

        scheduler_config = self.train_config.get("scheduler")
        if scheduler_config is not None and not isinstance(
            scheduler_config,
            dict,
        ):
            raise ValueError("train.scheduler must be a mapping or null")
        self.scheduler_config = (
            None if scheduler_config is None else dict(scheduler_config)
        )
        if self.scheduler_config is not None:
            scheduler_name = str(
                self.scheduler_config.get("name", "none")
            ).strip().lower()
            aliases = {
                "cosine_annealing": "cosine",
                "cosineannealing": "cosine",
            }
            self.scheduler_config["name"] = aliases.get(
                scheduler_name,
                scheduler_name,
            )
        self.scheduler = None
        # Cosine schedules without an explicit T_max are step-based.
        self.scheduler_step_per_optimizer_step = False
        self.scheduler_total_steps = None
        self.scheduler_warmup_steps = 0

        ema_config = self.train_config.get("ema") or {}
        self.ema_enabled = bool(ema_config.get("enabled", False))
        self.ema_decay = float(ema_config.get("decay", 0.999))
        self.ema_update_after_step = int(ema_config.get("update_after_step", 0))
        self.ema_update_every = int(ema_config.get("update_every", 1))
        self.ema_use_for_validation = bool(
            ema_config.get("use_for_validation", True)
        )
        self.ema = None

        self.wandb_config = self.train_config.get("wandb") or {}
        self.wandb_enabled = bool(self.wandb_config.get("enabled", False))
        self.wandb_log_every_steps = int(
            self.wandb_config.get("log_every_steps", 10)
        )
        if self.wandb_log_every_steps < 1:
            raise ValueError("wandb.log_every_steps must be at least 1.")
        self.wandb_run = None

        amp_config = self.train_config.get("amp") or {}
        if not isinstance(amp_config, dict):
            raise ValueError("train.amp must be a mapping or null")
        self.amp_enabled = bool(amp_config.get("enabled", False))
        amp_dtype = str(amp_config.get("dtype", "bfloat16")).lower()
        amp_dtypes = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
        }
        if amp_dtype not in amp_dtypes:
            raise ValueError(
                "train.amp.dtype must be bfloat16/bf16 or float16/fp16"
            )
        self.amp_dtype = amp_dtypes[amp_dtype]
        if self.amp_enabled and device_type not in {"cuda", "npu"}:
            raise ValueError("train.amp.enabled requires a CUDA or NPU device")
        self.amp_scaler = None
        if self.amp_enabled and self.amp_dtype == torch.float16:
            self.amp_scaler = grad_scaler(torch.device(self.device), enabled=True)

    def set_seed(self):
        import random
        import numpy as np
        import torch

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        manual_seed_all(self.seed, str(self.device).split(":", 1)[0].lower())

        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        torch.backends.cudnn.deterministic = self.deterministic
        # Match the mixed-precision fast path used by the LeRobot trainer
        # without changing the default behavior of existing configurations.
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.allow_tf32
        torch.set_float32_matmul_precision("high" if self.allow_tf32 else "highest")

    def batch_to_device(self, batch):
        new_batch = {}

        for k, v in batch.items():
            if torch.is_tensor(v):
                if self.device_batch_keys is not None and k not in self.device_batch_keys:
                    new_batch[k] = v
                    continue
                # log.info(f"{v} is tensor")
                new_batch[k] = v.to(
                    self.device,
                    non_blocking=self.non_blocking_transfer,
                )
            else:
                log.warning(f"{v} is not tensor")
                new_batch[k] = v

        return new_batch

    def autocast_context(self, enabled=None):
        """Return the configured CUDA autocast context for model execution."""

        if enabled is False or not self.amp_enabled:
            return nullcontext()
        return backend_autocast(torch.device(self.device), True, self.amp_dtype)

    def _dataloader_kwargs(
        self,
        *,
        shuffle,
        num_workers=None,
        prefetch_factor=None,
        pin_memory=None,
        persistent_workers=None,
    ):
        """Build DataLoader arguments for train or validation.

        Optional overrides let validation use a smaller worker pool without
        changing the historical training-loader defaults.
        """

        num_workers = self.num_workers if num_workers is None else int(num_workers)
        prefetch_factor = (
            self.prefetch_factor
            if prefetch_factor is None
            else int(prefetch_factor)
        )
        pin_memory = self.pin_memory if pin_memory is None else bool(pin_memory)
        persistent_workers = (
            self.persistent_workers
            if persistent_workers is None
            else bool(persistent_workers)
        )
        if num_workers < 0:
            raise ValueError("DataLoader num_workers must be non-negative")
        if prefetch_factor < 1:
            raise ValueError("DataLoader prefetch_factor must be at least 1")
        if persistent_workers and num_workers <= 0:
            raise ValueError(
                "DataLoader persistent_workers requires num_workers > 0"
            )
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = prefetch_factor
            kwargs["persistent_workers"] = persistent_workers
        return kwargs
    
    def build_dataset(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def initialize_model_weights(self):
        """Optional initialization before EMA and optimizer are created."""

    def compute_loss(self, batch):
        raise NotImplementedError

    def fit_dataset_normalizer(self, train_dataset):
        """Hook for datasets that must fit normalization after the split."""
        return None

    def build_train_sampler(self, train_dataset):
        """Optional hook for trainers that need a non-uniform train sampler."""
        del train_dataset
        return None

    def build_scheduler(self):
        if self.scheduler_config is None:
            return None

        name = self.scheduler_config.get("name", "none")

        if name == "none":
            return None

        if name == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=float(self.scheduler_config.get("factor", 0.5)),
                patience=int(self.scheduler_config.get("patience", 20)),
                min_lr=float(self.scheduler_config.get("min_lr", 1e-6)),
            )
        
        if name == "cosine":
            eta_min = float(self.scheduler_config.get("eta_min", 1e-6))
            if eta_min < 0.0:
                raise ValueError("cosine scheduler eta_min must be non-negative")
            if eta_min > self.lr:
                raise ValueError(
                    "cosine scheduler eta_min must not exceed train.lr"
                )

            # Preserve the original epoch/step behavior for configurations
            # that explicitly provide T_max.  New configs can omit T_max and
            # get an optimizer-step schedule with automatic total steps.
            if "T_max" in self.scheduler_config:
                t_max = int(self.scheduler_config["T_max"])
                if t_max <= 0:
                    raise ValueError("cosine scheduler T_max must be positive")
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=t_max,
                    eta_min=eta_min,
                )

            if self.max_optimizer_steps is not None:
                total_steps = int(self.max_optimizer_steps)
            elif self.loader is not None:
                total_steps = int(self.num_epochs) * len(self.loader)
            else:
                # This fallback keeps build_scheduler usable in lightweight
                # callers that construct the optimizer before a data loader.
                total_steps = int(self.num_epochs)
            if total_steps <= 0:
                raise ValueError(
                    "cosine scheduler requires a positive number of training steps"
                )

            warmup_steps = int(
                self.scheduler_config.get(
                    "warmup_steps",
                    self.scheduler_config.get("lr_warmup_steps", 0),
                )
            )
            if warmup_steps < 0:
                raise ValueError("cosine scheduler warmup_steps must be non-negative")
            if warmup_steps >= total_steps:
                raise ValueError(
                    "cosine scheduler warmup_steps must be smaller than total steps"
                )

            self.scheduler_step_per_optimizer_step = True
            self.scheduler_total_steps = total_steps
            self.scheduler_warmup_steps = warmup_steps
            eta_ratio = eta_min / self.lr if self.lr > 0.0 else 0.0
            log.info(
                "cosine scheduler: total_steps=%d warmup_steps=%d "
                "eta_min=%.2e update=optimizer_step",
                total_steps,
                warmup_steps,
                eta_min,
            )

            def lr_lambda(current_step):
                if warmup_steps > 0 and current_step < warmup_steps:
                    return float(current_step) / float(warmup_steps)
                progress = float(current_step - warmup_steps) / float(
                    max(1, total_steps - warmup_steps)
                )
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return eta_ratio + (1.0 - eta_ratio) * cosine

            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lr_lambda,
            )
        raise ValueError(f"Unsupported scheduler: {name}")

    @staticmethod
    def resolve_resume_checkpoint(configured_path):
        """Resolve a checkpoint file or output directory for continuation.

        ``configured_path`` may be a direct ``.pt`` file, an output directory,
        or its ``checkpoints`` child.  ``latest.pt`` is preferred when present;
        otherwise the numerically newest step/epoch checkpoint is selected.
        Relative paths are checked from the caller's working directory and the
        repository root so rendered configs remain portable.
        """

        if configured_path is None or configured_path is False:
            return None
        raw_path = Path(str(configured_path)).expanduser()
        candidates = [raw_path]
        if not raw_path.is_absolute():
            repository_root = Path(__file__).resolve().parents[1]
            candidates.extend((Path.cwd() / raw_path, repository_root / raw_path))

        path = None
        seen = set()
        for candidate in candidates:
            try:
                key = candidate.resolve()
            except OSError:
                key = candidate.absolute()
            if key in seen:
                continue
            seen.add(key)
            if candidate.exists():
                path = candidate
                break
        if path is None:
            raise FileNotFoundError(
                f"resume checkpoint path does not exist: {configured_path}"
            )
        if path.is_file():
            return path.resolve()

        nested = path / "checkpoints"
        # An output directory may contain auxiliary .pt files.  Prefer its
        # dedicated checkpoint directory so an unrelated artifact cannot be
        # selected as the resume source.
        roots = [nested] if nested.is_dir() else [path]
        checkpoint_files = []
        for root in roots:
            checkpoint_files.extend(root.glob("*.pt"))
        if not checkpoint_files:
            raise FileNotFoundError(
                f"resume checkpoint directory contains no .pt files: {path}"
            )

        def filename_key(candidate):
            name = candidate.name
            for prefix in ("step_", "epoch_"):
                if name.startswith(prefix) and name.endswith(".pt"):
                    number = name[len(prefix) : -3]
                    if number.isdigit():
                        return (int(number), prefix == "step_")
            return (-1, False)

        # A run normally contains one checkpoint family.  If both families
        # are present, metadata gives the only reliable cross-family order;
        # fall back to filename numbering for partially-written files.
        metadata_candidates = []
        for candidate in checkpoint_files:
            try:
                payload = torch.load(
                    candidate,
                    map_location="cpu",
                    weights_only=False,
                )
                if isinstance(payload, Mapping):
                    global_step = payload.get("global_step")
                    epoch = payload.get("epoch")
                    if global_step is None and epoch is None and candidate.name != "latest.pt":
                        continue
                    metadata_candidates.append(
                        (
                            int(global_step if global_step is not None else -1),
                            int(epoch if epoch is not None else -1),
                            filename_key(candidate),
                            candidate,
                        )
                    )
            except Exception as exc:
                log.debug("ignoring unreadable resume candidate %s: %s", candidate, exc)
                continue
        if metadata_candidates:
            # ``latest.pt`` is a pointer, not necessarily the newest payload
            # when a run was interrupted between two save calls.  Prefer the
            # greatest recorded progress and use latest only to break ties.
            return max(
                metadata_candidates,
                key=lambda item: (
                    item[0],
                    item[1],
                    item[3].name == "latest.pt",
                    item[2],
                ),
            )[-1].resolve()
        latest = [candidate for candidate in checkpoint_files if candidate.name == "latest.pt"]
        if latest:
            return latest[0].resolve()
        return max(checkpoint_files, key=filename_key).resolve()

    def _capture_rng_state(self):
        """Capture process RNG state so a resumed run is reproducible."""

        state = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
        }
        try:
            import numpy as np

            state["numpy"] = np.random.get_state()
        except ImportError:
            pass
        device_type = str(self.device).split(":", 1)[0].lower()
        backend_state = get_rng_state_all(device_type)
        if backend_state is not None:
            state[device_type] = backend_state
        return state

    @staticmethod
    def _restore_rng_state(state):
        if not isinstance(state, Mapping):
            return
        try:
            if state.get("python") is not None:
                random.setstate(state["python"])
            if state.get("torch") is not None:
                torch.set_rng_state(state["torch"])
            if state.get("numpy") is not None:
                import numpy as np

                np.random.set_state(state["numpy"])
            if state.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["cuda"])
            if state.get("npu") is not None:
                set_rng_state_all(state["npu"], "npu")
        except (RuntimeError, TypeError, ValueError) as exc:
            log.warning("could not restore checkpoint RNG state: %s", exc)

    @staticmethod
    def _move_optimizer_state_to_device(optimizer, device):
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    def _restore_checkpoint_normalizer(self, checkpoint):
        payload = checkpoint.get("normalizer")
        if not isinstance(payload, Mapping) or not payload.get("stats"):
            return
        if self.dataset is None or not hasattr(self.dataset, "set_normalizer"):
            return
        try:
            from train.nomalizer import Normalizer

            normalizer = Normalizer(
                copy.deepcopy(payload["stats"]),
                eps=float(payload.get("eps", 1.0e-6)),
            )
            self.dataset.set_normalizer(normalizer)
            loss_calculator = getattr(self, "loss_calculator", None)
            if loss_calculator is not None and hasattr(
                loss_calculator, "set_normalizer"
            ):
                loss_calculator.set_normalizer(normalizer)
        except (ImportError, TypeError, ValueError) as exc:
            log.warning("could not restore checkpoint normalizer: %s", exc)

    @staticmethod
    def _serialize_checkpoint_records(records):
        serialized = []
        for record in records or []:
            if not isinstance(record, Mapping):
                continue
            item = dict(record)
            if item.get("path") is not None:
                path = Path(str(item["path"]))
                # ``latest.pt`` is an overwriteable pointer, not a retained
                # checkpoint.  Persisting it in the history would make a
                # later resume point at a different state than the record's
                # epoch/step metadata.
                if path.name == "latest.pt":
                    continue
                item["path"] = str(path)
            serialized.append(item)
        return serialized

    @staticmethod
    def _immutable_checkpoint_path(checkpoint_path, checkpoint):
        """Find the immutable payload represented by ``latest.pt``.

        A latest pointer is useful when it is the only file left, but it must
        not be tracked as a retained checkpoint because every subsequent save
        overwrites it.  Scheduled and step checkpoints have deterministic
        filenames; top-k checkpoints are matched by their metadata.
        """

        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.name != "latest.pt":
            return checkpoint_path.resolve()

        root = checkpoint_path.parent
        checkpoint_type = str(checkpoint.get("checkpoint_type", ""))
        epoch = checkpoint.get("epoch")
        global_step = checkpoint.get("global_step")
        candidates = []
        try:
            if checkpoint_type == "scheduled_epoch" and epoch is not None:
                value = int(epoch)
                candidates.extend(
                    (root / f"epoch_{value:07d}.pt", root / f"epoch_{value:03d}.pt")
                )
            elif checkpoint_type == "optimizer_step" and global_step is not None:
                candidates.append(root / f"step_{int(global_step):08d}.pt")
            elif epoch is not None:
                value = int(epoch)
                candidates.extend(
                    (root / f"epoch_{value:03d}.pt", root / f"epoch_{value:07d}.pt")
                )
        except (TypeError, ValueError):
            candidates = []
        for candidate in candidates:
            if candidate.is_file() and candidate.name != "latest.pt":
                return candidate.resolve()

        # Top-k filenames contain a floating-point score, so use metadata as a
        # fallback.  This path is taken only when resuming from latest.pt.
        target_epoch = None
        target_step = None
        try:
            target_epoch = None if epoch is None else int(epoch)
        except (TypeError, ValueError):
            pass
        try:
            target_step = None if global_step is None else int(global_step)
        except (TypeError, ValueError):
            pass
        for candidate in root.glob("*.pt"):
            if candidate.name == "latest.pt":
                continue
            try:
                payload = torch.load(
                    candidate,
                    map_location="cpu",
                    weights_only=False,
                )
                if not isinstance(payload, Mapping):
                    continue
                candidate_epoch = payload.get("epoch")
                candidate_step = payload.get("global_step")
                if target_epoch is not None and candidate_epoch is not None:
                    if int(candidate_epoch) != target_epoch:
                        continue
                if target_step is not None and candidate_step is not None:
                    if int(candidate_step) != target_step:
                        continue
                if target_epoch is not None or target_step is not None:
                    return candidate.resolve()
            except (OSError, TypeError, ValueError, RuntimeError, EOFError):
                continue
        return None

    def _checkpoint_runtime_state(self, *, resume_epoch):
        """Return state needed to continue optimization after a checkpoint."""

        return {
            "resume_epoch": int(resume_epoch),
            "rng_state": self._capture_rng_state(),
            "amp_scaler": (
                self.amp_scaler.state_dict() if self.amp_scaler is not None else None
            ),
            "trainer_state": {
                "resume_epoch": int(resume_epoch),
                "global_step": int(self.global_step),
                "loss_history": copy.deepcopy(self.loss_history),
                "train_step_history": copy.deepcopy(self.train_step_history),
                "inference_history": copy.deepcopy(self.inference_history),
                "best_checkpoints": self._serialize_checkpoint_records(
                    self.best_checkpoints
                ),
                "early_stopping_best": float(self.early_stopping_best),
                "early_stopping_bad_epochs": int(self.early_stopping_bad_epochs),
            },
        }

    def _model_checkpoint_metadata(self):
        contract = getattr(self.model, "checkpoint_contract", None)
        contract_key = getattr(self.model, "CHECKPOINT_CONTRACT_KEY", "model_contract")
        metadata = {contract_key: contract()} if callable(contract) else {}
        extra = getattr(self, "checkpoint_metadata", None)
        if isinstance(extra, dict):
            metadata.update(copy.deepcopy(extra))
        return metadata

    @staticmethod
    def _save_checkpoint_atomic(checkpoint, path):
        """Write a checkpoint atomically so an interrupted save is ignored."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
            torch.save(checkpoint, temporary_path)
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    pass

    def _load_resume_checkpoint(self):
        """Load model and all available optimizer/trainer state for resuming."""

        if self.resume_from is None:
            return
        checkpoint_path = self.resolve_resume_checkpoint(self.resume_from)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint, Mapping):
            raise ValueError(
                f"resume checkpoint must contain a mapping: {checkpoint_path}"
            )
        expected_model_version = getattr(self.model, "MODEL_VERSION", None)
        if expected_model_version is not None and checkpoint.get("model_version") != expected_model_version:
            raise ValueError(
                f"resume checkpoint model_version {checkpoint.get('model_version')!r} "
                f"does not match {expected_model_version!r}; retrain with the current model"
            )
        validate_contract = getattr(self.model, "validate_checkpoint_contract", None)
        validate_checkpoint = getattr(self.model, "validate_checkpoint", None)
        if callable(validate_checkpoint):
            validate_checkpoint(checkpoint)
        elif callable(validate_contract):
            contract_key = getattr(self.model, "CHECKPOINT_CONTRACT_KEY", "model_contract")
            validate_contract(checkpoint.get(contract_key))

        model_state = checkpoint.get("model")
        raw_model_state = checkpoint.get("model_raw")
        if model_state is None and raw_model_state is None:
            raise KeyError(
                f"resume checkpoint has no model/model_raw state: {checkpoint_path}"
            )

        # ``model`` is the EMA copy when EMA is enabled.  Continue optimizing
        # the raw model, while restoring EMA independently when available.
        if callable(validate_contract) and (
            self.ema is not None or (checkpoint.get("ema") or {}).get("enabled") is not False
        ):
            # EMA WM resumes require both recorded states. With EMA explicitly
            # disabled, native checkpoints store raw weights only under model.
            if not isinstance(raw_model_state, Mapping):
                raise KeyError("WM resume checkpoint is missing model_raw weights")
            self.model.load_state_dict(raw_model_state, strict=True)
            if self.ema is not None:
                if not isinstance(model_state, Mapping):
                    raise KeyError("WM resume checkpoint is missing EMA model weights")
                self.ema.model.load_state_dict(model_state, strict=True)
        elif self.ema is not None:
            self.model.load_state_dict(raw_model_state or model_state, strict=True)
            self.ema.model.load_state_dict(model_state or raw_model_state, strict=True)
        else:
            self.model.load_state_dict(raw_model_state or model_state, strict=True)

        optimizer_state = checkpoint.get("optimizer")
        if optimizer_state is not None:
            try:
                self.optimizer.load_state_dict(optimizer_state)
                self._move_optimizer_state_to_device(self.optimizer, self.device)
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    f"could not restore optimizer state from {checkpoint_path}: {exc}"
                ) from exc
        else:
            log.warning("resume checkpoint has no optimizer state; optimizer reset")

        scheduler_state = checkpoint.get("scheduler")
        if self.scheduler is not None and scheduler_state is not None:
            try:
                self.scheduler.load_state_dict(scheduler_state)
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    f"could not restore scheduler state from {checkpoint_path}: {exc}"
                ) from exc
        elif self.scheduler is not None:
            log.warning("resume checkpoint has no scheduler state; scheduler reset")

        scaler_state = checkpoint.get("amp_scaler")
        if self.amp_scaler is not None and scaler_state is not None:
            self.amp_scaler.load_state_dict(scaler_state)
        elif self.amp_scaler is not None:
            log.warning("resume checkpoint has no AMP scaler state; scaler reset")

        trainer_state = checkpoint.get("trainer_state") or {}
        if not isinstance(trainer_state, Mapping):
            trainer_state = {}
        self.global_step = int(
            trainer_state.get("global_step", checkpoint.get("global_step", 0)) or 0
        )
        checkpoint_type = str(checkpoint.get("checkpoint_type", ""))
        if "resume_epoch" in trainer_state:
            self.resume_epoch = int(trainer_state["resume_epoch"])
        elif "resume_epoch" in checkpoint:
            self.resume_epoch = int(checkpoint["resume_epoch"])
        elif checkpoint_type == "scheduled_epoch":
            # Scheduled checkpoints store the number of completed epochs.
            self.resume_epoch = int(checkpoint.get("epoch", 0) or 0)
        else:
            # Epoch/top-k and optimizer-step checkpoints store the zero-based
            # epoch currently being finalized.
            self.resume_epoch = int(checkpoint.get("epoch", -1) or -1) + 1
        self.resume_epoch = max(self.resume_epoch, 0)

        history = trainer_state.get("loss_history", checkpoint.get("loss_history"))
        if isinstance(history, list):
            self.loss_history = copy.deepcopy(history)
        for name in ("train_step_history", "inference_history"):
            history = trainer_state.get(name)
            if isinstance(history, list):
                setattr(self, name, copy.deepcopy(history))
        records = trainer_state.get(
            "best_checkpoints", checkpoint.get("best_checkpoints", [])
        )
        if isinstance(records, list):
            self.best_checkpoints = []
            for record in records:
                if not isinstance(record, Mapping):
                    continue
                item = dict(record)
                if item.get("path") is not None:
                    item["path"] = Path(str(item["path"]))
                    if item["path"].name == "latest.pt":
                        continue
                self.best_checkpoints.append(item)
        record_checkpoint_path = self._immutable_checkpoint_path(
            checkpoint_path,
            checkpoint,
        )
        if record_checkpoint_path is not None:
            source_record = {
                "score": checkpoint.get("monitor_score"),
                "epoch": int(checkpoint.get("epoch", self.resume_epoch)),
                "global_step": self.global_step,
                "path": record_checkpoint_path,
                "metrics": dict(checkpoint.get("metrics") or {}),
            }
            if not any(
                Path(item.get("path")).resolve() == record_checkpoint_path
                for item in self.best_checkpoints
                if item.get("path")
            ):
                self.best_checkpoints.append(source_record)

        if "early_stopping_best" in trainer_state:
            self.early_stopping_best = float(trainer_state["early_stopping_best"])
        if "early_stopping_bad_epochs" in trainer_state:
            self.early_stopping_bad_epochs = int(trainer_state["early_stopping_bad_epochs"])

        self._restore_checkpoint_normalizer(checkpoint)
        self._restore_rng_state(checkpoint.get("rng_state"))
        self.resume_checkpoint_path = checkpoint_path
        self._resume_loaded = True
        self._last_step_checkpoint = self.global_step if checkpoint_type == "optimizer_step" else None
        self.current_epoch = self.resume_epoch
        log.info(
            "resumed training from %s (epoch=%d global_step=%d)",
            checkpoint_path,
            self.resume_epoch,
            self.global_step,
        )

    def split_dataset_by_episode(self):
        episodes = list(self.dataset.dataset.meta.episodes)
        if len(episodes) < 2:
            raise ValueError(
                "episode split requires at least two episodes; use "
                "split_mode=purged_temporal for a single time-series episode."
            )

        episode_indices = [
            int(episode.get("episode_index", position))
            for position, episode in enumerate(episodes)
        ]
        if len(set(episode_indices)) != len(episode_indices):
            raise ValueError("episode metadata contains duplicate episode_index values")

        if self.val_episode_indices is not None:
            val_episode_indices = set(self.val_episode_indices)
            unknown = sorted(val_episode_indices - set(episode_indices))
            if unknown:
                raise ValueError(
                    "train.val_episode_indices contains unknown episode indices: "
                    f"{unknown}; available={sorted(episode_indices)}"
                )
            if len(val_episode_indices) >= len(episodes):
                raise ValueError(
                    "train.val_episode_indices must leave at least one training episode"
                )
        else:
            num_val_episodes = int(len(episodes) * self.val_ratio)
            num_val_episodes = max(1, num_val_episodes)
            num_val_episodes = min(num_val_episodes, len(episodes) - 1)

            generator = torch.Generator().manual_seed(self.seed)
            perm = torch.randperm(len(episodes), generator=generator).tolist()
            val_episode_indices = {
                episode_indices[position] for position in perm[:num_val_episodes]
            }

        episode_start_to_idx = {
            int(ep["dataset_from_index"]): episode_indices[position]
            for position, ep in enumerate(episodes)
        }

        train_indices = []
        val_indices = []

        for sample_idx, raw_idx in enumerate(self.dataset.valid_indices):
            episode_start = self.dataset.raw_idx_to_episode_start[raw_idx]
            episode_idx = episode_start_to_idx[episode_start]

            if episode_idx in val_episode_indices:
                val_indices.append(sample_idx)
            else:
                train_indices.append(sample_idx)

        log.info(
            f"episode split: "
            f"train_episodes={len(episodes) - len(val_episode_indices)} "
            f"val_episodes={len(val_episode_indices)} "
            f"train_samples={len(train_indices)} "
            f"val_samples={len(val_indices)}"
        )
        log.info(f"val episode indices: {sorted(val_episode_indices)}")

        train_dataset = torch.utils.data.Subset(self.dataset, train_indices)
        val_dataset = torch.utils.data.Subset(self.dataset, val_indices)

        return train_dataset, val_dataset

    def split_dataset_purged_temporal(self):
        """Chronological split with disjoint raw-frame windows on both sides."""
        horizon = int(getattr(self.dataset, "horizon", 1))
        train_indices = []
        val_indices = []
        purged_samples = 0
        samples_by_episode_start = {}
        for sample_idx, raw_idx in enumerate(self.dataset.valid_indices):
            episode_start = self.dataset.raw_idx_to_episode_start[raw_idx]
            samples_by_episode_start.setdefault(episode_start, []).append(
                (sample_idx, raw_idx)
            )

        for episode in self.dataset.dataset.meta.episodes:
            episode_start = int(episode["dataset_from_index"])
            episode_samples = samples_by_episode_start.get(episode_start, [])
            if len(episode_samples) < 2:
                continue

            num_val = max(1, int(len(episode_samples) * self.val_ratio))
            num_val = min(num_val, len(episode_samples) - 1)
            first_val_position = len(episode_samples) - num_val
            first_val_raw_idx = episode_samples[first_val_position][1]
            last_train_raw_idx = first_val_raw_idx - horizon

            for sample_idx, raw_idx in episode_samples:
                if raw_idx <= last_train_raw_idx:
                    train_indices.append(sample_idx)
                elif raw_idx >= first_val_raw_idx:
                    val_indices.append(sample_idx)
                else:
                    purged_samples += 1

        if not train_indices or not val_indices:
            raise ValueError(
                "purged_temporal split produced an empty train or validation set; "
                "reduce dataloader.horizon or train.val_ratio, or collect a longer episode."
            )

        log.info(
            "purged temporal split: train_samples=%d val_samples=%d "
            "purged_samples=%d horizon=%d",
            len(train_indices),
            len(val_indices),
            purged_samples,
            horizon,
        )
        return (
            torch.utils.data.Subset(self.dataset, train_indices),
            torch.utils.data.Subset(self.dataset, val_indices),
        )

    def split_dataset_purged_kfold(self, fold_index=None, num_folds=None):
        """Hold out one contiguous block per episode without window overlap."""
        horizon = int(getattr(self.dataset, "horizon", 1))
        num_folds = int(
            self.purged_kfold_num_folds if num_folds is None else num_folds
        )
        fold_index = int(
            self.purged_kfold_fold_index if fold_index is None else fold_index
        )
        if num_folds < 2:
            raise ValueError("purged K-fold requires at least two folds")
        if not 0 <= fold_index < num_folds:
            raise ValueError(
                f"fold_index must be in [0, {num_folds - 1}], got {fold_index}"
            )

        samples_by_episode_start = {}
        for sample_idx, raw_idx in enumerate(self.dataset.valid_indices):
            episode_start = self.dataset.raw_idx_to_episode_start[raw_idx]
            samples_by_episode_start.setdefault(int(episode_start), []).append(
                (sample_idx, int(raw_idx))
            )

        train_indices = []
        val_indices = []
        purged_samples = 0
        episode_metadata = []

        for episode_position, episode in enumerate(
            self.dataset.dataset.meta.episodes
        ):
            episode_start = int(episode["dataset_from_index"])
            episode_samples = sorted(
                samples_by_episode_start.get(episode_start, []),
                key=lambda item: item[1],
            )
            sample_count = len(episode_samples)
            if sample_count < num_folds:
                raise ValueError(
                    "purged_kfold requires every episode to contain at least "
                    f"num_folds valid windows; episode {episode_position} has "
                    f"{sample_count}, num_folds={num_folds}."
                )

            val_start_position = sample_count * fold_index // num_folds
            val_end_position = sample_count * (fold_index + 1) // num_folds
            validation_samples = episode_samples[
                val_start_position:val_end_position
            ]
            first_val_raw_idx = validation_samples[0][1]
            last_val_raw_idx = validation_samples[-1][1]
            first_val_frame = max(
                episode_start,
                first_val_raw_idx - horizon + 1,
            )

            episode_train_count = 0
            episode_purged_count = 0
            for sample_position, (sample_idx, raw_idx) in enumerate(
                episode_samples
            ):
                if val_start_position <= sample_position < val_end_position:
                    val_indices.append(sample_idx)
                    continue

                first_window_frame = max(
                    episode_start,
                    raw_idx - horizon + 1,
                )
                if (
                    raw_idx < first_val_frame
                    or first_window_frame > last_val_raw_idx
                ):
                    train_indices.append(sample_idx)
                    episode_train_count += 1
                else:
                    purged_samples += 1
                    episode_purged_count += 1

            if episode_train_count == 0:
                raise ValueError(
                    "purged_kfold removed all training windows from episode "
                    f"{episode_position} in fold {fold_index}; reduce "
                    "dataloader.horizon or train.purged_kfold.num_folds, or "
                    "collect a longer episode."
                )

            episode_metadata.append(
                {
                    "episode_index": int(
                        episode.get("episode_index", episode_position)
                    ),
                    "train_samples": episode_train_count,
                    "val_samples": len(validation_samples),
                    "purged_samples": episode_purged_count,
                    "val_target_start": first_val_raw_idx,
                    "val_target_end": last_val_raw_idx,
                }
            )

        if not train_indices or not val_indices:
            raise ValueError(
                "purged_kfold split produced an empty train or validation set"
            )

        self.current_split_metadata = {
            "mode": "purged_kfold",
            "fold_index": fold_index,
            "num_folds": num_folds,
            "horizon": horizon,
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "purged_samples": purged_samples,
            "episodes": episode_metadata,
        }
        log.info(
            "purged K-fold split: fold=%d/%d train_samples=%d val_samples=%d "
            "purged_samples=%d horizon=%d episodes=%d",
            fold_index + 1,
            num_folds,
            len(train_indices),
            len(val_indices),
            purged_samples,
            horizon,
            len(episode_metadata),
        )
        return (
            torch.utils.data.Subset(self.dataset, train_indices),
            torch.utils.data.Subset(self.dataset, val_indices),
        )

    def setup(self):
        self.set_seed()
        self.dataset = self.build_dataset()

        if self.dataset_on_device:
            if torch.device(self.device).type not in {"cuda", "npu"}:
                raise ValueError("train.dataset_on_device requires a CUDA or NPU device")
            if not hasattr(self.dataset, "to_device"):
                raise TypeError(
                    "train.dataset_on_device is enabled but the dataset does not "
                    "implement to_device()"
                )
            log.info("moving decoded dataset tensors to %s", self.device)
            self.dataset.to_device(self.device)
            # Accelerator tensors must stay in the main process.  Disable pinned
            # memory and worker prefetching because samples are already on GPU.
            self.num_workers = 0
            self.val_num_workers = 0
            self.pin_memory = False
            self.val_pin_memory = False
            self.persistent_workers = False
            self.val_persistent_workers = False

        if self.validation_enabled and (
            self.val_ratio > 0
            or self.split_mode == "purged_kfold"
            or self.val_episode_indices is not None
        ):
            if self.split_mode == "episode":
                train_dataset, val_dataset = self.split_dataset_by_episode()
            elif self.split_mode == "sample":
                if int(getattr(self.dataset, "horizon", 1)) > 1:
                    log.warning(
                        "sample split leaks overlapping temporal windows across train and "
                        "validation; use split_mode=episode, purged_temporal, or "
                        "purged_kfold."
                    )
                val_size = int(len(self.dataset) * self.val_ratio)
                train_size = len(self.dataset) - val_size
        
                generator = torch.Generator().manual_seed(self.seed)
                train_dataset, val_dataset = torch.utils.data.random_split(
                    self.dataset,
                    [train_size, val_size],
                    generator=generator,
                )
            elif self.split_mode == "purged_temporal":
                train_dataset, val_dataset = self.split_dataset_purged_temporal()
            elif self.split_mode == "purged_kfold":
                train_dataset, val_dataset = self.split_dataset_purged_kfold()
            else:
                raise ValueError(f"unknown split_mode: {self.split_mode}")
        else:
            train_dataset = self.dataset
            val_dataset = None
            if not self.validation_enabled:
                log.info("validation disabled: using all %d samples for training", len(train_dataset))

        self.fit_dataset_normalizer(train_dataset)

        train_sampler = self.build_train_sampler(train_dataset)
        # Do not pre-materialize every window on GPU.  That path duplicates
        # the raw columns and all derived sample fields (history/future
        # indices, timestamps, masks), often exhausting both host and device
        # memory.  GPU-resident training now indexes raw columns per batch;
        # callers can still invoke materialize_samples_on_device() explicitly
        # for a deliberately small dataset.
        train_loader_kwargs = self._dataloader_kwargs(shuffle=train_sampler is None)
        if self.dataset_on_device and hasattr(self.dataset, "batch_collate"):
            train_loader_kwargs["collate_fn"] = self.dataset.batch_collate
        if train_sampler is not None:
            train_loader_kwargs.pop("shuffle", None)
            train_loader_kwargs["sampler"] = train_sampler
        self.loader = torch.utils.data.DataLoader(train_dataset, **train_loader_kwargs)

        self.val_loader = None
        if val_dataset is not None:
            val_loader_kwargs = self._dataloader_kwargs(
                shuffle=False,
                num_workers=self.val_num_workers,
                prefetch_factor=self.val_prefetch_factor,
                pin_memory=self.val_pin_memory,
                persistent_workers=self.val_persistent_workers,
            )
            if self.dataset_on_device and hasattr(self.dataset, "batch_collate"):
                val_loader_kwargs["collate_fn"] = self.dataset.batch_collate
            self.val_loader = torch.utils.data.DataLoader(
                val_dataset,
                **val_loader_kwargs,
            )

        if self.step_based_training:
            if len(self.loader) < 1:
                raise ValueError("step-based training requires a non-empty train loader")
            self.max_optimizer_steps = (
                self.max_train_steps
                if self.max_train_steps is not None
                else self.num_epochs * len(self.loader)
            )
            if self.checkpoint_every_steps <= 0:
                self.checkpoint_every_steps = self.max_optimizer_steps

        self.model = self.build_model().to(self.device)
        if self.resume_from is None:
            self.initialize_model_weights()
        if self.ema_enabled:
            self.ema = ModelEMA(
                self.model,
                decay=self.ema_decay,
                update_after_step=self.ema_update_after_step,
                update_every=self.ema_update_every,
            )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = self.build_scheduler()
        # Restore only after every stateful training object has been created.
        # This also restores the fitted normalizer after ``fit_dataset_normalizer``
        # has initialized the dataset.
        self._load_resume_checkpoint()
        self.setup_wandb()

    def setup_wandb(self):
        if not self.wandb_enabled:
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging is enabled but wandb is not installed. "
                "Run setup.sh or install the project again."
            ) from exc

        self.output_dir.mkdir(parents=True, exist_ok=True)
        init_kwargs = {
            "project": self.wandb_config.get("project", "cad-wm"),
            "name": self.wandb_config.get("name"),
            "entity": self.wandb_config.get("entity"),
            "group": self.wandb_config.get("group"),
            "tags": self.wandb_config.get("tags"),
            "notes": self.wandb_config.get("notes"),
            "mode": self.wandb_config.get("mode", "online"),
            "config": self.config,
            "dir": str(self.output_dir),
            "resume": self.wandb_config.get("resume"),
        }
        self.wandb_run = wandb.init(
            **{key: value for key, value in init_kwargs.items() if value is not None}
        )

    def log_wandb(self, metrics, step=None):
        if self.wandb_run is not None:
            metrics = dict(metrics)
            if step is not None:
                metrics.setdefault("trainer/global_step", step)
            self.wandb_run.log(metrics)

    def finish_wandb(self, exit_code=0):
        if self.wandb_run is not None:
            self.wandb_run.finish(exit_code=exit_code)
            self.wandb_run = None

    @staticmethod
    def _batch_size(batch):
        for value in batch.values():
            if torch.is_tensor(value) and value.ndim > 0:
                return int(value.shape[0])
        return 1

    @staticmethod
    def _accumulate_scalar_metrics(
        metric_sums,
        metric_counts,
        metrics,
        weight,
        *,
        defer_device_sync=False,
    ):
        for key, value in metrics.items():
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                value = value.detach()
                if defer_device_sync:
                    # Store scalar tensors and reduce once at epoch end.  A
                    # list avoids launching a tiny GPU add kernel per metric.
                    metric_sums.setdefault(key, []).append(value)
                    metric_counts.setdefault(key, []).append(weight)
                    continue
                value = value.item()
            if not isinstance(value, (int, float)):
                continue
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value) * weight
            metric_counts[key] = metric_counts.get(key, 0) + weight

    @staticmethod
    def _average_scalar_metrics(metric_sums, metric_counts):
        result = {}
        for key, total in metric_sums.items():
            counts = metric_counts[key]
            if isinstance(total, list):
                if not total:
                    result[key] = 0.0
                    continue
                values = torch.stack(total)
                weights = torch.as_tensor(
                    counts, device=values.device, dtype=values.dtype
                )
                result[key] = (
                    (values * weights).sum() / weights.sum().clamp_min(1.0)
                ).item()
            else:
                result[key] = total / max(counts, 1)
        return result

    def train_one_epoch(self, epoch):
        self.model.train()

        total_loss = 0.0
        num_samples = 0
        metric_sums = {}
        metric_counts = {}
        pbar = tqdm(
            self.loader,
            desc=f"train epoch {epoch}",
            unit="batch",
            leave=False,
        )
        self.optimizer.zero_grad(set_to_none=True)
        micro_batches = 0
        update_loss_sum, update_samples = 0.0, 0
        for step, batch in enumerate(pbar):
            if (
                self.step_based_training
                and self.max_optimizer_steps is not None
                and self.global_step >= self.max_optimizer_steps
            ):
                break
            batch = self.batch_to_device(batch)

            with self.autocast_context():
                loss, out = self.compute_loss(batch)
            batch_size = self._batch_size(batch)
            loss_value = loss.detach().item()
            update_loss_sum += loss_value * batch_size
            update_samples += batch_size

            micro_batches += 1
            scaled_loss = loss / float(self.gradient_every)
            if self.amp_scaler is not None:
                self.amp_scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            should_step = micro_batches % self.gradient_every == 0 or step + 1 == len(self.loader)
            if should_step:
                remainder = micro_batches % self.gradient_every
                if remainder:
                    correction = float(self.gradient_every) / float(remainder)
                    for parameter in self.model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                if self.amp_scaler is not None:
                    self.amp_scaler.unscale_(self.optimizer)
                if self.gradient_clip_norm is not None and self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_norm)
                if self.amp_scaler is not None:
                    self.amp_scaler.step(self.optimizer)
                    self.amp_scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                if self.ema is not None:
                    self.ema.update(self.model, self.global_step)
                if (self.scheduler is not None and (self.step_based_training or self.scheduler_step_per_optimizer_step)
                    and self.scheduler_config.get("name", "none") != "reduce_on_plateau"):
                    self.scheduler.step()
                self.train_step_history.append({
                    "epoch": epoch, "global_step": self.global_step,
                    "avg_loss": update_loss_sum / update_samples,
                })
                update_loss_sum, update_samples = 0.0, 0
                if self.plot_every_steps > 0 and self.global_step % self.plot_every_steps == 0:
                    self.save_loss_plot()

            # Step-budgeted runs checkpoint on the exact optimizer update,
            # independent of DataLoader/epoch boundaries.
            if (
                should_step
                and self.step_based_training
                and self.checkpoint_every_steps > 0
                and self.global_step % self.checkpoint_every_steps == 0
            ):
                self.save_step_checkpoint(
                    epoch,
                    {
                        "avg_loss": float(loss.detach().item()),
                        "train_loss_online": float(loss.detach().item()),
                        "val_loss": None,
                    },
                )

            total_loss += loss_value * batch_size
            num_samples += batch_size
            self._accumulate_scalar_metrics(
                metric_sums,
                metric_counts,
                out.get("loss_dict") or {},
                batch_size,
                defer_device_sync=self.defer_metric_sync,
            )

            if (
                self.wandb_run is not None
                and self.global_step % self.wandb_log_every_steps == 0
            ):
                step_metrics = {
                    "train/loss": loss.detach().item(),
                    "train/epoch": epoch,
                }
                for key, value in (out.get("loss_dict") or {}).items():
                    if torch.is_tensor(value):
                        value = value.detach().item()
                    if isinstance(value, (int, float)):
                        step_metrics[f"train/{key}"] = value
                self.log_wandb(step_metrics, step=self.global_step)

            pbar.set_postfix({
                "loss": f"{loss_value:.6f}",
                "step": self.global_step,
            })

        self.last_train_epoch_metrics = self._average_scalar_metrics(
            metric_sums,
            metric_counts,
        )
        return total_loss / max(num_samples, 1)
    
    @torch.no_grad()
    def evaluate_loader(self, loader, epoch, description):
        training_model = self.model
        if self.ema is not None and self.ema_use_for_validation:
            self.model = self.ema.model
        self.model.eval()

        total_loss = 0.0
        num_samples = 0
        metric_sums = {}
        metric_counts = {}

        pbar = tqdm(
            loader,
            desc=f"{description} epoch {epoch}",
            unit="batch",
            leave=False,
        )

        try:
            for batch in pbar:
                batch = self.batch_to_device(batch)
                with self.autocast_context():
                    loss, out = self.compute_loss(batch)
                batch_size = self._batch_size(batch)

                loss_value = loss.detach().item()
                total_loss += loss_value * batch_size
                num_samples += batch_size
                self._accumulate_scalar_metrics(
                    metric_sums,
                    metric_counts,
                    out.get("loss_dict") or {},
                    batch_size,
                    defer_device_sync=self.defer_metric_sync,
                )

                pbar.set_postfix({
                    "loss": f"{loss_value:.6f}",
                })
        finally:
            self.model = training_model

        metrics = self._average_scalar_metrics(
            metric_sums,
            metric_counts,
        )
        loss = total_loss / max(num_samples, 1)
        return loss, metrics

    def validate_one_epoch(self, epoch):
        if not self.validation_enabled or self.val_loader is None:
            self.last_val_epoch_metrics = {}
            return None

        val_loss, self.last_val_epoch_metrics = self.evaluate_loader(
            self.val_loader,
            epoch,
            "val",
        )
        return val_loss

    def evaluate_train_one_epoch(self, epoch):
        if not self.train_eval_enabled:
            self.last_train_eval_epoch_metrics = {}
            return None

        train_eval_loss, self.last_train_eval_epoch_metrics = self.evaluate_loader(
            self.loader,
            epoch,
            "train-eval",
        )
        return train_eval_loss

    def evaluate_epoch_metrics(self, epoch):
        """Optional inference diagnostics, independent of validation splitting."""
        return {}
    
    def train(self):
        try:
            return self._train_impl()
        except BaseException:
            self.finish_wandb(exit_code=1)
            raise

    def _train_impl(self):

        self.setup()

        # Checkpoints represent completed epochs.  Continue with the next
        # epoch while retaining the configured ``num_epochs`` as the total
        # budget, rather than adding another full budget on every invocation.
        epoch = int(self.resume_epoch)
        self.current_epoch = epoch
        while True:
            if self.step_based_training:
                if self.max_optimizer_steps is None or self.global_step >= self.max_optimizer_steps:
                    break
            elif epoch >= self.num_epochs:
                break

            step_before_epoch = self.global_step
            self.current_epoch = epoch
            avg_loss = self.train_one_epoch(epoch)
            if self.global_step == step_before_epoch:
                log.warning("training loader produced no optimizer steps; stopping")
                break
            train_eval_loss = self.evaluate_train_one_epoch(epoch)
            should_validate = (
                self.validation_enabled
                and self.val_loader is not None
                and epoch % self.val_every == 0
            )
            if should_validate:
                val_loss = self.validate_one_epoch(epoch)
            else:
                # Do not retain per-batch diagnostics from an earlier
                # validation epoch when this epoch intentionally skips val.
                self.last_val_epoch_metrics = {}
                val_loss = None

            metrics = {
                "avg_loss": avg_loss,
                "train_loss_online": avg_loss,
                "train_eval_loss": train_eval_loss,
                "val_loss": val_loss,
            }
            metrics.update(self.evaluate_epoch_metrics(epoch))
            metrics.update(
                {
                    f"train_{key}": value
                    for key, value in self.last_train_epoch_metrics.items()
                }
            )
            if train_eval_loss is not None:
                metrics.update(
                    {
                        f"train_eval_{key}": value
                        for key, value in self.last_train_eval_epoch_metrics.items()
                    }
                )
            if val_loss is not None:
                metrics.update(
                    {
                        f"val_{key}": value
                        for key, value in self.last_val_epoch_metrics.items()
                    }
                )

            if self.scheduler is not None:
                name = self.scheduler_config.get("name", "none")
                if name == "reduce_on_plateau":
                    # With a validation loader, only advance a plateau
                    # scheduler on epochs that actually ran validation.  If
                    # no validation set exists, retain the historical
                    # fallback to the training loss.
                    if should_validate or self.val_loader is None:
                        monitor_loss = metrics.get(self.scheduler_monitor_key)
                        if monitor_loss is None:
                            monitor_loss = (
                                val_loss if val_loss is not None else avg_loss
                            )
                        self.scheduler.step(monitor_loss)
                elif (
                    not self.step_based_training
                    and not self.scheduler_step_per_optimizer_step
                ):
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            val_mae_nm = metrics.get("val_mae_nm")
            physical_metric_text = ""
            if val_mae_nm is not None:
                joint_mae_nm = [
                    value
                    for key, value in sorted(
                        self.last_val_epoch_metrics.items(),
                        key=lambda item: item[0],
                    )
                    if key.startswith("mae_nm_j")
                ]
                joint_metric_text = ",".join(
                    f"{value:.4f}" for value in joint_mae_nm
                )
                physical_metric_text = (
                    f" val_mae_nm={val_mae_nm:.6f}"
                    f" val_joint_mae_nm=[{joint_metric_text}]"
                )
            train_eval_text = (
                f" train_eval_loss={train_eval_loss:.6f}"
                if train_eval_loss is not None
                else ""
            )
            if val_loss is None:
                log.info(
                    "epoch=%d train_loss_online=%.6f%s lr=%.2e",
                    epoch,
                    avg_loss,
                    train_eval_text,
                    current_lr,
                )
            else:
                log.info(
                    "epoch=%d train_loss_online=%.6f%s val_loss=%.6f%s lr=%.2e",
                    epoch,
                    avg_loss,
                    train_eval_text,
                    val_loss,
                    physical_metric_text,
                    current_lr,
                )

            history_item = {
                "epoch": epoch,
                "global_step": self.global_step,
                **metrics,
            }
            self.loss_history.append(history_item)

            self.save_loss_plot()

            epoch_metrics = {
                "epoch": epoch,
                "train/epoch_loss": avg_loss,
                "train/loss_online": avg_loss,
                "train/learning_rate": current_lr,
            }
            epoch_metrics.update(
                {
                    f"train/epoch_{key}": value
                    for key, value in self.last_train_epoch_metrics.items()
                }
            )
            if train_eval_loss is not None:
                epoch_metrics["train_eval/loss"] = train_eval_loss
                epoch_metrics.update(
                    {
                        f"train_eval/{key}": value
                        for key, value in self.last_train_eval_epoch_metrics.items()
                    }
                )
            if val_loss is not None:
                epoch_metrics["val/loss"] = val_loss
                epoch_metrics.update(
                    {
                        f"val/{key}": value
                        for key, value in self.last_val_epoch_metrics.items()
                    }
            )
            self.log_wandb(epoch_metrics, step=self.global_step)

            # Update early-stopping state before persisting the epoch so a
            # resumed run observes the same patience counter and stopping
            # decision as an uninterrupted run.
            stop_early = self.should_stop_early(epoch, metrics)

            if not self.step_based_training:
                completed_epoch = epoch + 1
                if self.checkpoint_every_epochs > 0:
                    if completed_epoch % self.checkpoint_every_epochs == 0:
                        self.save_scheduled_epoch_checkpoint(
                            completed_epoch, metrics
                        )
                elif not self.validation_enabled and self.monitor_key.startswith("val_"):
                    # A validation score cannot rank checkpoints in this mode.
                    self.save_scheduled_epoch_checkpoint(completed_epoch, metrics)
                else:
                    self.save_topk_checkpoint(epoch, metrics)

            if stop_early:
                log.info(
                    "early stopping at epoch=%d: monitor=%s best=%.6f patience=%d",
                    epoch,
                    self.early_stopping_monitor_key,
                    self.early_stopping_best,
                    self.early_stopping_patience,
                )
                break

            epoch += 1
            self.current_epoch = epoch

        if (
            self.step_based_training
            and self.global_step > 0
            and self._last_step_checkpoint != self.global_step
        ):
            final_metrics = self.loss_history[-1] if self.loss_history else {}
            self.save_step_checkpoint(epoch, final_metrics)
        elif (
            not self.step_based_training
            and self.checkpoint_every_epochs > 0
            and self.loss_history
        ):
            completed_epochs = len(self.loss_history)
            if completed_epochs % self.checkpoint_every_epochs != 0:
                self.save_scheduled_epoch_checkpoint(
                    completed_epochs, self.loss_history[-1]
                )

        self.last_summary = {
            "num_epochs": len(self.loss_history),
            "global_step": self.global_step,
            "last_loss": self.loss_history[-1]["avg_loss"] if self.loss_history else None,
            "last_val_loss": self.loss_history[-1]["val_loss"] if self.loss_history else None,
            "last_train_eval_loss": (
                self.loss_history[-1].get("train_eval_loss")
                if self.loss_history
                else None
            ),
            "last_train_metrics": dict(self.last_train_epoch_metrics),
            "last_train_eval_metrics": dict(self.last_train_eval_epoch_metrics),
            "last_val_metrics": dict(self.last_val_epoch_metrics),
            "last_inference_metrics": dict(self.inference_history[-1]) if self.inference_history else {},
            "validation_enabled": self.validation_enabled,
            "val_every": self.val_every,
            "split": self.current_split_metadata,
            "monitor_key": self.monitor_key,
            "scheduler_monitor_key": self.scheduler_monitor_key,
            "early_stopping_monitor_key": self.early_stopping_monitor_key,
            "top_k": self.top_k,
            "best_checkpoints": self.best_checkpoints,
            "max_optimizer_steps": self.max_optimizer_steps,
            "checkpoint_every_steps": self.checkpoint_every_steps,
            "checkpoint_every_epochs": self.checkpoint_every_epochs,
            "step_based_training": self.step_based_training,
            "stopped_early": (
                self.step_based_training
                and self.max_optimizer_steps is not None
                and self.global_step < self.max_optimizer_steps
            ),
            "output_dir": self.output_dir,
            "ckpt_dir": self.ckpt_dir,
            "ema_enabled": self.ema_enabled,
            "resumed_from": (
                str(self.resume_checkpoint_path)
                if self.resume_checkpoint_path is not None
                else None
            ),
            "resume_epoch": int(self.resume_epoch),
            "wandb_run_id": (
                self.wandb_run.id if self.wandb_run is not None else None
            ),
        }

        self.finish_wandb(exit_code=0)
        return self.last_summary

    def should_stop_early(self, epoch, metrics):
        if not self.early_stopping_enabled:
            return False

        score = metrics.get(self.early_stopping_monitor_key)
        if score is None:
            return False

        if score < self.early_stopping_best - self.early_stopping_min_delta:
            self.early_stopping_best = score
            self.early_stopping_bad_epochs = 0
            return False

        if epoch + 1 < self.early_stopping_warmup:
            return False

        self.early_stopping_bad_epochs += 1
        return self.early_stopping_bad_epochs >= self.early_stopping_patience
        
    
    def save_loss_plot(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        self.output_dir.mkdir(parents=True, exist_ok=True)

        steps = [item["global_step"] for item in self.loss_history]
        epochs = [item["epoch"] for item in self.loss_history]
        losses = [item["avg_loss"] for item in self.loss_history]
        train_eval_losses = [
            item.get("train_eval_loss") for item in self.loss_history
        ]
        val_losses = [item.get("val_loss") for item in self.loss_history]

        def finite_series(x_values, y_values):
            """Drop skipped validation epochs and non-finite metrics."""

            points = []
            for x_value, y_value in zip(x_values, y_values):
                if y_value is None:
                    continue
                try:
                    y_value = float(y_value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(y_value):
                    points.append((x_value, y_value))
            if not points:
                return [], []
            return tuple(zip(*points))

        train_epochs, train_values = finite_series(epochs, losses)
        eval_epochs, eval_values = finite_series(epochs, train_eval_losses)
        val_epochs, val_values = finite_series(epochs, val_losses)
        train_steps, train_step_values = finite_series(steps, losses)
        eval_steps, eval_step_values = finite_series(steps, train_eval_losses)
        val_steps, val_step_values = finite_series(steps, val_losses)
        if self.train_step_history:
            train_steps, train_step_values = finite_series(
                [item["global_step"] for item in self.train_step_history],
                [item["avg_loss"] for item in self.train_step_history],
            )
        inference_steps, inference_values = finite_series(
            [item["global_step"] for item in self.inference_history],
            [item.get("full_dataset_wrench_mse") for item in self.inference_history],
        )
        show_inference = bool(inference_values) or self.checkpoint_monitor_key == "full_dataset_wrench_mse"
        has_train_eval = bool(eval_values)
        has_val = bool(val_values)

        def style_axis(axis, xlabel, ylabel):
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.legend()
            axis.grid(True)

        def save_figure(figure, filename):
            # A viewer can keep refreshing the PNG while training replaces it.
            temporary_path = self.output_dir / f".{filename}.tmp"
            try:
                figure.tight_layout()
                figure.savefig(temporary_path, format="png")
                os.replace(temporary_path, self.output_dir / filename)
            finally:
                plt.close(figure)
                temporary_path.unlink(missing_ok=True)

        figure, axis = plt.subplots()
        axis.plot(train_epochs, train_values, label="train online")
        if has_train_eval:
            axis.plot(eval_epochs, eval_values, label="train eval")
        style_axis(axis, "epoch", "avg_loss")
        save_figure(figure, "loss_epoch.png")

        if has_val:
            figure, axis = plt.subplots()
            axis.plot(val_epochs, val_values, label="val", color="tab:orange")
            style_axis(axis, "epoch", "val_loss")
            save_figure(figure, "val_loss_epoch.png")

        if show_inference:
            figure, axis = plt.subplots()
            axis.plot(inference_steps, inference_values, marker="o", markersize=3,
                      label="full dataset wrench MSE", color="tab:orange")
            style_axis(axis, "optimizer steps", "physical wrench MSE")
            axis.set_title("Full Dataset Wrench MSE")
            if not inference_values:
                axis.text(0.5, 0.5, "Waiting for full-dataset evaluation",
                          ha="center", va="center", transform=axis.transAxes)
            save_figure(figure, "full_dataset_mse_steps.png")

        figure, axis = plt.subplots()
        axis.plot(train_steps, train_step_values, label="train online")
        if has_train_eval:
            axis.plot(eval_steps, eval_step_values, label="train eval")
        if has_val:
            axis.plot(val_steps, val_step_values, label="val")
        style_axis(axis, "steps", "avg_loss")
        save_figure(figure, "loss_steps.png")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(train_epochs, train_values, label="train online")
        if has_train_eval:
            axes[0].plot(eval_epochs, eval_values, label="train eval")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("avg_loss")
        axes[0].set_title("Train Loss / Epoch")
        axes[0].legend()
        axes[0].grid(True)

        if show_inference:
            axes[1].plot(inference_steps, inference_values, marker="o", markersize=3,
                         label="full dataset wrench MSE", color="tab:orange")
            axes[1].set_title("Full Dataset Wrench MSE / Steps")
            style_axis(axes[1], "optimizer steps", "physical wrench MSE")
            if not inference_values:
                axes[1].text(0.5, 0.5, "Waiting for full-dataset evaluation",
                             ha="center", va="center", transform=axes[1].transAxes)
        else:
            if has_val:
                axes[1].plot(val_epochs, val_values, label="val", color="tab:orange")
                axes[1].legend()
            axes[1].set_xlabel("epoch")
            axes[1].set_ylabel("val_loss")
            axes[1].set_title("Val Loss / Epoch")
            axes[1].grid(True)

        axes[2].plot(train_steps, train_step_values, label="train online")
        if has_train_eval:
            axes[2].plot(eval_steps, eval_step_values, label="train eval")
        if has_val:
            axes[2].plot(val_steps, val_step_values, label="val")
        axes[2].set_xlabel("steps")
        axes[2].set_ylabel("loss")
        axes[2].set_title("Loss / Steps")
        axes[2].legend()
        axes[2].grid(True)

        save_figure(fig, "loss_summary.png")

    def evaluate_checkpoint(self):
        """Optional inference metrics used to rank scheduled checkpoints."""
        return {}

    def _save_retained_checkpoint(self, checkpoint, path, *, resume_epoch):
        metrics = checkpoint["metrics"]
        score = None
        if self.checkpoint_monitor_key is not None:
            score = float(metrics[self.checkpoint_monitor_key])
            if not math.isfinite(score):
                raise ValueError("checkpoint ranking metric must be finite")
        checkpoint.update(monitor_key=self.checkpoint_monitor_key, monitor_score=score)
        records = [item for item in self.best_checkpoints if Path(item["path"]) != path]
        if self.checkpoint_monitor_key is not None:
            # Historical checkpoints from another selection policy cannot be ranked.
            records = [item for item in records if item.get("score") is not None
                       and self.checkpoint_monitor_key in (item.get("metrics") or {})]
        records.append({"score": score, "epoch": checkpoint["epoch"],
                        "global_step": checkpoint["global_step"], "path": path,
                        "metrics": dict(metrics)})
        if self.checkpoint_monitor_key is None:
            records.sort(key=lambda item: item["global_step"], reverse=True)
        else:
            records.sort(key=lambda item: (item["score"], -item["global_step"]))
        self.best_checkpoints = records[:self.top_k]
        if self.checkpoint_monitor_key is None:
            self.best_checkpoints.reverse()
        # Persist the updated ranking, including the current checkpoint when retained.
        checkpoint.update(self._checkpoint_runtime_state(resume_epoch=resume_epoch))
        self._save_checkpoint_atomic(checkpoint, path)
        if self.save_latest_checkpoint:
            self._save_checkpoint_atomic(checkpoint, self.ckpt_dir / "latest.pt")
        for removed in records[self.top_k:]:
            removed_path = Path(removed["path"])
            if removed_path.exists():
                removed_path.unlink()

    def save_step_checkpoint(self, epoch, metrics):
        """Save a step checkpoint, ranked by inference score when configured."""

        if self.global_step <= 0:
            return
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        step = int(self.global_step)
        metrics = dict(metrics or {})
        metrics.update(self.evaluate_checkpoint())
        dataloader_config = self.config.get("dataloader") or {}
        normalizer = getattr(self.dataset, "normalizer", None)
        model_state = (
            self.ema.model.state_dict() if self.ema is not None
            else self.model.state_dict()
        )
        checkpoint = {
            "checkpoint_type": "optimizer_step",
            "model_version": getattr(self.model, "MODEL_VERSION", None),
            "epoch": int(epoch),
            "global_step": step,
            "metrics": metrics,
            "avg_loss": metrics.get("avg_loss"),
            "val_loss": metrics.get("val_loss"),
            "model": model_state,
            "model_raw": self.model.state_dict() if self.ema is not None else None,
            "ema": self._ema_checkpoint_metadata(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "config": self.config,
            "dataloader_filters": getattr(self.dataset, "filter_config", {}),
            "sample_rate_hz": getattr(self.dataset, "sample_rate_hz", None),
            "derived_target_config": getattr(
                self, "derived_target_config", {"enabled": False}
            ),
            "normalizer": {
                "stats": getattr(normalizer, "stats", {}),
                "eps": getattr(normalizer, "eps", 1.0e-6),
                "normalize_mode": dataloader_config.get("normalize_mode"),
                "normalize_lowdim_keys": dataloader_config.get(
                    "normalize_lowdim_keys"
                ),
            },
            **self._model_checkpoint_metadata(),
        }
        path = self.ckpt_dir / f"step_{step:08d}.pt"
        self._save_retained_checkpoint(checkpoint, path, resume_epoch=int(epoch) + 1)
        self._last_step_checkpoint = step
        log.info(
            "saved optimizer-step checkpoint: step=%d path=%s retained=%s%s",
            step,
            path,
            [item["global_step"] for item in self.best_checkpoints],
            " latest.pt" if self.save_latest_checkpoint else "",
        )
        
    def save_checkpoint(self, epoch, avg_loss, val_loss=None):

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        model_state = (
            self.ema.model.state_dict() if self.ema is not None
            else self.model.state_dict()
        )
        ckpt = {
            "checkpoint_type": "epoch",
            "model_version": getattr(self.model, "MODEL_VERSION", None),
            "epoch": epoch,
            "avg_loss": avg_loss,
            "val_loss": val_loss,
            "model": model_state,
            "model_raw": self.model.state_dict() if self.ema is not None else None,
            "ema": self._ema_checkpoint_metadata(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "dataloader_filters": getattr(self.dataset, "filter_config", {}),
            "sample_rate_hz": getattr(self.dataset, "sample_rate_hz", None),
            "derived_target_config": getattr(
                self, "derived_target_config", {"enabled": False}
            ),
            "normalizer": {
                "stats": self.dataset.normalizer.stats,
                "eps": self.dataset.normalizer.eps,
                "normalize_mode": self.config["dataloader"].get("normalize_mode"),
                "normalize_lowdim_keys": self.config["dataloader"].get("normalize_lowdim_keys"),
            },
            **self._model_checkpoint_metadata(),
        }
        ckpt.update(self._checkpoint_runtime_state(resume_epoch=int(epoch) + 1))

        path = self.ckpt_dir / f"epoch_{epoch:03d}.pt"
        self._save_checkpoint_atomic(ckpt, path)
        # log.info(f"saved checkpoint: {path}")

    def save_scheduled_epoch_checkpoint(self, epoch, metrics):
        """Save a scheduled epoch checkpoint, optionally ranked by inference."""

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        epoch = int(epoch)
        metrics = dict(metrics or {})
        metrics.update(self.evaluate_checkpoint())
        dataloader_config = self.config.get("dataloader") or {}
        normalizer = getattr(self.dataset, "normalizer", None)
        model_state = (
            self.ema.model.state_dict() if self.ema is not None
            else self.model.state_dict()
        )
        checkpoint = {
            "checkpoint_type": "scheduled_epoch",
            "model_version": getattr(self.model, "MODEL_VERSION", None),
            "epoch": epoch,
            "global_step": int(self.global_step),
            "metrics": metrics,
            "avg_loss": metrics.get("avg_loss"),
            "val_loss": metrics.get("val_loss"),
            "model": model_state,
            "model_raw": self.model.state_dict() if self.ema is not None else None,
            "ema": self._ema_checkpoint_metadata(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "config": self.config,
            "dataloader_filters": getattr(self.dataset, "filter_config", {}),
            "sample_rate_hz": getattr(self.dataset, "high_fps", None),
            "normalizer": {
                "stats": getattr(normalizer, "stats", {}),
                "eps": getattr(normalizer, "eps", 1.0e-6),
                "normalize_mode": dataloader_config.get("normalize_mode"),
                "normalize_lowdim_keys": dataloader_config.get(
                    "normalize_lowdim_keys"
                ),
            },
            **self._model_checkpoint_metadata(),
        }
        path = self.ckpt_dir / f"epoch_{epoch:07d}.pt"
        self._save_retained_checkpoint(checkpoint, path, resume_epoch=epoch)
        log.info(
            "saved scheduled epoch checkpoint: epoch=%d path=%s retained=%s%s",
            epoch,
            path,
            [item["epoch"] for item in self.best_checkpoints],
            " latest.pt" if self.save_latest_checkpoint else "",
        )

    def save_topk_checkpoint(self, epoch, metrics, checkpoint_label=None):
        score = metrics.get(self.monitor_key)

        if score is None:
            log.warning(f"monitor_key={self.monitor_key} is None, skip checkpoint")
            return

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        label = (
            f"epoch_{epoch:03d}"
            if checkpoint_label is None
            else str(checkpoint_label)
        )
        path = self.ckpt_dir / f"{label}_{self.monitor_key}_{score:.6f}.pt"

        model_state = (
            self.ema.model.state_dict() if self.ema is not None
            else self.model.state_dict()
        )
        ckpt = {
            "checkpoint_type": "topk_epoch",
            "model_version": getattr(self.model, "MODEL_VERSION", None),
            "epoch": epoch,
            "global_step": self.global_step,
            "monitor_key": self.monitor_key,
            "monitor_score": score,
            "metrics": metrics,
            "model": model_state,
            "model_raw": self.model.state_dict() if self.ema is not None else None,
            "ema": self._ema_checkpoint_metadata(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "config": self.config,
            "dataloader_filters": getattr(self.dataset, "filter_config", {}),
            "sample_rate_hz": getattr(self.dataset, "sample_rate_hz", None),
            "derived_target_config": getattr(
                self, "derived_target_config", {"enabled": False}
            ),
            "normalizer": {
                "stats": self.dataset.normalizer.stats,
                "eps": self.dataset.normalizer.eps,
                "normalize_mode": self.config["dataloader"].get("normalize_mode"),
                "normalize_lowdim_keys": self.config["dataloader"].get("normalize_lowdim_keys"),
            },
            **self._model_checkpoint_metadata(),
        }
        ckpt.update(self._checkpoint_runtime_state(resume_epoch=int(epoch) + 1))

        self._save_checkpoint_atomic(ckpt, path)
        # log.info(f"saved checkpoint: {path}")

        self.best_checkpoints.append({
            "score": score,
            "path": path,
            "epoch": epoch,
            "global_step": self.global_step,
            "metrics": dict(metrics),
        })

        self.best_checkpoints.sort(key=lambda item: item["score"])

        while len(self.best_checkpoints) > self.top_k:
            removed = self.best_checkpoints.pop(-1)
            removed_path = removed["path"]
            if removed_path.exists():
                removed_path.unlink()
                # log.info(f"removed checkpoint: {removed_path}")

    def _ema_checkpoint_metadata(self):
        if self.ema is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "decay": self.ema.decay,
            "update_after_step": self.ema.update_after_step,
            "update_every": self.ema.update_every,
            "global_step": self.global_step,
        }

    def format_summary(self, summary):
        lines = []
        lines.append("Training finished")
        lines.append(f"num_epochs: {summary['num_epochs']}")
        lines.append(f"global_step: {summary['global_step']}")
        if summary.get("max_optimizer_steps") is not None:
            lines.append(
                f"max_optimizer_steps: {summary['max_optimizer_steps']}"
            )
            lines.append(f"stopped_early: {summary['stopped_early']}")
        lines.append(f"last_loss: {summary['last_loss']}")
        lines.append(f"last_train_eval_loss: {summary.get('last_train_eval_loss')}")
        lines.append(f"last_val_loss: {summary['last_val_loss']}")
        if summary.get("last_val_metrics", {}).get("mae_nm") is not None:
            lines.append(
                f"last_val_mae_nm: {summary['last_val_metrics']['mae_nm']}"
            )
            joint_mae = [
                summary["last_val_metrics"][key]
                for key in sorted(summary["last_val_metrics"])
                if key.startswith("mae_nm_j")
            ]
            lines.append(f"last_val_joint_mae_nm: {joint_mae}")
        lines.append(f"monitor_key: {summary['monitor_key']}")
        lines.append(f"output_dir: {summary['output_dir']}")
        lines.append(f"ckpt_dir: {summary['ckpt_dir']}")
        lines.append(f"ema_enabled: {summary['ema_enabled']}")
        if summary.get("resumed_from"):
            lines.append(f"resumed_from: {summary['resumed_from']}")
        lines.append(f"wandb_run_id: {summary['wandb_run_id']}")
        lines.append("best_checkpoints:")
    
        for item in summary["best_checkpoints"]:
            step_text = (
                f" step={item['global_step']}"
                if "global_step" in item
                else ""
            )
            score = item.get("score")
            score_text = f" score={score:.6f}" if score is not None else ""
            lines.append(
                f"  epoch={item['epoch']}{step_text}{score_text} "
                f"path={item['path']}"
            )
    
        return "\n".join(lines)
