if __name__ == "__main__":
    import os
    import pathlib
    import sys

    root_dir = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(root_dir)
    os.chdir(root_dir)

import copy
import contextlib
import dill
import logging
import math
import os
import pathlib
import random
from dataclasses import dataclass
from typing import Optional

import hydra
import numpy as np
import torch
import torch.distributed as dist
import tqdm
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from omegaconf import open_dict
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.common.phase_sampler import PhaseWeightedDistributedSampler
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.policy.force_aware_diffusion_transformer_policy import (
    ForceAwareDiffusionTransformerPolicy,
)
from diffusion_policy.workspace.base_workspace import BaseWorkspace


OmegaConf.register_new_resolver("eval", eval, replace=True)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistributedContext:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


class _NullLogger(contextlib.AbstractContextManager):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def log(self, value, **kwargs) -> None:
        pass


def resolve_amp_dtype(dtype_name: str) -> torch.dtype:
    normalized = str(dtype_name).lower()
    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if normalized not in dtype_by_name:
        raise ValueError(
            "training.amp.dtype must be one of: bfloat16, bf16, float16, fp16"
        )
    return dtype_by_name[normalized]


def per_rank_batch_size(
    batch_size: int,
    world_size: int,
    batch_size_is_global: bool,
) -> int:
    batch_size = int(batch_size)
    world_size = int(world_size)
    if batch_size < 1 or world_size < 1:
        raise ValueError("batch_size and world_size must be positive")
    if not batch_size_is_global or world_size == 1:
        return batch_size
    if batch_size % world_size != 0:
        raise ValueError(
            f"global batch_size={batch_size} must be divisible by world_size={world_size}"
        )
    return batch_size // world_size


def initialize_distributed(training_cfg) -> DistributedContext:
    distributed_cfg = training_cfg.get("distributed", {})
    requested = bool(distributed_cfg.get("enabled", False))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not requested:
        raise RuntimeError(
            "torchrun provided WORLD_SIZE > 1, but training.distributed.enabled is false"
        )
    if not requested or world_size == 1:
        if requested and world_size == 1 and torch.cuda.device_count() > 1:
            print(
                "[DDP] training.distributed.enabled=true but WORLD_SIZE=1; "
                "this is single-GPU training. Launch with torchrun for DDP.",
                flush=True,
            )
        return DistributedContext()
    if not torch.cuda.is_available():
        raise RuntimeError("multi-GPU training requires CUDA")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} exceeds visible CUDA device count "
            f"{torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=str(distributed_cfg.get("backend", "nccl")),
        init_method="env://",
    )
    print(
        f"[DDP] rank={rank}/{world_size} local_rank={local_rank} "
        f"device=cuda:{local_rank} pid={os.getpid()}",
        flush=True,
    )
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def shared_output_dir(
    distributed: DistributedContext,
    output_dir: Optional[str] = None,
) -> str:
    if output_dir is None:
        output_dir = str(HydraConfig.get().runtime.output_dir)
    else:
        output_dir = str(output_dir)
    if not distributed.enabled:
        return output_dir
    values = [output_dir if distributed.is_main else None]
    dist.broadcast_object_list(
        values,
        src=0,
        device=torch.device("cuda", distributed.local_rank),
    )
    return str(values[0])


def optimizer_updates_per_epoch(num_batches: int, accumulation_steps: int) -> int:
    if num_batches < 0 or accumulation_steps < 1:
        raise ValueError("num_batches must be non-negative and accumulation_steps positive")
    return math.ceil(num_batches / accumulation_steps)


def resolve_training_limits(training_cfg) -> tuple[Optional[int], Optional[int]]:
    num_epochs = training_cfg.get("num_epochs")
    num_optimizer_steps = training_cfg.get("num_optimizer_steps")
    if (num_epochs is None) == (num_optimizer_steps is None):
        raise ValueError(
            "exactly one of training.num_epochs and "
            "training.num_optimizer_steps must be set"
        )

    if num_epochs is not None:
        num_epochs = int(num_epochs)
        if num_epochs < 1:
            raise ValueError("training.num_epochs must be positive or null")
    if num_optimizer_steps is not None:
        num_optimizer_steps = int(num_optimizer_steps)
        if num_optimizer_steps < 1:
            raise ValueError(
                "training.num_optimizer_steps must be positive or null"
            )
    return num_epochs, num_optimizer_steps


class TrainForceAwareDiffusionWorkspace(BaseWorkspace):
    include_keys = ("global_step", "optimizer_step", "epoch", "freeze_config", "topk_state")

    def __init__(
        self,
        cfg: OmegaConf,
        output_dir: Optional[str] = None,
        distributed: Optional[DistributedContext] = None,
    ):
        super().__init__(cfg, output_dir=output_dir)
        self.distributed = distributed or DistributedContext()
        seed = int(cfg.training.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: ForceAwareDiffusionTransformerPolicy = hydra.utils.instantiate(
            cfg.policy
        )
        if not isinstance(self.model, ForceAwareDiffusionTransformerPolicy):
            raise TypeError("cfg.policy must instantiate ForceAwareDiffusionTransformerPolicy")
        # This is saved separately from model weights because requires_grad is
        # not part of a PyTorch state_dict.  It makes a checkpoint self-
        # describing when DINOv3 or the fusion block is frozen.
        self.freeze_config = self.model.get_freeze_config()

        self.ema_model: Optional[ForceAwareDiffusionTransformerPolicy] = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)
        amp_cfg = cfg.training.get("amp", {})
        amp_enabled = bool(amp_cfg.get("enabled", False))
        amp_dtype = resolve_amp_dtype(amp_cfg.get("dtype", "bfloat16"))
        scaler_enabled = amp_enabled and amp_dtype == torch.float16
        self.grad_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=scaler_enabled,
        )
        self.global_step = 0
        self.optimizer_step = 0
        self.epoch = 0
        self.topk_state = {}

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        """Restore freeze settings before loading optimizer state.

        A fine-tuning checkpoint may have a different set of trainable
        parameters from the current config.  Rebuild the optimizer first so
        its parameter groups agree with the saved model; this also lets a
        checkpoint switch between full and partial freezing safely.
        """
        serialized = payload.get("pickles", {}).get("freeze_config")
        if serialized is not None:
            saved_config = dill.loads(serialized)
            if not isinstance(saved_config, dict):
                saved_config = {"freeze_modules": tuple(saved_config)}
            saved_config = dict(saved_config)
            if saved_config != self.freeze_config:
                self.model.apply_freeze_config(saved_config)
                if self.ema_model is not None:
                    self.ema_model.apply_freeze_config(saved_config)
                self.freeze_config = self.model.get_freeze_config()
                optimizer_cfg = OmegaConf.to_container(
                    self.cfg.optimizer, resolve=True
                )
                self.optimizer = self.model.get_optimizer(**optimizer_cfg)
        return super().load_payload(
            payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs,
        )

    @property
    def is_main_process(self) -> bool:
        return getattr(self, "distributed", DistributedContext()).is_main

    def _autocast(self, cfg, device):
        amp_cfg = cfg.training.get("amp", {})
        enabled = bool(amp_cfg.get("enabled", False))
        if enabled and device.type != "cuda":
            raise RuntimeError("training.amp.enabled requires a CUDA device")
        dtype = resolve_amp_dtype(amp_cfg.get("dtype", "bfloat16"))
        return torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=enabled,
        )

    def _distributed_mean(self, value: float, device) -> float:
        distributed = getattr(self, "distributed", DistributedContext())
        if not distributed.enabled:
            return float(value)
        tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float((tensor / distributed.world_size).item())

    def run(self) -> None:
        cfg = copy.deepcopy(self.cfg)
        distributed = self.distributed
        num_epochs, num_optimizer_steps = resolve_training_limits(cfg.training)
        if cfg.training.debug:
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 2
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            if cfg.training.get("checkpoint_every_optimizer_steps") is not None:
                cfg.training.checkpoint_every_optimizer_steps = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            if num_epochs is not None:
                num_epochs = min(num_epochs, 2)
                cfg.training.num_epochs = num_epochs
            else:
                num_optimizer_steps = min(num_optimizer_steps, 3)
                cfg.training.num_optimizer_steps = num_optimizer_steps

        latest_checkpoint = self.get_checkpoint_path()
        if cfg.training.resume and latest_checkpoint.is_file():
            self.load_checkpoint(path=latest_checkpoint)

        # The dataset may construct a CUDA TorchCodec decoder during
        # instantiation.  Select the configured device first so single-GPU
        # runs do not accidentally decode on cuda:0 while training on cuda:1.
        configured_device = torch.device(str(cfg.training.device))
        if (
            configured_device.type == "cuda"
            and torch.cuda.is_available()
            and not distributed.enabled
        ):
            torch.cuda.set_device(
                configured_device.index
                if configured_device.index is not None
                else torch.cuda.current_device()
            )

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        if not isinstance(dataset, BaseImageDataset):
            raise TypeError("cfg.task.dataset must instantiate BaseImageDataset")
        if hasattr(dataset, "relative_pose_actions"):
            if bool(dataset.relative_pose_actions) != bool(
                self.model.relative_pose_actions
            ):
                raise ValueError(
                    "dataset and policy relative_pose_actions settings must match"
                )
            if dataset.action_reference_key != self.model.action_reference_key:
                raise ValueError(
                    "dataset and policy action_reference_key settings must match"
                )
        validation_dataset = dataset.get_validation_dataset()
        distributed_cfg = cfg.training.get("distributed", {})
        batch_size_is_global = bool(
            distributed_cfg.get("batch_size_is_global", True)
        )
        train_loader_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
        val_loader_cfg = OmegaConf.to_container(cfg.val_dataloader, resolve=True)
        train_loader_cfg["batch_size"] = per_rank_batch_size(
            train_loader_cfg["batch_size"],
            distributed.world_size,
            batch_size_is_global,
        )
        val_loader_cfg["batch_size"] = per_rank_batch_size(
            val_loader_cfg["batch_size"],
            distributed.world_size,
            batch_size_is_global,
        )

        train_sampler = None
        val_sampler = None
        phase_resampling_cfg = cfg.get("phase_resampling", {})
        if OmegaConf.is_config(phase_resampling_cfg):
            phase_resampling_cfg = OmegaConf.to_container(
                phase_resampling_cfg, resolve=True
            )
        else:
            phase_resampling_cfg = dict(phase_resampling_cfg or {})
        phase_resampling_enabled = bool(
            phase_resampling_cfg.get("enabled", False)
        )
        if phase_resampling_enabled:
            get_phase_labels = getattr(dataset, "get_phase_labels", None)
            if not callable(get_phase_labels):
                raise TypeError(
                    "phase_resampling.enabled requires the dataset to expose "
                    "get_phase_labels()"
                )
            phase_weights_cfg = phase_resampling_cfg.get(
                "weights",
                {"free": 1.0, "precontact": 5.0, "contact": 2.0},
            )
            phase_weights = {
                0: float(phase_weights_cfg.get("free", 1.0)),
                1: float(phase_weights_cfg.get("precontact", 5.0)),
                2: float(phase_weights_cfg.get("contact", 2.0)),
            }
            train_sampler = PhaseWeightedDistributedSampler(
                dataset,
                phase_labels=get_phase_labels(),
                phase_weights=phase_weights,
                num_replicas=distributed.world_size,
                rank=distributed.rank,
                seed=int(cfg.training.seed),
            )
            train_loader_cfg.pop("shuffle", None)
            if self.is_main_process:
                fractions = train_sampler.expected_phase_fractions
                log.info(
                    "Phase resampling enabled: counts=%s, weights=%s, "
                    "expected fractions={free=%.3f, precontact=%.3f, contact=%.3f}",
                    train_sampler.phase_counts,
                    phase_weights,
                    fractions[0],
                    fractions[1],
                    fractions[2],
                )
        elif distributed.enabled:
            train_sampler = DistributedSampler(
                dataset,
                num_replicas=distributed.world_size,
                rank=distributed.rank,
                shuffle=bool(train_loader_cfg.pop("shuffle", True)),
                seed=int(cfg.training.seed),
            )
        if distributed.enabled:
            # Validation has one reduction after the loop, so ranks can have
            # unequal lengths. Avoid padding/repeating validation samples.
            val_sampler = range(
                distributed.rank, len(validation_dataset), distributed.world_size
            )
            val_loader_cfg.pop("shuffle", None)
        train_dataloader = DataLoader(
            dataset,
            sampler=train_sampler,
            **train_loader_cfg,
        )
        val_dataloader = DataLoader(
            validation_dataset,
            sampler=val_sampler,
            **val_loader_cfg,
        )
        if distributed.enabled:
            print(
                f"[DDP] rank={distributed.rank} train_batch={train_loader_cfg['batch_size']} "
                f"train_batches={len(train_dataloader)} "
                f"val_batches={len(val_dataloader)}",
                flush=True,
            )

        normalizer = dataset.get_normalizer(**cfg.task.get("normalizer", {}))
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        accumulation_steps = int(cfg.training.gradient_accumulate_every)
        max_train_batches = len(train_dataloader)
        if cfg.training.max_train_steps is not None:
            max_train_batches = min(max_train_batches, int(cfg.training.max_train_steps))
        optimizer_steps_per_epoch = optimizer_updates_per_epoch(
            max_train_batches,
            accumulation_steps,
        )
        if optimizer_steps_per_epoch < 1:
            raise ValueError("training dataloader must produce at least one batch")
        total_optimizer_steps = (
            optimizer_steps_per_epoch * num_epochs
            if num_epochs is not None
            else num_optimizer_steps
        )
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=int(cfg.training.lr_warmup_steps),
            num_training_steps=total_optimizer_steps,
            last_epoch=self.optimizer_step - 1,
        )

        ema: Optional[EMAModel] = None
        if self.ema_model is not None:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
            ema.optimization_step = self.optimizer_step

        env_runner: Optional[BaseImageRunner] = None
        # Rollouts are rank-0-only by design.  Running them inside the DDP
        # epoch makes every other rank idle at the next collective.  Keep the
        # behavior for single-GPU runs; DDP users can run the same evaluator
        # from a saved checkpoint after training.
        run_rank0_eval = not distributed.enabled or bool(
            cfg.training.get("distributed", {}).get("run_rank0_eval", False)
        )
        if self.is_main_process and run_rank0_eval and cfg.task.get("env_runner") is not None:
            env_runner_cfg = cfg.task.env_runner
            env_runner_kwargs = {"output_dir": self.output_dir}
            # OfflineValidationRunner declares dataset_cfg and can share the
            # already-preloaded training dataset instead of decoding videos
            # again during runner construction.
            if env_runner_cfg.get("dataset_cfg") is not None:
                env_runner_kwargs["dataset"] = dataset
                # OfflineValidationRunner can use the exact validation view
                # already attached to this workspace.  This avoids rebuilding
                # dataset state (and, for lazy video datasets, decoding again).
                target = str(env_runner_cfg.get("_target_", ""))
                if target.endswith("OfflineValidationRunner"):
                    env_runner_kwargs["validation_dataset"] = validation_dataset
                    env_runner_kwargs["dataloader"] = val_dataloader
            env_runner = hydra.utils.instantiate(
                env_runner_cfg,
                **env_runner_kwargs,
            )
            if not isinstance(env_runner, BaseImageRunner):
                raise TypeError("cfg.task.env_runner must instantiate BaseImageRunner")

        if self.is_main_process:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging,
            )
            self.update_wandb_output_dir()
        else:
            wandb_run = _NullLogger()
        checkpoint_every_optimizer_steps = cfg.training.get(
            "checkpoint_every_optimizer_steps"
        )
        if (
            checkpoint_every_optimizer_steps is not None
            and int(checkpoint_every_optimizer_steps) < 1
        ):
            raise ValueError(
                "training.checkpoint_every_optimizer_steps must be positive or null"
            )
        topk_manager = None
        if self.is_main_process:
            topk_manager = self._make_topk_manager(cfg)

        device = torch.device(cfg.training.device)
        amp_cfg = cfg.training.get("amp", {})
        allow_tf32 = bool(amp_cfg.get("allow_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        self.optimizer.zero_grad(set_to_none=True)

        training_model = self.model
        if distributed.enabled:
            training_model = DistributedDataParallel(
                self.model,
                device_ids=[distributed.local_rank],
                output_device=distributed.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=bool(
                    distributed_cfg.get("find_unused_parameters", False)
                ),
            )

        # Keep initialization deterministic across ranks, then decorrelate runtime
        # randomness used by augmentation, noise, and curriculum masking.
        runtime_seed = int(cfg.training.seed) + distributed.rank
        torch.manual_seed(runtime_seed)
        np.random.seed(runtime_seed)
        random.seed(runtime_seed)

        sample_batch = None
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        logger_context = (
            JsonLogger(log_path) if self.is_main_process else _NullLogger()
        )
        with logger_context as json_logger:
            def training_is_complete() -> bool:
                if num_epochs is not None:
                    return self.epoch >= num_epochs
                return self.optimizer_step >= num_optimizer_steps

            while not training_is_complete():
                if train_sampler is not None:
                    train_sampler.set_epoch(self.epoch)
                step_log = self._train_epoch(
                    cfg=cfg,
                    dataloader=train_dataloader,
                    device=device,
                    training_model=training_model,
                    lr_scheduler=lr_scheduler,
                    ema=ema,
                    wandb_run=wandb_run,
                    json_logger=json_logger,
                    topk_manager=topk_manager,
                    target_optimizer_steps=num_optimizer_steps,
                    val_dataloader=val_dataloader,
                )
                if sample_batch is None and self.is_main_process:
                    sample_batch = next(iter(train_dataloader))

                policy = self.ema_model if self.ema_model is not None else self.model
                policy.eval()
                if (
                    self.is_main_process
                    and env_runner is not None
                    and self.epoch % int(cfg.training.rollout_every) == 0
                ):
                    step_log.update(env_runner.run(policy))

                if self.epoch % int(cfg.training.val_every) == 0:
                    val_metrics = self._validate_and_log(
                        cfg, val_dataloader, device, wandb_run, json_logger,
                    )
                    if val_metrics is not None:
                        step_log.update(val_metrics)

                if (
                    self.is_main_process
                    and run_rank0_eval
                    and self.epoch % int(cfg.training.sample_every) == 0
                ):
                    batch = dict_apply(
                        sample_batch,
                        lambda value: value.to(device, non_blocking=True),
                    )
                    with torch.no_grad(), self._autocast(cfg, device):
                        sample_generator = torch.Generator(device=device)
                        sample_generator.manual_seed(int(cfg.training.seed))
                        prediction = policy.predict_action(
                            batch["obs"],
                            generator=sample_generator,
                        )["model_action_pred"]
                        # Keep this metric in the same normalized action space
                        # as OfflineValidationRunner.val_action_mse.  The raw
                        # value is retained separately for physical-unit plots.
                        normalized_prediction = policy.normalizer["action"].normalize(
                            prediction
                        )
                        normalized_target = policy.normalizer["action"].normalize(
                            batch["action"]
                        )
                        step_log["train_action_mse_error"] = torch.nn.functional.mse_loss(
                            normalized_prediction,
                            normalized_target,
                        ).item()
                        step_log["train_action_mse_error_raw"] = torch.nn.functional.mse_loss(
                            prediction,
                            batch["action"],
                        ).item()

                completed_epoch = self.epoch
                step_log.update(
                    epoch=completed_epoch,
                    global_step=self.global_step,
                    optimizer_step=self.optimizer_step,
                )
                if self.is_main_process:
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                self.epoch += 1

                checkpoint_every_optimizer_steps = cfg.training.get(
                    "checkpoint_every_optimizer_steps"
                )
                use_optimizer_step_checkpoints = (
                    checkpoint_every_optimizer_steps is not None
                )
                should_checkpoint = (
                    not use_optimizer_step_checkpoints
                    and self.epoch % int(cfg.training.checkpoint_every) == 0
                )
                should_save_final = training_is_complete()
                if use_optimizer_step_checkpoints:
                    checkpoint_interval = int(checkpoint_every_optimizer_steps)
                    should_save_final = (
                        should_save_final
                        and self.optimizer_step % checkpoint_interval != 0
                    )
                should_checkpoint = should_checkpoint or should_save_final
                if should_checkpoint:
                    if self._uses_validation_checkpoint(cfg):
                        step_log.update(
                            self._validate_and_log(
                                cfg, val_dataloader, device, wandb_run, json_logger,
                            )
                        )
                    if self.is_main_process:
                        if use_optimizer_step_checkpoints:
                            self._save_optimizer_step_checkpoint(
                                cfg,
                                topk_manager=topk_manager,
                                metric_dict=step_log,
                            )
                        else:
                            self._save_checkpoints(cfg, step_log, topk_manager)

                policy.train()
                # DDP's gradient collectives already synchronize at the next
                # batch.  An unconditional epoch barrier only amplifies the
                # rank-0 checkpoint/evaluation stall.

        if self._saving_thread is not None:
            self._saving_thread.join()
        if self.is_main_process:
            wandb_run.finish()

    def _train_epoch(
        self,
        cfg,
        dataloader,
        device,
        lr_scheduler,
        ema,
        wandb_run,
        json_logger,
        topk_manager=None,
        target_optimizer_steps: Optional[int] = None,
        training_model=None,
        val_dataloader=None,
    ) -> dict:
        self.model.train()
        if training_model is None:
            training_model = self.model
        maximum = len(dataloader)
        if cfg.training.max_train_steps is not None:
            maximum = min(maximum, int(cfg.training.max_train_steps))
        log_interval = int(cfg.training.get("log_every_optimizer_steps", 50))
        if log_interval < 1:
            raise ValueError("training.log_every_optimizer_steps must be positive")
        loss_sum = None
        loss_count = 0
        last_log = {}

        with tqdm.tqdm(
            dataloader,
            total=maximum,
            desc=f"Training epoch {self.epoch}",
            leave=False,
            mininterval=float(cfg.training.tqdm_interval_sec),
            disable=not self.is_main_process,
        ) as progress:
            batch_iterator = iter(progress)
            batch_idx = 0
            while (
                batch_idx < maximum
                and (
                    target_optimizer_steps is None
                    or self.optimizer_step < target_optimizer_steps
                )
            ):
                group_size = min(
                    int(cfg.training.gradient_accumulate_every),
                    maximum - batch_idx,
                )
                group_losses = []
                for microbatch_idx in range(group_size):
                    batch = next(batch_iterator)
                    batch = dict_apply(
                        batch,
                        lambda value: value.to(device, non_blocking=True),
                    )
                    should_sync = microbatch_idx == group_size - 1
                    sync_context = contextlib.nullcontext()
                    if (
                        isinstance(training_model, DistributedDataParallel)
                        and not should_sync
                    ):
                        sync_context = training_model.no_sync()
                    with sync_context, self._autocast(cfg, device):
                        if isinstance(training_model, DistributedDataParallel):
                            raw_loss = training_model(
                                batch,
                                optimizer_step=self.optimizer_step,
                            )
                        else:
                            raw_loss = self.model.compute_loss(
                                batch,
                                optimizer_step=self.optimizer_step,
                            )
                        scaled_loss = raw_loss / group_size
                    grad_scaler = getattr(self, "grad_scaler", None)
                    if grad_scaler is None:
                        grad_scaler = torch.amp.GradScaler(
                            device.type,
                            enabled=False,
                        )
                    grad_scaler.scale(scaled_loss).backward()
                    # Keep loss tensors on device through the microbatch loop;
                    # calling ``item`` here forces a CUDA synchronization.
                    group_losses.append(raw_loss.detach())
                    self.global_step += 1
                    batch_idx += 1

                if grad_scaler.is_enabled():
                    grad_scaler.unscale_(self.optimizer)
                if cfg.training.get("max_grad_norm") is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        float(cfg.training.max_grad_norm),
                    )
                grad_scaler.step(self.optimizer)
                grad_scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                self.optimizer_step += 1
                self.model.set_optimizer_step(self.optimizer_step)
                if ema is not None:
                    ema.step(self.model)
                    self.ema_model.set_optimizer_step(self.optimizer_step)

                group_loss_tensor = torch.stack(group_losses).mean()
                loss_sum = group_loss_tensor if loss_sum is None else loss_sum + group_loss_tensor
                loss_count += 1
                should_log = (
                    self.optimizer_step % log_interval == 0
                    or batch_idx >= maximum
                    or (
                        target_optimizer_steps is not None
                        and self.optimizer_step >= target_optimizer_steps
                    )
                )
                loss_value = None
                if should_log:
                    loss_value = float(group_loss_tensor.item())
                    progress.set_postfix(loss=loss_value, refresh=False)
                last_log = {
                    # The scalar reduction is deferred to the epoch summary;
                    # one all-reduce per optimizer step is unnecessary for
                    # training and was measurable with small per-rank batches.
                    "train_loss": loss_value,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": self.epoch,
                    "global_step": self.global_step,
                    "optimizer_step": self.optimizer_step,
                    **self.model.last_curriculum_metrics,
                }
                if self.is_main_process and should_log:
                    wandb_run.log(last_log, step=self.global_step)
                    json_logger.log(last_log)
                checkpoint_every_optimizer_steps = cfg.training.get(
                    "checkpoint_every_optimizer_steps"
                )
                if checkpoint_every_optimizer_steps is not None:
                    checkpoint_interval = int(checkpoint_every_optimizer_steps)
                    if checkpoint_interval < 1:
                        raise ValueError(
                            "training.checkpoint_every_optimizer_steps must be "
                            "positive or null"
                        )
                    if self.optimizer_step % checkpoint_interval == 0:
                        if last_log["train_loss"] is None:
                            last_log["train_loss"] = float(group_loss_tensor.item())
                        if self._uses_validation_checkpoint(cfg):
                            last_log.update(
                                self._validate_and_log(
                                    cfg, val_dataloader, device, wandb_run, json_logger,
                                )
                            )
                        if self.is_main_process:
                            self._save_optimizer_step_checkpoint(
                                cfg,
                                topk_manager=topk_manager,
                                metric_dict=last_log,
                            )

        epoch_loss = (
            float((loss_sum / loss_count).item()) if loss_sum is not None else float("nan")
        )
        last_log["train_loss"] = self._distributed_mean(epoch_loss, device)
        return last_log

    @staticmethod
    def _uses_validation_checkpoint(cfg) -> bool:
        monitor = OmegaConf.select(cfg, "checkpoint.topk.monitor_key")
        return monitor is not None and str(monitor).startswith("val_")

    def _make_topk_manager(self, cfg):
        manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )
        state = getattr(self, "topk_state", {})
        if all(state.get(key) == getattr(manager, key) for key in ("monitor_key", "mode", "format_str")):
            candidates = [
                (os.path.join(manager.save_dir, pathlib.Path(name).name), value)
                for name, value in state.get("path_value_map", {}).items()
                if math.isfinite(value)
            ]
            candidates = [(path, value) for path, value in candidates if os.path.isfile(path)]
            if manager.monitor_key is None:
                selected = candidates[-manager.k:] if manager.k else []
            else:
                candidates.sort(key=lambda item: item[1], reverse=manager.mode == "max")
                selected = candidates[:manager.k]
            manager.path_value_map = dict(selected)
        return manager

    def _topk_checkpoint_path(self, topk_manager, metrics):
        if topk_manager is None:
            return None
        data = {key.replace("/", "_"): value for key, value in metrics.items()}
        if topk_manager.monitor_key is not None:
            value = data.get(topk_manager.monitor_key)
            if value is None or not math.isfinite(float(value)):
                return None
        # A final/epoch checkpoint can coincide with a step checkpoint. Do not
        # let a duplicate candidate evict a different Top-K entry.
        candidate = os.path.join(topk_manager.save_dir, topk_manager.format_str.format(**data))
        if candidate in topk_manager.path_value_map:
            return None
        path = topk_manager.get_ckpt_path(data)
        self.topk_state = {
            "monitor_key": topk_manager.monitor_key,
            "mode": topk_manager.mode,
            "format_str": topk_manager.format_str,
            "path_value_map": {
                pathlib.Path(name).name: value
                for name, value in topk_manager.path_value_map.items()
            },
        }
        return path

    def _update_best_checkpoint(self, topk_manager) -> None:
        if topk_manager is None or topk_manager.monitor_key is None or not topk_manager.path_value_map:
            return
        choose = min if topk_manager.mode == "min" else max
        best = choose(topk_manager.path_value_map, key=topk_manager.path_value_map.get)
        directory = pathlib.Path(topk_manager.save_dir)
        temporary_link = directory.joinpath(f".best-{os.getpid()}.tmp")
        temporary_link.unlink(missing_ok=True)
        temporary_link.symlink_to(pathlib.Path(best).name)
        os.replace(temporary_link, directory.joinpath("best.ckpt"))

    def _save_latest_checkpoint(self) -> None:
        # latest may be a symlink to a Top-K file. Replace the directory entry
        # instead of opening that symlink and overwriting the retained model.
        latest = self.get_checkpoint_path()
        temporary_path = latest.with_name(f".latest-{os.getpid()}.tmp.ckpt")
        try:
            self.save_checkpoint(path=temporary_path, use_thread=False)
            os.replace(temporary_path, latest)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _save_checkpoints(self, cfg, step_log, topk_manager) -> None:
        topk_path = self._topk_checkpoint_path(topk_manager, step_log)
        if topk_path is not None:
            self.save_checkpoint(path=topk_path, use_thread=False)
        self._update_best_checkpoint(topk_manager)
        if cfg.checkpoint.save_last_ckpt:
            self._save_latest_checkpoint()
        if cfg.checkpoint.save_last_snapshot:
            self.save_snapshot()

    def _save_optimizer_step_checkpoint(
        self,
        cfg,
        topk_manager=None,
        metric_dict=None,
    ) -> None:
        step_tag = f"optimizer_step={self.optimizer_step:08d}"
        checkpoint_dir = pathlib.Path(self.output_dir).joinpath("checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # In step-based mode, let TopKCheckpointManager choose the filename.
        # With monitor_key=null it keeps the most recent K checkpoints; with a
        # metric it keeps the best K according to the configured mode.
        checkpoint_path = None
        if topk_manager is not None:
            topk_path = self._topk_checkpoint_path(topk_manager, metric_dict or {})
            if topk_path is not None:
                checkpoint_path = pathlib.Path(topk_path)

        if checkpoint_path is None and topk_manager is None:
            checkpoint_path = checkpoint_dir.joinpath(f"{step_tag}.ckpt")

        if checkpoint_path is not None:
            self.save_checkpoint(path=checkpoint_path, use_thread=False)

            # Keep resume compatibility without duplicating the checkpoint payload.
            if cfg.checkpoint.save_last_ckpt:
                latest_path = self.get_checkpoint_path()
                temporary_link = checkpoint_dir.joinpath(
                    f".latest-{os.getpid()}.tmp"
                )
                temporary_link.unlink(missing_ok=True)
                temporary_link.symlink_to(checkpoint_path.name)
                os.replace(temporary_link, latest_path)
        elif topk_manager is not None and cfg.checkpoint.save_last_ckpt:
            # If a metric-based Top-K manager rejects this checkpoint, still keep
            # a single latest checkpoint for reliable resume.
            self._save_latest_checkpoint()
        self._update_best_checkpoint(topk_manager)
        if cfg.checkpoint.save_last_snapshot:
            self.save_snapshot(tag=step_tag)

    def _validate_and_log(self, cfg, dataloader, device, wandb_run, json_logger):
        if getattr(self, "_last_validation_step", None) == self.optimizer_step:
            return self._last_validation_metrics
        if dataloader is None:
            raise ValueError("Validation-based checkpoint selection requires a validation dataloader")
        policy = self.ema_model if getattr(self, "ema_model", None) is not None else self.model
        was_training = policy.training
        policy.eval()
        try:
            # DataLoader iteration also consumes RNG state. Validation must
            # not change the subsequent training noise/augmentation sequence.
            devices = [device] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                metrics = self._validate_metrics(
                    policy, dataloader, device, cfg.training.max_val_steps,
                    cfg=cfg, distributed=getattr(self, "distributed", None),
                )
        finally:
            policy.train(was_training)
        if metrics is None:
            if self._uses_validation_checkpoint(cfg):
                raise ValueError("Validation-based checkpoint selection requires a non-empty validation split")
            return None
        self._last_validation_step = self.optimizer_step
        self._last_validation_metrics = metrics
        if self.is_main_process:
            record = dict(metrics, epoch=self.epoch, global_step=self.global_step, optimizer_step=self.optimizer_step)
            wandb_run.log(record, step=self.global_step)
            json_logger.log(record)
            plot_path = self._update_validation_plot(self.epoch, self.optimizer_step, metrics)
            if plot_path is not None:
                wandb_run.log({"val_loss_plot": wandb.Image(str(plot_path))}, step=self.global_step)
        return metrics

    def _validate_metrics(
        self,
        policy,
        dataloader,
        device,
        max_steps,
        cfg,
        distributed: Optional[DistributedContext] = None,
        compute_action_mse: bool = True,
    ) -> Optional[dict]:
        loss_sum = 0.0
        loss_count = 0
        action_mse_sum = 0.0
        action_mse_raw_sum = 0.0
        action_count = 0
        maximum = len(dataloader)
        if max_steps is not None:
            maximum = min(maximum, int(max_steps))
        distributed = distributed or DistributedContext()
        generator = None
        loss_generator = torch.Generator(device=device)
        loss_generator.manual_seed(int(cfg.training.seed) + distributed.rank + 1)
        if compute_action_mse:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(cfg.training.seed) + distributed.rank)
        with torch.no_grad(), self._autocast(cfg, device):
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= maximum:
                    break
                batch = dict_apply(
                    batch,
                    lambda value: value.to(device, non_blocking=True),
                )
                batch_size = int(batch["action"].shape[0])
                loss_sum += policy.compute_loss(batch, generator=loss_generator).item() * batch_size
                loss_count += batch_size

                if compute_action_mse:
                    # Run the complete image-conditioned diffusion sampler and
                    # compare its predicted action chunk with the demonstration.
                    # Keep both normalized and physical-unit MSEs: the former is
                    # balanced across action dimensions, while the latter is the
                    # direct regression error requested for inspecting actions.
                    prediction = policy.predict_action(
                        batch["obs"],
                        generator=generator,
                    )["model_action_pred"]
                    # Align the same future chunk returned as `action`, omitting
                    # context/latency steps. model_action_pred and the dataset
                    # also agree when relative-pose training is enabled.
                    start = policy.action_start_index
                    end = start + policy.n_action_steps
                    prediction = prediction[:, start:end].float()
                    target = batch["action"][:, start:end].float()
                    if prediction.shape != target.shape or target.shape[1] != policy.n_action_steps:
                        raise ValueError("Validation action prediction and target chunks must have matching shapes")
                    normalized_prediction = policy.normalizer["action"].normalize(
                        prediction
                    )
                    normalized_target = policy.normalizer["action"].normalize(target)
                    action_mse_sum += torch.nn.functional.mse_loss(
                        normalized_prediction,
                        normalized_target,
                        reduction="sum",
                    ).item()
                    action_mse_raw_sum += torch.nn.functional.mse_loss(
                        prediction,
                        target,
                        reduction="sum",
                    ).item()
                    action_count += int(target.numel())

        if distributed.enabled:
            totals = torch.tensor(
                [
                    loss_sum,
                    loss_count,
                    action_mse_sum,
                    action_mse_raw_sum,
                    action_count,
                ],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            loss_sum = float(totals[0].item())
            loss_count = int(totals[1].item())
            action_mse_sum = float(totals[2].item())
            action_mse_raw_sum = float(totals[3].item())
            action_count = int(totals[4].item())
        if not loss_count:
            return None
        metrics = {
            "val_loss": loss_sum / loss_count,
        }
        if compute_action_mse:
            metrics.update(
                {
                    "val_action_mse": (
                        action_mse_sum / action_count if action_count else float("nan")
                    ),
                    "val_action_mse_raw": (
                        action_mse_raw_sum / action_count if action_count else float("nan")
                    ),
                }
            )
        return metrics

    def _validate(
        self,
        policy,
        dataloader,
        device,
        max_steps,
        cfg,
        distributed: Optional[DistributedContext] = None,
    ) -> Optional[float]:
        """Backward-compatible diffusion-loss-only validation helper."""
        metrics = self._validate_metrics(
            policy=policy,
            dataloader=dataloader,
            device=device,
            max_steps=max_steps,
            cfg=cfg,
            distributed=distributed,
            compute_action_mse=False,
        )
        return None if metrics is None else metrics["val_loss"]

    def _update_validation_plot(
        self,
        epoch: int,
        optimizer_step: int,
        metrics: dict,
    ) -> Optional[pathlib.Path]:
        """Append validation metrics and save the output-dir loss figure."""
        history = getattr(self, "_validation_history", None)
        if history is None:
            history = []
            self._validation_history = history
        history.append(
            {
                "epoch": int(epoch),
                "optimizer_step": int(optimizer_step),
                "val_loss": float(metrics["val_loss"]),
                "val_action_mse": float(metrics["val_action_mse"]),
                "val_action_mse_raw": float(metrics["val_action_mse_raw"]),
            }
        )

        # Use a non-interactive backend because training may run without a
        # display (remote server, torchrun, or a container).
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            from matplotlib import pyplot as plt
        except Exception as exc:  # pragma: no cover - depends on environment
            log.warning("Could not create validation loss plot: %s", exc)
            return None

        output_path = pathlib.Path(self.output_dir).joinpath("val_loss.png")
        x = [item["optimizer_step"] for item in history]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(
            x,
            [item["val_loss"] for item in history],
            marker="o",
            label="diffusion val_loss",
        )
        axes[0].plot(
            x,
            [item["val_action_mse"] for item in history],
            marker="o",
            label="normalized action MSE",
        )
        axes[0].set_xlabel("optimizer step")
        axes[0].set_ylabel("loss")
        axes[0].set_title("Validation losses")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(
            x,
            [item["val_action_mse_raw"] for item in history],
            marker="o",
            color="tab:orange",
        )
        axes[1].set_xlabel("optimizer step")
        axes[1].set_ylabel("MSE in action units")
        axes[1].set_title("Raw validation action MSE")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name="train_force_aware_diffusion_workspace",
)
def main(cfg):
    distributed = initialize_distributed(cfg.training)
    try:
        if distributed.enabled:
            with open_dict(cfg):
                cfg.training.device = f"cuda:{distributed.local_rank}"
        workspace = TrainForceAwareDiffusionWorkspace(
            cfg,
            output_dir=shared_output_dir(distributed),
            distributed=distributed,
        )
        workspace.run()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
