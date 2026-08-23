if __name__ == "__main__":
    import os
    import pathlib
    import sys

    root_dir = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(root_dir)
    os.chdir(root_dir)

import copy
import contextlib
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
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.policy.force_aware_diffusion_transformer_policy import (
    ForceAwareDiffusionTransformerPolicy,
)
from diffusion_policy.workspace.base_workspace import BaseWorkspace


OmegaConf.register_new_resolver("eval", eval, replace=True)


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
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def shared_output_dir(distributed: DistributedContext) -> str:
    output_dir = str(HydraConfig.get().runtime.output_dir)
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
    include_keys = ("global_step", "optimizer_step", "epoch")

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
        if distributed.enabled:
            train_sampler = DistributedSampler(
                dataset,
                num_replicas=distributed.world_size,
                rank=distributed.rank,
                shuffle=bool(train_loader_cfg.pop("shuffle", True)),
                seed=int(cfg.training.seed),
            )
            val_sampler = DistributedSampler(
                validation_dataset,
                num_replicas=distributed.world_size,
                rank=distributed.rank,
                shuffle=False,
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
        if self.is_main_process and cfg.task.get("env_runner") is not None:
            env_runner_cfg = cfg.task.env_runner
            env_runner_kwargs = {"output_dir": self.output_dir}
            # OfflineValidationRunner declares dataset_cfg and can share the
            # already-preloaded training dataset instead of decoding videos
            # again during runner construction.
            if env_runner_cfg.get("dataset_cfg") is not None:
                env_runner_kwargs["dataset"] = dataset
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
        if checkpoint_every_optimizer_steps is None and self.is_main_process:
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, "checkpoints"),
                **cfg.checkpoint.topk,
            )

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
                    val_loss = self._validate(
                        policy,
                        val_dataloader,
                        device,
                        cfg.training.max_val_steps,
                        cfg=cfg,
                        distributed=distributed,
                    )
                    if val_loss is not None:
                        step_log["val_loss"] = val_loss

                if (
                    self.is_main_process
                    and self.epoch % int(cfg.training.sample_every) == 0
                ):
                    batch = dict_apply(
                        sample_batch,
                        lambda value: value.to(device, non_blocking=True),
                    )
                    with torch.no_grad(), self._autocast(cfg, device):
                        prediction = policy.predict_action(batch["obs"])["model_action_pred"]
                        step_log["train_action_mse_error"] = torch.nn.functional.mse_loss(
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
                if should_checkpoint and self.is_main_process:
                    if use_optimizer_step_checkpoints:
                        self._save_optimizer_step_checkpoint(cfg)
                    else:
                        self._save_checkpoints(cfg, step_log, topk_manager)

                policy.train()
                if distributed.enabled:
                    dist.barrier()

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
    ) -> dict:
        self.model.train()
        if training_model is None:
            training_model = self.model
        maximum = len(dataloader)
        if cfg.training.max_train_steps is not None:
            maximum = min(maximum, int(cfg.training.max_train_steps))
        losses = []
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
                    loss_value = raw_loss.item()
                    losses.append(loss_value)
                    group_losses.append(loss_value)
                    progress.set_postfix(loss=loss_value, refresh=False)
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

                last_log = {
                    "train_loss": self._distributed_mean(
                        float(np.mean(group_losses)),
                        device,
                    ),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": self.epoch,
                    "global_step": self.global_step,
                    "optimizer_step": self.optimizer_step,
                    **self.model.last_curriculum_metrics,
                }
                if self.is_main_process:
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
                    if (
                        self.optimizer_step % checkpoint_interval == 0
                        and self.is_main_process
                    ):
                        self._save_optimizer_step_checkpoint(cfg)

        last_log["train_loss"] = self._distributed_mean(
            float(np.mean(losses)),
            device,
        )
        return last_log

    def _save_checkpoints(self, cfg, step_log, topk_manager) -> None:
        if cfg.checkpoint.save_last_ckpt:
            self.save_checkpoint(use_thread=False)
        if cfg.checkpoint.save_last_snapshot:
            self.save_snapshot()
        if topk_manager is None:
            return

        monitor_key = str(cfg.checkpoint.topk.monitor_key)
        metric_dict = {
            key.replace("/", "_"): value for key, value in step_log.items()
        }
        if monitor_key not in metric_dict:
            return
        topk_path = topk_manager.get_ckpt_path(metric_dict)
        if topk_path is not None:
            self.save_checkpoint(path=topk_path, use_thread=False)

    def _save_optimizer_step_checkpoint(self, cfg) -> None:
        step_tag = f"optimizer_step={self.optimizer_step:08d}"
        if cfg.checkpoint.save_last_ckpt:
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath("checkpoints")
            checkpoint_path = checkpoint_dir.joinpath(f"{step_tag}.ckpt")
            self.save_checkpoint(path=checkpoint_path, use_thread=False)

            # Keep resume compatibility without duplicating the checkpoint payload.
            latest_path = self.get_checkpoint_path()
            temporary_link = checkpoint_dir.joinpath(
                f".latest-{os.getpid()}.tmp"
            )
            temporary_link.unlink(missing_ok=True)
            temporary_link.symlink_to(checkpoint_path.name)
            os.replace(temporary_link, latest_path)
        if cfg.checkpoint.save_last_snapshot:
            self.save_snapshot(tag=step_tag)

    def _validate(
        self,
        policy,
        dataloader,
        device,
        max_steps,
        cfg,
        distributed: Optional[DistributedContext] = None,
    ) -> Optional[float]:
        loss_sum = 0.0
        loss_count = 0
        maximum = len(dataloader)
        if max_steps is not None:
            maximum = min(maximum, int(max_steps))
        with torch.no_grad(), self._autocast(cfg, device):
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= maximum:
                    break
                batch = dict_apply(
                    batch,
                    lambda value: value.to(device, non_blocking=True),
                )
                loss_sum += policy.compute_loss(batch).item()
                loss_count += 1

        distributed = distributed or DistributedContext()
        if distributed.enabled:
            totals = torch.tensor(
                [loss_sum, loss_count],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            loss_sum = float(totals[0].item())
            loss_count = int(totals[1].item())
        return loss_sum / loss_count if loss_count else None


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
