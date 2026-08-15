if __name__ == "__main__":
    import os
    import pathlib
    import sys

    root_dir = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(root_dir)
    os.chdir(root_dir)

import copy
import math
import os
import pathlib
import random
from typing import Optional

import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

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

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        super().__init__(cfg, output_dir=output_dir)
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
        self.global_step = 0
        self.optimizer_step = 0
        self.epoch = 0

    def run(self) -> None:
        cfg = copy.deepcopy(self.cfg)
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
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        val_dataloader = DataLoader(dataset.get_validation_dataset(), **cfg.val_dataloader)

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
        if cfg.task.get("env_runner") is not None:
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir,
            )
            if not isinstance(env_runner, BaseImageRunner):
                raise TypeError("cfg.task.env_runner must instantiate BaseImageRunner")

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        self.update_wandb_output_dir()
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
        if checkpoint_every_optimizer_steps is None:
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, "checkpoints"),
                **cfg.checkpoint.topk,
            )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        self.optimizer.zero_grad(set_to_none=True)

        sample_batch = None
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            def training_is_complete() -> bool:
                if num_epochs is not None:
                    return self.epoch >= num_epochs
                return self.optimizer_step >= num_optimizer_steps

            while not training_is_complete():
                step_log = self._train_epoch(
                    cfg=cfg,
                    dataloader=train_dataloader,
                    device=device,
                    lr_scheduler=lr_scheduler,
                    ema=ema,
                    wandb_run=wandb_run,
                    json_logger=json_logger,
                    topk_manager=topk_manager,
                    target_optimizer_steps=num_optimizer_steps,
                )
                if sample_batch is None:
                    sample_batch = next(iter(train_dataloader))

                policy = self.ema_model if self.ema_model is not None else self.model
                policy.eval()
                if env_runner is not None and self.epoch % int(cfg.training.rollout_every) == 0:
                    step_log.update(env_runner.run(policy))

                if self.epoch % int(cfg.training.val_every) == 0:
                    val_loss = self._validate(
                        policy,
                        val_dataloader,
                        device,
                        cfg.training.max_val_steps,
                    )
                    if val_loss is not None:
                        step_log["val_loss"] = val_loss

                if self.epoch % int(cfg.training.sample_every) == 0:
                    batch = dict_apply(
                        sample_batch,
                        lambda value: value.to(device, non_blocking=True),
                    )
                    with torch.no_grad():
                        prediction = policy.predict_action(batch["obs"])[
                            "model_action_pred"
                        ]
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
                    if use_optimizer_step_checkpoints:
                        self._save_optimizer_step_checkpoint(cfg)
                    else:
                        self._save_checkpoints(cfg, step_log, topk_manager)

                policy.train()

        if self._saving_thread is not None:
            self._saving_thread.join()
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
    ) -> dict:
        self.model.train()
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
                for _ in range(group_size):
                    batch = next(batch_iterator)
                    batch = dict_apply(
                        batch,
                        lambda value: value.to(device, non_blocking=True),
                    )
                    raw_loss = self.model.compute_loss(
                        batch,
                        optimizer_step=self.optimizer_step,
                    )
                    (raw_loss / group_size).backward()
                    loss_value = raw_loss.item()
                    losses.append(loss_value)
                    progress.set_postfix(loss=loss_value, refresh=False)
                    self.global_step += 1
                    batch_idx += 1

                if cfg.training.get("max_grad_norm") is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        float(cfg.training.max_grad_norm),
                    )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                self.optimizer_step += 1
                self.model.set_optimizer_step(self.optimizer_step)
                if ema is not None:
                    ema.step(self.model)
                    self.ema_model.set_optimizer_step(self.optimizer_step)

                last_log = {
                    "train_loss": float(np.mean(losses[-group_size:])),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": self.epoch,
                    "global_step": self.global_step,
                    "optimizer_step": self.optimizer_step,
                    **self.model.last_curriculum_metrics,
                }
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
                        self._save_optimizer_step_checkpoint(cfg)

        last_log["train_loss"] = float(np.mean(losses))
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

    @staticmethod
    def _validate(policy, dataloader, device, max_steps) -> Optional[float]:
        losses = []
        maximum = len(dataloader)
        if max_steps is not None:
            maximum = min(maximum, int(max_steps))
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= maximum:
                    break
                batch = dict_apply(
                    batch,
                    lambda value: value.to(device, non_blocking=True),
                )
                losses.append(policy.compute_loss(batch).item())
        return float(np.mean(losses)) if losses else None


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name="train_force_aware_diffusion_workspace",
)
def main(cfg):
    workspace = TrainForceAwareDiffusionWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
