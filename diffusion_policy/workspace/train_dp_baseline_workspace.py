from __future__ import annotations

import copy
import pathlib
import random

import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf, open_dict

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.workspace.train_force_aware_diffusion_workspace import (
    DistributedContext,
    TrainForceAwareDiffusionWorkspace,
    initialize_distributed,
    resolve_amp_dtype,
    shared_output_dir,
)


class TrainDPBaselineWorkspace(TrainForceAwareDiffusionWorkspace):
    """FDP-compatible trainer for a pure visual diffusion Transformer."""

    def __init__(
        self,
        cfg: OmegaConf,
        output_dir=None,
        distributed: DistributedContext | None = None,
    ) -> None:
        # Reuse the FDP workspace run loop, but avoid its force-aware policy
        # type assertion and use the baseline policy's optimizer protocol.
        BaseWorkspace.__init__(self, cfg, output_dir=output_dir)
        self.distributed = distributed or DistributedContext()
        seed = int(cfg.training.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model = hydra.utils.instantiate(cfg.policy)
        self.ema_model = copy.deepcopy(self.model) if cfg.training.use_ema else None
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


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name="train_dp_baseline",
)
def main(cfg: OmegaConf) -> None:
    distributed = initialize_distributed(cfg.training)
    try:
        if distributed.enabled:
            with open_dict(cfg):
                cfg.training.device = f"cuda:{distributed.local_rank}"
        workspace = TrainDPBaselineWorkspace(
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
