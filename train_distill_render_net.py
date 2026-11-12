"""Distill a frozen image-query policy's behavior into a wrench-conditioned renderer."""

import pathlib
import logging
import sys

import hydra
import torch.distributed as dist
from omegaconf import OmegaConf, open_dict

from diffusion_policy.common.config_cli import rewrite_config_reference_argv
from diffusion_policy.workspace.train_distill_render_net_workspace import TrainDistillRenderNetWorkspace
from diffusion_policy.workspace.train_force_aware_diffusion_workspace import (
    initialize_distributed, shared_output_dir,
)


CONFIG_DIR = pathlib.Path(__file__).resolve().parent / "diffusion_policy" / "config"
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="train_distill_render_net_student")
def main(cfg):
    OmegaConf.resolve(cfg)
    distributed = initialize_distributed(cfg.training)
    try:
        if distributed.enabled:
            with open_dict(cfg):
                cfg.training.device = f"cuda:{distributed.local_rank}"
            if not distributed.is_main:
                # Hydra configures logging before DDP is initialized. Keep the
                # shared training log exclusively owned by rank 0.
                for handler in list(logging.getLogger().handlers):
                    if isinstance(handler, logging.FileHandler):
                        logging.getLogger().removeHandler(handler)
                        handler.close()
        TrainDistillRenderNetWorkspace(
            cfg, output_dir=shared_output_dir(distributed), distributed=distributed,
        ).run()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    sys.argv = rewrite_config_reference_argv(sys.argv, default_config_dir=str(CONFIG_DIR))
    main()
