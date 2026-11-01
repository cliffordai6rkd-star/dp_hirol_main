"""Train the LIBERO wrench-only force-aware diffusion policy.

Examples:
    python sim_force_aware_dp_train.py
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
        sim_force_aware_dp_train.py
"""

import logging
import pathlib
import sys

import hydra
import torch.distributed as dist
from omegaconf import OmegaConf, open_dict

from diffusion_policy.common.config_cli import rewrite_config_reference_argv
from diffusion_policy.workspace.train_force_aware_diffusion_workspace import (
    initialize_distributed,
    shared_output_dir,
)
from diffusion_policy.workspace.train_force_aware_diffusion_workspace import (
    TrainForceAwareDiffusionWorkspace,
)


CONFIG_DIR = pathlib.Path(__file__).resolve().parent / "diffusion_policy" / "config"
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="sim_force_aware_dp_train")
def main(cfg):
    OmegaConf.resolve(cfg)
    distributed = initialize_distributed(cfg.training)
    try:
        if distributed.enabled:
            with open_dict(cfg):
                cfg.training.device = f"cuda:{distributed.local_rank}"
            if not distributed.is_main:
                for handler in list(logging.getLogger().handlers):
                    if isinstance(handler, logging.FileHandler):
                        logging.getLogger().removeHandler(handler)
                        handler.close()
        TrainForceAwareDiffusionWorkspace(
            cfg, output_dir=shared_output_dir(distributed), distributed=distributed,
        ).run()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    sys.argv = rewrite_config_reference_argv(sys.argv, default_config_dir=str(CONFIG_DIR))
    main()
