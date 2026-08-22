from __future__ import annotations

from typing import Dict, Optional, Union

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class OfflineValidationRunner(BaseImageRunner):
    """Evaluate predicted action chunks on the held-out dataset split.

    This is an offline evaluator rather than a closed-loop environment runner:
    observations come from validation demonstrations and are not replaced by
    states generated from the policy's predicted actions.
    """

    def __init__(
        self,
        output_dir,
        dataset_cfg: Union[DictConfig, BaseImageDataset],
        batch_size: int = 8,
        num_workers: int = 2,
        max_steps: Optional[int] = None,
        sampling_passes: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__(output_dir)
        if sampling_passes < 1:
            raise ValueError("sampling_passes must be positive")

        if isinstance(dataset_cfg, BaseImageDataset):
            dataset = dataset_cfg
        else:
            dataset = hydra.utils.instantiate(dataset_cfg)
        self.validation_dataset = dataset.get_validation_dataset()
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.max_steps = max_steps
        self.sampling_passes = int(sampling_passes)
        self.seed = int(seed)

    def run(self, policy) -> Dict[str, float]:
        loader_cfg = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "shuffle": False,
            "pin_memory": policy.device.type == "cuda",
        }
        if self.num_workers > 0:
            loader_cfg["persistent_workers"] = False
        dataloader = DataLoader(self.validation_dataset, **loader_cfg)

        device = policy.device
        total_loss = 0.0
        total_count = 0

        policy.eval()
        with torch.no_grad():
            for pass_idx in range(self.sampling_passes):
                generator = torch.Generator(device=device)
                generator.manual_seed(self.seed + pass_idx)
                for batch_idx, batch in enumerate(dataloader):
                    if self.max_steps is not None and batch_idx >= int(self.max_steps):
                        break

                    batch = dict_apply(
                        batch,
                        lambda value: value.to(device, non_blocking=True),
                    )
                    result = policy.predict_action(
                        batch["obs"],
                        generator=generator,
                    )
                    pred_action = result["model_action_pred"]
                    target_action = batch["action"]

                    # Compare in the policy's normalized action space so joint
                    # dimensions with different physical scales are balanced.
                    pred_normalized = policy.normalizer["action"].normalize(
                        pred_action
                    )
                    target_normalized = policy.normalizer["action"].normalize(
                        target_action
                    )
                    total_loss += F.mse_loss(
                        pred_normalized,
                        target_normalized,
                        reduction="sum",
                    ).item()
                    total_count += target_normalized.numel()

        if total_count == 0:
            raise RuntimeError("validation dataset produced no samples")
        return {
            "val_action_mse": float(total_loss / total_count / self.sampling_passes),
        }
