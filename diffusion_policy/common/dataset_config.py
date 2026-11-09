"""Resolve saved dataset configurations to the current public class names."""

from omegaconf import open_dict


V3_IMAGE_DATASET_TARGET = "diffusion_policy.dataset.lerobot_v3_dataset.LeRobotV3ImageDataset"


def canonical_dataset_config(config):
    """Update the previous v3 class suffix without retaining deprecated modules.

    Checkpoints store the dataset target as configuration text. Resolve that
    target in memory so existing weights and their fitted normalizer remain
    usable after the dataset module rename; checkpoint files are not rewritten.
    """
    target = str(config.get("_target_", ""))
    if target.startswith("diffusion_policy.dataset.") and target.endswith("LeRobotV3Dataset"):
        with open_dict(config):
            config._target_ = V3_IMAGE_DATASET_TARGET
    return config
