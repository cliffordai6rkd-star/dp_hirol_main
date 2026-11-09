"""Renamed dataset modules still accept targets stored in earlier checkpoints."""

import dill
import hydra
from omegaconf import OmegaConf
import torch

from diffusion_policy.common.dataset_config import V3_IMAGE_DATASET_TARGET, canonical_dataset_config
from diffusion_policy.dataset.lerobot_v3_dataset import LeRobotV3ImageDataset
from diffusion_policy.scripts.visualize_cross_attention_episode import load_checkpoint_payload
from diffusion_policy.workspace.train_distill_render_net_workspace import TrainDistillRenderNetWorkspace


LEGACY_TARGET = "diffusion_policy.dataset.legacy_lerobot_v3_dataset.LegacyLeRobotV3Dataset"


def test_current_target_resolves_and_other_dataset_types_are_preserved():
    assert hydra.utils.get_class(V3_IMAGE_DATASET_TARGET) is LeRobotV3ImageDataset
    assert hydra.utils.get_class("diffusion_policy.dataset.zarr_image_dataset.ZarrImageDataset").__name__ == "ZarrImageDataset"
    assert hydra.utils.get_class("diffusion_policy.env_runner.fr3_runner.Fr3Runner").__name__ == "Fr3Runner"
    for target in (V3_IMAGE_DATASET_TARGET, "diffusion_policy.dataset.pusht_image_dataset.PushTImageDataset"):
        config = OmegaConf.create({"_target_": target})
        assert canonical_dataset_config(config)._target_ == target


def test_checkpoint_dataset_target_is_resolved_in_memory_only(tmp_path):
    checkpoint = tmp_path / "teacher.ckpt"
    torch.save({"cfg": {"task": {"dataset": {
        "_target_": LEGACY_TARGET, "dataset_path": "/data/recordings",
        "wrench_filter": {"enabled": True, "cutoff_hz": 2.0},
    }}}}, checkpoint, pickle_module=dill)
    before = checkpoint.read_bytes()
    _, config = load_checkpoint_payload(checkpoint)
    assert config.task.dataset._target_ == V3_IMAGE_DATASET_TARGET
    assert config.task.dataset.wrench_filter.cutoff_hz == 2.0
    assert checkpoint.read_bytes() == before


def test_student_resolves_saved_dataset_target_after_merging_overrides():
    workspace = TrainDistillRenderNetWorkspace.__new__(TrainDistillRenderNetWorkspace)
    workspace.teacher_source_cfg = None
    workspace.teacher_cfg = OmegaConf.create({"task": {"dataset": {
        "_target_": LEGACY_TARGET, "wrench_filter": {"enabled": True, "cutoff_hz": 2.0},
    }}})
    workspace.cfg = OmegaConf.create({"dataset": {"val_ratio": .1}})
    config = workspace._dataset_config()
    assert config._target_ == V3_IMAGE_DATASET_TARGET
    assert config.wrench_filter.cutoff_hz == 2.0 and config.val_ratio == .1
