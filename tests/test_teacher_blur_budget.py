"""Bind percentage-based blur to the actual teacher optimizer budget."""

import pytest
from omegaconf import OmegaConf

from diffusion_policy.model.vision.contact_curriculum import FactrBlurScheduler
from diffusion_policy.workspace import train_force_aware_diffusion_workspace as workspace_module
from diffusion_policy.workspace.train_force_aware_diffusion_workspace import (
    TrainForceAwareDiffusionWorkspace,
)
from test_teacher_early_stopping_integration import (
    _EpochMarkerPolicy,
    _TinyDataset,
    _config,
)


class _BlurMarkerPolicy(_EpochMarkerPolicy):
    def __init__(self, total_steps):
        super().__init__()
        self.blur_scheduler = FactrBlurScheduler(
            total_steps=total_steps, start_fraction=0.1, end_fraction=0.8,
        )


class _NineSamples(_TinyDataset):
    def __len__(self):
        return 9


class _TrainingReached(Exception):
    pass


def _prepare(monkeypatch, *, total_steps=None):
    instantiate = workspace_module.hydra.utils.instantiate

    def instantiate_small(config, *args, **kwargs):
        if config.get("_target_") == "integration_test_policy":
            return _BlurMarkerPolicy(total_steps)
        if config.get("_target_") == "integration_test_dataset":
            return _NineSamples()
        return instantiate(config, *args, **kwargs)

    monkeypatch.setattr(workspace_module.hydra.utils, "instantiate", instantiate_small)
    # Inspect the real initialization path immediately before any training.
    def stop_at_training(self, **kwargs):
        raise _TrainingReached

    monkeypatch.setattr(TrainForceAwareDiffusionWorkspace, "_train_epoch", stop_at_training)
    cfg = _config(num_epochs=10)
    cfg.training.use_ema = True
    cfg.training.gradient_accumulate_every = 2
    cfg.ema = OmegaConf.create({
        "_target_": "diffusion_policy.model.diffusion.ema_model.EMAModel",
    })
    return cfg


@pytest.mark.parametrize("mode, expected", [
    ("epochs", 30),  # ceil(ceil(9 / batch_size=2) / accumulation=2) * 10 epochs
    ("limited_batches", 20),
    ("optimizer_steps", 37),
    ("fixed_schedule", 100),
])
def test_training_initializes_online_and_ema_blur_budgets(monkeypatch, tmp_path, mode, expected):
    cfg = _prepare(monkeypatch, total_steps=100 if mode == "fixed_schedule" else None)
    if mode == "limited_batches":
        cfg.training.max_train_steps = 3
    elif mode == "optimizer_steps":
        cfg.training.num_epochs = None
        cfg.training.num_optimizer_steps = 37
    workspace = TrainForceAwareDiffusionWorkspace(cfg, output_dir=str(tmp_path))

    with pytest.raises(_TrainingReached):
        workspace.run()

    for policy in (workspace.model, workspace.ema_model):
        assert policy.blur_scheduler.total_steps == expected
        assert policy.optimizer_step == 0
        assert policy.blur_scheduler.sigma(0) == 5.0
        assert policy.blur_scheduler.sigma(expected) == 0.0


def test_resume_rebinds_budget_and_restores_online_and_ema_progress(monkeypatch, tmp_path):
    cfg = _prepare(monkeypatch)
    original = TrainForceAwareDiffusionWorkspace(cfg, output_dir=str(tmp_path))
    with pytest.raises(_TrainingReached):
        original.run()
    original.optimizer_step = 7
    original.global_step = 14
    original.epoch = 2
    original.save_checkpoint(use_thread=False)

    cfg.training.resume = True
    resumed = TrainForceAwareDiffusionWorkspace(cfg, output_dir=str(tmp_path))
    with pytest.raises(_TrainingReached):
        resumed.run()

    assert resumed.optimizer_step == 7
    for policy in (resumed.model, resumed.ema_model):
        assert policy.blur_scheduler.total_steps == 30
        assert policy.optimizer_step == 7
        assert policy.blur_scheduler.sigma(policy.optimizer_step) == pytest.approx(
            5.0 * (0.8 - 7 / 30) / 0.7,
        )
