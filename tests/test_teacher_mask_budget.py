"""Fractional token masking follows the actual teacher optimizer budget."""

import pytest
from omegaconf import OmegaConf

from diffusion_policy.model.vision.contact_curriculum import MaskProbabilityScheduler
from diffusion_policy.workspace import train_force_aware_diffusion_workspace as workspace_module
from diffusion_policy.workspace.train_force_aware_diffusion_workspace import TrainForceAwareDiffusionWorkspace
from test_teacher_blur_budget import _NineSamples, _TrainingReached
from test_teacher_early_stopping_integration import _EpochMarkerPolicy, _config


class _MaskMarkerPolicy(_EpochMarkerPolicy):
    def __init__(self, total_steps=None):
        super().__init__()
        self.mask_scheduler = MaskProbabilityScheduler(
            schedule_type="piecewise", total_steps=total_steps,
            piecewise_fractions=[0., .25, .75, 1.],
            piecewise_probabilities=[1., 1., 0., 0.],
        )


def prepare(monkeypatch, *, total_steps=None, sample_count=9):
    instantiate = workspace_module.hydra.utils.instantiate

    class Dataset(_NineSamples):
        def __len__(self):
            return sample_count

    def instantiate_small(config, *args, **kwargs):
        if config.get("_target_") == "integration_test_policy":
            return _MaskMarkerPolicy(total_steps)
        if config.get("_target_") == "integration_test_dataset":
            return Dataset()
        return instantiate(config, *args, **kwargs)

    def stop_at_training(self, **kwargs):
        raise _TrainingReached

    monkeypatch.setattr(workspace_module.hydra.utils, "instantiate", instantiate_small)
    monkeypatch.setattr(TrainForceAwareDiffusionWorkspace, "_train_epoch", stop_at_training)
    cfg = _config(num_epochs=10)
    cfg.training.use_ema = True
    cfg.training.gradient_accumulate_every = 2
    cfg.ema = OmegaConf.create({"_target_": "diffusion_policy.model.diffusion.ema_model.EMAModel"})
    return cfg


@pytest.mark.parametrize("mode, expected", [
    ("epochs", 30), ("limited_batches", 20), ("steps150k", 150000),
    ("epochs600", 367200), ("fixed", 100),
])
def test_online_and_ema_mask_curricula_bind_actual_training_budget(monkeypatch, tmp_path, mode, expected):
    cfg = prepare(monkeypatch, total_steps=100 if mode == "fixed" else None,
                  sample_count=19571 if mode == "epochs600" else 9)
    if mode == "limited_batches":
        cfg.training.max_train_steps = 3
    elif mode == "steps150k":
        cfg.training.num_epochs = None
        cfg.training.num_optimizer_steps = 150000
    elif mode == "epochs600":
        cfg.training.num_epochs = 600
        cfg.dataloader.batch_size = 32
        cfg.training.gradient_accumulate_every = 1
    workspace = TrainForceAwareDiffusionWorkspace(cfg, output_dir=str(tmp_path))
    with pytest.raises(_TrainingReached):
        workspace.run()
    for policy in (workspace.model, workspace.ema_model):
        schedule = policy.mask_scheduler
        assert schedule.total_steps == expected
        assert schedule.probability(0) == 1.
        assert schedule.probability(round(expected * .25)) == 1.
        assert schedule.probability(round(expected * .75)) == 0.
        assert schedule.probability(expected) == 0.


def test_resume_rebinds_mask_budget_and_restores_progress(monkeypatch, tmp_path):
    cfg = prepare(monkeypatch)
    original = TrainForceAwareDiffusionWorkspace(cfg, output_dir=str(tmp_path))
    with pytest.raises(_TrainingReached):
        original.run()
    original.optimizer_step, original.global_step, original.epoch = 15, 30, 5
    original.save_checkpoint(use_thread=False)
    cfg.training.resume = True
    resumed = TrainForceAwareDiffusionWorkspace(cfg, output_dir=str(tmp_path))
    with pytest.raises(_TrainingReached):
        resumed.run()
    for policy in (resumed.model, resumed.ema_model):
        assert policy.optimizer_step == 15
        assert policy.mask_scheduler.total_steps == 30
        assert policy.mask_scheduler.probability(15) == .5
