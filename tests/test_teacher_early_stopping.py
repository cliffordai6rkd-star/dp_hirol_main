import copy

import dill
import pytest
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace.train_force_aware_diffusion_workspace import (
    TrainForceAwareDiffusionWorkspace,
)


def config(**options):
    return OmegaConf.create({
        "training": {
            "val_epoch_every": 5,
            "early_stopping": {
                "enabled": True, "monitor": "val_loss", "patience": 100,
                "min_delta": 0.0, **options,
            },
        },
    })


def workspace():
    result = TrainForceAwareDiffusionWorkspace.__new__(TrainForceAwareDiffusionWorkspace)
    result.early_stopping_state = {}
    return result


def test_validation_is_due_at_completed_intervals_and_final_epoch_only():
    cfg = config()
    due = TrainForceAwareDiffusionWorkspace._epoch_validation_due
    assert [epoch for epoch in range(13) if due(cfg, epoch)] == [5, 10]
    assert due(cfg, 12, final=True)
    assert not due(cfg, 0, final=True)
    assert TrainForceAwareDiffusionWorkspace._validation_epoch_interval(
        OmegaConf.create({"training": {"val_every": 5}})
    ) is None


def test_patience_counts_completed_epochs_not_validation_calls():
    state = workspace()
    cfg = config()
    state._update_early_stopping(cfg, {"val_loss": 1.0}, 5)
    for epoch in range(10, 101, 5):
        state._update_early_stopping(cfg, {"val_loss": 1.1}, epoch)
        assert not state.early_stopping_state["should_stop"]
    metrics = state._update_early_stopping(cfg, {"val_loss": 1.1}, 105)
    assert metrics["early_stopping/stale_epochs"] == 100
    assert state.early_stopping_state["should_stop"]
    assert state.early_stopping_state["best_epoch"] == 5


def test_improvement_resets_elapsed_epochs_and_duplicate_validation_is_ignored():
    state = workspace()
    cfg = config(min_delta=0.1)
    state._update_early_stopping(cfg, {"val_loss": 1.0}, 5)
    state._update_early_stopping(cfg, {"val_loss": 0.95}, 10)
    assert state.early_stopping_state["best_epoch"] == 5
    state._update_early_stopping(cfg, {"val_loss": 0.8}, 20)
    saved = copy.deepcopy(state.early_stopping_state)
    state._update_early_stopping(cfg, {"val_loss": 0.1}, 20)
    state._update_early_stopping(cfg, {"val_loss": 0.1}, 15)
    assert state.early_stopping_state == saved
    state._update_early_stopping(cfg, {"val_loss": 0.9}, 115)
    assert not state.early_stopping_state["should_stop"]
    state._update_early_stopping(cfg, {"val_loss": 0.9}, 120)
    assert state.early_stopping_state["should_stop"]


def test_nonfinite_validation_is_not_an_improvement():
    state = workspace()
    cfg = config()
    state._update_early_stopping(cfg, {"val_loss": float("nan")}, 5)
    assert state.early_stopping_state["best_value"] is None
    assert state.early_stopping_state["stale_epochs"] == 0
    state._update_early_stopping(cfg, {"val_loss": 1.0}, 10)
    state._update_early_stopping(cfg, {"val_loss": float("inf")}, 110)
    assert state.early_stopping_state["best_value"] == 1.0
    assert state.early_stopping_state["best_epoch"] == 10
    assert state.early_stopping_state["should_stop"]


def test_early_stopping_state_roundtrips_through_real_checkpoint(tmp_path):
    original = workspace()
    original.cfg = config()
    original._output_dir = str(tmp_path)
    original._saving_thread = None
    original.epoch = 95
    original._update_early_stopping(original.cfg, {"val_loss": 1.0}, 5)
    original._update_early_stopping(original.cfg, {"val_loss": 1.2}, 95)
    path = original.save_checkpoint(use_thread=False)
    payload = torch.load(path, map_location="cpu", pickle_module=dill)
    assert "early_stopping_state" in payload["pickles"]
    resumed = workspace()
    resumed.load_payload(payload)
    assert resumed.epoch == 95
    assert resumed.early_stopping_state == original.early_stopping_state
    resumed._update_early_stopping(original.cfg, {"val_loss": 1.2}, 100)
    assert not resumed.early_stopping_state["should_stop"]
    resumed._update_early_stopping(original.cfg, {"val_loss": 1.2}, 105)
    assert resumed.early_stopping_state["should_stop"]


def test_disabled_early_stopping_does_not_change_state():
    state = workspace()
    assert state._update_early_stopping(config(enabled=False), {}, 0) == {}
    assert state.early_stopping_state == {}


@pytest.mark.parametrize("options", [
    {"patience": 0}, {"min_delta": -1}, {"min_delta": float("nan")},
    {"monitor": "train_loss"},
])
def test_invalid_early_stopping_settings_fail_before_training(options):
    with pytest.raises(ValueError):
        TrainForceAwareDiffusionWorkspace._early_stopping_options(config(**options))


def test_missing_monitor_and_invalid_epoch_interval_fail_clearly():
    with pytest.raises(ValueError, match="missing from validation"):
        workspace()._update_early_stopping(config(), {"val_action_mse": 1.0}, 5)
    cfg = config()
    cfg.training.val_epoch_every = 0
    with pytest.raises(ValueError, match="val_epoch_every must be positive"):
        TrainForceAwareDiffusionWorkspace._validation_epoch_interval(cfg)
