"""Validation-driven learning rates survive student checkpoint continuation."""

import json

import dill
import pytest
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace import train_distill_render_net_workspace as distill
from test_render_net_ddp import _config, TinyDataset, TinyPreprocessor, TinyRenderer, TinyTeacher


@pytest.fixture
def make_workspace(monkeypatch, tmp_path):
    teacher_config = OmegaConf.create({"task": {"dataset": {}}})
    monkeypatch.setattr(distill, "load_checkpoint_payload", lambda _: ({"state_dicts": {}}, teacher_config))
    monkeypatch.setattr(distill, "instantiate_policy", lambda *args, **kwargs: TinyTeacher())
    monkeypatch.setattr(distill, "build_render_net", lambda *args: TinyRenderer())
    monkeypatch.setattr(distill.WrenchPreprocessor, "from_policy", lambda _: TinyPreprocessor())
    monkeypatch.setattr(distill.RenderNetTrainingLogger, "save_plot", lambda self: None)

    def make(name, *, epochs, losses=(), resume=None, enabled=True, validation=True,
             learning_rate=3e-4):
        dataset = TinyDataset(tmp_path / "data")
        if not validation:
            dataset.get_validation_dataset = lambda: []
        monkeypatch.setattr(distill.hydra.utils, "instantiate", lambda _: dataset)
        config = _config(tmp_path)
        config.teacher_cache.enabled = False
        config.optimizer.lr = learning_rate
        config.training.num_epochs = epochs
        config.training.early_stopping.enabled = False
        if enabled == "missing":
            config.training.pop("lr_scheduler", None)
        else:
            config.training.lr_scheduler = {
                "enabled": enabled, "factor": 0.5, "patience": 2,
                "threshold": 1e-4, "min_lr": 1e-5,
            }
        config.training.resume_from = str(resume) if resume is not None else None
        workspace = distill.TrainDistillRenderNetWorkspace(config, str(tmp_path / name))
        validation_losses = iter(losses)
        monkeypatch.setattr(workspace, "_validate", lambda loader: next(validation_losses))
        return workspace

    return make


def checkpoint_payload(workspace, tag="latest"):
    return torch.load(workspace.get_checkpoint_path(tag), pickle_module=dill)


def training_rates(workspace):
    with open(f"{workspace.output_dir}/logs.json.txt") as source:
        records = [json.loads(line) for line in source]
    return {record["epoch"]: record["lr"] for record in records if "train_mse" in record}


def test_plateau_reduces_after_third_bad_validation_and_respects_minimum(make_workspace):
    # These tiny improvements must not reset an absolute 1e-4 threshold.
    workspace = make_workspace("plateau", epochs=9, losses=[1.0] + [0.99995] * 9,
                               learning_rate=4e-5)
    workspace.run()
    rates = training_rates(workspace)
    assert [rates[epoch] for epoch in (1, 2, 3)] == pytest.approx([4e-5] * 3)
    assert [rates[epoch] for epoch in (4, 5, 6)] == pytest.approx([2e-5] * 3)
    assert [rates[epoch] for epoch in (7, 8, 9)] == pytest.approx([1e-5] * 3)
    assert workspace.optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)
    payload = checkpoint_payload(workspace)
    assert payload["state_dicts"]["optimizer"]["param_groups"][0]["lr"] == pytest.approx(1e-5)
    assert payload["state_dicts"]["lr_scheduler"]["best"] == pytest.approx(1.0)
    # The best checkpoint is saved after consuming that validation measurement.
    assert checkpoint_payload(workspace, "best")["state_dicts"]["lr_scheduler"]["num_bad_epochs"] == 1


def test_resume_preserves_pending_plateau_reduction(make_workspace):
    first = make_workspace("first", epochs=2, losses=[1.0, 1.1, 1.1])
    first.run()
    assert first.lr_scheduler.num_bad_epochs == 2
    resumed = make_workspace("resumed", epochs=3, losses=[1.1],
                             resume=first.get_checkpoint_path("latest"))
    assert resumed.lr_scheduler.num_bad_epochs == 2
    resumed.run()
    assert resumed.epoch == 3 and resumed.global_step == 6
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(1.5e-4)
    assert checkpoint_payload(resumed)["state_dicts"]["lr_scheduler"]["num_bad_epochs"] == 0


def test_old_checkpoint_seeds_best_metric_without_inventing_plateau_history(make_workspace):
    old = make_workspace("old", epochs=2, losses=[1.0, 1.1, 1.1], enabled="missing")
    old.run()
    assert "lr_scheduler" not in checkpoint_payload(old)["state_dicts"]
    resumed = make_workspace("upgraded", epochs=5, losses=[1.1] * 3,
                             resume=old.get_checkpoint_path("latest"))
    assert resumed.lr_scheduler.best == pytest.approx(1.0)
    assert resumed.lr_scheduler.num_bad_epochs == 0
    resumed.run()
    assert list(training_rates(resumed).values()) == pytest.approx([3e-4] * 3)
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(1.5e-4)


def test_scheduler_can_be_disabled_when_resuming_checkpoint_with_scheduler(make_workspace):
    first = make_workspace("scheduled", epochs=2, losses=[1.0, 1.1, 1.1])
    first.run()
    resumed = make_workspace("disabled_resume", epochs=4, losses=[1.1, 1.1],
                             resume=first.get_checkpoint_path("latest"), enabled=False)
    resumed.run()
    assert resumed.lr_scheduler is None
    assert list(training_rates(resumed).values()) == pytest.approx([3e-4] * 2)
    assert "lr_scheduler" not in checkpoint_payload(resumed)["state_dicts"]


@pytest.mark.parametrize("enabled,validation", [(False, True), ("missing", True), (True, False)])
def test_disabled_legacy_and_no_validation_runs_keep_learning_rate(make_workspace, enabled, validation):
    workspace = make_workspace("constant", epochs=4, losses=[1.0] + [1.1] * 4,
                               enabled=enabled, validation=validation)
    workspace.run()
    assert list(training_rates(workspace).values()) == pytest.approx([3e-4] * 4)
    assert workspace.optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
