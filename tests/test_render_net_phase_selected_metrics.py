"""Selected-phase metrics remain invariant to batch composition and resume."""

import hashlib
import json
import math

import dill
import pytest
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace import train_distill_render_net_workspace as distill
from test_render_net_ddp import _config, TinyDataset, TinyPreprocessor, TinyRenderer, TinyTeacher


class PhaseDataset(TinyDataset):
    phases = torch.tensor([[1, 0], [0, 0], [1, 2], [2, 0], [0, 0], [0, 0]])
    token_errors = torch.tensor([[2., 999.], [999., 999.], [4., 6.],
                                 [8., 999.], [999., 999.], [999., 999.]])
    region_errors = torch.tensor([[1., 999.], [999., 999.], [8., 10.],
                                  [9., 999.], [999., 999.], [999., 999.]])

    def __init__(self, root):
        super().__init__(root, range(6))

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["obs"].update({
            "contact_phase": self.phases[index].clone(),
            "token_error": self.token_errors[index].clone(),
            "region_error": self.region_errors[index].clone(),
            "condition_ce": torch.tensor(float([1, 3, 3, 5, 5, 7][index])),
            "alpha_error": torch.tensor(float([0, 2, 2, 4, 4, 6][index])),
        })
        return sample

    def get_validation_dataset(self):
        return PhaseDataset(self.dataset_path)


@pytest.fixture
def make_workspace(monkeypatch, tmp_path):
    teacher_cfg = OmegaConf.create({"task": {"dataset": {}}})
    monkeypatch.setattr(distill, "load_checkpoint_payload", lambda _: ({"state_dicts": {}}, teacher_cfg))
    monkeypatch.setattr(distill, "instantiate_policy", lambda *args, **kwargs: TinyTeacher())
    monkeypatch.setattr(distill, "build_render_net", lambda *args: TinyRenderer())
    monkeypatch.setattr(distill.WrenchPreprocessor, "from_policy", lambda _: TinyPreprocessor())
    monkeypatch.setattr(distill.hydra.utils, "instantiate", lambda _: PhaseDataset(tmp_path / "data"))
    monkeypatch.setattr(distill.RenderNetTrainingLogger, "save_plot", lambda self: None)

    def make(name="metrics", *, selected=True, resume=None, epochs=1):
        cfg = _config(tmp_path)
        cfg.teacher_cache.enabled = False
        cfg.training.num_epochs = epochs
        cfg.training.early_stopping.enabled = False
        cfg.training.use_ema = False
        cfg.training.lr_scheduler = {
            "enabled": True, "factor": .5, "patience": 2,
            "threshold": 1e-4, "min_lr": 1e-5,
        }
        cfg.distillation.token_region_phases = [1, 2] if selected else None
        cfg.training.resume_from = str(resume) if resume is not None else None
        cfg.dataloader.batch_size = cfg.val_dataloader.batch_size = 2
        workspace = distill.TrainDistillRenderNetWorkspace(cfg, str(tmp_path / name))

        def batch_loss(batch):
            obs = batch["obs"]
            selected_mask = obs["contact_phase"] > 0
            count = int(selected_mask.sum())
            token = float(obs["token_error"][selected_mask].mean()) if count else 0.
            region = float(obs["region_error"][selected_mask].mean()) if count else 0.
            ce = float(obs["condition_ce"].mean())
            alpha = float(obs["alpha_error"].mean())
            workspace.loss_metrics = {
                "token_mse": token, "token_mse_measured": count > 0,
                "token_region_observation_count": count,
                "mask_mse": region, "context_patch_mse": region,
                "intensity_smoothness": region, "intensity_loss": region,
                "phase_ce": ce, "alpha_mse": alpha,
            }
            # The expensive network's outputs are fixed for this aggregation
            # test; keep a valid graph so the real optimizer loop can run.
            zero = sum(parameter.sum() * 0 for parameter in workspace.model.parameters())
            return zero + token + ((region + ce + alpha) if workspace.model.training else 0)

        monkeypatch.setattr(workspace, "_batch_loss", batch_loss)
        return workspace

    return make


@pytest.mark.parametrize("batch_size", [2, 3, 6])
def test_validation_weights_token_and_regions_by_selected_observations(make_workspace, batch_size):
    workspace = make_workspace()
    workspace.cfg.val_dataloader.batch_size = batch_size
    loader = workspace._loader(PhaseDataset("unused"), workspace.cfg.val_dataloader, validation=True)
    assert workspace._validate(loader) == pytest.approx(5.)
    assert workspace.validation_samples == 6
    assert workspace.validation_selected_observations == 4
    for key in ("token_mse",):
        assert workspace.validation_metrics[key] == pytest.approx(5.)
    for key in ("mask_mse", "context_patch_mse", "intensity_smoothness", "intensity_loss"):
        assert workspace.validation_metrics[key] == pytest.approx(7.)
    assert workspace.validation_metrics["phase_ce"] == pytest.approx(4.)
    assert workspace.validation_metrics["alpha_mse"] == pytest.approx(3.)


def test_validation_with_no_selected_observations_rejects_false_zero_metric(make_workspace):
    workspace = make_workspace()
    data = PhaseDataset("unused")
    data.phases = torch.zeros_like(data.phases)
    loader = workspace._loader(data, workspace.cfg.val_dataloader, validation=True)
    with pytest.raises(RuntimeError, match="(?i)(selected|pre.contact|phase)"):
        workspace._validate(loader)


def test_training_epoch_metrics_exclude_free_batches_from_token_region_denominators(make_workspace):
    workspace = make_workspace()
    workspace.run()
    records = [json.loads(line) for line in
               (workspace.get_checkpoint_path().parent.parent / "logs.json.txt").read_text().splitlines()]
    epoch = next(record for record in records if "epoch_train_mse" in record)
    assert epoch["epoch_samples"] == 6
    assert epoch["epoch_token_region_observation_count"] == 4
    assert epoch["epoch_train_mse"] == pytest.approx(5.)
    assert epoch["epoch_train_token_mse"] == pytest.approx(5.)
    assert epoch["epoch_train_mask_mse"] == pytest.approx(7.)
    assert epoch["epoch_train_intensity_loss"] == pytest.approx(7.)
    assert epoch["epoch_train_phase_ce"] == pytest.approx(4.)
    assert epoch["epoch_train_alpha_mse"] == pytest.approx(3.)
    free = next(record for record in records if "train_mse" in record
                and record["train_token_region_observation_count"] == 0)
    assert free["objective"] == "token_mse"
    assert not free["train_token_mse_measured"]
    assert free["train_phase_ce"] == pytest.approx(6.)
    assert free["train_alpha_mse"] == pytest.approx(5.)


def test_changed_selection_scope_keeps_warmup_weights_and_rebuilds_best_baseline(make_workspace):
    first = make_workspace("resume", selected=False)
    with torch.no_grad():
        for parameter in first.model.parameters():
            parameter.fill_(.25)
    first.epoch = 1
    first.global_step = 1109
    first.training_stage = "condition_warmup"
    first.stage_step = 1109
    first.stage_epoch = 1
    first.best_val_loss = first.early_stopping_best_loss = .012
    first.initial_train_mse = .01
    first.initial_val_mse = first.latest_val_loss = .012
    first.epochs_without_improvement = 4
    first.lr_scheduler.step(.012)
    first.lr_scheduler.num_bad_epochs = 2
    first.lr_scheduler.cooldown_counter = 1
    first.get_checkpoint_path().parent.mkdir(parents=True, exist_ok=True)
    first.save_checkpoint(tag="best", use_thread=False)
    first.save_checkpoint(tag="latest", use_thread=False)
    best = first.get_checkpoint_path("best")
    old_best_digest = hashlib.sha256(best.read_bytes()).hexdigest()
    resumed = make_workspace("resume", selected=True, resume=first.get_checkpoint_path("latest"))
    assert resumed.epoch == 1 and resumed.global_step == 1109
    assert resumed.training_stage == "condition_warmup"
    assert resumed.stage_step == 1109 and resumed.stage_epoch == 1
    for value in resumed.model.state_dict().values():
        torch.testing.assert_close(value, torch.full_like(value, .25))
    assert resumed.selection_metric_changed
    assert math.isinf(resumed.best_val_loss)
    assert resumed.initial_train_mse is None and resumed.initial_val_mse is None
    assert resumed.epochs_without_improvement == 0
    assert math.isinf(resumed.lr_scheduler.best)
    assert resumed.lr_scheduler.num_bad_epochs == resumed.lr_scheduler.cooldown_counter == 0
    resumed.run()  # epoch limit reached: rebuild the baseline without another update.
    assert resumed.best_val_loss == pytest.approx(5.)
    assert resumed.lr_scheduler.best == pytest.approx(5.)
    assert resumed.best_epoch == 1 and resumed.best_step == 1109
    assert hashlib.sha256(best.with_name("best_previous_metric.ckpt").read_bytes()).hexdigest() == old_best_digest
    payload = torch.load(best, pickle_module=dill)
    assert dill.loads(payload["pickles"]["best_val_loss"]) == pytest.approx(5.)
    assert tuple(payload["cfg"].distillation.token_region_phases) == (1, 2)
    assert resumed.epoch == 1 and resumed.global_step == 1109
