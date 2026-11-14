"""Contact-only sample selection preserves context, actions and teacher cache rows."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.dataset.render_net_teacher_cache import CachedTeacherDataset, TeacherTargetCache
from diffusion_policy.workspace import train_distill_render_net_workspace as distill
from test_render_net_denoiser_distillation import make_workspace, RecordingDenoiser


def test_selected_windows_keep_original_cache_indices_and_full_history(tmp_path):
    workspace, batch = make_workspace()
    workspace.cfg.training = {"sample_phases": [2]}

    class Samples:
        def __len__(self):
            return 6

        def get_phase_labels(self):
            return np.array([0, 2, 1, 0, 2, 0])

        def __getitem__(self, index):
            return {"obs": {"contact_phase": torch.tensor([0, self.get_phase_labels()[index]]),
                            "wrench": torch.full((2, 8, 7), float(index))},
                    "action": torch.full((16, 7), float(index))}

    cache = TeacherTargetCache(tmp_path / "cache", {"test": "contact"}, 6)
    tokens = torch.arange(6).float()[:, None, None].expand(6, 4, 3)
    cache.append(tokens, {"side": torch.arange(6).float()[:, None, None]})
    cached = CachedTeacherDataset(Samples(), cache)
    selected = workspace._select_phase_dataset(cached, split="train")
    assert selected.indices == [1, 4]
    for item, original in zip(selected, (1, 4)):
        torch.testing.assert_close(item["teacher_targets"]["tokens"], tokens[original])
        assert item["action"].shape == (16, 7)
        assert item["obs"]["wrench"].shape == (2, 8, 7)
        assert item["obs"]["contact_phase"].tolist() == [0, 2]
        assert (item["action"] == original).all()
    assert workspace.phase_selection["train"]["selected_samples"] == 2
    assert workspace.metric_scope == "anchor_phases_2"
    workspace.cfg.training.sample_phases = None
    assert workspace._select_phase_dataset(cached, split="train") is cached


def test_empty_contact_split_fails_instead_of_reporting_zero_loss():
    workspace, _ = make_workspace()
    workspace.cfg.training = {"sample_phases": [2]}

    class FreeSamples:
        def __len__(self):
            return 3

        def get_phase_labels(self):
            return np.array([0, 1, 0])

    with pytest.raises(ValueError, match="No validation samples"):
        workspace._select_phase_dataset(FreeSamples(), split="validation")


@pytest.mark.parametrize("phases", [[], [3], [-1], [2.0], [True], "contact", 2])
def test_invalid_sample_phases_are_rejected(phases):
    workspace, _ = make_workspace()
    workspace.cfg.training = {"sample_phases": phases}
    with pytest.raises(ValueError, match="sample_phases"):
        workspace._validate_distillation_config()


def test_production_loop_trains_validates_and_selects_only_contact_samples(monkeypatch, tmp_path):
    from test_render_net_ddp import _config, TinyDataset, TinyTeacher, TinyRenderer, TinyPreprocessor

    class ContactDataset(TinyDataset):
        def get_phase_labels(self):
            return np.where(self.indices[:, 0] % 2 == 1, 2, 0)

        def __getitem__(self, index):
            sample = super().__getitem__(index)
            sample["obs"]["contact_phase"] = torch.tensor([self.get_phase_labels()[index]])
            sample["action"] = torch.full((4, 3), float(self.indices[index, 0]) * .03)
            return sample

        def get_validation_dataset(self):
            return ContactDataset(self.dataset_path, range(8, 13))

    teacher = TinyTeacher()
    teacher.model = RecordingDenoiser()
    teacher.noise_scheduler = DDIMScheduler(num_train_timesteps=10)
    teacher.normalizer.fit({"action": torch.randn(32, 3)})
    teacher_cfg = OmegaConf.create({"task": {"dataset": {}}})
    monkeypatch.setattr(distill, "load_checkpoint_payload", lambda _: ({"state_dicts": {}}, teacher_cfg))
    monkeypatch.setattr(distill, "instantiate_policy", lambda *args, **kwargs: teacher)
    monkeypatch.setattr(distill, "build_render_net", lambda *args: TinyRenderer())
    monkeypatch.setattr(distill.WrenchPreprocessor, "from_policy", lambda _: TinyPreprocessor())
    monkeypatch.setattr(distill.hydra.utils, "instantiate", lambda _: ContactDataset(tmp_path / "data"))
    cfg = _config(tmp_path)
    cfg.teacher_cache.enabled = False
    cfg.training.sample_phases = [2]
    cfg.distillation = {"objective": "denoiser_mse", "validation_seed": 42}
    workspace = distill.TrainDistillRenderNetWorkspace(cfg, str(tmp_path / "run"))
    workspace.run()
    summary = json.loads((tmp_path / "run/training_summary.json").read_text())
    assert summary["phase_selection"]["train"]["selected_samples"] == 4
    assert summary["phase_selection"]["validation"]["selected_samples"] == 2
    assert summary["validation_samples"] == 2
    assert summary["global_step"] == summary["batching"]["train_batches_per_epoch"] == 1
    assert summary["validation_metric"] == "denoiser MSE on anchor phases [2] (excludes auxiliary losses)"
    records = [json.loads(line) for line in (tmp_path / "run/logs.json.txt").read_text().splitlines()]
    assert records[0]["metric_scope"] == "anchor_phases_2"
    assert next(r for r in records if "epoch_samples" in r)["epoch_samples"] == 4
    assert next(r for r in records if "initial_train_samples" in r)["initial_train_samples"] == 4
    assert next(r for r in records if "initial_train_contact_samples" in r)["initial_train_contact_samples"] == 4
    assert next(r for r in records if "epoch_train_contact_samples" in r)["epoch_train_contact_samples"] == 4
    assert summary["latest_validation_metrics"]["contact_samples"] == 2
    curve_log = json.loads((tmp_path / "run/log.json").read_text())
    assert any("initial_train_contact_mse" in row for row in curve_log["records"])
    assert (tmp_path / "run/contact_loss.png").is_file()
    # Same-objective resume cannot silently reuse all-phase checkpoint scores.
    cfg.training.resume_from = str(workspace.get_checkpoint_path("latest"))
    cfg.training.sample_phases = None
    with pytest.raises(ValueError, match="different training.sample_phases"):
        distill.TrainDistillRenderNetWorkspace(cfg, str(tmp_path / "wrong_scope"))
