"""Exercise teacher epoch scheduling, early stopping and real checkpoint IO."""

import dill
import torch
from omegaconf import OmegaConf
from torch import nn

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.policy.force_aware_diffusion_transformer_policy import (
    ForceAwareDiffusionTransformerPolicy,
)
from diffusion_policy.workspace import train_force_aware_diffusion_workspace as workspace_module
from diffusion_policy.workspace.train_force_aware_diffusion_workspace import (
    TrainForceAwareDiffusionWorkspace,
)


class _EpochMarkerPolicy(ForceAwareDiffusionTransformerPolicy):
    """The weight identifies the completed epoch in saved checkpoints."""

    def __init__(self):
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.normalizer = LinearNormalizer()
        self.action_start_index = 0
        self.n_action_steps = 1
        self.action_head_backbone = "act"
        self.optimizer_step = 0
        self.last_curriculum_metrics = {}
        self.blur_scheduler = None

    def get_freeze_config(self):
        return {"freeze_modules": (), "freeze_dino_backbone": True}

    def get_optimizer(self, learning_rate, **kwargs):
        return torch.optim.SGD([self.weight], lr=learning_rate)

    def predict_action(self, obs, **kwargs):
        actions = self.weight.expand(obs["value"].shape[0], 1, 1)
        return {"action": actions, "action_pred": actions, "model_action_pred": actions}


class _TinyDataset(BaseImageDataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {"obs": {"value": torch.tensor([float(index)])},
                "action": torch.zeros(1, 1)}

    def get_validation_dataset(self):
        return _TinyDataset()

    def get_normalizer(self, **kwargs):
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
        return normalizer


def _config(*, num_epochs=1000, resume=False):
    return OmegaConf.create({
        "policy": {"_target_": "integration_test_policy", "action_head_backbone": "act"},
        "optimizer": {"learning_rate": 0.01},
        "task": {"dataset": {"_target_": "integration_test_dataset"}, "env_runner": None},
        "dataloader": {"batch_size": 2, "num_workers": 0, "shuffle": False},
        "val_dataloader": {"batch_size": 2, "num_workers": 0, "shuffle": False},
        "phase_resampling": {"enabled": False},
        "training": {
            "device": "cpu", "seed": 42, "debug": False, "resume": resume,
            "num_epochs": num_epochs, "num_optimizer_steps": None,
            "amp": {"enabled": False}, "distributed": {"enabled": False},
            "use_ema": False, "lr_scheduler": "constant", "lr_warmup_steps": 0,
            "gradient_accumulate_every": 1, "max_train_steps": None,
            "max_val_steps": None, "max_grad_norm": None, "tqdm_interval_sec": 1.0,
            "rollout_every": 10000, "sample_every": 10000,
            "val_every": 5, "val_epoch_every": 5,
            "validation_start_optimizer_step": 0,
            "checkpoint_every": 5, "checkpoint_every_optimizer_steps": None,
            "early_stopping": {
                "enabled": True, "monitor": "val_loss", "patience": 100,
                "min_delta": 0.0,
            },
        },
        "logging": {"mode": "disabled"},
        "checkpoint": {
            "topk": {
                "monitor_key": "val_loss", "mode": "min", "k": 1,
                "format_str": "epoch={epoch:04d}-val_loss={val_loss:.4f}.ckpt",
            },
            "save_last_ckpt": True, "save_last_snapshot": False,
            "action_mse": {"enabled": False},
        },
    })


def _instrument(monkeypatch, *, improve_at=None):
    """Mock expensive work; keep the actual epoch loop, validation and saving."""
    real_instantiate = workspace_module.hydra.utils.instantiate
    validation_epochs = []

    def instantiate(config, *args, **kwargs):
        if config.get("_target_") == "integration_test_policy":
            return _EpochMarkerPolicy()
        if config.get("_target_") == "integration_test_dataset":
            return _TinyDataset()
        return real_instantiate(config, *args, **kwargs)

    def train_epoch(self, *, lr_scheduler, **kwargs):
        with torch.no_grad():
            self.model.weight.fill_(self.epoch + 1)
        self.optimizer.step()
        lr_scheduler.step()
        self.optimizer_step += 1
        self.global_step += 1
        self.model.set_optimizer_step(self.optimizer_step)
        return {"train_loss": 0.1}

    def validate_metrics(self, policy, *args, **kwargs):
        completed_epoch = int(policy.weight.item())
        validation_epochs.append(completed_epoch)
        value = 1.0 if completed_epoch == 5 else 1.5
        if improve_at is not None and completed_epoch >= improve_at:
            value = 0.5 if completed_epoch == improve_at else 0.75
        return {"val_loss": value, "val_action_mse": value, "val_action_mse_raw": value}

    monkeypatch.setattr(workspace_module.hydra.utils, "instantiate", instantiate)
    monkeypatch.setattr(TrainForceAwareDiffusionWorkspace, "_train_epoch", train_epoch)
    monkeypatch.setattr(TrainForceAwareDiffusionWorkspace, "_validate_metrics", validate_metrics)
    monkeypatch.setattr(TrainForceAwareDiffusionWorkspace, "_update_validation_plot", lambda *a, **k: None)
    return validation_epochs


def _read_checkpoint(path):
    payload = torch.load(path, map_location="cpu", pickle_module=dill, weights_only=False)
    state = {key: dill.loads(value) for key, value in payload["pickles"].items()}
    return payload, state


def test_validation_every_five_epochs_stops_after_100_stale_epochs(monkeypatch, tmp_path):
    validation_epochs = _instrument(monkeypatch)
    workspace = TrainForceAwareDiffusionWorkspace(_config(), output_dir=str(tmp_path))
    workspace.run()

    assert validation_epochs == list(range(5, 106, 5))
    assert workspace.epoch == 105
    assert workspace.early_stopping_state["best_epoch"] == 5
    assert workspace.early_stopping_state["stale_epochs"] == 100
    assert workspace.early_stopping_state["should_stop"]
    best, best_state = _read_checkpoint(tmp_path / "checkpoints" / "best.ckpt")
    latest, latest_state = _read_checkpoint(tmp_path / "checkpoints" / "latest.ckpt")
    assert best["state_dicts"]["model"]["weight"].item() == 5
    assert best_state["epoch"] == 5
    assert latest["state_dicts"]["model"]["weight"].item() == 105
    assert latest_state["epoch"] == 105
    assert latest_state["early_stopping_state"]["should_stop"]


def test_validation_improvement_resets_patience_and_updates_best(monkeypatch, tmp_path):
    validation_epochs = _instrument(monkeypatch, improve_at=10)
    workspace = TrainForceAwareDiffusionWorkspace(_config(), output_dir=str(tmp_path))
    workspace.run()

    assert validation_epochs == list(range(5, 111, 5))
    assert workspace.epoch == 110
    assert workspace.early_stopping_state["best_epoch"] == 10
    assert workspace.early_stopping_state["best_value"] == 0.5
    assert workspace.early_stopping_state["stale_epochs"] == 100
    best, state = _read_checkpoint(tmp_path / "checkpoints" / "best.ckpt")
    assert best["state_dicts"]["model"]["weight"].item() == 10
    assert state["epoch"] == 10


def test_resume_restores_patience_instead_of_starting_a_new_100_epochs(monkeypatch, tmp_path):
    validation_epochs = _instrument(monkeypatch)
    first = TrainForceAwareDiffusionWorkspace(_config(num_epochs=95), output_dir=str(tmp_path))
    first.run()
    assert first.epoch == 95
    assert first.early_stopping_state["stale_epochs"] == 90
    assert not first.early_stopping_state["should_stop"]

    validation_epochs.clear()
    resumed = TrainForceAwareDiffusionWorkspace(_config(resume=True), output_dir=str(tmp_path))
    resumed.run()

    assert validation_epochs == [100, 105]
    assert resumed.epoch == 105
    assert resumed.early_stopping_state["best_epoch"] == 5
    assert resumed.early_stopping_state["stale_epochs"] == 100
    best, best_state = _read_checkpoint(tmp_path / "checkpoints" / "best.ckpt")
    _, latest_state = _read_checkpoint(tmp_path / "checkpoints" / "latest.ckpt")
    assert best["state_dicts"]["model"]["weight"].item() == 5
    assert best_state["epoch"] == 5
    assert latest_state["epoch"] == 105
