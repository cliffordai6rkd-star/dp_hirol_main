from pathlib import Path

import torch

from train.base_trainer import BaseTrainer


class _Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.values = torch.arange(4, dtype=torch.float32)[:, None]
        self.normalizer = None
        self.filter_config = {}
        self.sample_rate_hz = 100.0

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return {"x": self.values[index]}


class _Trainer(BaseTrainer):
    def build_dataset(self):
        return _Dataset()

    def build_model(self):
        return torch.nn.Linear(1, 1)

    def compute_loss(self, batch):
        prediction = self.model(batch["x"])
        loss = prediction.square().mean()
        return loss, {"loss_dict": {"total_loss": loss.detach()}}


def _config(output_dir, num_epochs, resume_from=None):
    train = {
        "device": "cpu",
        "batch_size": 2,
        "num_workers": 0,
        "num_epochs": num_epochs,
        "checkpoint_every_epochs": 1,
        "top_k": 5,
        "save_latest_checkpoint": True,
        "val_ratio": 0.0,
        "output_dir": str(output_dir),
        "scheduler": {"name": "cosine", "T_max": 4, "eta_min": 0.0},
        "ema": {"enabled": True, "decay": 0.9},
        "wandb": {"enabled": False},
    }
    if resume_from is not None:
        train["resume_from"] = str(resume_from)
    return {"train": train}


def test_resume_restores_optimizer_scheduler_ema_and_progress(tmp_path):
    first = _Trainer(_config(tmp_path, num_epochs=2))
    first.save_loss_plot = lambda: None
    first_summary = first.train()
    assert first_summary["global_step"] == 4

    latest = tmp_path / "checkpoints" / "latest.pt"
    assert latest.exists()
    checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
    assert checkpoint["trainer_state"]["resume_epoch"] == 2
    assert checkpoint["trainer_state"]["global_step"] == 4
    assert checkpoint["scheduler"] is not None
    assert checkpoint["optimizer"]
    assert checkpoint["model_raw"] is not None

    resumed = _Trainer(_config(tmp_path, num_epochs=4, resume_from=tmp_path))
    resumed.save_loss_plot = lambda: None
    resumed.setup()

    assert resumed.resume_checkpoint_path == latest.resolve()
    assert resumed.resume_epoch == 2
    assert resumed.global_step == 4
    assert all(
        Path(item["path"]).name != "latest.pt"
        for item in resumed.best_checkpoints
        if item.get("path") is not None
    )
    assert resumed.scheduler.last_epoch == checkpoint["scheduler"]["last_epoch"]
    assert resumed.optimizer.state
    for expected, actual in zip(
        checkpoint["model"].values(), resumed.ema.model.state_dict().values()
    ):
        assert torch.equal(expected, actual)

    # ``train`` is intentionally called after setup above to exercise the
    # normal entry point as well; a second setup is harmless and reloads the
    # same checkpoint before continuing from the configured total budget.
    resumed = _Trainer(_config(tmp_path, num_epochs=4, resume_from=tmp_path))
    resumed.save_loss_plot = lambda: None
    summary = resumed.train()
    assert summary["global_step"] == 8
    assert summary["num_epochs"] == 4


def test_resolve_resume_checkpoint_prefers_newest_progress_and_handles_file(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    payload = {"global_step": 2, "epoch": 1, "model": {}}
    torch.save(payload, checkpoint_dir / "step_00000002.pt")
    torch.save({"global_step": 4, "epoch": 2, "model": {}}, checkpoint_dir / "step_00000004.pt")
    torch.save({"global_step": 4, "epoch": 2, "model": {}}, checkpoint_dir / "latest.pt")

    assert BaseTrainer.resolve_resume_checkpoint(tmp_path).name == "latest.pt"
    assert (
        BaseTrainer.resolve_resume_checkpoint(checkpoint_dir / "step_00000002.pt").name
        == "step_00000002.pt"
    )


def test_resolve_resume_checkpoint_ignores_stale_latest(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    torch.save({"global_step": 10, "epoch": 1, "model": {}}, checkpoint_dir / "step_00000010.pt")
    torch.save({"global_step": 9, "epoch": 1, "model": {}}, checkpoint_dir / "latest.pt")
    assert BaseTrainer.resolve_resume_checkpoint(tmp_path).name == "step_00000010.pt"


def test_resolve_resume_checkpoint_uses_metadata_when_latest_missing(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    torch.save({"global_step": 10, "epoch": 1, "model": {}}, checkpoint_dir / "epoch_0000001.pt")
    torch.save({"global_step": 20, "epoch": 1, "model": {}}, checkpoint_dir / "step_00000002.pt")
    resolved = BaseTrainer.resolve_resume_checkpoint(tmp_path)
    assert resolved.name == "step_00000002.pt"
