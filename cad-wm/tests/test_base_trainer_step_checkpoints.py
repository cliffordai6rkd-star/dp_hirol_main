from types import SimpleNamespace

import torch

from train.base_trainer import BaseTrainer


class _Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.values = torch.arange(3, dtype=torch.float32)[:, None]
        self.normalizer = SimpleNamespace(stats={}, eps=1.0e-6)
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


class _ValidationCountingTrainer(_Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.validation_epochs = []

    def setup(self):
        super().setup()
        # The base toy dataset has no episode metadata, so use a sentinel to
        # exercise validation cadence without changing dataset split logic.
        self.val_loader = object()

    def validate_one_epoch(self, epoch):
        self.validation_epochs.append(epoch)
        self.last_val_epoch_metrics = {"marker": float(epoch)}
        return float(epoch)


def test_step_training_retains_latest_topk_and_latest_file(tmp_path):
    config = {
        "train": {
            "device": "cpu",
            "batch_size": 2,
            "num_workers": 0,
            "num_epochs": 100,
            "max_optimizer_steps": 5,
            "checkpoint_every_steps": 2,
            "top_k": 2,
            "save_latest_checkpoint": True,
            "val_ratio": 0.0,
            "output_dir": str(tmp_path),
            "scheduler": {"name": "cosine", "T_max": 5, "eta_min": 0.0},
            "ema": {"enabled": False},
            "wandb": {"enabled": False},
        }
    }
    trainer = _Trainer(config)
    trainer.save_loss_plot = lambda: None
    summary = trainer.train()

    checkpoint_dir = tmp_path / "checkpoints"
    assert sorted(path.name for path in checkpoint_dir.glob("step_*.pt")) == [
        "step_00000004.pt",
        "step_00000005.pt",
    ]
    latest = torch.load(
        checkpoint_dir / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert latest["global_step"] == 5
    assert summary["max_optimizer_steps"] == 5
    assert summary["checkpoint_every_steps"] == 2
    assert [item["global_step"] for item in summary["best_checkpoints"]] == [4, 5]
    assert trainer.scheduler.last_epoch == 5


def test_step_checkpoint_is_not_delayed_to_epoch_boundary(tmp_path):
    config = {
        "train": {
            "device": "cpu",
            "batch_size": 1,
            "num_workers": 0,
            "num_epochs": 100,
            "max_optimizer_steps": 5,
            "checkpoint_every_steps": 2,
            "top_k": 3,
            "save_latest_checkpoint": True,
            "val_ratio": 0.0,
            "output_dir": str(tmp_path),
            "scheduler": {"name": "none"},
            "ema": {"enabled": False},
            "wandb": {"enabled": False},
        }
    }
    trainer = _Trainer(config)
    trainer.save_loss_plot = lambda: None
    trainer.train()

    checkpoint_dir = tmp_path / "checkpoints"
    assert sorted(path.name for path in checkpoint_dir.glob("step_*.pt")) == [
        "step_00000002.pt",
        "step_00000004.pt",
        "step_00000005.pt",
    ]


def test_step_training_defaults_null_top_k(tmp_path):
    config = {
        "train": {
            "device": "cpu",
            "batch_size": 2,
            "num_workers": 0,
            "num_epochs": 100,
            "max_train_steps": 2,
            "checkpoint_every_steps": 2,
            "top_k": None,
            "save_latest_checkpoint": True,
            "val_ratio": 0.0,
            "output_dir": str(tmp_path),
            "scheduler": {"name": "cosine", "T_max": 2, "eta_min": 0.0},
            "ema": {"enabled": False},
            "wandb": {"enabled": False},
        }
    }
    trainer = _Trainer(config)

    assert trainer.top_k == 3


def test_cosine_without_tmax_infers_optimizer_steps_and_warmup(tmp_path):
    config = {
        "train": {
            "device": "cpu",
            "batch_size": 2,
            "num_workers": 0,
            "num_epochs": 2,
            "val_ratio": 0.0,
            "output_dir": str(tmp_path),
            "scheduler": {"name": "cosine", "warmup_steps": 2, "eta_min": 0.0},
            "ema": {"enabled": False},
            "wandb": {"enabled": False},
        }
    }
    trainer = _Trainer(config)
    trainer.setup()

    # The toy loader has two batches per epoch, so two epochs imply four
    # optimizer steps. LambdaLR starts at zero during the warmup.
    assert trainer.scheduler_total_steps == 4
    assert trainer.scheduler_warmup_steps == 2
    assert trainer.scheduler_step_per_optimizer_step is True
    assert trainer.optimizer.param_groups[0]["lr"] == 0.0

    trainer.train_one_epoch(0)
    assert trainer.scheduler.last_epoch == 2
    assert trainer.optimizer.param_groups[0]["lr"] == trainer.lr

    trainer.train_one_epoch(1)
    assert trainer.scheduler.last_epoch == 4
    assert trainer.optimizer.param_groups[0]["lr"] == 0.0


def test_val_every_skips_intermediate_validation_epochs(tmp_path):
    config = {
        "dataloader": {},
        "train": {
            "device": "cpu",
            "batch_size": 2,
            "num_workers": 0,
            "num_epochs": 5,
            "val_ratio": 0.0,
            "val_every": 2,
            "output_dir": str(tmp_path),
            "scheduler": None,
            "ema": {"enabled": False},
            "wandb": {"enabled": False},
        }
    }
    trainer = _ValidationCountingTrainer(config)
    trainer.save_loss_plot = lambda: None
    summary = trainer.train()

    assert trainer.validation_epochs == [0, 2, 4]
    assert summary["val_every"] == 2
    assert [item["val_loss"] for item in trainer.loss_history] == [
        0.0,
        None,
        2.0,
        None,
        4.0,
    ]


def test_epoch_training_retains_latest_scheduled_topk(tmp_path):
    config = {
        "train": {
            "device": "cpu",
            "batch_size": 2,
            "num_workers": 0,
            "num_epochs": 6,
            "checkpoint_every_epochs": 2,
            "top_k": 3,
            "save_latest_checkpoint": False,
            "val_ratio": 0.0,
            "output_dir": str(tmp_path),
            "scheduler": {"name": "cosine", "T_max": 6, "eta_min": 0.0},
            "ema": {"enabled": False},
            "wandb": {"enabled": False},
        }
    }
    trainer = _Trainer(config)
    trainer.save_loss_plot = lambda: None
    summary = trainer.train()

    checkpoint_dir = tmp_path / "checkpoints"
    assert sorted(path.name for path in checkpoint_dir.glob("epoch_*.pt")) == [
        "epoch_0000002.pt",
        "epoch_0000004.pt",
        "epoch_0000006.pt",
    ]
    assert [item["epoch"] for item in summary["best_checkpoints"]] == [2, 4, 6]
    assert summary["checkpoint_every_epochs"] == 2
    assert not (checkpoint_dir / "latest.pt").exists()


def test_validation_dataloader_can_use_independent_low_memory_settings(tmp_path):
    config = {
        "train": {
            "device": "cpu",
            "batch_size": 2,
            "num_workers": 2,
            "prefetch_factor": 2,
            "persistent_workers": True,
            "val_num_workers": 0,
            "val_prefetch_factor": 1,
            "val_persistent_workers": False,
            "output_dir": str(tmp_path),
            "val_ratio": 0.0,
            "scheduler": None,
            "ema": {"enabled": False},
            "wandb": {"enabled": False},
        }
    }
    trainer = _Trainer(config)

    train_kwargs = trainer._dataloader_kwargs(shuffle=True)
    assert train_kwargs["num_workers"] == 2
    assert train_kwargs["prefetch_factor"] == 2
    assert train_kwargs["persistent_workers"] is True

    val_kwargs = trainer._dataloader_kwargs(
        shuffle=False,
        num_workers=trainer.val_num_workers,
        prefetch_factor=trainer.val_prefetch_factor,
        pin_memory=trainer.val_pin_memory,
        persistent_workers=trainer.val_persistent_workers,
    )
    assert val_kwargs == {
        "batch_size": 2,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": False,
    }
