"""Concise console progress must not discard per-step training metrics."""

from functools import partial
import io
import json
import logging
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from diffusion_policy.workspace import train_distill_render_net_workspace as distill
from test_render_net_ddp import _config, TinyDataset, TinyPreprocessor, TinyRenderer, TinyTeacher
from test_render_net_teacher_cache import SourceDataset


class ManualClock:
    def __init__(self):
        self.seconds = 0.0

    def monotonic(self):
        return self.seconds

    def advance(self, seconds):
        self.seconds += seconds


def configure_console(monkeypatch, *, tty):
    output = io.StringIO()
    monkeypatch.setattr(distill, "sys", SimpleNamespace(stderr=SimpleNamespace(isatty=lambda: tty)))
    monkeypatch.setattr(distill, "tqdm", partial(tqdm, file=output))
    return output


@pytest.mark.parametrize("tty", [False, True])
def test_console_progress_is_throttled_without_losing_step_or_epoch_metrics(monkeypatch, tmp_path, caplog, tty):
    output = configure_console(monkeypatch, tty=tty)
    clock = ManualClock()
    monkeypatch.setattr(distill, "time", clock)
    teacher_config = OmegaConf.create({"task": {"dataset": {}}})
    monkeypatch.setattr(distill, "load_checkpoint_payload", lambda _: ({"state_dicts": {}}, teacher_config))
    monkeypatch.setattr(distill, "instantiate_policy", lambda *args, **kwargs: TinyTeacher())
    monkeypatch.setattr(distill, "build_render_net", lambda *args: TinyRenderer())
    monkeypatch.setattr(distill.WrenchPreprocessor, "from_policy", lambda _: TinyPreprocessor())
    monkeypatch.setattr(distill.hydra.utils, "instantiate", lambda _: TinyDataset(tmp_path / "data"))
    config = _config(tmp_path)
    config.teacher_cache.enabled = False
    config.training.num_epochs = 2
    # The legacy setting must not restore verbose per-batch console messages.
    config.training.log_every = 1
    config.logging.update({"progress_bar": True, "progress_bar_non_tty": False,
                           "tqdm_interval_sec": 0.0, "tqdm_leave": False,
                           "status_interval_sec": 25.0})
    config.dataloader.batch_size = config.val_dataloader.batch_size = 2
    workspace = distill.TrainDistillRenderNetWorkspace(config, str(tmp_path / "run"))
    workspace.shape_info = {}
    batch_loss = workspace._batch_loss

    def timed_batch_loss(batch):
        loss = batch_loss(batch)
        clock.advance(10.0)
        return loss

    monkeypatch.setattr(workspace, "_batch_loss", timed_batch_loss)
    with caplog.at_level(logging.INFO, logger=distill.log.name):
        workspace.run()
    messages = [record.getMessage() for record in caplog.records if record.name == distill.log.name]
    train_status = [message for message in messages if message.startswith("Train ")]
    val_status = [message for message in messages if message.startswith("Val ")]
    summaries = [message for message in messages if message.startswith("Epoch ")]
    assert len(summaries) == 2
    assert all("loss=" in message and "train=" in message and "field=" in message
               and "val=" in message and "best=" in message and "stale=" in message
               for message in summaries)
    assert not any(message.startswith("epoch=") or message.startswith("Validating epoch=")
                   for message in messages)
    if tty:
        assert not train_status and not val_status
        assert "Train 1/2" in output.getvalue() and "Val 1" in output.getvalue()
    else:
        # Four ten-second training batches per epoch: one report after batch 3,
        # not four lines. Three validation batches likewise produce one report.
        assert len(train_status) == 2
        assert all("batch=3/4" in message for message in train_status)
        assert len(val_status) == 3  # Initial evaluation and both epochs.
        assert all("batch=3/3" in message for message in val_status)
        assert output.getvalue() == ""

    rows = [json.loads(line) for line in (tmp_path / "run/logs.json.txt").read_text().splitlines()]
    steps = [row for row in rows if "train_mse" in row]
    assert [row["global_step"] for row in steps] == list(range(1, 9))
    required = {"train_loss", "train_token_mse", "train_denoiser_mse", "train_demo_mse",
                "train_mask_mse", "grad_norm", "lr", "step_seconds", "samples_per_second"}
    assert all(required <= row.keys() for row in steps)
    epochs = [row for row in rows if "epoch_train_mse" in row]
    assert [row["epoch"] for row in epochs] == [1, 2]
    assert all(row["epoch_samples"] == 8 for row in epochs)
    assert len([row for row in rows if "val_mse" in row]) == 2
    assert (tmp_path / "run/checkpoints/latest.ckpt").is_file()
    assert (tmp_path / "run/checkpoints/best.ckpt").is_file()
    assert not (tmp_path / "run/training_metrics.png").exists()


@pytest.mark.parametrize("tty", [False, True])
def test_cache_has_one_completion_summary_and_only_periodic_non_tty_status(monkeypatch, tmp_path, caplog, tty):
    output = configure_console(monkeypatch, tty=tty)
    clock = ManualClock()
    monkeypatch.setattr(distill, "time", clock)
    workspace = object.__new__(distill.TrainDistillRenderNetWorkspace)
    workspace._output_dir = str(tmp_path / "run")
    workspace.cfg = OmegaConf.create({
        "teacher_cache": {"enabled": True, "batch_size": 2, "num_workers": 0,
                          "dtype": "float32", "precision": "float32"},
        "training": {"seed": 42},
        "logging": {"progress_bar": True, "progress_bar_non_tty": False,
                    "status_interval_sec": 25.0, "tqdm_interval_sec": 0.0},
    })
    workspace.device = torch.device("cpu")
    workspace.attention_enabled = False

    def encode(obs, **kwargs):
        clock.advance(10.0)
        return obs["value"][:, None, :].expand(-1, 2, 3)

    workspace.teacher_policy = SimpleNamespace(_encode_context=encode)
    monkeypatch.setattr(workspace, "_teacher_cache_spec", lambda *args: (tmp_path / "cache", {}))
    with caplog.at_level(logging.INFO, logger=distill.log.name):
        train, _ = workspace._prepare_cached_datasets(
            SourceDataset(tmp_path, range(5)), [], OmegaConf.create({}),
        )
    messages = [record.getMessage() for record in caplog.records if record.name == distill.log.name]
    assert len([message for message in messages if message.startswith("Teacher cache train ready:")]) == 1
    periodic = [message for message in messages if message.startswith("Teacher cache train:")
                and "samples/s" in message]
    assert len(periodic) == (0 if tty else 1)
    if periodic:
        assert "5/5" in periodic[0] and "elapsed=30.0s" in periodic[0]
    assert bool(output.getvalue()) is tty
    assert train.cache.complete and train.cache.completed == 5


@pytest.mark.parametrize("tty", [False, True])
def test_attention_status_obeys_configured_interval_and_does_not_duplicate_bar(monkeypatch, caplog, tty):
    configure_console(monkeypatch, tty=tty)
    workspace = object.__new__(distill.TrainDistillRenderNetWorkspace)
    workspace.cfg = OmegaConf.create({"logging": {"progress_bar": True,
                                                 "progress_bar_non_tty": False,
                                                 "status_interval_sec": 60.0}})
    with caplog.at_level(logging.INFO, logger=distill.log.name):
        workspace._attention_progress(0, 100, 0)
        workspace._attention_progress(20, 100, 30)
        workspace._attention_progress(40, 100, 59)
        assert not caplog.records
        workspace._attention_progress(60, 100, 60)
        workspace._attention_progress(80, 100, 90)
        workspace._attention_progress(100, 100, 120)
    messages = [record.getMessage() for record in caplog.records if record.name == distill.log.name]
    assert len(messages) == (0 if tty else 2)
    if messages:
        assert "samples=60/100" in messages[0] and "samples=100/100" in messages[1]
