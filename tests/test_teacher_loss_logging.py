"""Teacher losses remain visible in stdout and in a CPU-only live plot."""

import json
import os
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace import train_force_aware_diffusion_workspace as workspace_module
from diffusion_policy.workspace.train_force_aware_diffusion_workspace import (
    DistributedContext,
    TrainForceAwareDiffusionWorkspace,
)
from test_force_aware_workspace import _FakePolicy, _Logger
from test_teacher_early_stopping_integration import _config, _instrument


def _workspace(directory=None, rank=0, world_size=1):
    workspace = TrainForceAwareDiffusionWorkspace.__new__(TrainForceAwareDiffusionWorkspace)
    workspace._output_dir = str(directory) if directory is not None else None
    workspace.distributed = DistributedContext(rank=rank, world_size=world_size)
    workspace.epoch = workspace.global_step = workspace.optimizer_step = 0
    return workspace


def test_step_loss_prints_first_interval_and_final_updates_and_returns_epoch_mean(capsys):
    workspace = _workspace()
    workspace.model = _FakePolicy()
    workspace.optimizer = torch.optim.SGD(workspace.model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(workspace.optimizer, lambda _: 1.0)
    cfg = OmegaConf.create({"training": {
        "max_train_steps": None, "gradient_accumulate_every": 1,
        "log_every_optimizer_steps": 3, "max_grad_norm": None, "tqdm_interval_sec": 1.0,
    }})
    metrics = _Logger()
    result = workspace._train_epoch(
        cfg=cfg, dataloader=[{"value": torch.tensor([1.0])} for _ in range(5)],
        device=torch.device("cpu"), lr_scheduler=scheduler, ema=None,
        wandb_run=_Logger(), json_logger=metrics,
    )
    output = capsys.readouterr().out
    assert [row["optimizer_step"] for row in metrics.records] == [1, 3, 5]
    assert output.count("[train]") == 3
    assert "optimizer_step=1 train_loss=" in output
    assert "optimizer_step=3 train_loss=" in output
    assert "optimizer_step=5 train_loss=" in output
    assert "lr=" in output
    assert result["epoch_train_loss"] == result["train_loss"]


def test_loss_print_labels_ddp_local_step_and_global_epoch_and_ignores_other_ranks(capsys):
    record = {"epoch": 4, "optimizer_step": 51, "train_loss": .4,
              "epoch_train_loss": .35, "lr": 1e-4, "curriculum/blur_sigma": 2.5,
              "val_loss": .2, "val_action_mse": .1, "val_action_mse_raw": .01}
    main = _workspace(rank=0, world_size=2)
    main._print_loss_record("train", record)
    main._print_loss_record("epoch", record)
    main._print_loss_record("val", record)
    output = capsys.readouterr().out
    assert "train_loss(rank0)=0.4" in output
    assert "epoch_train_loss(global)=0.35" in output
    assert "blur_sigma=2.5" in output
    assert "val_loss(clear)=0.2" in output
    assert "val_action_mse(normalized)=0.1" in output
    assert "val_action_mse(raw)=0.01" in output
    _workspace(rank=1, world_size=2)._print_loss_record("train", record)
    assert capsys.readouterr().out == ""


def test_epoch_and_validation_loss_are_persisted_and_printed(monkeypatch, tmp_path, capsys):
    _instrument(monkeypatch)
    workspace = TrainForceAwareDiffusionWorkspace(_config(num_epochs=5), output_dir=str(tmp_path))
    workspace.run()
    output = capsys.readouterr().out
    assert output.count("[epoch]") == 5
    assert "[val] epoch=5 optimizer_step=5 val_loss(clear)=1" in output
    assert "best_val=1" in output
    rows = [json.loads(line) for line in (tmp_path / "logs.json.txt").read_text().splitlines()]
    epoch_rows = [row for row in rows if "epoch_train_loss" in row]
    assert len(epoch_rows) == 5
    assert all(row["epoch_train_loss"] == row["train_loss"] == .1 for row in epoch_rows)


def test_plotter_cleanup_and_final_render_run_on_training_error(monkeypatch, tmp_path):
    workspace = _workspace(tmp_path)
    process = Mock()
    process.poll.return_value = None
    start = Mock(return_value=process)
    finish = Mock()
    monkeypatch.setattr(workspace_module.subprocess, "Popen", start)
    monkeypatch.setattr(workspace_module.subprocess, "run", finish)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    cfg = OmegaConf.create({"logging": {"enable_plots": True, "plot_interval_sec": 10}})
    with pytest.raises(RuntimeError, match="training failed"):
        with workspace._training_plotter(cfg):
            raise RuntimeError("training failed")
    assert start.call_count == finish.call_count == 1
    command = start.call_args.args[0]
    assert "--watch" in command and "--quiet" in command
    assert command[command.index("--pid") + 1] == str(os.getpid())
    assert start.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"
    process.terminate.assert_called_once()
    process.wait.assert_called_once()
    assert "--watch" not in finish.call_args.args[0]


@pytest.mark.parametrize("rank,enabled", [(0, False), (1, True)])
def test_plotter_is_optional_and_only_runs_on_rank_zero(monkeypatch, tmp_path, rank, enabled):
    workspace = _workspace(tmp_path, rank=rank, world_size=2)
    start = Mock()
    monkeypatch.setattr(workspace_module.subprocess, "Popen", start)
    cfg = OmegaConf.create({"logging": {"enable_plots": enabled}})
    with workspace._training_plotter(cfg):
        pass
    start.assert_not_called()


def test_automatic_plotter_renders_complete_final_record(tmp_path):
    (tmp_path / "logs.json.txt").write_text(json.dumps({
        "optimizer_step": 50, "epoch": 1, "train_loss": .6,
        "curriculum/blur_sigma": 5.,
    }) + "\n")
    workspace = _workspace(tmp_path)
    cfg = OmegaConf.create({"logging": {"enable_plots": True, "plot_interval_sec": 10}})
    with workspace._training_plotter(cfg):
        pass
    assert (tmp_path / "training_losses.png").stat().st_size > 3000
