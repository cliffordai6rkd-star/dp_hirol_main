"""Teacher plots distinguish measured objectives, epoch means and action MSE."""

import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image

from diffusion_policy.scripts.plot_teacher_losses import (
    TeacherLossHistory, infer_objective_label, save_loss_plot,
)


def test_history_uses_optimizer_steps_and_keeps_epoch_average_separate():
    history = TeacherLossHistory()
    history.update([
        {"global_step": 100, "optimizer_step": 50, "epoch": 1, "train_loss": .7},
        {"global_step": 100, "optimizer_step": 50, "epoch": 1,
         "train_loss": .6, "epoch_train_loss": .6},
        {"optimizer_step": 50, "val_loss": .5, "val_action_mse": .3,
         "val_action_mse_raw": .02, "curriculum/blur_sigma": 5},
        {"optimizer_step": 100, "epoch": 2, "train_loss": None, "val_loss": float("nan")},
    ])
    assert history.points("train_loss") == ([50], [.7])
    assert history.points("epoch_train_loss") == ([50], [.6])
    assert history.points("val_loss") == ([50], [.5])
    assert history.points("val_action_mse_raw") == ([50], [.02])
    assert "val_loss=0.5@step50" in history.summary()
    assert history.step == 100 and history.epoch == 2
    history.update([{"optimizer_step": 1, "train_loss": .9}], reset=True)
    assert history.points("train_loss") == ([1], [.9])
    assert not history.points("val_loss")[0]


def test_single_record_creates_plot_without_inventing_validation(tmp_path):
    history = TeacherLossHistory()
    history.update([{"optimizer_step": 50, "epoch": 0, "train_loss": .675,
                     "curriculum/blur_sigma": 5}])
    output = tmp_path / "training_losses.png"
    assert save_loss_plot(history, output) == output
    assert "val_loss=not_recorded" in history.summary()
    assert not history.series["val_loss"]
    with Image.open(output) as figure:
        assert figure.format == "PNG" and figure.width >= 1500 and figure.height >= 900
    assert not list(tmp_path.glob("*.tmp.png"))


def test_objective_label_reads_plain_saved_config(tmp_path):
    config = tmp_path / ".hydra" / "config.yaml"
    config.parent.mkdir()
    config.write_text("action_head_backbone: act\npolicy:\n  act_loss_type: l1\n")
    assert infer_objective_label(tmp_path) == "Normalized action L1 (ACT)"
    config.write_text("policy:\n  action_head_backbone: dit\n")
    assert infer_objective_label(tmp_path) == "Denoising MSE (DiT)"


def test_cli_prints_latest_measurements_ignores_partial_tail_and_uses_no_torch(tmp_path):
    source = tmp_path / "logs.json.txt"
    source.write_text(json.dumps({"optimizer_step": 50, "train_loss": .675104439})
                      + '\n{"optimizer_step": 100')
    code = (
        "import os,runpy,sys; "
        "sys.argv=['plot_teacher_losses',sys.argv[1]]; "
        "runpy.run_module('diffusion_policy.scripts.plot_teacher_losses',run_name='__main__'); "
        "assert os.environ['CUDA_VISIBLE_DEVICES']==''; assert 'torch' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "step=50" in result.stdout and "train_loss=0.67510444" in result.stdout
    assert "val_loss=not_recorded" in result.stdout
    assert (tmp_path / "training_losses.png").is_file()


def test_cli_quiet_mode_still_writes_plot(tmp_path):
    source = tmp_path / "logs.json.txt"
    source.write_text('{"optimizer_step": 10, "train_loss": 0.7, "val_loss": 0.4}\n')
    result = subprocess.run([
        sys.executable, "-m", "diffusion_policy.scripts.plot_teacher_losses",
        str(tmp_path), "--quiet",
    ], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == ""
    assert (tmp_path / "training_losses.png").is_file()


def test_import_does_not_hide_training_cuda_devices():
    code = (
        "import os,sys; import diffusion_policy.scripts.plot_teacher_losses; "
        "assert os.environ['CUDA_VISIBLE_DEVICES']=='2,3'; assert 'torch' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "2,3"},
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
