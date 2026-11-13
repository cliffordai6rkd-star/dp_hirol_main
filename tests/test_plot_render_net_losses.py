"""Live CPU loss plots must preserve measurements while logs are being written."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from diffusion_policy.scripts.plot_render_net_losses import (
    LossHistory, MetricsReader, save_loss_plot,
)


def test_jsonl_reader_waits_for_complete_lines_and_only_returns_new_records(tmp_path):
    source = tmp_path / "logs.json.txt"
    first = {"global_step": 1, "train_loss": .9}
    source.write_text(json.dumps(first) + '\n{"global_step": 2')
    reader = MetricsReader(source)
    records, _ = reader.read_new()
    assert records == [first]
    assert reader.read_new() == ([], False)
    with source.open("a") as stream:
        stream.write(', "train_loss": 0.8}\n')
    records, reset = reader.read_new()
    assert not reset
    assert records == [{"global_step": 2, "train_loss": .8}]
    assert reader.read_new() == ([], False)


def test_jsonl_truncation_and_atomic_replacement_reset_history(tmp_path):
    source = tmp_path / "logs.json.txt"
    source.write_text(json.dumps({"global_step": 200, "train_loss": 123.456}) + "\n")
    reader = MetricsReader(source)
    history = LossHistory()
    records, reset = reader.read_new()
    history.update(records, reset=reset)
    source.write_text('{"step": 1, "train_loss": 1}\n')
    records, reset = reader.read_new()
    assert reset
    history.update(records, reset=reset)
    np.testing.assert_array_equal(history.points("train_loss")[0], [1])
    replacement = tmp_path / "replacement"
    replacement.write_text('{"step": 9, "train_loss": 2}\n')
    replacement.replace(source)
    records, reset = reader.read_new()
    assert reset
    history.update(records, reset=reset)
    np.testing.assert_array_equal(history.points("train_loss")[0], [9])


def test_json_snapshot_waits_for_complete_rewrite(tmp_path):
    source = tmp_path / "loss.json"
    source.write_text(json.dumps([{"step": 1, "train_loss": .7}]))
    reader = MetricsReader(source)
    records, reset = reader.read_new()
    assert reset and records == [{"step": 1, "train_loss": .7}]
    assert reader.read_new() == ([], False)
    source.write_text('[{"step": 2, "train_loss":')
    assert reader.read_new() == ([], False)
    source.write_text(json.dumps({"records": [{"step": 2, "train_loss": .5}]}))
    records, reset = reader.read_new()
    assert reset and records == [{"step": 2, "train_loss": .5}]


def test_history_excludes_unmeasured_losses_and_preserves_other_components():
    history = LossHistory()
    history.update([
        {"global_step": 1, "objective": "condition_warmup",
         "train_token_mse": 0., "train_token_mse_measured": False,
         "train_phase_ce": .6, "train_alpha_mse": .1, "train_loss": .7,
         "train_token_region_observation_count": 0, "train_context_patch_mse": 0.,
         "train_open_token_mse": 0., "train_open_token_count": 0},
        {"global_step": 2, "objective": "token_mse",
         "train_token_mse": .04, "train_token_mse_measured": True,
         "train_token_region_observation_count": 2, "train_context_patch_mse": .02,
         "train_open_token_mse": .04, "train_open_token_count": 8,
         "train_closed_token_mse": 0., "train_closed_token_count": 0},
        {"global_step": 2, "train_token_mse": .03},
    ])
    np.testing.assert_array_equal(history.points("train_token_mse")[0], [2])
    np.testing.assert_allclose(history.points("train_token_mse")[1], [.03])
    np.testing.assert_allclose(history.points("train_phase_ce")[1], [.6])
    np.testing.assert_allclose(history.points("train_context_patch_mse")[1], [.02])
    assert not len(history.points("train_closed_token_mse")[0])
    history.update([{"global_step": 3, "metric_scope": "phases_1_2",
                     "train_token_mse": .02}])
    np.testing.assert_array_equal(history.points("train_token_mse")[0], [3])


def test_loss_plot_overwrites_png_and_leaves_no_partial_file(tmp_path):
    history = LossHistory()
    history.update([{"global_step": 1, "epoch": 1, "train_token_mse": .04,
                     "train_phase_ce": .3, "train_alpha_mse": .02, "train_loss": .4}])
    output = tmp_path / "live.png"
    assert save_loss_plot(history, output, smooth_window=2, max_points=20) == output
    before = output.read_bytes()
    history.update([{"global_step": 2, "epoch": 1, "train_token_mse": .03,
                     "train_phase_ce": .2, "train_alpha_mse": .01, "train_loss": .3}])
    save_loss_plot(history, output, smooth_window=2, max_points=20)
    assert output.read_bytes() != before
    assert list(tmp_path.glob("*.png")) == [output]
    with Image.open(output) as image:
        assert image.format == "PNG" and image.width > 1000 and image.height > 800


def test_condition_validation_preserves_auxiliary_losses_but_hides_unmeasured_tokens(tmp_path):
    history = LossHistory()
    history.update([
        {"global_step": 0, "objective": "condition_warmup", "initial_val_mse": .9,
         "initial_val_token_mse": 0., "initial_val_phase_ce": .7,
         "initial_val_alpha_mse": .1, "initial_val_intensity_loss": .1,
         "val_token_mse_measured": False},
        {"global_step": 2, "objective": "condition_warmup", "val_mse": .6,
         "val_token_mse": 0., "val_closed_token_mse": 0., "val_phase_ce": .4,
         "val_alpha_mse": .1, "val_intensity_loss": .1,
         "val_token_mse_measured": False},
        # Explicitly measured warmup diagnostics remain available for old runs.
        {"global_step": 3, "objective": "condition_warmup", "val_mse": .04,
         "val_token_mse_measured": True},
        {"global_step": 4, "objective": "token_mse", "val_mse": .03},
    ])
    assert not len(history.points("initial_val_token_mse")[0])
    assert not len(history.points("val_closed_token_mse")[0])
    np.testing.assert_allclose(history.points("initial_val_condition_loss")[1], [.9])
    np.testing.assert_allclose(history.points("val_condition_loss")[1], [.6])
    np.testing.assert_allclose(history.points("initial_val_phase_ce")[1], [.7])
    np.testing.assert_allclose(history.points("val_phase_ce")[1], [.4])
    np.testing.assert_allclose(history.points("val_alpha_mse")[1], [.1])
    np.testing.assert_allclose(history.points("val_intensity_loss")[1], [.1])
    np.testing.assert_array_equal(history.points("val_token_mse")[0], [3, 4])
    np.testing.assert_allclose(history.points("val_token_mse")[1], [.04, .03])
    save_loss_plot(history, tmp_path / "stages.png")


def test_cli_uses_cpu_and_accepts_run_directory_with_loss_json(tmp_path):
    (tmp_path / "loss.json").write_text(json.dumps([
        {"global_step": 1, "train_loss": .8},
        {"global_step": 2, "train_loss": .6},
    ]))
    script = Path(__file__).resolve().parents[1] / "diffusion_policy/scripts/plot_render_net_losses.py"
    code = (
        "import os,runpy,sys; "
        "sys.argv=[sys.argv[1],sys.argv[2]]; "
        "runpy.run_path(sys.argv[0],run_name='__main__'); "
        "assert os.environ['CUDA_VISIBLE_DEVICES']==''; "
        "assert 'torch' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code, str(script), str(tmp_path)],
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "training_losses.png").is_file()
