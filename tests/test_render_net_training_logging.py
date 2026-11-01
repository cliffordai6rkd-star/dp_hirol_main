import json

import numpy as np
from PIL import Image

from diffusion_policy.common.render_net_training_logger import (
    MetricHistory, RenderNetTrainingLogger, save_training_plot,
)


def test_local_json_and_png_preserve_initial_and_validation_metrics(tmp_path):
    with RenderNetTrainingLogger(tmp_path, plot_every_steps=2) as logger:
        logger.log({"global_step": 0, "epoch": 0, "initial_train_mse": 1.0})
        logger.log({"global_step": 0, "epoch": 0, "initial_train_mse": 1.0, "initial_val_mse": 1.2})
        for step, mse in ((1, 0.9), (2, 0.7)):
            logger.log({"global_step": step, "epoch": 1, "train_mse": mse,
                        "train_mask_mse": 0.1, "train_loss": mse + 0.1, "grad_norm": 0.4,
                        "lr": 0.0003, "samples_per_second": 25.0, "gpu_allocated_mib": 1024.0})
        logger.log({"global_step": 2, "epoch": 1, "epoch_train_mse": 0.8})
        logger.log({"global_step": 2, "epoch": 1, "val_mse": 0.75, "best_val_mse": 0.75})
        logger.save_plot()

    records = [json.loads(line) for line in (tmp_path / "logs.json.txt").read_text().splitlines()]
    assert records[-1]["val_mse"] == 0.75
    history = MetricHistory.from_json(tmp_path / "logs.json.txt")
    np.testing.assert_array_equal(history.points("initial_train_mse")[1], [1.0])
    np.testing.assert_array_equal(history.points("val_mse")[0], [0, 2])
    np.testing.assert_allclose(history.points("epoch_val_mse")[1], [1.2, 0.75])
    np.testing.assert_allclose(history.points("epoch_train_mse")[1], [0.8])
    np.testing.assert_allclose(history.points("train_loss")[1], [1.0, 0.8])
    with Image.open(tmp_path / "training_metrics.png") as image:
        assert image.format == "PNG" and image.width > 1000 and image.height > 600
        assert np.asarray(image).std() > 10

    # Offline regeneration can read a live log with an unfinished last record.
    with (tmp_path / "logs.json.txt").open("a") as source:
        source.write('{"global_step": 3')
    reloaded = MetricHistory.from_json(tmp_path / "logs.json.txt")
    assert reloaded.step == 2 and reloaded.epoch == 1
    save_training_plot(reloaded, tmp_path / "rebuilt.png")
    assert (tmp_path / "rebuilt.png").is_file()


def test_epoch_only_plotting_keeps_step_metrics_without_rendering(tmp_path, monkeypatch):
    with RenderNetTrainingLogger(tmp_path, plot_every_steps=0) as logger:
        def unexpected_plot():
            raise AssertionError("Step-triggered plotting must be disabled")
        monkeypatch.setattr(logger, "save_plot", unexpected_plot)
        logger.log({"global_step": 1, "epoch": 1, "train_mse": 0.9})
        assert logger.history.points("train_mse")[1][0] == 0.9


def test_denoiser_objective_and_token_diagnostic_survive_offline_plot_rebuild(tmp_path):
    with RenderNetTrainingLogger(tmp_path, plot_every_steps=0) as logger:
        logger.log({"global_step": 1, "epoch": 1, "objective": "denoiser_mse",
                    "train_mse": 0.02, "train_denoiser_mse": 0.02, "train_token_mse": 1.4,
                    "train_demo_mse": 0.03, "train_mask_mse": 0.1, "train_loss": 0.12})
        logger.save_plot()
    history = MetricHistory.from_json(tmp_path / "logs.json.txt")
    assert history.objective == "denoiser_mse"
    np.testing.assert_allclose(history.points("train_mse")[1], [0.02])
    np.testing.assert_allclose(history.points("train_token_mse")[1], [1.4])
    np.testing.assert_allclose(history.points("train_demo_mse")[1], [0.03])
    save_training_plot(history, tmp_path / "rebuilt.png")
    assert (tmp_path / "rebuilt.png").is_file()
