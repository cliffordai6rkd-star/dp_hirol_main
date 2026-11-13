import json

import numpy as np
from PIL import Image

from diffusion_policy.common.render_net_training_logger import (
    MetricHistory, RenderNetTrainingLogger, save_training_plot,
)


def test_condition_validation_and_initial_losses_never_enter_token_curves():
    history = MetricHistory()
    history.record({"global_step": 0, "epoch": 0, "objective": "condition_warmup",
                    "initial_train_mse": .8, "initial_val_mse": .9,
                    "train_token_mse_measured": False, "val_token_mse_measured": False})
    history.record({"global_step": 2, "epoch": 1, "objective": "condition_warmup",
                    "val_mse": .7, "best_val_mse": .7, "val_token_mse_measured": False,
                    "val_token_region_observation_count": 0})
    # Explicitly measured legacy warmup token diagnostics remain supported.
    history.record({"global_step": 3, "epoch": 2, "objective": "condition_warmup",
                    "val_mse": .04, "val_token_mse_measured": True})
    history.record({"global_step": 4, "epoch": 3, "objective": "token_mse",
                    "val_mse": .03, "best_val_mse": .03})
    np.testing.assert_allclose(history.points("train_condition_loss")[1], [.8])
    np.testing.assert_allclose(history.points("val_condition_loss")[1], [.9, .7])
    assert not len(history.points("initial_train_mse")[0])
    np.testing.assert_array_equal(history.points("val_mse")[0], [3, 4])
    np.testing.assert_allclose(history.points("val_mse")[1], [.04, .03])
    np.testing.assert_allclose(history.points("epoch_best_val_mse")[1], [.04, .03])


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


def test_disabled_plots_only_appends_metrics_and_allows_offline_plotting(tmp_path, monkeypatch):
    previous = {"global_step": 0, "epoch": 0, "initial_train_mse": .8}
    log_path = tmp_path / "logs.json.txt"
    log_path.write_text(json.dumps(previous) + "\n")
    records = [
        {"global_step": 1, "epoch": 1, "objective": "token_mse",
         "metric_scope": "phases_1_2", "train_mse": .4, "train_loss": .7,
         "train_phase_ce": .2, "train_alpha_mse": .1, "grad_norm": .5,
         "train_token_region_observation_count": 2, "lr": .0002},
        {"global_step": 2, "epoch": 1, "train_mse": .3, "train_loss": .5,
         "samples_per_second": 40., "gpu_peak_allocated_mib": 2048.},
        {"global_step": 2, "epoch": 1, "epoch_train_mse": .35,
         "val_mse": .32, "best_val_mse": .32},
    ]

    def unexpected_plot_work(*args, **kwargs):
        raise AssertionError("Disabled plots must not load, record, or render history")

    with monkeypatch.context() as patch:
        patch.setattr(MetricHistory, "from_json", unexpected_plot_work)
        patch.setattr(MetricHistory, "record", unexpected_plot_work)
        patch.setattr("diffusion_policy.common.render_net_training_logger.save_training_plot",
                      unexpected_plot_work)
        with RenderNetTrainingLogger(tmp_path, plot_every_steps=1, enable_plots=False) as logger:
            for record in records:
                logger.log(record)
            assert logger.save_plot() is None
            assert not logger.history.series
    assert not (tmp_path / "training_metrics.png").exists()
    assert [json.loads(line) for line in log_path.read_text().splitlines()] == [previous, *records]
    history = MetricHistory.from_json(log_path)
    np.testing.assert_allclose(history.points("train_mse")[1], [.4, .3])
    np.testing.assert_allclose(history.points("val_mse")[1], [.32])
    assert save_training_plot(history, tmp_path / "offline.png") == tmp_path / "offline.png"
    assert (tmp_path / "offline.png").is_file()


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


def test_condition_warmup_never_plots_unmeasured_token_mse(tmp_path):
    with RenderNetTrainingLogger(tmp_path, plot_every_steps=0) as logger:
        logger.log({"global_step": 1, "epoch": 1, "objective": "condition_warmup",
                    "train_mse": .7, "train_token_mse": 0., "train_token_mse_measured": False,
                    "train_open_token_mse": 0., "train_closed_token_mse": 0.,
                    "train_phase_ce": .5, "train_alpha_mse": .2, "train_loss": .7})
        logger.log({"global_step": 1, "epoch": 1, "objective": "condition_warmup",
                    "epoch_train_mse": .7, "epoch_train_token_mse_measured": False})
        assert logger.history.objective == "token_mse"
        assert not len(logger.history.points("train_mse")[1])
        assert not len(logger.history.points("train_token_mse")[1])
        assert not len(logger.history.points("train_open_token_mse")[1])
        assert not len(logger.history.points("epoch_train_mse")[1])
        logger.log({"global_step": 2, "epoch": 2, "objective": "token_mse",
                    "train_mse": .3, "train_token_mse": .3, "train_token_mse_measured": True})
        logger.save_plot()
    history = MetricHistory.from_json(tmp_path / "logs.json.txt")
    np.testing.assert_allclose(history.points("train_mse")[1], [.3])
    np.testing.assert_allclose(history.points("train_token_mse")[1], [.3])
    np.testing.assert_allclose(history.points("train_condition_loss")[1], [.7])
    np.testing.assert_allclose(history.points("train_phase_ce")[1], [.5])


def test_metric_scope_change_preserves_json_but_resets_incomparable_curves(tmp_path):
    with RenderNetTrainingLogger(tmp_path, plot_every_steps=0) as logger:
        logger.log({"global_step": 0, "epoch": 0,
                    "initial_train_mse": .01, "initial_val_mse": .012})
        logger.log({"global_step": 1, "epoch": 1, "train_mse": .009,
                    "train_token_mse": .009, "train_mask_mse": .003,
                    "train_loss": .2, "train_phase_ce": .15, "train_alpha_mse": .04,
                    "grad_norm": 1., "lr": .0002})
        logger.log({"global_step": 1, "epoch": 1, "val_mse": .011,
                    "best_val_mse": .011, "epoch_train_mse": .009})
        logger.log({"global_step": 1, "epoch": 1, "metric_scope": "phases_1_2",
                    "initial_train_mse": .04, "initial_val_mse": .05})
        logger.log({"global_step": 2, "epoch": 2, "metric_scope": "phases_1_2",
                    "train_mse": .03, "train_token_mse": .03, "train_mask_mse": .02,
                    "train_loss": .1, "train_phase_ce": .05, "train_alpha_mse": .02,
                    "train_token_region_observation_count": 2,
                    "grad_norm": .5, "lr": .0002})
        # Records without an explicit scope retain the active scope.
        logger.log({"global_step": 2, "epoch": 2, "val_mse": .04,
                    "best_val_mse": .04, "epoch_train_mse": .03})
        logger.save_plot()

    records = [json.loads(line) for line in (tmp_path / "logs.json.txt").read_text().splitlines()]
    assert records[0]["initial_val_mse"] == .012
    assert records[2]["val_mse"] == .011
    assert records[3]["metric_scope"] == "phases_1_2"
    history = MetricHistory.from_json(tmp_path / "logs.json.txt")
    assert history.metric_scope == "phases_1_2"
    np.testing.assert_allclose(history.points("initial_train_mse")[1], [.04])
    np.testing.assert_allclose(history.points("val_mse")[1], [.05, .04])
    np.testing.assert_allclose(history.points("epoch_best_val_mse")[1], [.05, .04])
    np.testing.assert_allclose(history.points("train_mse")[1], [.03])
    np.testing.assert_allclose(history.points("train_mask_mse")[1], [.02])
    np.testing.assert_allclose(history.points("train_loss")[1], [.1])
    np.testing.assert_allclose(history.points("train_phase_ce")[1], [.15, .05])
    np.testing.assert_allclose(history.points("train_alpha_mse")[1], [.04, .02])
    np.testing.assert_allclose(history.points("grad_norm")[1], [1., .5])


def test_free_only_batch_does_not_plot_zero_token_or_region_supervision(tmp_path):
    with RenderNetTrainingLogger(tmp_path, plot_every_steps=0) as logger:
        logger.log({"global_step": 1, "epoch": 2, "metric_scope": "phases_1_2",
                    "objective": "token_mse", "train_token_mse_measured": False,
                    "train_token_region_observation_count": 0,
                    "train_mse": 0., "train_token_mse": 0., "train_mask_mse": 0.,
                    "train_context_patch_mse": 0., "train_intensity_loss": 0.,
                    "train_phase_ce": .3, "train_alpha_mse": .1, "train_loss": .4})
        logger.log({"global_step": 2, "epoch": 2, "metric_scope": "phases_1_2",
                    "objective": "token_mse", "train_token_mse_measured": True,
                    "train_token_region_observation_count": 1,
                    "train_mse": .2, "train_token_mse": .2, "train_mask_mse": .1,
                    "train_context_patch_mse": .05, "train_intensity_loss": .05,
                    "train_phase_ce": .2, "train_alpha_mse": .1, "train_loss": .55})
    history = MetricHistory.from_json(tmp_path / "logs.json.txt")
    assert history.objective == "token_mse"
    assert not len(history.points("train_condition_loss")[1])
    np.testing.assert_allclose(history.points("train_mse")[1], [.2])
    np.testing.assert_allclose(history.points("train_token_mse")[1], [.2])
    np.testing.assert_allclose(history.points("train_mask_mse")[1], [.1])
    np.testing.assert_allclose(history.points("train_context_patch_mse")[1], [.05])
    np.testing.assert_allclose(history.points("train_intensity_loss")[1], [.05])
    np.testing.assert_allclose(history.points("train_phase_ce")[1], [.3, .2])
    np.testing.assert_allclose(history.points("train_loss")[1], [.4, .55])


def test_open_and_closed_token_diagnostics_skip_only_explicitly_unmeasured_windows():
    history = MetricHistory()
    history.record({"global_step": 1, "train_open_token_mse": .2,
                    "train_closed_token_mse": 0., "train_open_token_count": 4,
                    "train_closed_token_count": 0})
    history.record({"global_step": 2, "train_open_token_mse": 0.,
                    "train_closed_token_mse": .1, "train_open_token_count": 0,
                    "train_closed_token_count": 4})
    # Older logs omitted counts; their diagnostics must remain visible.
    history.record({"global_step": 3, "train_open_token_mse": .3,
                    "train_closed_token_mse": .05})
    np.testing.assert_array_equal(history.points("train_open_token_mse")[0], [1, 3])
    np.testing.assert_allclose(history.points("train_open_token_mse")[1], [.2, .3])
    np.testing.assert_array_equal(history.points("train_closed_token_mse")[0], [2, 3])
    np.testing.assert_allclose(history.points("train_closed_token_mse")[1], [.1, .05])
