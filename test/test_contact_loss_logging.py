"""Contact metrics must select measured anchors and preserve the training objective."""

from pathlib import Path
import copy
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import numpy as np
import pytest
import torch
from matplotlib.figure import Figure

from diffusion_policy.common.render_net_training_logger import MetricHistory, RenderNetTrainingLogger, save_training_plot
from test_render_net_denoiser_distillation import make_workspace


def test_contact_dit_statistics_use_current_anchor_without_changing_loss_or_gradients():
    workspace, batch = make_workspace()
    workspace.teacher_policy.n_obs_steps = 1
    # First entry is the anchor; later labels must not influence selection.
    batch["obs"]["contact_phase"] = torch.tensor([[2, 0], [0, 2], [1, 2]])
    rng = torch.random.get_rng_state()
    loss = workspace._batch_loss(batch)
    loss.backward()
    gradient = workspace.model.offset.weight.grad.clone()
    teacher, student = workspace.teacher_policy.model.calls[-2:]
    expected = (student["output"][0] - teacher["output"][0]).square().mean()
    assert workspace.loss_metrics["contact_samples"] == 1
    assert workspace.loss_metrics["contact_mse"] == pytest.approx(expected.item())
    assert len(workspace.teacher_policy.model.calls) == 2
    del batch["obs"]["contact_phase"]
    workspace.model.zero_grad()
    torch.random.set_rng_state(rng)
    unchanged_loss = workspace._batch_loss(batch)
    unchanged_loss.backward()
    torch.testing.assert_close(loss, unchanged_loss, rtol=0, atol=0)
    torch.testing.assert_close(gradient, workspace.model.offset.weight.grad, rtol=0, atol=0)


def test_contact_validation_aggregates_samples_not_batch_means_and_skips_free_batches():
    workspace, batch = make_workspace()
    batches = []
    for phases in ([0, 2, 1], [2, 2, 0], [0, 0, 1]):
        item = copy.deepcopy(batch)
        item["obs"]["contact_phase"] = torch.tensor(phases)[:, None]
        batches.append(item)
    workspace._validate(torch.utils.data.DataLoader(batches, batch_size=None))
    calls = workspace.teacher_policy.model.calls
    expected = []
    for index, item in enumerate(batches):
        teacher, student = calls[2 * index:2 * index + 2]
        per_sample = (student["output"] - teacher["output"]).square().flatten(1).mean(1)
        expected.extend(per_sample[item["obs"]["contact_phase"][:, 0] == 2].tolist())
    assert workspace.validation_metrics["contact_samples"] == 3
    assert workspace.validation_metrics["contact_mse"] == pytest.approx(np.mean(expected))
    assert workspace.loss_metrics["contact_samples"] == 0
    assert "contact_mse" not in workspace.loss_metrics


def test_response_statistics_normalize_over_contact_weights_across_batches():
    workspace, _ = make_workspace()
    workspace.cfg.distillation.update(objective="token_mse", response_loss_weight=2., response_noise_floor=0.)
    workspace.teacher_policy.n_obs_steps = 1
    # Contact squared errors are [1, 9] and [4, 16], with weights [1, 0] and [0, 3].
    student = torch.tensor([[[1.], [3.]], [[100.], [100.]], [[2.], [4.]]])
    target = torch.zeros_like(student)
    response = torch.tensor([[1., 0.], [1000., 1000.], [0., 3.]])
    phases = torch.tensor([[2], [0], [2]])
    totals = np.zeros(len(workspace.contact_total_keys))
    for ids in (slice(0, 2), slice(2, 3)):
        metrics = workspace._contact_loss_metrics({"contact_phase": phases[ids]}, student[ids], target[ids], response=response[ids])
        totals += [metrics[key] for key in workspace.contact_total_keys]
    mean = workspace._contact_means(totals)
    assert mean["contact_samples"] == 2
    assert mean["contact_token_mse"] == pytest.approx(7.5)
    assert mean["contact_response_mse"] == pytest.approx(49 / 4)
    assert mean["contact_mse"] == pytest.approx(7.5 + 2 * 49 / 4)


def test_two_curve_plots_connect_initial_points_and_export_json(tmp_path, monkeypatch):
    records = [
        {"global_step": 0, "epoch": 0, "objective": "denoiser_mse", "initial_train_mse": .8,
         "initial_train_samples": 10, "initial_train_contact_mse": .7, "initial_train_contact_samples": 4},
        {"global_step": 0, "epoch": 0, "initial_val_mse": .9,
         "initial_val_contact_mse": .85, "initial_val_contact_samples": 3},
        {"global_step": 1, "epoch": 1, "train_mse": 25., "train_contact_mse": 27.},
        {"global_step": 2, "epoch": 1, "epoch_train_mse": .6,
         "epoch_train_contact_mse": .5, "epoch_train_contact_samples": 4},
        {"global_step": 2, "epoch": 1, "val_mse": .65, "val_contact_mse": .55, "val_contact_samples": 3},
    ]
    with RenderNetTrainingLogger(tmp_path, enable_plots=False) as logger:
        for record in records:
            logger.log(record)
    saved = json.loads((tmp_path / "log.json").read_text())
    assert saved["records"] == [records[i] for i in (0, 1, 3, 4)]
    # Resume does not lose the initial measurements or duplicate history.
    with RenderNetTrainingLogger(tmp_path, enable_plots=False):
        pass
    assert json.loads((tmp_path / "log.json").read_text()) == saved
    from_json = MetricHistory.from_json(tmp_path / "log.json")
    np.testing.assert_allclose(from_json.train_curve("epoch_train_contact_mse", "initial_train_contact_mse")[1], [.7, .5])
    captured = {}
    original = Figure.savefig

    def capture(figure, path, *args, **kwargs):
        captured[Path(path).name] = [{line.get_label(): (list(line.get_xdata()), list(line.get_ydata()))
                                    for line in axis.lines} for axis in figure.axes if axis.get_visible()]
        return original(figure, path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    save_training_plot(MetricHistory.from_json(tmp_path / "logs.json.txt"), tmp_path / "training_metrics.png")
    assert captured["training_metrics.tmp.png"][0] == {
        "Train loss": ([0, 2], [.8, .6]), "Validation loss": ([0, 2], [.9, .65])}
    assert captured["contact_loss.tmp.png"][0] == {
        "Train loss": ([0, 2], [.7, .5]), "Validation loss": ([0, 2], [.85, .55])}
    assert (tmp_path / "contact_loss.png").is_file()


def test_unmeasured_contact_values_are_not_plotted_as_zero(tmp_path):
    history = MetricHistory()
    history.record({"epoch": 1, "global_step": 2, "epoch_train_contact_mse": 0.,
                    "epoch_train_contact_samples": 0, "val_contact_mse": 0., "val_contact_samples": 0})
    save_training_plot(history, tmp_path / "training_metrics.png")
    assert not (tmp_path / "contact_loss.png").exists()


def test_stage_boundary_zero_placeholders_do_not_replace_initial_losses():
    history = MetricHistory()
    history.record({"global_step": 0, "objective": "adapter_warmup", "initial_train_mse": .9,
                    "initial_train_adapter_mse": .9, "initial_train_condition_loss": 0.})
    history.record({"global_step": 3, "objective": "adapter_warmup", "epoch_train_mse": .8,
                    "epoch_train_adapter_mse": .8, "epoch_train_condition_loss": 0.})
    history.record({"global_step": 3, "objective": "condition_warmup", "initial_train_mse": .7,
                    "initial_train_adapter_mse": 0., "initial_train_condition_loss": .7})
    history.record({"global_step": 6, "objective": "condition_warmup", "epoch_train_mse": .6,
                    "epoch_train_adapter_mse": 0., "epoch_train_condition_loss": .6})
    history.record({"global_step": 6, "objective": "token_mse", "initial_train_mse": .5,
                    "initial_train_adapter_mse": 0., "initial_train_condition_loss": 0.})
    assert history.train_curve("epoch_train_adapter_mse", "initial_train_adapter_mse")[1].tolist() == [.9, .8]
    assert history.train_curve("epoch_train_condition_loss", "initial_train_condition_loss")[1].tolist() == [.7, .6]
