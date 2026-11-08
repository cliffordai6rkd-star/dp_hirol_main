"""Temporal alignment, offline phase labels, and training/inference integration."""

import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from data_process.libero_wrench_dataset import LiberoWrenchDataset, prediction_offsets, wrench_phase_labels
from model.cad_wm import CADWorldModel
from model.wrench_predictor import WrenchPredictor
from train.trainer.cad_wm_train import CADWMTrainer


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config(tmp_path):
    path = Path(__file__).resolve().parents[1] / "config/bench_train_config/libero_drawer_wrench_h5.yaml"
    config = yaml.safe_load(path.read_text())
    source = tmp_path / "demos.hdf5"
    with h5py.File(source, "w") as file:
        demos = file.create_group("data")
        demos.attrs["env_args"] = json.dumps({"env_kwargs": {"control_freq": 20}})
        for episode in range(2):
            obs = demos.create_group(f"demo_{episode}").create_group("obs")
            rows = np.arange(40, dtype=np.float32)[:, None]
            obs.create_dataset("joint_states", data=np.broadcast_to(rows + episode * 1000, (40, 7)))
            for name in ("dq", "tau"):
                values = np.broadcast_to(rows + episode * 1000, (40, 24)).copy()
                values[:, 7:] = 999999
                dataset = obs.create_dataset(name, data=values)
                if name == "tau":
                    dataset.attrs["dt"] = 0.05
            wrench = np.zeros((40, 6), dtype=np.float32)
            if episode == 0:
                wrench[:, 0] = rows[:, 0] ** 2
            obs.create_dataset("wrenches", data=wrench)
    config["dataloader"].update(root=str(source), state_history_horizon=4)
    config["model"].update(hidden_dim=8, lstm_layers=2, flow_layers=1,
                           free_dynamics_hidden_dim=8, flow_inference_steps=2,
                           attention_heads=2, ffn_multiplier=2, dropout=0.0)
    config["train"].update(device="cpu", output_dir=str(tmp_path / "run"), batch_size=2,
                           validation_enabled=True, val_ratio=0.5, val_episode_indices=[1],
                           max_optimizer_steps=1, checkpoint_every_steps=1,
                           scheduler={"name": "none"}, amp={"enabled": False})
    return config


@pytest.mark.parametrize("split_mode", ["episode", "sample", "purged_temporal", "purged_kfold"])
def test_no_validation_uses_all_demos_for_normalization_and_sampling(config, split_mode):
    config["train"].update(validation_enabled=False, split_mode=split_mode)
    trainer = CADWMTrainer(config)
    trainer.setup()
    assert trainer.val_loader is None
    assert trainer.loader.dataset is trainer.dataset
    assert len(trainer.loader.dataset) == trainer.loader.sampler.num_samples == 64
    torch.testing.assert_close(trainer.dataset.normalizer.stats["q"]["mean"], torch.full((7,), 519.5))
    assert sum(trainer.checkpoint_metadata["contact_sampling"]["source_counts"]) == 64
    assert trainer.validate_one_epoch(0) is None


def test_checkpoint_inference_covers_all_windows_in_physical_units(config, monkeypatch):
    # Keep validation enabled to prove evaluation also includes held-out demos.
    config["train"]["checkpoint_eval"]["batch_size"] = 7
    trainer = CADWMTrainer(config)
    trainer.setup()
    seen = []

    def predict(conditions, *, source_noise):
        assert set(conditions) == {"q", "dq", "tau"}
        assert not trainer.ema.model.training
        raw_q = trainer.dataset.normalizer.gaussian_denormalize("q", conditions["q"])
        anchors = raw_q[:, -1, 0].round().long().tolist()
        for row, anchor in enumerate(anchors):
            # Demo 1 starts at raw row 40, with synthetic q starting at 1000.
            raw_index = anchor if anchor < 1000 else anchor - 1000 + 40
            generator = torch.Generator().manual_seed(trainer.checkpoint_eval_seed + raw_index)
            torch.testing.assert_close(source_noise[row], torch.randn(8, 6, generator=generator))
        seen.extend(anchors)
        return {"wrench_pred": torch.zeros_like(source_noise)}

    def unexpected_raw_prediction(*args, **kwargs):
        pytest.fail("checkpoint inference must use the saved EMA weights")

    monkeypatch.setattr(trainer.ema.model, "predict", predict)
    monkeypatch.setattr(trainer.model, "predict", unexpected_raw_prediction)
    trainer.ema.model.train()
    rng_before = torch.get_rng_state().clone()
    metrics = trainer.evaluate_checkpoint()
    assert trainer.ema.model.training and trainer.model.training
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert seen == list(range(32)) + list(range(1000, 1032))
    assert metrics["checkpoint_eval_samples"] == 64
    assert metrics["checkpoint_eval_elements"] == 64 * 8 * 6
    targets = torch.stack([trainer.dataset[index]["wrench_future_raw"] for index in range(64)])
    mean = trainer.dataset.normalizer.stats["wrench"]["mean"]
    error_squared = (mean - targets).double().square()
    assert metrics["full_dataset_wrench_mse"] == pytest.approx(error_squared.mean().item())
    assert metrics["full_dataset_force_mse"] == pytest.approx(error_squared[..., :3].mean().item())
    assert metrics["full_dataset_moment_mse"] == pytest.approx(error_squared[..., 3:].mean().item())
    assert trainer.evaluate_checkpoint() == metrics


def test_checkpoint_topk_uses_lowest_mse_and_resumes_from_latest(config, monkeypatch):
    config["train"].update(validation_enabled=False, top_k=2)
    trainer = CADWMTrainer(config)
    trainer.setup()
    scores = iter([3.0, 1.0, 2.0, 4.0])
    monkeypatch.setattr(trainer, "evaluate_checkpoint", lambda: {"full_dataset_wrench_mse": next(scores)})
    for step in range(1, 5):
        trainer.global_step = step
        trainer.save_step_checkpoint(0, {"val_loss": None})
    assert sorted(path.name for path in trainer.ckpt_dir.glob("step_*.pt")) == [
        "step_00000002.pt", "step_00000003.pt"]
    assert [record["score"] for record in trainer.best_checkpoints] == [1.0, 2.0]
    latest = torch.load(trainer.ckpt_dir / "latest.pt", weights_only=False)
    assert latest["global_step"] == 4 and latest["monitor_score"] == 4.0
    assert latest["monitor_key"] == "full_dataset_wrench_mse"
    assert [record["score"] for record in latest["trainer_state"]["best_checkpoints"]] == [1.0, 2.0]
    config["train"]["resume_from"] = str(trainer.ckpt_dir / "latest.pt")
    resumed = CADWMTrainer(config)
    resumed.setup()
    assert resumed.global_step == 4
    assert [record["score"] for record in resumed.best_checkpoints] == [1.0, 2.0]
    monkeypatch.setattr(resumed, "evaluate_checkpoint", lambda: {"full_dataset_wrench_mse": 0.5})
    resumed.global_step = 5
    resumed.save_step_checkpoint(1, {})
    assert [record["global_step"] for record in resumed.best_checkpoints] == [5, 2]
    assert not (trainer.ckpt_dir / "step_00000003.pt").exists()


@pytest.mark.parametrize("step_based", [True, False])
def test_no_validation_training_reuses_checkpoint_and_epoch_evaluation(config, monkeypatch, step_based):
    config["train"].update(validation_enabled=False, batch_size=32, num_epochs=2, val_every=1,
                           early_stopping={"enabled": True, "patience": 1},
                           early_stopping_monitor_key="val_loss")
    if step_based:
        config["train"].update(max_optimizer_steps=3, checkpoint_every_steps=2)
    else:
        config["train"].pop("max_optimizer_steps")
        config["train"]["checkpoint_every_steps"] = 0
    trainer = CADWMTrainer(config)
    evaluations = []
    evaluate = trainer._evaluate_full_dataset

    def checkpoint_metrics():
        evaluations.append(trainer.global_step)
        return evaluate()

    def unexpected_validation(*args):
        pytest.fail("validation must not run when validation_enabled=false")

    monkeypatch.setattr(trainer, "_evaluate_full_dataset", checkpoint_metrics)
    monkeypatch.setattr(trainer, "validate_one_epoch", unexpected_validation)
    monkeypatch.setattr(trainer, "save_loss_plot", lambda: None)
    summary = trainer.train()
    assert evaluations == ([2, 3] if step_based else [2, 4])
    assert trainer.val_loader is None and not trainer.early_stopping_enabled
    assert summary["last_val_loss"] is None and not summary["validation_enabled"]
    latest = torch.load(trainer.ckpt_dir / "latest.pt", weights_only=False)
    assert latest["metrics"]["checkpoint_eval_samples"] == 64
    assert np.isfinite(latest["monitor_score"])


def test_full_mse_cadence_plot_updates_and_histories_survive_resume(config, monkeypatch):
    config["train"].update(validation_enabled=False, batch_size=32, num_epochs=4,
                           max_optimizer_steps=8, checkpoint_every_steps=6,
                           val_every=2, plot_every_steps=3)
    trainer = CADWMTrainer(config)
    evaluations, plot_updates = [], []
    evaluate = trainer._evaluate_full_dataset

    def full_evaluation():
        evaluations.append(trainer.global_step)
        return evaluate()

    monkeypatch.setattr(trainer, "_evaluate_full_dataset", full_evaluation)
    monkeypatch.setattr(trainer, "save_loss_plot", lambda: plot_updates.append(
        (trainer.global_step, len(trainer.train_step_history), len(trainer.inference_history))))
    summary = trainer.train()
    # Epoch 2, the scheduled step-6 save, then epoch 4/final save share step 8.
    assert evaluations == [4, 6, 8]
    assert [item["global_step"] for item in trainer.inference_history] == [4, 6, 8]
    assert [item["global_step"] for item in trainer.train_step_history] == list(range(1, 9))
    assert [item.get("full_dataset_wrench_mse") is not None for item in trainer.loss_history] == [
        False, True, False, True]
    assert (3, 3, 0) in plot_updates
    assert (4, 4, 1) in plot_updates and (8, 8, 3) in plot_updates
    assert summary["last_inference_metrics"]["global_step"] == 8
    config["train"]["resume_from"] = str(trainer.ckpt_dir / "latest.pt")
    resumed = CADWMTrainer(config)
    resumed.setup()
    assert resumed.train_step_history == trainer.train_step_history
    assert resumed.inference_history == trainer.inference_history


def test_grouped_h5_slices_alignment_and_episode_boundaries(config):
    dataset = LiberoWrenchDataset(config)
    assert len(dataset.episodes) == 2 and len(dataset) == 64
    first = dataset[0]
    assert first["q"].shape == (4, 7) and torch.count_nonzero(first["q"]) == 0
    torch.testing.assert_close(first["wrench_future"][:, 0], torch.arange(1, 9).float().square())
    torch.testing.assert_close(first["future_time"], torch.arange(1, 9) * 0.05)
    assert torch.equal(first["wrench_future"], dataset.high_tensors["wrench"][1:9])
    assert torch.equal(first["contact_future"], dataset.contact[1:9])
    for key in ("q", "dq", "tau"):
        torch.testing.assert_close(dataset[3][key][:, 0], torch.arange(4).float())
        assert dataset[3][key].max() < 999999
    assert torch.equal(dataset[31]["wrench_future"], dataset.high_tensors["wrench"][32:40])
    assert torch.equal(dataset[31]["contact_future"], dataset.contact[32:40])
    assert torch.all(dataset[32]["q"] == 1000)
    assert torch.count_nonzero(dataset.contact[40:]) == 0
    assert "wrench" not in first and "action" not in first


def test_offline_equal_threshold_debounce_and_strict_boundaries(config):
    config["contact_gate"]["precontact_seconds"] = 0
    signal = [9] * 4 + [8] + [9] * 5 + [8] + [7] * 4 + [8] + [7] * 5
    wrench = torch.zeros(len(signal), 6)
    wrench[:, 0] = torch.tensor(signal)
    wrench[:, 3:] = 10000
    assert wrench_phase_labels(wrench, config, 20).tolist() == [0] * 5 + [2] * 11 + [0] * 5
    config["contact_gate"]["precontact_seconds"] = 0.2
    assert wrench_phase_labels(wrench, config, 20).tolist() == [0] + [1] * 4 + [2] * 11 + [0] * 5


@pytest.mark.parametrize("signal,expected", [
    ([9] * 4, [0] * 4),
    ([9] * 5, [2] * 5),
    ([9] * 5 + [7] * 4, [2] * 9),
    ([9] * 5 + [7] * 5, [2] * 5 + [0] * 5),
    ([9] * 5 + [7] * 4 + [9] * 2 + [7] * 5, [2] * 11 + [0] * 5),
    ([9] * 5 + [7] * 5 + [9] * 4, [2] * 5 + [0] * 9),
])
def test_offline_contact_requires_complete_windows_and_preserves_short_dips(config, signal, expected):
    config["contact_gate"]["precontact_seconds"] = 0
    wrench = torch.zeros(len(signal), 6)
    wrench[:, 0] = torch.tensor(signal)
    assert wrench_phase_labels(wrench, config, 20).tolist() == expected


def test_offline_contact_backfills_different_entry_and_exit_lengths(config):
    config["contact_gate"].update(contact_enter_frames=3, contact_exit_frames=2, precontact_seconds=0)
    wrench = torch.zeros(6, 6)
    wrench[:, 0] = torch.tensor([9, 9, 9, 8, 7, 7])
    assert wrench_phase_labels(wrench, config, 20).tolist() == [2, 2, 2, 2, 0, 0]


def test_normalizer_uses_only_training_demo(config):
    dataset = LiberoWrenchDataset(config)
    dataset.fit_normalizer(range(32))
    torch.testing.assert_close(dataset.normalizer.stats["q"]["mean"], torch.full((7,), 19.5))
    assert torch.isfinite(dataset[0]["wrench_future"]).all()
    bad = copy.deepcopy(config)
    bad["dataloader"]["high_fps"] = 25
    with pytest.raises(ValueError, match="control_freq"):
        LiberoWrenchDataset(bad)


def test_reject_output_rate_different_from_native_input_rate(config):
    config["dataloader"]["prediction_step_s"] = 0.04
    with pytest.raises(ValueError, match="native rate"):
        LiberoWrenchDataset(config)
    with pytest.raises(ValueError, match="native rate"):
        CADWorldModel(config)


@pytest.mark.parametrize("fps", [20, 100])
def test_future_time_uses_configured_native_rate(config, fps):
    config["dataloader"]["high_fps"] = fps
    expected = torch.arange(1, 9, dtype=torch.float64) / fps
    torch.testing.assert_close(prediction_offsets(config), expected)


def test_no_target_or_future_action_inputs_and_gradients(config):
    dataset = LiberoWrenchDataset(config)
    batch = torch.utils.data.default_collate([dataset[0], dataset[1]])
    model = CADWorldModel(config).eval()
    source = torch.randn(2, 8, 6)
    first = model.predict(batch, source_noise=source)
    batch["wrench_future"].fill_(float("nan"))
    batch["contact_future"].fill_(-100)
    batch["action"] = torch.full((2, 8, 7), float("nan"))
    second = model.predict(batch, source_noise=source)
    torch.testing.assert_close(first["wrench_pred"], second["wrench_pred"])
    assert first["wrench_pred"].shape == (2, 8, 6)
    assert first["contact_logits"].shape == (2, 8, 3)
    out = model({key: value for key, value in batch.items() if key != "wrench_future"}
                | {"wrench_future": torch.randn(2, 8, 6)}, source_noise=source, flow_time=0.5)
    (out["flow_velocity_pred"].square().mean() + out["contact_logits"].square().mean()).backward()
    for module in (model.motion_lstm, model.torque_lstm, model.flow_output, model.contact_head,
                   model.flow_blocks[0].motion_cross_attn, model.flow_blocks[0].torque_cross_attn):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert model.motion_lstm.input_size == 14
    assert model.torque_lstm.input_size == 7
    assert model.motion_lstm.num_layers == model.torque_lstm.num_layers == 2
    assert out["motion_tokens"].shape == out["torque_tokens"].shape == (2, 4, 8)


def test_conditional_flow_matching_and_noise_integration(config):
    dataset = LiberoWrenchDataset(config)
    values = torch.utils.data.default_collate([dataset[0], dataset[1]])
    model = CADWorldModel(config).eval()
    source = torch.randn(2, 8, 6)
    out = model(values, source_noise=source, flow_time=0.25)
    torch.testing.assert_close(out["flow_velocity_target"], values["wrench_future"] - source)
    intermediate = 0.75 * source + 0.25 * values["wrench_future"]
    torch.testing.assert_close(out["wrench_pred"], intermediate + 0.75 * out["flow_velocity_pred"])
    different = model.predict(values, source_noise=source + 1)
    first = model.predict(values, source_noise=source)
    assert not torch.allclose(first["wrench_pred"], different["wrench_pred"])
    torch.testing.assert_close(first["contact_logits"], different["contact_logits"])
    # With a constant learned velocity, both ODE solvers must transport noise
    # from s=0 to s=1 exactly, independently of the number of steps.
    def constant(trajectory, time, encoded):
        return torch.full_like(trajectory, 2.0)
    model.flow_velocity = constant
    for solver in ("euler", "heun"):
        torch.testing.assert_close(model.predict(values, source_noise=source, steps=3,
                                                  solver=solver)["wrench_pred"], source + 2)


def test_next_auxiliary_only_updates_motion_lstm_and_two_layer_mlp(config):
    model = CADWorldModel(config)
    batch = {key: torch.randn(2, 4, 7, requires_grad=True) for key in ("q", "dq", "tau")}
    batch.update(free_dynamics_mask=torch.tensor([True, False]), importance_weight=torch.tensor([2.0, 1.0]))
    out = model.encode_conditions(batch, compute_free_dynamics=True)
    actual, count = CADWMTrainer.free_dynamics_loss(out, batch)
    expected = (out["free_tau_pred"][0] - batch["tau"][0, -1].detach()).square().mean()
    torch.testing.assert_close(actual, expected)
    assert count == 1
    actual.backward()
    assert sum(isinstance(layer, torch.nn.Linear) for layer in model.free_dynamics_head) == 2
    for module in (model.motion_lstm, model.free_dynamics_head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert batch["q"].grad.abs().sum() > 0 and batch["dq"].grad.abs().sum() > 0
    assert batch["tau"].grad is None
    for module in (model.torque_lstm, model.flow_output, model.flow_blocks):
        assert all(p.grad is None for p in module.parameters())
    model.zero_grad(set_to_none=True)
    batch["free_dynamics_mask"].zero_()
    empty, count = CADWMTrainer.free_dynamics_loss(
        model.encode_conditions(batch, compute_free_dynamics=True), batch)
    assert empty == 0 and count == 0 and torch.isfinite(empty)
    empty.backward()


def test_free_auxiliary_mask_excludes_padding_precontact_and_high_force(config):
    dataset = LiberoWrenchDataset(config)
    dataset.contact.zero_()
    dataset.high_tensors["wrench"].zero_()
    assert not dataset[0]["free_dynamics_mask"]
    assert dataset[3]["free_dynamics_mask"]
    dataset.contact[2] = 1
    assert not dataset[3]["free_dynamics_mask"]
    dataset.contact.zero_()
    dataset.high_tensors["wrench"][2, 0] = 9
    assert not dataset[3]["free_dynamics_mask"]
    assert not dataset[32]["free_dynamics_mask"]


def test_phase_sampler_emphasizes_training_windows_only(config):
    trainer = CADWMTrainer(config)
    trainer.dataset = trainer.build_dataset()
    trainer.dataset.contact[:40] = torch.tensor([0] * 12 + [1] * 12 + [2] * 16)
    train, validation = trainer.split_dataset_by_episode()
    sampler = trainer.build_train_sampler(train)
    assert len(train) == len(validation) == 32
    torch.testing.assert_close(sampler.weights, torch.tensor([1.] * 4 + [5.] * 28, dtype=torch.double))
    assert sampler.num_samples == 32 and sampler.replacement
    assert trainer.checkpoint_metadata["contact_sampling"]["source_counts"] == [4, 12, 16]
    assert not trainer.dataset.importance_weight_by_sample_index
    trainer.train_config["contact_sampling"]["importance_correction"] = True
    sampler = trainer.build_train_sampler(train)
    corrections = trainer.dataset.importance_weight_by_sample_index
    assert set(corrections) == set(range(32))
    values = torch.tensor([corrections[index] for index in range(32)], dtype=torch.double)
    torch.testing.assert_close((sampler.weights / sampler.weights.sum() * values).sum(), torch.tensor(1., dtype=torch.double))


def test_train_checkpoint_reload_and_dp_physical_units(config):
    trainer = CADWMTrainer(config)
    trainer.setup()
    trainer.loader = [next(iter(trainer.loader))]
    trainer.val_loader = [next(iter(trainer.val_loader))]
    assert np.isfinite(trainer.train_one_epoch(0)) and trainer.global_step == 1
    assert np.isfinite(trainer.validate_one_epoch(0))
    checkpoint = trainer.ckpt_dir / "step_00000001.pt"
    predictor = WrenchPredictor(checkpoint)
    noise = torch.randn(1, 8, 6)
    actual = predictor({key: torch.zeros(1, 1, 7) for key in ("q", "dq", "tau")}, source_noise=noise)
    assert actual["future_wrench"].shape == (1, 8, 6)
    assert actual["contact_probability"].shape == (1, 8, 3)
    torch.testing.assert_close(actual["future_time"], torch.arange(1, 9) * 0.05)
    sample = trainer.dataset[0]
    normalized = {key: sample[key][None] for key in ("q", "dq", "tau")}
    trainer.ema.model.eval()
    with torch.no_grad():
        expected = trainer.ema.model.predict(normalized, source_noise=noise)["wrench_pred"]
    expected = trainer.dataset.normalizer.gaussian_denormalize("wrench", expected)
    torch.testing.assert_close(actual["future_wrench"], expected)
    resumed_config = copy.deepcopy(config)
    resumed_config["train"]["resume_from"] = str(checkpoint)
    resumed_config["train"]["output_dir"] = str(checkpoint.parent.parent / "resumed")
    resumed = CADWMTrainer(resumed_config)
    resumed.setup()
    assert resumed.global_step == 1
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[key], value)
