from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from data_process.torque_target_filter import (
    filter_torque_target_dataset,
    torque_target_filter_config,
)
from train.tau_free_wrench_loss import TauFreeTorqueWrenchLoss
from train.trainer.tau_free_sequence_train import TauFreeSequenceTrainer
from train.trainer.tau_other_sequence_train import TauOtherTrainer


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "train_cfg"
    / "tau_free_sequence"
    / "tau_free_sequence.yaml"
)
PHYSICAL_MSE_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "train_cfg"
    / "tau_free_sequence"
    / "tau_free_sequence_sample_lstm_physical_mse_noema.yaml"
)


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def load_physical_mse_config():
    with PHYSICAL_MSE_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def test_next_config_matches_q_history_to_measured_torque_contract():
    config = load_config()

    assert config["dataloader"]["horizon"] == 50
    assert config["dataloader"]["pad_history"] is False
    assert config["dataloader"]["contact_free"] is True
    assert config["dataloader"]["lowdim_keys"]["q"] == "observation.joint"
    assert config["dataloader"]["lowdim_keys"]["dq"] == "observation.velocity"
    assert config["dataloader"]["lowdim_keys"]["delta_q"] == (
        "observation.delta_q"
    )
    assert config["dataloader"]["lowdim_keys"]["tau"] == "observation.torque"
    assert config["model"]["inputs"] == ["q", "dq", "delta_q"]
    assert config["model"]["target_key"] == "tau"
    assert config["model"]["architecture"] == "lstm"
    assert config["model"]["hidden_dim"] == 128
    assert config["model"]["num_layers"] == 2
    assert config["train"]["val_ratio"] == pytest.approx(0.1)
    assert config["train"]["split_mode"] == "sample"
    assert config["train"]["lr"] == pytest.approx(1.0e-4)
    assert config["train"]["monitor_key"] == "val_peak_cvar_rmse_nm"
    assert config["loss"] == {
        "type": "mse",
        "tau_weight": 1.0,
        "wrench_weight": 0,
        "force_scale_n": 1.0,
        "moment_scale_nm": 0.1,
        "joint_weights": None,
    }


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("model", "inputs"), ["q", "tau"], "inputs"),
        (("model", "target_key"), "tau_other", "target_key"),
        (("model", "architecture"), "transformer", "architecture"),
        (("dataloader", "horizon"), 25, "horizon"),
        (("dataloader", "pad_history"), True, "pad_history"),
        (("dataloader", "contact_free"), False, "contact_free"),
    ],
)
def test_next_rejects_training_inference_contract_drift(path, value, message):
    config = load_config()
    config["train"]["device"] = "cpu"
    config[path[0]][path[1]] = value

    with pytest.raises(ValueError, match=message):
        TauFreeSequenceTrainer(config)


@pytest.mark.parametrize("architecture", ["lstm", "gru", "tcn"])
def test_next_accepts_each_independent_sequence_branch(architecture):
    config = load_config()
    config["model"]["architecture"] = architecture

    TauFreeSequenceTrainer._validate_contract(config)


def test_next_pure_mse_disables_jacobian_pipeline():
    config = load_config()
    config["train"]["device"] = "cpu"
    config["loss"]["wrench_weight"] = 0.0

    trainer = TauFreeSequenceTrainer(config)

    assert trainer.use_wrench_objective is False
    assert trainer.wrench_dynamics is None


def test_physical_mse_config_uses_direct_h5_and_current_aligned_target():
    config = load_physical_mse_config()

    assert config["dataloader"]["backend"] == "h5"
    assert config["dataloader"]["h5_timestamp_path"] == "teleop/timestamp_us"
    assert config["dataloader"]["expected_fps"] == 100
    assert config["dataloader"]["horizon"] == 50
    assert config["dataloader"]["pad_history"] is False
    assert config["dataloader"]["h5_fields"]["observation.delta_q"] == {
        "operation": "subtract",
        "paths": ["teleop/q_cmd", "teleop/q_follower"],
    }
    assert config["model"]["target_key"] == "tau"
    assert config["dataloader"]["filters"]["tau"] == {
        "enabled": True,
        "dataset_preprocessed_operations": [
            {"type": "lowpass", "cutoff_hz": 10.0}
        ],
        "operations": [
            {"type": "lowpass", "cutoff_hz": 10.0},
            {"type": "median", "window": 3},
        ],
    }
    assert "target_filter" not in config["model"]
    assert config["loss"]["torque_loss_space"] == "physical_nm"
    assert config["loss"]["joint_weight_mode"] == "manual"
    assert config["loss"]["joint_weights"] == [1, 2.5, 1, 2.5, 1, 1, 1]
    assert config["loss"]["wrench_weight"] == 0.0
    assert config["train"]["lr"] == pytest.approx(1.0e-3)
    assert config["train"]["split_mode"] == "episode"
    assert config["train"]["val_episode_indices"] == [7, 8, 31, 32]
    assert config["train"]["output_dir"] == "outputs/tau_free_sequence/final_v"
    assert config["train"]["ema"]["enabled"] is False
    assert config["train"]["monitor_key"] == "val_tau_mse_nm2"


def test_physical_mse_computes_seven_joint_losses_before_equal_average():
    objective = TauFreeTorqueWrenchLoss(
        {
            "loss": {
                "torque_loss_space": "physical_nm",
                "joint_weight_mode": "equal",
                "tau_weight": 1.0,
                "wrench_weight": 0.0,
            }
        }
    )
    prediction = torch.zeros(2, 7)
    target = torch.zeros_like(prediction)
    physical_error_nm = torch.tensor(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    ).expand_as(prediction)

    loss, metrics = objective.torque_objective(
        prediction,
        target,
        physical_error_nm,
        torch.zeros_like(physical_error_nm),
    )

    expected_joint_mse = physical_error_nm[0].square()
    torch.testing.assert_close(loss, expected_joint_mse.mean())
    for joint_index, expected in enumerate(expected_joint_mse, start=1):
        torch.testing.assert_close(
            metrics[f"tau_mse_nm2_j{joint_index}"],
            expected,
        )
        torch.testing.assert_close(
            metrics[f"tau_joint_weight_j{joint_index}"],
            torch.tensor(1.0),
        )


def test_physical_mse_backpropagates_physical_axis_scales():
    objective = TauFreeTorqueWrenchLoss(
        {
            "loss": {
                "torque_loss_space": "physical_nm",
                "joint_weight_mode": "equal",
                "tau_weight": 1.0,
                "wrench_weight": 0.0,
            }
        }
    )
    prediction = torch.ones(1, 7, requires_grad=True)
    target = torch.zeros_like(prediction)
    physical_scale = torch.arange(1.0, 8.0).reshape(1, 7)

    loss, _ = objective.torque_objective(
        prediction,
        target,
        prediction * physical_scale,
        target * physical_scale,
    )
    loss.backward()

    expected_gradient = 2.0 * physical_scale.square() / 7.0
    torch.testing.assert_close(prediction.grad, expected_gradient)


@pytest.mark.parametrize("mode", ["mean_abs", "max_abs"])
def test_automatic_joint_weights_use_physical_training_targets(mode):
    objective = TauFreeTorqueWrenchLoss(
        {
            "loss": {
                "torque_loss_space": "physical_nm",
                "joint_weight_mode": mode,
                "joint_weights": None,
                "tau_weight": 1.0,
                "wrench_weight": 0.0,
            }
        }
    )
    targets = torch.tensor(
        [
            [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0],
            [3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0],
        ]
    )

    weights = objective.resolve_joint_weights(targets)
    scale = (
        targets.abs().mean(dim=0)
        if mode == "mean_abs"
        else targets.abs().max(dim=0).values
    )

    torch.testing.assert_close(weights, scale / scale.mean())
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))


def test_tau_target_moving_average_resets_at_episode_boundaries_and_skips_second_lowpass():
    config = {
        "model": {
            "target_filter": {
                "moving_average_window": 3,
                "apply_additional_lowpass": False,
            }
        }
    }
    filter_config = torque_target_filter_config(config)
    timestamps = torch.tensor([0.00, 0.01, 0.02, 0.00, 0.01, 0.02])
    tau = torch.tensor([[0.0], [10.0], [0.0], [5.0], [50.0], [5.0]])

    filtered = filter_torque_target_dataset(
        timestamps,
        tau,
        [(0, 3), (3, 6)],
        filter_config,
    )

    torch.testing.assert_close(
        filtered,
        torch.tensor(
            [[0.0], [10.0 / 3.0], [10.0 / 3.0], [5.0], [20.0], [20.0]]
        ),
    )


def test_legacy_tau_target_median_filter_remains_checkpoint_compatible():
    config = {
        "model": {
            "target_filter": {
                "median_window": 3,
                "apply_additional_lowpass": False,
            }
        }
    }
    filter_config = torque_target_filter_config(config)
    timestamps = torch.tensor([0.00, 0.01, 0.02])
    tau = torch.tensor([[0.0], [10.0], [0.0]])

    filtered = filter_torque_target_dataset(
        timestamps,
        tau,
        [(0, 3)],
        filter_config,
    )

    assert filter_config["mode"] == "median"
    torch.testing.assert_close(filtered, torch.zeros_like(tau))


def test_additional_lowpass_requires_an_explicit_cutoff():
    config = {
        "model": {
            "target_filter": {
                "moving_average_window": 3,
                "apply_additional_lowpass": True,
            }
        }
    }

    with pytest.raises(ValueError, match="cutoff_hz is required"):
        torque_target_filter_config(config)


def test_next_model_uses_configured_inputs_and_last_measured_torque():
    config = load_config()
    config["train"]["device"] = "cpu"
    config["model"].update(
        hidden_dim=8,
        num_layers=1,
        head_hidden_dim=8,
        dropout=0.0,
    )
    trainer = TauFreeSequenceTrainer(config)
    trainer.model = trainer.build_model()
    tau = torch.randn(3, 50, 7)
    batch = {
        "q": torch.randn(3, 50, 7),
        "dq": torch.randn(3, 50, 7),
        "delta_q": torch.randn(3, 50, 7),
        "tau": tau,
        "frame_jacobian": torch.eye(6, 7).expand(3, 6, 7).clone(),
    }

    loss, out = trainer.compute_loss(batch)

    assert trainer.model.active_inputs == ["q", "dq", "delta_q"]
    assert trainer.model.input_dim == 21
    assert trainer.model.recurrent.input_size == 21
    assert loss.ndim == 0
    torch.testing.assert_close(out["tau_other_target"], tau[:, -1])
    torch.testing.assert_close(out["_target_nm"], tau[:, -1])
    assert "wrench_pred" not in out
    assert "wrench_mse_scaled" not in out["loss_dict"]


def test_next_validation_reports_joint_normalized_mae_metrics():
    class FixedPredictionModel(torch.nn.Module):
        def forward(self, batch):
            return {
                "tau_other_pred": batch["prediction"],
                "tau_other_target": batch["target"],
            }

    config = load_config()
    config["train"]["device"] = "cpu"
    trainer = TauFreeSequenceTrainer(config)
    trainer.device = "cpu"
    trainer.dataset = SimpleNamespace(
        normalize_mode=None,
        normalize_lowdim_keys=[],
    )
    trainer.model = FixedPredictionModel()
    levels = torch.tensor([0.0, 1.0, 2.0, 3.0])
    trainer.val_loader = torch.utils.data.DataLoader(
        [
            {
                "prediction": torch.full((7,), level + 1.0),
                "target": torch.full((7,), level),
            }
            for level in levels
        ],
        batch_size=2,
    )

    trainer.validate_one_epoch(epoch=0)

    metrics = trainer.last_val_epoch_metrics
    expected_std = levels.std().item()
    expected_range = (
        torch.quantile(levels, 0.95) - torch.quantile(levels, 0.05)
    ).item()
    for joint_index in range(1, 8):
        assert metrics[f"joint_mae_nm_j{joint_index}"] == pytest.approx(1.0)
        assert metrics[f"joint_tau_std_nm_j{joint_index}"] == pytest.approx(
            expected_std
        )
        assert metrics[f"joint_nmae_std_j{joint_index}"] == pytest.approx(
            1.0 / (expected_std + 1.0e-8)
        )
        assert metrics[f"joint_tau_p90_range_nm_j{joint_index}"] == pytest.approx(
            expected_range
        )
        assert metrics[f"joint_nmae_p90_j{joint_index}"] == pytest.approx(
            1.0 / (expected_range + 1.0e-8)
        )


def test_tau_wrench_loss_backpropagates_through_torque_prediction():
    config = load_config()
    objective = TauFreeTorqueWrenchLoss(config)
    prediction = torch.full((2, 7), 0.2, requires_grad=True)
    target = torch.zeros_like(prediction)
    frame_jacobian = torch.eye(6, 7).expand(2, 6, 7).clone()

    loss, metrics, diagnostics = objective(
        prediction,
        target,
        prediction,
        target,
        frame_jacobian,
    )
    loss.backward()

    assert prediction.grad is not None
    assert prediction.grad.abs().sum() > 0
    assert metrics["tau_mse"].item() > 0
    assert metrics["wrench_mse_scaled"].item() > 0
    torch.testing.assert_close(
        diagnostics["tau_ext_pred_nm"],
        torch.full((2, 7), -0.2),
    )
    assert diagnostics["tau_ext_pred_nm"].shape == (2, 7)
    assert diagnostics["wrench_pred"].shape == (2, 6)


@pytest.mark.parametrize(
    "inputs",
    [
        ["q"],
        ["q", "dq", "delta_q"],
    ],
)
def test_next_allows_next_input_ablations_from_config(inputs):
    config = load_config()
    config["train"]["device"] = "cpu"
    config["model"]["inputs"] = inputs

    trainer = TauFreeSequenceTrainer(config)

    assert trainer.build_model().active_inputs == inputs


def test_next_reports_missing_optional_dataset_column_before_cache(monkeypatch):
    config = load_config()
    config["train"]["device"] = "cpu"
    config["model"]["inputs"] = ["q", "dq", "delta_q"]
    dataset = SimpleNamespace(
        stats_dataset=SimpleNamespace(
            column_names=[
                "observation.joint",
                "observation.velocity",
                "observation.torque",
            ]
        )
    )
    monkeypatch.setattr(TauOtherTrainer, "build_dataset", lambda _self: dataset)
    trainer = TauFreeSequenceTrainer(config)

    with pytest.raises(ValueError, match="meaningful q_cmd"):
        trainer.build_dataset()


def test_next_windows_do_not_carry_hidden_state_between_samples():
    config = load_config()
    config["train"]["device"] = "cpu"
    config["model"].update(
        hidden_dim=8,
        num_layers=1,
        head_hidden_dim=8,
        dropout=0.0,
    )
    model = TauFreeSequenceTrainer(config).build_model().eval()
    first = {
        "q": torch.randn(2, 50, 7),
        "dq": torch.randn(2, 50, 7),
        "delta_q": torch.randn(2, 50, 7),
    }
    second = {
        "q": torch.randn(2, 50, 7),
        "dq": torch.randn(2, 50, 7),
        "delta_q": torch.randn(2, 50, 7),
    }

    with torch.no_grad():
        model(first)
        after_first = model(second)["tau_other_pred"]
        repeated = model(second)["tau_other_pred"]

    torch.testing.assert_close(after_first, repeated)
