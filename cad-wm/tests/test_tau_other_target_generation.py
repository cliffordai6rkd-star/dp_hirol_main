from types import SimpleNamespace

import numpy as np
import torch

from data_process.tau_other_target_generation import (
    build_causal_tau_other_target,
    normalize_tau_other_target_generation,
    resolve_tau_other_target_generation,
    timestamps_to_seconds,
)
from train.trainer.tau_other_sequence_train import TauOtherTrainer


class FakeDynamics:
    def inverse_dynamics(self, q, dq, ddq):
        return q + 2.0 * dq + 3.0 * ddq

    def gravity_torque(self, q):
        return q + 10.0


def target_generation_config():
    return {
        "target_generation": {
            "enabled": True,
            "method": "causal_gravity_residual_v1",
            "target_key": "tau_other",
            "timestamp_key": "timestamp_us",
            "timestamp_unit": "us",
            "source_keys": {"q": "q", "dq": "dq", "tau": "tau"},
            "dq_sign": [1, -1],
            "torque_filter_key": "tau",
        }
    }


def test_gravity_target_generation_matches_observation_torque_per_episode():
    config = target_generation_config()
    resolved = resolve_tau_other_target_generation(config, {})
    timestamps = np.concatenate((np.arange(6) * 0.01, 1.0 + np.arange(6) * 0.01))
    q = torch.zeros(12, 2)
    dq = torch.stack(
        (torch.linspace(0.0, 0.5, 12), torch.linspace(0.2, 0.8, 12)), dim=-1
    )
    tau_measured = torch.full((12, 2), 4.0)
    episodes = [
        {"dataset_from_index": 0, "dataset_to_index": 6},
        {"dataset_from_index": 6, "dataset_to_index": 12},
    ]

    result = build_causal_tau_other_target(
        timestamps_s=timestamps,
        q=q,
        dq=dq,
        tau_measured=tau_measured,
        episodes=episodes,
        target_config=resolved,
        dynamics=FakeDynamics(),
    )

    expected_dq = dq * torch.tensor([1.0, -1.0])
    expected_gravity = q + 10.0
    expected_tau_other = tau_measured - expected_gravity

    torch.testing.assert_close(result.dq, expected_dq)
    torch.testing.assert_close(result.tau_other, expected_tau_other)
    torch.testing.assert_close(result.tau_g, expected_gravity)
    assert resolved["ddq_source"] == "unused"
    assert resolved["residual_formula"] == "tau_other=tau_measured-tau_g"


def test_causal_target_prefix_does_not_change_when_future_measurements_change():
    config = target_generation_config()
    resolved = resolve_tau_other_target_generation(config, {})
    timestamps = np.arange(12, dtype=np.float64) * 0.01
    q = torch.zeros(12, 2)
    dq = torch.zeros(12, 2)
    tau = torch.ones(12, 2)
    episode = [{"dataset_from_index": 0, "dataset_to_index": 12}]

    baseline = build_causal_tau_other_target(
        timestamps_s=timestamps,
        q=q,
        dq=dq,
        tau_measured=tau,
        episodes=episode,
        target_config=resolved,
        dynamics=FakeDynamics(),
    )
    changed_q = q.clone()
    changed_dq = dq.clone()
    changed_q[8:] = 100.0
    changed_dq[8:] = -100.0
    changed = build_causal_tau_other_target(
        timestamps_s=timestamps,
        q=changed_q,
        dq=changed_dq,
        tau_measured=tau,
        episodes=episode,
        target_config=resolved,
        dynamics=FakeDynamics(),
    )

    torch.testing.assert_close(changed.ddq[:8], baseline.ddq[:8])
    torch.testing.assert_close(changed.tau_other[:8], baseline.tau_other[:8])


def test_gravity_target_generation_uses_rnea_g_of_q_only():
    config = target_generation_config()
    resolved = resolve_tau_other_target_generation(config, {})
    timestamps = np.arange(6, dtype=np.float64) * 0.01
    q = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    dq = torch.zeros_like(q)
    tau = torch.full_like(q, 20.0)

    result = build_causal_tau_other_target(
        timestamps_s=timestamps,
        q=q,
        dq=dq,
        tau_measured=tau,
        episodes=[{"dataset_from_index": 0, "dataset_to_index": 6}],
        target_config=resolved,
        dynamics=FakeDynamics(),
    )

    expected_gravity = q + 10.0
    torch.testing.assert_close(result.tau_g, expected_gravity)
    torch.testing.assert_close(result.tau_other, tau - expected_gravity)
    assert resolved["residual_formula"] == "tau_other=tau_measured-tau_g"


def test_causal_rnea_target_uses_estimated_dq_ddq_and_full_inverse_dynamics():
    config = target_generation_config()
    config["target_generation"]["method"] = "causal_rnea_residual_v1"
    config["target_generation"]["dq_sign"] = [1, 1]
    resolved = resolve_tau_other_target_generation(config, {})
    timestamps = np.arange(20, dtype=np.float64) * 0.02
    q = torch.stack(
        (torch.linspace(0.0, 1.0, 20) ** 2, torch.linspace(0.0, 0.5, 20)),
        dim=-1,
    )
    dq = torch.stack(
        (torch.linspace(0.0, 2.0, 20), torch.full((20,), 0.5)),
        dim=-1,
    )
    tau = torch.full_like(q, 20.0)

    result = build_causal_tau_other_target(
        timestamps_s=timestamps,
        q=q,
        dq=dq,
        tau_measured=tau,
        episodes=[{"dataset_from_index": 0, "dataset_to_index": 20}],
        target_config=resolved,
        dynamics=FakeDynamics(),
    )

    expected_tau_id = q + 2.0 * result.dq + 3.0 * result.ddq
    torch.testing.assert_close(result.tau_id, expected_tau_id)
    torch.testing.assert_close(result.tau_other, tau - expected_tau_id)
    assert torch.count_nonzero(result.ddq).item() > 0
    assert resolved["ddq_source"] == "causal_state_estimator(q,dq)"
    assert resolved["residual_formula"] == "tau_other=tau_measured-tau_id"


def test_timestamp_units_and_tau_input_leakage_contract():
    normalized = normalize_tau_other_target_generation(target_generation_config())
    timestamps = timestamps_to_seconds(torch.tensor([0, 10_000, 20_000]), "us")

    np.testing.assert_allclose(timestamps, [0.0, 0.01, 0.02])
    assert normalized["source_keys"] == {"q": "q", "dq": "dq", "tau": "tau"}

    trainer_config = target_generation_config()
    trainer_config.update(
        {
            "dataloader": {"horizon": 3},
            "model": {
                "architecture": "lstm",
                "inputs": ["q", "dq", "tau"],
                "input_dims": {"q": 2, "dq": 2, "tau": 2},
                "target_key": "tau_other",
            },
            "loss": {},
            "train": {},
        }
    )
    try:
        TauOtherTrainer(trainer_config)
    except ValueError as exc:
        assert "leaks the tau_other supervision target" in str(exc)
    else:
        raise AssertionError("Measured tau must be rejected as a model input")


def test_trainer_builds_derived_target_without_dataset_tau_other_column():
    config = target_generation_config()
    config.update(
        {
            "dataloader": {"horizon": 3},
            "model": {
                "architecture": "lstm",
                "inputs": ["q", "dq"],
                "input_dims": {"q": 2, "dq": 2},
                "target_key": "tau_other",
            },
            "loss": {},
            "train": {"device": "cpu"},
        }
    )
    trainer = TauOtherTrainer(config)
    trainer.tau_other_dynamics = FakeDynamics()
    trainer.dataset = SimpleNamespace(
        filter_config={},
        dataset=SimpleNamespace(
            meta=SimpleNamespace(
                episodes=[{"dataset_from_index": 0, "dataset_to_index": 6}]
            )
        ),
    )
    trainer._load_dataset_column = lambda _key: torch.arange(6) * 10_000
    cpu_cache = {
        "q": torch.zeros(6, 2),
        "dq": torch.zeros(6, 2),
        "tau": torch.ones(6, 2),
    }

    trainer._build_derived_target(cpu_cache, "tau_other")

    assert set(cpu_cache) == {"q", "dq", "tau", "tau_other"}
    torch.testing.assert_close(cpu_cache["tau_other"], torch.full((6, 2), -9.0))


def test_checkpoint_explicitly_stores_resolved_target_generation(tmp_path):
    config = target_generation_config()
    config.update(
        {
            "dataloader": {"horizon": 3, "normalize_mode": None},
            "model": {
                "architecture": "lstm",
                "inputs": ["q", "dq"],
                "input_dims": {"q": 2, "dq": 2},
                "target_key": "tau_other",
            },
            "loss": {},
            "train": {"device": "cpu", "output_dir": str(tmp_path)},
        }
    )
    trainer = TauOtherTrainer(config)
    trainer.derived_target_config = {
        **trainer.derived_target_config,
        "torque_filter_operations": [],
    }
    trainer.model = torch.nn.Linear(1, 1)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters())
    trainer.scheduler = None
    trainer.dataset = SimpleNamespace(
        filter_config={"tau": {"enabled": True}},
        sample_rate_hz=99.0,
        normalizer=SimpleNamespace(stats={}, eps=1.0e-6),
    )

    trainer.save_checkpoint(epoch=0, avg_loss=1.0)
    checkpoint = torch.load(
        trainer.ckpt_dir / "epoch_000.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint["derived_target_config"] == trainer.derived_target_config
    assert checkpoint["derived_target_config"]["method"] == "causal_gravity_residual_v1"
