from pathlib import Path

import h5py
import numpy as np

from data_process.offline_tau_labels import (
    KalmanRTSConfig,
    causal_median_one_pole_filter,
    estimate_joint_states_causal,
    estimate_joint_states_rts,
    fill_missing_measurements,
    residual_torque,
)
from data_process.tool.build_offline_tau_labels import process_episode


def _noisy_motion(sample_count=800):
    rng = np.random.default_rng(7)
    dt = np.clip(0.01 + rng.normal(0.0, 0.001, sample_count - 1), 0.004, None)
    timestamps = np.concatenate(([0.0], np.cumsum(dt)))
    omega = 2.0 * np.pi * 0.7
    q_true = 0.3 * np.sin(omega * timestamps) + 0.04 * timestamps**2
    dq_true = 0.3 * omega * np.cos(omega * timestamps) + 0.08 * timestamps
    ddq_true = -0.3 * omega**2 * np.sin(omega * timestamps) + 0.08
    q = q_true + rng.normal(0.0, 5.0e-4, sample_count)
    dq = dq_true + rng.normal(0.0, 3.0e-2, sample_count)
    return timestamps, q[:, None], dq[:, None], ddq_true


def test_rts_acceleration_beats_velocity_differencing_with_jittered_timestamps():
    timestamps, q, dq, ddq_true = _noisy_motion()

    estimate = estimate_joint_states_rts(timestamps, q, dq)
    naive_ddq = np.zeros(len(timestamps))
    naive_ddq[1:] = np.diff(dq[:, 0]) / np.diff(timestamps)
    interior = slice(20, -20)
    naive_rmse = np.sqrt(np.mean((naive_ddq[interior] - ddq_true[interior]) ** 2))
    causal_rmse = np.sqrt(
        np.mean((estimate.ddq_filtered[interior, 0] - ddq_true[interior]) ** 2)
    )
    smoothed_rmse = np.sqrt(
        np.mean((estimate.ddq_smoothed[interior, 0] - ddq_true[interior]) ** 2)
    )

    assert smoothed_rmse < causal_rmse < naive_rmse
    assert estimate.ddq_smoothed_std.shape == q.shape
    assert np.all(estimate.ddq_smoothed_std >= 0.0)


def test_causal_estimator_matches_rts_forward_pass_and_ignores_future_changes():
    timestamps, q, dq, _ = _noisy_motion(sample_count=120)

    causal = estimate_joint_states_causal(timestamps, q, dq)
    rts = estimate_joint_states_rts(timestamps, q, dq)
    changed_q = q.copy()
    changed_dq = dq.copy()
    changed_q[80:] += 100.0
    changed_dq[80:] -= 100.0
    changed = estimate_joint_states_causal(timestamps, changed_q, changed_dq)

    np.testing.assert_allclose(causal.q_filtered, rts.q_filtered)
    np.testing.assert_allclose(causal.dq_filtered, rts.dq_filtered)
    np.testing.assert_allclose(causal.ddq_filtered, rts.ddq_filtered)
    np.testing.assert_allclose(changed.ddq_filtered[:80], causal.ddq_filtered[:80])


def test_missing_observations_are_smoothed_and_long_gaps_start_new_segments():
    timestamps, q, dq, _ = _noisy_motion(sample_count=120)
    timestamps[70:] += 0.25
    q[20:24] = np.nan
    dq[50:55] = np.nan

    estimate = estimate_joint_states_rts(
        timestamps,
        q,
        dq,
        KalmanRTSConfig(max_gap_s=0.1),
    )

    assert estimate.segment_starts == (0, 70)
    assert np.all(np.isfinite(estimate.q_smoothed))
    assert np.all(np.isfinite(estimate.dq_smoothed))
    assert np.all(np.isfinite(estimate.ddq_smoothed))
    q_filled = fill_missing_measurements(q, estimate.q_smoothed)
    np.testing.assert_array_equal(q_filled[np.isfinite(q)], q[np.isfinite(q)])
    assert np.all(np.isfinite(q_filled))


def test_residual_torque_uses_measured_minus_inverse_dynamics():
    measured = np.asarray([[4.0, -1.0]])
    inverse_dynamics = np.asarray([[1.5, 2.0]])
    np.testing.assert_allclose(
        residual_torque(measured, inverse_dynamics),
        [[2.5, -3.0]],
    )


def test_causal_torque_filter_uses_measured_dt_and_first_sample_initialization():
    timestamps = np.asarray([0.0, 0.01, 0.03])
    values = np.asarray([[0.0], [1.0], [1.0]])

    filtered = causal_median_one_pole_filter(
        timestamps,
        values,
        cutoff_hz=10.0,
        median_window=1,
    )

    alpha_1 = 1.0 - np.exp(-2.0 * np.pi * 10.0 * 0.01)
    alpha_2 = 1.0 - np.exp(-2.0 * np.pi * 10.0 * 0.02)
    expected_1 = alpha_1
    expected_2 = alpha_2 + (1.0 - alpha_2) * expected_1
    np.testing.assert_allclose(filtered[:, 0], [0.0, expected_1, expected_2])


class _FakeModel:
    nq = 2
    nv = 2

    @staticmethod
    def createData():
        return object()


class _FakePinocchio:
    @staticmethod
    def rnea(model, data, q, dq, ddq):
        return q + 2.0 * dq + 3.0 * ddq


def _write_episode(path: Path):
    timestamps = np.arange(80, dtype=np.int64) * 10_000
    time_s = timestamps * 1.0e-6
    q = np.stack((0.2 * time_s**2, -0.1 * time_s**2), axis=-1)
    dq = np.stack((0.4 * time_s, -0.2 * time_s), axis=-1)
    tau = 1.0 + q + 2.0 * dq
    with h5py.File(path, "w") as h5_file:
        teleop = h5_file.create_group("teleop")
        teleop.create_dataset("timestamp_us", data=timestamps)
        teleop.create_dataset("q_follower", data=q)
        teleop.create_dataset("dq_follower", data=dq)
        teleop.create_dataset("tau_follower", data=tau)
        teleop["tau_follower"].attrs.update(
            {
                "causal": True,
                "lowpass": True,
                "lowpass_cutoff_hz": 10.0,
                "median_window": 1,
                "zero_phase": False,
            }
        )
        teleop.create_dataset("ddq_follower", data=np.zeros_like(q))
        teleop.create_dataset("tau_other_cal", data=np.zeros_like(q))


def test_h5_pipeline_copies_source_and_writes_self_describing_labels(tmp_path):
    source = tmp_path / "source.h5"
    destination = tmp_path / "output" / "episode.h5"
    _write_episode(source)
    config = {
        "keys": {},
        "processing": {
            "timestamp_scale_to_s": 1.0e-6,
            "dq_sign": [1, 1],
            "rnea_state_source": "measured",
            "torque_filter": {"cutoff_hz": 10.0, "median_window": 1},
        },
        "estimator": {"max_gap_s": 0.1},
    }

    result = process_episode(
        source,
        destination,
        pin=_FakePinocchio(),
        model=_FakeModel(),
        config=config,
        overwrite=False,
    )

    with h5py.File(source, "r") as source_h5:
        assert "tau_id_rts" not in source_h5["teleop"]
        assert not source_h5.attrs.get("offline_tau_labels_built", False)
    with h5py.File(destination, "r") as output_h5:
        teleop = output_h5["teleop"]
        tau = np.asarray(teleop["tau_follower"])
        tau_id = np.asarray(teleop["tau_id_rts_filtered"])
        tau_other = np.asarray(teleop["tau_other_cal"])
        np.testing.assert_allclose(tau_other, tau - tau_id)
        assert teleop["tau_other_cal"].attrs["formula"].startswith(
            "tau_other=tau_filtered-tau_id_filtered"
        )
        assert bool(teleop["tau_id_rts_filtered"].attrs["lowpass"])
        assert bool(teleop["ddq_follower"].attrs["offline_only"])
        assert output_h5.attrs["offline_tau_residual_convention"] == (
            "tau_filtered_minus_tau_id_filtered"
        )
    assert result["frames"] == 80
    assert result["segments"] == 1
