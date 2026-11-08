"""Offline joint-state estimation for inverse-dynamics label generation.

The estimator uses the measured timestamps directly.  A causal Kalman filter
is followed by a Rauch-Tung-Striebel backward pass, so the smoothed acceleration
is suitable for offline labels but must not be used by an online controller.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class KalmanRTSConfig:
    """Noise parameters for an independent [q, dq, ddq] model per joint."""

    position_std: float | Sequence[float] = 5.0e-4
    velocity_std: float | Sequence[float] = 3.0e-2
    jerk_std: float | Sequence[float] = 2.0
    initial_position_std: float | Sequence[float] = 1.0e-2
    initial_velocity_std: float | Sequence[float] = 2.0e-1
    initial_acceleration_std: float | Sequence[float] = 5.0
    max_gap_s: float | None = 0.1


@dataclass(frozen=True)
class JointStateEstimate:
    """Causal and offline-smoothed state estimates in physical units."""

    q_filtered: np.ndarray
    dq_filtered: np.ndarray
    ddq_filtered: np.ndarray
    q_smoothed: np.ndarray
    dq_smoothed: np.ndarray
    ddq_smoothed: np.ndarray
    ddq_smoothed_std: np.ndarray
    segment_starts: tuple[int, ...]


@dataclass(frozen=True)
class CausalJointStateEstimate:
    """Forward-filtered joint state that can be replayed online."""

    q_filtered: np.ndarray
    dq_filtered: np.ndarray
    ddq_filtered: np.ndarray
    segment_starts: tuple[int, ...]


def causal_median_one_pole_filter(
    timestamps_s: np.ndarray,
    values: np.ndarray,
    *,
    cutoff_hz: float,
    median_window: int = 1,
) -> np.ndarray:
    """Apply the same timestamp-aware causal torque filter used online."""
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if timestamps_s.ndim != 1 or values.ndim != 2:
        raise ValueError("causal filtering expects timestamps [N] and values [N, D].")
    if len(timestamps_s) != len(values) or len(values) == 0:
        raise ValueError("causal filter timestamps and values must be non-empty and aligned.")
    if not np.isfinite(timestamps_s).all() or not np.isfinite(values).all():
        raise ValueError("causal filter inputs must be finite.")
    if len(timestamps_s) > 1 and np.any(np.diff(timestamps_s) <= 0.0):
        raise ValueError("causal filter timestamps must increase strictly.")
    if not np.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
        raise ValueError("cutoff_hz must be positive and finite.")
    if median_window < 1 or median_window % 2 == 0:
        raise ValueError("median_window must be a positive odd integer.")

    median_values = causal_trailing_median_filter(
        values,
        median_window=median_window,
    )
    filtered = np.empty_like(median_values)
    state: np.ndarray | None = None
    previous_timestamp_s: float | None = None
    for index, (timestamp_s, median_value) in enumerate(
        zip(timestamps_s, median_values)
    ):
        if state is None:
            state = median_value.copy()
        else:
            assert previous_timestamp_s is not None
            dt = float(timestamp_s - previous_timestamp_s)
            alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz * dt)
            state = alpha * median_value + (1.0 - alpha) * state
        filtered[index] = state
        previous_timestamp_s = float(timestamp_s)
    return filtered


def causal_trailing_median_filter(
    values: np.ndarray,
    *,
    median_window: int,
) -> np.ndarray:
    """Reject isolated spikes with a causal trailing median window."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("causal median filtering expects non-empty values [N, D].")
    if not np.isfinite(values).all():
        raise ValueError("causal median filter inputs must be finite.")
    if median_window < 1 or median_window % 2 == 0:
        raise ValueError("median_window must be a positive odd integer.")

    history: deque[np.ndarray] = deque()
    filtered = np.empty_like(values)
    for index, value in enumerate(values):
        history.append(value.copy())
        while len(history) > median_window:
            history.popleft()
        samples = list(history)
        samples = [samples[0]] * (median_window - len(samples)) + samples
        filtered[index] = np.median(np.stack(samples, axis=0), axis=0)
    return filtered


def _joint_parameter(
    value: float | Sequence[float],
    joint_count: int,
    name: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 0:
        result = np.full(joint_count, float(result), dtype=np.float64)
    if result.shape != (joint_count,):
        raise ValueError(f"{name} must be scalar or have shape ({joint_count},).")
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError(f"{name} must contain positive finite values.")
    return result


def _validate_measurements(
    timestamps_s: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    dq = np.asarray(dq, dtype=np.float64)
    if timestamps_s.ndim != 1 or len(timestamps_s) < 2:
        raise ValueError("timestamps_s must be a 1D array with at least two samples.")
    if q.ndim != 2 or dq.shape != q.shape or len(q) != len(timestamps_s):
        raise ValueError(
            "q and dq must have the same [time, joint] shape matching timestamps_s."
        )
    if np.any(~np.isfinite(timestamps_s)):
        raise ValueError("timestamps_s must be finite.")
    dt = np.diff(timestamps_s)
    if np.any(dt <= 0.0):
        raise ValueError("timestamps_s must be strictly increasing.")
    if np.any(np.isinf(q)) or np.any(np.isinf(dq)):
        raise ValueError("q and dq may contain NaN dropouts, but not infinities.")
    return timestamps_s, q, dq


def _transition(dt: float) -> np.ndarray:
    return np.asarray(
        [
            [1.0, dt, 0.5 * dt * dt],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _white_jerk_covariance(dt: float, spectral_density: float) -> np.ndarray:
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    dt5 = dt4 * dt
    return spectral_density * np.asarray(
        [
            [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
            [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
            [dt3 / 6.0, dt2 / 2.0, dt],
        ],
        dtype=np.float64,
    )


def _symmetrize(covariance: np.ndarray) -> np.ndarray:
    return 0.5 * (covariance + covariance.T)


def _measurement_update(
    state: np.ndarray,
    covariance: np.ndarray,
    q_value: float,
    dq_value: float,
    position_variance: float,
    velocity_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray([q_value, dq_value], dtype=np.float64)
    observed = np.isfinite(values)
    if not np.any(observed):
        return state, covariance

    full_h = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    h = full_h[observed]
    measurement = values[observed]
    r = np.diag(
        np.asarray([position_variance, velocity_variance], dtype=np.float64)[
            observed
        ]
    )
    innovation_covariance = h @ covariance @ h.T + r
    gain = np.linalg.solve(
        innovation_covariance,
        h @ covariance,
    ).T
    updated_state = state + gain @ (measurement - h @ state)

    # Joseph form avoids negative variances after long sequences.
    identity = np.eye(3, dtype=np.float64)
    correction = identity - gain @ h
    updated_covariance = (
        correction @ covariance @ correction.T + gain @ r @ gain.T
    )
    return updated_state, _symmetrize(updated_covariance)


def _initial_value(value: float, fallback: float = 0.0) -> float:
    return float(value) if np.isfinite(value) else fallback


def _filter_joint(
    timestamps_s: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    *,
    position_std: float,
    velocity_std: float,
    jerk_std: float,
    initial_position_std: float,
    initial_velocity_std: float,
    initial_acceleration_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(timestamps_s)
    filtered_state = np.empty((count, 3), dtype=np.float64)
    filtered_covariance = np.empty((count, 3, 3), dtype=np.float64)
    predicted_state = np.empty((count, 3), dtype=np.float64)
    predicted_covariance = np.empty((count, 3, 3), dtype=np.float64)
    transitions = np.empty((max(count - 1, 0), 3, 3), dtype=np.float64)

    state = np.asarray(
        [_initial_value(q[0]), _initial_value(dq[0]), 0.0],
        dtype=np.float64,
    )
    covariance = np.diag(
        np.square(
            [
                initial_position_std,
                initial_velocity_std,
                initial_acceleration_std,
            ]
        )
    )
    predicted_state[0] = state
    predicted_covariance[0] = covariance
    state, covariance = _measurement_update(
        state,
        covariance,
        q[0],
        dq[0],
        position_std**2,
        velocity_std**2,
    )
    filtered_state[0] = state
    filtered_covariance[0] = covariance

    for index in range(1, count):
        dt = float(timestamps_s[index] - timestamps_s[index - 1])
        transition = _transition(dt)
        process_covariance = _white_jerk_covariance(dt, jerk_std**2)
        state = transition @ state
        covariance = _symmetrize(
            transition @ covariance @ transition.T + process_covariance
        )
        transitions[index - 1] = transition
        predicted_state[index] = state
        predicted_covariance[index] = covariance
        state, covariance = _measurement_update(
            state,
            covariance,
            q[index],
            dq[index],
            position_std**2,
            velocity_std**2,
        )
        filtered_state[index] = state
        filtered_covariance[index] = covariance

    return (
        filtered_state,
        filtered_covariance,
        predicted_state,
        predicted_covariance,
        transitions,
    )


def _filter_and_smooth_joint(
    timestamps_s: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    *,
    position_std: float,
    velocity_std: float,
    jerk_std: float,
    initial_position_std: float,
    initial_velocity_std: float,
    initial_acceleration_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    (
        filtered_state,
        filtered_covariance,
        predicted_state,
        predicted_covariance,
        transitions,
    ) = _filter_joint(
        timestamps_s,
        q,
        dq,
        position_std=position_std,
        velocity_std=velocity_std,
        jerk_std=jerk_std,
        initial_position_std=initial_position_std,
        initial_velocity_std=initial_velocity_std,
        initial_acceleration_std=initial_acceleration_std,
    )

    count = len(timestamps_s)
    smoothed_state = filtered_state.copy()
    smoothed_covariance = filtered_covariance.copy()
    for index in range(count - 2, -1, -1):
        cross_covariance = filtered_covariance[index] @ transitions[index].T
        smoother_gain = np.linalg.solve(
            predicted_covariance[index + 1],
            cross_covariance.T,
        ).T
        smoothed_state[index] += smoother_gain @ (
            smoothed_state[index + 1] - predicted_state[index + 1]
        )
        smoothed_covariance[index] = _symmetrize(
            filtered_covariance[index]
            + smoother_gain
            @ (smoothed_covariance[index + 1] - predicted_covariance[index + 1])
            @ smoother_gain.T
        )
    return filtered_state, smoothed_state, smoothed_covariance


def _segment_bounds(
    timestamps_s: np.ndarray,
    max_gap_s: float | None,
) -> tuple[tuple[int, int], ...]:
    if max_gap_s is None:
        return ((0, len(timestamps_s)),)
    if not np.isfinite(max_gap_s) or max_gap_s <= 0.0:
        raise ValueError("max_gap_s must be positive and finite, or null.")
    starts = np.concatenate(
        ([0], np.flatnonzero(np.diff(timestamps_s) > max_gap_s) + 1)
    )
    stops = np.concatenate((starts[1:], [len(timestamps_s)]))
    return tuple((int(start), int(stop)) for start, stop in zip(starts, stops))


def _estimator_parameters(
    config: KalmanRTSConfig,
    joint_count: int,
) -> dict[str, np.ndarray]:
    return {
        name: _joint_parameter(getattr(config, name), joint_count, name)
        for name in (
            "position_std",
            "velocity_std",
            "jerk_std",
            "initial_position_std",
            "initial_velocity_std",
            "initial_acceleration_std",
        )
    }


def estimate_joint_states_causal(
    timestamps_s: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    config: KalmanRTSConfig | None = None,
) -> CausalJointStateEstimate:
    """Estimate ``[q, dq, ddq]`` using only current and previous samples."""

    timestamps_s, q, dq = _validate_measurements(timestamps_s, q, dq)
    config = config or KalmanRTSConfig()
    joint_count = q.shape[1]
    parameters = _estimator_parameters(config, joint_count)
    bounds = _segment_bounds(timestamps_s, config.max_gap_s)
    filtered = np.empty((len(q), joint_count, 3), dtype=np.float64)

    for start, stop in bounds:
        for joint in range(joint_count):
            joint_filtered, _, _, _, _ = _filter_joint(
                timestamps_s[start:stop],
                q[start:stop, joint],
                dq[start:stop, joint],
                **{name: value[joint] for name, value in parameters.items()},
            )
            filtered[start:stop, joint] = joint_filtered

    return CausalJointStateEstimate(
        q_filtered=filtered[..., 0],
        dq_filtered=filtered[..., 1],
        ddq_filtered=filtered[..., 2],
        segment_starts=tuple(start for start, _ in bounds),
    )


def estimate_joint_states_rts(
    timestamps_s: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    config: KalmanRTSConfig | None = None,
) -> JointStateEstimate:
    """Estimate [q, dq, ddq] with a variable-dt Kalman filter and RTS smoother."""

    timestamps_s, q, dq = _validate_measurements(timestamps_s, q, dq)
    config = config or KalmanRTSConfig()
    joint_count = q.shape[1]
    parameters = _estimator_parameters(config, joint_count)
    bounds = _segment_bounds(timestamps_s, config.max_gap_s)
    filtered = np.empty((len(q), joint_count, 3), dtype=np.float64)
    smoothed = np.empty_like(filtered)
    smoothed_variance = np.empty_like(filtered)

    for start, stop in bounds:
        for joint in range(joint_count):
            joint_filtered, joint_smoothed, joint_covariance = (
                _filter_and_smooth_joint(
                    timestamps_s[start:stop],
                    q[start:stop, joint],
                    dq[start:stop, joint],
                    **{name: value[joint] for name, value in parameters.items()},
                )
            )
            filtered[start:stop, joint] = joint_filtered
            smoothed[start:stop, joint] = joint_smoothed
            smoothed_variance[start:stop, joint] = np.diagonal(
                joint_covariance,
                axis1=-2,
                axis2=-1,
            )

    return JointStateEstimate(
        q_filtered=filtered[..., 0],
        dq_filtered=filtered[..., 1],
        ddq_filtered=filtered[..., 2],
        q_smoothed=smoothed[..., 0],
        dq_smoothed=smoothed[..., 1],
        ddq_smoothed=smoothed[..., 2],
        ddq_smoothed_std=np.sqrt(np.maximum(smoothed_variance[..., 2], 0.0)),
        segment_starts=tuple(start for start, _ in bounds),
    )


def fill_missing_measurements(
    measured: np.ndarray,
    smoothed: np.ndarray,
) -> np.ndarray:
    """Keep real measurements and use the smoother only for NaN dropouts."""

    measured = np.asarray(measured, dtype=np.float64)
    smoothed = np.asarray(smoothed, dtype=np.float64)
    if measured.shape != smoothed.shape:
        raise ValueError("measured and smoothed arrays must have the same shape.")
    return np.where(np.isfinite(measured), measured, smoothed)


def residual_torque(
    tau_measured: np.ndarray,
    tau_inverse_dynamics: np.ndarray,
) -> np.ndarray:
    """Return the repository-wide residual convention: tau_other = tau - tau_id."""

    tau_measured = np.asarray(tau_measured, dtype=np.float64)
    tau_inverse_dynamics = np.asarray(tau_inverse_dynamics, dtype=np.float64)
    if tau_measured.shape != tau_inverse_dynamics.shape:
        raise ValueError("tau_measured and tau_inverse_dynamics shapes must match.")
    if np.any(~np.isfinite(tau_measured)) or np.any(
        ~np.isfinite(tau_inverse_dynamics)
    ):
        raise ValueError("Torque arrays must be finite.")
    return tau_measured - tau_inverse_dynamics
