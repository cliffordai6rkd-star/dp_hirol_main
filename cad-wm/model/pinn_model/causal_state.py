"""Differentiable counterpart of Nero's online joint-state estimator.

The implementation mirrors ``nero_collection.state_alignment`` for a
uniformly sampled trajectory:

    moving-average(q) -> low-pass(q) -> difference -> low-pass(dq)
                      -> difference -> low-pass(ddq)

It intentionally returns the unfiltered input position together with the
filtered derivatives, matching the values written to Nero collection files.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class CausalStateEstimatorConfig:
    sampling_dt: float
    q_mean_window_samples: int = 10
    q_lowpass_cutoff_hz: float | None = 10.0
    dq_lowpass_cutoff_hz: float | None = 6.0
    ddq_lowpass_cutoff_hz: float | None = 3.0

    def __post_init__(self):
        if not math.isfinite(self.sampling_dt) or self.sampling_dt <= 0.0:
            raise ValueError("state_estimator.sampling_dt must be positive")
        if self.q_mean_window_samples <= 0:
            raise ValueError(
                "state_estimator.q_mean_window_samples must be positive"
            )
        for name in (
            "q_lowpass_cutoff_hz",
            "dq_lowpass_cutoff_hz",
            "ddq_lowpass_cutoff_hz",
        ):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(value) or value <= 0.0
            ):
                raise ValueError(f"state_estimator.{name} must be positive or null")

    @classmethod
    def from_model_config(cls, config: Mapping):
        model_config = config.get("model") or {}
        estimator = model_config.get("state_estimator") or {}
        loss_config = config.get("loss") or {}
        sampling_dt = estimator.get(
            "sampling_dt",
            loss_config.get("sampling_dt", 1.0 / 80.0),
        )
        return cls(
            sampling_dt=float(sampling_dt),
            q_mean_window_samples=int(
                estimator.get("q_mean_window_samples", 10)
            ),
            q_lowpass_cutoff_hz=_optional_float(
                estimator.get("q_lowpass_cutoff_hz", 10.0)
            ),
            dq_lowpass_cutoff_hz=_optional_float(
                estimator.get("dq_lowpass_cutoff_hz", 6.0)
            ),
            ddq_lowpass_cutoff_hz=_optional_float(
                estimator.get("ddq_lowpass_cutoff_hz", 3.0)
            ),
        )


def causal_joint_state_from_position(
    q: torch.Tensor,
    config: CausalStateEstimatorConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw q and Nero-compatible filtered dq/ddq for ``[B,T,D]`` q.

    The filter state is initialized exactly like a fresh Nero episode: the
    moving-average window is left-filled with the first position and the
    derivative filter states start at zero. Callers should prepend the real
    observed history before a predicted future so the future rollout inherits
    a warmed, observation-conditioned filter state.
    """

    if q.ndim != 3:
        raise ValueError(f"q must have shape [B, T, D], got {tuple(q.shape)}")
    if q.shape[1] < 1 or q.shape[2] < 1:
        raise ValueError("q time and feature dimensions must be positive")
    if not q.is_floating_point():
        raise TypeError("q must be a floating-point tensor")

    dt = q.new_tensor(config.sampling_dt)
    q_alpha = _lowpass_alpha(q, config.q_lowpass_cutoff_hz, dt)
    dq_alpha = _lowpass_alpha(q, config.dq_lowpass_cutoff_hz, dt)
    ddq_alpha = _lowpass_alpha(q, config.ddq_lowpass_cutoff_hz, dt)

    window_size = config.q_mean_window_samples
    q_window = [q[:, 0]] * window_size
    q_window_sum = q[:, 0] * window_size
    previous_q_filtered = q[:, 0]
    previous_dq_filtered = torch.zeros_like(previous_q_filtered)
    previous_ddq_filtered = torch.zeros_like(previous_q_filtered)

    dq_values = [previous_dq_filtered]
    ddq_values = [previous_ddq_filtered]
    for time_index in range(1, q.shape[1]):
        q_window_sum = (
            q_window_sum - q_window[0] + q[:, time_index]
        )
        q_window = q_window[1:] + [q[:, time_index]]
        q_mean = q_window_sum / window_size
        q_filtered = _apply_one_pole(
            q_mean,
            previous_q_filtered,
            q_alpha,
        )
        dq_raw = (q_filtered - previous_q_filtered) / dt
        dq_filtered = _apply_one_pole(
            dq_raw,
            previous_dq_filtered,
            dq_alpha,
        )
        ddq_raw = (dq_filtered - previous_dq_filtered) / dt
        ddq_filtered = _apply_one_pole(
            ddq_raw,
            previous_ddq_filtered,
            ddq_alpha,
        )

        dq_values.append(dq_filtered)
        ddq_values.append(ddq_filtered)
        previous_q_filtered = q_filtered
        previous_dq_filtered = dq_filtered
        previous_ddq_filtered = ddq_filtered

    return (
        q,
        torch.stack(dq_values, dim=1),
        torch.stack(ddq_values, dim=1),
    )


def future_joint_state_from_position(
    q_history: torch.Tensor,
    q_future: torch.Tensor,
    config: CausalStateEstimatorConfig,
) -> dict[str, torch.Tensor]:
    """Warm the causal estimator on history and return a future state mapping."""

    if q_history.ndim != 3 or q_future.ndim != 3:
        raise ValueError("q_history and q_future must have shape [B, T, D]")
    if q_history.shape[0] != q_future.shape[0] or q_history.shape[2] != q_future.shape[2]:
        raise ValueError("q_history and q_future batch/feature dimensions differ")
    if q_history.shape[1] < 1 or q_future.shape[1] < 1:
        raise ValueError("q history and future horizons must be positive")

    future_horizon = q_future.shape[1]
    q_complete = torch.cat((q_history, q_future), dim=1)
    _, dq_complete, ddq_complete = causal_joint_state_from_position(
        q_complete,
        config,
    )
    return {
        "q": q_future,
        "v": dq_complete[:, -future_horizon:],
        "a": ddq_complete[:, -future_horizon:],
    }


def causal_joint_acceleration_from_velocity(
    dq: torch.Tensor,
    config: CausalStateEstimatorConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw dq and causally filtered ddq for ``[B,T,D]`` velocity."""

    if dq.ndim != 3:
        raise ValueError(f"dq must have shape [B, T, D], got {tuple(dq.shape)}")
    if dq.shape[1] < 1 or dq.shape[2] < 1:
        raise ValueError("dq time and feature dimensions must be positive")
    if not dq.is_floating_point():
        raise TypeError("dq must be a floating-point tensor")

    dt = dq.new_tensor(config.sampling_dt)
    ddq_alpha = _lowpass_alpha(dq, config.ddq_lowpass_cutoff_hz, dt)
    previous_dq = dq[:, 0]
    previous_ddq = torch.zeros_like(previous_dq)

    ddq_values = [previous_ddq]
    for time_index in range(1, dq.shape[1]):
        dq_value = dq[:, time_index]
        ddq_raw = (dq_value - previous_dq) / dt
        ddq_filtered = _apply_one_pole(ddq_raw, previous_ddq, ddq_alpha)
        ddq_values.append(ddq_filtered)
        previous_dq = dq_value
        previous_ddq = ddq_filtered

    return dq, torch.stack(ddq_values, dim=1)


def future_joint_acceleration_from_velocity(
    dq_history: torch.Tensor,
    dq_future: torch.Tensor,
    config: CausalStateEstimatorConfig,
) -> torch.Tensor:
    """Warm the causal ddq estimator on observed dq and return future ddq."""

    if dq_history.ndim != 3 or dq_future.ndim != 3:
        raise ValueError("dq_history and dq_future must have shape [B, T, D]")
    if (
        dq_history.shape[0] != dq_future.shape[0]
        or dq_history.shape[2] != dq_future.shape[2]
    ):
        raise ValueError("dq history and future batch/feature dimensions differ")
    if dq_history.shape[1] < 1 or dq_future.shape[1] < 1:
        raise ValueError("dq history and future horizons must be positive")

    future_horizon = dq_future.shape[1]
    dq_complete = torch.cat((dq_history, dq_future), dim=1)
    _, ddq_complete = causal_joint_acceleration_from_velocity(
        dq_complete,
        config,
    )
    return ddq_complete[:, -future_horizon:]


def _optional_float(value):
    return None if value is None else float(value)


def _lowpass_alpha(
    reference: torch.Tensor,
    cutoff_hz: float | None,
    dt: torch.Tensor,
) -> torch.Tensor | None:
    if cutoff_hz is None:
        return None
    return 1.0 - torch.exp(
        reference.new_tensor(-2.0 * math.pi * cutoff_hz) * dt
    )


def _apply_one_pole(
    value: torch.Tensor,
    previous: torch.Tensor,
    alpha: torch.Tensor | None,
) -> torch.Tensor:
    if alpha is None:
        return value
    return alpha * value + (1.0 - alpha) * previous
