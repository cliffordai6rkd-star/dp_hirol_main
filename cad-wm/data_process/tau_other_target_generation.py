"""Build causal tau_other supervision from reusable q/dq/tau episode streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from data_process.offline_tau_labels import (
    KalmanRTSConfig,
    estimate_joint_states_causal,
)

_TIMESTAMP_SCALES = {
    "s": 1.0,
    "ms": 1.0e-3,
    "us": 1.0e-6,
    "ns": 1.0e-9,
}


@dataclass(frozen=True)
class TauOtherTargetBuildResult:
    tau_other: torch.Tensor
    dq: torch.Tensor
    ddq: torch.Tensor
    tau_id: torch.Tensor

    @property
    def tau_g(self) -> torch.Tensor:
        """Backward-compatible alias for gravity-only target consumers."""
        return self.tau_id


def normalize_tau_other_target_generation(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize the checkpointed target-generation contract."""

    raw = config.get("target_generation") or {}
    if not isinstance(raw, Mapping):
        raise ValueError("target_generation must be a mapping")
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return {"enabled": False}

    allowed = {
        "enabled",
        "method",
        "target_key",
        "timestamp_key",
        "timestamp_unit",
        "source_keys",
        "dq_sign",
        "torque_filter_key",
        "state_estimator",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown target_generation options: {unknown}")

    method = str(raw.get("method", "causal_rnea_residual_v1")).lower()
    if method not in {"causal_rnea_residual_v1", "causal_gravity_residual_v1"}:
        raise ValueError(
            "target_generation.method must be 'causal_rnea_residual_v1'"
        )

    source_keys = raw.get("source_keys") or {"q": "q", "dq": "dq", "tau": "tau"}
    if not isinstance(source_keys, Mapping):
        raise ValueError("target_generation.source_keys must be a mapping")
    missing_sources = sorted({"q", "dq", "tau"} - set(source_keys))
    unknown_sources = sorted(set(source_keys) - {"q", "dq", "tau"})
    if missing_sources or unknown_sources:
        raise ValueError(
            "target_generation.source_keys must contain exactly q, dq, and tau"
        )
    normalized_sources = {key: str(source_keys[key]) for key in ("q", "dq", "tau")}
    if any(not value for value in normalized_sources.values()):
        raise ValueError("target_generation.source_keys values must be non-empty")

    timestamp_unit = str(raw.get("timestamp_unit", "s")).lower()
    if timestamp_unit not in _TIMESTAMP_SCALES:
        raise ValueError(
            "target_generation.timestamp_unit must be one of s, ms, us, or ns"
        )

    dq_sign = raw.get("dq_sign")
    if dq_sign is not None:
        if not isinstance(dq_sign, Sequence) or isinstance(dq_sign, (str, bytes)):
            raise ValueError("target_generation.dq_sign must be a list or null")
        dq_sign = [float(value) for value in dq_sign]
        if not dq_sign or any(value not in {-1.0, 1.0} for value in dq_sign):
            raise ValueError("target_generation.dq_sign must contain only -1 or 1")

    estimator_values = raw.get("state_estimator") or {}
    if not isinstance(estimator_values, Mapping):
        raise ValueError("target_generation.state_estimator must be a mapping")
    estimator_fields = (
        "position_std",
        "velocity_std",
        "jerk_std",
        "initial_position_std",
        "initial_velocity_std",
        "initial_acceleration_std",
        "max_gap_s",
    )
    estimator_unknown = sorted(set(estimator_values) - set(estimator_fields))
    if estimator_unknown:
        raise ValueError(
            "Unknown target_generation.state_estimator options: "
            f"{estimator_unknown}"
        )
    # Construct once here so invalid estimator parameters fail during config
    # loading rather than after the dataset has been decoded.
    estimator_config = KalmanRTSConfig(**estimator_values)
    state_estimator = {
        name: getattr(estimator_config, name)
        for name in estimator_fields
    }

    return {
        "enabled": True,
        "method": method,
        "target_key": str(raw.get("target_key", "tau_other")),
        "timestamp_key": str(raw.get("timestamp_key", "timestamp")),
        "timestamp_unit": timestamp_unit,
        "source_keys": normalized_sources,
        "dq_sign": dq_sign,
        "torque_filter_key": str(raw.get("torque_filter_key", "tau")),
        "state_estimator": state_estimator,
    }


def resolve_tau_other_target_generation(
    config: Mapping[str, Any],
    dataloader_filters: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Attach the exact torque source and operations used by the target."""

    normalized = normalize_tau_other_target_generation(config)
    if not normalized["enabled"]:
        return normalized
    filter_key = normalized["torque_filter_key"]
    filter_spec = dataloader_filters.get(filter_key) or {}
    operations = (
        list(filter_spec.get("operations") or [])
        if bool(filter_spec.get("enabled", False))
        else []
    )
    if bool(filter_spec.get("enabled", False)) and operations:
        raise ValueError(
            "tau_other target generation requires measured "
            "observation.torque without a dataloader torque filter"
        )
    physics = config.get("physics") or {}
    pinocchio = physics.get("pinocchio") or {}
    if not isinstance(pinocchio, Mapping):
        raise ValueError("physics.pinocchio must be a mapping")
    data_config = config.get("dataloader") or {}
    lowdim_keys = data_config.get("lowdim_keys") or {}
    measured_tau_source = str(
        lowdim_keys.get(
            normalized["source_keys"]["tau"],
            normalized["source_keys"]["tau"],
        )
    )
    return {
        **normalized,
        "torque_filter_operations": operations,
        "pinocchio": dict(pinocchio),
        "measured_tau_source": measured_tau_source,
        "state_estimator_contract": (
            "causal_q_dq_kalman_forward_filter"
            if normalized.get("method") == "causal_rnea_residual_v1"
            else "not_used"
        ),
        "ddq_source": (
            "causal_state_estimator(q,dq)"
            if normalized.get("method") == "causal_rnea_residual_v1"
            else "unused"
        ),
        "inverse_dynamics": (
            "RNEA(q,dq,ddq;urdf)"
            if normalized.get("method") == "causal_rnea_residual_v1"
            else "gravity_torque(q)"
        ),
        "residual_formula": (
            "tau_other=tau_measured-tau_id"
            if normalized.get("method") == "causal_rnea_residual_v1"
            else "tau_other=tau_measured-tau_g"
        ),
    }


def timestamps_to_seconds(values: torch.Tensor, unit: str) -> np.ndarray:
    timestamps = torch.as_tensor(values).detach().cpu().numpy().reshape(-1)
    timestamps = timestamps.astype(np.float64) * _TIMESTAMP_SCALES[unit]
    if not np.isfinite(timestamps).all():
        raise ValueError("target-generation timestamps must be finite")
    return timestamps


def build_causal_tau_other_target(
    *,
    timestamps_s: np.ndarray,
    q: torch.Tensor,
    dq: torch.Tensor,
    tau_measured: torch.Tensor,
    episodes: Sequence[Mapping[str, Any]],
    target_config: Mapping[str, Any],
    dynamics: Any,
) -> TauOtherTargetBuildResult:
    """Build causal RNEA residual labels independently inside each episode."""

    q_np = torch.as_tensor(q).detach().cpu().to(torch.float64).numpy()
    dq_np = torch.as_tensor(dq).detach().cpu().to(torch.float64).numpy()
    tau_np = torch.as_tensor(tau_measured).detach().cpu().to(torch.float64).numpy()
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    if q_np.ndim != 2 or dq_np.shape != q_np.shape or tau_np.shape != q_np.shape:
        raise ValueError("q, dq, and tau must share shape [frames, joints]")
    if len(timestamps_s) != len(q_np):
        raise ValueError("timestamps must align with q, dq, and tau")
    if not np.isfinite(q_np).all() or not np.isfinite(dq_np).all():
        raise ValueError("causal tau_other generation requires finite q and dq")
    if not np.isfinite(tau_np).all():
        raise ValueError("causal tau_other generation requires finite measured tau")

    joint_count = q_np.shape[1]
    configured_sign = target_config.get("dq_sign")
    dq_sign = (
        np.ones(joint_count, dtype=np.float64)
        if configured_sign is None
        else np.asarray(configured_sign, dtype=np.float64)
    )
    if dq_sign.shape != (joint_count,):
        raise ValueError(
            f"target_generation.dq_sign must have {joint_count} entries"
        )
    dq_corrected = dq_np * dq_sign[None, :]
    tau_other = np.empty_like(tau_np)
    ddq = np.zeros_like(q_np)
    tau_id_all = np.empty_like(tau_np)
    dq_output = dq_corrected.copy()
    use_causal_rnea = target_config.get("method") == "causal_rnea_residual_v1"
    estimator_config = KalmanRTSConfig(**target_config["state_estimator"])

    covered = np.zeros(len(q_np), dtype=bool)
    for episode in episodes:
        start = int(episode["dataset_from_index"])
        stop = int(episode["dataset_to_index"])
        if start < 0 or stop > len(q_np) or stop - start < 2:
            raise ValueError(
                "target generation requires every episode to contain at least two "
                f"aligned frames, got [{start}, {stop})"
            )
        if covered[start:stop].any():
            raise ValueError("episode metadata overlaps during target generation")
        if use_causal_rnea:
            estimate = estimate_joint_states_causal(
                timestamps_s[start:stop],
                q_np[start:stop],
                dq_corrected[start:stop],
                estimator_config,
            )
            ddq[start:stop] = estimate.ddq_filtered
            dq_output[start:stop] = estimate.dq_filtered
            tau_id = dynamics.inverse_dynamics(
                torch.as_tensor(q_np[start:stop], dtype=torch.float64),
                torch.as_tensor(estimate.dq_filtered, dtype=torch.float64),
                torch.as_tensor(estimate.ddq_filtered, dtype=torch.float64),
            ).detach().cpu().numpy().astype(np.float64)
        else:
            tau_id = dynamics.gravity_torque(
                torch.as_tensor(q_np[start:stop], dtype=torch.float64)
            ).detach().cpu().numpy().astype(np.float64)
        tau_other[start:stop] = tau_np[start:stop] - tau_id
        tau_id_all[start:stop] = tau_id
        covered[start:stop] = True

    if not covered.all():
        missing = int((~covered).sum())
        raise ValueError(f"episode metadata leaves {missing} frames uncovered")
    return TauOtherTargetBuildResult(
        tau_other=torch.as_tensor(tau_other, dtype=torch.float32),
        dq=torch.as_tensor(dq_output, dtype=torch.float32),
        ddq=torch.as_tensor(ddq, dtype=torch.float32),
        tau_id=torch.as_tensor(tau_id_all, dtype=torch.float32),
    )
