from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy.signal import butter, sosfilt, sosfilt_zi

from data_process.causal_data_filter import (
    filter_episode_values,
    normalize_dataloader_filters,
)
from data_process.offline_tau_labels import (
    KalmanRTSConfig,
    causal_median_one_pole_filter,
    estimate_joint_states_causal,
)
from data_process.tau_other_target_generation import (
    build_causal_tau_other_target,
    normalize_tau_other_target_generation,
    resolve_tau_other_target_generation,
    timestamps_to_seconds,
)
from data_process.torque_target_filter import (
    filter_torque_target_episode,
    torque_target_filter_config,
)
from model.tau_other_sequence import build_tau_other_sequence_model
from physics.nero_dynamics import (
    PinocchioDynamics,
    damped_wrench_from_joint_torque,
)


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TorqueVisualizationTask:
    name: str
    target_key: str
    target_label: str
    prediction_label: str
    default_output_dir: Path
    rollout_mode: str


class CheckpointNormalizer:
    def __init__(self, checkpoint: Mapping[str, Any], config: Mapping[str, Any]):
        data_config = config.get("dataloader") or {}
        payload = checkpoint.get("normalizer") or {}
        self.mode = payload.get(
            "normalize_mode",
            data_config.get("normalize_mode"),
        )
        self.normalized_keys = set(
            payload.get(
                "normalize_lowdim_keys",
                data_config.get("normalize_lowdim_keys") or [],
            )
            or []
        )
        self.stats = payload.get("stats") or {}
        self.eps = float(payload.get("eps", 1.0e-6))
        if self.mode not in {None, "gaussian", "limit", "quantile"}:
            raise ValueError(f"Unsupported checkpoint normalize_mode: {self.mode!r}")

    def _stat(self, key: str, name: str, value: torch.Tensor) -> torch.Tensor:
        if key not in self.stats or name not in self.stats[key]:
            raise KeyError(
                f"Checkpoint normalizer is missing {name!r} for key {key!r}"
            )
        return torch.as_tensor(
            self.stats[key][name],
            device=value.device,
            dtype=value.dtype,
        )

    def normalize(self, key: str, value: torch.Tensor) -> torch.Tensor:
        if self.mode is None or key not in self.normalized_keys:
            return value
        if self.mode == "gaussian":
            return (
                value - self._stat(key, "mean", value)
            ) / (self._stat(key, "std", value) + self.eps)
        if self.mode == "limit":
            minimum = self._stat(key, "min", value)
            maximum = self._stat(key, "max", value)
            return 2.0 * (value - minimum) / (maximum - minimum + self.eps) - 1.0
        q01 = self._stat(key, "q01", value)
        q99 = self._stat(key, "q99", value)
        return torch.clamp(
            2.0 * (value - q01) / (q99 - q01 + self.eps) - 1.0,
            -1.0,
            1.0,
        )

    def denormalize(self, key: str, value: torch.Tensor) -> torch.Tensor:
        if self.mode is None or key not in self.normalized_keys:
            return value
        if self.mode == "gaussian":
            return value * (self._stat(key, "std", value) + self.eps) + self._stat(
                key,
                "mean",
                value,
            )
        if self.mode == "limit":
            minimum = self._stat(key, "min", value)
            maximum = self._stat(key, "max", value)
            return (value + 1.0) * (maximum - minimum + self.eps) / 2.0 + minimum
        q01 = self._stat(key, "q01", value)
        q99 = self._stat(key, "q99", value)
        return (value + 1.0) * (q99 - q01 + self.eps) / 2.0 + q01


def resolve_checkpoint_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return resolved
    if not resolved.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {resolved}")
    candidates = list(resolved.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"Checkpoint directory contains no .pt files: {resolved}")

    signatures = {}
    for candidate in candidates:
        checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
        config = checkpoint.get("config") if isinstance(checkpoint, Mapping) else None
        if not isinstance(config, Mapping):
            raise ValueError(f"Checkpoint has no config mapping: {candidate}")
        model_config = config.get("model") or {}
        data_config = config.get("dataloader") or {}
        train_config = config.get("train") or {}
        signature = json.dumps(
            {
                "target_key": model_config.get("target_key"),
                "architecture": model_config.get("architecture"),
                "inputs": model_config.get("inputs"),
                "horizon": data_config.get("horizon"),
                "split_mode": train_config.get("split_mode"),
            },
            sort_keys=True,
        )
        signatures.setdefault(signature, []).append(candidate.name)
    if len(signatures) > 1:
        raise ValueError(
            "Checkpoint directory mixes different model/split contracts; pass an "
            f"explicit .pt file instead: {resolved}"
        )

    def score(candidate: Path) -> tuple[float, float]:
        match = re.search(r"_([0-9]+(?:\.[0-9]+)?)\.pt$", candidate.name)
        monitor_score = float(match.group(1)) if match else math.inf
        return monitor_score, -candidate.stat().st_mtime

    return min(candidates, key=score).resolve()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return resolved


def load_sequence_checkpoint(
    checkpoint_path: Path,
    expected_target_key: str,
    device: torch.device,
):
    path = resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint root must be a mapping")
    for key in ("config", "model"):
        if key not in checkpoint:
            raise KeyError(f"Checkpoint is missing required key {key!r}")

    config = checkpoint["config"]
    model_config = config.get("model") or {}
    target_key = str(model_config.get("target_key", "tau_other"))
    if target_key != expected_target_key:
        raise ValueError(
            f"Checkpoint target_key={target_key!r}, expected {expected_target_key!r}"
        )

    model = build_tau_other_sequence_model(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    normalizer = CheckpointNormalizer(checkpoint, config)
    return path, checkpoint, config, model, normalizer


def _column_to_tensor(column: Any) -> torch.Tensor:
    if torch.is_tensor(column):
        return column
    if isinstance(column, list) and column and torch.is_tensor(column[0]):
        return torch.stack(column, dim=0)
    return torch.as_tensor(column)


def load_episode_columns(
    dataset,
    episode: Mapping[str, Any],
    column_mapping: Mapping[str, str],
    timestamp_key: str,
) -> dict[str, torch.Tensor]:
    start = int(episode["dataset_from_index"])
    end = int(episode["dataset_to_index"])
    dataset_columns = list(dict.fromkeys([*column_mapping.values(), timestamp_key]))
    missing = sorted(set(dataset_columns) - set(dataset.hf_dataset.column_names))
    if missing:
        raise KeyError(f"Dataset is missing columns: {missing}")

    formatted = dataset.hf_dataset.with_format(
        "torch",
        columns=dataset_columns,
        output_all_columns=False,
    )
    raw_columns = formatted[start:end]
    columns = {
        key: _column_to_tensor(raw_columns[dataset_key]).to(dtype=torch.float32)
        for key, dataset_key in column_mapping.items()
    }
    columns["timestamp"] = _column_to_tensor(
        raw_columns[timestamp_key]
    ).to(dtype=torch.float64).reshape(-1)
    return columns


def checkpoint_dataloader_filters(
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    data_config = dict(config.get("dataloader") or {})
    if "dataloader_filters" in checkpoint:
        data_config["filters"] = checkpoint["dataloader_filters"]
    return normalize_dataloader_filters(
        data_config,
        data_config.get("lowdim_keys") or {},
    )


def checkpoint_derived_target_config(
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    filters: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    resolved = checkpoint.get("derived_target_config")
    if resolved is not None:
        if not isinstance(resolved, Mapping):
            raise ValueError("checkpoint derived_target_config must be a mapping")
        return dict(resolved)
    return resolve_tau_other_target_generation(config, filters)


def apply_checkpoint_filters_to_episode(
    columns: dict[str, torch.Tensor],
    filters: Mapping[str, Mapping[str, Any]],
) -> None:
    timestamps = columns["timestamp"]
    for key, spec in filters.items():
        if key not in columns or not bool(spec.get("enabled", False)):
            continue
        operations = list(spec.get("operations", ()))
        preprocessed = list(spec.get("dataset_preprocessed_operations", ()))
        pending = operations[len(preprocessed) :]
        if not pending:
            continue
        columns[key] = torch.as_tensor(
            filter_episode_values(timestamps, columns[key], pending),
            dtype=columns[key].dtype,
        )


@torch.inference_mode()
def infer_sequence_columns(
    model: torch.nn.Module,
    normalizer: CheckpointNormalizer,
    columns: Mapping[str, torch.Tensor],
    *,
    target_key: str,
    horizon: int,
    start_frame: int,
    end_frame: int | None,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    if horizon < 1:
        raise ValueError("Checkpoint history horizon must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if target_key not in columns:
        raise KeyError(f"Episode columns are missing target {target_key!r}")

    episode_length = int(columns[target_key].shape[0])
    end_frame = episode_length if end_frame is None else int(end_frame)
    if not 0 <= start_frame < episode_length:
        raise ValueError(
            f"start_frame must be in [0, {episode_length - 1}], got {start_frame}"
        )
    if not start_frame < end_frame <= episode_length:
        raise ValueError(
            f"end_frame must be in ({start_frame}, {episode_length}], got {end_frame}"
        )

    first_target = max(int(start_frame), horizon - 1)
    if first_target >= end_frame:
        raise ValueError(
            "Selected interval contains no target with a complete history; "
            f"need end_frame > {first_target}."
        )

    active_inputs = tuple(model.active_inputs)
    missing_inputs = sorted(set(active_inputs) - set(columns))
    if missing_inputs:
        raise KeyError(f"Episode columns are missing model inputs: {missing_inputs}")

    device_columns = {
        key: columns[key].to(device=device, dtype=torch.float32)
        for key in set(active_inputs) | {target_key}
    }
    offsets = torch.arange(-horizon + 1, 1, device=device)
    prediction_batches = []
    target_batches = []
    target_positions = torch.arange(first_target, end_frame, dtype=torch.long)

    for position_batch in target_positions.split(batch_size):
        positions_device = position_batch.to(device)
        window_indices = positions_device[:, None] + offsets[None, :]
        model_batch = {
            key: normalizer.normalize(key, device_columns[key][window_indices])
            for key in active_inputs
        }
        normalized_prediction = model(model_batch)["tau_other_pred"]
        prediction = normalizer.denormalize(target_key, normalized_prediction)
        target = device_columns[target_key].index_select(0, positions_device)
        prediction_batches.append(prediction.cpu())
        target_batches.append(target.cpu())

    prediction = torch.cat(prediction_batches, dim=0).numpy()
    target = torch.cat(target_batches, dim=0).numpy()
    positions = target_positions.numpy()
    timestamps = columns["timestamp"].index_select(0, target_positions).numpy()
    return {
        "target_frame": positions,
        "timestamp_s": timestamps,
        "target_nm": target,
        "prediction_nm": prediction,
        "error_nm": prediction - target,
    }


def torque_error_metrics(result: Mapping[str, np.ndarray]) -> dict[str, Any]:
    error = np.asarray(result["error_nm"], dtype=np.float64)
    absolute_error = np.abs(error)
    joint_metrics = []
    for joint_index in range(error.shape[1]):
        joint_metrics.append(
            {
                "joint": joint_index + 1,
                "mae_nm": float(absolute_error[:, joint_index].mean()),
                "rmse_nm": float(np.sqrt(np.mean(error[:, joint_index] ** 2))),
                "bias_nm": float(error[:, joint_index].mean()),
                "p95_abs_nm": float(np.quantile(absolute_error[:, joint_index], 0.95)),
                "max_abs_nm": float(absolute_error[:, joint_index].max()),
            }
        )
    return {
        "sample_count": int(error.shape[0]),
        "overall_mae_nm": float(absolute_error.mean()),
        "overall_rmse_nm": float(np.sqrt(np.mean(error**2))),
        "overall_p95_abs_nm": float(np.quantile(absolute_error, 0.95)),
        "joint_metrics": joint_metrics,
    }


def _rollout_state_estimator_config(config: Mapping[str, Any]) -> KalmanRTSConfig:
    target_generation = normalize_tau_other_target_generation(config)
    if target_generation["enabled"]:
        return KalmanRTSConfig(**target_generation["state_estimator"])
    rollout = config.get("rollout") or {}
    values = rollout.get("state_estimator") or {}
    allowed = {
        "position_std",
        "velocity_std",
        "jerk_std",
        "initial_position_std",
        "initial_velocity_std",
        "initial_acceleration_std",
        "max_gap_s",
    }
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown rollout.state_estimator options: {unknown}")
    return KalmanRTSConfig(**values)


def causal_trailing_moving_average(
    values: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    """Apply a causal boxcar average, padding startup with the first sample."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("moving average expects non-empty values [N, D]")
    if not np.isfinite(values).all():
        raise ValueError("moving average inputs must be finite")
    if window < 1:
        raise ValueError("moving average window must be positive")
    if window == 1:
        return values.copy()
    padded = np.concatenate(
        [np.repeat(values[:1], window - 1, axis=0), values],
        axis=0,
    )
    cumulative = np.concatenate(
        [np.zeros((1, values.shape[1]), dtype=np.float64), padded.cumsum(axis=0)],
        axis=0,
    )
    return (cumulative[window:] - cumulative[:-window]) / float(window)


def causal_trailing_hampel_filter(
    values: np.ndarray,
    *,
    window: int,
    n_sigma: float,
) -> np.ndarray:
    """Replace causal trailing-window outliers using a Hampel MAD threshold."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("Hampel filter expects non-empty values [N, D]")
    if not np.isfinite(values).all():
        raise ValueError("Hampel filter inputs must be finite")
    if window < 1 or window % 2 == 0:
        raise ValueError("Hampel window must be a positive odd integer")
    if not math.isfinite(n_sigma) or n_sigma <= 0.0:
        raise ValueError("Hampel n_sigma must be positive and finite")

    filtered = values.copy()
    for index in range(len(values)):
        start = max(0, index - window + 1)
        samples = values[start : index + 1]
        if len(samples) < 3:
            continue
        median = np.median(samples, axis=0)
        mad = np.median(np.abs(samples - median), axis=0)
        robust_sigma = 1.4826 * mad
        deviation = np.abs(values[index] - median)
        is_outlier = deviation > n_sigma * robust_sigma
        filtered[index] = np.where(is_outlier, median, values[index])
    return filtered


def causal_butterworth_lowpass(
    values: np.ndarray,
    *,
    sample_rate_hz: float,
    cutoff_hz: float,
    order: int,
) -> np.ndarray:
    """Apply a causal Butterworth SOS filter with steady first-sample startup."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("Butterworth filter expects non-empty values [N, D]")
    if not np.isfinite(values).all():
        raise ValueError("Butterworth filter inputs must be finite")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("Butterworth sample_rate_hz must be positive and finite")
    if (
        not math.isfinite(cutoff_hz)
        or cutoff_hz <= 0.0
        or cutoff_hz >= 0.5 * sample_rate_hz
    ):
        raise ValueError(
            "Butterworth cutoff_hz must be positive and below the Nyquist frequency"
        )
    if order < 1:
        raise ValueError("Butterworth order must be positive")

    sos = butter(
        order,
        cutoff_hz,
        btype="lowpass",
        fs=sample_rate_hz,
        output="sos",
    )
    initial_state = (
        sosfilt_zi(sos)[:, :, np.newaxis]
        * values[0][np.newaxis, np.newaxis, :]
    )
    filtered, _ = sosfilt(sos, values, axis=0, zi=initial_state)
    return filtered


def add_external_wrench_rollout(
    rollout_mode: str,
    result: dict[str, np.ndarray],
    columns: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    *,
    dynamics: PinocchioDynamics | None = None,
    target_generation: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Reconstruct the deployment tau_ext and map it to tool wrench."""

    if rollout_mode not in {"tau_other", "tau_free"}:
        raise ValueError(f"Unknown torque rollout mode: {rollout_mode!r}")
    positions = np.asarray(result["target_frame"], dtype=np.int64)
    q = columns["q"].detach().cpu().to(torch.float64)
    dynamics = dynamics or PinocchioDynamics(config)

    if rollout_mode == "tau_free":
        tau_ext_nm = -np.asarray(result["error_nm"], dtype=np.float64)
    else:
        target_generation = (
            dict(target_generation)
            if target_generation is not None
            else normalize_tau_other_target_generation(config)
        )
        target_method = str(target_generation.get("method", "")).lower()
        if not target_generation.get("enabled", False) or target_method not in {
            "causal_rnea_residual_v1",
            "causal_gravity_residual_v1",
        }:
            raise RuntimeError(
                "tau_other rollout requires a supported causal residual contract"
            )
        required = {"tau", "timestamp"}
        missing = sorted(required - set(columns))
        if missing:
            raise KeyError(f"tau_other rollout is missing episode columns: {missing}")
        if "_tau_id" in columns:
            tau_id = columns["_tau_id"].detach().cpu().numpy()
        elif target_method == "causal_gravity_residual_v1":
            tau_id = dynamics.gravity_torque(q).cpu().numpy()
        else:
            source_keys = target_generation["source_keys"]
            derived = build_causal_tau_other_target(
                timestamps_s=columns["timestamp"].detach().cpu().numpy(),
                q=columns[source_keys["q"]],
                dq=columns[source_keys["dq"]],
                tau_measured=columns[source_keys["tau"]],
                episodes=[{"dataset_from_index": 0, "dataset_to_index": len(q)}],
                target_config=target_generation,
                dynamics=dynamics,
            )
            tau_id = derived.tau_id.cpu().numpy()
        rollout_config = config.get("rollout") or {}
        tau_measured = columns["tau"].detach().cpu().numpy().astype(np.float64)
        tau_ext_nm = (
            tau_measured[positions]
            - tau_id[positions]
            - np.asarray(result["prediction_nm"], dtype=np.float64)
        )
        result["tau_id_nm"] = tau_id[positions]
        result["tau_measured_nm"] = tau_measured[positions]

    tau_ext_raw_nm = np.asarray(tau_ext_nm, dtype=np.float64)
    rollout_config = config.get("rollout") or {}
    tau_ext_filter = rollout_config.get("tau_ext_filter") or {}
    filter_enabled = bool(tau_ext_filter.get("enabled", False))
    if filter_enabled:
        filter_mode = str(
            tau_ext_filter.get("mode", "hampel_butterworth")
        ).lower()
        filter_window = int(tau_ext_filter.get("window", 5))
        timestamps_s = np.asarray(result["timestamp_s"], dtype=np.float64)
        if filter_mode == "hampel_butterworth":
            hampel_tau_ext = causal_trailing_hampel_filter(
                tau_ext_raw_nm,
                window=filter_window,
                n_sigma=float(tau_ext_filter.get("hampel_n_sigma", 3.0)),
            )
            tau_ext_nm = causal_butterworth_lowpass(
                hampel_tau_ext,
                sample_rate_hz=float(
                    tau_ext_filter.get("sample_rate_hz", 100.0)
                ),
                cutoff_hz=float(tau_ext_filter.get("cutoff_hz", 8.0)),
                order=int(tau_ext_filter.get("order", 4)),
            )
            result["tau_ext_hampel_nm"] = hampel_tau_ext
        elif filter_mode == "moving_average":
            averaged_tau_ext = causal_trailing_moving_average(
                tau_ext_raw_nm,
                window=filter_window,
            )
            tau_ext_nm = causal_median_one_pole_filter(
                timestamps_s,
                averaged_tau_ext,
                cutoff_hz=float(tau_ext_filter.get("cutoff_hz", 20.0)),
                median_window=1,
            )
        elif filter_mode == "median":
            tau_ext_nm = causal_median_one_pole_filter(
                timestamps_s,
                tau_ext_raw_nm,
                cutoff_hz=float(tau_ext_filter.get("cutoff_hz", 20.0)),
                median_window=filter_window,
            )
        else:
            raise ValueError(
                "rollout.tau_ext_filter.mode must be hampel_butterworth, "
                "moving_average, or median, "
                f"got {filter_mode!r}"
            )
    else:
        tau_ext_nm = tau_ext_raw_nm.copy()

    q_targets = q.index_select(0, torch.as_tensor(positions, dtype=torch.long))
    frame_jacobian = dynamics.frame_jacobians(q_targets)
    damping = float((config.get("physics") or {}).get("wrench_damping", 0.02))
    tau_ext_raw_tensor = torch.as_tensor(tau_ext_raw_nm, dtype=torch.float64)
    tau_ext_tensor = torch.as_tensor(tau_ext_nm, dtype=torch.float64)
    wrench_ext_raw = damped_wrench_from_joint_torque(
        frame_jacobian,
        tau_ext_raw_tensor,
        damping=damping,
    )
    wrench_ext = damped_wrench_from_joint_torque(
        frame_jacobian,
        tau_ext_tensor,
        damping=damping,
    )

    # The damped inverse maps joint torque to wrench as A @ tau, where
    # A = (J J.T + lambda^2 I)^-1 J.  Keeping A and A[:, j] * tau_j makes it
    # possible to distinguish pose-dependent sensitivity from the wrench that
    # was actually induced by each joint's prediction residual.
    lhs = frame_jacobian @ frame_jacobian.transpose(-1, -2)
    identity = torch.eye(
        frame_jacobian.shape[-2],
        device=frame_jacobian.device,
        dtype=frame_jacobian.dtype,
    )
    lhs = lhs + damping**2 * identity
    wrench_joint_sensitivity = torch.linalg.solve(lhs, frame_jacobian)
    wrench_joint_contribution = (
        wrench_joint_sensitivity.transpose(-1, -2)
        * tau_ext_tensor.unsqueeze(-1)
    )
    result["tau_ext_raw_nm"] = tau_ext_raw_nm
    result["tau_ext_nm"] = tau_ext_nm
    result["wrench_ext_raw"] = wrench_ext_raw.cpu().numpy()
    result["wrench_ext"] = wrench_ext.cpu().numpy()
    result["wrench_joint_sensitivity"] = (
        wrench_joint_sensitivity.transpose(-1, -2).cpu().numpy()
    )
    result["wrench_joint_contribution"] = wrench_joint_contribution.cpu().numpy()
    return result


def rollout_metrics(result: Mapping[str, np.ndarray]) -> dict[str, Any]:
    tau_ext = np.asarray(result["tau_ext_nm"], dtype=np.float64)
    wrench = np.asarray(result["wrench_ext"], dtype=np.float64)
    force_norm = np.linalg.norm(wrench[:, :3], axis=1)
    moment_norm = np.linalg.norm(wrench[:, 3:], axis=1)
    metrics = {
        "tau_ext": torque_error_metrics({"error_nm": tau_ext}),
        "wrench_component_rmse": np.sqrt(np.mean(wrench**2, axis=0)).tolist(),
        "force_norm_mean_n": float(force_norm.mean()),
        "force_norm_p95_n": float(np.quantile(force_norm, 0.95)),
        "force_norm_max_n": float(force_norm.max()),
        "moment_norm_mean_nm": float(moment_norm.mean()),
        "moment_norm_p95_nm": float(np.quantile(moment_norm, 0.95)),
        "moment_norm_max_nm": float(moment_norm.max()),
    }
    if "tau_ext_hampel_nm" in result:
        tau_ext_raw = np.asarray(result["tau_ext_raw_nm"], dtype=np.float64)
        tau_ext_hampel = np.asarray(result["tau_ext_hampel_nm"], dtype=np.float64)
        replaced = ~np.isclose(tau_ext_raw, tau_ext_hampel, rtol=0.0, atol=1.0e-12)
        metrics["hampel_replacement_ratio"] = float(replaced.mean())
        metrics["hampel_replacement_ratio_by_joint"] = replaced.mean(axis=0).tolist()
    if "tau_ext_raw_nm" in result and "wrench_ext_raw" in result:
        tau_ext_raw = np.asarray(result["tau_ext_raw_nm"], dtype=np.float64)
        wrench_raw = np.asarray(result["wrench_ext_raw"], dtype=np.float64)
        force_norm_raw = np.linalg.norm(wrench_raw[:, :3], axis=1)
        moment_norm_raw = np.linalg.norm(wrench_raw[:, 3:], axis=1)
        metrics["unfiltered"] = {
            "tau_ext": torque_error_metrics({"error_nm": tau_ext_raw}),
            "wrench_component_rmse": np.sqrt(
                np.mean(wrench_raw**2, axis=0)
            ).tolist(),
            "force_norm_mean_n": float(force_norm_raw.mean()),
            "force_norm_p95_n": float(np.quantile(force_norm_raw, 0.95)),
            "force_norm_max_n": float(force_norm_raw.max()),
            "force_norm_gt_1n_ratio": float(np.mean(force_norm_raw > 1.0)),
            "force_norm_gt_2n_ratio": float(np.mean(force_norm_raw > 2.0)),
            "moment_norm_mean_nm": float(moment_norm_raw.mean()),
            "moment_norm_p95_nm": float(np.quantile(moment_norm_raw, 0.95)),
            "moment_norm_max_nm": float(moment_norm_raw.max()),
        }
        metrics["force_norm_gt_1n_ratio"] = float(np.mean(force_norm > 1.0))
        metrics["force_norm_gt_2n_ratio"] = float(np.mean(force_norm > 2.0))
    if "wrench_joint_contribution" in result:
        contribution = np.asarray(
            result["wrench_joint_contribution"], dtype=np.float64
        )
        sensitivity = np.asarray(
            result["wrench_joint_sensitivity"], dtype=np.float64
        )
        force_rms = np.sqrt(
            np.mean(np.sum(contribution[..., :3] ** 2, axis=-1), axis=0)
        )
        moment_rms = np.sqrt(
            np.mean(np.sum(contribution[..., 3:] ** 2, axis=-1), axis=0)
        )
        force_energy = force_rms**2
        moment_energy = moment_rms**2
        force_energy_share = np.divide(
            force_energy,
            force_energy.sum(),
            out=np.zeros_like(force_energy),
            where=force_energy.sum() > 0.0,
        )
        moment_energy_share = np.divide(
            moment_energy,
            moment_energy.sum(),
            out=np.zeros_like(moment_energy),
            where=moment_energy.sum() > 0.0,
        )
        metrics["joint_wrench_influence"] = [
            {
                "joint": joint_index + 1,
                "force_contribution_rms_n": float(force_rms[joint_index]),
                "moment_contribution_rms_nm": float(moment_rms[joint_index]),
                "force_contribution_energy_share": float(
                    force_energy_share[joint_index]
                ),
                "moment_contribution_energy_share": float(
                    moment_energy_share[joint_index]
                ),
                "force_sensitivity_rms_n_per_nm": float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                sensitivity[:, joint_index, :3] ** 2,
                                axis=-1,
                            )
                        )
                    )
                ),
                "moment_sensitivity_rms": float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                sensitivity[:, joint_index, 3:] ** 2,
                                axis=-1,
                            )
                        )
                    )
                ),
            }
            for joint_index in range(contribution.shape[1])
        ]
    return metrics


def _plot_indices(
    sample_count: int,
    max_plot_points: int | None,
) -> np.ndarray:
    if max_plot_points is None:
        return np.arange(sample_count)
    if max_plot_points < 1:
        raise ValueError("max_plot_points must be positive or null")
    if sample_count <= max_plot_points:
        return np.arange(sample_count)
    return np.unique(
        np.linspace(0, sample_count - 1, max_plot_points, dtype=np.int64)
    )


def save_plots(
    result: Mapping[str, np.ndarray],
    metrics: Mapping[str, Any],
    *,
    task: TorqueVisualizationTask,
    episode_index: int,
    output_dir: Path,
    max_plot_points: int | None,
    dpi: int,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    selection = _plot_indices(len(result["timestamp_s"]), max_plot_points)
    time_s = result["timestamp_s"] - result["timestamp_s"][0]
    plot_time = time_s[selection]
    target = result["target_nm"][selection]
    prediction = result["prediction_nm"][selection]
    error = result["error_nm"][selection]
    joint_count = target.shape[1]

    figure, axes = plt.subplots(
        joint_count,
        1,
        figsize=(14, 2.15 * joint_count),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for joint_index, axis in enumerate(axes):
        axis.plot(
            plot_time,
            target[:, joint_index],
            color="#2563EB",
            linewidth=0.9,
            label=task.target_label,
        )
        axis.plot(
            plot_time,
            prediction[:, joint_index],
            color="#F2B701",
            linewidth=0.9,
            alpha=0.9,
            label=task.prediction_label,
        )
        axis.set_ylabel(f"J{joint_index + 1}\nNm")
        axis.grid(True, alpha=0.3)
        if joint_index == 0:
            axis.legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("time from first prediction (s)")
    figure.suptitle(f"{task.name}: episode {episode_index} label vs prediction")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    comparison_path = output_dir / "torque_label_vs_prediction.png"
    figure.savefig(comparison_path, dpi=dpi)
    plt.close(figure)

    figure, axes = plt.subplots(
        joint_count,
        1,
        figsize=(14, 2.0 * joint_count),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for joint_index, axis in enumerate(axes):
        joint_mae = metrics["joint_metrics"][joint_index]["mae_nm"]
        axis.plot(plot_time, error[:, joint_index], color="tab:red", linewidth=0.85)
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.7)
        axis.set_ylabel(f"J{joint_index + 1}\nNm")
        axis.text(
            0.995,
            0.92,
            f"MAE {joint_mae:.4f} Nm",
            transform=axis.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
        )
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time from first prediction (s)")
    figure.suptitle(f"{task.name}: prediction error (prediction - label)")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    error_path = output_dir / "torque_prediction_error.png"
    figure.savefig(error_path, dpi=dpi)
    plt.close(figure)

    joint_labels = [f"J{index + 1}" for index in range(joint_count)]
    x = np.arange(joint_count)
    width = 0.25
    mae = [item["mae_nm"] for item in metrics["joint_metrics"]]
    rmse = [item["rmse_nm"] for item in metrics["joint_metrics"]]
    p95 = [item["p95_abs_nm"] for item in metrics["joint_metrics"]]
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(x - width, mae, width, label="MAE", color="tab:blue")
    axis.bar(x, rmse, width, label="RMSE", color="tab:orange")
    axis.bar(x + width, p95, width, label="P95 |error|", color="tab:red")
    axis.set_xticks(x, joint_labels)
    axis.set_ylabel("torque error (Nm)")
    axis.set_title(f"{task.name}: episode {episode_index} error summary")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    summary_path = output_dir / "torque_error_summary.png"
    figure.savefig(summary_path, dpi=dpi)
    plt.close(figure)
    return [comparison_path, error_path, summary_path]


def save_rollout_plots(
    result: Mapping[str, np.ndarray],
    *,
    task: TorqueVisualizationTask,
    episode_index: int,
    output_dir: Path,
    max_plot_points: int | None,
    dpi: int,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    selection = _plot_indices(len(result["timestamp_s"]), max_plot_points)
    time_s = result["timestamp_s"] - result["timestamp_s"][0]
    plot_time = time_s[selection]
    tau_ext = result["tau_ext_nm"][selection]
    tau_ext_raw = result.get("tau_ext_raw_nm", result["tau_ext_nm"])[selection]
    wrench_ext = result["wrench_ext"][selection]
    wrench_ext_raw = result.get("wrench_ext_raw", result["wrench_ext"])[selection]

    figure, axes = plt.subplots(7, 1, figsize=(14, 14), sharex=True)
    for joint_index, axis in enumerate(axes):
        axis.plot(
            plot_time,
            tau_ext_raw[:, joint_index],
            color="0.65",
            linewidth=0.65,
            alpha=0.75,
            label="raw",
        )
        axis.plot(
            plot_time,
            tau_ext[:, joint_index],
            color="#2563EB",
            linewidth=0.85,
            label="filtered",
        )
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.7)
        axis.set_ylabel(f"J{joint_index + 1}\nNm")
        axis.grid(True, alpha=0.3)
        if joint_index == 0:
            axis.legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("time from first prediction (s)")
    figure.suptitle(f"{task.name}: episode {episode_index} tau_ext rollout")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    tau_path = output_dir / "tau_ext_rollout.png"
    figure.savefig(tau_path, dpi=dpi)
    plt.close(figure)

    labels = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
    units = ("N", "N", "N", "Nm", "Nm", "Nm")
    figure, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
    for component, axis in enumerate(axes):
        color = "#2563EB" if component < 3 else "#F2B701"
        axis.plot(
            plot_time,
            wrench_ext_raw[:, component],
            color="0.65",
            linewidth=0.65,
            alpha=0.75,
            label="raw",
        )
        axis.plot(
            plot_time,
            wrench_ext[:, component],
            color=color,
            linewidth=0.85,
            label="filtered",
        )
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.7)
        axis.set_ylabel(f"{labels[component]}\n{units[component]}")
        axis.grid(True, alpha=0.3)
        if component == 0:
            axis.legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("time from first prediction (s)")
    figure.suptitle(f"{task.name}: episode {episode_index} wrench_ext rollout")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    wrench_path = output_dir / "wrench_ext_rollout.png"
    figure.savefig(wrench_path, dpi=dpi)
    plt.close(figure)

    plot_paths = [tau_path, wrench_path]
    if "wrench_joint_contribution" in result:
        contribution = np.asarray(result["wrench_joint_contribution"])
        force_rms = np.sqrt(
            np.mean(np.sum(contribution[..., :3] ** 2, axis=-1), axis=0)
        )
        moment_rms = np.sqrt(
            np.mean(np.sum(contribution[..., 3:] ** 2, axis=-1), axis=0)
        )
        joint_labels = [f"J{index + 1}" for index in range(contribution.shape[1])]
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].bar(joint_labels, force_rms, color="#2563EB")
        axes[0].set_ylabel("RMS force contribution (N)")
        axes[0].set_title("Per-joint force influence")
        axes[1].bar(joint_labels, moment_rms, color="#F2B701")
        axes[1].set_ylabel("RMS moment contribution (Nm)")
        axes[1].set_title("Per-joint moment influence")
        for axis in axes:
            axis.grid(True, axis="y", alpha=0.3)
        figure.suptitle(f"{task.name}: episode {episode_index} joint wrench influence")
        figure.tight_layout()
        influence_path = output_dir / "joint_wrench_influence.png"
        figure.savefig(influence_path, dpi=dpi)
        plt.close(figure)
        plot_paths.append(influence_path)
    return plot_paths


def _select_episodes(dataset, requested: list[int] | None, all_episodes: bool):
    episodes = list(dataset.meta.episodes)
    by_index = {
        int(episode.get("episode_index", position)): episode
        for position, episode in enumerate(episodes)
    }
    if all_episodes or not requested:
        return [(index, by_index[index]) for index in sorted(by_index)]
    missing = sorted(set(requested) - set(by_index))
    if missing:
        raise ValueError(
            f"Unknown episode indices {missing}; available={sorted(by_index)}"
        )
    return [(index, by_index[index]) for index in requested]


def load_visualization_dataset(config, *, root=None, repo_id=None, video_backend=None):
    """Load the checkpoint's dataset backend without changing time semantics."""

    data_config = config.get("dataloader") or {}
    backend = str(data_config.get("backend", "lerobot")).lower()
    configured_root = data_config.get("root")
    resolved_root = Path(root) if root is not None else (
        Path(configured_root) if configured_root is not None else None
    )
    if resolved_root is None:
        raise ValueError("Dataset root must be provided by the checkpoint or CLI")

    if backend == "h5":
        from data_process.dataloader import PINNDataset

        config.setdefault("dataloader", {})["root"] = str(resolved_root)
        return (
            PINNDataset(config, compute_normalizer=False).dataset,
            resolved_root,
            None,
        )
    if backend != "lerobot":
        raise ValueError(
            "dataloader.backend must be 'lerobot' or 'h5', "
            f"got {backend!r}"
        )

    resolved_repo_id = repo_id or data_config.get("repo_id")
    if not resolved_repo_id:
        raise ValueError(
            "LeRobot repo-id must be provided by the checkpoint or CLI"
        )
    resolved_video_backend = video_backend or data_config.get(
        "video_backend", "torchcodec"
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return (
        LeRobotDataset(
            repo_id=resolved_repo_id,
            root=resolved_root,
            video_backend=resolved_video_backend,
        ),
        resolved_root,
        resolved_repo_id,
    )


def build_parser(task: TorqueVisualizationTask) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run {task.name} checkpoint inference on H5/LeRobot data and plot it."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint .pt file or a directory whose lowest-score .pt is selected.",
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--video-backend", default=None)
    parser.add_argument(
        "--episode-index",
        type=int,
        action="append",
        default=None,
        help=(
            "Episode index to process; repeat for multiple episodes. "
            "When omitted, all episodes are processed."
        ),
    )
    parser.add_argument("--all-episodes", action="store_true")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--timestamp-key", default="timestamp")
    parser.add_argument(
        "--frame-name",
        default="gripper_tcp",
        help=(
            "Pinocchio frame used for wrench rollout. Defaults to the closed-"
            "fingertip center and overrides the frame stored in old checkpoints."
        ),
    )
    parser.add_argument(
        "--tau-ext-filter-mode",
        choices=("hampel_butterworth", "moving_average", "median"),
        default="hampel_butterworth",
        help=(
            "tau_ext filter chain. The default applies a causal Hampel outlier "
            "filter followed by a causal Butterworth low-pass."
        ),
    )
    parser.add_argument(
        "--tau-ext-filter-window",
        "--tau-ext-median-window",
        dest="tau_ext_filter_window",
        type=int,
        default=5,
        help=(
            "Causal Hampel, moving-average, or median window. The old "
            "median-window option remains as an alias."
        ),
    )
    parser.add_argument(
        "--tau-ext-hampel-n-sigma",
        type=float,
        default=3.0,
        help="Hampel outlier threshold in robust standard deviations.",
    )
    parser.add_argument(
        "--tau-ext-butterworth-order",
        type=int,
        default=4,
        help="Order of the causal Butterworth low-pass.",
    )
    parser.add_argument(
        "--tau-ext-sample-rate-hz",
        type=float,
        default=100.0,
        help="Fixed tau_ext sampling rate used to design the Butterworth filter.",
    )
    parser.add_argument(
        "--tau-ext-lowpass-cutoff-hz",
        type=float,
        default=8.0,
        help="Low-pass cutoff in Hz; defaults to 8 Hz.",
    )
    parser.add_argument(
        "--no-tau-ext-filter",
        action="store_true",
        help="Disable all tau_ext outlier and low-pass filtering.",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=None,
        help=(
            "Optional plot downsampling limit. By default every inference point "
            "is plotted."
        ),
    )
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=task.default_output_dir,
    )
    return parser


def run_visualization(task: TorqueVisualizationTask, args: argparse.Namespace) -> None:
    if args.all_episodes and args.episode_index:
        raise ValueError("Use either --all-episodes or --episode-index, not both")

    device = resolve_device(args.device)
    path, checkpoint, config, model, normalizer = load_sequence_checkpoint(
        args.checkpoint,
        task.target_key,
        device,
    )
    physics_config = config.setdefault("physics", {})
    pinocchio_config = physics_config.setdefault("pinocchio", {})
    pinocchio_config["frame_name"] = str(args.frame_name)
    rollout_config = config.setdefault("rollout", {})
    rollout_config["tau_ext_filter"] = {
        "enabled": not bool(args.no_tau_ext_filter),
        "mode": str(args.tau_ext_filter_mode),
        "window": int(args.tau_ext_filter_window),
        "cutoff_hz": float(args.tau_ext_lowpass_cutoff_hz),
        "hampel_n_sigma": float(args.tau_ext_hampel_n_sigma),
        "order": int(args.tau_ext_butterworth_order),
        "sample_rate_hz": float(args.tau_ext_sample_rate_hz),
    }
    data_config = config.get("dataloader") or {}
    filters = checkpoint_dataloader_filters(checkpoint, config)
    data_config["filters"] = filters
    derived_target_config = checkpoint_derived_target_config(
        checkpoint,
        config,
        filters,
    )
    model_config = config.get("model") or {}
    lowdim_keys = data_config.get("lowdim_keys") or {}
    active_inputs = list(model_config.get("inputs") or [])
    rollout_keys = ["q"] if task.rollout_mode == "tau_free" else ["q", "dq", "tau"]
    build_target_from_sources = bool(
        task.rollout_mode == "tau_other"
        and derived_target_config.get("enabled", False)
        and derived_target_config.get("target_key") == task.target_key
    )
    target_keys = (
        list(derived_target_config["source_keys"].values())
        if build_target_from_sources
        else [task.target_key]
    )
    required_keys = list(
        dict.fromkeys([*active_inputs, *target_keys, *rollout_keys])
    )
    missing_mappings = [key for key in required_keys if key not in lowdim_keys]
    if missing_mappings:
        raise KeyError(
            f"Checkpoint dataloader.lowdim_keys is missing: {missing_mappings}"
        )

    dataset, root, repo_id = load_visualization_dataset(
        config,
        root=args.root,
        repo_id=args.repo_id,
        video_backend=args.video_backend,
    )
    selected_episodes = _select_episodes(
        dataset,
        args.episode_index,
        args.all_episodes,
    )
    horizon = int(data_config.get("horizon", 50))
    timestamp_dataset_key = (
        str(derived_target_config["timestamp_key"])
        if build_target_from_sources
        else str(args.timestamp_key)
    )
    dynamics = PinocchioDynamics(config) if task.rollout_mode == "tau_other" else None
    log.info(
        "%s inference: checkpoint=%s dataset=%s episodes=%s device=%s horizon=%d",
        task.name,
        path,
        root,
        [index for index, _ in selected_episodes],
        device,
        horizon,
    )

    for episode_index, episode in selected_episodes:
        columns = load_episode_columns(
            dataset,
            episode,
            {key: lowdim_keys[key] for key in required_keys},
            timestamp_dataset_key,
        )
        if build_target_from_sources:
            columns["timestamp"] = torch.as_tensor(
                timestamps_to_seconds(
                    columns["timestamp"],
                    derived_target_config["timestamp_unit"],
                ),
                dtype=torch.float64,
            )
        apply_checkpoint_filters_to_episode(columns, filters)
        if build_target_from_sources:
            source_keys = derived_target_config["source_keys"]
            derived = build_causal_tau_other_target(
                timestamps_s=columns["timestamp"].numpy(),
                q=columns[source_keys["q"]],
                dq=columns[source_keys["dq"]],
                tau_measured=columns[source_keys["tau"]],
                episodes=[
                    {
                        "dataset_from_index": 0,
                        "dataset_to_index": len(columns["timestamp"]),
                    }
                ],
                target_config=derived_target_config,
                dynamics=dynamics,
            )
            columns[source_keys["dq"]] = derived.dq
            columns[task.target_key] = derived.tau_other
            columns["_tau_id"] = derived.tau_id
        target_filter = torque_target_filter_config(config)
        if (
            task.rollout_mode == "tau_free"
            and not filters
            and target_filter is not None
        ):
            columns[task.target_key] = torch.as_tensor(
                filter_torque_target_episode(
                    columns["timestamp"],
                    columns[task.target_key],
                    target_filter,
                ),
                dtype=columns[task.target_key].dtype,
            )
        result = infer_sequence_columns(
            model,
            normalizer,
            columns,
            target_key=task.target_key,
            horizon=horizon,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            batch_size=args.batch_size,
            device=device,
        )
        result = add_external_wrench_rollout(
            task.rollout_mode,
            result,
            columns,
            config,
            dynamics=dynamics,
            target_generation=derived_target_config,
        )
        metrics = torque_error_metrics(result)
        metrics["rollout"] = rollout_metrics(result)
        episode_output_dir = args.output_dir / f"episode_{episode_index:03d}"
        plot_paths = save_plots(
            result,
            metrics,
            task=task,
            episode_index=episode_index,
            output_dir=episode_output_dir,
            max_plot_points=args.max_plot_points,
            dpi=args.dpi,
        )
        plot_paths.extend(
            save_rollout_plots(
                result,
                task=task,
                episode_index=episode_index,
                output_dir=episode_output_dir,
                max_plot_points=args.max_plot_points,
                dpi=args.dpi,
            )
        )
        npz_path = episode_output_dir / "inference_data.npz"
        np.savez_compressed(npz_path, **result)
        report = {
            "task": task.name,
            "checkpoint": str(path),
            "dataset_root": str(Path(root).resolve()),
            "repo_id": repo_id,
            "episode_index": episode_index,
            "history_horizon": horizon,
            "target_key": task.target_key,
            "derived_target_config": derived_target_config,
            "active_inputs": active_inputs,
            "rollout_mode": task.rollout_mode,
            "wrench_frame_name": str(args.frame_name),
            "tau_ext_filter": rollout_config["tau_ext_filter"],
            "metrics": metrics,
            "artifacts": {
                "inference_data": str(npz_path),
                "plots": [str(plot_path) for plot_path in plot_paths],
            },
        }
        report_path = episode_output_dir / "metrics.json"
        with report_path.open("w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2, sort_keys=True)
            report_file.write("\n")

        joint_mae = ", ".join(
            f"{item['mae_nm']:.4f}" for item in metrics["joint_metrics"]
        )
        log.info(
            "episode=%d samples=%d MAE=%.6f Nm RMSE=%.6f Nm "
            "joint_MAE=[%s] output=%s",
            episode_index,
            metrics["sample_count"],
            metrics["overall_mae_nm"],
            metrics["overall_rmse_nm"],
            joint_mae,
            episode_output_dir,
        )
