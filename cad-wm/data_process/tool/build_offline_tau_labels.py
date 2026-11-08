#!/usr/bin/env python3
"""Build offline inverse-dynamics residual labels in copied HDF5 episodes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import yaml

from data_process.offline_tau_labels import (
    KalmanRTSConfig,
    estimate_joint_states_rts,
    fill_missing_measurements,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate ddq with a variable-dt Kalman filter + RTS smoother, "
            "then build tau_other = tau_measured - RNEA(q, 0, 0) in copied HDF5 files."
        )
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/data_process/offline_tau_labels.yaml"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N sorted episodes (for validation runs).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace episode files that already exist in output_dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and list inputs without creating output files.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return payload


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section {key!r} must be a mapping.")
    return value


def _dataset(h5_file: h5py.File, path: str) -> h5py.Dataset:
    if path not in h5_file:
        raise KeyError(f"HDF5 dataset is missing: {path}")
    value = h5_file[path]
    if not isinstance(value, h5py.Dataset):
        raise TypeError(f"HDF5 path is not a dataset: {path}")
    return value


def _read_vector(h5_file: h5py.File, path: str) -> np.ndarray:
    value = np.asarray(_dataset(h5_file, path), dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"{path} must have shape [time, joint], got {value.shape}.")
    return value


def _write_dataset(
    h5_file: h5py.File,
    path: str,
    values: np.ndarray,
    attrs: Mapping[str, Any],
) -> None:
    values = np.asarray(values)
    if path in h5_file:
        dataset = _dataset(h5_file, path)
        if dataset.shape != values.shape:
            raise ValueError(
                f"Cannot update {path}: existing shape {dataset.shape}, "
                f"new shape {values.shape}."
            )
        dataset[...] = values.astype(dataset.dtype, copy=False)
        dataset.attrs.clear()
    else:
        group_path, _, dataset_name = path.rpartition("/")
        group = h5_file.require_group(group_path) if group_path else h5_file
        dataset = group.create_dataset(dataset_name, data=values)
    for key, value in attrs.items():
        dataset.attrs[key] = value


def _build_reduced_model(robot_config: Mapping[str, Any]):
    try:
        import pinocchio as pin
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Pinocchio is required for offline RNEA label generation."
        ) from exc

    urdf_path = Path(
        robot_config.get("urdf_path", "sim_mesh/nero/nero_with_gripper.urdf")
    )
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF does not exist: {urdf_path}")
    full_model = pin.buildModelFromUrdf(str(urdf_path))
    locked_names = tuple(
        robot_config.get(
            "locked_joint_names",
            ("gripper", "gripper_joint1", "gripper_joint2"),
        )
    )
    locked_ids = []
    for name in locked_names:
        joint_id = full_model.getJointId(str(name))
        if joint_id == full_model.njoints:
            raise ValueError(f"Pinocchio joint not found: {name}")
        locked_ids.append(joint_id)
    model = (
        pin.buildReducedModel(full_model, locked_ids, pin.neutral(full_model))
        if locked_ids
        else full_model
    )
    return pin, model, urdf_path


def _batched_rnea(pin, model, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray):
    if q.shape != dq.shape or q.shape != ddq.shape:
        raise ValueError("RNEA q, dq, and ddq arrays must have the same shape.")
    if q.shape[1] != model.nq or q.shape[1] != model.nv:
        raise ValueError(
            f"Joint width {q.shape[1]} does not match reduced model "
            f"nq={model.nq}, nv={model.nv}."
        )
    data = model.createData()
    result = np.empty_like(q, dtype=np.float64)
    for index in range(len(q)):
        result[index] = np.asarray(
            pin.rnea(model, data, q[index], dq[index], ddq[index]),
            dtype=np.float64,
        )
    return result


def _kalman_config(config: Mapping[str, Any]) -> KalmanRTSConfig:
    allowed = set(KalmanRTSConfig.__dataclass_fields__)
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown estimator options: {sorted(unknown)}")
    return KalmanRTSConfig(**config)


def _joint_sign(value: Any, joint_count: int) -> np.ndarray:
    sign = np.asarray(value if value is not None else 1.0, dtype=np.float64)
    if sign.ndim == 0:
        sign = np.full(joint_count, float(sign), dtype=np.float64)
    if sign.shape != (joint_count,) or np.any(~np.isin(sign, (-1.0, 1.0))):
        raise ValueError(
            f"processing.dq_sign must contain {joint_count} entries, each +/-1."
        )
    return sign


def _torque_filter_config(processing: Mapping[str, Any]) -> tuple[float, int]:
    filter_config = processing.get("torque_filter") or {}
    if not isinstance(filter_config, Mapping):
        raise ValueError("processing.torque_filter must be a mapping.")
    unknown = set(filter_config) - {"cutoff_hz", "median_window"}
    if unknown:
        raise ValueError(
            f"Unknown processing.torque_filter options: {sorted(unknown)}"
        )
    cutoff_hz = float(filter_config.get("cutoff_hz", 10.0))
    median_window = int(filter_config.get("median_window", 1))
    if not np.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
        raise ValueError("processing.torque_filter.cutoff_hz must be positive.")
    if median_window < 1 or median_window % 2 == 0:
        raise ValueError(
            "processing.torque_filter.median_window must be a positive odd integer."
        )
    return cutoff_hz, median_window


def _validate_measured_torque_filter(
    dataset: h5py.Dataset,
    *,
    cutoff_hz: float,
    median_window: int,
) -> None:
    attrs = dataset.attrs
    errors = []
    if not bool(attrs.get("lowpass", False)):
        errors.append("lowpass=true")
    if not bool(attrs.get("causal", False)):
        errors.append("causal=true")
    if bool(attrs.get("zero_phase", True)):
        errors.append("zero_phase=false")
    actual_cutoff = float(attrs.get("lowpass_cutoff_hz", float("nan")))
    if not np.isclose(actual_cutoff, cutoff_hz, rtol=0.0, atol=1.0e-12):
        errors.append(f"lowpass_cutoff_hz={cutoff_hz:g}")
    actual_median = int(attrs.get("median_window", -1))
    if actual_median != median_window:
        errors.append(f"median_window={median_window}")
    if errors:
        raise ValueError(
            f"Measured torque dataset {dataset.name} does not match the required "
            f"causal filter contract ({', '.join(errors)})."
        )


def process_episode(
    source_path: Path,
    destination_path: Path,
    *,
    pin,
    model,
    config: Mapping[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    if destination_path.exists() and not overwrite:
        raise FileExistsError(
            "Output already exists (pass --overwrite to replace it): "
            f"{destination_path}"
        )
    keys = _mapping(config, "keys")
    processing = _mapping(config, "processing")
    estimator_config = _kalman_config(_mapping(config, "estimator"))

    key = {
        "timestamp": str(keys.get("timestamp", "teleop/timestamp_us")),
        "q": str(keys.get("q", "teleop/q_follower")),
        "dq": str(keys.get("dq", "teleop/dq_follower")),
        "tau": str(keys.get("tau", "teleop/tau_follower")),
        "q_rts": str(keys.get("q_rts", "teleop/q_rts")),
        "dq_rts": str(keys.get("dq_rts", "teleop/dq_rts")),
        "ddq_causal": str(keys.get("ddq_causal", "teleop/ddq_kf_causal")),
        "ddq_rts": str(keys.get("ddq_rts", "teleop/ddq_follower")),
        "ddq_rts_std": str(keys.get("ddq_rts_std", "teleop/ddq_rts_std")),
        "tau_id": str(keys.get("tau_id", "teleop/tau_id_rts")),
        "tau_g": str(keys.get("tau_g", "teleop/tau_g")),
        "tau_id_filtered": str(
            keys.get("tau_id_filtered", "teleop/tau_id_rts_filtered")
        ),
        "tau_other": str(keys.get("tau_other", "teleop/tau_other_cal")),
    }
    timestamp_scale = float(processing.get("timestamp_scale_to_s", 1.0e-6))
    if not np.isfinite(timestamp_scale) or timestamp_scale <= 0.0:
        raise ValueError("processing.timestamp_scale_to_s must be positive and finite.")
    rnea_source = str(processing.get("rnea_state_source", "measured"))
    if rnea_source not in {"measured", "smoothed"}:
        raise ValueError("processing.rnea_state_source must be measured or smoothed.")
    torque_filter_hz, torque_median_window = _torque_filter_config(processing)

    temporary_path = destination_path.with_suffix(destination_path.suffix + ".building")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if temporary_path.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary_path}")
    shutil.copy2(source_path, temporary_path)

    try:
        with h5py.File(temporary_path, "r+") as h5_file:
            timestamps_s = (
                np.asarray(_dataset(h5_file, key["timestamp"]), dtype=np.float64)
                * timestamp_scale
            )
            q_measured = _read_vector(h5_file, key["q"])
            dq_measured = _read_vector(h5_file, key["dq"])
            tau_dataset = _dataset(h5_file, key["tau"])
            _validate_measured_torque_filter(
                tau_dataset,
                cutoff_hz=torque_filter_hz,
                median_window=torque_median_window,
            )
            tau_measured = np.asarray(tau_dataset, dtype=np.float64)
            if (
                q_measured.shape != dq_measured.shape
                or q_measured.shape != tau_measured.shape
            ):
                raise ValueError(
                    "Measured q, dq, and tau arrays must have the same shape."
                )

            dq_sign = _joint_sign(processing.get("dq_sign"), q_measured.shape[1])
            dq_measured = dq_measured * dq_sign[None, :]
            estimate = estimate_joint_states_rts(
                timestamps_s,
                q_measured,
                dq_measured,
                estimator_config,
            )
            q_filled = fill_missing_measurements(q_measured, estimate.q_smoothed)
            dq_filled = fill_missing_measurements(dq_measured, estimate.dq_smoothed)
            if rnea_source == "measured":
                q_rnea, dq_rnea = q_filled, dq_filled
            else:
                q_rnea, dq_rnea = estimate.q_smoothed, estimate.dq_smoothed
            tau_id = _batched_rnea(
                pin,
                model,
                q_rnea,
                dq_rnea,
                estimate.ddq_smoothed,
            )
            tau_g = _batched_rnea(
                pin,
                model,
                q_rnea,
                np.zeros_like(q_rnea),
                np.zeros_like(q_rnea),
            )
            tau_id_filtered = tau_id.copy()
            tau_other = tau_measured - tau_g

            estimator_attrs = {
                "estimator": "variable_dt_constant_acceleration_kalman_rts",
                "offline_only": True,
                "config_json": json.dumps(
                    _mapping(config, "estimator"),
                    sort_keys=True,
                ),
                "timestamp_dataset": key["timestamp"],
                "segment_starts_json": json.dumps(estimate.segment_starts),
            }
            _write_dataset(
                h5_file,
                key["dq"],
                dq_measured,
                {
                    "processing_method": "measured_velocity_with_coordinate_sign",
                    "coordinate_sign_json": json.dumps(dq_sign.astype(int).tolist()),
                },
            )
            _write_dataset(h5_file, key["q_rts"], estimate.q_smoothed, estimator_attrs)
            _write_dataset(
                h5_file,
                key["dq_rts"],
                estimate.dq_smoothed,
                estimator_attrs,
            )
            _write_dataset(
                h5_file,
                key["ddq_causal"],
                estimate.ddq_filtered,
                {
                    **estimator_attrs,
                    "offline_only": False,
                    "estimator_pass": "causal_forward_filter",
                },
            )
            _write_dataset(
                h5_file,
                key["ddq_rts"],
                estimate.ddq_smoothed,
                {**estimator_attrs, "estimator_pass": "rts_backward_smoother"},
            )
            _write_dataset(
                h5_file,
                key["ddq_rts_std"],
                estimate.ddq_smoothed_std,
                {**estimator_attrs, "quantity": "posterior_standard_deviation"},
            )
            label_attrs = {
                **estimator_attrs,
                "rnea_q_dq_source": rnea_source,
                "ddq_source": key["ddq_rts"],
            }
            _write_dataset(
                h5_file,
                key["tau_id"],
                tau_id,
                {
                    **label_attrs,
                    "definition": "RNEA(q,dq,ddq_rts)",
                    "unit": "N*m",
                },
            )
            _write_dataset(
                h5_file,
                key["tau_g"],
                tau_g,
                {
                    **label_attrs,
                    "definition": "RNEA(q,0,0) = tau_g",
                    "processing_method": "pinocchio_gravity_rnea",
                    "dq_source": "zero",
                    "ddq_source": "zero",
                    "unit": "N*m",
                },
            )
            _write_dataset(
                h5_file,
                key["tau_id_filtered"],
                tau_id_filtered,
                {
                    **label_attrs,
                    "definition": "duplicate of RNEA(q,dq,ddq_rts) = tau_id",
                    "processing_method": "duplicate_full_rnea",
                    "source_dataset": key["tau_id"],
                    "unit": "N*m",
                },
            )
            _write_dataset(
                h5_file,
                key["tau_other"],
                tau_other,
                {
                    **label_attrs,
                    "definition": "tau_measured-tau_g",
                    "formula": "tau_other=tau_measured-tau_g",
                    "target_contract": "causal_gravity_residual_v1",
                    "tau_source_dataset": key["tau"],
                    "tau_g_source_dataset": key["tau_g"],
                    "unit": "N*m",
                },
            )
            h5_file.attrs["offline_tau_labels_built"] = True
            h5_file.attrs["offline_tau_residual_convention"] = "tau_measured_minus_tau_g"
            h5_file.attrs["offline_tau_target_contract"] = "causal_gravity_residual_v1"
            h5_file.flush()
        os.replace(temporary_path, destination_path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    dt = np.diff(timestamps_s)
    return {
        "file": source_path.name,
        "frames": int(len(timestamps_s)),
        "segments": len(estimate.segment_starts),
        "missing_q_values": int(np.isnan(q_measured).sum()),
        "missing_dq_values": int(np.isnan(dq_measured).sum()),
        "dt_ms_median": float(np.median(dt) * 1.0e3),
        "dt_ms_max": float(np.max(dt) * 1.0e3),
        "ddq_abs_p99": float(np.quantile(np.abs(estimate.ddq_smoothed), 0.99)),
        "ddq_abs_max": float(np.max(np.abs(estimate.ddq_smoothed))),
        "ddq_std_p99": float(np.quantile(estimate.ddq_smoothed_std, 0.99)),
        "tau_other_abs_p99": float(np.quantile(np.abs(tau_other), 0.99)),
        "tau_other_abs_max": float(np.max(np.abs(tau_other))),
        "tau_g_abs_p99": float(np.quantile(np.abs(tau_g), 0.99)),
        "tau_id_filter_cutoff_hz": torque_filter_hz,
        "tau_id_filter_median_window": torque_median_window,
    }


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    config = load_config(args.config)
    io_config = _mapping(config, "io")
    input_dir = Path(io_config.get("input_dir", ""))
    output_dir = Path(io_config.get("output_dir", ""))
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("io.output_dir must differ from io.input_dir.")
    patterns = tuple(io_config.get("patterns", ("*.h5", "*.hdf5")))
    files = sorted({path for pattern in patterns for path in input_dir.glob(pattern)})
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No input episodes found under {input_dir}")
    if args.dry_run:
        print(json.dumps({"input_files": [str(path) for path in files]}, indent=2))
        return

    pin, model, urdf_path = _build_reduced_model(_mapping(config, "robot"))
    results = [
        process_episode(
            source,
            output_dir / source.name,
            pin=pin,
            model=model,
            config=config,
            overwrite=args.overwrite,
        )
        for source in files
    ]
    manifest = {
        "config": str(args.config),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "urdf_path": str(urdf_path),
        "residual_convention": "tau_other=tau_measured-tau_g",
        "target_contract": "causal_gravity_residual_v1",
        "results": results,
    }
    manifest_path = output_dir / "offline_tau_label_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
