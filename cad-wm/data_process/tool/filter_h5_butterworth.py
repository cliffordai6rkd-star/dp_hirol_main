from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


FILTER_CONTRACT = "causal_variable_dt_butterworth_lowpass_v1"
DEFAULT_CUTOFF_HZ = 20.0
FILTER_ORDER = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy H5 episodes and apply a causal second-order Butterworth "
            "low-pass to their floating-point time-series datasets."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cutoff-hz",
        type=float,
        default=DEFAULT_CUTOFF_HZ,
        help="Low-pass cutoff in Hz (default: 15).",
    )
    parser.add_argument(
        "--timestamp-path",
        default="teleop/timestamp_us",
        help="H5 timestamp dataset shared by the filtered signals.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help=(
            "Exact H5 dataset path to filter. Repeat this option to select "
            "multiple datasets; no dataset is selected implicitly."
        ),
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Only process the first N sorted episodes.",
    )
    return parser.parse_args()


def infer_sample_rate_hz(timestamps_us: np.ndarray) -> tuple[float, dict[str, float]]:
    timestamps_us = np.asarray(timestamps_us, dtype=np.float64).reshape(-1)
    if len(timestamps_us) < 2 or not np.isfinite(timestamps_us).all():
        raise ValueError("Timestamp dataset must contain at least two finite values.")
    intervals_s = np.diff(timestamps_us) * 1.0e-6
    if np.any(intervals_s <= 0.0):
        raise ValueError("Timestamp dataset must be strictly increasing.")
    median_interval_s = float(np.median(intervals_s))
    sample_rate_hz = 1.0 / median_interval_s
    return sample_rate_hz, {
        "median_interval_s": median_interval_s,
        "min_interval_s": float(np.min(intervals_s)),
        "max_interval_s": float(np.max(intervals_s)),
    }


def causal_butterworth_lowpass(
    values: np.ndarray,
    timestamps_us: np.ndarray,
    *,
    cutoff_hz: float,
) -> np.ndarray:
    """Apply the variable-dt causal Butterworth used by online inference."""

    source = np.asarray(values)
    if source.ndim < 1 or len(source) == 0:
        raise ValueError("Butterworth input must be a non-empty time series.")
    if source.dtype.kind != "f":
        raise TypeError("Butterworth input must use a floating-point dtype.")
    if not np.isfinite(source).all():
        raise ValueError("Butterworth input contains NaN or infinity.")
    validate_filter_settings(cutoff_hz)
    timestamps_us = np.asarray(timestamps_us, dtype=np.float64).reshape(-1)
    if len(timestamps_us) != len(source):
        raise ValueError(
            "Butterworth values and timestamps must have the same length."
        )
    intervals_s = np.diff(timestamps_us) * 1.0e-6
    if not np.isfinite(timestamps_us).all() or np.any(intervals_s <= 0.0):
        raise ValueError("Butterworth timestamps must be finite and increasing.")

    original_shape = source.shape
    channels = source.astype(np.float64, copy=False).reshape(len(source), -1)
    omega = 2.0 * np.pi * float(cutoff_hz)
    state_matrix = np.asarray(
        [[0.0, 1.0], [-omega * omega, -np.sqrt(2.0) * omega]],
        dtype=np.float64,
    )
    input_vector = np.asarray([0.0, omega * omega], dtype=np.float64)
    identity = np.eye(2, dtype=np.float64)
    state = np.stack((channels[0].copy(), np.zeros_like(channels[0])), axis=0)
    filtered = np.empty_like(channels)
    filtered[0] = channels[0]
    for index, dt in enumerate(intervals_s, start=1):
        left = identity - 0.5 * dt * state_matrix
        right = (
            (identity + 0.5 * dt * state_matrix) @ state
            + 0.5
            * dt
            * input_vector[:, np.newaxis]
            * (channels[index - 1] + channels[index])[np.newaxis, :]
        )
        state = np.linalg.solve(left, right)
        filtered[index] = state[0]
    return filtered.reshape(original_shape).astype(source.dtype, copy=False)


def validate_filter_settings(cutoff_hz: float) -> None:
    if not math.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
        raise ValueError("Cutoff must be positive and finite.")


def resolve_filter_datasets(
    h5_file: h5py.File,
    *,
    frame_count: int,
    requested: Sequence[str],
) -> list[str]:
    selected = list(dict.fromkeys(str(name).strip() for name in requested))
    if any(not name for name in selected):
        raise ValueError("--dataset paths must not be empty.")
    if not selected:
        raise ValueError("At least one --dataset must be specified.")

    for name in selected:
        if name not in h5_file:
            raise KeyError(f"Requested dataset is missing: {name}")
        dataset = h5_file[name]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"Requested path is not a dataset: {name}")
        if dataset.dtype.kind != "f":
            raise TypeError(f"Dataset {name!r} is not floating point: {dataset.dtype}")
        if dataset.ndim < 1 or dataset.shape[0] != frame_count:
            raise ValueError(
                f"Dataset {name!r} does not share the timestamp length "
                f"{frame_count}: shape={dataset.shape}"
            )
        if dataset.ndim >= 3 and dataset.shape[-2:] == (4, 4):
            raise ValueError(
                f"Refusing element-wise filtering of pose matrix dataset {name!r}."
            )
    return selected


def filter_file(
    source_path: Path,
    output_path: Path,
    *,
    cutoff_hz: float,
    timestamp_path: str,
    requested_datasets: Sequence[str],
) -> dict[str, object]:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite output file: {output_path}")
    temporary_path = output_path.with_suffix(output_path.suffix + ".filtering")
    if temporary_path.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, temporary_path)
    try:
        with h5py.File(temporary_path, "r+") as h5_file:
            if bool(h5_file.attrs.get("butterworth_preprocessed", False)):
                raise ValueError(f"Source already records Butterworth preprocessing: {source_path}")
            if timestamp_path not in h5_file:
                raise KeyError(f"Timestamp dataset is missing: {timestamp_path}")
            timestamps_us = np.asarray(h5_file[timestamp_path])
            inferred_hz, timing = infer_sample_rate_hz(timestamps_us)
            validate_filter_settings(cutoff_hz)
            dataset_names = resolve_filter_datasets(
                h5_file,
                frame_count=len(timestamps_us),
                requested=requested_datasets,
            )

            for name in dataset_names:
                dataset = h5_file[name]
                dataset[:] = causal_butterworth_lowpass(
                    np.asarray(dataset),
                    timestamps_us,
                    cutoff_hz=cutoff_hz,
                )
                dataset.attrs["preprocessing_filter"] = FILTER_CONTRACT
                dataset.attrs["filter_type"] = "butterworth_lowpass"
                dataset.attrs["filter_order"] = FILTER_ORDER
                dataset.attrs["filter_cutoff_hz"] = float(cutoff_hz)
                dataset.attrs["filter_causal"] = True
                dataset.attrs["filter_variable_dt"] = True
                dataset.attrs["filter_timestamp_path"] = timestamp_path
                dataset.attrs["filter_discretization"] = "trapezoidal_tustin"
                dataset.attrs["filter_initialization"] = "steady_first_sample"

            h5_file.attrs["butterworth_preprocessed"] = True
            h5_file.attrs["butterworth_filter_contract"] = FILTER_CONTRACT
            h5_file.attrs["butterworth_filter_order"] = FILTER_ORDER
            h5_file.attrs["butterworth_cutoff_hz"] = float(cutoff_hz)
            h5_file.attrs["butterworth_variable_dt"] = True
            h5_file.attrs["butterworth_discretization"] = "trapezoidal_tustin"
            h5_file.attrs["butterworth_timestamp_path"] = timestamp_path
            h5_file.attrs["butterworth_filtered_datasets_json"] = json.dumps(
                dataset_names,
                sort_keys=True,
            )
            h5_file.flush()

        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    return {
        "source": str(source_path),
        "output": str(output_path),
        "frames": int(len(timestamps_us)),
        "inferred_sample_rate_hz": float(inferred_hz),
        "median_interval_s": timing["median_interval_s"],
        "min_interval_s": timing["min_interval_s"],
        "max_interval_s": timing["max_interval_s"],
        "datasets": dataset_names,
    }


def find_h5_files(input_dir: Path) -> list[Path]:
    return sorted((*input_dir.glob("*.h5"), *input_dir.glob("*.hdf5")))


def run(args: argparse.Namespace) -> dict[str, object]:
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output_dir}")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive.")
    if not args.dataset:
        raise ValueError("At least one --dataset must be specified.")

    files = find_h5_files(input_dir)
    if args.max_episodes is not None:
        files = files[: args.max_episodes]
    if not files:
        raise FileNotFoundError(f"No H5 files found in {input_dir}")

    output_dir.mkdir(parents=True)
    results: list[dict[str, object]] = []
    try:
        for index, source_path in enumerate(files, start=1):
            print(f"[{index}/{len(files)}] filtering {source_path.name}")
            results.append(
                filter_file(
                    source_path,
                    output_dir / source_path.name,
                    cutoff_hz=float(args.cutoff_hz),
                    timestamp_path=str(args.timestamp_path),
                    requested_datasets=tuple(args.dataset),
                )
            )
    except BaseException:
        # Keep completed outputs for inspection; the missing manifest marks an
        # incomplete run and prevents accidental use as a finished dataset.
        raise

    manifest: dict[str, object] = {
        "filter_contract": FILTER_CONTRACT,
        "filter_type": "butterworth_lowpass",
        "filter_order": FILTER_ORDER,
        "cutoff_hz": float(args.cutoff_hz),
        "causal": True,
        "variable_dt": True,
        "discretization": "trapezoidal_tustin",
        "timestamp_path": str(args.timestamp_path),
        "datasets": list(dict.fromkeys(args.dataset)),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "episodes": results,
    }
    manifest_path = output_dir / "butterworth_filter_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    manifest = run(parse_args())
    print(
        f"Filtered {len(manifest['episodes'])} episodes into "
        f"{manifest['output_dir']}"
    )


if __name__ == "__main__":
    main()
