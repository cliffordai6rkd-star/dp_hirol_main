#!/usr/bin/env python3
"""Recompute LeRobot v3 joint acceleration from low-pass filtered velocity.

Example:
    python data_process/tool/lerobot_recompute_acceleration.py \
        --input-root /path/to/lerobot_v3_dataset \
        --cutoff-hz 10

The acceleration column and its entry in ``meta/stats.json`` are replaced in
place. Additional low-dimensional columns can be filtered in place with
``--filter-keys``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class ParquetSlice:
    path: Path
    start: int
    stop: int
    indexes: np.ndarray
    row_group_sizes: tuple[int, ...]


@dataclass(frozen=True)
class DatasetFrames:
    files: tuple[ParquetSlice, ...]
    indexes: np.ndarray
    episode_indexes: np.ndarray
    frame_indexes: np.ndarray
    timestamps: np.ndarray
    velocities: np.ndarray
    old_accelerations: np.ndarray
    filter_values: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Low-pass observation.velocity, differentiate it per episode, and "
            "overwrite observation.acceleration in a LeRobot v3 dataset."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="LeRobot v3 dataset root containing data/ and meta/.",
    )
    parser.add_argument(
        "--cutoff-hz",
        type=float,
        default=10.0,
        help="Cutoff frequency of the causal first-order low-pass filter. Default: 10 Hz.",
    )
    parser.add_argument("--velocity-key", default="observation.velocity")
    parser.add_argument("--acceleration-key", default="observation.acceleration")
    parser.add_argument("--timestamp-key", default="timestamp")
    parser.add_argument("--episode-key", default="episode_index")
    parser.add_argument("--frame-key", default="frame_index")
    parser.add_argument("--index-key", default="index")
    parser.add_argument(
        "--filter-keys",
        nargs="+",
        default=[],
        metavar="KEY",
        help=(
            "Additional 1D float features to low-pass and overwrite in place. "
            "Multiple keys may be provided."
        ),
    )
    parser.add_argument(
        "--min-dt",
        type=float,
        default=1e-6,
        help="Clamp positive timestamp deltas to at least this many seconds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and report the result without changing the dataset.",
    )
    return parser.parse_args()


def parquet_files(root: Path) -> list[Path]:
    files = sorted((root / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")
    return files


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON metadata file: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def validate_v3_metadata(
    root: Path,
    *,
    velocity_key: str,
    acceleration_key: str,
    filter_keys: list[str],
) -> tuple[dict[str, Any], Path]:
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"
    info = load_json(info_path)
    load_json(stats_path)

    version = str(info.get("codebase_version", ""))
    if not version.startswith("v3"):
        raise ValueError(
            f"Expected a LeRobot v3 dataset, but {info_path} has codebase_version={version!r}."
        )

    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"Missing features object in {info_path}")
    for key in dict.fromkeys((velocity_key, acceleration_key, *filter_keys)):
        if key not in features:
            raise KeyError(f"Missing feature {key!r} in {info_path}")

    velocity_shape = tuple(features[velocity_key].get("shape", ()))
    acceleration_shape = tuple(features[acceleration_key].get("shape", ()))
    if len(velocity_shape) != 1 or velocity_shape != acceleration_shape:
        raise ValueError(
            f"Velocity and acceleration must have the same 1D shape, got "
            f"{velocity_shape} and {acceleration_shape}."
        )
    for key in filter_keys:
        shape = tuple(features[key].get("shape", ()))
        dtype = str(features[key].get("dtype", ""))
        if len(shape) != 1 or not dtype.startswith("float"):
            raise ValueError(
                f"--filter-keys only supports 1D float features; {key!r} has "
                f"shape={shape}, dtype={dtype!r}."
            )
    return info, stats_path


def scalar_column_to_numpy(table: pa.Table, key: str) -> np.ndarray:
    column = table[key].combine_chunks()
    if column.null_count:
        raise ValueError(f"Column {key!r} contains null values.")

    if pa.types.is_fixed_size_list(column.type):
        if column.type.list_size != 1:
            raise ValueError(f"Column {key!r} must be scalar, got {column.type}.")
        column = column.values
    elif pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
        values = column.to_pylist()
        if any(len(value) != 1 for value in values):
            raise ValueError(f"Column {key!r} must contain one scalar per row.")
        return np.asarray([value[0] for value in values])

    values = np.asarray(column.to_numpy(zero_copy_only=False))
    if values.ndim != 1 or len(values) != table.num_rows:
        raise ValueError(f"Column {key!r} must contain one scalar per row.")
    return values


def vector_column_to_numpy(table: pa.Table, key: str) -> np.ndarray:
    column = table[key].combine_chunks()
    if column.null_count:
        raise ValueError(f"Column {key!r} contains null values.")

    if pa.types.is_fixed_size_list(column.type):
        flat = column.values
        if flat.null_count:
            raise ValueError(f"Column {key!r} contains null values.")
        values = np.asarray(flat.to_numpy(zero_copy_only=False))
        return values.reshape(table.num_rows, column.type.list_size)

    if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
        values = np.asarray(column.to_pylist())
        if values.ndim != 2:
            raise ValueError(f"Column {key!r} must contain equal-length vectors.")
        return values

    raise TypeError(f"Column {key!r} must be a list vector, got {column.type}.")


def require_integer_column(values: np.ndarray, key: str) -> np.ndarray:
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError(f"Column {key!r} must have an integer dtype, got {values.dtype}.")
    return values.astype(np.int64, copy=False)


def load_dataset_frames(
    root: Path,
    *,
    velocity_key: str,
    acceleration_key: str,
    timestamp_key: str,
    episode_key: str,
    frame_key: str,
    index_key: str,
    filter_keys: list[str],
) -> DatasetFrames:
    required_keys = list(
        dict.fromkeys(
            (
                index_key,
                episode_key,
                frame_key,
                timestamp_key,
                velocity_key,
                acceleration_key,
                *filter_keys,
            )
        )
    )
    file_slices: list[ParquetSlice] = []
    indexes: list[np.ndarray] = []
    episode_indexes: list[np.ndarray] = []
    frame_indexes: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    old_accelerations: list[np.ndarray] = []
    filter_values: dict[str, list[np.ndarray]] = {key: [] for key in filter_keys}
    vector_width: int | None = None
    offset = 0

    for path in parquet_files(root):
        parquet_file = pq.ParquetFile(path)
        available = set(parquet_file.schema_arrow.names)
        missing = [key for key in required_keys if key not in available]
        if missing:
            raise KeyError(f"Missing columns {missing} in {path}")

        table = parquet_file.read(columns=required_keys)
        file_indexes = require_integer_column(scalar_column_to_numpy(table, index_key), index_key)
        file_episodes = require_integer_column(
            scalar_column_to_numpy(table, episode_key), episode_key
        )
        file_frames = require_integer_column(scalar_column_to_numpy(table, frame_key), frame_key)
        file_timestamps = scalar_column_to_numpy(table, timestamp_key).astype(np.float64)
        file_velocities = vector_column_to_numpy(table, velocity_key).astype(np.float64)
        file_accelerations = vector_column_to_numpy(table, acceleration_key).astype(np.float64)
        file_filter_values = {
            key: (
                file_velocities
                if key == velocity_key
                else file_accelerations
                if key == acceleration_key
                else vector_column_to_numpy(table, key).astype(np.float64)
            )
            for key in filter_keys
        }

        row_count = table.num_rows
        if any(
            len(values) != row_count
            for values in (
                file_indexes,
                file_episodes,
                file_frames,
                file_timestamps,
                file_velocities,
                file_accelerations,
                *file_filter_values.values(),
            )
        ):
            raise ValueError(f"Column length mismatch in {path}")
        if file_velocities.shape != file_accelerations.shape:
            raise ValueError(
                f"Velocity/acceleration shape mismatch in {path}: "
                f"{file_velocities.shape} vs {file_accelerations.shape}."
            )
        if vector_width is None:
            vector_width = file_velocities.shape[1]
        elif file_velocities.shape[1] != vector_width:
            raise ValueError(f"Velocity vector width changes in {path}.")

        stop = offset + row_count
        row_group_sizes = tuple(
            parquet_file.metadata.row_group(i).num_rows
            for i in range(parquet_file.num_row_groups)
        )
        file_slices.append(
            ParquetSlice(
                path=path,
                start=offset,
                stop=stop,
                indexes=file_indexes.copy(),
                row_group_sizes=row_group_sizes,
            )
        )
        indexes.append(file_indexes)
        episode_indexes.append(file_episodes)
        frame_indexes.append(file_frames)
        timestamps.append(file_timestamps)
        velocities.append(file_velocities)
        old_accelerations.append(file_accelerations)
        for key, values in file_filter_values.items():
            filter_values[key].append(values)
        offset = stop

    loaded = DatasetFrames(
        files=tuple(file_slices),
        indexes=np.concatenate(indexes),
        episode_indexes=np.concatenate(episode_indexes),
        frame_indexes=np.concatenate(frame_indexes),
        timestamps=np.concatenate(timestamps),
        velocities=np.concatenate(velocities, axis=0),
        old_accelerations=np.concatenate(old_accelerations, axis=0),
        filter_values={
            key: np.concatenate(value_chunks, axis=0)
            for key, value_chunks in filter_values.items()
        },
    )
    validate_loaded_frames(loaded)
    return loaded


def validate_loaded_frames(frames: DatasetFrames) -> None:
    if len(frames.indexes) == 0:
        raise ValueError("The dataset contains no frames.")
    if len(np.unique(frames.indexes)) != len(frames.indexes):
        raise ValueError("The dataset contains duplicate values in the index column.")
    if not np.all(np.isfinite(frames.timestamps)):
        raise ValueError("The timestamp column contains NaN or infinite values.")
    if not np.all(np.isfinite(frames.velocities)):
        raise ValueError("The velocity column contains NaN or infinite values.")
    if not np.all(np.isfinite(frames.old_accelerations)):
        raise ValueError("The existing acceleration column contains NaN or infinite values.")
    for key, values in frames.filter_values.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"The selected filter column {key!r} contains NaN or infinite values.")


def lowpass_values(
    values: np.ndarray,
    timestamps: np.ndarray,
    *,
    cutoff_hz: float,
    min_dt: float,
) -> tuple[np.ndarray, int]:
    """Apply a causal first-order RC low-pass filter."""
    values = np.asarray(values, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"values must be 2D, got shape {values.shape}.")
    if timestamps.ndim != 1 or len(timestamps) != len(values):
        raise ValueError("timestamps must be 1D and match the values length.")
    if len(values) == 0:
        return values.copy(), 0
    if not math.isfinite(cutoff_hz) or cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be a positive finite number.")
    if not math.isfinite(min_dt) or min_dt <= 0:
        raise ValueError("min_dt must be a positive finite number.")

    timestamp_deltas = np.diff(timestamps)
    bad_delta = np.flatnonzero(timestamp_deltas <= 0)
    if len(bad_delta):
        idx = int(bad_delta[0])
        raise ValueError(
            "Timestamps must be strictly increasing within an episode: "
            f"t[{idx}]={timestamps[idx]:.9g}, t[{idx + 1}]={timestamps[idx + 1]:.9g}."
        )

    filtered = np.empty_like(values)
    filtered[0] = values[0]
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    clamped_dt_count = 0

    for idx in range(1, len(values)):
        raw_dt = float(timestamp_deltas[idx - 1])
        dt = max(raw_dt, min_dt)
        if dt != raw_dt:
            clamped_dt_count += 1
        alpha = dt / (rc + dt)
        filtered[idx] = filtered[idx - 1] + alpha * (values[idx] - filtered[idx - 1])

    return filtered, clamped_dt_count


def lowpass_velocity_and_differentiate(
    velocities: np.ndarray,
    timestamps: np.ndarray,
    *,
    cutoff_hz: float,
    min_dt: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Low-pass velocity, then compute acceleration using timestamp differences."""
    filtered, clamped_dt_count = lowpass_values(
        velocities,
        timestamps,
        cutoff_hz=cutoff_hz,
        min_dt=min_dt,
    )
    accelerations = np.zeros_like(filtered)
    for idx, raw_dt in enumerate(np.diff(timestamps), start=1):
        dt = max(float(raw_dt), min_dt)
        accelerations[idx] = (filtered[idx] - filtered[idx - 1]) / dt
    return filtered, accelerations, clamped_dt_count


def compute_outputs_by_episode(
    frames: DatasetFrames,
    *,
    cutoff_hz: float,
    min_dt: float,
    velocity_key: str,
    acceleration_key: str,
    filter_keys: list[str],
) -> tuple[dict[str, np.ndarray], int, np.ndarray]:
    order = np.lexsort((frames.indexes, frames.frame_indexes, frames.episode_indexes))
    ordered_episodes = frames.episode_indexes[order]
    episode_starts = np.r_[0, np.flatnonzero(np.diff(ordered_episodes) != 0) + 1]
    episode_stops = np.r_[episode_starts[1:], len(order)]
    outputs = {acceleration_key: np.empty_like(frames.velocities)}
    for key in filter_keys:
        if key != acceleration_key:
            outputs[key] = np.empty_like(frames.filter_values[key])
    positive_dts: list[np.ndarray] = []
    clamped_dt_count = 0

    for start, stop in zip(episode_starts, episode_stops, strict=True):
        positions = order[start:stop]
        episode = int(frames.episode_indexes[positions[0]])
        episode_frames = frames.frame_indexes[positions]
        duplicate_or_reversed = np.flatnonzero(np.diff(episode_frames) <= 0)
        if len(duplicate_or_reversed):
            idx = int(duplicate_or_reversed[0])
            raise ValueError(
                f"frame_index must increase within episode {episode}: "
                f"{episode_frames[idx]} then {episode_frames[idx + 1]}."
            )

        episode_timestamps = frames.timestamps[positions]
        try:
            filtered_velocity, episode_accelerations, episode_clamped_count = (
                lowpass_velocity_and_differentiate(
                    frames.velocities[positions],
                    episode_timestamps,
                    cutoff_hz=cutoff_hz,
                    min_dt=min_dt,
                )
            )
        except ValueError as error:
            raise ValueError(f"Invalid data in episode {episode}: {error}") from error

        outputs[acceleration_key][positions] = episode_accelerations
        for key in filter_keys:
            if key == velocity_key:
                filtered_values = filtered_velocity
            elif key == acceleration_key:
                filtered_values, _ = lowpass_values(
                    episode_accelerations,
                    episode_timestamps,
                    cutoff_hz=cutoff_hz,
                    min_dt=min_dt,
                )
            else:
                filtered_values, _ = lowpass_values(
                    frames.filter_values[key][positions],
                    episode_timestamps,
                    cutoff_hz=cutoff_hz,
                    min_dt=min_dt,
                )
            outputs[key][positions] = filtered_values
        clamped_dt_count += episode_clamped_count
        if len(episode_timestamps) > 1:
            positive_dts.append(np.diff(episode_timestamps))

    all_dts = np.concatenate(positive_dts) if positive_dts else np.empty(0, dtype=np.float64)
    return outputs, clamped_dt_count, all_dts


def vector_arrow_array(values: np.ndarray, field: pa.Field) -> pa.Array:
    arrow_type = field.type
    if pa.types.is_fixed_size_list(arrow_type):
        if values.shape[1] != arrow_type.list_size:
            raise ValueError(
                f"Computed value width {values.shape[1]} does not match {arrow_type}."
            )
        if not pa.types.is_floating(arrow_type.value_type):
            raise TypeError(f"Filtered values must be floating point, got {arrow_type}.")
        flat = pa.array(values.reshape(-1), type=arrow_type.value_type)
        return pa.FixedSizeListArray.from_arrays(flat, arrow_type.list_size)

    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        if not pa.types.is_floating(arrow_type.value_type):
            raise TypeError(f"Filtered values must be floating point, got {arrow_type}.")
        return pa.array(values.tolist(), type=arrow_type)

    raise TypeError(f"Filtered column must be a list vector, got {arrow_type}.")


def create_temp_path(target: Path) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temp_path = Path(raw_path)
    os.chmod(temp_path, target.stat().st_mode)
    return temp_path


def write_parquet_temp(
    table: pa.Table,
    *,
    target: Path,
    row_group_sizes: tuple[int, ...],
) -> Path:
    temp_path = create_temp_path(target)
    try:
        with pq.ParquetWriter(temp_path, table.schema, compression="snappy") as writer:
            offset = 0
            for row_count in row_group_sizes:
                writer.write_table(table.slice(offset, row_count), row_group_size=row_count)
                offset += row_count
            if offset != table.num_rows:
                raise ValueError(
                    f"Stored row-group sizes do not match the row count for {target}."
                )
        return temp_path
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def write_json_temp(target: Path, payload: dict[str, Any]) -> Path:
    temp_path = create_temp_path(target)
    try:
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
        )
        return temp_path
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def feature_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("Cannot compute feature statistics for an empty or non-vector array.")
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(len(values))],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def stage_dataset_update(
    frames: DatasetFrames,
    columns: dict[str, np.ndarray],
    *,
    index_key: str,
    stats_path: Path,
) -> list[tuple[Path, Path]]:
    staged: list[tuple[Path, Path]] = []
    try:
        for file_slice in frames.files:
            parquet_file = pq.ParquetFile(file_slice.path)
            table = parquet_file.read()
            current_indexes = require_integer_column(
                scalar_column_to_numpy(table, index_key), index_key
            )
            if not np.array_equal(current_indexes, file_slice.indexes):
                raise RuntimeError(f"Dataset changed while processing: {file_slice.path}")

            updated_table = table
            for key, all_values in columns.items():
                column_index = updated_table.column_names.index(key)
                field = updated_table.schema.field(key)
                values = all_values[file_slice.start : file_slice.stop]
                array = vector_arrow_array(values, field)
                updated_table = updated_table.set_column(column_index, field, array)
            temp_path = write_parquet_temp(
                updated_table,
                target=file_slice.path,
                row_group_sizes=file_slice.row_group_sizes,
            )
            staged.append((temp_path, file_slice.path))

        stats = load_json(stats_path)
        for key, values in columns.items():
            stats[key] = feature_stats(values)
        staged.append((write_json_temp(stats_path, stats), stats_path))
        return staged
    except BaseException:
        for temp_path, _ in staged:
            temp_path.unlink(missing_ok=True)
        raise


def commit_staged_files(staged: list[tuple[Path, Path]]) -> None:
    try:
        for temp_path, target in staged:
            os.replace(temp_path, target)
    finally:
        for temp_path, _ in staged:
            temp_path.unlink(missing_ok=True)


def magnitude_summary(values: np.ndarray) -> tuple[float, float, float]:
    absolute = np.abs(np.asarray(values, dtype=np.float64)).reshape(-1)
    return (
        float(np.quantile(absolute, 0.99)),
        float(np.quantile(absolute, 0.999)),
        float(absolute.max()),
    )


def main() -> None:
    args = parse_args()
    root = args.input_root.expanduser().resolve()
    filter_keys = list(dict.fromkeys(args.filter_keys))
    if not math.isfinite(args.cutoff_hz) or args.cutoff_hz <= 0:
        raise ValueError("--cutoff-hz must be a positive finite number.")
    if not math.isfinite(args.min_dt) or args.min_dt <= 0:
        raise ValueError("--min-dt must be a positive finite number.")

    info, stats_path = validate_v3_metadata(
        root,
        velocity_key=args.velocity_key,
        acceleration_key=args.acceleration_key,
        filter_keys=filter_keys,
    )
    frames = load_dataset_frames(
        root,
        velocity_key=args.velocity_key,
        acceleration_key=args.acceleration_key,
        timestamp_key=args.timestamp_key,
        episode_key=args.episode_key,
        frame_key=args.frame_key,
        index_key=args.index_key,
        filter_keys=filter_keys,
    )
    expected_frames = info.get("total_frames")
    if expected_frames is not None and int(expected_frames) != len(frames.indexes):
        raise ValueError(
            f"meta/info.json declares {expected_frames} frames, but data/ contains "
            f"{len(frames.indexes)}."
        )

    computed, clamped_dt_count, dts = compute_outputs_by_episode(
        frames,
        cutoff_hz=args.cutoff_hz,
        min_dt=args.min_dt,
        velocity_key=args.velocity_key,
        acceleration_key=args.acceleration_key,
        filter_keys=filter_keys,
    )
    stored_columns = {key: values.astype(np.float32) for key, values in computed.items()}
    stored_accelerations = stored_columns[args.acceleration_key]
    old_p99, old_p999, old_max = magnitude_summary(frames.old_accelerations)
    new_p99, new_p999, new_max = magnitude_summary(stored_accelerations)
    episode_count = len(np.unique(frames.episode_indexes))

    print(
        f"frames={len(frames.indexes)}, episodes={episode_count}, "
        f"parquet_files={len(frames.files)}, cutoff_hz={args.cutoff_hz:g}"
    )
    if len(dts):
        print(
            f"median_dt={np.median(dts):.6g} s "
            f"(median_rate={1.0 / np.median(dts):.3f} Hz), "
            f"clamped_dt={clamped_dt_count}"
        )
    print("absolute acceleration over all frames and joints:")
    print(f"  before: p99={old_p99:.6g}, p99.9={old_p999:.6g}, max={old_max:.6g}")
    print(f"  after:  p99={new_p99:.6g}, p99.9={new_p999:.6g}, max={new_max:.6g}")
    if filter_keys:
        print(f"filtered in place: {', '.join(filter_keys)}")

    if args.dry_run:
        print("dry run: no files changed")
        return

    staged = stage_dataset_update(
        frames,
        stored_columns,
        index_key=args.index_key,
        stats_path=stats_path,
    )
    commit_staged_files(staged)
    print(f"updated in place: {root}")


if __name__ == "__main__":
    main()
