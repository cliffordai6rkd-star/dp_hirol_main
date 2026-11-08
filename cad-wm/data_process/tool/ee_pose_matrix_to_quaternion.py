from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any


DEFAULT_FIELDS = (
    "observation.ee_state.ee_pose",
    "action.ee_pose",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace LeRobot 4x4 ee_pose fields with xyz + quaternion."
    )
    parser.add_argument("--datapath", type=Path, required=True, help="LeRobot dataset folder.")
    return parser.parse_args()


def load_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: numpy is required to convert poses.") from exc
    return np


def load_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: pyarrow is required for in-place replacement.") from exc
    return pa, pq


def to_numpy(value: Any) -> np.ndarray:
    np = load_numpy()
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def rotation_matrix_to_quat_xyzw(matrix: Any) -> np.ndarray:
    """Convert a 3x3 rotation matrix to quaternion in [qx, qy, qz, qw] order."""

    np = load_numpy()
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"rotation matrix must have shape (3, 3), got {m.shape}")

    trace = np.trace(m)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm > 0:
        quat = quat / norm

    # q and -q represent the same orientation. Keep qw non-negative to reduce
    # discontinuous sign flips in learning code.
    if quat[3] < 0:
        quat = -quat
    return quat.astype(np.float32)


def matrix_to_xyz_quat_xyzw(pose: Any) -> np.ndarray:
    """Convert a 4x4 homogeneous transform to [x, y, z, qx, qy, qz, qw]."""

    np = load_numpy()
    pose = to_numpy(pose).astype(np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"pose must have shape (4, 4), got {pose.shape}")

    xyz = pose[:3, 3].astype(np.float32)
    quat = rotation_matrix_to_quat_xyzw(pose[:3, :3])
    return np.concatenate([xyz, quat], axis=0).astype(np.float32)


def pose_to_quat7(value: Any) -> np.ndarray:
    """Accept either a 4x4 matrix pose or an already flattened quat7 pose."""

    np = load_numpy()
    array = to_numpy(value)
    if array.shape == (4, 4):
        return matrix_to_xyz_quat_xyzw(array)

    flat = array.reshape(-1).astype(np.float32)
    if flat.shape == (7,):
        quat = flat[3:7].astype(np.float64)
        norm = np.linalg.norm(quat)
        if norm > 0:
            flat[3:7] = (quat / norm).astype(np.float32)
        if flat[6] < 0:
            flat[3:7] = -flat[3:7]
        return flat

    raise ValueError(f"pose must be 4x4 matrix or quat7, got shape {array.shape}")


def backup_file(path: Path) -> None:
    backup_path = path.with_suffix(path.suffix + ".bak")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)


def data_parquet_files(root: Path) -> list[Path]:
    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data parquet files found under {root / 'data'}")
    return files


def table_column_to_quat7_array(column) -> list[list[float]]:
    return [pose_to_quat7(value.as_py()).tolist() for value in column]


def replace_table_fields(table, fields: list[str]):
    pa, _ = load_pyarrow()
    updated = table

    for field in fields:
        if field not in updated.column_names:
            raise KeyError(f"{field!r} is missing from parquet columns: {updated.column_names}")

        quat7_values = table_column_to_quat7_array(updated[field])
        quat7_array = pa.array(quat7_values, type=pa.list_(pa.float32(), list_size=7))
        field_index = updated.schema.get_field_index(field)
        updated = updated.set_column(field_index, field, quat7_array)

    return update_huggingface_schema_metadata(updated, fields)


def update_huggingface_schema_metadata(table, fields: list[str]):
    metadata = dict(table.schema.metadata or {})
    raw = metadata.get(b"huggingface")
    if raw is None:
        return table

    try:
        payload = json.loads(raw.decode("utf-8"))
        features = payload["info"]["features"]
    except (KeyError, TypeError, ValueError):
        return table

    for field in fields:
        if field in features:
            features[field] = {
                "feature": {
                    "dtype": "float32",
                    "_type": "Sequence",
                },
                "length": 7,
                "_type": "Sequence",
            }

    metadata[b"huggingface"] = json.dumps(payload).encode("utf-8")
    return table.replace_schema_metadata(metadata)


def replace_parquet_files(root: Path, fields: list[str], backup: bool) -> dict[str, list[float]]:
    np = load_numpy()
    _, pq = load_pyarrow()
    collected = {field: [] for field in fields}

    for parquet_path in data_parquet_files(root):
        table = pq.read_table(parquet_path)
        updated = replace_table_fields(table, fields)

        for field in fields:
            values = np.asarray(updated[field].to_pylist(), dtype=np.float32)
            collected[field].append(values)

        if backup:
            backup_file(parquet_path)
        pq.write_table(updated, parquet_path)

    stats = {}
    for field, chunks in collected.items():
        values = np.concatenate(chunks, axis=0)
        stats[field] = compute_stats(values)
    return stats


def compute_stats(values):
    np = load_numpy()
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])] * int(values.shape[1]),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def replace_meta_files(root: Path, fields: list[str], stats: dict[str, Any], backup: bool) -> None:
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"

    if backup:
        backup_file(info_path)
        backup_file(stats_path)

    info = json.loads(info_path.read_text())
    for field in fields:
        if field not in info["features"]:
            raise KeyError(f"{field!r} is missing from {info_path}")
        info["features"][field]["dtype"] = "float32"
        info["features"][field]["shape"] = [7]
    info_path.write_text(json.dumps(info, indent=4), encoding="utf-8")

    stats_data = json.loads(stats_path.read_text())
    for field, field_stats in stats.items():
        stats_data[field] = field_stats
    stats_path.write_text(json.dumps(stats_data, indent=4), encoding="utf-8")


def replace_lerobot_dataset(root: Path, fields: list[str], backup: bool) -> None:
    if not root.exists():
        raise FileNotFoundError(f"datapath does not exist: {root}")
    if not (root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"missing LeRobot meta/info.json under: {root}")
    if not (root / "meta" / "stats.json").exists():
        raise FileNotFoundError(f"missing LeRobot meta/stats.json under: {root}")

    stats = replace_parquet_files(root, fields, backup=backup)
    replace_meta_files(root, fields, stats, backup=backup)


def main() -> None:
    args = parse_args()
    replace_lerobot_dataset(args.datapath, list(DEFAULT_FIELDS), backup=False)
    print(f"replaced fields in-place under: {args.datapath}")
    for field in DEFAULT_FIELDS:
        print(f"  {field}: shape=[7], order=[x, y, z, qx, qy, qz, qw]")


if __name__ == "__main__":
    main()
