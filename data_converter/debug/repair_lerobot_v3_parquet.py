from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


VIDEO_HELPER_SUFFIXES = (".__video_file__", ".__frame_index__")


def _feature_scalar_type(dtype: str) -> pa.DataType:
    if dtype == "float32":
        return pa.float32()
    if dtype == "float64":
        return pa.float64()
    if dtype == "int64":
        return pa.int64()
    if dtype == "int32":
        return pa.int32()
    if dtype == "bool":
        return pa.bool_()
    raise ValueError(f"Unsupported scalar dtype for repair: {dtype!r}")


def _is_shape_one(spec: dict[str, Any] | None) -> bool:
    return spec is not None and tuple(spec.get("shape", ())) == (1,)


def _unwrap_singleton_column(column: pa.ChunkedArray, dtype: str) -> pa.Array:
    values = []
    for value in column.to_pylist():
        if value is None:
            values.append(None)
        elif isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"Expected singleton list, got length {len(value)}")
            values.append(value[0])
        else:
            values.append(value)
    return pa.array(values, type=_feature_scalar_type(dtype))


def _repair_table(table: pa.Table, features: dict[str, dict[str, Any]]) -> tuple[pa.Table, list[str]]:
    arrays = []
    names = []
    changes = []

    for name in table.schema.names:
        if name.endswith(VIDEO_HELPER_SUFFIXES):
            changes.append(f"drop {name}")
            continue

        column = table[name]
        spec = features.get(name)
        if _is_shape_one(spec) and (
            pa.types.is_fixed_size_list(column.type) or pa.types.is_list(column.type)
        ):
            column = _unwrap_singleton_column(column, str(spec["dtype"]))
            changes.append(f"unwrap {name}")

        arrays.append(column)
        names.append(name)

    return pa.table(arrays, names=names), changes


def _column_or_none(table: pa.Table, *names: str) -> pa.ChunkedArray | None:
    for name in names:
        if name in table.schema.names:
            return table[name]
    return None


def _append_column_if_missing(
    table: pa.Table,
    name: str,
    column: pa.Array | pa.ChunkedArray,
    changes: list[str],
) -> pa.Table:
    if name in table.schema.names:
        return table
    changes.append(f"add {name}")
    return table.append_column(name, column)


def _scalar_pylist(column: pa.ChunkedArray) -> list[Any]:
    values = []
    for value in column.to_pylist():
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"Expected singleton list, got length {len(value)}")
            values.append(value[0])
        else:
            values.append(value)
    return values


def _set_column(table: pa.Table, name: str, column: pa.Array, changes: list[str]) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        return table
    changes.append(f"replace {name}")
    return table.set_column(index, name, column)


def _repair_timestamps(table: pa.Table, features: dict[str, dict[str, Any]], *, fps: int) -> tuple[pa.Table, list[str]]:
    changes = []
    if "frame_index" not in table.schema.names:
        return table, changes

    frame_seconds = [float(value) / float(fps) for value in _scalar_pylist(table["frame_index"])]
    timestamp_array = pa.array(frame_seconds, type=pa.float32())
    for name, spec in features.items():
        if name == "timestamp" or name.endswith(".timestamp"):
            if name in table.schema.names and _is_shape_one(spec):
                table = _set_column(table, name, timestamp_array, changes)
    return table, changes


def _repair_episode_table(
    table: pa.Table,
    *,
    fps: int,
    video_keys: list[str],
) -> tuple[pa.Table, list[str]]:
    changes = []
    from_index = _column_or_none(table, "dataset_from_index", "from_index", "start_frame_index")
    to_index = _column_or_none(table, "dataset_to_index", "to_index", "end_frame_index")
    data_chunk = _column_or_none(table, "data/chunk_index", "chunk_index")
    data_file = _column_or_none(table, "data/file_index", "file_index")

    if from_index is None or to_index is None:
        return table, changes

    table = _append_column_if_missing(table, "dataset_from_index", from_index, changes)
    table = _append_column_if_missing(table, "dataset_to_index", to_index, changes)
    if data_chunk is not None:
        table = _append_column_if_missing(table, "data/chunk_index", data_chunk, changes)
    if data_file is not None:
        table = _append_column_if_missing(table, "data/file_index", data_file, changes)

    from_values = [float(value) / float(fps) for value in from_index.to_pylist()]
    to_values = [float(value) / float(fps) for value in to_index.to_pylist()]
    video_chunk = data_chunk if data_chunk is not None else pa.array([0] * table.num_rows, type=pa.int64())
    video_file = data_file if data_file is not None else pa.array([0] * table.num_rows, type=pa.int64())

    for video_key in video_keys:
        prefix = f"videos/{video_key}"
        table = _append_column_if_missing(table, f"{prefix}/chunk_index", video_chunk, changes)
        table = _append_column_if_missing(table, f"{prefix}/file_index", video_file, changes)
        table = _append_column_if_missing(
            table,
            f"{prefix}/from_timestamp",
            pa.array(from_values, type=pa.float64()),
            changes,
        )
        table = _append_column_if_missing(
            table,
            f"{prefix}/to_timestamp",
            pa.array(to_values, type=pa.float64()),
            changes,
        )

    return table, changes


def _write_repaired_table(path: Path, table: pa.Table, *, backup: bool) -> None:
    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
            print(f"  backup: {backup_path}")

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp_path, compression="snappy", use_dictionary=True)
    tmp_path.replace(path)
    print("  repaired")


def repair_dataset(dataset_path: Path, *, backup: bool, dry_run: bool) -> None:
    dataset_path = dataset_path.expanduser().resolve()
    info_path = dataset_path / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    features = dict(info.get("features", {}))
    data_parquet_paths = sorted((dataset_path / "data").glob("*/*.parquet"))
    if not data_parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {dataset_path / 'data'}")

    for path in data_parquet_paths:
        table = pq.read_table(path)
        repaired, changes = _repair_table(table, features)
        repaired, timestamp_changes = _repair_timestamps(
            repaired,
            features,
            fps=int(info["fps"]),
        )
        changes.extend(timestamp_changes)
        if not changes:
            print(f"{path}: no changes")
            continue

        print(f"{path}:")
        for change in changes:
            print(f"  - {change}")
        if dry_run:
            continue

        _write_repaired_table(path, repaired, backup=backup)

    episode_parquet_paths = sorted((dataset_path / "meta" / "episodes").glob("*/*.parquet"))
    for path in episode_parquet_paths:
        table = pq.read_table(path)
        repaired, changes = _repair_episode_table(
            table,
            fps=int(info["fps"]),
            video_keys=list(info.get("video_keys", [])),
        )
        if not changes:
            print(f"{path}: no changes")
            continue

        print(f"{path}:")
        for change in changes:
            print(f"  - {change}")
        if dry_run:
            continue

        _write_repaired_table(path, repaired, backup=backup)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repair local LeRobot v3 parquet files produced by older converter code."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak copies.")
    parser.add_argument("--dry-run", action="store_true", help="Print repairs without writing files.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    repair_dataset(
        dataset_path=args.dataset_path,
        backup=not args.no_backup,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
