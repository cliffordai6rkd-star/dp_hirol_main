"""Direct, episode-aware H5 loading without resampling or interpolation.

The training datasets use this module when ``dataloader.backend=h5``.  Every
configured field is read at the row selected by the H5 master timestamp.  The
loader deliberately has no alignment fallback: fields with a different length
must be fixed at collection time or handled by a task-specific dataset.
"""

from __future__ import annotations

import logging
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch


log = logging.getLogger(__name__)

_EPISODE_INDEX_PATTERN = re.compile(r"(?:^|_)episode_(\d+)(?:_|\.|$)")
_TIMESTAMP_SCALES_TO_SECONDS = {
    "s": 1.0,
    "ms": 1.0e-3,
    "us": 1.0e-6,
    "ns": 1.0e-9,
}


def load_h5py():
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Direct H5 dataset loading requires h5py. Activate the PINN "
            "training environment or install h5py there."
        ) from exc
    return h5py


def discover_h5_files(
    root: Path | str,
    patterns: Sequence[str] = ("*.h5", "*.hdf5"),
    max_episodes: int | None = None,
) -> list[Path]:
    """Return a stable, de-duplicated list of episode files."""

    root = Path(root).expanduser()
    if root.is_file():
        files = [root]
    else:
        files = sorted({path for pattern in patterns for path in root.glob(pattern)})
    if not files:
        raise FileNotFoundError(
            f"No H5 episode files found under {root} with patterns={list(patterns)}"
        )
    if max_episodes is not None:
        max_episodes = int(max_episodes)
        if max_episodes <= 0:
            raise ValueError("dataloader.max_episodes must be positive or null")
        files = files[:max_episodes]
    return files


def episode_index_from_path(path: Path, fallback: int | None = None) -> int:
    """Use the raw episode number in the filename, not its sorted position."""

    match = _EPISODE_INDEX_PATTERN.search(path.name)
    if match is not None:
        return int(match.group(1))
    if fallback is None:
        raise ValueError(
            f"Cannot infer episode index from {path.name!r}; expected "
            "a name containing episode_<integer>."
        )
    return int(fallback)


def timestamp_scale_to_seconds(unit: str) -> float:
    unit = str(unit).lower()
    try:
        return _TIMESTAMP_SCALES_TO_SECONDS[unit]
    except KeyError as exc:
        raise ValueError(
            "timestamp unit must be one of s, ms, us, ns, "
            f"got {unit!r}"
        ) from exc


def read_h5_array(h5_file: Any, path: str):
    if path not in h5_file:
        raise KeyError(f"H5 file {h5_file.filename} is missing dataset {path!r}")
    value = h5_file[path]
    if not hasattr(value, "shape"):
        raise TypeError(f"H5 path {path!r} in {h5_file.filename} is not a dataset")
    return value[...]


def _read_field(h5_file: Any, spec: str | Mapping[str, Any]):
    if isinstance(spec, str):
        return read_h5_array(h5_file, spec)
    if not isinstance(spec, Mapping):
        raise TypeError(f"H5 field spec must be a path or mapping, got {spec!r}")

    if "path" in spec:
        unknown = set(spec) - {"path", "dtype", "transform"}
        if unknown:
            raise ValueError(f"H5 path field has unknown options: {sorted(unknown)}")
        value = read_h5_array(h5_file, str(spec["path"]))
        transform = spec.get("transform")
        if transform is None:
            return value
        if transform != "ee_pose_matrix_to_quaternion":
            raise ValueError(f"unsupported direct H5 transform {transform!r}")
        from data_process.tool.ee_pose_matrix_to_quaternion import pose_to_quat7

        if value.shape[-2:] != (4, 4):
            raise ValueError(
                "ee_pose_matrix_to_quaternion expects (...,4,4), "
                f"got {value.shape}"
            )
        import numpy as np

        flat = value.reshape(-1, 4, 4)
        converted = np.stack([pose_to_quat7(pose) for pose in flat], axis=0)
        return converted.reshape(value.shape[:-2] + (7,))

    operation = str(spec.get("operation", "")).lower()
    paths = spec.get("paths")
    if operation != "subtract" or not isinstance(paths, (list, tuple)) or len(paths) != 2:
        raise ValueError(
            "Derived H5 fields currently require "
            "{operation: subtract, paths: [minuend, subtrahend]}"
        )
    unknown = set(spec) - {"operation", "paths", "dtype"}
    if unknown:
        raise ValueError(f"H5 subtract field has unknown options: {sorted(unknown)}")
    return read_h5_array(h5_file, str(paths[0])) - read_h5_array(
        h5_file, str(paths[1])
    )


class TensorColumnDataset(torch.utils.data.Dataset):
    """Small HuggingFace-like tensor table used by the existing trainers."""

    def __init__(self, columns: Mapping[str, torch.Tensor], selected=None):
        self._columns = dict(columns)
        self._selected = tuple(selected) if selected is not None else None
        lengths = {int(value.shape[0]) for value in self._columns.values()}
        if len(lengths) != 1:
            raise ValueError(f"tensor columns have inconsistent lengths: {lengths}")
        self._length = lengths.pop() if lengths else 0

    @property
    def column_names(self) -> list[str]:
        return list(self._columns)

    @property
    def columns(self) -> Mapping[str, torch.Tensor]:
        """Expose the materialized columns to cache writers."""

        return self._columns

    def with_format(self, _format=None, columns=None, output_all_columns=False):
        del _format, output_all_columns
        if columns is None:
            selected = None
        else:
            selected = tuple(columns)
            missing = sorted(set(selected) - set(self._columns))
            if missing:
                raise KeyError(f"tensor table is missing columns: {missing}")
        return TensorColumnDataset(self._columns, selected=selected)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        names = self._selected or tuple(self._columns)
        return {name: self._columns[name][index] for name in names}


class DirectH5EpisodeDataset(torch.utils.data.Dataset):
    """Concatenate aligned H5 episodes while retaining exact boundaries."""

    def __init__(
        self,
        *,
        root: Path | str,
        fields: Mapping[str, str | Mapping[str, Any]],
        timestamp_path: str,
        timestamp_output_key: str = "timestamp",
        timestamp_unit: str = "us",
        timestamp_output_unit: str = "s",
        patterns: Sequence[str] = ("*.h5", "*.hdf5"),
        max_episodes: int | None = None,
        expected_fps: float | None = None,
        max_cadence_error_s: float = 1.0e-6,
    ):
        if not fields:
            raise ValueError("dataloader.h5_fields must not be empty")
        if timestamp_output_key in fields:
            raise ValueError(
                f"timestamp output key {timestamp_output_key!r} conflicts with h5_fields"
            )
        self.root = Path(root)
        self.files = discover_h5_files(self.root, patterns, max_episodes)
        self.fields = dict(fields)
        self.timestamp_path = str(timestamp_path)
        self.timestamp_output_key = str(timestamp_output_key)
        self.timestamp_unit = str(timestamp_unit).lower()
        self.timestamp_output_unit = str(timestamp_output_unit).lower()
        self.timestamp_scale_s = timestamp_scale_to_seconds(self.timestamp_unit)
        self.timestamp_output_scale_s = timestamp_scale_to_seconds(
            self.timestamp_output_unit
        )
        self.expected_fps = (
            None if expected_fps is None else float(expected_fps)
        )
        self.max_cadence_error_s = float(max_cadence_error_s)
        if self.expected_fps is not None and self.expected_fps <= 0.0:
            raise ValueError("dataloader.expected_fps must be positive or null")
        if self.max_cadence_error_s < 0.0:
            raise ValueError("dataloader.max_cadence_error_s must be non-negative")

        columns, episodes = self._load()
        self.hf_dataset = TensorColumnDataset(columns)
        self.meta = SimpleNamespace(episodes=episodes)

    def _load(self):
        h5py = load_h5py()
        buffers = {name: [] for name in self.fields}
        timestamp_buffers = []
        episodes = []
        offset = 0
        seen_episode_indices = set()

        for position, path in enumerate(self.files):
            episode_index = episode_index_from_path(path, fallback=position)
            if episode_index in seen_episode_indices:
                raise ValueError(f"duplicate H5 episode index {episode_index}")
            seen_episode_indices.add(episode_index)

            with h5py.File(path, "r") as h5_file:
                raw_timestamps = read_h5_array(h5_file, self.timestamp_path)
                timestamps = torch.as_tensor(raw_timestamps).reshape(-1)
                if timestamps.numel() == 0:
                    raise ValueError(f"H5 episode {path} has no timestamps")
                timestamps_s = timestamps.to(dtype=torch.float64) * self.timestamp_scale_s
                deltas_s = torch.diff(timestamps_s)
                if deltas_s.numel() and torch.any(deltas_s <= 0.0):
                    raise ValueError(
                        f"H5 timestamps must be strictly increasing in {path}"
                    )
                if self.expected_fps is not None and deltas_s.numel():
                    expected_dt_s = 1.0 / self.expected_fps
                    max_error_s = float((deltas_s - expected_dt_s).abs().max())
                    if max_error_s > self.max_cadence_error_s:
                        raise ValueError(
                            f"H5 episode {path.name} is not uniformly sampled at "
                            f"{self.expected_fps:g} Hz: max cadence error is "
                            f"{max_error_s:.9f} s, allowed "
                            f"{self.max_cadence_error_s:.9f} s. Direct H5 mode "
                            "does not resample."
                        )

                frame_count = int(timestamps.numel())
                for output_key, spec in self.fields.items():
                    array = _read_field(h5_file, spec)
                    if getattr(array, "ndim", 0) == 0 or int(array.shape[0]) != frame_count:
                        raise ValueError(
                            f"H5 field {output_key!r} in {path.name} has shape "
                            f"{getattr(array, 'shape', None)}, expected first dimension "
                            f"{frame_count} from {self.timestamp_path!r}. Direct H5 "
                            "mode will not align or resample mismatched fields."
                        )
                    dtype = spec.get("dtype") if isinstance(spec, Mapping) else None
                    tensor = torch.as_tensor(array)
                    if dtype is not None:
                        torch_dtype = getattr(torch, str(dtype), None)
                        if torch_dtype is None:
                            raise ValueError(f"unknown torch dtype {dtype!r}")
                        tensor = tensor.to(dtype=torch_dtype)
                    elif tensor.is_floating_point():
                        tensor = tensor.to(dtype=torch.float32)
                    buffers[output_key].append(tensor.contiguous())

            output_ratio = self.timestamp_scale_s / self.timestamp_output_scale_s
            rounded_ratio = round(output_ratio)
            if (
                not timestamps.is_floating_point()
                and abs(output_ratio - rounded_ratio) < 1.0e-12
            ):
                output_timestamps = timestamps.to(dtype=torch.int64) * int(
                    rounded_ratio
                )
            else:
                output_timestamps = (
                    timestamps.to(dtype=torch.float64) * output_ratio
                )
            timestamp_buffers.append(output_timestamps.contiguous())
            episodes.append(
                {
                    "episode_index": episode_index,
                    "dataset_from_index": offset,
                    "dataset_to_index": offset + frame_count,
                    "length": frame_count,
                    "h5_path": str(path),
                }
            )
            offset += frame_count

        columns = {
            name: torch.cat(values, dim=0).contiguous()
            for name, values in buffers.items()
        }
        columns[self.timestamp_output_key] = torch.cat(
            timestamp_buffers, dim=0
        ).contiguous()
        log.info(
            "direct H5 dataset loaded: root=%s episodes=%d frames=%d fields=%s",
            self.root,
            len(episodes),
            offset,
            sorted(columns),
        )
        return columns, episodes

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        return self.hf_dataset[index]


class ZarrTensorColumn:
    """Lazy first-axis reader for a cached v3 numeric column.

    The Zarr handle is opened lazily and once per worker process.  This avoids
    inheriting an open HDF5/Zarr handle through ``fork`` while keeping sample
    indexing compatible with the existing tensor-table dataset.
    """

    def __init__(self, store_path: Path | str, array_name: str):
        self.store_path = str(store_path)
        self.array_name = str(array_name)
        self._group = None
        self._array = None
        self._pid = None

    def _ensure_open(self):
        import os

        pid = os.getpid()
        if self._array is not None and self._pid == pid:
            return self._array
        try:
            import zarr
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ssd_zarr cache mode requires zarr; install the optional "
                "zarr dependency or use train_data.cache.mode=ram."
            ) from exc
        self._group = zarr.open_group(self.store_path, mode="r")
        self._array = self._group[self.array_name]
        self._pid = pid
        return self._array

    @property
    def shape(self):
        return tuple(self._ensure_open().shape)

    @property
    def dtype(self):
        return self._ensure_open().dtype

    def index_select(self, dim: int, index: torch.Tensor) -> torch.Tensor:
        if dim != 0:
            raise ValueError("ZarrTensorColumn only supports first-axis selection")
        if not torch.is_tensor(index):
            index = torch.as_tensor(index)
        if index.ndim != 1:
            raise ValueError("index_select index must be one-dimensional")
        indices = index.detach().cpu().to(dtype=torch.long).numpy()
        array = self._ensure_open()
        try:
            values = array.oindex[indices]
        except AttributeError:
            values = array[indices]
        return torch.as_tensor(values).clone()

    def __getitem__(self, index):
        values = self._ensure_open()[index]
        return torch.as_tensor(values).clone()


class ConcatTensorColumn:
    """Concatenate RAM or Zarr columns without materializing Zarr data."""

    def __init__(self, columns: Sequence[Any]):
        self.columns = tuple(columns)
        if not self.columns:
            raise ValueError("ConcatTensorColumn requires at least one column")
        trailing = tuple(self.columns[0].shape[1:])
        if any(tuple(column.shape[1:]) != trailing for column in self.columns):
            raise ValueError("concatenated columns have incompatible shapes")
        self._lengths = tuple(int(column.shape[0]) for column in self.columns)
        self._offsets = torch.as_tensor(
            [0, *torch.tensor(self._lengths, dtype=torch.long).cumsum(0).tolist()],
            dtype=torch.long,
        )

    @property
    def shape(self):
        return (sum(self._lengths), *self.columns[0].shape[1:])

    @property
    def dtype(self):
        return getattr(self.columns[0], "dtype", torch.float32)

    def to_tensor(self) -> torch.Tensor:
        """Materialize the full column when a global pass is unavoidable."""

        return self.index_select(0, torch.arange(self.shape[0], dtype=torch.long))

    def index_select(self, dim: int, index: torch.Tensor) -> torch.Tensor:
        if dim != 0:
            raise ValueError("ConcatTensorColumn only supports first-axis selection")
        index = torch.as_tensor(index, dtype=torch.long)
        if index.ndim != 1:
            raise ValueError("index_select index must be one-dimensional")
        if index.numel() == 0:
            return torch.empty((0, *self.shape[1:]), dtype=torch.float32)
        if torch.any(index < 0) or torch.any(index >= self.shape[0]):
            raise IndexError("column index is out of range")
        result = None
        for position, column in enumerate(self.columns):
            start = int(self._offsets[position])
            end = int(self._offsets[position + 1])
            mask = (index >= start) & (index < end)
            if not torch.any(mask):
                continue
            values = column.index_select(0, index[mask] - start)
            if result is None:
                result = torch.empty(
                    (index.numel(), *values.shape[1:]),
                    dtype=values.dtype,
                )
            result[mask] = values
        if result is None:
            raise RuntimeError("failed to select concatenated column values")
        return result

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.shape[0])
            return self.index_select(
                0, torch.arange(start, stop, step, dtype=torch.long)
            )
        return self.index_select(0, torch.as_tensor([index], dtype=torch.long))[0]


def _cache_signature(files, *, fields, timestamp_path, timestamp_unit, anchor_path, anchor_unit, expected_fps):
    payload = {
        "files": [
            {
                "path": str(path),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in files
        ],
        "fields": fields,
        "timestamp_path": timestamp_path,
        "timestamp_unit": timestamp_unit,
        "anchor_path": anchor_path,
        "anchor_unit": anchor_unit,
        "expected_fps": expected_fps,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


class V3H5Collection:
    """Load multiple raw H5 folders through one v3 numeric interface.

    Each source is decoded with the same v3 field contract.  ``ram`` keeps
    tensor columns resident, while ``ssd_zarr`` writes one cache per source and
    reads numeric columns lazily during training.
    """

    def __init__(
        self,
        *,
        sources: Sequence[Mapping[str, Any] | str | Path],
        source_base_root: Path | str | None = None,
        fields: Mapping[str, str | Mapping[str, Any]],
        timestamp_path: str,
        timestamp_unit: str,
        anchor_timestamp_path: str,
        anchor_timestamp_unit: str,
        patterns: Sequence[str] = ("*.h5", "*.hdf5"),
        expected_fps: float | None = None,
        max_cadence_error_s: float = 1.0e-6,
        cache_mode: str = "ram",
        cache_root: Path | str | None = None,
        cache_rebuild: bool = False,
        cache_chunk_rows: int = 4096,
    ):
        if not sources:
            raise ValueError("train_data.sources must not be empty")
        self.cache_mode = str(cache_mode).lower()
        self.source_base_root = (
            None if source_base_root is None else Path(source_base_root).expanduser()
        )
        if self.cache_mode not in {"ram", "ssd_zarr"}:
            raise ValueError("train_data.cache.mode must be 'ram' or 'ssd_zarr'")
        if self.cache_mode == "ssd_zarr" and cache_root is None:
            raise ValueError("train_data.cache.root is required for ssd_zarr mode")
        if int(cache_chunk_rows) <= 0:
            raise ValueError("train_data.cache.chunk_rows must be positive")

        self.files = []
        source_columns = []
        source_timestamps = []
        source_anchors = []
        episodes = []
        raw_offset = 0
        anchor_offset = 0
        for source_position, source in enumerate(sources):
            if isinstance(source, (str, Path)):
                source_path = Path(source)
                if self.source_base_root is not None and not source_path.is_absolute():
                    source_path = self.source_base_root / source_path
                source = {"name": source_path.name, "root": str(source_path)}
            if not isinstance(source, Mapping):
                raise TypeError(
                    "each train_data.sources item must be a folder string or mapping"
                )
            source_name = str(source.get("name", f"source_{source_position}"))
            source_root = source.get("root") or source.get("path")
            if source_root is None:
                raise ValueError(f"train_data source {source_name!r} needs root")
            source_patterns = tuple(source.get("patterns", patterns))
            source_fields = source.get("h5_fields", source.get("fields", fields))
            source_timestamp_path = str(source.get("timestamp_path", timestamp_path))
            source_timestamp_unit = str(source.get("timestamp_unit", timestamp_unit))
            source_anchor_path = str(source.get("anchor_timestamp_path", anchor_timestamp_path))
            source_anchor_unit = str(source.get("anchor_timestamp_unit", anchor_timestamp_unit))
            source_expected_fps = source.get("expected_fps", expected_fps)
            source_max_cadence = float(source.get("max_cadence_error_s", max_cadence_error_s))
            source_files = discover_h5_files(source_root, source_patterns, source.get("max_episodes"))
            cache_path = None
            if self.cache_mode == "ssd_zarr":
                signature = _cache_signature(
                    source_files,
                    fields=source_fields,
                    timestamp_path=source_timestamp_path,
                    timestamp_unit=source_timestamp_unit,
                    anchor_path=source_anchor_path,
                    anchor_unit=source_anchor_unit,
                    expected_fps=source_expected_fps,
                )
                cache_path = Path(cache_root).expanduser() / source_name / f"{signature}.zarr"
            direct = self._load_source(
                source_fields,
                source_root,
                source_patterns,
                source_timestamp_path,
                source_timestamp_unit,
                source_anchor_path,
                source_anchor_unit,
                source_expected_fps,
                source_max_cadence,
                cache_path,
                cache_rebuild,
                int(cache_chunk_rows),
            )
            columns, high_timestamps, anchors, source_episodes, source_paths = direct
            source_columns.append(columns)
            source_timestamps.append(high_timestamps)
            source_anchors.append(anchors)
            self.files.extend(source_paths)
            for episode in source_episodes:
                updated = dict(episode)
                updated["episode_index"] = len(episodes)
                updated["source_index"] = source_position
                updated["source_name"] = source_name
                updated["source_episode_index"] = int(episode.get("episode_index", len(episodes)))
                updated["dataset_from_index"] = int(updated["dataset_from_index"]) + raw_offset
                updated["dataset_to_index"] = int(updated["dataset_to_index"]) + raw_offset
                updated["anchor_from_index"] = int(updated.get("anchor_from_index", 0)) + anchor_offset
                updated["anchor_to_index"] = int(updated.get("anchor_to_index", 0)) + anchor_offset
                episodes.append(updated)
            raw_offset += int(high_timestamps.shape[0])
            anchor_offset += int(anchors.shape[0])

        names = tuple(source_columns[0])
        if any(tuple(columns) != names for columns in source_columns[1:]):
            raise ValueError("all train_data sources must expose the same v3 fields")
        if self.cache_mode == "ram":
            self.columns = {
                name: torch.cat([columns[name] for columns in source_columns], dim=0).contiguous()
                for name in names
            }
        else:
            self.columns = {
                name: ConcatTensorColumn([columns[name] for columns in source_columns])
                for name in names
            }
        self.high_timestamps = torch.cat(source_timestamps, dim=0).contiguous()
        self.anchor_timestamps = torch.cat(source_anchors, dim=0).contiguous()
        self.episodes = episodes
        self.meta = SimpleNamespace(episodes=episodes)

    @staticmethod
    def _load_source(
        fields, root, patterns, timestamp_path, timestamp_unit, anchor_path,
        anchor_unit, expected_fps, max_cadence_error_s, cache_path, cache_rebuild,
        cache_chunk_rows,
    ):
        if cache_path is not None and cache_path.exists() and not cache_rebuild:
            return V3H5Collection._read_zarr_cache(cache_path)
        direct = DirectH5EpisodeDataset(
            root=root,
            fields=fields,
            timestamp_path=timestamp_path,
            timestamp_output_key="__h5_timestamp_ns",
            timestamp_unit=timestamp_unit,
            timestamp_output_unit="ns",
            patterns=patterns,
            expected_fps=expected_fps,
            max_cadence_error_s=max_cadence_error_s,
        )
        columns = dict(direct.hf_dataset.columns)
        high_timestamps = columns.pop("__h5_timestamp_ns").to(dtype=torch.int64)
        anchors = V3H5Collection._read_anchor_timestamps(
            direct.files,
            direct.meta.episodes,
            anchor_path,
            anchor_unit,
            high_timestamps,
        )
        if cache_path is not None:
            V3H5Collection._write_zarr_cache(
                cache_path, columns, high_timestamps, anchors,
                direct.meta.episodes, direct.files, cache_chunk_rows,
            )
            return V3H5Collection._read_zarr_cache(cache_path)
        return columns, high_timestamps, anchors, list(direct.meta.episodes), list(direct.files)

    @staticmethod
    def _read_anchor_timestamps(files, episodes, path, unit, high_timestamps):
        h5py = load_h5py()
        scale = timestamp_scale_to_seconds(unit) / 1.0e-9
        rounded = round(scale)
        values = []
        anchor_offset = 0
        for episode, file_path in zip(episodes, files):
            with h5py.File(file_path, "r") as h5_file:
                raw = torch.as_tensor(read_h5_array(h5_file, path)).reshape(-1)
            if raw.numel() == 0:
                raise ValueError(f"H5 episode {file_path.name} has no action anchors")
            if not raw.is_floating_point() and abs(scale - rounded) < 1.0e-12:
                anchor = raw.to(dtype=torch.int64) * int(rounded)
            else:
                anchor = (raw.to(dtype=torch.float64) * scale).round().to(dtype=torch.int64)
            if anchor.numel() > 1 and torch.any(torch.diff(anchor) <= 0):
                raise ValueError(f"H5 action anchors must be strictly increasing in {file_path.name}")
            start = int(episode["dataset_from_index"])
            end = int(episode["dataset_to_index"])
            state_times = high_timestamps[start:end]
            if int(anchor[-1]) < int(state_times[0]) or int(anchor[0]) > int(state_times[-1]):
                raise ValueError(f"H5 action-anchor clock does not overlap state clock in {file_path.name}")
            episode["anchor_from_index"] = anchor_offset
            episode["anchor_to_index"] = anchor_offset + int(anchor.numel())
            values.append(anchor.contiguous())
            anchor_offset += int(anchor.numel())
        return torch.cat(values, dim=0)

    @staticmethod
    def _write_zarr_cache(path, columns, timestamps, anchors, episodes, files, chunk_rows):
        try:
            import zarr
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ssd_zarr cache mode requires zarr; use cache.mode=ram otherwise"
            ) from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        group = zarr.open_group(str(path), mode="w", zarr_format=2)
        for name, value in {**columns, "__h5_timestamp_ns": timestamps, "__anchor_timestamp_ns": anchors}.items():
            array = value.detach().cpu().numpy() if torch.is_tensor(value) else value
            chunks = (min(int(chunk_rows), int(array.shape[0])), *array.shape[1:])
            group.create_dataset(name, data=array, chunks=chunks, overwrite=True)
        (path / "manifest.json").write_text(
            json.dumps({"episodes": episodes, "files": [str(file) for file in files]}, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _read_zarr_cache(path):
        try:
            import zarr
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ssd_zarr cache mode requires zarr; use cache.mode=ram otherwise"
            ) from exc
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Zarr cache is missing manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        group = zarr.open_group(str(path), mode="r")
        names = [name for name in group.array_keys() if name not in {"__h5_timestamp_ns", "__anchor_timestamp_ns"}]
        columns = {name: ZarrTensorColumn(path, name) for name in names}
        timestamps = torch.as_tensor(group["__h5_timestamp_ns"][:]).to(dtype=torch.int64)
        anchors = torch.as_tensor(group["__anchor_timestamp_ns"][:]).to(dtype=torch.int64)
        files = [Path(value) for value in manifest["files"]]
        return columns, timestamps, anchors, list(manifest["episodes"]), files
