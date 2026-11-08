from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np


DEFAULT_TIMESTAMP_PATH = "teleop/timestamp_us"
DEFAULT_MAX_GAP_S = 0.1
MANIFEST_NAME = "timestamp_gap_split_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split H5 episodes wherever the master timestamp has a long gap."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--timestamp-path",
        default=DEFAULT_TIMESTAMP_PATH,
    )
    parser.add_argument(
        "--max-gap-s",
        type=float,
        default=DEFAULT_MAX_GAP_S,
        help="Split when adjacent timestamps differ by more than this value.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=2,
        help="Reject a split that would create a shorter segment.",
    )
    return parser.parse_args()


def segment_bounds(
    timestamps_us: np.ndarray,
    *,
    max_gap_s: float,
    min_frames: int = 2,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    timestamps = np.asarray(timestamps_us).reshape(-1)
    if timestamps.size < min_frames:
        raise ValueError(
            f"Timestamp series has {timestamps.size} frames; minimum is {min_frames}."
        )
    if not np.isfinite(timestamps).all():
        raise ValueError("Timestamps must be finite.")
    intervals_s = np.diff(timestamps.astype(np.float64)) * 1.0e-6
    if np.any(intervals_s <= 0.0):
        raise ValueError("Timestamps must be strictly increasing.")
    if not np.isfinite(max_gap_s) or max_gap_s <= 0.0:
        raise ValueError("max_gap_s must be positive and finite.")
    if min_frames <= 0:
        raise ValueError("min_frames must be positive.")

    split_indices = np.flatnonzero(intervals_s > max_gap_s) + 1
    edges = np.concatenate(([0], split_indices, [timestamps.size]))
    bounds = [
        (int(start), int(stop))
        for start, stop in zip(edges[:-1], edges[1:])
    ]
    short = [(start, stop) for start, stop in bounds if stop - start < min_frames]
    if short:
        raise ValueError(
            f"Timestamp-gap split would create segments shorter than {min_frames}: "
            f"{short}."
        )
    return bounds, intervals_s[split_indices - 1]


def _copy_attributes(source: Any, destination: Any) -> None:
    for key, value in source.attrs.items():
        destination.attrs[key] = value


def _dataset_creation_options(source: h5py.Dataset, shape: Sequence[int]) -> dict:
    if not shape or source.chunks is None:
        return {}
    chunks = tuple(min(chunk, size) for chunk, size in zip(source.chunks, shape))
    options: dict[str, Any] = {"chunks": chunks}
    if source.compression is not None:
        options["compression"] = source.compression
        options["compression_opts"] = source.compression_opts
    if source.shuffle:
        options["shuffle"] = True
    if source.fletcher32:
        options["fletcher32"] = True
    if source.scaleoffset is not None:
        options["scaleoffset"] = source.scaleoffset
    return options


def write_segment(
    source_path: Path,
    output_path: Path,
    *,
    frame_count: int,
    start: int,
    stop: int,
    part_index: int,
    part_count: int,
    timestamp_path: str,
    max_gap_s: float,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite output file: {output_path}")
    temporary_path = output_path.with_suffix(output_path.suffix + ".splitting")
    if temporary_path.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(source_path, "r") as source_file, h5py.File(
            temporary_path,
            "w",
        ) as output_file:
            _copy_attributes(source_file, output_file)
            output_file.attrs["timestamp_gap_split"] = True
            output_file.attrs["timestamp_gap_split_source"] = source_path.name
            output_file.attrs["timestamp_gap_split_part_index"] = part_index
            output_file.attrs["timestamp_gap_split_part_count"] = part_count
            output_file.attrs["timestamp_gap_split_start_index"] = start
            output_file.attrs["timestamp_gap_split_stop_index"] = stop
            output_file.attrs["timestamp_gap_split_max_gap_s"] = float(max_gap_s)
            output_file.attrs["timestamp_gap_split_timestamp_path"] = timestamp_path

            def copy_node(name: str, node: Any) -> None:
                if isinstance(node, h5py.Group):
                    group = output_file.require_group(name)
                    _copy_attributes(node, group)
                    return
                if not isinstance(node, h5py.Dataset):
                    return

                if node.ndim == 0:
                    data = node[()]
                elif node.shape[0] == frame_count:
                    data = node[start:stop]
                else:
                    raise ValueError(
                        f"Dataset {name!r} in {source_path} has shape {node.shape}; "
                        f"expected a scalar or first dimension {frame_count}."
                    )
                options = _dataset_creation_options(node, np.shape(data))
                dataset = output_file.create_dataset(
                    name,
                    data=data,
                    dtype=node.dtype,
                    **options,
                )
                _copy_attributes(node, dataset)

            source_file.visititems(copy_node)
            output_file.flush()
        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def split_file(
    source_path: Path,
    output_dir: Path,
    *,
    timestamp_path: str,
    max_gap_s: float,
    min_frames: int,
) -> dict[str, Any]:
    with h5py.File(source_path, "r") as source_file:
        if timestamp_path not in source_file:
            raise KeyError(f"Timestamp dataset is missing: {timestamp_path}")
        timestamps = np.asarray(source_file[timestamp_path])
    bounds, gap_sizes_s = segment_bounds(
        timestamps,
        max_gap_s=max_gap_s,
        min_frames=min_frames,
    )

    outputs = []
    if len(bounds) == 1:
        output_path = output_dir / source_path.name
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite output file: {output_path}")
        shutil.copy2(source_path, output_path)
        outputs.append(
            {"path": output_path.name, "start": 0, "stop": int(len(timestamps))}
        )
    else:
        digits = max(2, len(str(len(bounds) - 1)))
        for part_index, (start, stop) in enumerate(bounds):
            output_path = output_dir / (
                f"{source_path.stem}_part{part_index:0{digits}d}{source_path.suffix}"
            )
            write_segment(
                source_path,
                output_path,
                frame_count=len(timestamps),
                start=start,
                stop=stop,
                part_index=part_index,
                part_count=len(bounds),
                timestamp_path=timestamp_path,
                max_gap_s=max_gap_s,
            )
            outputs.append({"path": output_path.name, "start": start, "stop": stop})

    return {
        "source": source_path.name,
        "frame_count": int(len(timestamps)),
        "split": len(bounds) > 1,
        "gap_sizes_s": [float(value) for value in gap_sizes_s],
        "outputs": outputs,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output_dir}")
    files = sorted((*input_dir.glob("*.h5"), *input_dir.glob("*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No H5 files found in {input_dir}")

    output_dir.mkdir(parents=True)
    episodes = []
    for index, source_path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] checking {source_path.name}", flush=True)
        result = split_file(
            source_path,
            output_dir,
            timestamp_path=str(args.timestamp_path),
            max_gap_s=float(args.max_gap_s),
            min_frames=int(args.min_frames),
        )
        episodes.append(result)
        if result["split"]:
            gaps_ms = [round(value * 1000.0, 3) for value in result["gap_sizes_s"]]
            print(
                f"  split into {len(result['outputs'])} parts at gaps {gaps_ms} ms",
                flush=True,
            )

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "timestamp_path": str(args.timestamp_path),
        "max_gap_s": float(args.max_gap_s),
        "min_frames": int(args.min_frames),
        "source_episode_count": len(files),
        "output_episode_count": sum(len(item["outputs"]) for item in episodes),
        "episodes": episodes,
    }
    (output_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
