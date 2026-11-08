from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tqdm import tqdm

from data_process.tool.h5_2_lerobotev3 import (
    H5Dataset,
    LeRobotV3Dataset,
    build_conversion_spec,
    config_bool,
    config_int,
    config_path,
    config_str,
    load_conversion_deps,
    load_shape_meta,
)


INDEX_SAMPLING_MODE = "index_stride"
MANIFEST_NAME = "index_stride_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a 100 Hz H5 stream into even/odd 50 Hz LeRobot v3 datasets."
        )
    )
    parser.add_argument("--config", "-c", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=None, help="Override io.input.")
    parser.add_argument(
        "--output-phase0",
        type=Path,
        default=None,
        help="Override io.output_phase0. The target must not exist.",
    )
    parser.add_argument(
        "--output-phase1",
        type=Path,
        default=None,
        help="Override io.output_phase1. The target must not exist.",
    )
    return parser.parse_args()


def normalize_index_sampling(
    config: Mapping[str, Any],
    *,
    target_fps: int,
) -> dict[str, Any]:
    value = config.get("downsample")
    if not isinstance(value, Mapping):
        raise ValueError("Config must define a 'downsample' mapping.")
    unknown = set(value) - {"mode", "source_fps", "stride", "phases"}
    if unknown:
        raise ValueError(f"downsample has unknown options: {sorted(unknown)}")

    mode = str(value.get("mode", "")).strip().lower()
    source_fps = int(value.get("source_fps", 0))
    stride = int(value.get("stride", 0))
    phases = tuple(int(phase) for phase in value.get("phases", ()))
    if mode != INDEX_SAMPLING_MODE:
        raise ValueError(f"downsample.mode must be {INDEX_SAMPLING_MODE!r}.")
    if source_fps <= 0 or stride <= 0:
        raise ValueError("downsample.source_fps and stride must be positive integers.")
    if source_fps % stride != 0 or source_fps // stride != target_fps:
        raise ValueError(
            f"source_fps={source_fps} with stride={stride} does not produce "
            f"fps={target_fps}."
        )
    if phases != tuple(range(stride)):
        raise ValueError(
            f"downsample.phases must contain every phase exactly once: "
            f"{list(range(stride))}."
        )
    return {
        "mode": mode,
        "source_fps": source_fps,
        "stride": stride,
        "phases": phases,
    }


def conversion_spec(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = build_conversion_spec(config)
    if spec.get("sampling") is not None or spec.get("timeline") is not None:
        raise ValueError(
            "Index-stride conversion must not define sampling or timeline."
        )
    for mapping in spec["mappings"]:
        for source in mapping["sources"]:
            if source.get("method") != "index":
                raise ValueError(
                    f"Feature {mapping['lerobot_key']!r} source "
                    f"{source['h5_path']!r} must use align='index'."
                )
    spec["downsample"] = normalize_index_sampling(config, target_fps=spec["fps"])
    return spec


def build_phase_cache(
    h5_dataset: H5Dataset,
    h5_file: Any,
    spec: Mapping[str, Any],
    h5_path: Path,
    *,
    phase_index: int,
) -> dict[str, Any]:
    master_timestamp_path = spec["master_timestamp_path"]
    stride = int(spec["downsample"]["stride"])
    if phase_index not in spec["downsample"]["phases"]:
        raise ValueError(f"Unknown phase index: {phase_index}")

    dataset_paths = {master_timestamp_path}
    for mapping in spec["mappings"]:
        dataset_paths.update(source["h5_path"] for source in mapping["sources"])
    dataset_cache = {
        path: h5_dataset._dataset(h5_file, path, h5_path)
        for path in dataset_paths
    }
    timestamp_cache = {
        master_timestamp_path: dataset_cache[master_timestamp_path][:]
    }
    h5_dataset._validate_index_sources(
        dataset_cache=dataset_cache,
        timestamp_cache=timestamp_cache,
        mappings=spec["mappings"],
        master_timestamp_path=master_timestamp_path,
        h5_path=h5_path,
    )

    master_timestamps = timestamp_cache[master_timestamp_path]
    source_indices = h5_dataset.np.arange(
        phase_index,
        len(master_timestamps),
        stride,
        dtype=h5_dataset.np.int64,
    )
    if source_indices.size == 0:
        raise ValueError(
            f"Episode {h5_path} has no source row for phase {phase_index}."
        )
    target_timestamps = master_timestamps[source_indices]
    cache = {
        "datasets": dataset_cache,
        "timestamps": timestamp_cache,
        "timestamp_seconds": {},
        "master_timestamp_path": master_timestamp_path,
        "target_timestamps": target_timestamps,
        "snapshot_indices": source_indices,
        "aligned": {},
        "index_stride": True,
        "phase_index": phase_index,
        "stride": stride,
    }
    for mapping in spec["mappings"]:
        cache["aligned"][mapping["lerobot_key"]] = (
            h5_dataset._sample_snapshot_mapping(mapping, source_indices, cache)
        )
    return cache


def ensure_new_outputs(output_paths: Sequence[Path]) -> None:
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("Phase output paths must be different.")
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Phase output already exists; refusing to mix datasets: "
            + ", ".join(str(path) for path in existing)
        )


def write_phase_manifest(
    output_path: Path,
    *,
    input_path: Path,
    fps: int,
    source_fps: int,
    stride: int,
    phase_index: int,
    episodes: Sequence[Mapping[str, Any]],
) -> Path:
    manifest_path = output_path / "meta" / MANIFEST_NAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "mode": INDEX_SAMPLING_MODE,
        "input": str(input_path),
        "fps": fps,
        "source_fps": source_fps,
        "stride": stride,
        "phase_index": phase_index,
        "source_index_rule": f"{phase_index} + k * {stride}",
        "episodes": list(episodes),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def run_conversion(args: argparse.Namespace) -> None:
    config = load_shape_meta(args.config)
    spec = conversion_spec(config)
    input_path = config_path(config, "input", override=getattr(args, "input", None))
    output_paths = {
        0: config_path(
            config,
            "output_phase0",
            override=getattr(args, "output_phase0", None),
        ),
        1: config_path(
            config,
            "output_phase1",
            override=getattr(args, "output_phase1", None),
        ),
    }
    phases = spec["downsample"]["phases"]
    if phases != (0, 1):
        raise ValueError("This converter writes exactly the two phases [0, 1].")
    ensure_new_outputs([output_paths[phase] for phase in phases])

    h5py, np, LeRobotDataset = load_conversion_deps()
    h5_dataset = H5Dataset(
        input_path,
        h5py=h5py,
        np=np,
        max_episodes=config_int(config, "max_episodes"),
    )
    writers = {
        phase: LeRobotV3Dataset(
            LeRobotDataset,
            repo_id=config_str(
                config,
                f"repo_id_phase{phase}",
                f"local/index_stride_phase{phase}",
            ),
            root=output_paths[phase],
            fps=spec["fps"],
            features=spec["lerobot_features"],
            no_videos=config_bool(config, "no_videos"),
        )
        for phase in phases
    }
    episode_manifests = {phase: [] for phase in phases}

    h5_files = h5_dataset.files()
    episode_iter = tqdm(h5_files, desc="source episodes", unit="episode")
    try:
        for episode_index, h5_path in enumerate(episode_iter):
            episode_iter.set_postfix_str(h5_path.name)
            with h5_dataset.open_episode(h5_path) as h5_file:
                source_frame_count = len(h5_file[spec["master_timestamp_path"]])
                for phase in phases:
                    cache = None
                    try:
                        cache = build_phase_cache(
                            h5_dataset,
                            h5_file,
                            spec,
                            h5_path,
                            phase_index=phase,
                        )
                        output_frame_count = h5_dataset.episode_length(
                            h5_file,
                            spec["master_timestamp_path"],
                            h5_path,
                            cache,
                        )
                        for frame_idx in range(output_frame_count):
                            frame = h5_dataset.read_frame(
                                h5_file,
                                frame_idx,
                                spec["mappings"],
                                h5_path,
                                spec["master_timestamp_path"],
                                cache,
                            )
                            writers[phase].add_frame(frame, task=spec["task"])
                        indices = cache["snapshot_indices"]
                        episode_manifests[phase].append(
                            {
                                "episode_index": episode_index,
                                "source_h5": h5_path.name,
                                "source_frame_count": source_frame_count,
                                "output_frame_count": output_frame_count,
                                "first_source_index": int(indices[0]),
                                "last_source_index": int(indices[-1]),
                            }
                        )
                    finally:
                        h5_dataset.clear_episode_cache(cache)
                    writers[phase].save_episode(task=spec["task"])
    finally:
        for writer in writers.values():
            writer.finalize()

    for phase in phases:
        manifest_path = write_phase_manifest(
            output_paths[phase],
            input_path=input_path,
            fps=spec["fps"],
            source_fps=spec["downsample"]["source_fps"],
            stride=spec["downsample"]["stride"],
            phase_index=phase,
            episodes=episode_manifests[phase],
        )
        print(f"phase {phase} manifest: {manifest_path}")

    if config_bool(config, "push_to_hub"):
        for writer in writers.values():
            writer.push_to_hub()


def main() -> None:
    run_conversion(parse_args())


if __name__ == "__main__":
    main()
