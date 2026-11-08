"""Convert synchronized camera-row H5 episodes to policy-ready LeRobot v3.

Every output row is anchored by one recorded master-camera timestamp. Media
features are read by the same H5 row index, while all numeric features select
the newest source sample not later than that timestamp. Actions are single
``[Da]`` vectors; the policy loader builds future action chunks across rows.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from tqdm import tqdm

from data_process.tool.h5_2_lerobotev3 import (
    H5Dataset,
    LeRobotV3Dataset,
    build_conversion_spec as build_base_conversion_spec,
    config_bool,
    config_int,
    config_path,
    config_str,
    load_conversion_deps,
    load_h5py,
    load_shape_meta,
)


EXAMPLE_SHAPE_META = {
    "io": {
        "input": "nero_ws/runs/example",
        "output": "data/train_episode/example_va_lbv3",
        "repo_id": "example_va_lbv3",
        "no_videos": False,
        "push_to_hub": False,
    },
    "task": "example",
    "fps": 25,
    "master_timestamp_path": "cameras/wrist/timestamp_us",
    "master_timeline": {"max_gap_s": 0.02, "store_timestamps": True},
    "features": {
        "observation.images.wrist": {
            "dtype": "video",
            "shape": [192, 256, 3],
            "h5_path": "cameras/wrist/frames",
            "align": "index",
        },
        "observation.images.side": {
            "dtype": "video",
            "shape": [192, 256, 3],
            "h5_path": "cameras/side/frames",
            "align": "index",
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [7],
            "h5_path": "teleop/q_follower",
            "timestamp_path": "teleop/timestamp_us",
            "align": "previous",
        },
        "observation.delta_q": {
            "dtype": "float32",
            "shape": [7],
            "sources": [
                {"h5_path": "teleop/q_cmd", "timestamp_path": "teleop/timestamp_us", "align": "previous"},
                {"h5_path": "teleop/q_follower", "timestamp_path": "teleop/timestamp_us", "align": "previous"},
            ],
            "combine": "subtract",
        },
        "observation.tau_ext": {
            "dtype": "float32",
            "shape": [7],
            "h5_path": "teleop/tau_ext_cal",
            "timestamp_path": "teleop/timestamp_us",
            "align": "previous",
        },
        "action.joint": {
            "dtype": "float32",
            "shape": [7],
            "h5_path": "teleop/q_cmd",
            "timestamp_path": "teleop/timestamp_us",
            "align": "previous",
        },
        "action.ee_pose": {
            "dtype": "float32",
            "shape": [7],
            "h5_path": "teleop/ee_pose_follower",
            "timestamp_path": "teleop/timestamp_us",
            "align": "previous",
            "transform": "ee_pose_matrix_to_quaternion",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert H5 episodes on a recorded master-camera timeline with "
            "per-row single-step actions."
        )
    )
    parser.add_argument("--config", "-c", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--print-example-shape-meta", action="store_true")
    return parser.parse_args()


def _is_media_feature(feature_spec: Mapping[str, Any]) -> bool:
    dtype = str(
        feature_spec.get("dtype", feature_spec.get("type", ""))
    ).lower()
    return dtype in {"image", "video"}


def _normalize_master_timeline(value: Any) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("shape_meta.master_timeline must be a mapping.")
    unknown = set(value) - {"max_gap_s", "store_timestamps"}
    if unknown:
        raise ValueError(
            f"shape_meta.master_timeline has unknown options: {sorted(unknown)}"
        )
    max_gap_s = value.get("max_gap_s")
    if max_gap_s is not None:
        max_gap_s = float(max_gap_s)
        if not math.isfinite(max_gap_s) or max_gap_s <= 0.0:
            raise ValueError("master_timeline.max_gap_s must be positive and finite.")
    return {
        "max_gap_s": max_gap_s,
        "store_timestamps": bool(value.get("store_timestamps", True)),
    }


def homogeneous_pose_to_xyz_quat_xyzw(value: Any, np_module=None):
    """Convert homogeneous pose matrices to ``xyz + quaternion(xyzw)``.

    Args:
        value: One ``[4,4]`` pose or a batch with shape ``[...,4,4]``.
        np_module: Optional NumPy module injection used by the H5 reader/tests.

    Returns:
        A float32 array with shape ``[...,7]`` ordered as
        ``[x, y, z, qx, qy, qz, qw]``. Quaternions are unit-normalized and use
        a non-negative ``qw`` representative to avoid arbitrary sign flips.
    """

    if np_module is None:
        import numpy as np_module  # type: ignore

    poses = np_module.asarray(value, dtype=np_module.float64)
    if poses.shape[-2:] != (4, 4):
        raise ValueError(
            "homogeneous pose must have shape (..., 4, 4), "
            f"got {poses.shape}"
        )
    if not np_module.isfinite(poses).all():
        raise ValueError("homogeneous pose contains non-finite values")

    flat_poses = poses.reshape(-1, 4, 4)
    expected_bottom = np_module.asarray(
        [0.0, 0.0, 0.0, 1.0], dtype=np_module.float64
    )
    if not np_module.allclose(
        flat_poses[:, 3, :], expected_bottom[None, :], rtol=0.0, atol=1.0e-5
    ):
        raise ValueError(
            "pose is not a standard homogeneous transform: expected bottom "
            "row [0, 0, 0, 1]"
        )

    converted = np_module.empty((flat_poses.shape[0], 7), dtype=np_module.float32)
    converted[:, :3] = flat_poses[:, :3, 3]
    for index, pose in enumerate(flat_poses):
        matrix = pose[:3, :3]
        trace = float(np_module.trace(matrix))
        if trace > 0.0:
            scale = float(np_module.sqrt(trace + 1.0) * 2.0)
            qw = 0.25 * scale
            qx = (matrix[2, 1] - matrix[1, 2]) / scale
            qy = (matrix[0, 2] - matrix[2, 0]) / scale
            qz = (matrix[1, 0] - matrix[0, 1]) / scale
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            scale = float(
                np_module.sqrt(
                    1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
                )
                * 2.0
            )
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif matrix[1, 1] > matrix[2, 2]:
            scale = float(
                np_module.sqrt(
                    1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
                )
                * 2.0
            )
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = float(
                np_module.sqrt(
                    1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
                )
                * 2.0
            )
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale

        quaternion = np_module.asarray(
            [qx, qy, qz, qw], dtype=np_module.float64
        )
        norm = float(np_module.linalg.norm(quaternion))
        if not math.isfinite(norm) or norm <= 1.0e-12:
            raise ValueError("rotation matrix produced an invalid quaternion")
        quaternion /= norm
        if quaternion[3] < 0.0:
            quaternion = -quaternion
        converted[index, 3:] = quaternion

    return converted.reshape(poses.shape[:-2] + (7,))


def build_conversion_spec(shape_meta: Mapping[str, Any]) -> dict[str, Any]:
    """Build and validate the dedicated vision-action conversion contract."""

    if "timeline" in shape_meta or "sampling" in shape_meta:
        raise ValueError(
            "VA_h5_v3 uses master_timeline and must not define timeline or sampling."
        )
    base_config = dict(shape_meta)
    master_timeline = _normalize_master_timeline(
        base_config.pop("master_timeline", None)
    )
    spec = build_base_conversion_spec(base_config)

    action_mappings = []
    for mapping in spec["mappings"]:
        key = mapping["lerobot_key"]
        media = _is_media_feature(mapping["feature_spec"])
        expected_method = "index" if media else "previous"
        for source in mapping["sources"]:
            method = source["method"]
            if method != expected_method:
                kind = "camera/media" if media else "low-dimensional"
                raise ValueError(
                    f"Feature {key!r} is {kind} data and must use "
                    f"align={expected_method!r}, got {method!r}."
                )
            if not media and not source.get("timestamp_path"):
                raise ValueError(
                    f"Low-dimensional feature {key!r} needs timestamp_path."
                )
        if media and len(mapping["sources"]) != 1:
            raise ValueError(
                f"Camera/media feature {key!r} must have exactly one H5 source."
            )

        if key == "action" or key.startswith("action."):
            action_mappings.append(mapping)
            shape = tuple(mapping["feature_spec"]["shape"])
            if len(shape) != 1:
                raise ValueError(
                    f"Action feature {key!r} must be one [Da] vector per row; "
                    f"got shape={shape}. Action chunks belong in the policy loader."
                )

    if not action_mappings:
        raise ValueError(
            "VA_h5_v3 requires at least one single-step action feature."
        )

    if master_timeline["store_timestamps"]:
        timing_key = "timing.master_timestamp_ns"
        if timing_key in spec["lerobot_features"]:
            raise ValueError(f"Generated feature {timing_key!r} is declared manually.")
        spec["lerobot_features"][timing_key] = {
            "dtype": "int64",
            "shape": (1,),
        }
    spec["master_timeline"] = master_timeline
    return spec


class VAH5Dataset(H5Dataset):
    """Preserve camera rows and causally snapshot all numeric features."""

    def _ee_pose_matrix_to_quaternion(self, value):
        """Use the VA-local homogeneous-pose conversion implementation."""

        if self.np is None:
            raise RuntimeError("VA pose conversion requires numpy.")
        return homogeneous_pose_to_xyz_quat_xyzw(value, self.np)

    def build_episode_cache(
        self,
        h5_file,
        mappings,
        master_timestamp_path,
        fps,
        h5_path,
        *,
        master_timeline,
    ):
        del fps  # Metadata only; no synthetic timeline is generated.
        if self.np is None:
            raise RuntimeError("VA conversion requires numpy.")

        dataset_cache = {}
        timestamp_cache = {}
        dataset_paths = {master_timestamp_path}
        timestamp_paths = {master_timestamp_path}
        for mapping in mappings:
            for source in mapping["sources"]:
                dataset_paths.add(source["h5_path"])
                if source.get("timestamp_path"):
                    timestamp_paths.add(source["timestamp_path"])

        for field_path in dataset_paths | timestamp_paths:
            dataset_cache[field_path] = self._dataset(
                h5_file, field_path, h5_path
            )
        for timestamp_path in timestamp_paths:
            timestamp_cache[timestamp_path] = dataset_cache[timestamp_path][:]

        master_raw = self.np.asarray(
            timestamp_cache[master_timestamp_path]
        ).reshape(-1)
        master_seconds = self._timestamps_seconds(
            master_raw, master_timestamp_path
        )
        if master_seconds.size == 0:
            raise ValueError(f"Master-camera timeline is empty in {h5_path}.")

        timestamp_seconds = {master_timestamp_path: master_seconds}
        media_lengths = [len(master_raw)]
        for mapping in mappings:
            if not _is_media_feature(mapping["feature_spec"]):
                continue
            source = mapping["sources"][0]
            dataset = dataset_cache[source["h5_path"]]
            if not dataset.shape:
                raise ValueError(
                    f"Camera/media feature {mapping['lerobot_key']!r} in "
                    f"{h5_path} must have a row dimension."
                )
            media_lengths.append(int(dataset.shape[0]))

        # Cameras are paired strictly by their original H5 index. Acquisition
        # can leave one unmatched terminal frame, so keep only their common
        # prefix rather than timestamp-matching or shifting either stream.
        common_media_rows = min(media_lengths)
        candidate_rows = self.np.arange(common_media_rows, dtype=self.np.int64)
        candidate_seconds = master_seconds[candidate_rows]
        history_valid = self.np.ones(common_media_rows, dtype=bool)
        for mapping in mappings:
            if _is_media_feature(mapping["feature_spec"]):
                continue
            for source in mapping["sources"]:
                timestamp_path = source["timestamp_path"]
                source_seconds = timestamp_seconds.get(timestamp_path)
                if source_seconds is None:
                    source_seconds = self._timestamps_seconds(
                        timestamp_cache[timestamp_path], timestamp_path
                    )
                    timestamp_seconds[timestamp_path] = source_seconds
                history_indices = self.np.searchsorted(
                    source_seconds, candidate_seconds, side="right"
                ) - 1
                history_valid &= history_indices >= 0

        selected_master_indices = candidate_rows[history_valid]
        if selected_master_indices.size == 0:
            raise ValueError(
                f"No master-camera row has a causal low-dimensional sample in {h5_path}."
            )
        selected_master_seconds = master_seconds[selected_master_indices]
        selected_master_raw = master_raw[selected_master_indices]
        cache = {
            "datasets": dataset_cache,
            "timestamps": timestamp_cache,
            "timestamp_seconds": timestamp_seconds,
            "target_timestamps": selected_master_raw,
            "aligned": {},
            "master_timeline": True,
            "selected_master_indices": selected_master_indices,
            "dropped_leading_rows": int(selected_master_indices[0]),
            "dropped_media_tail_rows": int(len(master_raw) - common_media_rows),
            "master_timestamp_ns": self._timestamps_ns(
                selected_master_raw, master_timestamp_path
            ),
            "store_timestamps": bool(master_timeline["store_timestamps"]),
        }

        for mapping in mappings:
            if _is_media_feature(mapping["feature_spec"]):
                continue
            for source in mapping["sources"]:
                timestamp_path = source["timestamp_path"]
                source_seconds = timestamp_seconds.get(timestamp_path)
                if source_seconds is None:
                    source_seconds = self._timestamps_seconds(
                        timestamp_cache[timestamp_path], timestamp_path
                    )
                    timestamp_seconds[timestamp_path] = source_seconds

                indices = self._point_sample_indices(
                    source_seconds, selected_master_seconds, "previous"
                )
                max_gap_s = source.get("max_gap_s")
                if max_gap_s is None:
                    max_gap_s = mapping.get("max_gap_s")
                if max_gap_s is None:
                    max_gap_s = master_timeline.get("max_gap_s")
                if max_gap_s is not None and not source.get("allow_stale"):
                    ages = selected_master_seconds - source_seconds[indices]
                    invalid = ages > float(max_gap_s) + 1.0e-12
                    if self.np.any(invalid):
                        first = int(self.np.flatnonzero(invalid)[0])
                        raise ValueError(
                            f"Feature {mapping['lerobot_key']!r} in {h5_path} "
                            f"is stale by {ages[first]:.6f}s at master row "
                            f"{first}; maximum is {float(max_gap_s):.6f}s."
                        )

            cache["aligned"][mapping["lerobot_key"]] = (
                self._resample_mapping(mapping, selected_master_seconds, cache)
            )
        return cache

    def read_frame(
        self,
        h5_file,
        frame_idx,
        mappings,
        h5_path,
        master_timestamp_path,
        cache=None,
    ):
        if cache is None or not cache.get("master_timeline"):
            return super().read_frame(
                h5_file,
                frame_idx,
                mappings,
                h5_path,
                master_timestamp_path,
                cache,
            )

        frame = {}
        master_index = int(cache["selected_master_indices"][frame_idx])
        for mapping in mappings:
            key = mapping["lerobot_key"]
            if _is_media_feature(mapping["feature_spec"]):
                value = self._read_paths_at_index(
                    h5_file,
                    mapping["h5_paths"],
                    master_index,
                    h5_path,
                    mapping["feature_spec"],
                    cache,
                    combine=mapping.get("combine"),
                    feature_name=key,
                )
                frame[key] = self._apply_transform(
                    value, mapping.get("transform")
                )
            else:
                frame[key] = cache["aligned"][key][frame_idx]
        if cache is not None and cache.get("store_timestamps"):
            frame["timing.master_timestamp_ns"] = self.np.asarray(
                [cache["master_timestamp_ns"][frame_idx]], dtype=self.np.int64
            )
        return frame


def run_inspect(args: argparse.Namespace) -> None:
    config = load_shape_meta(args.config)
    dataset = VAH5Dataset(
        config_path(config, "input", override=args.input),
        h5py=load_h5py(),
        max_episodes=config_int(config, "max_episodes"),
    )
    dataset.inspect()


def run_conversion(args: argparse.Namespace) -> None:
    config = load_shape_meta(args.config)
    h5py, np, LeRobotDataset = load_conversion_deps()
    spec = build_conversion_spec(config)
    h5_dataset = VAH5Dataset(
        config_path(config, "input", override=args.input),
        h5py=h5py,
        np=np,
        max_episodes=config_int(config, "max_episodes"),
    )
    lerobot_dataset = LeRobotV3Dataset(
        LeRobotDataset,
        repo_id=config_str(config, "repo_id", "local/va_h5_v3"),
        root=config_path(config, "output", override=args.output),
        fps=spec["fps"],
        features=spec["lerobot_features"],
        no_videos=config_bool(config, "no_videos"),
    )

    try:
        for h5_path in tqdm(h5_dataset.files(), desc="episodes", unit="episode"):
            with h5_dataset.open_episode(h5_path) as h5_file:
                cache = None
                try:
                    cache = h5_dataset.build_episode_cache(
                        h5_file,
                        spec["mappings"],
                        spec["master_timestamp_path"],
                        spec["fps"],
                        h5_path,
                        master_timeline=spec["master_timeline"],
                    )
                    dropped_start = cache["dropped_leading_rows"]
                    dropped_end = cache["dropped_media_tail_rows"]
                    if dropped_start or dropped_end:
                        tqdm.write(
                            f"{h5_path.name}: kept common camera-index rows; "
                            f"dropped start={dropped_start}, end={dropped_end}"
                        )
                    length = h5_dataset.episode_length(
                        h5_file,
                        spec["master_timestamp_path"],
                        h5_path,
                        cache,
                    )
                    for frame_idx in tqdm(
                        range(length),
                        desc=f"frames {h5_path.name}",
                        unit="frame",
                        leave=False,
                    ):
                        frame = h5_dataset.read_frame(
                            h5_file,
                            frame_idx,
                            spec["mappings"],
                            h5_path,
                            spec["master_timestamp_path"],
                            cache,
                        )
                        lerobot_dataset.add_frame(frame, task=spec["task"])
                finally:
                    h5_dataset.clear_episode_cache(cache)
            lerobot_dataset.save_episode(task=spec["task"])
    finally:
        lerobot_dataset.finalize()

    if config_bool(config, "push_to_hub"):
        lerobot_dataset.push_to_hub()


def main() -> None:
    args = parse_args()
    if args.print_example_shape_meta:
        print(json.dumps(EXAMPLE_SHAPE_META, indent=2, ensure_ascii=False))
        return
    if args.inspect_only:
        run_inspect(args)
        return
    run_conversion(args)


if __name__ == "__main__":
    main()
