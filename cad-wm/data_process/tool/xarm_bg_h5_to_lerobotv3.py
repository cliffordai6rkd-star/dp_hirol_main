"""Convert dual xArm background H5 recordings to one 7-DoF LeRobot v3 dataset.

Each input H5 has left-arm joints in columns 0:7 and right-arm joints in 7:14.
The four training features are sampled from the same recorder row on a 100 Hz
Unix-aligned clock. One H5 file becomes two single-arm episodes, left then right.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.tool.h5_2_lerobotev3 import (  # noqa: E402
    H5Dataset,
    LeRobotV3Dataset,
    build_conversion_spec,
    load_conversion_deps,
    resolve_io_path,
)


FEATURE_PATHS = {
    "observation.joint": "teleop/q_follower",
    "observation.velocity": "teleop/dq_follower",
    "observation.torque": "teleop/tau_follower",
}
REQUIRED_PATHS = (*FEATURE_PATHS.values(), "teleop/q_cmd")
VALIDITY_PATHS = (
    "teleop/q_follower_valid",
    "teleop/dq_valid_follower",
    "teleop/torque_valid_follower",
)
ARM_SLICES = {"left": slice(0, 7), "right": slice(7, 14)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/xarm_bg"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/train_episode/xarm_bg_lbv3"),
        help="Directory for the combined LeRobot v3 dataset.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


def conversion_spec() -> dict:
    return build_conversion_spec(
        {
            "task": "xarm_background",
            "fps": 100,
            "master_timestamp_path": "teleop/timestamp_us",
            "sampling": {
                "mode": "fixed_rate_causal_snapshot",
                "phase": "unix_epoch",
                "max_staleness_s": 0.03,
            },
            "features": {
                **{
                    key: {
                        "dtype": "float32",
                        "shape": [7],
                        "h5_path": path,
                        "align": "index",
                    }
                    for key, path in FEATURE_PATHS.items()
                },
                "observation.delta_q": {
                    "dtype": "float32",
                    "shape": [7],
                    "sources": [
                        {"h5_path": "teleop/q_cmd", "align": "index"},
                        {"h5_path": "teleop/q_follower", "align": "index"},
                    ],
                    "combine": "subtract",
                },
            },
        }
    )


def validate_episode(h5_file, h5_path: Path, np) -> None:
    arm_names_path = "metadata/arm_names_json"
    if arm_names_path not in h5_file or json.loads(h5_file[arm_names_path][()]) != ["left", "right"]:
        raise ValueError(f"{h5_path}: expected metadata arm order ['left', 'right']")
    timestamp_path = "teleop/timestamp_us"
    if timestamp_path not in h5_file:
        raise ValueError(f"{h5_path}: missing {timestamp_path}")
    timestamps = np.asarray(h5_file[timestamp_path])
    if timestamps.ndim != 1 or len(timestamps) == 0:
        raise ValueError(f"{h5_path}: {timestamp_path} must be a nonempty 1-D array")
    if not np.issubdtype(timestamps.dtype, np.integer) or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{h5_path}: {timestamp_path} must contain increasing integer microseconds")

    for path in REQUIRED_PATHS:
        if path not in h5_file:
            raise ValueError(f"{h5_path}: missing {path}")
        dataset = h5_file[path]
        if dataset.shape != (len(timestamps), 14):
            raise ValueError(f"{h5_path}: {path} must have shape ({len(timestamps)}, 14), got {dataset.shape}")
        if not np.isfinite(dataset[:]).all():
            raise ValueError(f"{h5_path}: {path} contains nonfinite values")

    for path in VALIDITY_PATHS:
        if path in h5_file and not np.all(h5_file[path][:]):
            raise ValueError(f"{h5_path}: {path} contains invalid recorder rows")


def convert(*, input_path: Path, output: Path, max_episodes: int | None = None) -> Path:
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")

    h5py, np, LeRobotDataset = load_conversion_deps()
    input_path = resolve_io_path(input_path)
    output = resolve_io_path(output)
    if output.exists():
        raise FileExistsError(f"Output dataset already exists: {output}")

    h5_dataset = H5Dataset(input_path, h5py=h5py, np=np, max_episodes=max_episodes)
    h5_files = h5_dataset.files()
    for h5_path in h5_files:
        with h5_dataset.open_episode(h5_path) as h5_file:
            validate_episode(h5_file, h5_path, np)

    spec = conversion_spec()
    dataset = LeRobotV3Dataset(
        LeRobotDataset,
        repo_id="xarm_bg_lbv3",
        root=output,
        fps=spec["fps"],
        features=spec["lerobot_features"],
        no_videos=True,
    )
    try:
        for h5_path in h5_files:
            with h5_dataset.open_episode(h5_path) as h5_file:
                cache = h5_dataset.build_episode_cache(
                    h5_file,
                    spec["mappings"],
                    spec["master_timestamp_path"],
                    spec["fps"],
                    h5_path,
                    timeline=spec["timeline"],
                    sampling=spec["sampling"],
                )
                try:
                    frame_count = h5_dataset.episode_length(
                        h5_file, spec["master_timestamp_path"], h5_path, cache
                    )
                    print(f"{h5_path.name}: {frame_count} frames × 2 arms", flush=True)
                    for name in ARM_SLICES:
                        arm_task = f"xarm_background_{name}"
                        for frame_idx in range(frame_count):
                            full_frame = h5_dataset.read_frame(
                                h5_file,
                                frame_idx,
                                spec["mappings"],
                                h5_path,
                                spec["master_timestamp_path"],
                                cache,
                            )
                            arm_frame = {
                                key: np.asarray(value[ARM_SLICES[name]], dtype=np.float32)
                                for key, value in full_frame.items()
                            }
                            dataset.add_frame(arm_frame, task=arm_task)
                        dataset.save_episode(task=arm_task)
                finally:
                    h5_dataset.clear_episode_cache(cache)
    finally:
        dataset.finalize()
    return output


def main() -> None:
    args = parse_args()
    output = convert(
        input_path=args.input,
        output=args.output,
        max_episodes=args.max_episodes,
    )
    print(f"Dataset: {output} (repo_id=xarm_bg_lbv3)")


if __name__ == "__main__":
    main()
