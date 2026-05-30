from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

from data_converter.hirol_reader import HiROLEpisodeReader
from diffusion_policy.common.lerobot_v3_io import LeRobotV3Writer


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.3f}s"


def _progress_bar(current: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = int(width * current / total)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _selected_indices(reader: HiROLEpisodeReader, missing_policy: str) -> List[int]:
    if missing_policy == "zeros":
        return list(range(len(reader)))
    selected: List[int] = []
    for i, record in enumerate(reader.records):
        missing = reader.missing_cameras(record)
        if missing_policy == "error" and missing:
            raise FileNotFoundError(
                f"Episode {reader.episode_dir.name} step={record.get('idx', i)} has missing cameras: {missing}"
            )
        if missing_policy == "skip" and missing:
            continue
        selected.append(i)
    return selected


def _infer_fps(episode_dirs: Iterable[Path], missing_policy: str) -> int:
    for ep_dir in episode_dirs:
        reader = HiROLEpisodeReader(ep_dir)
        indices = _selected_indices(reader, missing_policy)
        timestamps: List[float] = []
        for idx in indices:
            timestamps.append(reader.extract_primary_timestamp(reader.records[idx]))
        timestamps_np = np.asarray(timestamps, dtype=np.float64)
        finite = timestamps_np[np.isfinite(timestamps_np)]
        if finite.shape[0] < 2:
            continue
        diffs = np.diff(finite)
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            continue
        median_dt = float(np.median(diffs))
        if median_dt > 0:
            return max(1, int(round(1.0 / median_dt)))
    return 10


def _build_feature_spec(image_shape: Sequence[int], camera_keys: Sequence[str]) -> Dict[str, Dict]:
    state_names = [
        "ee_x",
        "ee_y",
        "ee_z",
        "ee_qx",
        "ee_qy",
        "ee_qz",
        "ee_qw",
        "joint_1.pos",
        "joint_2.pos",
        "joint_3.pos",
        "joint_4.pos",
        "joint_5.pos",
        "joint_6.pos",
        "joint_7.pos",
        "gripper_width",
    ]
    joint_names = [f"joint_{i}.pos" for i in range(1, 8)]
    ee_names = ["ee_x", "ee_y", "ee_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw"]
    features: Dict[str, Dict] = {
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (15,), "names": state_names},
        "observation.state.ee_pose": {"dtype": "float32", "shape": (7,), "names": ee_names},
        "observation.state.joint_position": {"dtype": "float32", "shape": (7,), "names": joint_names},
        "observation.state.gripper_width": {"dtype": "float32", "shape": (1,), "names": ["gripper_width"]},
        "action": {"dtype": "float32", "shape": (15,), "names": [*ee_names, *joint_names, "gripper_width"]},
        "action.ee_pose": {"dtype": "float32", "shape": (7,), "names": ee_names},
        "action.joint_position": {"dtype": "float32", "shape": (7,), "names": joint_names},
        "action.gripper_width": {"dtype": "float32", "shape": (1,), "names": ["gripper_width"]},
    }
    for camera_key in camera_keys:
        features[f"observation.images.{camera_key}"] = {
            "dtype": "video",
            "shape": tuple(image_shape),
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.fps": None,
                "video.codec": "mp4v",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    return features


def convert_dataset(
    input_root: Path,
    output_dir: Path,
    missing_policy: str,
    fps: int | None,
    use_videos: bool,
    robot_type: str,
) -> None:
    episode_dirs = HiROLEpisodeReader.list_episode_dirs(input_root)
    if not episode_dirs:
        raise RuntimeError(f"No episode_* directories found in: {input_root}")

    first_reader = HiROLEpisodeReader(episode_dirs[0])
    image_shape = first_reader.infer_image_shape()
    camera_keys = first_reader.camera_keys
    if fps is None:
        fps = _infer_fps(episode_dirs, missing_policy)

    video_keys = [f"observation.images.{camera_key}" for camera_key in camera_keys]
    dataset = LeRobotV3Writer(
        root=str(output_dir),
        fps=fps,
        features=_build_feature_spec(image_shape=image_shape, camera_keys=camera_keys),
        video_keys=video_keys if use_videos else [],
        robot_type=robot_type,
        image_color_space="bgr" if use_videos else "rgb",
    )

    summaries: List[Dict] = []
    total_episodes = len(episode_dirs)
    dataset_start = time.perf_counter()
    task_to_index: Dict[str, int] = {}
    print(f"Found {total_episodes} episode(s) under: {input_root}")
    print(
        f"{_progress_bar(0, total_episodes)} 0/{total_episodes} 0.0% total_elapsed=0.000s",
        flush=True,
    )

    for ep_idx, ep_dir in enumerate(episode_dirs, start=1):
        episode_start = time.perf_counter()
        reader = HiROLEpisodeReader(ep_dir)
        indices = _selected_indices(reader, missing_policy)
        fill_missing = "zeros" if missing_policy in {"zeros", "skip"} else "error"
        image_shape = reader.infer_image_shape()
        task_text = ""
        if isinstance(reader.text, dict):
            task_text = str(
                reader.text.get("task")
                or reader.text.get("description")
                or reader.text.get("instruction")
                or ""
            )
        if task_text not in task_to_index:
            task_to_index[task_text] = len(task_to_index)
        task_index = task_to_index[task_text]

        written_steps = 0
        for local_step_idx, raw_idx in enumerate(indices):
            fallback_timestamp = local_step_idx / max(fps, 1)
            frame = reader.get_lerobot_frame(
                raw_idx,
                fallback_timestamp=fallback_timestamp,
                episode_index=ep_idx - 1,
                task_index=task_index,
                fill_missing=fill_missing,
                image_shape=image_shape,
                image_color_space="bgr" if use_videos else "rgb",
            )
            dataset.add_frame(frame=frame)
            written_steps += 1

        dataset.save_episode(task=task_text)
        episode_elapsed = time.perf_counter() - episode_start
        total_elapsed = time.perf_counter() - dataset_start
        summary = {
            "episode": ep_dir.name,
            "raw_steps": len(reader),
            "written_steps": written_steps,
            "skipped": len(reader) - written_steps,
            "primary_stream": reader.primary_stream,
        }
        summaries.append(summary)
        progress = (ep_idx / total_episodes) * 100 if total_episodes else 100.0
        print(
            f"{_progress_bar(ep_idx, total_episodes)} {ep_idx}/{total_episodes} {progress:.1f}% "
            f"[{summary['episode']}] episode_elapsed={_format_seconds(episode_elapsed)} "
            f"total_elapsed={_format_seconds(total_elapsed)} raw={summary['raw_steps']} "
            f"written={summary['written_steps']} skipped={summary['skipped']} "
            f"primary_stream={summary['primary_stream']}",
            flush=True,
        )

    total_raw = sum(summary["raw_steps"] for summary in summaries)
    total_written = sum(summary["written_steps"] for summary in summaries)
    dataset.finalize()
    print(f"Done. episodes={total_episodes} raw_steps={total_raw} written_steps={total_written}")
    print(f"LeRobot v3 output: {output_dir}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert HIROL episodes to an official LeRobot v3 dataset.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("dataset/duo_unitree_pick_n_place"),
        help="Input dataset root containing episode_* folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/duo_unitree_pick_n_place_lerobot_v3"),
        help="Output LeRobot v3 dataset directory path.",
    )
    parser.add_argument(
        "--missing-policy",
        choices=["error", "skip", "zeros"],
        default="zeros",
        help="How to handle steps with missing image files.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Dataset FPS. Defaults to inference from timestamps.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="fr3",
        help="robot_type written into meta/info.json.",
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Disable MP4 packing and keep RGB frames as parquet values.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    start = time.perf_counter()
    convert_dataset(
        input_root=args.input_root,
        output_dir=args.output_dir,
        missing_policy=args.missing_policy,
        fps=args.fps,
        use_videos=not args.no_videos,
        robot_type=args.robot_type,
    )
    elapsed = time.perf_counter() - start
    print(f"elapsed={_format_seconds(elapsed)}")


if __name__ == "__main__":
    main()
