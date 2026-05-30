"""将 wipe_board 的 episode_*.h5 文件转换成官方 LeRobot v3 数据集。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

#  拿父目录路径
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

h5py = None
np = None
LeRobotV3Writer = None

DEFAULT_CAMERAS: Sequence[str] = ("wrist", "side_1", "side_2")
TELEOP_GROUP = "teleop"
DEFAULT_STATE_FIELDS: Sequence[str] = ("ee_pose", "q_follower", "gripper_state")
DEFAULT_ACTION_FIELDS: Sequence[str] = ("cmd_ee_pose", "q_cmd", "gripper_action")


def _require_h5py():
    """延迟导入 h5py，让缺少依赖的环境也能正常查看 --help。"""
    global h5py
    if h5py is not None:
        return h5py
    try:
        import h5py as h5py_module
    except ImportError as exc:
        raise ImportError(
            "h5py is required for .h5 conversion. Please run this script in the project "
            "environment that has h5py installed."
        ) from exc
    h5py = h5py_module
    return h5py_module


def _require_h5_runtime() -> None:
    """在读取 H5 前加载 numpy 和 h5py。"""
    global np
    _require_h5py()
    if np is None:
        try:
            import numpy as np_module
        except ImportError as exc:
            raise ImportError(
                "numpy is required for .h5 conversion. Please run this script in the project "
                "environment that has numpy installed."
            ) from exc
        np = np_module


def _require_writer() -> None:
    """在正式转换前加载官方 LeRobot writer。"""
    global LeRobotV3Writer
    if LeRobotV3Writer is None:
        from diffusion_policy.common.lerobot_v3_io import LeRobotV3Writer as writer_cls

        LeRobotV3Writer = writer_cls


def _format_seconds(seconds: float) -> str:
    """把秒数格式化成简短的进度耗时文本。"""
    return f"{seconds:.3f}s"


def _progress_bar(current: int, total: int, width: int = 28) -> str:
    """根据 episode 转换进度生成命令行进度条。"""
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = int(width * current / total)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _parse_csv(value: Optional[str], default: Sequence[str]) -> List[str]:
    """解析逗号分隔的 CLI 参数；未传入时使用默认字段。"""
    if value is None:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _episode_paths(input_path: Path, max_episodes: Optional[int]) -> List[Path]:
    """收集待转换的 episode_*.h5 文件，并按文件名排序。"""
    input_path = input_path.expanduser()
    if input_path.is_file():
        paths = [input_path]
    else:
        paths = sorted(input_path.glob("episode_*.h5"))
    if max_episodes is not None:
        paths = paths[:max_episodes]
    if not paths:
        raise RuntimeError(f"No episode_*.h5 files found: {input_path}")
    return paths


def _as_1d_float(value) -> "np.ndarray":
    """把任意 H5 数值字段压平成 float32 一维向量。"""
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _to_hwc_uint8(image: "np.ndarray") -> "np.ndarray":
    """把单帧图像整理成 HWC uint8，兼容 CHW/HWC 和 0-1/0-255 输入。"""
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected one 3-D image frame, got shape {image.shape}")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.dtype != np.uint8:
        image = image.astype(np.float32, copy=False)
        max_value = float(np.nanmax(image)) if image.size else 0.0
        if max_value <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _decode_attr(value) -> str:
    """把 HDF5 attribute 转成可打印字符串。"""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _dataset_shape(h5_file, path: str) -> tuple[int, ...]:
    """读取指定 HDF5 dataset 的 shape，并在缺字段时给出明确错误。"""
    if path not in h5_file:
        raise KeyError(f"Missing HDF5 dataset: {path}")
    dataset = h5_file[path]
    if not isinstance(dataset, h5py.Dataset):
        raise KeyError(f"HDF5 path is not a dataset: {path}")
    return tuple(int(dim) for dim in dataset.shape)


def _teleop_path(field: str) -> str:
    """把低维字段短名映射到当前 H5 里的 teleop/<field> 路径。"""
    return f"{TELEOP_GROUP}/{field}"


def _field_dim(h5_file, field: str) -> int:
    """计算单个低维字段在每一帧展开后的维度。"""
    return int(np.prod(_dataset_shape(h5_file, _teleop_path(field))[1:], dtype=np.int64))


def _feature_dim(h5_file, fields: Sequence[str]) -> int:
    """计算多个低维字段拼接后的总维度。"""
    return int(sum(_field_dim(h5_file, field) for field in fields))


def _image_shape(h5_file, camera: str) -> tuple[int, int, int]:
    """读取相机图像 shape，并统一返回 LeRobot 需要的 HWC 形状。"""
    shape = _dataset_shape(h5_file, f"cameras/{camera}/frames")
    if len(shape) != 4:
        raise ValueError(f"Expected cameras/{camera}/frames to be 4-D, got {shape}")
    frame_shape = shape[1:]
    if frame_shape[0] in (1, 3) and frame_shape[-1] not in (1, 3):
        c, h, w = frame_shape
        return (h, w, c)
    h, w, c = frame_shape
    return (h, w, c)


def _field_names(fields: Sequence[str], h5_file) -> List[str]:
    """为拼接后的低维向量生成可读的维度名。"""
    names: List[str] = []
    for field in fields:
        dim = _field_dim(h5_file, field)
        if dim == 1:
            names.append(field)
        else:
            names.extend([f"{field}.{i}" for i in range(dim)])
    return names


class H5EpisodeReader:
    """读取一个 wipe_board episode_*.h5，并输出 LeRobot 单帧字典。"""

    def __init__(
        self,
        path: Path,
        *,
        cameras: Sequence[str],
        state_fields: Sequence[str],
        action_fields: Sequence[str],
    ):
        """打开 episode 文件，并检查本转换器需要的字段是否齐全。"""
        self.path = Path(path)
        self.file = h5py.File(self.path, "r")
        self.cameras = list(cameras)
        self.state_fields = list(state_fields)
        self.action_fields = list(action_fields)

        for camera in self.cameras:
            _dataset_shape(self.file, f"cameras/{camera}/frames")
            _dataset_shape(self.file, f"cameras/{camera}/timestamp_us")
        _dataset_shape(self.file, f"{TELEOP_GROUP}/timestamp_us")
        for field in [*self.state_fields, *self.action_fields]:
            _dataset_shape(self.file, _teleop_path(field))

        lengths = [
            int(self.file[_teleop_path(field)].shape[0])
            for field in [*self.state_fields, *self.action_fields]
        ]
        lengths.append(int(self.file[f"{TELEOP_GROUP}/timestamp_us"].shape[0]))
        lengths.extend(int(self.file[f"cameras/{camera}/frames"].shape[0]) for camera in self.cameras)
        self.length = min(lengths)
        if self.length <= 0:
            raise RuntimeError(f"Episode has no frames: {self.path}")

    def close(self) -> None:
        """关闭当前 HDF5 文件句柄。"""
        self.file.close()

    def __len__(self) -> int:
        """返回该 episode 可安全读取的帧数。"""
        return self.length

    def state_dim(self) -> int:
        """返回 observation.state 拼接后的总维度。"""
        return _feature_dim(self.file, self.state_fields)

    def action_dim(self) -> int:
        """返回 action 拼接后的总维度。"""
        return _feature_dim(self.file, self.action_fields)

    def state_names(self) -> List[str]:
        """返回 observation.state 每个维度对应的来源名字。"""
        return _field_names(self.state_fields, self.file)

    def action_names(self) -> List[str]:
        """返回 action 每个维度对应的来源名字。"""
        return _field_names(self.action_fields, self.file)

    def camera_shape(self, camera: str) -> tuple[int, int, int]:
        """返回指定相机图像的 HWC 形状。"""
        return _image_shape(self.file, camera)

    def field_dim(self, field: str) -> int:
        """返回单个 H5 低维字段展开后的维度。"""
        return _field_dim(self.file, field)

    def frame(self, index: int) -> Dict[str, object]:
        """读取一帧，并组装成 LeRobotV3Writer.add_frame 接收的字典。"""
        if index < 0 or index >= self.length:
            raise IndexError(f"index out of range: {index}")

        frame: Dict[str, object] = {}
        for camera in self.cameras:
            frame[f"observation.images.{camera}"] = _to_hwc_uint8(
                self.file[f"cameras/{camera}/frames"][index]
            )
            camera_timestamp = (
                float(np.asarray(self.file[f"cameras/{camera}/timestamp_us"][index]).reshape(-1)[0])
                / 1_000_000.0
            )
            frame[f"observation.images.{camera}.timestamp"] = np.asarray(
                [camera_timestamp],
                dtype=np.float32,
            )
            frame[f"observation.images.{camera}.is_valid"] = np.asarray([True], dtype=np.bool_)

        state_parts = [
            _as_1d_float(self.file[_teleop_path(field)][index])
            for field in self.state_fields
        ]
        action_parts = [
            _as_1d_float(self.file[_teleop_path(field)][index])
            for field in self.action_fields
        ]
        frame["observation.state"] = np.concatenate(state_parts, axis=0).astype(np.float32, copy=False)
        frame["action"] = np.concatenate(action_parts, axis=0).astype(np.float32, copy=False)
        for field, value in zip(self.state_fields, state_parts):
            frame[f"observation.{field}"] = value
        for field, value in zip(self.action_fields, action_parts):
            frame[f"action.{field}"] = value
        return frame


def _build_feature_spec(reader: H5EpisodeReader, fps: int) -> Dict[str, Dict]:
    """根据首个 episode 的真实 shape 构造官方 LeRobot features。"""
    features: Dict[str, Dict] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (reader.state_dim(),),
            "names": reader.state_names(),
        },
        "action": {
            "dtype": "float32",
            "shape": (reader.action_dim(),),
            "names": reader.action_names(),
        },
    }

    for field in reader.state_fields:
        features[f"observation.{field}"] = {
            "dtype": "float32",
            "shape": (reader.field_dim(field),),
            "names": None,
        }
    for field in reader.action_fields:
        features[f"action.{field}"] = {
            "dtype": "float32",
            "shape": (reader.field_dim(field),),
            "names": None,
        }
    for camera in reader.cameras:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": reader.camera_shape(camera),
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.fps": int(fps),
                "video.codec": "mp4v",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
        features[f"observation.images.{camera}.timestamp"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        }
        features[f"observation.images.{camera}.is_valid"] = {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        }
    return features


def inspect_episode(
    input_path: Path,
    cameras: Sequence[str],
    state_fields: Sequence[str],
    action_fields: Sequence[str],
) -> None:
    """打印首个 episode 的关键 schema，用于转换前人工检查。"""
    _require_h5_runtime()
    first_path = _episode_paths(input_path, max_episodes=1)[0]
    reader = H5EpisodeReader(
        first_path,
        cameras=cameras,
        state_fields=state_fields,
        action_fields=action_fields,
    )
    try:
        print(f"episode_file: {first_path}")
        print(f"frames: {len(reader)}")
        print("root_attrs:", sorted(reader.file.attrs.keys()))
        if "format" in reader.file.attrs:
            print("format:", _decode_attr(reader.file.attrs["format"]))
        keys = (
            "config_yaml",
            "saved_at_us",
            f"{TELEOP_GROUP}/timestamp_us",
            *[_teleop_path(field) for field in state_fields],
            *[_teleop_path(field) for field in action_fields],
        )
        for key in keys:
            if key in reader.file:
                value = reader.file[key]
                if isinstance(value, h5py.Dataset):
                    print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
        for camera in cameras:
            frames = reader.file[f"cameras/{camera}/frames"]
            timestamps = reader.file[f"cameras/{camera}/timestamp_us"]
            print(f"cameras/{camera}/frames: shape={tuple(frames.shape)} dtype={frames.dtype}")
            print(f"cameras/{camera}/timestamp_us: shape={tuple(timestamps.shape)} dtype={timestamps.dtype}")
        print("state_fields:", list(state_fields), "state_dim:", reader.state_dim())
        print("action_fields:", list(action_fields), "action_dim:", reader.action_dim())
    finally:
        reader.close()


def convert_dataset(
    input_path: Path,
    output_dir: Path,
    *,
    fps: int,
    cameras: Sequence[str],
    state_fields: Sequence[str],
    action_fields: Sequence[str],
    use_videos: bool,
    robot_type: str,
    task: str,
    max_episodes: Optional[int],
) -> None:
    """执行完整转换：读取 H5 episode，写出官方 LeRobot v3 数据集。"""
    _require_h5_runtime()
    _require_writer()
    episode_paths = _episode_paths(input_path, max_episodes=max_episodes)
    first_reader = H5EpisodeReader(
        episode_paths[0],
        cameras=cameras,
        state_fields=state_fields,
        action_fields=action_fields,
    )
    try:
        state_dim = first_reader.state_dim()
        action_dim = first_reader.action_dim()
        features = _build_feature_spec(first_reader, fps=fps)
    finally:
        first_reader.close()

    video_keys = [f"observation.images.{camera}" for camera in cameras]
    dataset = LeRobotV3Writer(
        root=str(output_dir),
        fps=fps,
        features=features,
        video_keys=video_keys if use_videos else [],
        robot_type=robot_type,
        image_color_space="rgb",
    )

    print(f"input_path: {input_path}")
    print(f"output_dir: {output_dir}")
    print(f"episodes: {len(episode_paths)}")
    print(f"cameras: {list(cameras)}")
    print(f"state_fields: {list(state_fields)} state_dim={state_dim}")
    print(f"action_fields: {list(action_fields)} action_dim={action_dim}")

    total_frames = 0
    start = time.perf_counter()
    print(
        f"{_progress_bar(0, len(episode_paths))} 0/{len(episode_paths)} "
        "0.0% total_elapsed=0.000s",
        flush=True,
    )
    for ep_idx, episode_path in enumerate(episode_paths, start=1):
        ep_start = time.perf_counter()
        reader = H5EpisodeReader(
            episode_path,
            cameras=cameras,
            state_fields=state_fields,
            action_fields=action_fields,
        )
        try:
            for frame_idx in range(len(reader)):
                dataset.add_frame(reader.frame(frame_idx))
            dataset.save_episode(task=task)
            total_frames += len(reader)
            total_elapsed = time.perf_counter() - start
            progress = (ep_idx / len(episode_paths)) * 100 if episode_paths else 100.0
            print(
                f"{_progress_bar(ep_idx, len(episode_paths))} "
                f"{ep_idx}/{len(episode_paths)} {progress:.1f}% "
                f"[{episode_path.name}] "
                f"episode_elapsed={_format_seconds(time.perf_counter() - ep_start)} "
                f"total_elapsed={_format_seconds(total_elapsed)} "
                f"frames={len(reader)}",
                flush=True,
            )
        finally:
            reader.close()

    dataset.finalize()
    print(f"Done. episodes={len(episode_paths)} frames={total_frames}")
    print(f"LeRobot v3 output: {output_dir}")


def build_argparser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Convert wipe_board episode_*.h5 files to the official LeRobot v3 format."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("/home/hirol/code/data/train_episode/wipe_board"),
        help="Input episode_*.h5 file or directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output LeRobot v3 dataset directory path. Required unless --inspect-only is used.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print the first episode schema and exit.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Dataset FPS written into meta/info.json.")
    parser.add_argument("--cameras", type=str, default=None, help="Comma-separated camera names.")
    parser.add_argument(
        "--state-fields",
        type=str,
        default=None,
        help="Comma-separated teleop fields for observation.state.",
    )
    parser.add_argument(
        "--action-fields",
        type=str,
        default=None,
        help="Comma-separated teleop fields for action.",
    )
    parser.add_argument("--robot-type", type=str, default="fr3", help="robot_type written into meta/info.json.")
    parser.add_argument("--task", type=str, default="wipe_board", help="Task text written into metadata.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional limit for quick conversion tests.")
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Disable MP4 packing and keep RGB frames as parquet values.",
    )
    return parser


def main() -> None:
    """解析 CLI 参数，并根据模式执行 inspect 或 convert。"""
    args = build_argparser().parse_args()
    cameras = _parse_csv(args.cameras, DEFAULT_CAMERAS)
    state_fields = _parse_csv(args.state_fields, DEFAULT_STATE_FIELDS)
    action_fields = _parse_csv(args.action_fields, DEFAULT_ACTION_FIELDS)
    if args.inspect_only:
        inspect_episode(args.input_path, cameras, state_fields, action_fields)
        return
    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --inspect-only is used.")
    start = time.perf_counter()
    convert_dataset(
        input_path=args.input_path,
        output_dir=args.output_dir,
        fps=args.fps,
        cameras=cameras,
        state_fields=state_fields,
        action_fields=action_fields,
        use_videos=not args.no_videos,
        robot_type=args.robot_type,
        task=args.task,
        max_episodes=args.max_episodes,
    )
    print(f"elapsed={_format_seconds(time.perf_counter() - start)}")


if __name__ == "__main__":
    main()
