from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CAMERA_MAP = {
    "ee_cam_color": "wrist",
    "third_person_cam_color": "side_1",
}


def load_h5py() -> Any:
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: h5py. Run this script in an environment with h5py.") from exc
    return h5py


def load_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: opencv-python/cv2. Install it before converting images.") from exc
    return cv2


def parse_camera_map(raw_items: list[str] | None) -> dict[str, str]:
    camera_map = dict(DEFAULT_CAMERA_MAP)
    if not raw_items:
        return camera_map

    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid --camera-map item {item!r}, expected source_key=target_name")
        source_key, target_name = item.split("=", 1)
        source_key = source_key.strip()
        target_name = target_name.strip()
        if not source_key or not target_name:
            raise ValueError(f"Invalid --camera-map item {item!r}")
        camera_map[source_key] = target_name
    return camera_map


def discover_episodes(input_path: Path, max_episodes: int | None) -> list[Path]:
    input_path = input_path.expanduser()
    if input_path.is_file():
        if input_path.name != "data.json":
            raise ValueError(f"Input file must be data.json, got {input_path}")
        episodes = [input_path.parent]
    else:
        episodes = sorted(path.parent for path in input_path.glob("episode_*/data.json"))
        if not episodes and (input_path / "data.json").exists():
            episodes = [input_path]

    if not episodes:
        raise FileNotFoundError(f"No EpisodeWriter episodes found under {input_path}")
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    return episodes


def timestamp_to_us(value: Any, *, base_wall_us: int | None = None, base_mono_s: float | None = None) -> int:
    if value is None:
        return 0
    ts = float(value)
    if base_wall_us is not None and base_mono_s is not None:
        return int(round(base_wall_us + (ts - base_mono_s) * 1_000_000.0))
    return int(round(ts * 1_000_000.0))


def timestamp_mapping_to_us(
    mapping: dict[str, Any],
    default: Any,
    *,
    base_wall_us: int | None = None,
    base_mono_s: float | None = None,
) -> int:
    if mapping.get("timestamp_us") is not None:
        return int(round(float(mapping["timestamp_us"])))
    if mapping.get("timestamp_ms") is not None:
        return int(round(float(mapping["timestamp_ms"]) * 1_000.0))
    if mapping.get("timestamp_ns") is not None:
        return int(round(float(mapping["timestamp_ns"]) / 1_000.0))
    if mapping.get("timestamp_s") is not None:
        return timestamp_to_us(mapping["timestamp_s"], base_wall_us=base_wall_us, base_mono_s=base_mono_s)
    return timestamp_to_us(
        mapping.get("time_stamp", mapping.get("timestamp", default)),
        base_wall_us=base_wall_us,
        base_mono_s=base_mono_s,
    )


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def pose7_to_matrix(pose: Any) -> np.ndarray:
    pose_arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    mat = np.eye(4, dtype=np.float64)
    if pose_arr.size < 7:
        return mat
    mat[:3, 3] = pose_arr[:3]
    mat[:3, :3] = quat_xyzw_to_matrix(pose_arr[3:7])
    return mat


def scalar_to_int8(value: Any) -> np.int8:
    arr = np.asarray(value)
    if arr.shape != ():
        arr = arr.reshape(-1)[0]
    val = float(arr)
    if val > 1.0:
        val = val / 100.0
    val = max(0.0, min(1.0, val))
    return np.int8(round(val))


def get_single(item: dict[str, Any], key: str) -> dict[str, Any]:
    value = item.get(key) or {}
    if "single" in value:
        return value["single"] or {}
    if value:
        first_key = sorted(value.keys())[0]
        return value[first_key] or {}
    return {}


def first_present(mapping: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def require_json_list(episode_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    json_path = episode_dir / "data.json"
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError(f"{json_path} does not contain non-empty data list")
    return payload, data


def read_image(cv2: Any, image_path: Path, output_size: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    if output_size:
        width, height = output_size
        if image.shape[1] != width or image.shape[0] != height:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image.astype(np.uint8, copy=False)


def collect_camera_frames(
    *,
    episode_dir: Path,
    data: list[dict[str, Any]],
    camera_map: dict[str, str],
    output_size: tuple[int, int],
    timestamp_base: tuple[int, float] | None,
) -> dict[str, dict[str, np.ndarray]]:
    cv2 = load_cv2()
    frames_by_target: dict[str, list[np.ndarray]] = {}
    ts_by_target: dict[str, list[int]] = {}
    base_wall_us, base_mono_s = timestamp_base if timestamp_base else (None, None)

    for item in data:
        colors = item.get("colors") or {}
        for source_key, color_info in colors.items():
            target_name = camera_map.get(source_key)
            if target_name is None:
                continue
            if not isinstance(color_info, dict):
                continue
            rel_path = color_info.get("path")
            if not rel_path:
                continue
            image_path = episode_dir / rel_path
            frames_by_target.setdefault(target_name, []).append(read_image(cv2, image_path, output_size))
            ts_by_target.setdefault(target_name, []).append(
                timestamp_mapping_to_us(
                    color_info,
                    0,
                    base_wall_us=base_wall_us,
                    base_mono_s=base_mono_s,
                )
            )

    cameras: dict[str, dict[str, np.ndarray]] = {}
    for target_name in sorted(frames_by_target.keys()):
        cameras[target_name] = {
            "frames": np.stack(frames_by_target[target_name], axis=0).astype(np.uint8, copy=False),
            "timestamp_us": np.asarray(ts_by_target[target_name], dtype=np.int64),
        }
    return cameras


def collect_teleop_arrays(
    data: list[dict[str, Any]],
    timestamp_base: tuple[int, float] | None,
) -> dict[str, np.ndarray]:
    base_wall_us, base_mono_s = timestamp_base if timestamp_base else (None, None)
    timestamp_us = []
    q_follower = []
    dq_follower = []
    ddq_follower = []
    tau_ext = []
    q_cmd = []
    ee_pose = []
    cmd_ee_pose = []
    gripper_state = []
    gripper_action = []

    for item in data:
        joint = get_single(item, "joint_states")
        ee = get_single(item, "ee_states")
        tool = get_single(item, "tools")
        action = get_single(item, "actions")
        action_joint = action.get("joint") or {}
        action_ee = action.get("ee") or {}
        action_tool = action.get("tool") or {}

        master_default = ee.get("time_stamp", ee.get("timestamp", item.get("idx", 0)))
        timestamp_us.append(
            timestamp_mapping_to_us(
                joint,
                master_default,
                base_wall_us=base_wall_us,
                base_mono_s=base_mono_s,
            )
        )
        q = np.asarray(joint.get("position", np.zeros(7)), dtype=np.float64).reshape(-1)
        dq = np.asarray(joint.get("velocity", np.zeros_like(q)), dtype=np.float64).reshape(-1)
        ddq = np.asarray(
            first_present(
                joint,
                (
                    "acceleration",
                    "accelerations",
                    "ddq",
                    "joint_acceleration",
                    "joint_accelerations",
                ),
                np.zeros_like(q),
            ),
            dtype=np.float64,
        ).reshape(-1)
        tau = np.asarray(joint.get("torque", np.zeros_like(q)), dtype=np.float64).reshape(-1)
        q_action = np.asarray(action_joint.get("position", q), dtype=np.float64).reshape(-1)

        q_follower.append(q)
        dq_follower.append(dq)
        ddq_follower.append(ddq)
        tau_ext.append(tau)
        q_cmd.append(q_action)
        ee_pose.append(pose7_to_matrix(ee.get("pose", np.zeros(7))))
        cmd_ee_pose.append(pose7_to_matrix(action_ee.get("pose", ee.get("pose", np.zeros(7)))))
        gripper_state.append(scalar_to_int8(tool.get("position", 0)))
        gripper_action.append(scalar_to_int8(action_tool.get("position", tool.get("position", 0))))

    return {
        "timestamp_us": np.asarray(timestamp_us, dtype=np.int64),
        "q_follower": np.asarray(q_follower, dtype=np.float64),
        "dq_follower": np.asarray(dq_follower, dtype=np.float64),
        "ddq_follower": np.asarray(ddq_follower, dtype=np.float64),
        "tau_ext": np.asarray(tau_ext, dtype=np.float64),
        "q_cmd": np.asarray(q_cmd, dtype=np.float64),
        "q_leader": np.asarray(q_cmd, dtype=np.float64),
        "dq_leader": np.zeros_like(np.asarray(q_cmd, dtype=np.float64)),
        "ddq_leader": np.zeros_like(np.asarray(q_cmd, dtype=np.float64)),
        "tau_leader": np.zeros_like(np.asarray(q_cmd, dtype=np.float64)),
        "ee_pose": np.asarray(ee_pose, dtype=np.float64),
        "cmd_ee_pose": np.asarray(cmd_ee_pose, dtype=np.float64),
        "gripper_state": np.asarray(gripper_state, dtype=np.int8),
        "gripper_action": np.asarray(gripper_action, dtype=np.int8),
    }


def collect_ati_json_arrays(
    ati_path: Path,
    timestamp_base: tuple[int, float] | None,
) -> dict[str, np.ndarray]:
    base_wall_us, base_mono_s = timestamp_base if timestamp_base else (None, None)
    with ati_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{ati_path} must contain non-empty data list")

    wrench = []
    ts = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{ati_path} data[{idx}] must be a JSON object")
        if "ft_data" not in row:
            raise KeyError(f"{ati_path} data[{idx}] missing required key 'ft_data'")
        if "time_stamp" not in row:
            raise KeyError(f"{ati_path} data[{idx}] missing required key 'time_stamp'")

        wrench_row = np.asarray(row["ft_data"], dtype=np.float64).reshape(-1)
        if wrench_row.size != 6:
            raise ValueError(f"{ati_path} data[{idx}].ft_data must have length 6, got {wrench_row.size}")
        wrench.append(wrench_row)
        ts.append(
            timestamp_to_us(
                row["time_stamp"],
                base_wall_us=base_wall_us,
                base_mono_s=base_mono_s,
            )
        )

    return build_force_dict(np.asarray(wrench, dtype=np.float64), np.asarray(ts, dtype=np.int64))


def empty_force_dict() -> dict[str, np.ndarray]:
    return build_force_dict(np.zeros((0, 6), dtype=np.float64), np.zeros((0,), dtype=np.int64))


def collect_force_arrays(
    data: list[dict[str, Any]],
    episode_dir: Path,
    timestamp_base: tuple[int, float] | None,
    *,
    prefer_async_ft: bool,
    ati_json_name: str,
) -> dict[str, np.ndarray]:
    base_wall_us, base_mono_s = timestamp_base if timestamp_base else (None, None)

    if prefer_async_ft:
        async_path = episode_dir / ati_json_name
        if async_path.exists():
            return collect_ati_json_arrays(async_path, timestamp_base)
        return empty_force_dict()

    wrench_rows = []
    ts_rows = []
    for item in data:
        ee = get_single(item, "ee_states")
        if "ft" not in ee:
            continue
        wrench_rows.append(np.asarray(ee["ft"], dtype=np.float64).reshape(6))
        ts_rows.append(timestamp_to_us(ee.get("ft_time_stamp"), base_wall_us=base_wall_us, base_mono_s=base_mono_s))

    if not wrench_rows:
        return empty_force_dict()
    return build_force_dict(np.stack(wrench_rows, axis=0), np.asarray(ts_rows, dtype=np.int64))


def build_force_dict(wrench_filtered: np.ndarray, timestamp_us: np.ndarray) -> dict[str, np.ndarray]:
    if timestamp_us.size:
        relative_time_s = (timestamp_us - timestamp_us[0]).astype(np.float64) * 1e-6
    else:
        relative_time_s = np.zeros((0,), dtype=np.float64)
    return {
        "timestamp_us": timestamp_us.astype(np.int64, copy=False),
        "relative_time_s": relative_time_s,
        "wrench_filtered": wrench_filtered.astype(np.float64, copy=False),
        "wrench_raw": wrench_filtered.astype(np.float64, copy=True),
    }


def build_config_yaml(payload: dict[str, Any], args: argparse.Namespace, source_episode: Path) -> str:
    text = {
        "source_episode": str(source_episode),
        "source_format": "HIROL EpisodeWriter data.json",
        "converter": Path(__file__).name,
        "image_output_size": [args.image_width, args.image_height],
        "task": payload.get("text", {}),
        "camera_map": parse_camera_map(args.camera_map),
    }
    try:
        import yaml  # type: ignore

        return yaml.safe_dump(text, sort_keys=False, allow_unicode=True)
    except ModuleNotFoundError:
        return json.dumps(text, ensure_ascii=False, indent=2)


def output_name_for_episode(episode_dir: Path) -> str:
    suffix = episode_dir.name
    if suffix.startswith("episode_"):
        episode_id = suffix.split("_", 1)[1]
    else:
        episode_id = suffix
    return f"episode_{episode_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"


def write_h5(
    *,
    h5_path: Path,
    payload: dict[str, Any],
    cameras: dict[str, dict[str, np.ndarray]],
    teleop: dict[str, np.ndarray],
    force: dict[str, np.ndarray],
    config_yaml: str,
    overwrite: bool,
) -> None:
    h5py = load_h5py()
    if h5_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {h5_path}. Use --overwrite to replace it.")
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "w") as h5:
        h5.attrs["format"] = "factr_multimodal_episode/v2"
        h5.attrs["saved_at_us"] = int(time.time_ns() // 1000)

        string_dtype = h5py.string_dtype(encoding="utf-8")
        h5.create_dataset("config_yaml", data=config_yaml, dtype=string_dtype)

        cameras_group = h5.create_group("cameras")
        for cam_name, cam_data in cameras.items():
            cam_group = cameras_group.create_group(cam_name)
            cam_group.create_dataset("frames", data=cam_data["frames"], dtype=np.uint8)
            cam_group.create_dataset("timestamp_us", data=cam_data["timestamp_us"], dtype=np.int64)

        force_group = h5.create_group("force_sensor")
        for key, value in force.items():
            force_group.create_dataset(key, data=value)

        teleop_group = h5.create_group("teleop")
        for key, value in teleop.items():
            if value.ndim >= 2 and value.size:
                teleop_group.create_dataset(key, data=value, compression="gzip", compression_opts=1)
            else:
                teleop_group.create_dataset(key, data=value)


def convert_episode(episode_dir: Path, output_dir: Path, args: argparse.Namespace) -> Path:
    payload, data = require_json_list(episode_dir)
    camera_map = parse_camera_map(args.camera_map)
    output_size = (int(args.image_width), int(args.image_height))

    timestamp_base = None
    if args.epoch_timestamps:
        first_item = data[0]
        first_joint = get_single(first_item, "joint_states")
        first_ee = get_single(first_item, "ee_states")
        base_mono_s = float(first_joint.get("time_stamp", first_ee.get("time_stamp", 0.0)))
        timestamp_base = (int(time.time_ns() // 1000), base_mono_s)

    if getattr(args, "skip_cameras", False):
        cameras = {}
    else:
        cameras = collect_camera_frames(
            episode_dir=episode_dir,
            data=data,
            camera_map=camera_map,
            output_size=output_size,
            timestamp_base=timestamp_base,
        )
    teleop = collect_teleop_arrays(data, timestamp_base)
    force = collect_force_arrays(
        data,
        episode_dir,
        timestamp_base,
        prefer_async_ft=args.prefer_async_ft,
        ati_json_name=args.ati_json_name,
    )

    if args.require_cameras and not cameras:
        raise ValueError(f"No mapped camera frames found in {episode_dir}")
    if args.require_ft and force["wrench_filtered"].shape[0] == 0:
        raise ValueError(f"No FT data found in {episode_dir}")

    out_name = args.output_name or output_name_for_episode(episode_dir)
    if getattr(args, "num_episodes", 1) > 1 and args.output_name:
        stem = Path(args.output_name).stem
        out_name = f"{stem}_{episode_dir.name}.h5"
    h5_path = output_dir / out_name
    write_h5(
        h5_path=h5_path,
        payload=payload,
        cameras=cameras,
        teleop=teleop,
        force=force,
        config_yaml=build_config_yaml(payload, args, episode_dir),
        overwrite=args.overwrite,
    )
    return h5_path


def inspect_h5(h5_path: Path) -> None:
    h5py = load_h5py()
    with h5py.File(h5_path, "r") as h5:
        print(f"\n# {h5_path}")
        print(f"attrs={dict(h5.attrs)}")

        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                print(f"{name}: shape={obj.shape} dtype={obj.dtype}")
            else:
                print(f"{name}/")

        h5.visititems(visit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HIROL EpisodeWriter episodes to factr_multimodal_episode/v2 H5 files."
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Episode dir, data.json, or task dir.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output directory for .h5 files.")
    parser.add_argument("--output-name", type=str, default=None, help="Output file name for a single episode.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit number of episodes.")
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=192)
    parser.add_argument(
        "--camera-map",
        action="append",
        default=None,
        help="Map source color key to target camera name, e.g. ee_cam_color=wrist. Can be repeated.",
    )
    parser.add_argument("--use-ati-json", dest="prefer_async_ft", action="store_true", help="Use the explicit ATI JSON file beside data.json.")
    parser.add_argument("--prefer-async-ft", dest="prefer_async_ft", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--ati-json-name",
        default="ee_ft_data.json",
        help="Exact ATI JSON filename beside data.json. Expected schema: data[*].ft_data + data[*].time_stamp.",
    )
    parser.add_argument("--epoch-timestamps", action="store_true", help="Convert perf_counter timestamps to wall-clock-like us.")
    parser.add_argument("--require-cameras", action="store_true", help="Fail if no mapped cameras are found.")
    parser.add_argument("--require-ft", action="store_true", help="Fail if no FT data is found.")
    parser.add_argument("--skip-cameras", action="store_true", help="Do not read/write camera frames.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--inspect", action="store_true", help="Print converted H5 structure.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = discover_episodes(args.input, args.max_episodes)
    args.num_episodes = len(episodes)
    output_dir = args.output.expanduser()
    print(f"Found {len(episodes)} episode(s).")

    written = []
    for episode_dir in episodes:
        h5_path = convert_episode(episode_dir, output_dir, args)
        written.append(h5_path)
        print(f"Wrote {h5_path}")
        if args.inspect:
            inspect_h5(h5_path)

    print(f"Done. Converted {len(written)} episode(s) into {output_dir}")


if __name__ == "__main__":
    main()
