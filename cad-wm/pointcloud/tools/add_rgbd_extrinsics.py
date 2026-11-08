from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DATASET = Path("data/train_episode/Ft_test_data")
DEFAULT_CALIBRATION = Path("pointcloud/calibration_v2/camera_extrinsics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Augment a HIROL RGB-D episode JSON with camera intrinsics, depth units, "
            "and per-frame camera extrinsics in the robot base frame."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET,
        help="Dataset folder containing data.json, colors/, and depths/.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help="JSON produced by pointcloud/calibration_v2/export_camera_extrinsics.py.",
    )
    parser.add_argument(
        "--input-json",
        default="data.json",
        help="Input episode JSON name inside --dataset-dir.",
    )
    parser.add_argument(
        "--output-json",
        default="data_with_rgbd_extrinsics.json",
        help="Output episode JSON name inside --dataset-dir unless an absolute path is provided.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input JSON after creating a .bak backup. By default a new JSON is written.",
    )
    parser.add_argument(
        "--robot-key",
        default="single",
        help="Robot key under ee_states used for the wrist camera pose.",
    )
    parser.add_argument(
        "--ee-pose-order",
        default="xyzw",
        choices=("xyzw", "wxyz"),
        help="Quaternion order in ee_states.<robot-key>.pose after xyz. HIROL datasets use xyzw.",
    )
    parser.add_argument(
        "--depth-scale-m-per-unit",
        type=float,
        default=0.001,
        help="Depth PNG scale. Existing HIROL spec stores 16-bit depth in millimeters.",
    )
    parser.add_argument(
        "--depth-aligned-to",
        default="color",
        help="Frame that depth pixels are aligned to. Use 'color' for RealSense rs.align(rs.stream.color).",
    )
    parser.add_argument(
        "--no-file-check",
        action="store_true",
        help="Do not check that referenced color/depth files exist.",
    )
    return parser.parse_args()


def resolve_existing_path(path: Path) -> Path:
    if path.exists():
        return path
    if not path.is_absolute():
        parent_candidate = Path("..") / path
        if parent_candidate.exists():
            return parent_candidate
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4), encoding="utf-8")


def matmul4(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def invert_transform(transform: list[list[float]]) -> list[list[float]]:
    rotation = [[float(transform[i][j]) for j in range(3)] for i in range(3)]
    translation = [float(transform[i][3]) for i in range(3)]
    rot_t = [[rotation[j][i] for j in range(3)] for i in range(3)]
    inv_translation = [-sum(rot_t[i][j] * translation[j] for j in range(3)) for i in range(3)]
    return [
        [rot_t[0][0], rot_t[0][1], rot_t[0][2], inv_translation[0]],
        [rot_t[1][0], rot_t[1][1], rot_t[1][2], inv_translation[1]],
        [rot_t[2][0], rot_t[2][1], rot_t[2][2], inv_translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def normalize_quaternion(q: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in q))
    if norm <= 0.0:
        raise ValueError("zero-norm quaternion in end-effector pose")
    return [float(value) / norm for value in q]


def quat_xyzw_to_matrix(q: list[float]) -> list[list[float]]:
    qx, qy, qz, qw = normalize_quaternion(q)
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def pose7_to_transform(pose: list[float], quat_order: str) -> list[list[float]]:
    if len(pose) != 7:
        raise ValueError(f"end-effector pose must have 7 values, got {len(pose)}")
    xyz = [float(value) for value in pose[:3]]
    quat = [float(value) for value in pose[3:7]]
    if quat_order == "wxyz":
        qw, qx, qy, qz = quat
        quat = [qx, qy, qz, qw]

    rotation = quat_xyzw_to_matrix(quat)
    return [
        [rotation[0][0], rotation[0][1], rotation[0][2], xyz[0]],
        [rotation[1][0], rotation[1][1], rotation[1][2], xyz[1]],
        [rotation[2][0], rotation[2][1], rotation[2][2], xyz[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def camera_color_key(camera_name: str) -> str:
    return f"{camera_name}_color"


def camera_depth_key(camera_name: str) -> str:
    return f"{camera_name}_depth"


def get_required_transform(camera: dict[str, Any], name: str) -> list[list[float]]:
    transform = camera.get(name)
    if transform is None:
        raise KeyError(f"camera {camera.get('camera_name')} is missing {name}")
    if len(transform) != 4 or any(len(row) != 4 for row in transform):
        raise ValueError(f"camera {camera.get('camera_name')} {name} must be a 4x4 matrix")
    return [[float(value) for value in row] for row in transform]


def build_camera_static_info(
    calibration: dict[str, Any],
    depth_scale_m_per_unit: float,
    depth_aligned_to: str,
) -> dict[str, Any]:
    cameras = {}
    for camera_name, camera in sorted(calibration.get("cameras", {}).items()):
        color_key = camera_color_key(camera_name)
        depth_key = camera_depth_key(camera_name)
        transform_name = camera["transform_name"]
        static_transform = get_required_transform(camera, transform_name)
        cameras[camera_name] = {
            "color_key": color_key,
            "depth_key": depth_key,
            "intrinsics": camera.get("intrinsics", {}),
            "depth_scale_m_per_unit": depth_scale_m_per_unit,
            "depth_unit": "millimeter" if abs(depth_scale_m_per_unit - 0.001) < 1e-12 else "raw_units",
            "depth_aligned_to": depth_aligned_to,
            "calibration_type": camera["calibration_type"],
            "static_transform_name": transform_name,
            transform_name: static_transform,
            "source_json": camera.get("source_json"),
        }
    return cameras


def camera_t_base_camera(
    camera_name: str,
    camera: dict[str, Any],
    t_base_ee: list[list[float]],
) -> list[list[float]]:
    cali_type = camera["calibration_type"]
    if cali_type == "eye_to_hand":
        return get_required_transform(camera, "T_base_camera")
    if cali_type == "eye_in_hand":
        return matmul4(t_base_ee, get_required_transform(camera, "T_ee_camera"))
    raise ValueError(f"Unsupported calibration_type for {camera_name}: {cali_type!r}")


def check_media_file(dataset_dir: Path, rel_path: str, warnings: list[str], frame_idx: int, key: str) -> None:
    if not (dataset_dir / rel_path).exists():
        warnings.append(f"frame {frame_idx} references missing {key}: {rel_path}")


def augment_frame(
    item: dict[str, Any],
    dataset_dir: Path,
    calibration_cameras: dict[str, Any],
    robot_key: str,
    ee_pose_order: str,
    check_files: bool,
    warnings: list[str],
) -> None:
    frame_idx = int(item.get("idx", -1))
    ee_state = ((item.get("ee_states") or {}).get(robot_key) or {})
    if "pose" not in ee_state:
        raise KeyError(f"frame {frame_idx} is missing ee_states.{robot_key}.pose")

    t_base_ee = pose7_to_transform(ee_state["pose"], ee_pose_order)
    camera_entries = {}
    for camera_name, camera in sorted(calibration_cameras.items()):
        color_key = camera_color_key(camera_name)
        depth_key = camera_depth_key(camera_name)
        color_info = (item.get("colors") or {}).get(color_key)
        depth_info = (item.get("depths") or {}).get(depth_key)
        if color_info is None:
            warnings.append(f"frame {frame_idx} is missing colors.{color_key}")
            continue
        if depth_info is None:
            warnings.append(f"frame {frame_idx} is missing depths.{depth_key}")
            continue

        if check_files:
            check_media_file(dataset_dir, color_info["path"], warnings, frame_idx, color_key)
            check_media_file(dataset_dir, depth_info["path"], warnings, frame_idx, depth_key)

        t_base_camera = camera_t_base_camera(camera_name, camera, t_base_ee)
        camera_entries[camera_name] = {
            "color_key": color_key,
            "depth_key": depth_key,
            "color_path": color_info.get("path"),
            "depth_path": depth_info.get("path"),
            "color_time_stamp": color_info.get("time_stamp"),
            "depth_time_stamp": depth_info.get("time_stamp"),
            "T_base_camera": t_base_camera,
            "T_camera_base": invert_transform(t_base_camera),
        }

    item["rgbd_extrinsics"] = {
        "world_frame": "robot_base",
        "robot_key": robot_key,
        "ee_pose_order": ee_pose_order,
        "T_base_ee": t_base_ee,
        "cameras": camera_entries,
    }


def warn_on_resolution_mismatch(
    episode: dict[str, Any],
    camera_static_info: dict[str, Any],
    warnings: list[str],
) -> None:
    image_info = ((episode.get("info") or {}).get("image") or {})
    width = image_info.get("width")
    height = image_info.get("height")
    if width is None or height is None:
        return

    for camera_name, camera in camera_static_info.items():
        intr = camera.get("intrinsics") or {}
        intr_width = intr.get("width")
        intr_height = intr.get("height")
        if intr_width is None or intr_height is None:
            warnings.append(f"{camera_name} calibration intrinsics do not include width/height")
            continue
        if int(intr_width) != int(width) or int(intr_height) != int(height):
            warnings.append(
                f"{camera_name} calibration intrinsics are {intr_width}x{intr_height}, "
                f"but dataset images are {width}x{height}"
            )


def output_path_for(dataset_dir: Path, input_path: Path, output_json: str, in_place: bool) -> Path:
    if in_place:
        return input_path
    candidate = Path(output_json)
    if candidate.is_absolute():
        return candidate
    return dataset_dir / candidate


def backup_for_in_place(path: Path) -> Path:
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def main() -> None:
    args = parse_args()
    dataset_dir = resolve_existing_path(args.dataset_dir)
    calibration_path = resolve_existing_path(args.calibration)
    input_path = dataset_dir / args.input_json
    if not input_path.exists():
        raise FileNotFoundError(f"Input episode JSON not found: {input_path}")
    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Calibration JSON not found: {calibration_path}. "
            "Run pointcloud/calibration_v2/export_camera_extrinsics.py first."
        )

    episode = read_json(input_path)
    calibration = read_json(calibration_path)
    calibration_cameras = calibration.get("cameras") or {}
    if not calibration_cameras:
        raise ValueError(f"No cameras found in calibration JSON: {calibration_path}")

    warnings: list[str] = []
    camera_static_info = build_camera_static_info(
        calibration,
        depth_scale_m_per_unit=args.depth_scale_m_per_unit,
        depth_aligned_to=args.depth_aligned_to,
    )
    warn_on_resolution_mismatch(episode, camera_static_info, warnings)

    data = episode.get("data")
    if not isinstance(data, list):
        raise ValueError(f"{input_path} does not contain a list at key 'data'")

    for item in data:
        augment_frame(
            item=item,
            dataset_dir=dataset_dir,
            calibration_cameras=calibration_cameras,
            robot_key=args.robot_key,
            ee_pose_order=args.ee_pose_order,
            check_files=not args.no_file_check,
            warnings=warnings,
        )

    episode["pointcloud_reconstruction"] = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "created_by": "pointcloud/tools/add_rgbd_extrinsics.py",
        "source_episode_json": input_path.as_posix(),
        "source_calibration_json": calibration_path.as_posix(),
        "world_frame": "robot_base",
        "camera_frame": "RealSense color optical frame: x right, y down, z forward",
        "transform_direction": "T_base_camera maps homogeneous points from camera frame into robot_base",
        "depth_scale_m_per_unit": args.depth_scale_m_per_unit,
        "depth_aligned_to": args.depth_aligned_to,
        "robot_key": args.robot_key,
        "ee_pose_order": args.ee_pose_order,
        "cameras": camera_static_info,
        "warnings": warnings,
    }

    output_path = output_path_for(dataset_dir, input_path, args.output_json, args.in_place)
    if args.in_place:
        backup_path = backup_for_in_place(input_path)
        print(f"Created backup: {backup_path}")

    write_json(output_path, episode)
    print(f"Saved augmented episode JSON: {output_path}")
    print(f"Frames: {len(data)}")
    print(f"Cameras: {', '.join(sorted(camera_static_info))}")
    if warnings:
        print(f"Warnings: {len(warnings)}")
        for warning in warnings[:10]:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
