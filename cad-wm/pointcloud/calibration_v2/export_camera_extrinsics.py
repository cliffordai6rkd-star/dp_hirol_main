from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve pointcloud/calibration_v2 AprilTag hand-eye datasets and export a single "
            "camera-extrinsics JSON for RGB-D point-cloud reconstruction."
        )
    )
    parser.add_argument(
        "--ee-calibration",
        type=Path,
        default=Path("pointcloud/calibration_v2/ee_cam_d435/data_ee_cam.json"),
        help="Eye-in-hand calibration JSON for the wrist camera.",
    )
    parser.add_argument(
        "--third-person-calibration",
        type=Path,
        default=Path("pointcloud/calibration_v2/third_person_cam_d455/data_third_person_cam.json"),
        help="Eye-to-hand calibration JSON for the fixed third-person camera.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pointcloud/calibration_v2/camera_extrinsics.json"),
        help="Output JSON path consumed by pointcloud/tools/add_rgbd_extrinsics.py.",
    )
    parser.add_argument(
        "--method",
        default="tsai",
        choices=("tsai", "park", "horaud", "andreff", "daniilidis"),
        help="OpenCV calibrateHandEye method.",
    )
    parser.add_argument(
        "--write-npy",
        action="store_true",
        help="Also write T_ee_camera.npy / T_base_camera.npy next to each source JSON.",
    )
    return parser.parse_args()


def load_numpy_cv2():
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency. Install the project vision extras with "
            "`python -m pip install -e \".[vision]\"`."
        ) from exc
    return np, cv2


def opencv_hand_eye_method(cv2, name: str) -> int:
    methods = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    return methods[name]


def normalize_cali_type(cali_type: str) -> str:
    normalized = str(cali_type).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in {"eye_in_hand", "eye_to_hand"}:
        raise ValueError(f"Unsupported calibration type: {cali_type!r}")
    return normalized


def load_calibration_dataset(path: Path, np) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    samples = raw.get("data") or []
    if len(samples) < 3:
        raise ValueError(f"{path} has {len(samples)} samples; at least 3 are required")

    robot_poses = []
    tag_poses = []
    for idx, sample in enumerate(samples):
        robot_pose = np.asarray(sample["matrix"], dtype=np.float64)
        tag_pose = np.asarray(sample["homogeneous_matrix"], dtype=np.float64)
        if robot_pose.shape != (4, 4):
            raise ValueError(f"{path} sample {idx} matrix shape is {robot_pose.shape}, expected (4, 4)")
        if tag_pose.shape != (4, 4):
            raise ValueError(
                f"{path} sample {idx} homogeneous_matrix shape is {tag_pose.shape}, expected (4, 4)"
            )
        robot_poses.append(robot_pose)
        tag_poses.append(tag_pose)

    metadata = raw.get("metadata", {}) or {}
    cali_type = normalize_cali_type(metadata.get("cali_type") or metadata.get("calibration_type"))
    camera_name = metadata.get("camera") or infer_camera_name(path, cali_type)
    intrinsics = raw.get("intrinsics") or {}

    return {
        "path": path,
        "raw": raw,
        "robot_poses": robot_poses,
        "tag_poses": tag_poses,
        "metadata": metadata,
        "camera_name": camera_name,
        "cali_type": cali_type,
        "intrinsics": intrinsics,
    }


def infer_camera_name(path: Path, cali_type: str) -> str:
    lower = path.as_posix().lower()
    if "third" in lower or "d455" in lower or cali_type == "eye_to_hand":
        return "third_person_cam"
    if "ee" in lower or "wrist" in lower or "d435" in lower or cali_type == "eye_in_hand":
        return "ee_cam"
    return path.stem.replace("data_", "")


def make_transform(np, rotation, translation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert_transform(np, transform):
    inverse = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def solve_eye_in_hand(np, cv2, robot_poses, tag_poses, method: int):
    r_gripper2base = [pose[:3, :3] for pose in robot_poses]
    t_gripper2base = [pose[:3, 3:4] for pose in robot_poses]
    r_target2cam = [pose[:3, :3] for pose in tag_poses]
    t_target2cam = [pose[:3, 3:4] for pose in tag_poses]

    r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        r_gripper2base,
        t_gripper2base,
        r_target2cam,
        t_target2cam,
        method=method,
    )
    return make_transform(np, r_cam2gripper, t_cam2gripper)


def solve_eye_to_hand(np, cv2, robot_poses, tag_poses, method: int):
    # OpenCV calibrateHandEye estimates T_camera_gripper from gripper/base motion.
    # For a fixed external camera, swap the robot frames by feeding T_ee_base as
    # the gripper-to-base sequence; the returned camera-to-"gripper" transform is T_base_camera.
    ee2base = [invert_transform(np, pose) for pose in robot_poses]
    r_gripper2base = [pose[:3, :3] for pose in ee2base]
    t_gripper2base = [pose[:3, 3:4] for pose in ee2base]
    r_target2cam = [pose[:3, :3] for pose in tag_poses]
    t_target2cam = [pose[:3, 3:4] for pose in tag_poses]

    r_cam2base, t_cam2base = cv2.calibrateHandEye(
        r_gripper2base,
        t_gripper2base,
        r_target2cam,
        t_target2cam,
        method=method,
    )
    return make_transform(np, r_cam2base, t_cam2base)


def rotation_error_deg(np, r_a, r_b) -> float:
    r_delta = r_a @ r_b.T
    cos_angle = np.clip((np.trace(r_delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def consistency_stats(np, transforms: list) -> dict[str, Any]:
    reference = transforms[0]
    pos_errors = [float(np.linalg.norm(pose[:3, 3] - reference[:3, 3])) for pose in transforms]
    rot_errors = [rotation_error_deg(np, pose[:3, :3], reference[:3, :3]) for pose in transforms]
    translations = np.asarray([pose[:3, 3] for pose in transforms], dtype=np.float64)
    return {
        "samples": len(transforms),
        "mean_position_error_m": float(np.mean(pos_errors)),
        "max_position_error_m": float(np.max(pos_errors)),
        "mean_position_error_mm": float(np.mean(pos_errors) * 1000.0),
        "max_position_error_mm": float(np.max(pos_errors) * 1000.0),
        "mean_rotation_error_deg": float(np.mean(rot_errors)),
        "max_rotation_error_deg": float(np.max(rot_errors)),
        "translation_mean_m": translations.mean(axis=0).tolist(),
        "translation_std_m": translations.std(axis=0).tolist(),
    }


def verify_eye_in_hand(np, robot_poses, tag_poses, t_ee_camera) -> dict[str, Any]:
    # The AprilTag is fixed in the scene, so T_base_tag should stay constant.
    t_base_tag = [t_base_ee @ t_ee_camera @ t_camera_tag for t_base_ee, t_camera_tag in zip(robot_poses, tag_poses)]
    stats = consistency_stats(np, t_base_tag)
    stats["constant_transform_checked"] = "T_base_tag"
    return stats


def verify_eye_to_hand(np, robot_poses, tag_poses, t_base_camera) -> dict[str, Any]:
    # The AprilTag is rigidly attached to the end-effector, so T_ee_tag should stay constant.
    t_ee_tag = [
        invert_transform(np, t_base_ee) @ t_base_camera @ t_camera_tag
        for t_base_ee, t_camera_tag in zip(robot_poses, tag_poses)
    ]
    stats = consistency_stats(np, t_ee_tag)
    stats["constant_transform_checked"] = "T_ee_tag"
    return stats


def solve_dataset(dataset: dict[str, Any], np, cv2, method: int) -> dict[str, Any]:
    cali_type = dataset["cali_type"]
    robot_poses = dataset["robot_poses"]
    tag_poses = dataset["tag_poses"]

    if cali_type == "eye_in_hand":
        transform_name = "T_ee_camera"
        transform = solve_eye_in_hand(np, cv2, robot_poses, tag_poses, method)
        verification = verify_eye_in_hand(np, robot_poses, tag_poses, transform)
        parent_frame = "end_effector"
    else:
        transform_name = "T_base_camera"
        transform = solve_eye_to_hand(np, cv2, robot_poses, tag_poses, method)
        verification = verify_eye_to_hand(np, robot_poses, tag_poses, transform)
        parent_frame = "robot_base"

    return {
        "camera_name": dataset["camera_name"],
        "calibration_type": cali_type,
        "parent_frame": parent_frame,
        "transform_name": transform_name,
        transform_name: transform.tolist(),
        "intrinsics": dataset["intrinsics"],
        "source_json": dataset["path"].as_posix(),
        "num_samples": len(robot_poses),
        "verification": verification,
    }


def write_npy_outputs(np, datasets: list[dict[str, Any]], results: dict[str, dict[str, Any]]) -> None:
    for dataset in datasets:
        result = results[dataset["camera_name"]]
        transform_name = result["transform_name"]
        output_path = dataset["path"].with_name(f"{transform_name}.npy")
        np.save(output_path, np.asarray(result[transform_name], dtype=np.float64))
        print(f"Saved {transform_name}: {output_path}")


def main() -> None:
    args = parse_args()
    np, cv2 = load_numpy_cv2()
    method = opencv_hand_eye_method(cv2, args.method)

    datasets = [
        load_calibration_dataset(args.ee_calibration, np),
        load_calibration_dataset(args.third_person_calibration, np),
    ]

    cameras = {}
    for dataset in datasets:
        result = solve_dataset(dataset, np, cv2, method)
        camera_name = result["camera_name"]
        if camera_name in cameras:
            raise ValueError(f"Duplicate camera name in calibration inputs: {camera_name}")
        cameras[camera_name] = result
        print(
            f"{camera_name}: {result['calibration_type']} -> {result['transform_name']} | "
            f"mean={result['verification']['mean_position_error_mm']:.2f} mm, "
            f"max={result['verification']['max_position_error_mm']:.2f} mm"
        )

    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "created_by": "pointcloud/calibration_v2/export_camera_extrinsics.py",
        "opencv_hand_eye_method": args.method,
        "frame_convention": {
            "robot_base": "Franka/base frame used by calibration robot pose matrices",
            "end_effector": "Franka end-effector frame used by Ft_test_data ee_states.single.pose",
            "camera": "RealSense color optical frame: x right, y down, z forward",
            "transform_direction": "T_parent_camera maps homogeneous points from camera frame into parent frame",
        },
        "cameras": cameras,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved camera extrinsics JSON: {args.output}")

    if args.write_npy:
        write_npy_outputs(np, datasets, cameras)


if __name__ == "__main__":
    main()
