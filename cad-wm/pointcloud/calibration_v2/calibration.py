from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def normalize_cali_type(cali_type: str) -> str:
    normalized = str(cali_type).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in {"eye_in_hand", "eye_to_hand"}:
        raise ValueError("cali_type must be 'eye_in_hand' or 'eye_to_hand'")
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve calibration_v2 hand-eye calibration data.")
    parser.add_argument(
        "--data",
        default="data.json",
        help="Path to calibration JSON produced by data_recording_robotmotion_auto.py.",
    )
    parser.add_argument(
        "--cali-type",
        default=None,
        help="Override calibration type: eye_in_hand or eye_to_hand. Defaults to JSON metadata.cali_type.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional .npy output path. Defaults to T_ee_camera.npy or T_base_camera.npy next to data file.",
    )
    return parser.parse_args()


def load_data(file_path: str | Path):
    with Path(file_path).open("r") as file:
        raw = json.load(file)

    robot_poses = []
    tag_poses = []
    for entry in raw["data"]:
        robot_poses.append(np.array(entry["matrix"], dtype=np.float64))
        tag_poses.append(np.array(entry["homogeneous_matrix"], dtype=np.float64))

    metadata = raw.get("metadata", {})
    return robot_poses, tag_poses, metadata


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation).reshape(3)
    return transform


def solve_eye_in_hand(robot_poses: list[np.ndarray], tag_poses: list[np.ndarray]) -> np.ndarray:
    """
    Camera is mounted on the Franka end-effector and the AprilTag is fixed.

    Input:
        robot_poses: T_base_ee for each sample.
        tag_poses: T_camera_tag for each sample.

    Output:
        T_ee_camera, so T_base_camera = T_base_ee @ T_ee_camera.
    """
    r_gripper2base = [pose[:3, :3] for pose in robot_poses]
    t_gripper2base = [pose[:3, 3:4] for pose in robot_poses]
    r_target2cam = [pose[:3, :3] for pose in tag_poses]
    t_target2cam = [pose[:3, 3:4] for pose in tag_poses]

    r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        r_gripper2base,
        t_gripper2base,
        r_target2cam,
        t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )
    return make_transform(r_cam2gripper, t_cam2gripper)


def solve_eye_to_hand(robot_poses: list[np.ndarray], tag_poses: list[np.ndarray]) -> np.ndarray:
    """
    Camera is fixed in the scene and the AprilTag is mounted on the Franka end-effector.

    Input:
        robot_poses: T_base_ee for each sample.
        tag_poses: T_camera_tag for each sample.

    Output:
        T_base_camera, so T_base_tag = T_base_camera @ T_camera_tag.
    """
    r_gripper2base = [pose[:3, :3] for pose in robot_poses]
    t_gripper2base = [pose[:3, 3:4] for pose in robot_poses]
    r_target2cam = [pose[:3, :3] for pose in tag_poses]
    t_target2cam = [pose[:3, 3:4] for pose in tag_poses]

    r_base2cam, t_base2cam, _, _ = cv2.calibrateRobotWorldHandEye(
        r_gripper2base,
        t_gripper2base,
        r_target2cam,
        t_target2cam,
        method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
    )
    return make_transform(r_base2cam, t_base2cam)


def rotation_error_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    r_delta = r_a @ r_b.T
    cos_angle = np.clip((np.trace(r_delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def verify_result(robot_poses: list[np.ndarray], tag_poses: list[np.ndarray], result: np.ndarray, cali_type: str) -> None:
    if cali_type == "eye_in_hand":
        estimated_fixed_tag = [t_base_ee @ result @ t_camera_tag for t_base_ee, t_camera_tag in zip(robot_poses, tag_poses)]
        label = "T_base_tag"
    else:
        estimated_fixed_tag = [result @ t_camera_tag for t_camera_tag in tag_poses]
        label = "T_base_tag"

    reference = estimated_fixed_tag[0]
    pos_errors = [np.linalg.norm(pose[:3, 3] - reference[:3, 3]) for pose in estimated_fixed_tag]
    rot_errors = [rotation_error_deg(pose[:3, :3], reference[:3, :3]) for pose in estimated_fixed_tag]

    print(f"\nVerification using fixed {label} consistency:")
    print(f"  mean position error: {np.mean(pos_errors) * 1000:.2f} mm")
    print(f"  max  position error: {np.max(pos_errors) * 1000:.2f} mm")
    print(f"  mean rotation error: {np.mean(rot_errors):.3f} deg")
    print(f"  max  rotation error: {np.max(rot_errors):.3f} deg")


def solve(robot_poses: list[np.ndarray], tag_poses: list[np.ndarray], cali_type: str) -> np.ndarray:
    if len(robot_poses) != len(tag_poses):
        raise ValueError(f"Data size mismatch: {len(robot_poses)} robot poses vs {len(tag_poses)} tag poses")
    if len(robot_poses) < 3:
        raise ValueError("At least 3 samples are required, 10-30 diverse samples are recommended")

    if cali_type == "eye_in_hand":
        return solve_eye_in_hand(robot_poses, tag_poses)
    return solve_eye_to_hand(robot_poses, tag_poses)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    robot_poses, tag_poses, metadata = load_data(data_path)

    cali_type = args.cali_type or metadata.get("cali_type") or metadata.get("calibration_type") or "eye_to_hand"
    cali_type = normalize_cali_type(cali_type)

    print(f"Loaded {len(robot_poses)} samples from {data_path}")
    print(f"Calibration type: {cali_type}")

    result = solve(robot_poses, tag_poses, cali_type)
    verify_result(robot_poses, tag_poses, result, cali_type)

    if cali_type == "eye_in_hand":
        result_name = "T_ee_camera"
        default_output = data_path.with_name("T_ee_camera.npy")
        print("\nResult: T_ee_camera (camera frame to end-effector frame)")
    else:
        result_name = "T_base_camera"
        default_output = data_path.with_name("T_base_camera.npy")
        print("\nResult: T_base_camera (camera frame to robot base frame)")

    print(result)
    output_path = Path(args.output) if args.output is not None else default_output
    np.save(output_path, result)
    print(f"\nSaved {result_name} to: {output_path}")


if __name__ == "__main__":
    main()
