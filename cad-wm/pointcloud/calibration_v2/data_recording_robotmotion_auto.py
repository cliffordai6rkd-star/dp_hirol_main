from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R
from sshkeyboard import stop_listening


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factory.tasks.robot_motion import RobotMotion
from hardware.base.utils import convert_7D_2_homo, dynamic_load_yaml


DEFAULT_CONFIG_PATH = "pointcloud/calibration_v2/robotmotion_auto_config.yaml"
DEFAULT_CONFIG = {
    "robot": {
        "motion_config": "factory/components/motion_configs/left_fr3_with_pika_ati_ik.yaml",
        "execute_hardware": False,
        "dry_run_record": False,
        "pose_source": "model",
        "initialize_gripper": False,
    },
    "camera": {
        "name": "ee_cam",
        "cali_type": "eye_in_hand",
        "warmup": 1.0,
        "display": True,
    },
    "tag": {
        "family": "tag36h11",
        "size": 0.08,
        "id": None,
    },
    "workspace": {
        "grid_size": [3, 3, 2],
        "spacing": [0.04, 0.04, 0.04],
        "center": None,
        "base_rpy_deg": None,
        "orientation_randomness": 12.0,
        "seed": 0,
    },
    "collection": {
        "settle_time": 2.5,
        "retry_interval": 0.2,
        "stop_key": "q",
    },
    "output": {
        "dir": None,
        "json_name": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic AprilTag hand-eye data collection with RobotMotion + Franka FR3."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="YAML config path. All calibration controls are read from this file.",
    )
    return parser.parse_args()


def merge_config(default: dict, override: Optional[dict]) -> dict:
    result = copy.deepcopy(default)
    if override is None:
        return result

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path) -> dict:
    loaded_config = dynamic_load_yaml(str(resolve_repo_path(config_path)))
    return merge_config(DEFAULT_CONFIG, loaded_config)


def normalize_cali_type(cali_type: str) -> str:
    normalized = str(cali_type).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in {"eye_in_hand", "eye_to_hand"}:
        raise ValueError("camera.cali_type must be 'eye_in_hand' or 'eye_to_hand'")
    return normalized


def validate_config(config: dict) -> None:
    robot_cfg = config["robot"]
    camera_cfg = config["camera"]
    tag_cfg = config["tag"]
    workspace_cfg = config["workspace"]

    if not robot_cfg["motion_config"]:
        raise ValueError("robot.motion_config is required")
    if not camera_cfg["name"]:
        raise ValueError("camera.name is required")
    normalize_cali_type(camera_cfg.get("cali_type", "eye_in_hand"))
    if not tag_cfg["family"]:
        raise ValueError("tag.family is required")
    if not isinstance(workspace_cfg["grid_size"], list) or len(workspace_cfg["grid_size"]) != 3:
        raise ValueError("workspace.grid_size must be a list of 3 integers")
    if not isinstance(workspace_cfg["spacing"], list) or len(workspace_cfg["spacing"]) != 3:
        raise ValueError("workspace.spacing must be a list of 3 numbers")
    if workspace_cfg["center"] is not None and len(workspace_cfg["center"]) != 3:
        raise ValueError("workspace.center must be null or a list of 3 numbers")
    if workspace_cfg["base_rpy_deg"] is not None and len(workspace_cfg["base_rpy_deg"]) != 3:
        raise ValueError("workspace.base_rpy_deg must be null or a list of 3 numbers")


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def make_output_dir(output_dir: Optional[str]) -> Path:
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"pointcloud/calibration_v2/robotmotion_runs/{stamp}"
    path = resolve_repo_path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "images").mkdir(parents=True, exist_ok=True)
    return path


def prepare_robot_motion_config(motion_config_path: str | Path, output_dir: Path, initialize_gripper: bool) -> Path:
    resolved_motion_config = resolve_repo_path(motion_config_path)
    motion_config = dynamic_load_yaml(str(resolved_motion_config))

    if not initialize_gripper:
        if "motion_config" in motion_config:
            raw_motion_config = motion_config["motion_config"]
            outer_config = copy.deepcopy(motion_config)
        else:
            raw_motion_config = motion_config
            outer_config = None

        raw_motion_config = copy.deepcopy(raw_motion_config)
        raw_motion_config["gripper"] = None
        raw_motion_config.pop("gripper_config", None)

        safe_motion_config_path = output_dir / "motion_config_without_gripper.yaml"
        with safe_motion_config_path.open("w") as file:
            yaml.safe_dump(raw_motion_config, file, sort_keys=False)

        if outer_config is not None:
            safe_wrapper_path = output_dir / "robot_motion_wrapper.yaml"
            outer_config["motion_config"] = raw_motion_config
            with safe_wrapper_path.open("w") as file:
                yaml.safe_dump(outer_config, file, sort_keys=False)
            return safe_wrapper_path

        return write_robot_motion_wrapper(safe_motion_config_path, output_dir)

    if "motion_config" in motion_config:
        return resolved_motion_config

    return write_robot_motion_wrapper(resolved_motion_config, output_dir)


def write_robot_motion_wrapper(raw_motion_config_path: Path, output_dir: Path) -> Path:
    wrapper_path = output_dir / "robot_motion_wrapper.yaml"
    relative_motion_config = os.path.relpath(raw_motion_config_path, PROJECT_ROOT)
    wrapper_config = {
        "motion_config": f"!include {relative_motion_config}",
        "data_collection": {
            "save_path_prefix": "calibration_v2_robotmotion_auto",
            "data_record_frequency": 30,
            "image_visualization": False,
            "rerun_visualization": False,
            "task_description": "Calibration v2 RobotMotion automatic collection",
            "task_description_goal": "Collect Franka robot poses and camera AprilTag detections",
            "task_description_steps": "Move through configured grid poses and save valid detections.",
        },
        "motion_control": {
            "control_loop_time": 0.02,
            "enable_keyboard_listener": False,
            "reset_space": "joint",
            "reset_arm_command": [
                -0.002240852524182456,
                -0.7834976804122445,
                -7.442107452836265e-05,
                -2.357436399024249,
                -0.00532840631613707,
                1.566669550357334,
                0.7724277178011111,
            ],
            "reset_tool_command": {"single": 1.0},
        },
    }

    with wrapper_path.open("w") as file:
        for key, value in wrapper_config.items():
            if key == "motion_config":
                file.write(f"motion_config: {value}\n")
            else:
                file.write(yaml.safe_dump({key: value}, sort_keys=False))
    return wrapper_path


def shutdown_robot_motion(robot_motion: Optional[RobotMotion]) -> None:
    if robot_motion is None:
        return

    try:
        robot_motion._main_thread_running = False
    except Exception:
        pass

    try:
        data_thread = getattr(robot_motion, "_data_recording_thread", None)
        if data_thread is not None and data_thread.is_alive():
            data_thread.join(timeout=2.0)
    except Exception:
        pass

    try:
        stop_listening()
    except Exception:
        pass

    try:
        motion_factory = getattr(robot_motion, "_motion_factory", None)
        if motion_factory is not None:
            motion_factory.close()
    except Exception:
        pass


def find_camera(robot_motion: RobotMotion, camera_name: str):
    robot_system = robot_motion._robot_system
    cameras = robot_system._sensors.get("camera", [])
    available = [camera["name"] for camera in cameras]

    for camera in cameras:
        if camera["name"] == camera_name:
            return camera["object"], available

    raise RuntimeError(
        f"Camera '{camera_name}' not found in RobotMotion sensors. "
        f"Available cameras: {available}"
    )


def get_intrinsics_dict(camera_obj) -> dict:
    if not hasattr(camera_obj, "_intrinsics"):
        raise RuntimeError(
            f"Camera object {type(camera_obj).__name__} has no _intrinsics attribute. "
            "This script currently expects a RealSense camera."
        )

    intrinsics = camera_obj._intrinsics
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "coeffs": [float(value) for value in getattr(intrinsics, "coeffs", [])],
    }


def capture_color_image(camera_obj) -> np.ndarray:
    image_data = camera_obj.capture_all_data()
    image = image_data.get("image")
    if image is None:
        raise RuntimeError("Failed to capture color image from selected camera.")
    return image.copy()


def random_quat_around(base_quat: np.ndarray, max_angle_deg: float, rng: np.random.Generator) -> np.ndarray:
    if max_angle_deg <= 0.0:
        return base_quat.copy()

    axis = rng.normal(size=3)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return base_quat.copy()

    axis /= axis_norm
    angle = rng.uniform(-max_angle_deg, max_angle_deg) * np.pi / 180.0
    return (R.from_rotvec(axis * angle) * R.from_quat(base_quat)).as_quat()


def generate_grid_poses(
    center: np.ndarray,
    grid_size: list[int],
    spacing: list[float],
    base_quat: np.ndarray,
    orientation_randomness: float,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    nx, ny, nz = grid_size
    dx, dy, dz = spacing

    start = np.array(
        [
            center[0] - (nx - 1) * dx / 2.0,
            center[1] - (ny - 1) * dy / 2.0,
            center[2] - (nz - 1) * dz / 2.0,
        ]
    )

    poses = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                position = start + np.array([ix * dx, iy * dy, iz * dz])
                quat = random_quat_around(base_quat, orientation_randomness, rng)
                poses.append(np.concatenate([position, quat]))
    return poses


def get_robot_pose_matrix(robot_motion: RobotMotion, pose_source: str) -> np.ndarray:
    robot_system = robot_motion._robot_system
    if pose_source == "hardware" and robot_system._use_hardware:
        robot = getattr(robot_system, "_robot", None)
        if robot is not None and hasattr(robot, "get_ee_pose"):
            return np.array(robot.get_ee_pose(), dtype=np.float64)

    state = robot_motion.get_state()
    return convert_7D_2_homo(np.array(state["pose"], dtype=np.float64))


def select_detection(detections, tag_id: Optional[int]):
    if tag_id is not None:
        detections = [detection for detection in detections if detection.tag_id == tag_id]
    if not detections:
        return None

    return max(detections, key=lambda detection: cv2.contourArea(detection.corners.astype(np.float32)))


def detect_tag_pose(
    image: np.ndarray,
    detector: Detector,
    camera_params: list[float],
    tag_size: float,
    tag_id: Optional[int],
):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=tag_size,
    )
    detection = select_detection(detections, tag_id)
    if detection is None:
        return None, None

    tag_pose = np.eye(4, dtype=np.float64)
    tag_pose[:3, :3] = detection.pose_R
    tag_pose[:3, 3] = detection.pose_t.flatten()
    return detection, tag_pose


def draw_detection(
    image: np.ndarray,
    detection,
    tag_pose: Optional[np.ndarray],
    intrinsics: dict,
    sample_count: int,
) -> np.ndarray:
    display = image.copy()
    status = f"Samples: {sample_count}"

    if detection is None or tag_pose is None:
        cv2.putText(display, status + " | No tag", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return display

    corners = detection.corners.astype(int)
    for i in range(4):
        pt1 = tuple(corners[i])
        pt2 = tuple(corners[(i + 1) % 4])
        cv2.line(display, pt1, pt2, (0, 255, 0), 2)

    center = tuple(map(int, detection.center))
    cv2.circle(display, center, 5, (0, 0, 255), -1)
    cv2.putText(display, f"ID: {detection.tag_id}", (center[0] - 10, center[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["ppx"], intrinsics["ppy"]
    t = tag_pose[:3, 3]
    rot = tag_pose[:3, :3]

    axis_length = 0.12
    axes = np.array([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for axis, color in zip(axes, colors):
        endpoint = t + rot @ axis
        if endpoint[2] <= 1e-6:
            continue
        x = int(fx * endpoint[0] / endpoint[2] + cx)
        y = int(fy * endpoint[1] / endpoint[2] + cy)
        cv2.line(display, center, (x, y), color, 2)

    text = f"{status} | XYZ: {t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f} m"
    cv2.putText(display, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return display


def save_json(json_path: Path, samples: list[dict], intrinsics: dict, metadata: dict) -> None:
    with json_path.open("w") as file:
        json.dump({"data": samples, "intrinsics": intrinsics, "metadata": metadata}, file, indent=4)


def main() -> None:
    args = parse_args()

    # RobotMotion's !include YAML loader resolves paths against cwd in this repo.
    os.chdir(PROJECT_ROOT)
    config = load_config(args.config)
    validate_config(config)

    robot_cfg = config["robot"]
    camera_cfg = config["camera"]
    tag_cfg = config["tag"]
    workspace_cfg = config["workspace"]
    collection_cfg = config["collection"]
    output_cfg = config["output"]

    pose_source = robot_cfg["pose_source"]
    if pose_source not in {"model", "hardware"}:
        raise ValueError("robot.pose_source must be 'model' or 'hardware'")

    camera_name = camera_cfg["name"]
    cali_type = normalize_cali_type(camera_cfg["cali_type"])
    execute_hardware = bool(robot_cfg["execute_hardware"])
    dry_run_record = bool(robot_cfg["dry_run_record"])
    initialize_gripper = bool(robot_cfg.get("initialize_gripper", False))
    display_enabled = bool(camera_cfg["display"])
    stop_key = str(collection_cfg.get("stop_key", "q"))[:1]

    output_dir = make_output_dir(output_cfg["dir"])
    json_name = output_cfg["json_name"] or f"data_{camera_name}.json"
    json_path = output_dir / json_name

    robot_motion = None
    samples: list[dict] = []
    intrinsics: dict = {}
    metadata = {
        "script": Path(__file__).name,
        "config": str(resolve_repo_path(args.config)),
        "created_at": datetime.now().isoformat(),
        "motion_config": robot_cfg["motion_config"],
        "camera": camera_name,
        "cali_type": cali_type,
        "tag_family": tag_cfg["family"],
        "tag_size": tag_cfg["size"],
        "tag_id": tag_cfg["id"],
        "pose_source": pose_source,
        "execute_hardware": execute_hardware,
        "dry_run_record": dry_run_record,
        "initialize_gripper": initialize_gripper,
        "resolved_config": config,
    }

    try:
        robot_motion_config = prepare_robot_motion_config(robot_cfg["motion_config"], output_dir, initialize_gripper)
        metadata["robot_motion_config"] = str(robot_motion_config)

        if not initialize_gripper:
            print("Gripper initialization is disabled for this calibration run.")
            print("The temporary RobotMotion config removes gripper/gripper_config to avoid opening the Pika gripper.")

        robot_motion = RobotMotion(str(robot_motion_config), auto_initialize=False)
        robot_motion._img_visualization = False
        robot_motion.initialize()

        if execute_hardware:
            robot_motion.enable_hardware()
        else:
            robot_motion.disable_hardware()
            print("Hardware execution is disabled. Set robot.execute_hardware: true when you are ready to move the robot.")
            if not dry_run_record:
                print("Dry run will not save samples. Set robot.dry_run_record: true to debug JSON/image writing without motion.")

        camera_obj, available_cameras = find_camera(robot_motion, camera_name)
        intrinsics = get_intrinsics_dict(camera_obj)
        metadata["available_cameras"] = available_cameras

        camera_params = [intrinsics["fx"], intrinsics["fy"], intrinsics["ppx"], intrinsics["ppy"]]
        detector = Detector(families=tag_cfg["family"])

        print(f"Selected camera: {camera_name}")
        print(f"Camera intrinsics: fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}, "
              f"cx={intrinsics['ppx']:.2f}, cy={intrinsics['ppy']:.2f}")
        print(f"Output JSON: {json_path}")

        time.sleep(float(camera_cfg["warmup"]))

        current_pose = robot_motion.get_state()["pose"]
        center_cfg = workspace_cfg["center"]
        center = np.array(center_cfg, dtype=np.float64) if center_cfg is not None else current_pose[:3]
        base_rpy_deg = workspace_cfg["base_rpy_deg"]
        if base_rpy_deg is None:
            base_quat = np.array(current_pose[3:7], dtype=np.float64)
        else:
            base_quat = R.from_euler("xyz", base_rpy_deg, degrees=True).as_quat()

        poses = generate_grid_poses(
            center=center,
            grid_size=workspace_cfg["grid_size"],
            spacing=workspace_cfg["spacing"],
            base_quat=base_quat,
            orientation_randomness=float(workspace_cfg["orientation_randomness"]),
            seed=int(workspace_cfg["seed"]),
        )
        metadata["grid"] = {
            "center": center.tolist(),
            "grid_size": workspace_cfg["grid_size"],
            "spacing": workspace_cfg["spacing"],
            "base_quat_xyzw": base_quat.tolist(),
            "orientation_randomness": workspace_cfg["orientation_randomness"],
            "seed": workspace_cfg["seed"],
        }

        print(f"Generated {len(poses)} target poses. Press '{stop_key}' in preview window to stop early.")

        stop_requested = False
        for pose_index, target_pose in enumerate(poses):
            print(f"Moving to pose {pose_index + 1}/{len(poses)}: {np.round(target_pose[:3], 4)}")
            robot_motion.send_pose_command(target_pose)
            time.sleep(float(collection_cfg["settle_time"]))

            attempts = 0
            while True:
                attempts += 1
                image = capture_color_image(camera_obj)
                detection, tag_pose = detect_tag_pose(
                    image=image,
                    detector=detector,
                    camera_params=camera_params,
                    tag_size=float(tag_cfg["size"]),
                    tag_id=tag_cfg["id"],
                )

                display_image = draw_detection(image, detection, tag_pose, intrinsics, len(samples))
                if display_enabled:
                    cv2.imshow("RobotMotion AprilTag Calibration", display_image)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(stop_key):
                        print("Stopped by user.")
                        stop_requested = True
                        break

                if detection is None or tag_pose is None:
                    # if attempts == 1 or attempts % 10 == 0:
                    print(
                        f"  Pose {pose_index + 1}/{len(poses)} has no valid AprilTag detection "
                        f"(attempt {attempts}); skip."
                    )
                    time.sleep(float(collection_cfg["retry_interval"]))
                    break

                if not execute_hardware and not dry_run_record:
                    print("  Detection succeeded, but sample was not saved because hardware execution is disabled.")
                    stop_requested = True
                    break

                robot_pose = get_robot_pose_matrix(robot_motion, pose_source)
                image_rel_path = Path("images") / f"sample_{len(samples):04d}.png"
                image_abs_path = output_dir / image_rel_path
                cv2.imwrite(str(image_abs_path), image)

                samples.append(
                    {
                        "image": str(image_rel_path),
                        "matrix": robot_pose.tolist(),
                        "homogeneous_matrix": tag_pose.tolist(),
                        "target_pose": target_pose.tolist(),
                        "tag_id": int(detection.tag_id),
                        "detection_attempts": attempts,
                    }
                )
                save_json(json_path, samples, intrinsics, metadata)
                print(f"  Recorded sample {len(samples)}/{len(poses)} -> {image_rel_path} (attempts: {attempts})")
                break

            if stop_requested:
                break

        save_json(json_path, samples, intrinsics, metadata)
        print(f"Finished. Recorded {len(samples)} samples.")
        print(f"Data saved to: {json_path}")

    except KeyboardInterrupt:
        save_json(json_path, samples, intrinsics, metadata)
        print("Interrupted; partial data saved.")
        raise
    finally:
        shutdown_robot_motion(robot_motion)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
