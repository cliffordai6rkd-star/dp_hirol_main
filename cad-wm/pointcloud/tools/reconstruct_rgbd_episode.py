from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DATASET = Path("data/train_episode/Ft_test_data")
DEFAULT_INPUT_JSON = "data_with_rgbd_extrinsics.json"
DEFAULT_OUTPUT_DIR = Path("outputs/rgbd_pointcloud/Ft_test_data")
DEFAULT_CONFIG = Path("pointcloud/config/rgbd_reconstruction/ft_test_data.yaml")


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8")) or {}

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: PyYAML is required for YAML config files.") from exc

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping/object: {path}")
    return payload


def flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    # Flat keys matching argparse destinations are also accepted.
    for key, value in config.items():
        if not isinstance(value, dict):
            flat[key.replace("-", "_")] = value

    groups = {
        "dataset": {
            "dir": "dataset_dir",
            "dataset_dir": "dataset_dir",
            "input_json": "input_json",
        },
        "output": {
            "dir": "output_dir",
            "output_dir": "output_dir",
            "name": "output_name",
            "output_name": "output_name",
            "ascii": "ascii",
            "save_per_frame": "save_per_frame",
        },
        "frames": {
            "start": "start_frame",
            "start_frame": "start_frame",
            "end": "end_frame",
            "end_frame": "end_frame",
            "step": "frame_step",
            "frame_step": "frame_step",
            "max": "max_frames",
            "max_frames": "max_frames",
        },
        "projection": {
            "cameras": "cameras",
            "stride": "stride",
            "depth_min": "depth_min",
            "depth_max": "depth_max",
            "depth_scale_m_per_unit": "depth_scale_m_per_unit",
        },
        "downsample": {
            "voxel_size": "voxel_size",
            "max_points": "max_points",
        },
        "visualization": {
            "view": "view",
            "view_existing": "view_existing",
            "coord_frame_size": "coord_frame_size",
            "window_width": "window_width",
            "window_height": "window_height",
        },
    }

    for group_name, mapping in groups.items():
        group = config.get(group_name)
        if group is None:
            continue
        if not isinstance(group, dict):
            raise ValueError(f"Config section {group_name!r} must be a mapping/object")
        for key, value in group.items():
            dest = mapping.get(key.replace("-", "_"))
            if dest is None:
                raise ValueError(f"Unsupported config key: {group_name}.{key}")
            flat[dest] = value

    return flat


def path_or_none(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value)


def normalize_cameras(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "all", "none", "null"}:
            return None
        if "," in stripped:
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return [stripped]
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError("projection.cameras must be null, a camera name string, or a list of camera names")


def normalize_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return value


def coerce_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"Config key {key!r} must be a boolean or boolean-like string")


def coerce_int(value: Any, key: str, *, allow_none: bool = False) -> int | None:
    value = normalize_none(value)
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"Config key {key!r} cannot be null")
    if isinstance(value, bool):
        raise ValueError(f"Config key {key!r} must be an integer, not a boolean")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError as exc:
            raise ValueError(f"Config key {key!r} must be an integer") from exc
    raise ValueError(f"Config key {key!r} must be an integer")


def coerce_float(value: Any, key: str, *, allow_none: bool = False) -> float | None:
    value = normalize_none(value)
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"Config key {key!r} cannot be null")
    if isinstance(value, bool):
        raise ValueError(f"Config key {key!r} must be a float, not a boolean")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError as exc:
            raise ValueError(f"Config key {key!r} must be a float") from exc
    raise ValueError(f"Config key {key!r} must be a float")


def coerce_path(value: Any, *, allow_none: bool = False) -> Path | None:
    value = normalize_none(value)
    if value is None:
        if allow_none:
            return None
        raise ValueError("path config cannot be null")
    return value if isinstance(value, Path) else Path(str(value))


def coerce_str(value: Any, key: str, *, allow_none: bool = False) -> str | None:
    value = normalize_none(value)
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"Config key {key!r} cannot be null")
    return str(value)


def config_defaults(config: dict[str, Any]) -> dict[str, Any]:
    defaults = flatten_config(config)
    if "dataset_dir" in defaults:
        defaults["dataset_dir"] = coerce_path(defaults["dataset_dir"])
    if "input_json" in defaults:
        defaults["input_json"] = coerce_str(defaults["input_json"], "input_json")
    if "output_dir" in defaults:
        defaults["output_dir"] = coerce_path(defaults["output_dir"])
    if "output_name" in defaults:
        defaults["output_name"] = coerce_str(defaults["output_name"], "output_name")
    if "cameras" in defaults:
        defaults["cameras"] = normalize_cameras(defaults["cameras"])
    for key in ("start_frame", "frame_step", "max_frames", "stride", "max_points", "window_width", "window_height"):
        if key in defaults:
            defaults[key] = coerce_int(defaults[key], key, allow_none=key in {"max_frames"})
    for key in ("end_frame",):
        if key in defaults:
            defaults[key] = coerce_int(defaults[key], key, allow_none=True)
    for key in ("depth_min", "depth_max", "depth_scale_m_per_unit", "voxel_size", "coord_frame_size"):
        if key in defaults:
            defaults[key] = coerce_float(defaults[key], key, allow_none=key == "depth_scale_m_per_unit")
    for key in ("save_per_frame", "ascii", "view"):
        if key in defaults:
            defaults[key] = coerce_bool(defaults[key], key)
    if "view_existing" in defaults:
        defaults["view_existing"] = coerce_path(defaults["view_existing"], allow_none=True)
    return defaults


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct fused RGB-D point clouds from an augmented HIROL episode JSON. "
            "All reconstruction parameters are read from a YAML/JSON config. "
            "Run pointcloud/tools/add_rgbd_extrinsics.py first."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML/JSON reconstruction config.",
    )
    cli_args = parser.parse_args()
    values = config_defaults(load_config(cli_args.config))

    defaults = {
        "dataset_dir": DEFAULT_DATASET,
        "input_json": DEFAULT_INPUT_JSON,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "output_name": "episode_fused.ply",
        "cameras": None,
        "start_frame": 0,
        "end_frame": None,
        "frame_step": 1,
        "max_frames": None,
        "stride": 4,
        "depth_min": 0.05,
        "depth_max": 2.0,
        "depth_scale_m_per_unit": None,
        "voxel_size": 0.003,
        "max_points": 2_000_000,
        "save_per_frame": False,
        "ascii": False,
        "view": False,
        "view_existing": None,
        "coord_frame_size": 0.08,
        "window_width": 1280,
        "window_height": 800,
    }
    defaults.update(values)
    defaults["config"] = cli_args.config
    return argparse.Namespace(**defaults)


def load_runtime_deps():
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: this script needs numpy and opencv-python.") from exc
    try:
        import open3d as o3d
    except ModuleNotFoundError:
        o3d = None
    return cv2, np, o3d


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_indices(num_frames: int, args: argparse.Namespace) -> list[int]:
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be positive")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative")

    end = num_frames if args.end_frame is None else min(args.end_frame, num_frames)
    indices = list(range(args.start_frame, end, args.frame_step))
    if args.max_frames is not None:
        indices = indices[: max(0, args.max_frames)]
    return indices


def image_path(dataset_dir: Path, rel_path: str | None) -> Path | None:
    if not rel_path:
        return None
    path = Path(rel_path)
    if path.is_absolute():
        return path
    return dataset_dir / path


def load_color(path: Path, cv2, np):
    color_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise FileNotFoundError(f"failed to read color image: {path}")
    return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8, copy=False)


def load_depth_meters(path: Path, scale_m_per_unit: float | None, cv2, np):
    if path.suffix.lower() == ".npy":
        depth = np.load(path)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"failed to read depth image: {path}")

    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth[..., 0]

    if np.issubdtype(depth.dtype, np.floating) and scale_m_per_unit is None:
        return depth.astype(np.float32, copy=False)

    scale = 0.001 if scale_m_per_unit is None else float(scale_m_per_unit)
    return depth.astype(np.float32) * scale


def resize_color_to_depth(color_rgb, depth_shape: tuple[int, int], cv2):
    if color_rgb.shape[:2] == depth_shape:
        return color_rgb
    return cv2.resize(color_rgb, (depth_shape[1], depth_shape[0]), interpolation=cv2.INTER_AREA)


def project_depth_to_camera(depth_m, color_rgb, intr: dict[str, Any], args: argparse.Namespace, np):
    height, width = depth_m.shape[:2]
    ys = np.arange(0, height, args.stride, dtype=np.float32)
    xs = np.arange(0, width, args.stride, dtype=np.float32)
    u, v = np.meshgrid(xs, ys)

    depth_sample = depth_m[:: args.stride, :: args.stride]
    valid = (
        np.isfinite(depth_sample)
        & (depth_sample >= args.depth_min)
        & (depth_sample <= args.depth_max)
    )
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    z = depth_sample[valid]
    x = (u[valid] - float(intr["ppx"])) / float(intr["fx"]) * z
    y = (v[valid] - float(intr["ppy"])) / float(intr["fy"]) * z
    points = np.stack([x, y, z], axis=-1).astype(np.float32)
    colors = color_rgb[:: args.stride, :: args.stride][valid].astype(np.uint8, copy=False)
    return points, colors


def transform_points(points_camera, t_base_camera, np):
    if points_camera.shape[0] == 0:
        return points_camera
    transform = np.asarray(t_base_camera, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return (points_camera.astype(np.float64) @ rotation.T + translation).astype(np.float32)


def voxel_downsample(points, colors, voxel_size: float, np, o3d):
    if voxel_size <= 0.0 or points.shape[0] == 0:
        return points, colors

    if o3d is not None:
        pcd = make_o3d_cloud(points, colors, o3d)
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))
        return o3d_to_arrays(pcd, np)

    voxels = np.floor(points / float(voxel_size)).astype(np.int64)
    _, keep = np.unique(voxels, axis=0, return_index=True)
    keep.sort()
    return points[keep], colors[keep]


def random_downsample(points, colors, max_points: int, np):
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors
    rng = np.random.default_rng(0)
    keep = rng.choice(points.shape[0], size=max_points, replace=False)
    keep.sort()
    return points[keep], colors[keep]


def make_o3d_cloud(points, colors, o3d):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype("float64", copy=False))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype("float64", copy=False) / 255.0)
    return pcd


def o3d_to_arrays(pcd, np):
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.clip(np.asarray(pcd.colors) * 255.0, 0, 255).astype(np.uint8)
    return points, colors


def view_point_cloud(pcd, args: argparse.Namespace, o3d, *, title: str) -> None:
    if o3d is None:
        raise SystemExit("Open3D is required for --view. Install open3d or omit the visualization flag.")

    geometries = [pcd]
    if args.coord_frame_size > 0.0:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.coord_frame_size))

    print("Opening Open3D viewer. Mouse: left-rotate, wheel-zoom, right/middle-pan.")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=int(args.window_width),
        height=int(args.window_height),
    )


def write_point_cloud(path: Path, points, colors, np, o3d, ascii_ply: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if o3d is not None:
        pcd = make_o3d_cloud(points, colors, o3d)
        ok = o3d.io.write_point_cloud(str(path), pcd, write_ascii=ascii_ply)
        if not ok:
            raise RuntimeError(f"Open3D failed to write point cloud: {path}")
        return

    with path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {points.shape[0]}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("end_header\n")
        for point, color in zip(points, colors):
            file.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def depth_scale_for(camera_static: dict[str, Any], args: argparse.Namespace) -> float | None:
    if args.depth_scale_m_per_unit is not None:
        return float(args.depth_scale_m_per_unit)
    scale = camera_static.get("depth_scale_m_per_unit")
    if scale is not None:
        return float(scale)
    return None


def available_camera_names(frame: dict[str, Any], requested: list[str] | None) -> list[str]:
    frame_cameras = sorted(((frame.get("rgbd_extrinsics") or {}).get("cameras") or {}).keys())
    if requested is None:
        return frame_cameras
    requested_set = set(requested)
    return [camera_name for camera_name in frame_cameras if camera_name in requested_set]


def validate_requested_cameras(camera_static: dict[str, Any], requested: list[str] | None) -> None:
    if requested is None:
        return
    available = sorted(camera_static.keys())
    unknown = sorted(set(requested) - set(available))
    if not unknown:
        return
    hints = []
    if "thrid_person_cam" in unknown and "third_person_cam" in available:
        hints.append("Did you mean 'third_person_cam'?")
    hint = f" {' '.join(hints)}" if hints else ""
    raise ValueError(
        f"Unknown camera name(s): {unknown}. Available cameras: {available}.{hint}"
    )


def reconstruct_camera(
    *,
    dataset_dir: Path,
    frame_camera: dict[str, Any],
    camera_static: dict[str, Any],
    args: argparse.Namespace,
    cv2,
    np,
) -> tuple[Any, Any]:
    color_path = image_path(dataset_dir, frame_camera.get("color_path"))
    depth_path = image_path(dataset_dir, frame_camera.get("depth_path"))
    if color_path is None or depth_path is None:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    color_rgb = load_color(color_path, cv2, np)
    depth_m = load_depth_meters(depth_path, depth_scale_for(camera_static, args), cv2, np)
    color_rgb = resize_color_to_depth(color_rgb, depth_m.shape[:2], cv2)
    points_camera, colors = project_depth_to_camera(
        depth_m=depth_m,
        color_rgb=color_rgb,
        intr=camera_static["intrinsics"],
        args=args,
        np=np,
    )
    points_base = transform_points(points_camera, frame_camera["T_base_camera"], np)
    return points_base, colors


def reconstruct_frame(
    *,
    dataset_dir: Path,
    frame: dict[str, Any],
    camera_static: dict[str, Any],
    camera_names: list[str] | None,
    args: argparse.Namespace,
    cv2,
    np,
) -> tuple[Any, Any, dict[str, Any]]:
    frame_extrinsics = frame.get("rgbd_extrinsics") or {}
    frame_cameras = frame_extrinsics.get("cameras") or {}
    points_parts = []
    color_parts = []
    camera_counts: dict[str, int] = {}

    for camera_name in available_camera_names(frame, camera_names):
        if camera_name not in camera_static:
            camera_counts[camera_name] = 0
            continue
        points, colors = reconstruct_camera(
            dataset_dir=dataset_dir,
            frame_camera=frame_cameras[camera_name],
            camera_static=camera_static[camera_name],
            args=args,
            cv2=cv2,
            np=np,
        )
        camera_counts[camera_name] = int(points.shape[0])
        if points.shape[0] > 0:
            points_parts.append(points)
            color_parts.append(colors)

    if not points_parts:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            {"idx": frame.get("idx"), "points": 0, "cameras": camera_counts},
        )

    points = np.concatenate(points_parts, axis=0)
    colors = np.concatenate(color_parts, axis=0)
    summary = {"idx": frame.get("idx"), "points": int(points.shape[0]), "cameras": camera_counts}
    return points, colors, summary


def bounds_summary(points, np) -> dict[str, Any]:
    if points.shape[0] == 0:
        return {"min": None, "max": None}
    return {
        "min": points.min(axis=0).astype(float).tolist(),
        "max": points.max(axis=0).astype(float).tolist(),
    }


def main() -> None:
    args = parse_args()
    cv2, np, o3d = load_runtime_deps()

    if args.view_existing is not None:
        if o3d is None:
            raise SystemExit("Open3D is required for --view-existing.")
        pcd = o3d.io.read_point_cloud(str(args.view_existing))
        if pcd.is_empty():
            raise RuntimeError(f"Point cloud is empty or unreadable: {args.view_existing}")
        view_point_cloud(pcd, args, o3d, title=f"RGB-D point cloud: {args.view_existing.name}")
        return

    dataset_dir = args.dataset_dir
    input_path = dataset_dir / args.input_json
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input JSON not found: {input_path}. "
            "Run pointcloud/tools/add_rgbd_extrinsics.py first."
        )

    episode = read_json(input_path)
    reconstruction_meta = episode.get("pointcloud_reconstruction") or {}
    camera_static = reconstruction_meta.get("cameras") or {}
    if not camera_static:
        raise ValueError(
            f"{input_path} does not contain pointcloud_reconstruction.cameras. "
            "Run pointcloud/tools/add_rgbd_extrinsics.py first."
        )

    data = episode.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError(f"{input_path} does not contain a non-empty data list")

    indices = selected_indices(len(data), args)
    if not indices:
        raise ValueError("No frames selected")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    per_frame_dir = output_dir / "frames"

    camera_names = args.cameras if args.cameras else None
    validate_requested_cameras(camera_static, camera_names)
    all_points = []
    all_colors = []
    frame_summaries = []

    for frame_number, data_index in enumerate(indices, start=1):
        frame = data[data_index]
        points, colors, frame_summary = reconstruct_frame(
            dataset_dir=dataset_dir,
            frame=frame,
            camera_static=camera_static,
            camera_names=camera_names,
            args=args,
            cv2=cv2,
            np=np,
        )
        frame_summary["data_index"] = data_index
        frame_summaries.append(frame_summary)

        if points.shape[0] == 0:
            print(f"[{frame_number}/{len(indices)}] frame {data_index}: no valid points")
            continue

        if args.save_per_frame:
            frame_points, frame_colors = voxel_downsample(points, colors, args.voxel_size, np, o3d)
            per_frame_path = per_frame_dir / f"frame_{data_index:06d}_fused.ply"
            write_point_cloud(per_frame_path, frame_points, frame_colors, np, o3d, args.ascii)

        all_points.append(points)
        all_colors.append(colors)
        print(f"[{frame_number}/{len(indices)}] frame {data_index}: {points.shape[0]} points")

    if not all_points:
        raise RuntimeError("No valid points were reconstructed from the selected frames")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    raw_points = int(points.shape[0])
    points, colors = voxel_downsample(points, colors, args.voxel_size, np, o3d)
    voxel_points = int(points.shape[0])
    points, colors = random_downsample(points, colors, args.max_points, np)

    output_path = output_dir / args.output_name
    write_point_cloud(output_path, points, colors, np, o3d, args.ascii)

    summary = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "created_by": "pointcloud/tools/reconstruct_rgbd_episode.py",
        "dataset_dir": dataset_dir.as_posix(),
        "input_json": input_path.as_posix(),
        "output_ply": output_path.as_posix(),
        "world_frame": reconstruction_meta.get("world_frame", "robot_base"),
        "camera_frame": reconstruction_meta.get("camera_frame"),
        "selected_data_indices": indices,
        "cameras": camera_names or "all",
        "depth_min_m": args.depth_min,
        "depth_max_m": args.depth_max,
        "stride": args.stride,
        "voxel_size_m": args.voxel_size,
        "max_points": args.max_points,
        "raw_points_before_downsample": raw_points,
        "points_after_voxel_downsample": voxel_points,
        "points_written": int(points.shape[0]),
        "bounds_robot_base_m": bounds_summary(points, np),
        "used_open3d": o3d is not None,
        "save_per_frame": bool(args.save_per_frame),
        "frame_summaries": frame_summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved point cloud: {output_path}")
    print(f"Saved summary: {summary_path}")
    print(
        f"Points: raw={raw_points}, after_voxel={voxel_points}, written={points.shape[0]}"
    )

    if args.view:
        if o3d is None:
            raise SystemExit("Open3D is required for --view.")
        view_point_cloud(
            make_o3d_cloud(points, colors, o3d),
            args,
            o3d,
            title=f"RGB-D point cloud: {output_path.name}",
        )


if __name__ == "__main__":
    main()
