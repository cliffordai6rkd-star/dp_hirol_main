
# python dataset/tool/lerobot_add_feature.py \
#   --input-root data/train_episode/wrench_background/wrench_bg_lerobotv3 \
#   --input-repo-id wrench_bg_lerobotv3 \
#   --features acceleration ee_velocity ee_acceleration \


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from tqdm import tqdm


FEATURE_CHOICES = ("acceleration", "ee_velocity", "ee_acceleration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append selected lowdim derivative features to an existing LeRobot dataset."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Source LeRobot dataset root.")
    parser.add_argument("--input-repo-id", required=True, help="Source LeRobot repo id.")
    parser.add_argument(
        "--features",
        nargs="+",
        choices=FEATURE_CHOICES,
        default=list(FEATURE_CHOICES),
        help="Features to add. Default: add all supported derivative features.",
    )
    parser.add_argument("--velocity-key", default="observation.velocity")
    parser.add_argument("--acceleration-key", default="observation.acceleration")
    parser.add_argument("--ee-pose-key", default="action.ee_pose")
    parser.add_argument("--ee-velocity-key", default="observation.ee_velocity")
    parser.add_argument("--ee-acceleration-key", default="observation.ee_acceleration")
    parser.add_argument("--timestamp-key", default="timestamp")
    parser.add_argument("--min-dt", type=float, default=1e-6, help="Lower bound for timestamp deltas.")
    parser.add_argument("--video-backend", default="torchcodec")
    return parser.parse_args()


def feature_to_plain_dict(feature: Any) -> dict[str, Any]:
    if isinstance(feature, dict):
        return dict(feature)
    if hasattr(feature, "to_dict"):
        return dict(feature.to_dict())
    if hasattr(feature, "__dict__"):
        return dict(feature.__dict__)
    raise TypeError(f"Unsupported feature spec type: {type(feature).__name__}")


def added_feature_specs(
    source_dataset: Any,
    *,
    selected_features: set[str],
    velocity_key: str,
    acceleration_key: str,
    ee_pose_key: str,
    ee_velocity_key: str,
    ee_acceleration_key: str,
) -> dict[str, Any]:
    features = {}
    if "acceleration" in selected_features:
        features[acceleration_key] = derivative_feature_spec(source_dataset, velocity_key)
    if "ee_velocity" in selected_features:
        features[ee_velocity_key] = ee_twist_feature_spec(source_dataset, ee_pose_key)
    if "ee_acceleration" in selected_features:
        features[ee_acceleration_key] = ee_twist_feature_spec(source_dataset, ee_pose_key)
    return features


def derivative_feature_spec(source_dataset: Any, source_key: str) -> dict[str, Any]:
    if source_key not in source_dataset.features:
        raise KeyError(f"missing source feature spec: {source_key}")
    spec = feature_to_plain_dict(source_dataset.features[source_key])
    if "shape" not in spec:
        raise KeyError(f"missing shape in source feature spec: {source_key}")
    return {
        "dtype": "float32",
        "shape": tuple(spec["shape"]),
    }


def ee_twist_feature_spec(source_dataset: Any, source_key: str) -> dict[str, Any]:
    if source_key not in source_dataset.features:
        raise KeyError(f"missing source feature spec: {source_key}")
    spec = feature_to_plain_dict(source_dataset.features[source_key])
    shape = tuple(spec.get("shape", ()))
    if shape != (4, 4):
        raise ValueError(f"{source_key} must have shape (4, 4) to compute ee twist, got {shape}.")
    return {
        "dtype": "float32",
        "shape": (6,),
    }


def compute_acceleration(
    velocity: Any,
    previous_velocity: Any | None,
    dt: float,
) -> Any:
    import torch

    if not torch.is_tensor(velocity):
        velocity = torch.as_tensor(velocity)
    velocity = velocity.to(dtype=torch.float32)
    if previous_velocity is None:
        return torch.zeros_like(velocity)
    if not torch.is_tensor(previous_velocity):
        previous_velocity = torch.as_tensor(previous_velocity)
    return (velocity - previous_velocity.to(dtype=torch.float32)) / dt


def scalar_timestamp(value: Any) -> float:
    import torch

    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def compute_episode_accelerations(
    velocities: list[Any],
    *,
    timestamps: list[Any],
    min_dt: float,
) -> list[Any]:
    import torch

    if not velocities:
        return []
    if len(velocities) != len(timestamps):
        raise ValueError("velocities and timestamps must have the same length.")

    accelerations = [torch.zeros_like(torch.as_tensor(velocities[0], dtype=torch.float32))]
    for idx in range(1, len(velocities)):
        dt = scalar_timestamp(timestamps[idx]) - scalar_timestamp(timestamps[idx - 1])
        if dt < min_dt:
            dt = min_dt
        accelerations.append(compute_acceleration(velocities[idx], velocities[idx - 1], dt))

    if len(accelerations) > 1:
        accelerations[0] = accelerations[1].clone()
    return accelerations


def ee_pose_value(ee_pose: Any) -> Any:
    import torch

    if not torch.is_tensor(ee_pose):
        ee_pose = torch.as_tensor(ee_pose)
    ee_pose = ee_pose.to(dtype=torch.float32)
    if tuple(ee_pose.shape) != (4, 4):
        raise ValueError(f"ee pose must have shape (4, 4), got {tuple(ee_pose.shape)}.")
    return ee_pose


def so3_log(rotation: Any) -> Any:
    import torch

    rotation = torch.as_tensor(rotation, dtype=torch.float32)
    trace = torch.trace(rotation)
    cos_theta = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    vee = torch.stack(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
    )

    sin_theta = torch.sin(theta)
    if torch.abs(sin_theta) < 1e-6:
        return 0.5 * vee
    return theta / (2.0 * sin_theta) * vee


def ee_velocity_from_poses(current_pose: Any, previous_pose: Any, dt: float) -> Any:
    import torch

    current_pose = ee_pose_value(current_pose)
    previous_pose = ee_pose_value(previous_pose)
    linear_velocity = (current_pose[:3, 3] - previous_pose[:3, 3]) / dt

    # Spatial angular velocity in the same frame as the pose translation.
    rotation_delta = current_pose[:3, :3] @ previous_pose[:3, :3].transpose(0, 1)
    angular_velocity = so3_log(rotation_delta) / dt
    return torch.cat((linear_velocity, angular_velocity), dim=0)


def compute_episode_ee_velocities(
    ee_poses: list[Any],
    *,
    timestamps: list[Any],
    min_dt: float,
) -> list[Any]:
    import torch

    if not ee_poses:
        return []
    if len(ee_poses) != len(timestamps):
        raise ValueError("ee_poses and timestamps must have the same length.")

    velocities = [torch.zeros(6, dtype=torch.float32)]
    for idx in range(1, len(ee_poses)):
        dt = scalar_timestamp(timestamps[idx]) - scalar_timestamp(timestamps[idx - 1])
        if dt < min_dt:
            dt = min_dt
        velocities.append(ee_velocity_from_poses(ee_poses[idx], ee_poses[idx - 1], dt))

    if len(velocities) > 1:
        velocities[0] = velocities[1].clone()
    return velocities


def compute_episode_derivative(values: list[Any], *, timestamps: list[Any], min_dt: float) -> list[Any]:
    import torch

    if not values:
        return []
    if len(values) != len(timestamps):
        raise ValueError("values and timestamps must have the same length.")

    derivatives = [torch.zeros_like(torch.as_tensor(values[0], dtype=torch.float32))]
    for idx in range(1, len(values)):
        dt = scalar_timestamp(timestamps[idx]) - scalar_timestamp(timestamps[idx - 1])
        if dt < min_dt:
            dt = min_dt
        current = torch.as_tensor(values[idx], dtype=torch.float32)
        previous = torch.as_tensor(values[idx - 1], dtype=torch.float32)
        derivatives.append((current - previous) / dt)

    if len(derivatives) > 1:
        derivatives[0] = derivatives[1].clone()
    return derivatives


def data_parquet_files(root: Path) -> list[Path]:
    files = sorted((root / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")
    return files


def tensor_to_list(value: Any) -> list[float]:
    import torch

    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value.detach().cpu().to(dtype=torch.float32).reshape(-1).tolist()


def compute_stats(values: list[Any]) -> dict[str, Any]:
    import numpy as np

    array = np.asarray([tensor_to_list(value) for value in values], dtype=np.float64)
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [int(array.shape[0])] * int(array.shape[1]),
        "q01": np.quantile(array, 0.01, axis=0).tolist(),
        "q10": np.quantile(array, 0.10, axis=0).tolist(),
        "q50": np.quantile(array, 0.50, axis=0).tolist(),
        "q90": np.quantile(array, 0.90, axis=0).tolist(),
        "q99": np.quantile(array, 0.99, axis=0).tolist(),
    }


def feature_arrow_type(feature: Mapping[str, Any]) -> Any:
    import pyarrow as pa

    shape = tuple(feature.get("shape", ()))
    if len(shape) != 1:
        raise ValueError(f"in-place add_feature only supports 1D lowdim features, got shape={shape}")
    return pa.list_(pa.float32(), list_size=int(shape[0]))


def huggingface_feature_metadata(feature: Mapping[str, Any]) -> dict[str, Any]:
    shape = tuple(feature.get("shape", ()))
    if len(shape) != 1:
        raise ValueError(f"in-place add_feature only supports 1D lowdim features, got shape={shape}")
    return {
        "feature": {
            "dtype": "float32",
            "_type": "Value",
        },
        "length": int(shape[0]),
        "_type": "Sequence",
    }


def update_huggingface_schema_metadata(table: Any, features: Mapping[str, Mapping[str, Any]]) -> Any:
    metadata = dict(table.schema.metadata or {})
    raw = metadata.get(b"huggingface")
    if raw is None:
        return table

    try:
        payload = json.loads(raw.decode("utf-8"))
        hf_features = payload["info"]["features"]
    except (KeyError, TypeError, ValueError):
        return table

    for key, feature in features.items():
        hf_features[key] = huggingface_feature_metadata(feature)

    metadata[b"huggingface"] = json.dumps(payload).encode("utf-8")
    return table.replace_schema_metadata(metadata)


def replace_or_append_column(table: Any, name: str, array: Any) -> Any:
    if name in table.column_names:
        col_idx = table.column_names.index(name)
        return table.set_column(col_idx, name, array)
    return table.append_column(name, array)


def update_meta_files_in_place(root: Path, features: Mapping[str, Mapping[str, Any]], stats: Mapping[str, Any]) -> None:
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"
    if not info_path.exists():
        raise FileNotFoundError(f"missing {info_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"missing {stats_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    info_features = info.setdefault("features", {})
    for key, feature in features.items():
        info_features[key] = {
            "dtype": "float32",
            "shape": list(feature["shape"]),
            "names": feature.get("names"),
        }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=4), encoding="utf-8")

    stats_data = json.loads(stats_path.read_text(encoding="utf-8"))
    for key, value in stats.items():
        stats_data[key] = value
    stats_path.write_text(json.dumps(stats_data, ensure_ascii=False, indent=4), encoding="utf-8")


def compute_added_features_by_index(
    source_dataset: Any,
    episodes: list[Mapping[str, Any]],
    selected_features: set[str],
    args: argparse.Namespace,
) -> dict[str, list[Any]]:
    added_keys = computed_key_names(args, selected_features)
    values_by_key = {key: [None] * len(source_dataset) for key in added_keys}

    episode_iter = tqdm(episodes, desc="episodes", unit="episode")
    for episode in episode_iter:
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        source_frames = [source_dataset.hf_dataset[raw_idx] for raw_idx in range(start, end)]

        velocities = []
        ee_poses = []
        timestamps = []
        for source_frame in source_frames:
            if "acceleration" in selected_features and args.velocity_key not in source_frame:
                raise KeyError(f"missing velocity key in source frame: {args.velocity_key}")
            if (
                {"ee_velocity", "ee_acceleration"} & selected_features
                and args.ee_pose_key not in source_frame
            ):
                raise KeyError(f"missing ee pose key in source frame: {args.ee_pose_key}")
            if args.timestamp_key not in source_frame:
                raise KeyError(f"missing timestamp key in source frame: {args.timestamp_key}")
            if "acceleration" in selected_features:
                velocities.append(source_frame[args.velocity_key])
            if {"ee_velocity", "ee_acceleration"} & selected_features:
                ee_poses.append(ee_pose_value(source_frame[args.ee_pose_key]))
            timestamps.append(source_frame[args.timestamp_key])

        computed_features: dict[str, list[Any]] = {}
        if "acceleration" in selected_features:
            computed_features[args.acceleration_key] = compute_episode_accelerations(
                velocities,
                timestamps=timestamps,
                min_dt=args.min_dt,
            )
        if "ee_velocity" in selected_features or "ee_acceleration" in selected_features:
            ee_velocities = compute_episode_ee_velocities(
                ee_poses,
                timestamps=timestamps,
                min_dt=args.min_dt,
            )
            if "ee_velocity" in selected_features:
                computed_features[args.ee_velocity_key] = ee_velocities
            if "ee_acceleration" in selected_features:
                computed_features[args.ee_acceleration_key] = compute_episode_derivative(
                    ee_velocities,
                    timestamps=timestamps,
                    min_dt=args.min_dt,
                )

        for frame_offset, raw_idx in enumerate(range(start, end)):
            for key, values in computed_features.items():
                values_by_key[key][raw_idx] = values[frame_offset]

    return values_by_key


def run_in_place(source_dataset: Any, features: Mapping[str, Mapping[str, Any]], episodes, args) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    selected_features = set(args.features)
    values_by_key = compute_added_features_by_index(source_dataset, list(episodes), selected_features, args)
    stats = {key: compute_stats(values) for key, values in values_by_key.items()}

    for parquet_path in tqdm(data_parquet_files(args.input_root), desc="parquet", unit="file"):
        table = pq.read_table(parquet_path)
        if "index" not in table.column_names:
            raise KeyError(f"'index' column is required for in-place update: {parquet_path}")
        row_indices = [int(index) for index in table["index"].to_pylist()]
        updated = table

        for key, feature in features.items():
            column_values = []
            for raw_idx in row_indices:
                value = values_by_key[key][raw_idx]
                if value is None:
                    raise ValueError(f"missing computed value for {key} at dataset index {raw_idx}")
                column_values.append(tensor_to_list(value))
            array = pa.array(column_values, type=feature_arrow_type(feature))
            updated = replace_or_append_column(updated, key, array)

        updated = update_huggingface_schema_metadata(updated, features)
        pq.write_table(updated, parquet_path)

    update_meta_files_in_place(args.input_root, features, stats)
    print(f"updated dataset in-place: root={args.input_root} repo_id={args.input_repo_id}")
    print(
        "added keys: "
        f"{', '.join(computed_key_names(args, selected_features))}, "
        f"timestamp_key={args.timestamp_key}"
    )


def main() -> None:
    args = parse_args()

    if args.min_dt <= 0:
        raise ValueError("--min-dt must be positive.")
    selected_features = set(args.features)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source_dataset = LeRobotDataset(
        repo_id=args.input_repo_id,
        root=args.input_root,
        video_backend=args.video_backend,
    )

    features = added_feature_specs(
        source_dataset,
        selected_features=selected_features,
        velocity_key=args.velocity_key,
        acceleration_key=args.acceleration_key,
        ee_pose_key=args.ee_pose_key,
        ee_velocity_key=args.ee_velocity_key,
        ee_acceleration_key=args.ee_acceleration_key,
    )
    run_in_place(source_dataset, features, source_dataset.meta.episodes, args)


def computed_key_names(args: argparse.Namespace, selected_features: set[str]) -> list[str]:
    keys = []
    if "acceleration" in selected_features:
        keys.append(args.acceleration_key)
    if "ee_velocity" in selected_features:
        keys.append(args.ee_velocity_key)
    if "ee_acceleration" in selected_features:
        keys.append(args.ee_acceleration_key)
    return keys


if __name__ == "__main__":
    main()
