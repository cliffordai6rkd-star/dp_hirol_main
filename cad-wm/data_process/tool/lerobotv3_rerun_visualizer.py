import os
import argparse
import subprocess
import pyarrow as pa
import pyarrow.parquet as pq

UID = os.getuid()
os.environ["XDG_CONFIG_HOME"] = f"/tmp/rerun-config-{UID}"
os.environ["XDG_DATA_HOME"] = f"/tmp/rerun-data-{UID}"
os.environ.setdefault("RUST_LOG", "error")
for path in (os.environ["XDG_CONFIG_HOME"], os.environ["XDG_DATA_HOME"]):
    os.makedirs(path, exist_ok=True)

try:
    subprocess.run(
        ["rerun", "analytics", "disable"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
except FileNotFoundError:
    pass

import rerun as rr
import numpy as np
import torch

from pathlib import Path
import json
import logging

from lerobot.datasets.lerobot_dataset import LeRobotDataset


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("lerobotv3_rerun_visualizer")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/train_episode/wipe_board_lbv3"),
        help="LeRobot dataset root to visualize.",
    )
    parser.add_argument(
        "--repo-id",
        default="wipe_board_lbv3",
        help="LeRobot repo id to visualize.",
    )
    parser.add_argument(
        "--video-backend",
        default="pyav",
        help="LeRobot video decoder backend (default: pyav).",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=None,
        help="First episode_index to review.",
    )
    parser.add_argument(
        "--end-episode",
        type=int,
        default=None,
        help="Last episode_index to review, inclusive.",
    )
    parser.add_argument(
        "--contact_add",
        action="store_true",
        help="Enable terminal contact interval annotation and write observation.contact_state.",
    )
    return parser.parse_args()


def feature_to_plain_dict(feature):
    if isinstance(feature, dict):
        return dict(feature)
    if hasattr(feature, "to_dict"):
        return dict(feature.to_dict())
    if hasattr(feature, "__dict__"):
        return dict(feature.__dict__)
    return {}


def feature_shapes_from_dataset(dataset):
    feature_shapes = {}
    for key, feature in dataset.features.items():
        spec = feature_to_plain_dict(feature)
        shape = spec.get("shape")
        if shape is not None:
            feature_shapes[key] = tuple(int(dim) for dim in shape)
    return feature_shapes


def build_rerun_blueprint():
    import rerun.blueprint as rrb

    feature_row_1 = rrb.Horizontal(
        rrb.TimeSeriesView(
            name="Wrench",
            origin="signals/features/wrench",
            contents="$origin/**",
        ),
        rrb.TimeSeriesView(
            name="Velocity",
            origin="signals/features/velocity",
            contents="$origin/**",
        ),
        rrb.TimeSeriesView(
            name="Acceleration",
            origin="signals/features/acceleration",
            contents="$origin/**",
        ),
        column_shares=[1, 1, 1],
    )
    feature_row_2 = rrb.Horizontal(
        rrb.TimeSeriesView(
            name="Torque",
            origin="signals/features/torque",
            contents="$origin/**",
        ),
        rrb.TimeSeriesView(
            name="EE Velocity",
            origin="signals/features/ee_velocity",
            contents="$origin/**",
        ),
        rrb.TimeSeriesView(
            name="EE Acceleration",
            origin="signals/features/ee_acceleration",
            contents="$origin/**",
        ),
        column_shares=[1, 1, 1],
    )
    feature_row_3 = rrb.Horizontal(
        rrb.TimeSeriesView(
            name="Joint Position",
            origin="signals/features/joint",
            contents="$origin/**",
        ),
        rrb.TimeSeriesView(
            name="Action Joint",
            origin="signals/features/action_joint",
            contents="$origin/**",
        ),
        column_shares=[1, 1],
    )

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Vertical(
                    feature_row_1,
                    feature_row_2,
                    feature_row_3,
                    row_shares=[1, 1, 1],
                ),
                rrb.Horizontal(
                    rrb.Spatial3DView(
                        name="EE Matrix",
                        origin="world/ee",
                        contents="$origin/**",
                    ),
                    rrb.Spatial3DView(
                        name="EE Quat7",
                        origin="world/ee_quat7",
                        contents="$origin/**",
                    ),
                    column_shares=[1, 1],
                ),
                row_shares=[2, 1],
            ),
            rrb.Vertical(
                rrb.Spatial2DView(
                    name="Current Wrist",
                    origin="current/cameras/wrist",
                    contents="$origin/**",
                ),
                rrb.Spatial2DView(
                    name="Current Side",
                    origin="current/cameras/side",
                    contents="$origin/**",
                ),
                rrb.Spatial2DView(
                    name="Current Side 2",
                    origin="current/cameras/side_2",
                    contents="$origin/**",
                ),
                row_shares=[1, 1, 1],
            ),
            column_shares=[2, 1],
        ),
        rrb.TimePanel(
            timeline="timestamp",
            state=rrb.PanelState.Collapsed,
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


class LeRobotv3Reader:
    def __init__(self, root, repo_id, video_backend="torchcodec"):
        self.root = Path(root)
        self.repo_id = repo_id
        self.video_backend = video_backend
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        return LeRobotDataset(
            repo_id=self.repo_id,
            root=self.root,
            video_backend=self.video_backend
        )

    def print_basic_info(self):
        print("root:", self.root)
        print("repo_id:", self.repo_id)
        print("len:", len(self.dataset))

    def print_features(self):
        print("features:")
        for key, spec in self.dataset.features.items():
            print(f"  {key}: {spec}")

    def print_sample(self, idx=0):
        sample = self.dataset[idx]

        print("sample idx:", idx)
        print("sample keys:", sample.keys())

        for key, value in sample.items():
            shape = getattr(value, "shape", None)
            dtype = getattr(value, "dtype", type(value))
            print(f"  {key}: shape={shape}, dtype={dtype}")

        return sample

    def _to_numpy(self, value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return value


class RerunLeRobotVisualizer:
    def __init__(self, reader, ee_pose_mode="matrix", log_force_window_detail=False, feature_shapes=None):
        self.reader = reader
        self.ee_pose_mode = ee_pose_mode
        self.log_force_window_detail = log_force_window_detail
        self.feature_shapes = feature_shapes or {}
        self.ee_trajectory = []
        self.ee_quat7_trajectory = []
        self.image_path_map = {
            "observation.images.wrist": "cameras/wrist",
            "observation.images.side": "cameras/side",
            "observation.images.side_1": "cameras/side",
            "observation.images.side_2": "cameras/side_2",
        }
        self.current_image_path_map = {
            "observation.images.wrist": "current/cameras/wrist",
            "observation.images.side": "current/cameras/side",
            "observation.images.side_1": "current/cameras/side",
            "observation.images.side_2": "current/cameras/side_2",
        }
        self.wrench_feature_keys = (
            "observation.wrench_ext",
            "observation.wrench_cal",
            "observation.ft_window",
        )
        self.ee_pose_feature_keys = (
            "observation.ee_state.ee_pose",
            "action.ee_pose",
        )
        self.force_names = ["fx", "fy", "fz", "tx", "ty", "tz"]
        self.gripper_path_map = {
            "observation.ee_state.gripper": "signals/gripper/observation",
            "action.gripper_state": "signals/gripper/action",
        }
        self.lowdim_feature_map = {
            "observation.joint": ("joint", [f"q{i}" for i in range(7)]),
            "observation.velocity": ("velocity", [f"v{i}" for i in range(7)]),
            "observation.acceleration": ("acceleration", [f"a{i}" for i in range(7)]),
            "observation.torque": ("torque", [f"tau{i}" for i in range(7)]),
            "observation.ee_velocity": ("ee_velocity", ["vx", "vy", "vz", "wx", "wy", "wz"]),
            "observation.ee_acceleration": ("ee_acceleration", ["ax", "ay", "az", "alphax", "alphay", "alphaz"]),
            "action.joint": ("action_joint", [f"q{i}" for i in range(7)]),
        }

    def reset_episode_state(self):
        self.ee_trajectory = []
        self.ee_quat7_trajectory = []

    def log_world_reference(self):
        rr.log(
            "world/origin",
            rr.Points3D([[0, 0, 0]], radii=0.01, colors=[[255, 255, 255]]),
        )
        rr.log(
            "world/reference_axes",
            rr.Arrows3D(
                origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                vectors=[[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )

    def log_frame(self, idx):
        sample = self.reader.dataset[idx]
        timestamp = float(sample["timestamp"])

        # Rerun stays on the dataset timestamp timeline; the annotation UI uses local_idx.
        rr.set_time("timestamp", duration=timestamp)

        self.log_images(sample)
        self.log_current_images(sample)
        self.log_force_window(sample)
        self.log_gripper(sample)
        self.log_lowdim_features(sample)
        self.log_ee_pose(sample)
        return timestamp

    def log_images(self, sample):
        for key, rerun_path in self.image_path_map.items():
            if key not in sample:
                continue

            image = self._image_to_uint8(sample[key])
            rr.log(rerun_path, rr.Image(image))

    def log_current_images(self, sample):
        for key, rerun_path in self.current_image_path_map.items():
            if key not in sample:
                continue

            image = self._image_to_uint8(sample[key])
            rr.log(rerun_path, rr.Image(image))

    def _image_to_uint8(self, value):
        image = self.reader._to_numpy(value)

        if image.ndim == 4:
            image = image[-1]

        # LeRobot 通常是 CHW，需要转 HWC 给 rerun
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.transpose(1, 2, 0)

        # LeRobot 读出来的图像一般是 float32 [0, 1]，Rerun 用 uint8 更直观。
        if np.issubdtype(image.dtype, np.floating):
            if image.size and float(np.nanmax(image)) > 1.0:
                image = image / 255.0
            image = np.clip(image, 0.0, 1.0)
            image = (image * 255.0).astype(np.uint8)

        return image

    def log_force_window(self, sample):
        key = next((key for key in self.wrench_feature_keys if key in sample), None)
        if key is None:
            return

        ft = np.asarray(self.reader._to_numpy(sample[key]))
        ft_shape = self.feature_shapes.get(key)

        # Final shape comes from the actual LeRobot dataset features:
        # [6] means current wrench, [window_size, 6] means a past window.
        current_ft = self._force_current(ft, ft_shape)
        if current_ft is None:
            return

        for ch_i in range(current_ft.shape[0]):
            name = self.force_names[ch_i] if ch_i < len(self.force_names) else f"ch_{ch_i}"
            value = float(current_ft[ch_i])
            rr.log(f"signals/features/wrench/{name}", rr.Scalars(value))

        if not self.log_force_window_detail:
            return

        ft_window = self._force_window_array(ft, ft_shape)
        if ft_window is None:
            return

        # 窗口细节：如果需要检查窗口方向/历史值，可在左侧树里打开 debug/force_window。
        for window_i in range(ft_window.shape[0]):
            for ch_i in range(ft_window.shape[1]):
                name = self.force_names[ch_i] if ch_i < len(self.force_names) else f"ch_{ch_i}"
                rr.log(
                    f"debug/force_window/{name}/past_{window_i}",
                    rr.Scalars(float(ft_window[window_i, ch_i])),
                )

    def _force_current(self, value, shape):
        ft = np.asarray(value, dtype=np.float64)
        if ft.dtype == object:
            ft = np.asarray(list(value), dtype=np.float64)

        if shape is not None and len(shape) == 1:
            return ft.reshape(-1)

        if shape is not None and len(shape) == 2:
            return ft.reshape(shape)[-1].reshape(-1)

        if ft.ndim == 1:
            return ft.reshape(-1)
        if ft.ndim == 2:
            return ft[-1].reshape(-1)
        return None

    def _force_window_array(self, value, shape):
        if shape is None or len(shape) != 2:
            return None
        ft = np.asarray(value, dtype=np.float64)
        if ft.dtype == object:
            ft = np.asarray(list(value), dtype=np.float64)
        return ft.reshape(shape)

    def log_gripper(self, sample):
        for key, rerun_path in self.gripper_path_map.items():
            if key not in sample:
                continue

            value = np.asarray(self.reader._to_numpy(sample[key])).reshape(-1)[0]
            rr.log(rerun_path, rr.Scalars(float(value)))

    def log_lowdim_features(self, sample):
        for key, (feature_name, channel_names) in self.lowdim_feature_map.items():
            if key not in sample:
                continue

            values = np.asarray(self.reader._to_numpy(sample[key]), dtype=np.float64).reshape(-1)
            for ch_i, value in enumerate(values):
                if ch_i < len(channel_names):
                    channel_name = channel_names[ch_i]
                else:
                    channel_name = f"ch_{ch_i}"
                rr.log(f"signals/features/{feature_name}/{channel_name}", rr.Scalars(float(value)))

    def log_ee_pose(self, sample):
        key = next((key for key in self.ee_pose_feature_keys if key in sample), None)
        if key is None:
            return

        pose = np.asarray(self.reader._to_numpy(sample[key]))
        mode = self.ee_pose_mode

        if mode in ("matrix", "both"):
            self._log_ee_pose_matrix(pose)

        if mode in ("quat7", "both"):
            self._log_ee_pose_quat7(pose)

    def _log_ee_pose_matrix(self, pose):
        # Accept both old [4, 4] matrix poses and converted quat7 poses:
        # [x, y, z, qx, qy, qz, qw].
        if pose.shape == (4, 4):
            # print(f"rotaion type : rotation matrix")
            translation = pose[:3, 3]
            rotation = pose[:3, :3]
        else:
            # print(f"rotaion type : quaternion")
            quat7 = self._pose_to_quat7(pose)
            if quat7 is None:
                return
            translation = quat7[:3]
            rotation = self._quat_xyzw_to_rotation_matrix(quat7[3:7])

        if translation is None or rotation is None:
            return

        self.ee_trajectory.append(translation.copy())

        rr.log(
            "world/ee/current_position",
            rr.Points3D([translation], radii=0.005),
        )

        rr.log(
            "world/ee/trajectory",
            rr.Points3D(
                np.asarray(self.ee_trajectory),
                radii=0.002,
                colors=[[255, 255, 255]],
            ),
        )
        if len(self.ee_trajectory) >= 2:
            rr.log(
                "world/ee/trajectory_line",
                rr.LineStrips3D(
                    [np.asarray(self.ee_trajectory)],
                    radii=0.003,
                    colors=[[255, 255, 255]],
                ),
            )

        # 额外画出 EE 坐标轴：x 红、y 绿、z 蓝。
        axis_length = 0.08
        origins = np.repeat(translation.reshape(1, 3), 3, axis=0)
        vectors = np.stack(
            [
                rotation[:, 0] * axis_length,
                rotation[:, 1] * axis_length,
                rotation[:, 2] * axis_length,
            ],
            axis=0,
        )
        rr.log(
            "world/ee/current_axes",
            rr.Arrows3D(
                origins=origins,
                vectors=vectors,
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )

    def _log_ee_pose_quat7(self, pose):
        # quat7 约定：[x, y, z, qx, qy, qz, qw]
        # 兼容旧数据: 如果当前数据仍是 4x4 矩阵，就先转换成 quat7。
        quat7 = self._pose_to_quat7(pose)
        if quat7 is None:
            return

        translation = quat7[:3]
        quat_xyzw = quat7[3:7]
        self.ee_quat7_trajectory.append(translation.copy())

        rr.log(
            "world/ee_quat7/current_position",
            rr.Points3D([translation], radii=0.005, colors=[[0, 0, 255]]),
        )

        rr.log(
            "world/ee_quat7/trajectory",
            rr.Points3D(
                np.asarray(self.ee_quat7_trajectory),
                radii=0.002,
                colors=[[255, 255, 255]],
            ),
            
        )
        if len(self.ee_quat7_trajectory) >= 2:
            rr.log(
                "world/ee_quat7/trajectory_line",
                rr.LineStrips3D(
                    [np.asarray(self.ee_quat7_trajectory)],
                    radii=0.003,
                    colors=[[0, 0, 255]],
                ),
            )

        rr.log(
            "world/ee_quat7/frame",
            rr.Transform3D(
                translation=translation,
                rotation=rr.Quaternion(xyzw=quat_xyzw),
            ),
        )

    def _matrix_to_quat7(self, pose):
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        quat_xyzw = self._rotation_matrix_to_quat_xyzw(rotation)
        return np.concatenate([translation, quat_xyzw])

    def _pose_to_quat7(self, pose):
        pose = np.asarray(pose)
        if pose.shape == (4, 4):
            return self._matrix_to_quat7(pose)

        quat7 = pose.reshape(-1).astype(np.float64)
        if quat7.shape[0] < 7:
            return None

        quat7 = quat7[:7]
        quat = quat7[3:7]
        norm = np.linalg.norm(quat)
        if norm > 0:
            quat7[3:7] = quat / norm
        if quat7[6] < 0:
            quat7[3:7] = -quat7[3:7]
        return quat7

    def _quat_xyzw_to_rotation_matrix(self, quat_xyzw):
        quat = np.asarray(quat_xyzw, dtype=np.float64)
        norm = np.linalg.norm(quat)
        if norm == 0:
            return np.eye(3, dtype=np.float64)
        qx, qy, qz, qw = quat / norm

        return np.array(
            [
                [
                    1 - 2 * (qy * qy + qz * qz),
                    2 * (qx * qy - qz * qw),
                    2 * (qx * qz + qy * qw),
                ],
                [
                    2 * (qx * qy + qz * qw),
                    1 - 2 * (qx * qx + qz * qz),
                    2 * (qy * qz - qx * qw),
                ],
                [
                    2 * (qx * qz - qy * qw),
                    2 * (qy * qz + qx * qw),
                    1 - 2 * (qx * qx + qy * qy),
                ],
            ],
            dtype=np.float64,
        )

    def _rotation_matrix_to_quat_xyzw(self, matrix):
        # 返回顺序为 xyzw，和 Rerun rr.Quaternion(xyzw=...) 对齐。
        m = np.asarray(matrix, dtype=np.float64)
        trace = np.trace(m)

        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s

        quat = np.array([qx, qy, qz, qw], dtype=np.float64)
        norm = np.linalg.norm(quat)
        if norm > 0:
            quat = quat / norm
        if quat[3] < 0:
            quat = -quat
        return quat


def find_episode_parquet(root, episode_index):
    root = Path(root)
    episode_index = int(episode_index)

    parquet_paths = sorted((root / "data").glob("**/*.parquet"))
    matched_paths = []

    for parquet_path in parquet_paths:
        try:
            table = pq.read_table(parquet_path, columns=["episode_index"])
        except Exception as exc:
            print(f"Skip parquet without episode_index: {parquet_path}, error={exc}")
            continue

        if (table["episode_index"].to_numpy() == episode_index).any():
            matched_paths.append(parquet_path)

    if len(matched_paths) != 1:
        raise FileNotFoundError(
            f"expected one parquet containing episode {episode_index}, "
            f"found {len(matched_paths)}: {matched_paths}"
        )

    return matched_paths[0]


def write_contact_state_to_episode(root, episode_index, contact_state):
    root = Path(root)
    episode_index = int(episode_index)

    parquet_path = find_episode_parquet(root, episode_index)

    table = pq.read_table(parquet_path)

    if "episode_index" not in table.column_names:
        raise KeyError(f"episode_index column not found in {parquet_path}")

    episode_indices = table["episode_index"].to_numpy()
    mask = episode_indices == episode_index
    row_count = int(mask.sum())

    if row_count != contact_state.shape[0]:
        raise ValueError(
            f"contact_state length {contact_state.shape[0]} does not match "
            f"episode {episode_index} rows {row_count} in {parquet_path}"
        )

    contact_values = contact_state.detach().cpu().numpy().astype(np.float32).reshape(-1)

    full_contact = np.zeros((table.num_rows,), dtype=np.float32)

    # 如果原本已经有 contact_state，则先保留其他 episode 的旧标注
    col_name = "observation.contact_state"
    if col_name in table.column_names:
        old_values = table[col_name].to_pylist()
        restored = []
        for x in old_values:
            if isinstance(x, (list, tuple)):
                restored.append(float(x[0]))
            elif x is None:
                restored.append(0.0)
            else:
                restored.append(float(x))
        full_contact[:] = np.asarray(restored, dtype=np.float32)

    # 只覆盖当前 episode
    full_contact[mask] = contact_values

    # 关键：写成 float32 标量列，不写成 list[1]
    contact_column = pa.array(
        full_contact.astype(np.float32),
        type=pa.float32(),
    )

    if col_name in table.column_names:
        col_idx = table.column_names.index(col_name)
        table = table.set_column(col_idx, col_name, contact_column)
    else:
        table = table.append_column(col_name, contact_column)

    pq.write_table(table, parquet_path)

    print(
        f"Wrote {col_name} for episode {episode_index} "
        f"to {parquet_path}, rows={row_count}"
    )

def ensure_contact_state_feature(root):
    info_path = Path(root) / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as file:
        info = json.load(file)

    features = info.setdefault("features", {})
    features["observation.contact_state"] = {
        "dtype": "float32",
        "shape": [1],
        "names": None,
    }

    with info_path.open("w", encoding="utf-8") as file:
        json.dump(info, file, ensure_ascii=False, indent=2)


def nearest_timestamp_index(timestamps, value):
    timestamps = np.asarray(timestamps, dtype=np.float64)
    return int(np.argmin(np.abs(timestamps - float(value))))


def parse_contact_interval(text):
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError("please input exactly two timestamps: start_timestamp end_timestamp")
    return float(parts[0]), float(parts[1])


def build_contact_state_from_intervals(episode_len, intervals):
    contact_state = torch.zeros((episode_len, 1), dtype=torch.float32)
    for start, end in intervals:
        contact_state[start:end + 1] = 1.0
    return contact_state


def run_terminal_contact_annotation(reader, episode_index, timestamps):
    intervals = []
    episode_len = len(timestamps)
    if episode_len == 0:
        raise ValueError(f"episode {episode_index} has no timestamps")

    start_t = float(timestamps[0])
    end_t = float(timestamps[-1])
    log.info(
        "episode %s contact annotation: input 'start_timestamp end_timestamp'; "
        "empty Enter saves and moves to next. timestamp range=[%.6f, %.6f]",
        episode_index,
        start_t,
        end_t,
    )

    while True:
        text = input(f"episode {episode_index} contact interval> ").strip()
        if not text:
            break
        try:
            interval_start_t, interval_end_t = parse_contact_interval(text)
        except ValueError as exc:
            log.info("invalid input: %s", exc)
            continue

        start = nearest_timestamp_index(timestamps, interval_start_t)
        end = nearest_timestamp_index(timestamps, interval_end_t)
        if start > end:
            start, end = end, start
        intervals.append((start, end))
        log.info(
            "added contact interval episode=%s timestamps=[%.6f, %.6f] frames=[%d, %d]",
            episode_index,
            interval_start_t,
            interval_end_t,
            start,
            end,
        )

    contact_state = build_contact_state_from_intervals(episode_len, intervals)
    ensure_contact_state_feature(reader.root)
    write_contact_state_to_episode(reader.root, episode_index, contact_state)
    log.info("episode %s annotation finished: intervals=%s", episode_index, intervals)
    return intervals


def close_rerun_ui():
    rr.disconnect()


def run_episode_review(
    reader,
    visualizer,
    blueprint,
    contact_add=False,
    start_episode=None,
    end_episode=None,
):
    episodes = reader.dataset.meta.episodes

    print("Episode review controls:")
    print("  Rerun will play each episode automatically.")
    if contact_add:
        print("  Use Rerun's timeline to inspect contact intervals.")
        print("  In terminal, input: start_timestamp end_timestamp")
        print("  Press Enter on an empty line to save contact_state and move to the next episode.")
    else:
        print("  Contact annotation is disabled. Press Enter to move to the next episode.")
        print("  Pass --contact_add to enable contact_state annotation.")

    for episode_info in episodes:
        episode_index = int(episode_info["episode_index"])
        if start_episode is not None and episode_index < start_episode:
            continue
        if end_episode is not None and episode_index > end_episode:
            continue

        start_idx = int(episode_info["dataset_from_index"])
        end_idx = int(episode_info["dataset_to_index"])
        length = int(episode_info["length"])

        visualizer.reset_episode_state()
        close_rerun_ui()

        rr.init(
            "lerobot_wipe_board",
            recording_id=f"episode_{episode_index:06d}",
            spawn=False,
            default_blueprint=blueprint,
        )
        rr.spawn(default_blueprint=blueprint, hide_welcome_screen=True)
        rr.send_blueprint(blueprint)
        visualizer.log_world_reference()

        try:
            print(f"\nPlaying episode {episode_index} [{start_idx}, {end_idx}) length={length}")

            timestamps = []
            for idx in range(start_idx, end_idx):
                timestamps.append(visualizer.log_frame(idx))

            if not contact_add:
                input(f"Episode {episode_index} played. Press Enter for next episode...")
                continue

            run_terminal_contact_annotation(reader, episode_index, timestamps)
            log.info("moving to next episode")

        finally:
            close_rerun_ui()


if __name__ == "__main__":
    import rerun as rr
    args = parse_args()

    EE_POSE_MODE = "both"  # 可选："matrix"、"quat7"、"both"
    LOG_FORCE_WINDOW_DETAIL = False

    reader = LeRobotv3Reader(
        root=args.root,
        repo_id=args.repo_id,
        video_backend=args.video_backend,
    )
    feature_shapes = feature_shapes_from_dataset(reader.dataset)

    blueprint = build_rerun_blueprint()
    visualizer = RerunLeRobotVisualizer(
        reader,
        ee_pose_mode=EE_POSE_MODE,
        log_force_window_detail=LOG_FORCE_WINDOW_DETAIL,
        feature_shapes=feature_shapes,
    )
    run_episode_review(
        reader=reader,
        visualizer=visualizer,
        blueprint=blueprint,
        contact_add=args.contact_add,
        start_episode=args.start_episode,
        end_episode=args.end_episode,
    )
