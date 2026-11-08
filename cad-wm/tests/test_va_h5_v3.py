from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from data_process.tool.VA_h5_v3 import (
    VAH5Dataset,
    build_conversion_spec,
    homogeneous_pose_to_xyz_quat_xyzw,
)


def _shape_meta() -> dict:
    return {
        "fps": 25,
        "master_timestamp_path": "cameras/wrist/timestamp_us",
        "master_timeline": {
            "max_gap_s": 0.0001,
            "store_timestamps": True,
        },
        "features": {
            "observation.images.wrist": {
                "dtype": "video",
                "shape": [1, 1, 3],
                "h5_path": "cameras/wrist/frames",
                "align": "index",
            },
            "observation.images.side": {
                "dtype": "video",
                "shape": [1, 1, 3],
                "h5_path": "cameras/side/frames",
                "align": "index",
            },
            "observation.joint": {
                "dtype": "float32",
                "shape": [1],
                "h5_path": "teleop/q",
                "timestamp_path": "teleop/timestamp_us",
                "align": "previous",
            },
            "action.joint": {
                "dtype": "float32",
                "shape": [1],
                "h5_path": "teleop/q",
                "timestamp_path": "teleop/timestamp_us",
                "align": "previous",
            },
            "action.ee_pose": {
                "dtype": "float32",
                "shape": [7],
                "h5_path": "teleop/ee_pose",
                "timestamp_path": "teleop/timestamp_us",
                "align": "previous",
                "transform": "ee_pose_matrix_to_quaternion",
            },
        },
    }


def _write_episode(
    path: Path,
    *,
    side_rows: int = 3,
    master_timestamps: tuple[int, ...] = (100, 200, 300),
) -> None:
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset(
            "cameras/wrist/timestamp_us", data=np.asarray(master_timestamps)
        )
        h5_file.create_dataset(
            "cameras/wrist/frames",
            data=(
                np.asarray([1, 2, 3], dtype=np.uint8)
                .reshape(3, 1, 1, 1)
                .repeat(3, axis=-1)
            ),
        )
        h5_file.create_dataset(
            "cameras/side/frames",
            data=(
                np.arange(10, 10 + side_rows, dtype=np.uint8)
                .reshape(side_rows, 1, 1, 1)
                .repeat(3, axis=-1)
            ),
        )
        h5_file.create_dataset(
            "teleop/timestamp_us", data=np.asarray([50, 120, 190, 250, 310])
        )
        h5_file.create_dataset(
            "teleop/q",
            data=np.asarray([[0], [1], [2], [3], [4]], dtype=np.float32),
        )
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], 5, axis=0)
        poses[:, 0, 3] = np.arange(5, dtype=np.float32)
        h5_file.create_dataset("teleop/ee_pose", data=poses)


def test_homogeneous_pose_conversion_is_xyz_quaternion_xyzw() -> None:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    pose[:3, :3] = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    result = homogeneous_pose_to_xyz_quat_xyzw(pose, np)

    assert result.shape == (7,)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        result[3:], [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], atol=1.0e-6
    )


def test_homogeneous_pose_conversion_rejects_nonstandard_bottom_row() -> None:
    pose = np.eye(4, dtype=np.float64)
    pose[3, 0] = 1.0

    with pytest.raises(ValueError, match="bottom row"):
        homogeneous_pose_to_xyz_quat_xyzw(pose, np)


def test_camera_rows_are_indexed_and_numeric_values_are_causal(tmp_path: Path) -> None:
    h5_path = tmp_path / "episode.h5"
    _write_episode(h5_path)
    spec = build_conversion_spec(_shape_meta())
    dataset = VAH5Dataset(tmp_path, h5py=h5py, np=np)

    with dataset.open_episode(h5_path) as h5_file:
        cache = dataset.build_episode_cache(
            h5_file,
            spec["mappings"],
            spec["master_timestamp_path"],
            spec["fps"],
            h5_path,
            master_timeline=spec["master_timeline"],
        )
        frames = [
            dataset.read_frame(
                h5_file,
                index,
                spec["mappings"],
                h5_path,
                spec["master_timestamp_path"],
                cache,
            )
            for index in range(3)
        ]

    assert [
        int(frame["observation.images.wrist"][0, 0, 0]) for frame in frames
    ] == [1, 2, 3]
    assert [
        int(frame["observation.images.side"][0, 0, 0]) for frame in frames
    ] == [10, 11, 12]
    np.testing.assert_array_equal(
        [frame["observation.joint"][0] for frame in frames], [0, 2, 3]
    )
    np.testing.assert_array_equal(
        [frame["action.joint"][0] for frame in frames], [0, 2, 3]
    )
    np.testing.assert_array_equal(
        [frame["action.ee_pose"][0] for frame in frames], [0, 2, 3]
    )
    np.testing.assert_array_equal(
        frames[0]["action.ee_pose"][3:], [0, 0, 0, 1]
    )
    np.testing.assert_array_equal(
        [frame["timing.master_timestamp_ns"][0] for frame in frames],
        [100_000, 200_000, 300_000],
    )


def test_action_chunk_shape_is_rejected() -> None:
    config = _shape_meta()
    config["features"]["action.joint"]["shape"] = [8, 1]

    with pytest.raises(ValueError, match=r"one \[Da\] vector"):
        build_conversion_spec(config)


def test_unmatched_terminal_camera_row_is_truncated_by_index(tmp_path: Path) -> None:
    h5_path = tmp_path / "episode.h5"
    _write_episode(h5_path, side_rows=2)
    spec = build_conversion_spec(_shape_meta())
    dataset = VAH5Dataset(tmp_path, h5py=h5py, np=np)

    with dataset.open_episode(h5_path) as h5_file:
        cache = dataset.build_episode_cache(
            h5_file,
            spec["mappings"],
            spec["master_timestamp_path"],
            spec["fps"],
            h5_path,
            master_timeline=spec["master_timeline"],
        )

    np.testing.assert_array_equal(cache["selected_master_indices"], [0, 1])
    assert cache["dropped_media_tail_rows"] == 1


def test_leading_camera_row_without_history_is_dropped_without_index_shift(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "episode.h5"
    _write_episode(h5_path, master_timestamps=(10, 100, 200))
    spec = build_conversion_spec(_shape_meta())
    dataset = VAH5Dataset(tmp_path, h5py=h5py, np=np)

    with dataset.open_episode(h5_path) as h5_file:
        cache = dataset.build_episode_cache(
            h5_file,
            spec["mappings"],
            spec["master_timestamp_path"],
            spec["fps"],
            h5_path,
            master_timeline=spec["master_timeline"],
        )
        first = dataset.read_frame(
            h5_file,
            0,
            spec["mappings"],
            h5_path,
            spec["master_timestamp_path"],
            cache,
        )

    np.testing.assert_array_equal(cache["selected_master_indices"], [1, 2])
    assert int(first["observation.images.wrist"][0, 0, 0]) == 2
    assert int(first["observation.images.side"][0, 0, 0]) == 11
    assert float(first["action.joint"][0]) == 0.0
