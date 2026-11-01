from argparse import Namespace
import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.h5_2_lerobotv3 import (
    EpisodeSource, _feature_specs, _load_episode, _parse_args,
    _resolve_source_settings, _rotvec_to_quat, _timestamp_ns,
)


def libero_source(tmp_path):
    path = tmp_path / "drawer.hdf5"
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        data.attrs["env_args"] = '{"env_kwargs": {"control_freq": 20}}'
        demo = data.create_group("demo_0")
        demo.create_dataset("states", data=np.column_stack((0.25 + np.arange(4) * 0.05, np.zeros((4, 50)))))
        demo.create_dataset("actions", data=np.full((4, 7), 0.5))
        demo.create_dataset("obs/joint_states", data=np.zeros((4, 7)))
        demo.create_dataset("obs/dq", data=np.zeros((4, 24)))
        demo.create_dataset("obs/tau", data=np.full((4, 24), 1e5))
        demo.create_dataset("obs/ee_states", data=np.zeros((4, 6)))
        demo.create_dataset("obs/wrenches", data=np.ones((4, 6)))
        for key in ("obs/eye_in_hand_rgb", "obs/agentview_rgb"):
            demo.create_dataset(key, data=np.zeros((4, 8, 8, 3), dtype=np.uint8))
    return EpisodeSource(path, "data/demo_0", 0)


def test_rotvec_conversion_matches_xyzw_rotation_reference():
    rotvecs = np.array([[0, 0, 0], [np.pi, 0, 0], [0.3, -0.4, 0.2]])
    with np.errstate(all="raise"):
        actual = _rotvec_to_quat(rotvecs)
    np.testing.assert_allclose(actual, Rotation.from_rotvec(rotvecs).as_quat(), atol=1e-7)


def test_libero_uses_source_clock_and_omits_invalid_joint_fields(tmp_path, monkeypatch):
    source = libero_source(tmp_path)
    monkeypatch.setattr("sys.argv", ["converter", "--input", str(source.path), "--output", str(tmp_path / "out"), "--allow-vector-truncate"])
    args = _parse_args()
    _resolve_source_settings([source], args)
    assert args.source_format == "libero"
    assert args.fps == args.camera_fps == 20
    args.image_height = args.image_width = None
    with h5py.File(source.path, "r") as file:
        arrays = _load_episode(file[source.group_name], args)
    np.testing.assert_array_equal(arrays.state_timestamp_ns, [250_000_000, 300_000_000, 350_000_000, 400_000_000])
    np.testing.assert_array_equal(arrays.camera_timestamp_ns, arrays.state_timestamp_ns)
    np.testing.assert_array_equal(arrays.ee_pose[:, 3:], np.tile([0, 0, 0, 1], (4, 1)))
    specs = _feature_specs(arrays.wrist_images, arrays.side_images, args.fps, args.source_format)
    assert "observation.wrench" in specs
    assert "observation.ee_pose" in specs
    assert "action" in specs
    assert not {"observation.q_cmd", "observation.delta_q", "observation.torque", "observation.tau_free", "observation.tau_ext"}.intersection(specs)
    assert specs["observation.images.wrist"]["info"]["video.fps"] == 20


def test_libero_cannot_be_relabelled_as_50hz(tmp_path):
    source = libero_source(tmp_path)
    args = Namespace(source_format="auto", fps=50, camera_fps=25)
    with pytest.raises(ValueError, match="source is 20 Hz"):
        _resolve_source_settings([source], args)


def test_relative_nanoseconds_keep_their_unit():
    timestamps = np.array([250_000_000, 300_000_000, 350_000_000], dtype=np.int64)
    np.testing.assert_array_equal(_timestamp_ns(timestamps, 3, 20, "state timestamp", "ns"), timestamps)
