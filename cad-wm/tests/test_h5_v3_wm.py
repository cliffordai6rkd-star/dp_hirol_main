from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from data_process.tool.h5_2_lerobotev3 import H5Dataset
from data_process.tool.h5_v3_wm import (
    build_wm_conversion_spec,
    build_wm_episode_cache,
    read_wm_frame,
    write_wm_manifest,
)


class _DatasetHandle:
    def __init__(self, values) -> None:
        self.values = np.asarray(values)
        self.shape = self.values.shape

    def __getitem__(self, item):
        return self.values[item]

    def __len__(self):
        return len(self.values)


class _H5Py:
    Dataset = _DatasetHandle


def _config():
    return {
        "fps": 100,
        "task": "wm_test",
        "timeline": {
            "mode": "raw_lowdim_action_hold",
            "state_timestamp_path": "state/timestamp_us",
            "action_anchor_timestamp_path": "camera/timestamp_us",
            "action_fps": 25,
            "max_action_gap_s": 0.02,
        },
        "features": {
            "observation.joint": {
                "rate": "state",
                "dtype": "float32",
                "shape": [1],
                "h5_path": "state/q",
                "align": "index",
            },
            "action.joint": {
                "rate": "action",
                "dtype": "float32",
                "shape": [1],
                "h5_path": "action/q",
                "timestamp_path": "action/timestamp_us",
                "resample": "previous",
            },
        },
    }


def _episode_cache(action_ts=None, *, action_fps=25):
    # State rows and camera timestamps are deliberately irregular. Expert and
    # WM conversion must preserve camera timestamps rather than regularize them.
    state_ts = np.asarray(
        [0, 9_000, 20_000, 31_000, 41_000, 52_000, 63_000, 79_000,
         81_000, 93_000, 105_000, 121_000],
        dtype=np.int64,
    )
    camera_ts = np.asarray([5_000, 46_000, 84_000], dtype=np.int64)
    # Expert anchors are 5, 46, 84 ms; previous selects 3, 43, 83 ms. Samples
    # just after each anchor prove that conversion is not using ``next``.
    if action_ts is None:
        action_ts = [3_000, 7_000, 43_000, 47_000, 83_000, 87_000]
    action_ts = np.asarray(action_ts, dtype=np.int64)
    h5_file = {
        "state/timestamp_us": _DatasetHandle(state_ts),
        "state/q": _DatasetHandle(np.arange(len(state_ts))[:, None]),
        "camera/timestamp_us": _DatasetHandle(camera_ts),
        "action/timestamp_us": _DatasetHandle(action_ts),
        "action/q": _DatasetHandle(
            np.asarray(
                [[10.0], [11.0], [20.0], [21.0], [30.0], [31.0]],
                dtype=np.float32,
            )[
                : len(action_ts)
            ]
        ),
    }
    dataset = H5Dataset(".", h5py=_H5Py, np=np)
    config = _config()
    config["timeline"]["action_fps"] = action_fps
    spec = build_wm_conversion_spec(config)
    return build_wm_episode_cache(dataset, h5_file, spec, Path("episode.h5"))


def test_spec_requires_raw_index_states_and_previous_expert_actions():
    config = _config()
    config["features"]["observation.joint"].update(
        {"align": "previous", "timestamp_path": "state/timestamp_us"}
    )
    with pytest.raises(ValueError, match="align='index'"):
        build_wm_conversion_spec(config)

    config = _config()
    config["features"]["action.joint"]["resample"] = "next"
    with pytest.raises(ValueError, match="resample='previous'"):
        build_wm_conversion_spec(config)


def test_action_labeled_raw_state_rows_and_timestamps_are_preserved():
    cache = _episode_cache()
    np.testing.assert_array_equal(
        cache["resampled"]["observation.joint"][:, 0], np.arange(1, 12)
    )
    np.testing.assert_array_equal(
        cache["timing"]["timing.state_timestamp_ns"][:, 0],
        np.asarray([9, 20, 31, 41, 52, 63, 79, 81, 93, 105, 121])
        * 1_000_000,
    )


def test_camera_row_expert_actions_are_zoh_held_on_raw_rows():
    cache = _episode_cache()
    np.testing.assert_array_equal(
        cache["resampled"]["action.joint"][:, 0],
        [10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30],
    )
    assert "timing.action_valid" not in cache["timing"]
    np.testing.assert_array_equal(
        cache["timing"]["timing.action_update"][:, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    )


def test_configured_action_fps_must_match_camera_timeline():
    with pytest.raises(ValueError, match="not action_fps=50"):
        _episode_cache(action_fps=50)


def test_action_anchor_and_previous_source_timestamps_are_preserved():
    cache = _episode_cache()
    # The jittered raw camera timestamps are preserved exactly.
    np.testing.assert_array_equal(
        cache["timing"]["timing.action_anchor_timestamp_ns"][:, 0],
        np.asarray([5, 5, 5, 5, 46, 46, 46, 46, 84, 84, 84])
        * 1_000_000,
    )
    np.testing.assert_array_equal(
        cache["timing"]["timing.action_source_timestamp_ns"][:, 0],
        np.asarray([3, 3, 3, 3, 43, 43, 43, 43, 83, 83, 83])
        * 1_000_000,
    )
    np.testing.assert_array_equal(
        cache["timing"]["timing.action_phase_ns"][:2, 0],
        [4_000_000, 15_000_000],
    )


def test_read_frame_returns_raw_state_and_held_action():
    frame = read_wm_frame(_episode_cache(), 7)
    np.testing.assert_array_equal(frame["observation.joint"], [8])
    np.testing.assert_array_equal(frame["action.joint"], [20])
    np.testing.assert_array_equal(frame["timing.action_update"], [0])
    np.testing.assert_array_equal(frame["timing.action_phase_ns"], [35_000_000])


def test_rows_owned_by_camera_anchor_without_fresh_previous_action_are_clipped():
    cache = _episode_cache(action_ts=[3_000, 7_000, 43_000, 47_000])
    np.testing.assert_array_equal(
        cache["resampled"]["observation.joint"][:, 0], np.arange(1, 9)
    )
    np.testing.assert_array_equal(
        cache["resampled"]["action.joint"][:, 0],
        [10, 10, 10, 10, 20, 20, 20, 20],
    )
    assert "timing.action_valid" not in cache["timing"]


def test_manifest_records_expert_camera_zoh_contract(tmp_path):
    path = write_wm_manifest(tmp_path, build_wm_conversion_spec(_config()))
    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 3
    assert manifest["action_fps"] == 25
    assert (
        manifest["action_fps_validation"]
        == "median_camera_period_within_10_percent"
    )
    assert manifest["action_anchor_timestamp_path"] == "camera/timestamp_us"
    assert manifest["action_sampling"] == "previous_expert_label"
    assert manifest["action_upsampling"] == "zoh_previous_camera_anchor"
    assert manifest["action_contract"] == "high_level_expert_camera_snapshot_v1"


@pytest.mark.parametrize(
    "name",
    ["push_button", "insert_usb", "cucumber_peeling"],
)
def test_swm_joint_action_matches_va_command_semantics(name):
    config_path = (
        Path(__file__).parents[1] / "config" / "shape_meta" / "swm" / f"{name}.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["timeline"]["action_fps"] == 25
    assert config["features"]["action.joint"]["h5_path"] == "teleop/q_cmd"
    assert config["features"]["action.joint"]["resample"] == "previous"
