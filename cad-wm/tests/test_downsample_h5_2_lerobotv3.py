from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from data_process.tool import downsample_h5_2_lerobotv3 as index_stride
from data_process.tool.downsample_h5_2_lerobotv3 import (
    build_phase_cache,
    conversion_spec,
    ensure_new_outputs,
)
from data_process.tool.h5_2_lerobotev3 import H5Dataset


class _DatasetHandle:
    def __init__(self, values) -> None:
        self.values = np.asarray(values)
        self.shape = self.values.shape

    def __getitem__(self, item):
        return self.values[item]


class _FakeH5(dict):
    pass


def _shape_meta() -> dict:
    return {
        "fps": 50,
        "master_timestamp_path": "teleop/timestamp_us",
        "downsample": {
            "mode": "index_stride",
            "source_fps": 100,
            "stride": 2,
            "phases": [0, 1],
        },
        "features": {
            "observation.joint": {
                "dtype": "float32",
                "shape": [1],
                "h5_path": "teleop/q",
                "align": "index",
            },
            "observation.delta_q": {
                "dtype": "float32",
                "shape": [1],
                "sources": [
                    {"h5_path": "teleop/q_cmd", "align": "index"},
                    {"h5_path": "teleop/q", "align": "index"},
                ],
                "combine": "subtract",
            },
        },
    }


def _dataset() -> H5Dataset:
    class _H5Py:
        Dataset = _DatasetHandle

    return H5Dataset(".", h5py=_H5Py, np=np)


def _cache(timestamps, *, phase_index: int):
    timestamps = np.asarray(timestamps, dtype=np.int64)
    q = np.arange(timestamps.size, dtype=np.float32)[:, None]
    h5_file = _FakeH5(
        {
            "teleop/timestamp_us": _DatasetHandle(timestamps),
            "teleop/q": _DatasetHandle(q),
            "teleop/q_cmd": _DatasetHandle(q + 10.0),
        }
    )
    spec = conversion_spec(_shape_meta())
    return build_phase_cache(
        _dataset(),
        h5_file,
        spec,
        Path("episode.h5"),
        phase_index=phase_index,
    )


def test_index_stride_produces_even_and_odd_views() -> None:
    timestamps = [0, 10_100, 20_300, 30_200, 40_500, 50_700, 61_000]

    phase_zero = _cache(timestamps, phase_index=0)
    phase_one = _cache(timestamps, phase_index=1)

    np.testing.assert_array_equal(phase_zero["snapshot_indices"], [0, 2, 4, 6])
    np.testing.assert_array_equal(phase_one["snapshot_indices"], [1, 3, 5])
    np.testing.assert_array_equal(
        phase_zero["target_timestamps"],
        np.asarray(timestamps)[[0, 2, 4, 6]],
    )
    np.testing.assert_array_equal(
        phase_one["target_timestamps"],
        np.asarray(timestamps)[[1, 3, 5]],
    )
    np.testing.assert_allclose(phase_zero["aligned"]["observation.delta_q"], 10.0)
    np.testing.assert_allclose(phase_one["aligned"]["observation.delta_q"], 10.0)


def test_index_stride_contract_rejects_non_index_mapping() -> None:
    shape_meta = _shape_meta()
    shape_meta["features"]["observation.joint"].update(
        {
            "align": "previous",
            "timestamp_path": "teleop/timestamp_us",
        }
    )

    with pytest.raises(ValueError, match="must use align='index'"):
        conversion_spec(shape_meta)


def test_index_stride_contract_requires_100_to_50_ratio() -> None:
    shape_meta = _shape_meta()
    shape_meta["downsample"]["source_fps"] = 117

    with pytest.raises(ValueError, match="does not produce fps=50"):
        conversion_spec(shape_meta)


def test_phase_output_paths_must_be_new_and_distinct(tmp_path: Path) -> None:
    ensure_new_outputs([tmp_path / "phase0", tmp_path / "phase1"])
    with pytest.raises(ValueError, match="must be different"):
        ensure_new_outputs([tmp_path / "same", tmp_path / "same"])
    with pytest.raises(FileExistsError, match="already exists"):
        ensure_new_outputs([tmp_path, tmp_path / "new"])


def test_conversion_writes_phases_to_two_datasets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "h5"
    phase0_path = tmp_path / "phase0"
    phase1_path = tmp_path / "phase1"
    input_path.mkdir()
    with h5py.File(input_path / "episode_0000.h5", "w") as h5_file:
        timestamps = np.arange(0, 70_000, 10_000, dtype=np.int64)
        q = np.arange(timestamps.size, dtype=np.float32)[:, None]
        h5_file.create_dataset("teleop/timestamp_us", data=timestamps)
        h5_file.create_dataset("teleop/q", data=q)
        h5_file.create_dataset("teleop/q_cmd", data=q + 10.0)

    config = _shape_meta()
    config.update(
        {
            "io": {
                "input": str(input_path),
                "output_phase0": str(phase0_path),
                "output_phase1": str(phase1_path),
                "repo_id_phase0": "phase0",
                "repo_id_phase1": "phase1",
                "no_videos": True,
                "push_to_hub": False,
            },
            "task": "index_stride_test",
        }
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    class _FakeLeRobotDataset:
        instances = {}

        @classmethod
        def create(cls, **kwargs):
            root = kwargs["root"]
            root.mkdir(parents=True)
            instance = cls()
            instance.current_frames = []
            instance.episodes = []
            cls.instances[root] = instance
            return instance

        def add_frame(self, frame, task=None):
            self.current_frames.append(frame)

        def save_episode(self, task=None):
            self.episodes.append(self.current_frames)
            self.current_frames = []

        def finalize(self):
            pass

    monkeypatch.setattr(
        index_stride,
        "load_conversion_deps",
        lambda: (h5py, np, _FakeLeRobotDataset),
    )
    index_stride.run_conversion(
        SimpleNamespace(
            config=config_path,
            input=None,
            output_phase0=None,
            output_phase1=None,
        )
    )

    phase0 = _FakeLeRobotDataset.instances[phase0_path].episodes[0]
    phase1 = _FakeLeRobotDataset.instances[phase1_path].episodes[0]
    np.testing.assert_array_equal(
        [frame["observation.joint"][0] for frame in phase0],
        [0, 2, 4, 6],
    )
    np.testing.assert_array_equal(
        [frame["observation.joint"][0] for frame in phase1],
        [1, 3, 5],
    )
    phase0_manifest = json.loads(
        (phase0_path / "meta" / "index_stride_manifest.json").read_text()
    )
    phase1_manifest = json.loads(
        (phase1_path / "meta" / "index_stride_manifest.json").read_text()
    )
    assert phase0_manifest["source_index_rule"] == "0 + k * 2"
    assert phase1_manifest["source_index_rule"] == "1 + k * 2"
