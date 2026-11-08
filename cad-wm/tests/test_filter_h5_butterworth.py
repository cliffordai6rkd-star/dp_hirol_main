from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import pytest

from data_process.tool.filter_h5_butterworth import (
    causal_butterworth_lowpass,
    run,
)


def test_causal_second_order_butterworth_preserves_dc_and_attenuates_high_frequency():
    sample_rate_hz = 120.0
    time_s = np.arange(600, dtype=np.float64) / sample_rate_hz
    timestamps_us = np.rint(time_s * 1.0e6).astype(np.int64)
    low = np.sin(2.0 * np.pi * 3.0 * time_s)
    high = np.sin(2.0 * np.pi * 40.0 * time_s)

    low_filtered = causal_butterworth_lowpass(
        low[:, None], timestamps_us, cutoff_hz=15.0
    )[:, 0]
    high_filtered = causal_butterworth_lowpass(
        high[:, None], timestamps_us, cutoff_hz=15.0
    )[:, 0]
    dc_filtered = causal_butterworth_lowpass(
        np.full((100, 2), 7.0),
        timestamps_us[:100],
        cutoff_hz=15.0,
    )

    np.testing.assert_allclose(dc_filtered, 7.0, atol=1.0e-12)
    assert np.std(low_filtered[100:]) > 0.9 * np.std(low[100:])
    assert np.std(high_filtered[100:]) < 0.2 * np.std(high[100:])
    assert low_filtered[0] == pytest.approx(low[0])


def _write_episode(path: Path) -> None:
    sample_rate_hz = 120.0
    count = 240
    time_s = np.arange(count, dtype=np.float64) / sample_rate_hz
    timestamps_us = np.rint(time_s * 1.0e6).astype(np.int64)
    signal = np.sin(2.0 * np.pi * 40.0 * time_s)[:, None]

    with h5py.File(path, "w") as h5_file:
        h5_file.attrs["source_attr"] = "preserved"
        teleop = h5_file.create_group("teleop")
        teleop.create_dataset("timestamp_us", data=timestamps_us)
        teleop.create_dataset("q_follower", data=np.repeat(signal, 7, axis=1))
        teleop.create_dataset("dq_follower", data=np.repeat(2.0 * signal, 7, axis=1))
        teleop.create_dataset("tau_follower", data=np.repeat(3.0 * signal, 7, axis=1))
        teleop.create_dataset("q_cmd", data=np.repeat(4.0 * signal, 7, axis=1))
        teleop.create_dataset("q_leader", data=np.repeat(5.0 * signal, 7, axis=1))
        teleop.create_dataset("integer_status", data=np.arange(count))
        pose = np.repeat(np.eye(4)[None, :, :], count, axis=0)
        teleop.create_dataset("ee_pose_follower", data=pose)
        h5_file.create_dataset("config_yaml", data="unchanged")


def _args(input_dir: Path, output_dir: Path) -> Namespace:
    return Namespace(
        input_dir=input_dir,
        output_dir=output_dir,
        cutoff_hz=15.0,
        timestamp_path="teleop/timestamp_us",
        dataset=[
            "teleop/q_follower",
            "teleop/dq_follower",
            "teleop/tau_follower",
        ],
        max_episodes=None,
    )


def test_run_copies_h5_and_filters_only_eligible_float_time_series(tmp_path: Path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "filtered"
    input_dir.mkdir()
    source_path = input_dir / "episode_0000.h5"
    _write_episode(source_path)

    manifest = run(_args(input_dir, output_dir))

    with h5py.File(source_path, "r") as source, h5py.File(
        output_dir / source_path.name, "r"
    ) as output:
        np.testing.assert_array_equal(
            output["teleop/timestamp_us"], source["teleop/timestamp_us"]
        )
        np.testing.assert_array_equal(
            output["teleop/integer_status"], source["teleop/integer_status"]
        )
        np.testing.assert_array_equal(
            output["teleop/ee_pose_follower"], source["teleop/ee_pose_follower"]
        )
        assert output["config_yaml"][()].decode() == "unchanged"
        assert output.attrs["source_attr"] == "preserved"
        assert output.attrs["butterworth_filter_order"] == 2
        assert not np.array_equal(
            output["teleop/q_follower"], source["teleop/q_follower"]
        )
        assert not np.array_equal(
            output["teleop/dq_follower"], source["teleop/dq_follower"]
        )
        assert not np.array_equal(
            output["teleop/tau_follower"], source["teleop/tau_follower"]
        )
        np.testing.assert_array_equal(output["teleop/q_cmd"], source["teleop/q_cmd"])
        np.testing.assert_array_equal(
            output["teleop/q_leader"], source["teleop/q_leader"]
        )
        assert output["teleop/q_follower"].attrs["filter_cutoff_hz"] == 15.0

    assert len(manifest["episodes"]) == 1
    assert (output_dir / "butterworth_filter_manifest.json").is_file()


def test_run_refuses_to_reuse_output_directory(tmp_path: Path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "filtered"
    input_dir.mkdir()
    output_dir.mkdir()
    _write_episode(input_dir / "episode_0000.h5")

    with pytest.raises(FileExistsError, match="output directory"):
        run(_args(input_dir, output_dir))


def test_run_requires_explicit_dataset_selection(tmp_path: Path):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    _write_episode(input_dir / "episode_0000.h5")
    args = _args(input_dir, tmp_path / "filtered")
    args.dataset = []

    with pytest.raises(ValueError, match="At least one --dataset"):
        run(args)


def test_run_rejects_missing_explicit_dataset(tmp_path: Path):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    _write_episode(input_dir / "episode_0000.h5")
    args = _args(input_dir, tmp_path / "filtered")
    args.dataset = ["teleop/not_present"]

    with pytest.raises(KeyError, match="teleop/not_present"):
        run(args)
