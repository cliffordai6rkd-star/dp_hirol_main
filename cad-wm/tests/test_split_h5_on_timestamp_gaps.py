from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from data_process.tool.split_h5_on_timestamp_gaps import (
    segment_bounds,
    split_file,
)


def test_segment_bounds_split_after_the_last_sample_before_gap() -> None:
    timestamps = np.asarray([0, 10_000, 20_000, 520_000, 530_000])

    bounds, gaps = segment_bounds(timestamps, max_gap_s=0.1)

    assert bounds == [(0, 3), (3, 5)]
    np.testing.assert_allclose(gaps, [0.5])


def test_split_file_slices_all_time_series_and_preserves_static_data(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.h5"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    with h5py.File(source_path, "w") as h5_file:
        h5_file.attrs["format"] = "test"
        h5_file.create_dataset("config_yaml", data="config")
        teleop = h5_file.create_group("teleop")
        teleop.attrs["owner"] = "robot"
        timestamps = np.asarray([0, 10_000, 20_000, 520_000, 530_000])
        teleop.create_dataset("timestamp_us", data=timestamps)
        q = teleop.create_dataset(
            "q",
            data=np.arange(10, dtype=np.float64).reshape(5, 2),
            compression="gzip",
        )
        q.attrs["units"] = "rad"

    result = split_file(
        source_path,
        output_dir,
        timestamp_path="teleop/timestamp_us",
        max_gap_s=0.1,
        min_frames=2,
    )

    assert [item["path"] for item in result["outputs"]] == [
        "source_part00.h5",
        "source_part01.h5",
    ]
    with h5py.File(output_dir / "source_part00.h5", "r") as first:
        np.testing.assert_array_equal(first["teleop/timestamp_us"][:], timestamps[:3])
        np.testing.assert_array_equal(first["teleop/q"][:], np.arange(6).reshape(3, 2))
        assert first["config_yaml"][()].decode() == "config"
        assert first.attrs["format"] == "test"
        assert first.attrs["timestamp_gap_split_stop_index"] == 3
        assert first["teleop"].attrs["owner"] == "robot"
        assert first["teleop/q"].attrs["units"] == "rad"
    with h5py.File(output_dir / "source_part01.h5", "r") as second:
        np.testing.assert_array_equal(second["teleop/timestamp_us"][:], timestamps[3:])
        np.testing.assert_array_equal(second["teleop/q"][:], np.arange(6, 10).reshape(2, 2))
