"""Validate causal filtering and episode selection without loading LeRobot videos."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from diffusion_policy.scripts.visualize_lerobot_v3_wrench import (
    first_order_lowpass,
    load_episode,
    visualize,
)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "v3"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0", "fps": 20}))
    epoch = 1_787_380_170_640_629_000
    # Episode 4 spans files and arrives out of order; episode 1 must be excluded.
    for chunk, frames, episodes, offsets in [
        (0, [1, 0, 0], [4, 1, 4], [20_000_001, 0, 0]),
        (1, [3, 2], [4, 4], [100_000_003, 60_000_002]),
    ]:
        path = root / "data" / f"chunk-{chunk:03d}" / "file-000.parquet"
        path.parent.mkdir(parents=True)
        table = pa.table({
            "episode_index": pa.array(episodes, type=pa.int64()),
            "frame_index": pa.array(frames, type=pa.int64()),
            "timestamp": pa.array(np.asarray(frames) * 0.05, type=pa.float32()),
            "timing.state_timestamp_ns": pa.array([epoch + t for t in offsets], type=pa.int64()),
            "observation.wrench": pa.array(
                [[float(frame + 10 * episode + channel) for channel in range(6)]
                 for frame, episode in zip(frames, episodes)],
                type=pa.list_(pa.float32(), 6),
            ),
        })
        pq.write_table(table, path)
    return root


def test_lowpass_matches_step_response_with_irregular_dt_and_is_causal():
    times = np.array([0, 0.03, 0.11, 0.17, 0.3])
    amplitudes = np.array([1, -2, 3, -4, 5, -6])
    values = np.tile(amplitudes, (len(times), 1))
    values[0] = 0
    cutoff = 2.0
    actual = first_order_lowpass(times, values, cutoff)
    expected = (1 - np.exp(-2 * np.pi * cutoff * times[:, None])) * amplitudes
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    # Changing a future input cannot affect the prefix already emitted.
    values[-1] = 1_000_000
    np.testing.assert_array_equal(first_order_lowpass(times, values, cutoff)[:-1], actual[:-1])
    constant = np.tile(amplitudes, (len(times), 1))
    np.testing.assert_array_equal(first_order_lowpass(times, constant, cutoff), constant)


@pytest.mark.parametrize("cutoff", [0, -1, float("nan"), float("inf")])
def test_invalid_cutoff_is_rejected(cutoff):
    with pytest.raises(ValueError, match="--cutoff-hz"):
        first_order_lowpass(np.arange(3), np.zeros((3, 6)), cutoff)


def test_episode_loading_across_chunks_preserves_nanosecond_intervals(dataset):
    times, values, frames, clock = load_episode(dataset, 4)
    np.testing.assert_array_equal(frames, [0, 1, 2, 3])
    np.testing.assert_allclose(times, np.array([0, 20_000_001, 60_000_002, 100_000_003]) * 1e-9,
                               rtol=0, atol=1e-15)
    np.testing.assert_array_equal(values, 40 + np.arange(4)[:, None] + np.arange(6))
    assert clock == "timing.state_timestamp_ns"
    np.testing.assert_allclose(load_episode(dataset, 4, timestamp_key="timestamp")[0], np.arange(4) * .05)
    with pytest.raises(ValueError, match="Episode 9 was not found"):
        load_episode(dataset, 9)


def test_visualize_and_overwrite_protect_outputs_and_leave_source_intact(dataset):
    source_bytes = {path: path.read_bytes() for path in dataset.rglob("*.parquet")}
    result = visualize(dataset, 4, 2)
    assert result["plot"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    rows = np.loadtxt(result["csv"], delimiter=",", skiprows=1)
    times, raw, frames, _ = load_episode(dataset, 4)
    np.testing.assert_array_equal(rows[:, 0], frames)
    np.testing.assert_allclose(rows[:, 1], times)
    np.testing.assert_array_equal(rows[:, 2:8], raw)
    np.testing.assert_allclose(rows[:, 8:], first_order_lowpass(times, raw, 2))
    png_bytes = result["plot"].read_bytes()
    with pytest.raises(FileExistsError, match="--overwrite"):
        visualize(dataset, 4, 3, output=result["plot"])
    assert result["plot"].read_bytes() == png_bytes
    visualize(dataset, 4, 3, output=result["plot"], overwrite=True)
    assert result["plot"].read_bytes() != png_bytes
    for path, original in source_bytes.items():
        assert path.read_bytes() == original


def test_cli_direct_script_and_timestamp_fallback(dataset, tmp_path):
    # Native v3 datasets without recorded nanosecond clocks use timestamp.
    for path in dataset.rglob("*.parquet"):
        table = pq.read_table(path).drop(["timing.state_timestamp_ns"])
        pq.write_table(table, path)
    script = Path(__file__).resolve().parents[1] / "diffusion_policy/scripts/visualize_lerobot_v3_wrench.py"
    output = tmp_path / "cli.png"
    run = subprocess.run([
        sys.executable, str(script), "--dataset", str(dataset), "--episode", "4",
        "--cutoff-hz", "1.5", "--output", str(output), "--overwrite",
    ], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "clock=timestamp" in run.stdout
    assert output.is_file() and output.with_suffix(".csv").is_file()


def test_non_increasing_clock_rejected():
    with pytest.raises(ValueError, match="strictly increase"):
        first_order_lowpass(np.array([0, 0.1, 0.1]), np.zeros((3, 6)), 2)
