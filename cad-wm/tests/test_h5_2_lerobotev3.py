from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from data_process.tool.h5_2_lerobotev3 import (
    H5Dataset,
    LeRobotV3Dataset,
    NERO_RUNS_ROOT,
    PINN_ROOT,
    build_conversion_spec,
    config_path,
    resolve_raw_index_fps,
    resolve_io_path,
)


def _dataset() -> H5Dataset:
    return H5Dataset(".", h5py=object(), np=np)


def test_next_point_sampling_selects_first_row_at_or_after_target() -> None:
    dataset = _dataset()
    source_times = np.asarray([0.00, 0.01, 0.02, 0.03])
    targets = np.asarray([0.001, 0.01, 0.019, 0.03])

    np.testing.assert_array_equal(
        dataset._point_sample_indices(source_times, targets, "next"),
        [1, 1, 2, 3],
    )


def test_io_paths_resolve_from_repo_roots_instead_of_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    assert resolve_io_path("data/train_episode/example") == (
        PINN_ROOT / "data/train_episode/example"
    ).resolve()
    assert resolve_io_path("nero_ws/runs/next_data") == (
        NERO_RUNS_ROOT / "next_data"
    ).resolve()
    assert resolve_io_path("../nero_ws/runs/next_data") == (
        NERO_RUNS_ROOT / "next_data"
    ).resolve()
    assert resolve_io_path("runs/next_data") == (
        NERO_RUNS_ROOT / "next_data"
    ).resolve()


def test_cli_io_override_uses_nero_runs_alias() -> None:
    config = {"io": {"input": "data/old"}}

    assert config_path(config, "input", override="runs/test") == (
        NERO_RUNS_ROOT / "test"
    ).resolve()


def _dual_rate_spec() -> dict:
    return build_conversion_spec(
        {
            "fps": 10,
            "master_timestamp_path": "teleop/timestamp_us",
            "timeline": {
                "mode": "dual_rate",
                "high_fps": 20,
                "low_fps": 10,
                "action_horizon": 3,
                "max_gap_s": 0.051,
                "store_timestamps": True,
            },
            "features": {
                "observation.images.wrist": {
                    "dtype": "video",
                    "shape": [2, 2, 3],
                    "h5_path": "camera/frames",
                    "timestamp_path": "camera/timestamp_us",
                    "grid": "low_anchor",
                    "resample": "previous",
                    "allow_stale": True,
                    "emit_age_key": "wrist",
                },
                "observation.joint": {
                    "dtype": "float32",
                    "shape": [2, 1],
                    "h5_path": "teleop/q",
                    "timestamp_path": "teleop/timestamp_us",
                    "grid": "high_past",
                    "resample": "linear",
                },
                "action.joint": {
                    "dtype": "float32",
                    "shape": [3, 1],
                    "h5_path": "teleop/q",
                    "timestamp_path": "teleop/timestamp_us",
                    "grid": "low_future",
                    "resample": "linear",
                },
            },
        }
    )


def _dual_rate_cache() -> tuple[dict, dict]:
    spec = _dual_rate_spec()
    teleop_us = np.arange(1_000_000, 1_600_001, 50_000, dtype=np.int64)
    camera_us = np.asarray([1_050_000, 1_250_000, 1_450_000], dtype=np.int64)
    datasets = {
        "teleop/timestamp_us": teleop_us,
        "teleop/q": ((teleop_us - 1_000_000) * 1e-6)[:, None],
        "camera/timestamp_us": camera_us,
        "camera/frames": np.stack(
            [np.full((2, 2, 3), value, dtype=np.uint8) for value in (10, 20, 30)]
        ),
    }
    timestamps = {
        "teleop/timestamp_us": teleop_us,
        "camera/timestamp_us": camera_us,
    }
    cache = _dataset()._build_dual_rate_episode_cache(
        dataset_cache=datasets,
        timestamp_cache=timestamps,
        mappings=spec["mappings"],
        master_timestamp_path=spec["master_timestamp_path"],
        h5_path=Path("episode.h5"),
        timeline=spec["timeline"],
    )
    return spec, cache


def _camera_rows_spec() -> dict:
    return build_conversion_spec(
        {
            "fps": 25,
            "master_timestamp_path": "wrist/timestamp_us",
            "timeline": {
                "mode": "camera_rows",
                "history_size": 4,
                "action_horizon": 8,
                "max_gap_s": 0.02,
                "store_timestamps": True,
            },
            "features": {
                "observation.images.wrist": {
                    "dtype": "video",
                    "shape": [1, 1, 3],
                    "h5_path": "wrist/frames",
                    "timestamp_path": "wrist/timestamp_us",
                    "grid": "low_anchor",
                    "resample": "previous",
                    "allow_stale": True,
                },
                "observation.images.side": {
                    "dtype": "video",
                    "shape": [1, 1, 3],
                    "h5_path": "side/frames",
                    "timestamp_path": "side/timestamp_us",
                    "grid": "low_anchor",
                    "resample": "previous",
                    "allow_stale": True,
                },
                "observation.wrench_ext": {
                    "dtype": "float32",
                    "shape": [4, 1],
                    "h5_path": "teleop/wrench",
                    "timestamp_path": "teleop/timestamp_us",
                    "grid": "high_past",
                    "resample": "previous",
                },
                "action.joint": {
                    "dtype": "float32",
                    "shape": [1],
                    "h5_path": "teleop/q",
                    "timestamp_path": "teleop/timestamp_us",
                    "grid": "low_anchor",
                    "resample": "next",
                },
            },
        }
    )


def test_camera_rows_preserve_real_wrist_rows_and_use_causal_alignment() -> None:
    spec = _camera_rows_spec()
    wrist_us = np.asarray([5_000, 46_000, 84_000], dtype=np.int64)
    side_us = np.asarray([4_000, 45_000, 83_000], dtype=np.int64)
    teleop_us = np.arange(0, 100_000, 10_000, dtype=np.int64)
    def image(values):
        return np.stack(
            [np.full((1, 1, 3), value, dtype=np.uint8) for value in values]
        )
    datasets = {
        "wrist/timestamp_us": wrist_us,
        "wrist/frames": image([1, 2, 3]),
        "side/timestamp_us": side_us,
        "side/frames": image([11, 12, 13]),
        "teleop/timestamp_us": teleop_us,
        "teleop/wrench": np.arange(10, dtype=np.float32)[:, None],
        "teleop/q": (100 + np.arange(10, dtype=np.float32))[:, None],
    }
    cache = _dataset()._build_camera_rows_episode_cache(
        dataset_cache=datasets,
        timestamp_cache={
            "wrist/timestamp_us": wrist_us,
            "side/timestamp_us": side_us,
            "teleop/timestamp_us": teleop_us,
        },
        mappings=spec["mappings"],
        master_timestamp_path=spec["master_timestamp_path"],
        h5_path=Path("episode.h5"),
        timeline=spec["timeline"],
    )

    # The first wrist row is dropped because four causal wrench rows do not yet exist.
    np.testing.assert_array_equal(cache["target_timestamps"], [46_000, 84_000])
    np.testing.assert_array_equal(
        cache["resampled"]["observation.wrench_ext"][:, :, 0],
        [[1, 2, 3, 4], [5, 6, 7, 8]],
    )
    np.testing.assert_array_equal(
        cache["resampled"]["action.joint"][:, 0], [105, 109]
    )
    np.testing.assert_array_equal(
        cache["resampled"]["observation.images.wrist"]["indices"], [1, 2]
    )
    np.testing.assert_array_equal(
        cache["resampled"]["observation.images.side"]["indices"], [1, 2]
    )
    np.testing.assert_array_equal(
        cache["anchor_timestamp_ns"], [46_000_000, 84_000_000]
    )
    np.testing.assert_array_equal(
        cache["action_source_timestamp_ns"], [50_000_000, 90_000_000]
    )
    assert "timing.action_source_timestamp_ns" in spec["lerobot_features"]
    assert "timing.action_timestamp_ns" not in spec["lerobot_features"]


def test_timestamp_ns_conversion_preserves_epoch_microseconds_exactly() -> None:
    raw = np.asarray([1_700_000_000_000_001], dtype=np.int64)
    np.testing.assert_array_equal(
        _dataset()._timestamps_ns(raw, "timestamp_us"),
        [1_700_000_000_000_001_000],
    )


def test_conversion_spec_requires_explicit_fps() -> None:
    with pytest.raises(ValueError, match="define a positive integer 'fps'"):
        build_conversion_spec(
            {
                "master_timestamp_path": "teleop/timestamp_us",
                "features": {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [1],
                        "h5_path": "teleop/state",
                    }
                },
            }
        )


def _raw_index_shape_meta() -> dict:
    return {
        "fps": 100,
        "master_timestamp_path": "teleop/timestamp_us",
        "sampling": {"mode": "raw_index"},
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


def test_raw_index_keeps_every_master_row_despite_timestamp_jitter(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "episode.h5"
    timestamps = np.asarray(
        [1_000_000, 1_010_100, 1_020_300, 1_030_200, 1_040_500],
        dtype=np.int64,
    )
    q = np.arange(len(timestamps), dtype=np.float64)[:, None]
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset("teleop/timestamp_us", data=timestamps)
        h5_file.create_dataset("teleop/q", data=q)
        h5_file.create_dataset("teleop/q_cmd", data=q + 10.0)

    spec = build_conversion_spec(_raw_index_shape_meta())
    dataset = H5Dataset(tmp_path, h5py=h5py, np=np)
    with dataset.open_episode(h5_path) as h5_file:
        cache = dataset.build_episode_cache(
            h5_file,
            spec["mappings"],
            spec["master_timestamp_path"],
            spec["fps"],
            h5_path,
            sampling=spec["sampling"],
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
            for index in range(dataset.episode_length(
                h5_file,
                spec["master_timestamp_path"],
                h5_path,
                cache,
            ))
        ]

    np.testing.assert_array_equal(cache["target_timestamps"], timestamps)
    np.testing.assert_array_equal(
        [frame["observation.joint"][0] for frame in frames],
        [0, 1, 2, 3, 4],
    )
    np.testing.assert_allclose(
        [frame["observation.delta_q"][0] for frame in frames],
        10.0,
    )


def test_raw_index_rejects_any_non_index_feature() -> None:
    shape_meta = _raw_index_shape_meta()
    shape_meta["features"]["observation.joint"].update(
        {
            "align": "previous",
            "timestamp_path": "teleop/timestamp_us",
        }
    )

    with pytest.raises(ValueError, match="raw_index.*must use align='index'"):
        build_conversion_spec(shape_meta)


def test_raw_index_infers_nominal_fps_from_median_interval(
    tmp_path: Path,
) -> None:
    timestamps = np.asarray(
        [0, 10_000, 20_100, 30_000, 530_000, 540_000],
        dtype=np.int64,
    )
    with h5py.File(tmp_path / "episode.h5", "w") as h5_file:
        h5_file.create_dataset("teleop/timestamp_us", data=timestamps)

    dataset = H5Dataset(tmp_path, h5py=h5py, np=np)
    shape_meta = _raw_index_shape_meta()
    shape_meta.pop("fps")

    resolved = resolve_raw_index_fps(shape_meta, dataset)

    assert resolved["fps"] == 100


def test_uniform_timestamps_use_configured_fps_and_master_bounds() -> None:
    master = np.asarray([1_000_000, 1_071_000, 1_129_000, 1_210_000], dtype=np.int64)

    result = _dataset().uniform_timestamps(master, "camera/timestamp_us", fps=10)

    np.testing.assert_array_equal(
        result,
        np.asarray([1_000_000, 1_100_000, 1_200_000], dtype=np.int64),
    )


def test_fixed_phase_timestamps_use_absolute_50hz_grid() -> None:
    master = np.asarray(
        [1_000_007, 1_011_000, 1_025_000, 1_043_000],
        dtype=np.int64,
    )

    result = _dataset().fixed_phase_timestamps(
        master,
        "teleop/timestamp_us",
        fps=50,
    )

    np.testing.assert_array_equal(
        result,
        np.asarray([1_020_000, 1_040_000], dtype=np.int64),
    )


def test_fixed_rate_causal_snapshot_uses_one_historical_row_for_all_features() -> None:
    shape_meta = {
        "fps": 50,
        "master_timestamp_path": "teleop/timestamp_us",
        "sampling": {
            "mode": "fixed_rate_causal_snapshot",
            "phase": "unix_epoch",
            "max_staleness_s": 0.03,
        },
        "features": {
            "observation.joint": {
                "dtype": "float32",
                "shape": [1],
                "h5_path": "teleop/q",
                "timestamp_path": "teleop/timestamp_us",
                "align": "previous",
            },
            "observation.delta_q": {
                "dtype": "float32",
                "shape": [1],
                "sources": [
                    {
                        "h5_path": "teleop/q_cmd",
                        "timestamp_path": "teleop/timestamp_us",
                        "align": "previous",
                    },
                    {
                        "h5_path": "teleop/q",
                        "timestamp_path": "teleop/timestamp_us",
                        "align": "previous",
                    },
                ],
                "combine": "subtract",
            },
        },
    }
    spec = build_conversion_spec(shape_meta)
    timestamps = np.asarray(
        [1_000_007, 1_011_000, 1_025_000, 1_043_000],
        dtype=np.int64,
    )
    datasets = {
        "teleop/timestamp_us": timestamps,
        "teleop/q": np.asarray([[0.0], [1.0], [2.0], [3.0]]),
        "teleop/q_cmd": np.asarray([[10.0], [11.0], [12.0], [13.0]]),
    }

    cache = _dataset()._build_causal_snapshot_episode_cache(
        dataset_cache=datasets,
        timestamp_cache={"teleop/timestamp_us": timestamps},
        mappings=spec["mappings"],
        master_timestamp_path=spec["master_timestamp_path"],
        fps=spec["fps"],
        h5_path=Path("episode.h5"),
        sampling=spec["sampling"],
    )

    np.testing.assert_array_equal(cache["snapshot_indices"], [1, 2])
    np.testing.assert_allclose(cache["aligned"]["observation.joint"][:, 0], [1, 2])
    np.testing.assert_allclose(cache["aligned"]["observation.delta_q"][:, 0], [10, 10])


def test_fixed_rate_causal_snapshot_index_uses_selected_master_row() -> None:
    shape_meta = {
        "fps": 50,
        "master_timestamp_path": "teleop/timestamp_us",
        "sampling": {
            "mode": "fixed_rate_causal_snapshot",
            "phase": "unix_epoch",
        },
        "features": {
            "observation.joint": {
                "dtype": "float32",
                "shape": [1],
                "h5_path": "teleop/q",
                "align": "index",
            }
        },
    }
    spec = build_conversion_spec(shape_meta)
    timestamps = np.asarray(
        [1_000_007, 1_011_000, 1_025_000, 1_043_000],
        dtype=np.int64,
    )
    datasets = {
        "teleop/timestamp_us": timestamps,
        "teleop/q": np.asarray([[0.0], [1.0], [2.0], [3.0]]),
    }

    cache = _dataset()._build_causal_snapshot_episode_cache(
        dataset_cache=datasets,
        timestamp_cache={"teleop/timestamp_us": timestamps},
        mappings=spec["mappings"],
        master_timestamp_path=spec["master_timestamp_path"],
        fps=spec["fps"],
        h5_path=Path("episode.h5"),
        sampling=spec["sampling"],
    )

    np.testing.assert_array_equal(cache["snapshot_indices"], [1, 2])
    np.testing.assert_allclose(cache["aligned"]["observation.joint"][:, 0], [1, 2])


def test_fixed_rate_causal_snapshot_rejects_noncausal_feature_alignment() -> None:
    shape_meta = {
        "fps": 50,
        "master_timestamp_path": "teleop/timestamp_us",
        "sampling": {"mode": "fixed_rate_causal_snapshot"},
        "features": {
            "observation.joint": {
                "dtype": "float32",
                "shape": [1],
                "h5_path": "teleop/q",
                "timestamp_path": "teleop/timestamp_us",
                "align": "nearest",
            }
        },
    }

    with pytest.raises(ValueError, match="must use align='previous'"):
        build_conversion_spec(shape_meta)


def test_structured_sources_support_direct_index_alignment() -> None:
    spec = build_conversion_spec(
        {
            "fps": 50,
            "master_timestamp_path": "teleop/timestamp_us",
            "features": {
                "observation.delta_q": {
                    "dtype": "float32",
                    "shape": [1],
                    "sources": [
                        {"h5_path": "teleop/q_cmd", "align": "index"},
                        {"h5_path": "teleop/q", "align": "index"},
                    ],
                    "combine": "subtract",
                }
            },
        }
    )

    mapping = spec["mappings"][0]
    assert [source["method"] for source in mapping["sources"]] == [
        "index",
        "index",
    ]
    assert all(source["timestamp_path"] is None for source in mapping["sources"])


def test_index_alignment_rejects_source_length_mismatch() -> None:
    spec = build_conversion_spec(
        {
            "fps": 50,
            "master_timestamp_path": "teleop/timestamp_us",
            "features": {
                "observation.joint": {
                    "dtype": "float32",
                    "shape": [1],
                    "h5_path": "teleop/q",
                    "align": "index",
                }
            },
        }
    )

    with pytest.raises(ValueError, match="requires exactly 3 rows"):
        _dataset()._validate_index_sources(
            dataset_cache={"teleop/q": np.zeros((2, 1), dtype=np.float32)},
            timestamp_cache={
                "teleop/timestamp_us": np.asarray([100, 200, 300], dtype=np.int64)
            },
            mappings=spec["mappings"],
            master_timestamp_path=spec["master_timestamp_path"],
            h5_path=Path("episode.h5"),
        )


def test_nearest_alignment_selects_latest_historical_sample() -> None:
    timestamps = np.asarray([100, 200, 300], dtype=np.int64)
    dataset = _dataset()

    assert dataset._history_index_from_timestamps(timestamps, 200) == 1
    assert dataset._history_index_from_timestamps(timestamps, 299) == 1


def test_nearest_alignment_rejects_target_before_first_sample() -> None:
    timestamps = np.asarray([100, 200, 300], dtype=np.int64)

    with pytest.raises(ValueError, match="No historical sample"):
        _dataset()._history_index_from_timestamps(timestamps, 99)


@pytest.mark.parametrize(
    ("target_t", "expected_index"),
    [
        (100, 0),
        (149, 0),
        (150, 0),
        (199, 0),
        (200, 1),
        (299, 1),
        (301, 2),
    ],
)
def test_nearest_point_sampling_selects_latest_historical_sample(
    target_t: int,
    expected_index: int,
) -> None:
    timestamps = np.asarray([100, 200, 300], dtype=np.float64)

    result = _dataset()._point_sample_indices(
        timestamps,
        np.asarray([target_t], dtype=np.float64),
        method="nearest",
    )

    assert int(result[0]) == expected_index


def test_nearest_point_sampling_rejects_target_before_first_sample() -> None:
    with pytest.raises(ValueError, match="outside the source timestamp range"):
        _dataset()._point_sample_indices(
            np.asarray([100, 200, 300], dtype=np.float64),
            np.asarray([99], dtype=np.float64),
            method="nearest",
        )


def test_nearest_source_allows_stale_samples() -> None:
    spec = build_conversion_spec(
        {
            "fps": 50,
            "master_timestamp_path": "teleop/timestamp_us",
            "features": {
                "observation.delta_q": {
                    "dtype": "float32",
                    "shape": [1],
                    "sources": [
                        {
                            "h5_path": "teleop/q_cmd",
                            "timestamp_path": "teleop/timestamp_us",
                            "align": "nearest",
                            "allow_stale": True,
                        },
                        {
                            "h5_path": "teleop/q_follower",
                            "timestamp_path": "teleop/timestamp_us",
                            "align": "nearest",
                        },
                    ],
                    "combine": "subtract",
                }
            },
        }
    )

    command_source = spec["mappings"][0]["sources"][0]
    assert command_source["method"] == "nearest"
    assert command_source["allow_stale"] is True


def test_pchip_alignment_requires_a_bounded_gap() -> None:
    shape_meta = {
        "fps": 100,
        "master_timestamp_path": "teleop/timestamp_us",
        "features": {
            "observation.joint": {
                "dtype": "float32",
                "shape": [1],
                "h5_path": "teleop/q",
                "timestamp_path": "teleop/timestamp_us",
                "align": "pchip",
            }
        },
    }

    with pytest.raises(ValueError, match="needs max_gap_s"):
        build_conversion_spec(shape_meta)


def test_pchip_resampling_preserves_knots_and_fills_missing_grid_points() -> None:
    source_times = np.asarray([0.00, 0.01, 0.03, 0.04], dtype=np.float64)
    source_values = (2.0 * source_times + 1.0)[:, None]
    targets = np.arange(5, dtype=np.float64) * 0.01

    result = _dataset()._resample_values(
        source_values,
        source_times=source_times,
        targets=targets,
        method="pchip",
        transform=None,
    )

    np.testing.assert_allclose(result[:, 0], 2.0 * targets + 1.0)
    np.testing.assert_array_equal(result[[0, 1, 3, 4]], source_values)


def test_pchip_alignment_rejects_a_gap_above_the_configured_limit() -> None:
    with pytest.raises(ValueError, match="exceeds max_gap_s"):
        _dataset()._validate_resample_targets(
            source_times=np.asarray([0.0, 0.06]),
            targets=np.asarray([0.01]),
            method="pchip",
            max_gap_s=0.05,
            feature_name="observation.joint",
            h5_path=Path("episode.h5"),
        )


def test_past_window_ends_at_latest_historical_sample() -> None:
    timestamps = np.asarray([100, 200, 300], dtype=np.int64)
    dataset = _dataset()

    np.testing.assert_array_equal(
        dataset._nearest_past_window_indices(timestamps, 299, window_size=3),
        np.asarray([0, 0, 1]),
    )
    np.testing.assert_array_equal(
        dataset._nearest_past_window_indices(timestamps, 350, window_size=3),
        np.asarray([0, 1, 2]),
    )


def test_future_window_starts_at_latest_historical_sample() -> None:
    timestamps = np.asarray([100, 200, 300, 400], dtype=np.int64)
    dataset = _dataset()

    np.testing.assert_array_equal(
        dataset._nearest_future_window_indices(timestamps, 299, window_size=3),
        np.asarray([1, 2, 3]),
    )
    np.testing.assert_array_equal(
        dataset._nearest_future_window_indices(timestamps, 350, window_size=3),
        np.asarray([2, 3, 3]),
    )


def test_ee_pose_transform_converts_single_matrix_to_quat7() -> None:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [1.0, 2.0, 3.0]

    result = _dataset()._ee_pose_matrix_to_quaternion(pose)

    assert result.shape == (7,)
    np.testing.assert_allclose(result, [1, 2, 3, 0, 0, 0, 1], atol=1e-6)


def test_ee_pose_transform_converts_pose_window_to_quat7_window() -> None:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 8, axis=0)
    poses[:, 0, 3] = np.arange(8, dtype=np.float32)

    result = _dataset()._ee_pose_matrix_to_quaternion(poses)

    assert result.shape == (8, 7)
    np.testing.assert_allclose(result[:, 0], np.arange(8), atol=1e-6)
    np.testing.assert_allclose(result[:, 3:], np.asarray([[0, 0, 0, 1]] * 8), atol=1e-6)


def test_conversion_spec_keeps_transform_out_of_lerobot_metadata() -> None:
    spec = build_conversion_spec(
        {
            "fps": 10,
            "master_timestamp_path": "camera/timestamp_us",
            "features": {
                "action.ee_pose": {
                    "dtype": "float32",
                    "shape": [8, 7],
                    "h5_path": "teleop/ee_pose_follower",
                    "timestamp_path": "teleop/timestamp_us",
                    "align": "nearest_future_window",
                    "window_size": 8,
                    "transform": "ee_pose_matrix_to_quaternion",
                }
            },
        }
    )

    assert spec["mappings"][0]["transform"] == "ee_pose_matrix_to_quaternion"
    assert "transform" not in spec["lerobot_features"]["action.ee_pose"]
    assert spec["lerobot_features"]["action.ee_pose"]["shape"] == (8, 7)


def test_conversion_spec_keeps_subtract_combination_out_of_metadata() -> None:
    spec = build_conversion_spec(
        {
            "fps": 100,
            "master_timestamp_path": "teleop/timestamp_us",
            "features": {
                "observation.delta_q": {
                    "dtype": "float32",
                    "shape": [2],
                    "h5_paths": ["teleop/q_cmd", "teleop/q"],
                    "combine": "subtract",
                }
            },
        }
    )

    assert spec["mappings"][0]["combine"] == "subtract"
    assert "combine" not in spec["lerobot_features"]["observation.delta_q"]


def test_subtract_combination_computes_first_h5_path_minus_second() -> None:
    mapping = {
        "lerobot_key": "observation.delta_q",
        "h5_paths": ["teleop/q_cmd", "teleop/q"],
        "timestamp_path": "teleop/timestamp_us",
        "resample": "linear",
        "transform": None,
        "combine": "subtract",
        "feature_spec": {"dtype": "float32", "shape": (2,)},
    }
    cache = {
        "datasets": {
            "teleop/q_cmd": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            "teleop/q": np.asarray([[0.5, 1.5], [2.0, 3.5]]),
        },
        "timestamp_seconds": {
            "teleop/timestamp_us": np.asarray([0.0, 1.0]),
        },
    }

    result = _dataset()._resample_mapping(
        mapping,
        targets=np.asarray([0.0, 0.5, 1.0]),
        cache=cache,
    )

    np.testing.assert_allclose(
        result,
        np.asarray([[0.5, 0.5], [0.75, 0.5], [1.0, 0.5]], dtype=np.float32),
    )


def _delta_q_source_spec() -> dict:
    return build_conversion_spec(
        {
            "fps": 100,
            "master_timestamp_path": "state/timestamp_us",
            "features": {
                "observation.delta_q": {
                    "dtype": "float32",
                    "shape": [1],
                    "sources": [
                        {
                            "h5_path": "control/q_cmd",
                            "timestamp_path": "control/timestamp_us",
                            "align": "previous",
                            "allow_stale": True,
                        },
                        {
                            "h5_path": "state/q_measured",
                            "timestamp_path": "state/timestamp_us",
                            "align": "pchip",
                            "max_gap_s": 0.05,
                        },
                    ],
                    "combine": "subtract",
                }
            },
        }
    )


def test_delta_q_resamples_command_zoh_and_state_continuously_before_subtracting():
    spec = _delta_q_source_spec()
    mapping = spec["mappings"][0]
    command_times = np.asarray([0.00, 0.03, 0.07])
    state_times = np.asarray([0.00, 0.02, 0.04, 0.06, 0.08])
    targets = np.arange(9, dtype=np.float64) * 0.01
    cache = {
        "datasets": {
            "control/q_cmd": np.asarray([[0.1], [0.2], [0.4]]),
            "state/q_measured": (0.05 + 0.5 * state_times)[:, None],
        },
        "timestamp_seconds": {
            "control/timestamp_us": command_times,
            "state/timestamp_us": state_times,
        },
    }

    result = _dataset()._resample_mapping(mapping, targets, cache)

    command_zoh = np.asarray(
        [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4]
    )
    measured_continuous = 0.05 + 0.5 * targets
    np.testing.assert_allclose(
        result[:, 0],
        command_zoh - measured_continuous,
        atol=1.0e-7,
    )
    assert mapping["sources"][0]["method"] == "previous"
    assert mapping["sources"][1]["method"] == "pchip"
    assert "sources" not in spec["lerobot_features"]["observation.delta_q"]


def test_delta_q_zoh_rejects_target_before_first_command_without_future_leakage():
    mapping = _delta_q_source_spec()["mappings"][0]
    cache = {
        "datasets": {
            "control/q_cmd": np.asarray([[0.1], [0.2]]),
            "state/q_measured": np.asarray([[0.0], [0.1]]),
        },
        "timestamp_seconds": {
            "control/timestamp_us": np.asarray([0.01, 0.02]),
            "state/timestamp_us": np.asarray([0.00, 0.02]),
        },
    }

    with pytest.raises(ValueError, match="outside the source timestamp range"):
        _dataset()._resample_mapping(
            mapping,
            targets=np.asarray([0.00]),
            cache=cache,
        )


def test_dual_rate_spec_generates_timing_and_image_age_features() -> None:
    spec = _dual_rate_spec()

    assert spec["timeline"]["high_window_size"] == 2
    assert spec["lerobot_features"]["timing.high_timestamp_ns"]["shape"] == (2, 1)
    assert spec["lerobot_features"]["timing.action_timestamp_ns"]["shape"] == (3, 1)
    assert spec["lerobot_features"]["wrist"]["shape"] == (1,)


def test_dual_rate_grids_are_uniform_contiguous_and_interpolated() -> None:
    spec, cache = _dual_rate_cache()

    assert cache["target_timestamps"].shape == (4,)
    np.testing.assert_array_equal(
        np.diff(cache["anchor_timestamp_ns"]),
        np.full(3, 100_000_000, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.diff(cache["high_timestamp_ns"], axis=1),
        np.full((4, 1), 50_000_000, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        cache["high_timestamp_ns"][1:, 0] - cache["high_timestamp_ns"][:-1, -1],
        np.full(3, 50_000_000, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.diff(cache["action_timestamp_ns"], axis=1),
        np.full((4, 2), 100_000_000, dtype=np.int64),
    )

    first = _dataset()._read_dual_rate_frame(0, spec["mappings"], cache)
    np.testing.assert_allclose(first["observation.joint"][:, 0], [0.05, 0.10])
    np.testing.assert_allclose(first["action.joint"][:, 0], [0.10, 0.20, 0.30])


def test_dual_rate_images_repeat_latest_history_without_future_leakage() -> None:
    spec, cache = _dual_rate_cache()
    frames = [
        _dataset()._read_dual_rate_frame(index, spec["mappings"], cache)
        for index in range(4)
    ]

    assert [int(frame["observation.images.wrist"][0, 0, 0]) for frame in frames] == [
        10,
        10,
        20,
        20,
    ]
    np.testing.assert_allclose(
        [float(frame["wrist"][0]) for frame in frames],
        [0.05, 0.15, 0.05, 0.15],
        atol=1e-6,
    )
    image_cache = cache["resampled"]["observation.images.wrist"]
    source_s = cache["timestamp_seconds"]["camera/timestamp_us"]
    selected_s = source_s[image_cache["indices"]]
    assert np.all(selected_s <= cache["target_timestamps"])


def test_dual_rate_rejects_large_numeric_interpolation_gap() -> None:
    with pytest.raises(ValueError, match="exceeds max_gap_s"):
        _dataset()._validate_resample_targets(
            source_times=np.asarray([1.0, 1.2]),
            targets=np.asarray([1.1]),
            method="linear",
            max_gap_s=0.051,
            feature_name="observation.joint",
            h5_path=Path("episode.h5"),
        )


def test_pose_resampling_uses_slerp_and_preserves_unit_quaternion() -> None:
    poses = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 1],
            [2, 0, 0, 0, 0, 1, 0],
        ],
        dtype=np.float64,
    )

    result = _dataset()._resample_values(
        poses,
        source_times=np.asarray([0.0, 2.0]),
        targets=np.asarray([1.0]),
        method="pose",
        transform=None,
    )

    np.testing.assert_allclose(result[0, :3], [1, 0, 0], atol=1e-7)
    np.testing.assert_allclose(
        result[0, 3:], [0, 0, np.sqrt(0.5), np.sqrt(0.5)], atol=1e-7
    )
    np.testing.assert_allclose(np.linalg.norm(result[0, 3:]), 1.0, atol=1e-7)


def test_lerobot_adapter_finalizes_buffered_metadata() -> None:
    class FakeDataset:
        finalized = False

        @classmethod
        def create(cls, **_kwargs):
            return cls()

        def finalize(self):
            self.finalized = True

    adapter = LeRobotV3Dataset(
        FakeDataset,
        repo_id="local/test",
        root=Path("unused"),
        fps=10,
        features={},
        no_videos=True,
    )

    adapter.finalize()

    assert adapter.dataset.finalized is True


@pytest.mark.parametrize(
    ("align", "window_size", "expected_x"),
    [
        ("nearest", 1, np.asarray(1.0, dtype=np.float32)),
        ("nearest_future_window", 3, np.asarray([1.0, 2.0, 2.0], dtype=np.float32)),
    ],
)
def test_pose_transform_runs_after_configured_alignment(
    align: str,
    window_size: int,
    expected_x: np.ndarray,
) -> None:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 3, axis=0)
    poses[:, 0, 3] = np.arange(3, dtype=np.float32)
    cache = {
        "datasets": {
            "teleop/ee_pose": poses,
            "teleop/timestamp_us": np.asarray([100, 200, 300], dtype=np.int64),
        },
        "timestamps": {
            "teleop/timestamp_us": np.asarray([100, 200, 300], dtype=np.int64),
        },
    }
    mapping = {
        "h5_paths": ["teleop/ee_pose"],
        "timestamp_path": "teleop/timestamp_us",
        "align": align,
        "window_size": window_size,
        "transform": "ee_pose_matrix_to_quaternion",
        "feature_spec": {"dtype": "float32"},
    }

    result = _dataset()._read_mapped_value(
        h5_file=None,
        mapping=mapping,
        frame_idx=0,
        target_t=250,
        h5_path=Path("episode.h5"),
        cache=cache,
    )

    np.testing.assert_allclose(result[..., 0], expected_x, atol=1e-6)
    expected_quaternion = np.broadcast_to(
        np.asarray([0, 0, 0, 1], dtype=np.float32),
        result[..., 3:].shape,
    )
    np.testing.assert_allclose(result[..., 3:], expected_quaternion, atol=1e-6)
