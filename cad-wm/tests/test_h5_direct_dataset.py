from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from data_process.dataloader import PINNDataset
from data_process.h5_direct_dataset import DirectH5EpisodeDataset


def _write_episode(path: Path, episode: int, frames: int = 5, dt_us: int = 10_000):
    del episode
    with h5py.File(path, "w") as h5_file:
        teleop = h5_file.create_group("teleop")
        timestamps = 1_000_000 + np.arange(frames, dtype=np.int64) * dt_us
        q = np.arange(frames * 2, dtype=np.float64).reshape(frames, 2)
        teleop.create_dataset("timestamp_us", data=timestamps)
        teleop.create_dataset("q_follower", data=q)
        teleop.create_dataset("q_cmd", data=q + 0.25)
        teleop.create_dataset("dq_follower", data=q + 10.0)
        teleop.create_dataset("tau_follower", data=q + 20.0)


def _fields():
    return {
        "observation.joint": "teleop/q_follower",
        "observation.velocity": "teleop/dq_follower",
        "observation.delta_q": {
            "operation": "subtract",
            "paths": ["teleop/q_cmd", "teleop/q_follower"],
        },
        "observation.torque": "teleop/tau_follower",
    }


def test_direct_h5_preserves_rows_and_raw_episode_ids(tmp_path):
    _write_episode(tmp_path / "episode_0007_first.h5", 7)
    _write_episode(tmp_path / "episode_0015_second.h5", 15)

    dataset = DirectH5EpisodeDataset(
        root=tmp_path,
        fields=_fields(),
        timestamp_path="teleop/timestamp_us",
        expected_fps=100,
    )

    assert [episode["episode_index"] for episode in dataset.meta.episodes] == [7, 15]
    assert dataset.meta.episodes[0]["dataset_from_index"] == 0
    assert dataset.meta.episodes[1]["dataset_from_index"] == 5
    torch.testing.assert_close(
        dataset.hf_dataset[:]["observation.delta_q"],
        torch.full((10, 2), 0.25),
    )
    torch.testing.assert_close(
        dataset.hf_dataset[:]["timestamp"][:5],
        torch.tensor([1.00, 1.01, 1.02, 1.03, 1.04], dtype=torch.float64),
    )


def test_direct_h5_rejects_cadence_mismatch_instead_of_resampling(tmp_path):
    _write_episode(tmp_path / "episode_0000_bad.h5", 0, dt_us=20_000)

    with pytest.raises(ValueError, match="does not resample"):
        DirectH5EpisodeDataset(
            root=tmp_path,
            fields=_fields(),
            timestamp_path="teleop/timestamp_us",
            expected_fps=100,
        )


def test_pinn_dataset_builds_episode_safe_windows_from_h5(tmp_path):
    _write_episode(tmp_path / "episode_0007_first.h5", 7)
    _write_episode(tmp_path / "episode_0015_second.h5", 15)
    config = {
        "dataloader": {
            "backend": "h5",
            "root": str(tmp_path),
            "load_images": False,
            "horizon": 3,
            "pad_history": False,
            "expected_fps": 100,
            "lowdim_keys": {
                "q": "observation.joint",
                "dq": "observation.velocity",
                "delta_q": "observation.delta_q",
                "tau": "observation.torque",
            },
            "h5_fields": _fields(),
            "normalize_mode": None,
        }
    }

    dataset = PINNDataset(config, compute_normalizer=False)

    assert len(dataset) == 6
    assert dataset.valid_indices == [2, 3, 4, 7, 8, 9]
    assert dataset[3]["q"].shape == (3, 2)
    torch.testing.assert_close(
        dataset[3]["q"],
        dataset.stats_dataset[:]["observation.joint"][5:8],
    )


def test_pinn_dataset_items_and_stats_share_the_filtered_view(tmp_path):
    _write_episode(tmp_path / "episode_0000_filter.h5", 0)
    config = {
        "dataloader": {
            "backend": "h5",
            "root": str(tmp_path),
            "load_images": False,
            "horizon": 1,
            "expected_fps": 100,
            "lowdim_keys": {"tau": "observation.torque"},
            "h5_fields": {"observation.torque": "teleop/tau_follower"},
            "filters": {
                "tau": {
                    "enabled": True,
                    "operations": [{"type": "moving_average", "window": 3}],
                }
            },
            "normalize_mode": None,
        }
    }

    dataset = PINNDataset(config, compute_normalizer=False)

    filtered = dataset.stats_dataset[:]["observation.torque"]
    torch.testing.assert_close(dataset[2]["tau"][0], filtered[2])
    assert dataset.sample_rate_hz == 100.0
