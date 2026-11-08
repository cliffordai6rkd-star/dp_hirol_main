import numpy as np
import torch

from data_process.causal_data_filter import (
    build_filtered_dataset_view,
    filter_episode_values,
)
from data_process.h5_direct_dataset import TensorColumnDataset


def test_filter_chain_is_causal_and_operation_order_is_preserved():
    timestamps = np.arange(6, dtype=np.float64) * 0.01
    values = np.asarray([[0.0], [9.0], [0.0], [3.0], [6.0], [0.0]])
    operations = [
        {"type": "median", "window": 3},
        {"type": "moving_average", "window": 2},
        {"type": "lowpass", "cutoff_hz": 10.0},
    ]

    reference = filter_episode_values(timestamps, values, operations)
    changed_future = values.copy()
    changed_future[4:] = 1_000.0
    changed = filter_episode_values(timestamps, changed_future, operations)
    reversed_order = filter_episode_values(timestamps, values, operations[::-1])

    np.testing.assert_allclose(changed[:4], reference[:4])
    assert not np.allclose(reversed_order, reference)


def test_filtered_dataset_view_resets_each_episode_and_is_used_for_reads():
    dataset = TensorColumnDataset(
        {
            "timestamp": torch.tensor([0.00, 0.01, 0.02, 0.00, 0.01, 0.02]),
            "observation.torque": torch.tensor(
                [[0.0], [10.0], [0.0], [5.0], [50.0], [5.0]]
            ),
        }
    )
    episodes = [
        {"dataset_from_index": 0, "dataset_to_index": 3},
        {"dataset_from_index": 3, "dataset_to_index": 6},
    ]
    view, canonical = build_filtered_dataset_view(
        dataset,
        data_config={
            "filter_timestamp_key": "timestamp",
            "filters": {
                "tau": {
                    "enabled": True,
                    "operations": [{"type": "median", "window": 3}],
                }
            },
        },
        lowdim_keys={"tau": "observation.torque"},
        episodes=episodes,
    )

    expected = torch.tensor([[0.0], [0.0], [0.0], [5.0], [5.0], [5.0]])
    torch.testing.assert_close(view[:]["observation.torque"], expected)
    torch.testing.assert_close(view[3]["observation.torque"], expected[3])
    assert canonical["tau"]["operations"] == [{"type": "median", "window": 3}]


def test_preprocessed_prefix_is_saved_but_not_applied_twice():
    dataset = TensorColumnDataset(
        {
            "timestamp": torch.tensor([0.00, 0.01, 0.02]),
            "observation.torque": torch.tensor([[0.0], [10.0], [0.0]]),
        }
    )
    lowpass = {"type": "lowpass", "cutoff_hz": 10.0}
    median = {"type": "median", "window": 3}
    view, canonical = build_filtered_dataset_view(
        dataset,
        data_config={
            "filters": {
                "tau": {
                    "enabled": True,
                    "dataset_preprocessed_operations": [lowpass],
                    "operations": [lowpass, median],
                }
            }
        },
        lowdim_keys={"tau": "observation.torque"},
        episodes=[{"dataset_from_index": 0, "dataset_to_index": 3}],
    )

    torch.testing.assert_close(
        view[:]["observation.torque"],
        torch.zeros(3, 1),
    )
    assert canonical["tau"]["dataset_preprocessed_operations"] == [lowpass]


def test_filtered_dataset_view_converts_integer_microsecond_timestamps():
    dataset = TensorColumnDataset(
        {
            "timestamp_us": torch.tensor([0, 10_000, 30_000], dtype=torch.int64),
            "observation.torque": torch.tensor([[0.0], [1.0], [1.0]]),
        }
    )
    view, _ = build_filtered_dataset_view(
        dataset,
        data_config={
            "filter_timestamp_key": "timestamp_us",
            "filter_timestamp_unit": "us",
            "filters": {
                "tau": {
                    "enabled": True,
                    "operations": [{"type": "lowpass", "cutoff_hz": 10.0}],
                }
            },
        },
        lowdim_keys={"tau": "observation.torque"},
        episodes=[{"dataset_from_index": 0, "dataset_to_index": 3}],
    )

    alpha_1 = 1.0 - np.exp(-2.0 * np.pi * 10.0 * 0.01)
    alpha_2 = 1.0 - np.exp(-2.0 * np.pi * 10.0 * 0.02)
    expected = torch.tensor(
        [[0.0], [alpha_1], [alpha_2 + (1.0 - alpha_2) * alpha_1]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(view[:]["observation.torque"], expected)
