"""CIVI torque histories stay causal, episode-local and physically zero padded."""

import copy

import numpy as np
import pytest
import torch

from diffusion_policy.dataset import lerobot_v3_dataset as dataset_module
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.contact_curriculum import ContactDetector, VectorThresholdGate
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor


@pytest.fixture(params=[25, 50])
def history_dataset(monkeypatch, request):
    history_steps = request.param
    size = 60
    torque = np.zeros((2 * size, 7), dtype=np.float32)
    torque[:, 0] = np.arange(2 * size) + 20
    torque[1, 0] = 12.49
    torque[2, 0] = 12.5
    columns = {
        "timestamp": np.tile(np.arange(size) * .04, 2),
        "observation.tau_ext": torque,
        "observation.joint": np.full((2 * size, 7), 3, dtype=np.float32),
        "observation.velocity": np.full((2 * size, 7), 4, dtype=np.float32),
        "action": np.zeros((2 * size, 7), dtype=np.float32),
    }

    class FakeV3:
        episode_data_index = {"from": [0, size], "to": [size, 2 * size]}

        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return size * 2

        def get_column(self, key):
            return columns[key]

    monkeypatch.setattr(dataset_module, "LeRobotV3Dataset", FakeV3)
    config = dict(
        shape_meta={"obs": {"tau_ext": {"shape": [history_steps, 7], "type": "low_dim"}},
                    "action": {"shape": [7]}},
        dataset_path="unused", horizon=2, n_obs_steps=2, pad_before=1,
        windowed_lowdim_feature_groups={
            "tau_ext": {"source_key": "observation.tau_ext", "steps": history_steps,
                        "start_offset": 1 - history_steps, "stride": 1, "pad_mode": "zero"},
        },
        wrench_threshold={"enabled": True, "source_key": "observation.tau_ext",
                          "threshold": 12.5, "norm": "l1"},
        phase_predictor={"enabled": True, "history_steps": history_steps,
                         "image_offsets": [-12, -8, -4, 0], "pad_mode": "zero",
                         "state_source_keys": ["observation.joint", "observation.velocity",
                                               "observation.tau_ext"]},
    )
    return dataset_module.LeRobotV3ImageDataset(**config), columns, config


def test_history50_threshold_padding_episode_reset_and_phase_input_agree(history_dataset):
    dataset, columns, _ = history_dataset
    windows = dataset.windowed_lowdim_data["tau_ext"]
    history_steps = windows.shape[1]
    for start in (0, 60):
        np.testing.assert_array_equal(windows[start, :-1], np.zeros((history_steps - 1, 7)))
        np.testing.assert_array_equal(windows[start, -1], columns["observation.tau_ext"][start])
    np.testing.assert_array_equal(windows[1, -1], np.zeros(7))
    assert windows[2, -1, 0] == 12.5
    np.testing.assert_array_equal(windows[55], columns["observation.tau_ext"][56 - history_steps:56])
    rows = np.array([0, 1, 2, 55, 60, 61])
    inputs = dataset._phase_inputs(rows, {})
    np.testing.assert_array_equal(inputs["state"][:, :, 14:], windows[rows])
    np.testing.assert_array_equal(inputs["state"][[0, 4], :-1], np.zeros((2, history_steps - 1, 21)))
    assert dataset._windowed_lowdim_valid_rows["tau_ext"].all()


def test_history50_cannot_read_future_torque(history_dataset):
    dataset, columns, config = history_dataset
    columns["observation.tau_ext"][56:60] += 1000
    changed = dataset_module.LeRobotV3ImageDataset(**config)
    np.testing.assert_array_equal(changed.windowed_lowdim_data["tau_ext"][:56],
                                  dataset.windowed_lowdim_data["tau_ext"][:56])
    np.testing.assert_array_equal(changed._phase_inputs(np.array([55]), {})["state"],
                                  dataset._phase_inputs(np.array([55]), {})["state"])


def test_timestamp_history50_zero_padding_uses_episode_clock_boundaries():
    dataset = dataset_module.LeRobotV3ImageDataset.__new__(dataset_module.LeRobotV3ImageDataset)
    dataset.dataset_length = 6
    dataset.episode_ranges = [range(3), range(3, 6)]
    dataset.timestamps = np.tile(np.arange(3) * .04, 2)
    dataset._window_anchor_timestamps = {"clock": dataset.timestamps}
    source = np.arange(6, dtype=np.float32)[:, None] + 20
    valid = np.zeros(6, dtype=bool)
    windows = dataset._build_windowed_lowdim(source, {
        "steps": 50, "stride": 1, "start_offset": -49, "pad_mode": "zero",
        "anchor_timestamp_key": "clock", "source_timestamp_key": "clock",
        "offsets_sec": (np.arange(-49, 1) * .04).tolist(),
    }, valid_rows=valid)
    assert valid.all()
    np.testing.assert_array_equal(windows[3, :-1], np.zeros((49, 1)))
    np.testing.assert_array_equal(windows[5, -3:], source[3:6])


def test_preprocessor_threshold_and_zero_padding_survive_nonzero_mean_normalizer():
    normalizer = LinearNormalizer()
    normalizer.fit({"tau_ext": torch.arange(70).reshape(10, 7).float() + 10}, mode="gaussian")
    processor = WrenchPreprocessor(
        2, "tau_ext", "tau_ext", None,
        ContactDetector(12.5, force_dims=tuple(range(7)), history_reducer="last"),
        VectorThresholdGate(12.5, norm="l1", enabled=True), normalizer,
        precontact_seconds=.5, fusion_gate_mode="phase_binary", preserve_zero_wrench=True,
    )
    torque = torch.zeros(1, 2, 50, 7)
    torque[0, 0, -2, 0] = 12.49
    torque[0, 0, -1, 0] = 12.5
    torque[0, 1, -1, 0] = 20
    assert normalizer["tau_ext"].normalize(torque)[0, 0, 0].abs().sum() > 0
    normalized = processor({"tau_ext": torque})[0]
    assert torch.count_nonzero(normalized[0, 0, :-1]) == 0
    torch.testing.assert_close(normalized[0, 0, -1], normalizer["tau_ext"].normalize(torque[0, 0, -1]))
    restored = WrenchPreprocessor.from_config(processor.get_config())
    restored.load_state_dict(processor.state_dict())
    torch.testing.assert_close(restored({"tau_ext": torque})[0], normalized)
    legacy = copy.deepcopy(processor.get_config())
    del legacy["preserve_zero_wrench"]
    assert not WrenchPreprocessor.from_config(legacy).preserve_zero_wrench
