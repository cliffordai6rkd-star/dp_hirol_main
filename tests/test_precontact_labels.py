"""Episode-aligned phase and alpha targets without decoding image frames."""

import numpy as np
import pytest
import torch

from diffusion_policy.common.contact_phase import contact_phase_alpha
from diffusion_policy.dataset import lerobot_v3_dataset as v3
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.contact_curriculum import HysteresisContactDetector, VectorThresholdGate
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor


@pytest.mark.parametrize("interval", [.05, .04])
def test_precontact_alpha_uses_elapsed_time_and_repeated_contact_onsets(interval):
    times = np.arange(50, dtype=np.float32) * interval
    contact = np.zeros(50, dtype=bool)
    contact[16:20] = True
    contact[35:39] = True
    phase, alpha = contact_phase_alpha(times, contact, [range(50)], .5)
    assert (phase[contact] == 2).all()
    np.testing.assert_array_equal(alpha[contact], np.ones(contact.sum()))
    for onset in (16, 35):
        remaining = times[onset].astype(np.float64) - times.astype(np.float64)
        before = (~contact) & (remaining > 0) & (remaining <= .5 + 1e-6)
        assert (phase[before] == 1).all()
        np.testing.assert_allclose(alpha[before], np.clip(1 - remaining[before] / .5, 0, 1), atol=1e-7)
    assert phase[0] == 0 and alpha[0] == 0
    assert phase[21] == 0 and alpha[21] == 0
    assert phase[45] == 0 and alpha[45] == 0


def test_precontact_boundary_and_episode_reset_with_absolute_clocks():
    times = np.tile(np.arange(12) * .05, 2) + 1000.
    contact = np.zeros(24, dtype=bool)
    contact[10:12] = True
    contact[14:17] = True
    phase, alpha = contact_phase_alpha(times, contact, [range(12), range(12, 24)], .5)
    assert phase[0] == 1 and alpha[0] == 0
    assert alpha[5] == pytest.approx(.5)
    assert phase[12] == 1 and alpha[12] == pytest.approx(.8)
    assert phase[17] == 0 and alpha[17] == 0
    assert phase[-1] == 0 and alpha[-1] == 0
    no_ramp_phase, no_ramp_alpha = contact_phase_alpha(times, contact, [range(12), range(12, 24)], 0)
    np.testing.assert_array_equal(no_ramp_phase, contact.astype(np.int64) * 2)
    np.testing.assert_array_equal(no_ramp_alpha, contact.astype(np.float32))


def make_dataset(monkeypatch, *, ramp=True, phase_resampling=True):
    episode_length = 40
    tau = np.zeros((80, 7), dtype=np.float32)
    tau[14:20, 0] = 10.
    tau[28:32, 0] = 11.
    tau[42:47, 0] = 12.
    columns = {
        "timestamp": np.tile((np.arange(episode_length) * .05).astype(np.float32), 2),
        "observation.tau_ext": tau,
        "action": np.zeros((80, 8), dtype=np.float32),
    }

    class FakeV3:
        episode_data_index = {"from": [0, 40], "to": [40, 80]}

        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 80

        def get_column(self, key):
            return columns[key]

        def close(self):
            pass

    monkeypatch.setattr(v3, "LeRobotV3Dataset", FakeV3)
    preprocessor = WrenchPreprocessor(
        2, "tau_ext", "tau_ext", None,
        HysteresisContactDetector(8., 8., force_dims=tuple(range(7)), norm="l2",
                                 enter_frames=2, exit_frames=2),
        VectorThresholdGate(0., enabled=False), LinearNormalizer(),
    )
    label_config = preprocessor.get_config()
    # Omitted duration exercises saved binary-alpha configurations unchanged.
    label_config.pop("precontact_seconds", None)
    if ramp:
        label_config["precontact_seconds"] = .5
    dataset = v3.LeRobotV3ImageDataset(
        shape_meta={"obs": {"tau_ext": {"shape": [8, 7], "type": "low_dim"}},
                    "action": {"shape": [8]}},
        dataset_path="unused", horizon=3, n_obs_steps=2, pad_before=1,
        val_ratio=.5, seed=42, window_sampling_strategy="timestamp", timestamp_step_sec=.05,
        action_feature_fields=["action"],
        windowed_lowdim_feature_groups={
            "tau_ext": {"source_key": "observation.tau_ext", "steps": 8,
                        "start_offset": -7, "stride": 1},
        },
        phase_resampling_enabled=phase_resampling, phase_source_key="observation.tau_ext",
        contact_threshold=8., contact_force_dims=tuple(range(7)), contact_norm="l2",
        contact_enter_frames=2, contact_exit_frames=2, precontact_seconds=.5,
        contact_label_config=label_config, video_decode_device="cpu",
    )
    return dataset, tau, columns["timestamp"], label_config


def test_dataset_labels_align_each_observation_and_share_validation_state(monkeypatch):
    dataset, tau, times, _ = make_dataset(monkeypatch)
    contact = np.zeros(80, dtype=bool)
    # The episode scan confirms contact/release after two consecutive frames.
    contact[15:21] = True
    contact[29:33] = True
    contact[43:48] = True
    phase, alpha = contact_phase_alpha(times, contact, [range(40), range(40, 80)], .5)
    for view in (dataset, dataset.get_validation_dataset(), dataset.get_all_episodes_dataset()):
        assert view._frame_contact_phase is dataset._frame_contact_phase
        assert view._frame_contact_alpha is dataset._frame_contact_alpha
        for index in range(len(view)):
            sequence = view._retime_sequence_indices(view._sample_indices_to_sequence(index))
            obs_rows = sequence[:2]
            obs = view[index]["obs"]
            assert obs["contact_phase"].dtype == torch.int64
            assert obs["contact_alpha"].dtype == torch.float32
            np.testing.assert_array_equal(obs["contact_phase"], phase[obs_rows])
            np.testing.assert_array_equal(obs["contact_alpha"], alpha[obs_rows])
            assert view.get_phase_labels()[index] == phase[sequence[view.anchor_position]]
            for step, row in enumerate(obs_rows):
                episode_start = (row // 40) * 40
                history = np.clip(np.arange(row - 7, row + 1), episode_start, row)
                np.testing.assert_array_equal(obs["tau_ext"][step], tau[history])
    assert dataset._frame_contact_alpha[38:40].sum() == 0
    assert dataset._frame_contact_alpha[40] == pytest.approx(.7)


def test_dataset_ramp_is_supervision_not_a_resampling_dependency_and_legacy_is_binary(monkeypatch):
    ramp, _, _, _ = make_dataset(monkeypatch, phase_resampling=False)
    evaluation = ramp.get_all_episodes_dataset()
    alpha = torch.cat([evaluation[index]["obs"]["contact_alpha"] for index in range(len(evaluation))])
    assert ((alpha > 0) & (alpha < 1)).any()
    assert (evaluation.get_phase_labels() == 1).any()
    legacy, _, _, config = make_dataset(monkeypatch, ramp=False)
    assert "precontact_seconds" not in config
    assert legacy._frame_contact_phase is None
    for index in range(len(legacy)):
        obs = legacy[index]["obs"]
        assert "contact_phase" not in obs
        assert ((obs["contact_alpha"] == 0) | (obs["contact_alpha"] == 1)).all()
