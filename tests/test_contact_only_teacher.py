"""Causal two-state labels, balanced sampling, fusion bypass and masking."""

from pathlib import Path

import numpy as np
import pytest
import torch

from test_configurable_force_fusion import IMAGE_SHAPES, make_policy
from test_precontact_labels import make_dataset
from diffusion_policy.common.config_cli import load_config_reference
from diffusion_policy.common.phase_sampler import PhaseWeightedDistributedSampler
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor
from diffusion_policy.model.vision.render_net_attention import zero_wrench_reference


def test_contact_only_labels_keep_five_frame_debounce_without_precontact(monkeypatch):
    dataset, tau, _, config = make_dataset(monkeypatch)
    config.update(precontact_seconds=0.0)
    config['contact_detector'].update(enter_frames=5, exit_frames=5, history_reducer='last')
    dataset.set_contact_label_config(config)
    expected = np.zeros(80, dtype=np.int64)
    expected[18:24] = 2
    expected[46:51] = 2
    # The four-frame second pulse at 28:32 never establishes contact.
    np.testing.assert_array_equal(dataset._frame_contact_phase, expected)
    assert not (dataset.get_phase_labels() == 1).any()
    for view in (dataset, dataset.get_validation_dataset()):
        for index in range(len(view)):
            sequence = view._retime_sequence_indices(view._sample_indices_to_sequence(index))
            np.testing.assert_array_equal(view[index]['obs']['contact_phase'], expected[sequence[:2]])
    restored = WrenchPreprocessor.from_config(config)
    assert restored.precontact_seconds == 0.0
    assert restored.contact_detector.enter_frames == 5


def test_two_state_sampler_balances_free_contact_and_preserves_empty_pre_pool():
    labels = np.array([0] * 9000 + [2] * 1000, dtype=np.int8)
    sampler = PhaseWeightedDistributedSampler(
        list(range(len(labels))), labels, {0: 1., 1: 0., 2: 1.}, seed=42,
    )
    observed = labels[list(sampler)]
    assert sampler.expected_phase_fractions == {0: .5, 1: 0., 2: .5}
    assert not (observed == 1).any()
    assert (observed == 2).mean() == pytest.approx(.5, abs=.02)
    with pytest.raises(ValueError, match='positive-weight phase pools'):
        PhaseWeightedDistributedSampler(list(range(len(labels))), labels, {0: 1., 1: 1., 2: 1.})


def test_contact_only_fusion_uses_causal_episode_labels_and_masks_only_contact(monkeypatch):
    teacher = make_policy(
        monkeypatch, cross_attn_query='image', cross_attn_layers=2,
        force_temporal_encoder='lstm', force_encoder_layers=2,
        precontact_seconds=0., preserve_zero_wrench=True,
        fusion_gate_mode='current_contact', image_mask_scope='current_contact',
        contact_enter_frames=5, contact_exit_frames=5,
        contact_threshold=8., free_threshold=8., contact_history_reducer='last',
        mask_schedule={'schedule_type': 'constant', 'start_probability': 1.},
    ).eval()
    obs = {key: torch.rand(2, 2, *shape) for key, shape in IMAGE_SHAPES.items()}
    obs['wrench_ext'] = torch.zeros(2, 2, 8, 6)
    obs['contact_phase'] = torch.tensor([[0, 2], [2, 0]])
    obs['contact_alpha'] = torch.ones(2, 2)
    captured = []
    handle = teacher.obs_encoder.register_forward_pre_hook(
        lambda module, args, kwargs: captured.append(kwargs), with_kwargs=True,
    )
    with torch.no_grad():
        actual = teacher._encode_context(obs, apply_curriculum_mask=True)
        visual = teacher.obs_encoder.encode_image_context({key: obs[key] for key in IMAGE_SHAPES})
    handle.remove()
    expected_gate = obs['contact_phase'] == 2
    torch.testing.assert_close(captured[0]['fusion_gate'], expected_gate)
    torch.testing.assert_close(captured[0]['image_token_mask'], expected_gate)
    torch.testing.assert_close(actual.reshape(2, 2, -1, 16)[~expected_gate],
                               visual.reshape(2, 2, -1, 16)[~expected_gate], rtol=0, atol=0)
    zero_reference = zero_wrench_reference(teacher, obs)
    assert not zero_reference['contact_phase'].any()
    zero_context = teacher._encode_context(zero_reference, False)
    torch.testing.assert_close(zero_context, visual, rtol=0, atol=0)
    obs['contact_phase'][0, 0] = 1
    with pytest.raises(ValueError, match='only free=0 and contact=2'):
        teacher._encode_context(obs, False)
    # With no dataset label (inference), the measured history drives the gate.
    obs.pop('contact_phase')
    with torch.no_grad():
        context = teacher._encode_context(obs, False)
    torch.testing.assert_close(context, visual, rtol=0, atol=0)


def test_contact_only_training_config_resolves_to_authorized_run():
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config_reference(str(repo / 'diffusion_policy/config/train_civi_teacher_contact_only.yaml'))
    assert cfg.tau_ext_history_steps == 25
    assert cfg.policy.cross_attn_layers == 2
    assert cfg.policy.precontact_seconds == 0
    assert cfg.policy.contact_enter_frames == cfg.policy.contact_exit_frames == 5
    assert cfg.policy.contact_history_reducer == 'last'
    assert cfg.policy.fusion_gate_mode == cfg.policy.image_mask_scope == 'current_contact'
    assert cfg.phase_resampling.weights.precontact == 0
    assert cfg.training.num_epochs == 600 and cfg.training.num_optimizer_steps is None
    assert cfg.dataloader.batch_size == 128
    assert cfg.task.dataset.windowed_lowdim_feature_groups.tau_ext.start_offset == -24
