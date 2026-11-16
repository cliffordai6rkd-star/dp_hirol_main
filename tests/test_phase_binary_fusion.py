"""November 7 whole-fusion bypass, adapted to board phase labels and blur."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from test_contact_alpha import make_current_teacher, current_observations
from test_configurable_force_fusion import IMAGE_SHAPES
from diffusion_policy.common.config_cli import load_config_reference
from diffusion_policy.model.vision.contact_curriculum import FactrBlurScheduler, gaussian_2d_smoothing
from diffusion_policy.model.vision.render_net import RenderNet
from diffusion_policy.model.vision.render_net_attention import teacher_context_effect_targets
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor, fusion_gate_from_phase
from diffusion_policy.scripts.visualize_render_net_episode import render_observations
from diffusion_policy.workspace.train_distill_render_net_workspace import student_context_tokens


def teacher_and_obs():
    teacher = make_current_teacher()
    teacher.precontact_seconds = .5
    teacher.fusion_gate_mode = "phase_binary"
    obs = current_observations()
    obs["wrench"].fill_(.2)
    obs["contact_phase"] = torch.tensor([[0, 1], [2, 0]])
    # Include the zero-alpha precontact boundary and contradictory free labels.
    obs["contact_alpha"] = torch.tensor([[1., 0.], [.3, 1.]])
    return teacher, obs


def test_phase_gate_bypasses_both_ffns_and_opens_full_attention_on_precontact():
    teacher, obs = teacher_and_obs()
    images, wrench, _, _ = teacher.prepare_context_inputs_with_future_contact(obs)
    gate = fusion_gate_from_phase(obs, 2)
    with torch.no_grad():
        actual = teacher._encode_context(obs, apply_curriculum_mask=False)
        visual = teacher.obs_encoder.encode_image_context(images)
        full = teacher.obs_encoder(images=images, wrench_history=wrench)
        ffn_only = teacher.obs_encoder.encode_image_context(images, apply_fusion_ffn=True)
    expected = torch.where(gate[..., None, None], full.reshape(2, 2, -1, 16),
                           visual.reshape(2, 2, -1, 16)).reshape_as(actual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.allclose(visual, ffn_only)
    assert not torch.allclose(actual[0], visual[0])
    with torch.no_grad():
        changed_alpha = teacher._encode_context(
            {**obs, "contact_alpha": 1 - obs["contact_alpha"]}, apply_curriculum_mask=False,
        )
    torch.testing.assert_close(actual, changed_alpha, rtol=0, atol=0)


@pytest.mark.parametrize("phase,active", [(0, False), (1, True), (2, True)])
def test_gradient_reaches_attention_and_ffn_only_for_open_phases(phase, active):
    teacher, obs = teacher_and_obs()
    teacher.train()
    obs["contact_phase"].fill_(phase)
    context = teacher._encode_context(obs, apply_curriculum_mask=False)
    (context * torch.randn_like(context)).mean().backward()
    for block in (teacher.obs_encoder, *teacher.obs_encoder.extra_cross_attention_layers):
        for parameter in (block.cross_attention.out_proj.weight, block.cross_ffn[0].weight):
            assert parameter.grad is not None
            assert bool(parameter.grad.abs().sum() > 0) == active
    assert teacher.obs_encoder.image_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.obs_encoder.backbone.parameters())


def test_phase_labels_are_required_even_when_blur_is_disabled():
    teacher, obs = teacher_and_obs()
    obs.pop("contact_phase")
    with pytest.raises(ValueError, match="phase_binary fusion requires"):
        teacher._encode_context(obs, apply_curriculum_mask=False)


def test_raw_zero_targets_close_fusion_and_free_effect_is_exactly_zero():
    teacher, obs = teacher_and_obs()
    teacher.requires_grad_(False)
    normal, effects, zero = teacher_context_effect_targets(teacher, obs, return_zero=True)
    images = {key: obs[key] for key in IMAGE_SHAPES}
    visual = teacher.obs_encoder.encode_image_context(images)
    torch.testing.assert_close(zero, visual, rtol=0, atol=0)
    gate = fusion_gate_from_phase(obs, 2)
    for values in effects.values():
        assert torch.count_nonzero(values[~gate]) == 0
        assert torch.all(values[gate] > 0)
    with torch.no_grad():
        ordinary = teacher._encode_context(obs, False)
    torch.testing.assert_close(normal, ordinary, rtol=0, atol=0)


def test_student_training_inference_and_preprocessor_roundtrip_share_binary_gate():
    teacher, obs = teacher_and_obs()
    teacher.requires_grad_(False)
    renderer = RenderNet(
        6, base_channels=4, conditioning_dim=16, embed_dim=16, num_heads=4,
        image_self_attn_layers=1, cross_attn_layers=1, decoder_layers=4,
        patch_size=16, image_keys=tuple(IMAGE_SHAPES),
    ).eval()
    with torch.no_grad():
        renderer.output.bias.fill_(.25)
    calls = []
    handle = renderer.register_forward_hook(
        lambda module, args, kwargs, output: calls.append((kwargs, output)), with_kwargs=True,
    )
    try:
        tokens, contact = student_context_tokens(teacher, renderer, obs, return_contact=True)
        preprocessor = WrenchPreprocessor.from_policy(teacher)
        restored = WrenchPreprocessor.from_config(preprocessor.get_config())
        restored.load_state_dict(preprocessor.state_dict(), strict=True)
        assert restored.fusion_gate_mode == "phase_binary"
        images, inference_rgb = render_observations(renderer, restored, obs, tuple(IMAGE_SHAPES), 2)
    finally:
        handle.remove()
    gate = fusion_gate_from_phase(obs, 2)
    torch.testing.assert_close(contact.squeeze(-1), gate, rtol=0, atol=0)
    for kwargs, _ in calls:
        torch.testing.assert_close(kwargs["contact_alpha"], gate.float(), rtol=0, atol=0)
    train_rgb = calls[0][1][0]
    for key in IMAGE_SHAPES:
        torch.testing.assert_close(train_rgb[key], inference_rgb[key], rtol=0, atol=0)
        torch.testing.assert_close(train_rgb[key][~gate], images[key][~gate], rtol=0, atol=0)
        assert not torch.equal(train_rgb[key][gate], images[key][gate])
    expected = teacher.obs_encoder.encode_image_context(inference_rgb, input_grad=True)
    torch.testing.assert_close(tokens, expected, rtol=0, atol=0)
    target = teacher._encode_context(obs, False).detach()
    torch.nn.functional.mse_loss(tokens, target).backward()
    assert renderer.output.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.parameters())


def test_phase_binary_fusion_preserves_selective_pixel_blur_and_clear_targets():
    teacher, obs = teacher_and_obs()
    teacher.image_blur_scope = "precontact_contact"
    teacher.blur_scheduler = FactrBlurScheduler(
        total_steps=150000, start_fraction=1 / 3, end_fraction=1.,
    )
    seen = []
    handle = teacher.obs_encoder.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs), with_kwargs=True,
    )
    try:
        teacher.set_optimizer_step(50000)
        teacher._encode_context(obs, apply_curriculum_mask=True)
        teacher._encode_context(obs, apply_curriculum_mask=False)
    finally:
        handle.remove()
    gate = fusion_gate_from_phase(obs, 2)
    for key in IMAGE_SHAPES:
        expected = obs[key].clone()
        expected[gate] = gaussian_2d_smoothing(obs[key][gate], 5.)
        torch.testing.assert_close(seen[0]["images"][key], expected)
        torch.testing.assert_close(seen[1]["images"][key], obs[key], rtol=0, atol=0)
    torch.testing.assert_close(seen[0]["fusion_gate"], gate, rtol=0, atol=0)
    assert "contact_alpha" not in seen[0]


def test_reproduction_configs_match_confirmed_teacher_and_old_student():
    root = Path(__file__).resolve().parents[1] / "diffusion_policy/config"
    teacher = load_config_reference(str(root / "train_civi_teacher_board_blur_legacy_fusion.yaml"))
    student = load_config_reference(str(root / "train_distill_render_net_board_legacy.yaml"))
    assert teacher.policy.fusion_gate_mode == "phase_binary"
    assert teacher.policy.cross_attn_layers == 2
    assert teacher.policy.precontact_seconds == .5
    assert teacher.policy.future_phase_wrench_key is None
    assert teacher.task.dataset.windowed_lowdim_feature_groups.tau_ext.start_offset == -7
    assert teacher.shape_meta.obs.tau_ext.shape == [8, 7]
    assert teacher.training.num_optimizer_steps == 150000
    assert teacher.dataloader.batch_size == 32
    schedule = FactrBlurScheduler(**OmegaConf.to_container(teacher.policy.image_blur_schedule))
    schedule.set_total_steps(teacher.training.num_optimizer_steps)
    assert [schedule.sigma(step) for step in (0, 50000, 100000, 150000)] == pytest.approx([5, 5, 2.5, 0])
    assert student.render_net.backbone.image_self_attn_layers == 4
    assert student.render_net.backbone.cross_attn_layers == 2
    assert student.render_net.backbone.decoder_layers == 4
    assert student.render_net.backbone.patch_size == 16
    assert student.render_net.high_attention_render.threshold == .5
    assert student.render_net.residual_scale == 1.
    assert student.render_net.separate_gate_path
    assert student.render_net.spatial_modulation.enabled
    assert student.training.render_stages.intensity_epochs == 5
    assert student.dataloader.batch_size == 16
    assert student.training.early_stopping.enabled
    assert student.training.early_stopping.patience == 5
    assert student.training.early_stopping.start_at_rgb
    assert student.distillation.objective == "token_mse"
    assert not student.distillation.use_teacher_ffn
