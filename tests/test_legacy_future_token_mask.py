"""Reference future-contact gating and image-token masking, restored together."""

import pytest
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from test_configurable_force_fusion import IMAGE_SHAPES, make_encoder
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.render_net import RenderNet
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor
from diffusion_policy.policy.force_aware_diffusion_transformer_policy import ForceAwareDiffusionTransformerPolicy
from diffusion_policy.scripts.visualize_render_net_episode import render_observations
from diffusion_policy.workspace.train_distill_render_net_workspace import student_context_tokens


def teacher_and_obs():
    encoder = make_encoder(
        wrench_dim=7, cross_attn_query="image", cross_attn_layers=1,
        force_temporal_encoder="lstm", image_token_masking=True,
    )
    teacher = ForceAwareDiffusionTransformerPolicy(
        shape_meta={"obs": {
            **{key: {"shape": shape, "type": "rgb"} for key, shape in IMAGE_SHAPES.items()},
            "tau_ext": {"shape": [8, 7], "type": "low_dim"},
            "future_tau_ext": {"shape": [8, 7], "type": "low_dim"},
        }, "action": {"shape": [7]}},
        noise_scheduler=DDIMScheduler(num_train_timesteps=10),
        horizon=9, n_obs_steps=2, n_action_steps=8,
        wrench_key="future_tau_ext", phase_wrench_key="tau_ext",
        future_phase_wrench_key="future_tau_ext", future_contact_mode="token_threshold",
        contact_fusion_gate=True, contact_force_dims=list(range(7)), contact_norm="l1",
        contact_threshold=12.5, free_threshold=12.5, contact_enter_frames=5, contact_exit_frames=5,
        mask_schedule={"schedule_type": "piecewise", "total_steps": 100000,
                       "piecewise_fractions": [0., .25, .75, 1.],
                       "piecewise_probabilities": [1., 1., 0., 0.]},
        obs_encoder=encoder, n_emb=16, n_head=4, n_layer=1, p_drop_attn=0.,
        action_representation="joint_position",
    ).eval()
    normalizer = LinearNormalizer()
    normalizer.fit({"tau_ext": torch.randn(32, 7), "future_tau_ext": torch.randn(32, 7),
                    "action": torch.randn(32, 7)})
    teacher.set_normalizer(normalizer)
    obs = {key: torch.rand(2, 2, *shape) for key, shape in IMAGE_SHAPES.items()}
    obs["tau_ext"] = torch.zeros(2, 2, 8, 7)
    obs["future_tau_ext"] = torch.zeros(2, 2, 8, 7)
    # A single future token must open complete fusion, even with debounce=5.
    # It has zero continuous alpha in the newer (m-1)/(K-1) formulation.
    obs["future_tau_ext"][1, :, -1, 6] = 13.
    obs["contact_alpha"] = torch.zeros(2, 2)
    return teacher, obs


def test_one_future_contact_opens_full_fusion_and_free_bypasses_ffn():
    teacher, obs = teacher_and_obs()
    images, wrench, _, future = teacher.prepare_context_inputs_with_future_contact(obs)
    with torch.no_grad():
        actual = teacher._encode_context(obs, apply_curriculum_mask=False)
        visual = teacher.obs_encoder.encode_image_context(images)
        full = teacher.obs_encoder(images=images, wrench_history=wrench)
    torch.testing.assert_close(actual[0], visual[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], full[1], rtol=0, atol=0)
    assert not torch.allclose(full[1], visual[1])
    assert future[1].sum(-1).tolist() == [1, 1]
    preprocessing = WrenchPreprocessor.from_policy(teacher)
    assert preprocessing.contact_alpha(obs).tolist() == [[0., 0.], [1., 1.]]
    restored = WrenchPreprocessor.from_config(preprocessing.get_config())
    restored.load_state_dict(preprocessing.state_dict(), strict=True)
    torch.testing.assert_close(restored.contact_alpha(obs), preprocessing.contact_alpha(obs))


@pytest.mark.parametrize("step,probability,masked", [
    (0, 1., True), (25000, 1., True), (50000, .5, None),
    (75000, 0., False), (100000, 0., False),
])
def test_curriculum_masks_tokens_after_dino_and_never_changes_pixels(step, probability, masked):
    teacher, obs = teacher_and_obs()
    teacher.set_optimizer_step(step)
    seen = {}

    def capture(module, args, kwargs):
        seen.update(kwargs)

    handle = teacher.obs_encoder.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        teacher._encode_context(obs, apply_curriculum_mask=True)
    finally:
        handle.remove()
    assert teacher.mask_scheduler.probability(step) == probability
    assert teacher.blur_scheduler is None
    for key in IMAGE_SHAPES:
        torch.testing.assert_close(seen["images"][key], obs[key], rtol=0, atol=0)
    assert not seen["image_token_mask"][0].any()
    if masked is not None:
        assert seen["image_token_mask"][1].all().item() == masked
    if masked:
        teacher.zero_grad()
        output = teacher._encode_context(obs, apply_curriculum_mask=True)
        (output * torch.randn_like(output)).sum().backward()
        assert teacher.obs_encoder.image_mask_token.grad.abs().sum() > 0
        assert all(p.grad is None for p in teacher.obs_encoder.backbone.parameters())


def test_student_training_and_rendering_use_binary_future_gate():
    teacher, obs = teacher_and_obs()
    renderer = RenderNet(
        7, base_channels=4, conditioning_dim=16, embed_dim=16, num_heads=4,
        wrench_layers=2, image_self_attn_layers=1, cross_attn_layers=1,
        decoder_layers=4, patch_size=16, image_keys=tuple(IMAGE_SHAPES),
        contact_fusion_gate=True, residual_scale=1.,
    ).eval()
    with torch.no_grad():
        renderer.output.bias.fill_(.25)
        images, rendered = render_observations(
            renderer, WrenchPreprocessor.from_policy(teacher), obs, tuple(IMAGE_SHAPES), 2,
        )
        tokens = student_context_tokens(teacher, renderer, obs)
        expected = teacher.obs_encoder.encode_image_context(rendered)
    for key in IMAGE_SHAPES:
        torch.testing.assert_close(rendered[key][0], images[key][0], rtol=0, atol=0)
        assert not torch.equal(rendered[key][1], images[key][1])
    torch.testing.assert_close(tokens, expected, rtol=0, atol=0)
