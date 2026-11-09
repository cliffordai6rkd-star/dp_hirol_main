"""Deterministic token-level contact alpha tests for the teacher fusion path."""

import torch
import pytest
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from test_configurable_force_fusion import IMAGE_SHAPES, make_encoder
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.wrench_preprocessing import (
    WrenchPreprocessor,
    compute_contact_alpha,
)
from diffusion_policy.policy.force_aware_diffusion_transformer_policy import (
    ForceAwareDiffusionTransformerPolicy,
)


def make_teacher(**overrides):
    torch.manual_seed(7)
    encoder = make_encoder(
        cross_attn_query="image", cross_attn_layers=2,
        pre_cross_self_attn_layers=0, force_temporal_encoder="lstm",
    )
    config = dict(
        shape_meta={"obs": {
            **{key: {"shape": shape, "type": "rgb"} for key, shape in IMAGE_SHAPES.items()},
            "tau_ext": {"shape": (8, 7), "type": "low_dim"},
            "future_tau_ext": {"shape": (8, 7), "type": "low_dim"},
            "future_wrench": {"shape": (8, 6), "type": "low_dim"},
        }, "action": {"shape": (7,)}},
        noise_scheduler=DDIMScheduler(num_train_timesteps=10, prediction_type="epsilon"),
        horizon=9, n_obs_steps=2, n_action_steps=8, num_inference_steps=2,
        obs_encoder=encoder, n_emb=16, n_head=4, n_layer=1, p_drop_attn=0.,
        wrench_key="future_wrench", phase_wrench_key="tau_ext",
        future_phase_wrench_key="future_tau_ext", contact_force_dims=(5, 6),
        contact_norm="l1", contact_threshold=2., free_threshold=1.,
        contact_enter_frames=5, contact_exit_frames=5,
        future_contact_mode="token_threshold",
    )
    config.update(overrides)
    policy = ForceAwareDiffusionTransformerPolicy(**config).eval()
    normalizer = LinearNormalizer()
    normalizer.fit({
        "future_wrench": torch.arange(600).reshape(100, 6).float() + 5,
        "action": torch.randn(100, 7),
    })
    policy.set_normalizer(normalizer)
    return policy


def observations(batch=2):
    return {
        **{key: torch.rand(batch, 2, *shape) for key, shape in IMAGE_SHAPES.items()},
        "tau_ext": torch.zeros(batch, 2, 8, 7),
        "future_tau_ext": torch.zeros(batch, 2, 8, 7),
        "future_wrench": torch.arange(batch * 2 * 8 * 6).reshape(batch, 2, 8, 6).float() + 1,
    }


def test_contact_alpha_formula_and_monotonicity():
    for token_count in (1, 2, 4, 8):
        values = []
        for contact_count in range(token_count + 1):
            mask = torch.zeros(1, 1, token_count, dtype=torch.bool)
            mask[..., :contact_count] = True
            values.append(float(compute_contact_alpha(mask).item()))
        expected = (
            [0.0, 1.0] if token_count == 1 else
            [0.0, 0.0, *[(m - 1) / (token_count - 1) for m in range(2, token_count)], 1.0]
        )
        assert values == pytest.approx(expected)
        assert all(left <= right for left, right in zip(values, values[1:]))


def test_alpha_zero_is_wrench_independent_and_alpha_one_matches_full_cross_attention():
    encoder = make_encoder(
        cross_attn_query="image", cross_attn_layers=2,
        pre_cross_self_attn_layers=0,
    ).eval()
    images = {key: torch.rand(2, 2, *shape) for key, shape in IMAGE_SHAPES.items()}
    wrench = torch.randn(2, 2, 8, 6)
    free = torch.zeros(2, 2, 4, dtype=torch.bool)
    full = torch.ones_like(free)
    with torch.no_grad():
        closed = encoder(images=images, wrench_history=wrench, future_contact=free)
        changed = encoder(images=images, wrench_history=torch.randn_like(wrench), future_contact=free)
        opened = encoder(images=images, wrench_history=wrench, future_contact=full)
        uncontrolled = encoder(images=images, wrench_history=wrench)
    torch.testing.assert_close(closed, changed, rtol=0, atol=0)
    torch.testing.assert_close(opened, uncontrolled, rtol=0, atol=0)


def test_partial_alpha_changes_the_update_continuously_between_closed_and_open():
    encoder = make_encoder(
        cross_attn_query="image", cross_attn_layers=1,
        pre_cross_self_attn_layers=0,
    ).eval()
    images = {key: torch.rand(1, 2, *shape) for key, shape in IMAGE_SHAPES.items()}
    wrench = torch.randn(1, 2, 8, 6)
    outputs = []
    with torch.no_grad():
        for contact_count in (0, 1, 2, 3, 4):
            mask = torch.zeros(1, 2, 4, dtype=torch.bool)
            mask[..., :contact_count] = True
            outputs.append(encoder(images=images, wrench_history=wrench, future_contact=mask))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    assert (outputs[2] - outputs[0]).abs().max() > 0
    assert (outputs[3] - outputs[2]).abs().max() > 0
    assert (outputs[4] - outputs[3]).abs().max() > 0


def test_full_future_contact_auxiliary_path_preserves_legacy_preprocessing_return():
    teacher = make_teacher()
    obs = observations()
    obs["future_tau_ext"][0, 0, -2:, 5] = 3.0
    _, _, _, reduced = teacher.prepare_context_inputs(obs)
    _, _, _, full = teacher.prepare_context_inputs_with_future_contact(obs)
    assert reduced.shape == (2, 2)
    assert full.shape == (2, 2, 8)
    assert torch.equal(reduced, full.any(dim=-1))
    preprocessor = WrenchPreprocessor.from_policy(teacher)
    assert preprocessor(obs)[2].shape == (2, 2)
    assert preprocessor.forward_with_future_contact(obs)[2].shape == (2, 2, 8)


def test_teacher_forward_smoke_uses_alpha_without_a_gate_argument():
    teacher = make_teacher()
    obs = observations()
    loss = teacher.compute_loss({"obs": obs, "action": torch.randn(2, 9, 7)}, optimizer_step=0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert not hasattr(teacher, "contact_fusion_gate")
    assert "contact_gate" not in teacher.obs_encoder.forward.__code__.co_varnames


def test_teacher_consumes_dataloader_alpha_and_raw_zero_ablation_discards_it():
    from diffusion_policy.scripts.visualize_wrench_effect_episode import make_zero_wrench_obs

    teacher = make_teacher()
    obs = observations()
    obs["future_tau_ext"][..., 5] = 3.0
    obs["contact_alpha"] = WrenchPreprocessor.from_policy(teacher).contact_alpha(obs)
    seen = []
    handle = teacher.obs_encoder.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs["contact_alpha"].clone()), with_kwargs=True,
    )
    try:
        teacher._encode_context(obs, apply_curriculum_mask=False)
    finally:
        handle.remove()
    torch.testing.assert_close(seen[0], obs["contact_alpha"], rtol=0, atol=0)
    zero = make_zero_wrench_obs(teacher, obs)
    assert "contact_alpha" not in zero and "contact_alpha" in obs
    assert WrenchPreprocessor.from_policy(teacher).contact_alpha(zero).count_nonzero() == 0


@pytest.mark.parametrize("mode", ["token_threshold", "legacy_rollout"])
def test_context_effect_targets_drop_cached_alpha_for_zero_wrench(mode):
    from diffusion_policy.model.vision.render_net_attention import teacher_context_effect_targets
    from diffusion_policy.scripts.visualize_wrench_effect_episode import make_zero_wrench_obs

    teacher = make_teacher(future_contact_mode=mode).eval().requires_grad_(False)
    obs = observations(3)
    # Include free, partially contacting and fully contacting future windows.
    contact_tokens = 4 if mode == "token_threshold" else 8
    obs["future_tau_ext"][1, :, -contact_tokens:, 5] = 3.
    obs["tau_ext"][2, ..., 5] = 3.
    obs["future_tau_ext"][2, ..., 5] = 3.
    obs["contact_alpha"] = WrenchPreprocessor.from_policy(teacher).contact_alpha(obs)
    assert obs["contact_alpha"][0].count_nonzero() == 0
    assert ((obs["contact_alpha"][1] > 0) & (obs["contact_alpha"][1] < 1)).all()
    assert (obs["contact_alpha"][2] == 1).all()
    original = {key: value.clone() for key, value in obs.items()}
    seen = []

    def capture_alpha(module, args, kwargs):
        alpha = kwargs.get("contact_alpha")
        seen.append((compute_contact_alpha(kwargs["future_contact"]) if alpha is None else alpha).clone())

    handle = teacher.obs_encoder.register_forward_pre_hook(capture_alpha, with_kwargs=True)
    try:
        tokens, maps = teacher_context_effect_targets(teacher, obs)
    finally:
        handle.remove()
    torch.testing.assert_close(seen[0], obs["contact_alpha"], rtol=0, atol=0)
    assert seen[1].count_nonzero() == 0
    with torch.no_grad():
        expected = teacher._encode_context(obs, apply_curriculum_mask=False)
        zero = teacher._encode_context(make_zero_wrench_obs(teacher, obs), apply_curriculum_mask=False)
    torch.testing.assert_close(tokens, expected, rtol=0, atol=0)
    patch_count = sum(teacher.obs_encoder.patch_counts.values())
    delta = (expected - zero).reshape(3, 2, patch_count, -1).norm(dim=-1)
    offset = 0
    for camera in teacher.obs_encoder.image_keys:
        end = offset + teacher.obs_encoder.patch_counts[camera]
        torch.testing.assert_close(maps[camera], delta[..., offset:end], rtol=0, atol=0)
        assert maps[camera][0].count_nonzero() == 0
        offset = end
    for key in original:
        torch.testing.assert_close(obs[key], original[key], rtol=0, atol=0)
