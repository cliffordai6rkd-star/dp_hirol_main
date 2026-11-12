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


def make_current_teacher():
    teacher = make_teacher(
        shape_meta={"obs": {
            **{key: {"shape": shape, "type": "rgb"} for key, shape in IMAGE_SHAPES.items()},
            "wrench": {"shape": (8, 6), "type": "low_dim"},
        }, "action": {"shape": (8,)}},
        wrench_key="wrench", phase_wrench_key="wrench", future_phase_wrench_key=None,
        contact_force_dims=(0, 1, 2), contact_norm="l2",
        contact_threshold=8., free_threshold=8.,
    )
    normalizer = LinearNormalizer()
    normalizer.fit({
        "wrench": torch.arange(600).reshape(100, 6).float() + 5,
        "action": torch.randn(100, 8),
    }, mode="gaussian")
    teacher.set_normalizer(normalizer)
    return teacher


def current_observations():
    wrench = torch.zeros(2, 2, 8, 6)
    # Only the last five observed samples establish contact. Earlier samples
    # must retain their values instead of receiving the future-prefix fill.
    wrench[1, ..., 0] = torch.tensor([.1, .2, .3, 9., 10., 11., 12., 13.])
    wrench[1, ..., 3] = torch.arange(8).float()
    return {
        **{key: torch.rand(2, 2, *shape) for key, shape in IMAGE_SHAPES.items()},
        "wrench": wrench,
    }


def test_current_wrench_preprocessing_uses_only_observed_history_and_preserves_free_zero():
    teacher = make_current_teacher()
    obs = current_observations()
    preprocessor = WrenchPreprocessor.from_policy(teacher)
    assert preprocessor.required_keys == ("wrench",)
    assert teacher.future_phase_wrench_key is None

    _, normalized, contact, mask = teacher.prepare_context_inputs_with_future_contact(obs)
    assert contact.tolist() == [[False, False], [True, True]]
    assert mask.shape == (2, 2, 1)
    torch.testing.assert_close(mask.squeeze(-1), contact, rtol=0, atol=0)
    torch.testing.assert_close(preprocessor.contact_alpha(obs), contact.float(), rtol=0, atol=0)
    assert teacher.normalizer["wrench"].normalize(obs["wrench"][0]).count_nonzero() > 0
    assert normalized[0].count_nonzero() == 0
    assert teacher._latest_processed_wrench[0].count_nonzero() == 0
    torch.testing.assert_close(teacher._latest_processed_wrench[1], obs["wrench"][1], rtol=0, atol=0)
    torch.testing.assert_close(
        normalized[1], teacher.normalizer["wrench"].normalize(obs["wrench"][1]), rtol=0, atol=0,
    )

    restored = WrenchPreprocessor.from_config(preprocessor.get_config())
    restored.load_state_dict(preprocessor.state_dict(), strict=True)
    actual_wrench, actual_contact, actual_mask = restored.forward_with_future_contact(obs)
    torch.testing.assert_close(actual_wrench, normalized, rtol=0, atol=0)
    torch.testing.assert_close(actual_contact, contact, rtol=0, atol=0)
    torch.testing.assert_close(actual_mask, mask, rtol=0, atol=0)


def test_current_wrench_teacher_and_student_share_inputs_and_student_backpropagates():
    from diffusion_policy.model.vision.render_net import RenderNet
    from diffusion_policy.workspace.train_distill_render_net_workspace import student_context_tokens

    teacher = make_current_teacher().requires_grad_(False)
    obs = current_observations()
    obs["contact_alpha"] = WrenchPreprocessor.from_policy(teacher).contact_alpha(obs)
    renderer = RenderNet(
        6, base_channels=4, conditioning_dim=16, embed_dim=16, num_heads=4,
        image_self_attn_layers=1, cross_attn_layers=1, decoder_layers=3,
        patch_size=8, image_keys=tuple(IMAGE_SHAPES),
    ).train()
    with torch.no_grad():
        renderer.output.bias.fill_(.25)
    seen = {}

    def capture_teacher(module, args, kwargs):
        seen["teacher_wrench"] = kwargs["wrench_history"].detach().clone()
        seen["teacher_mask"] = kwargs["future_contact"].clone()
        seen["teacher_alpha"] = kwargs["contact_alpha"].clone()

    def capture_student(module, args, kwargs, output):
        seen["student_wrench"] = args[1].detach().clone()
        seen["student_mask"] = kwargs["future_contact"].clone()
        seen["student_alpha"] = kwargs["contact_alpha"].clone()
        seen["rendered"] = output[0]

    handles = [
        teacher.obs_encoder.register_forward_pre_hook(capture_teacher, with_kwargs=True),
        renderer.register_forward_hook(capture_student, with_kwargs=True),
    ]
    try:
        with torch.no_grad():
            target = teacher._encode_context(obs, apply_curriculum_mask=False)
        tokens = student_context_tokens(teacher, renderer, obs)
    finally:
        for handle in handles:
            handle.remove()
    for field in ("wrench", "mask", "alpha"):
        torch.testing.assert_close(seen[f"student_{field}"], seen[f"teacher_{field}"], rtol=0, atol=0)
    for key in IMAGE_SHAPES:
        torch.testing.assert_close(seen["rendered"][key][0], obs[key][0], rtol=0, atol=0)
        assert not torch.equal(seen["rendered"][key][1], obs[key][1])
    loss = torch.nn.functional.mse_loss(tokens, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert renderer.output.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_current_wrench_teacher_trains_and_predicts_without_future_observations():
    teacher = make_current_teacher().train()
    obs = current_observations()
    loss = teacher.compute_loss({"obs": obs, "action": torch.randn(2, 9, 8)}, optimizer_step=0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert teacher.obs_encoder.wrench_projection.weight.grad.abs().sum() > 0
    teacher.eval()
    with torch.no_grad():
        prediction = teacher.predict_action(obs)
    assert prediction["action"].shape == (2, 8, 8)
    assert torch.isfinite(prediction["action"]).all()


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
