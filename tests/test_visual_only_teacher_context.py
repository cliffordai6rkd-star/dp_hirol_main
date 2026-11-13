"""Visual-only teacher outputs retain the internal torque-to-vision path."""

import io

import hydra
import pytest
import torch
from omegaconf import OmegaConf

from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.model.vision.force_aware_obs_encoder import ForceAwareObsEncoder
from diffusion_policy.policy import force_aware_diffusion_transformer_policy as policy_module
from test_civi_action_heads import _PatchBackbone, _batch
from test_coarse_residual_fusion import _encoder, _observations


@pytest.fixture(autouse=True, scope="module")
def _cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_backbone(monkeypatch):
    def create_encoder(**kwargs):
        return ForceAwareObsEncoder(**kwargs, backbone=_PatchBackbone())

    monkeypatch.setattr(policy_module, "ForceAwareObsEncoder", create_encoder)


def _policy_config(head, mode, upsample=None):
    config = {
        "_target_": (
            "diffusion_policy.policy.force_aware_diffusion_transformer_policy."
            "ForceAwareDiffusionTransformerPolicy"
        ),
        "shape_meta": {"obs": {
            "side": {"shape": [3, 16, 16], "type": "rgb"},
            "wrist": {"shape": [3, 16, 16], "type": "rgb"},
            "tau_ext": {"shape": [8, 7], "type": "low_dim"},
        }, "action": {"shape": [7]}},
        "noise_scheduler": None if head == "act" else {
            "_target_": "diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            "num_train_timesteps": 10,
            "beta_schedule": "squaredcos_cap_v2",
            "prediction_type": "epsilon",
        },
        "horizon": 9, "n_action_steps": 8, "n_obs_steps": 2,
        "wrench_key": "tau_ext", "wrench_history_steps": 8,
        "n_emb": 16, "n_head": 4, "n_layer": 1, "n_cond_layers": 0,
        "force_temporal_encoder": "lstm", "force_encoder_layers": 2,
        "p_drop_emb": 0.0, "p_drop_attn": 0.0,
        "action_head_backbone": head, "action_representation": "joint_position",
        "precontact_seconds": 0.5, "contact_force_dims": list(range(7)),
        "pre_cross_self_attn_layers": 1, "cross_attn_layers": 0,
        "fusion_downsample_factor": 2, "num_inference_steps": 2,
    }
    if mode is not None:
        config["fusion_output_mode"] = mode
    if upsample is not None:
        config["fusion_upsample"] = upsample
        config["fusion_highres_skip"] = upsample
    return OmegaConf.create(config)


def _policy(config):
    torch.manual_seed(23)
    policy = hydra.utils.instantiate(config)
    normalizer = LinearNormalizer()
    for key in ("tau_ext", "action"):
        normalizer[key] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)
    return policy


def _assert_gradients(module):
    gradients = [p.grad for p in module.parameters() if p.requires_grad]
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum().item() for g in gradients) > 0


@pytest.mark.parametrize("factor,skip,upsample,visual_count", [
    (1, True, True, 48), (2, True, True, 48),
    (2, False, True, 48), (2, False, False, 12),
])
def test_visual_output_exactly_matches_all_output_with_per_observation_torque_removed(factor, skip, upsample, visual_count):
    options = dict(factor=factor, fusion_highres_skip=skip, fusion_upsample=upsample)
    complete = _encoder(fusion_output_mode="all", **options)
    visual = _encoder(fusion_output_mode="visual", **options)
    visual.load_state_dict(complete.state_dict(), strict=True)
    assert complete.state_dict().keys() == visual.state_dict().keys()
    images, torque = _observations(complete)
    with torch.no_grad():
        full_context = complete(images=images, wrench_history=torque)
        visual_context = visual(images=images, wrench_history=torque)
    # Preserve each camera and observation position; dropping the last 16
    # tokens of the flattened sequence would corrupt the second observation.
    per_step = full_context.reshape(2, 2, visual_count + 8, 16)
    expected = per_step[:, :, :visual_count].flatten(1, 2)
    torch.testing.assert_close(visual_context, expected, rtol=0, atol=0)
    assert visual.context_tokens_per_observation == visual_count
    assert visual_context.shape == (2, 2 * visual_count, 16)


def test_coarse_output_is_joint_attention_visual_slice_without_interpolation_or_skip(monkeypatch):
    encoder = _encoder(fusion_output_mode="visual", fusion_upsample=False,
                       fusion_highres_skip=False)
    images, torque = _observations(encoder)
    captured = []
    handle = encoder.pre_cross_self_attention.register_forward_hook(
        lambda module, args, output: captured.append(output.detach().clone())
    )

    def forbidden_interpolation(*args, **kwargs):
        pytest.fail("coarse output must never upsample the fused visual tokens")

    monkeypatch.setattr(torch.nn.functional, "interpolate", forbidden_interpolation)
    try:
        with torch.no_grad():
            context = encoder(images=images, wrench_history=torque)
            expected = encoder.context_norm(captured[0][:, :12].reshape(2, 24, 16))
    finally:
        handle.remove()
    assert len(captured) == 1
    assert captured[0].shape == (4, 12 + 8, 16)
    assert encoder.context_tokens_per_observation == 12
    assert context.shape == (2, 24, 16)
    torch.testing.assert_close(context, expected, rtol=0, atol=0)


@pytest.mark.parametrize("upsample", [True, False])
def test_fixed_gray_images_still_receive_torque_information_and_gradients(upsample):
    encoder = _encoder(fusion_output_mode="visual", fusion_upsample=upsample,
                       fusion_highres_skip=upsample).train()
    images, torque = _observations(encoder)
    images = {key: torch.full_like(value, 0.5) for key, value in images.items()}
    torque.requires_grad_()
    seen = []
    handle = encoder.pre_cross_self_attention.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach().clone())
    )
    try:
        context = encoder(images=images, wrench_history=torque)
        zero_context = encoder(images=images, wrench_history=torch.zeros_like(torque))
    finally:
        handle.remove()
    # Both rectangular camera grids pool to six visual tokens. Only the
    # following eight LSTM tokens differ at the input to joint attention.
    torch.testing.assert_close(seen[0][:, :12], seen[1][:, :12], rtol=0, atol=0)
    assert not torch.allclose(seen[0][:, 12:], seen[1][:, 12:])
    assert not torch.allclose(context, zero_context)
    target = torch.randn(context.shape, generator=torch.Generator().manual_seed(71))
    torch.nn.functional.mse_loss(context, target).backward()
    for module in (encoder.wrench_projection, encoder.force_encoder,
                   encoder.pre_cross_self_attention):
        _assert_gradients(module)
    assert torch.isfinite(torque.grad).all()
    assert (torque.grad.abs().sum(dim=(0, 1, 3)) > 0).all()
    assert all(not p.requires_grad and p.grad is None for p in encoder.backbone.parameters())


@pytest.mark.parametrize("head", ["act", "dit"])
@pytest.mark.parametrize("upsample", [True, False])
def test_action_heads_train_and_predict_from_visual_context(head, upsample, tiny_backbone):
    policy = _policy(_policy_config(head, "visual", upsample)).train()
    batch = _batch()
    batch["obs"]["tau_ext"].requires_grad_()
    received = []
    handle = policy.model.register_forward_pre_hook(
        lambda module, args: received.append(tuple(args[-1].shape))
    )
    try:
        loss = policy.compute_loss(batch, generator=torch.Generator().manual_seed(8))
        assert torch.isfinite(loss)
        loss.backward()
    finally:
        handle.remove()
    # Each camera contributes its original 2x2 grid or its single pooled token.
    context_count = 16 if upsample else 4
    assert received == [(2, context_count, 16)]
    assert policy.obs_encoder.fusion_output_mode == "visual"
    assert policy.obs_encoder.fusion_upsample is upsample
    assert policy.obs_encoder.context_tokens_per_observation == context_count // 2
    if head == "dit":
        assert policy.model.max_context_tokens == context_count
    for module in (policy.obs_encoder.force_encoder,
                   policy.obs_encoder.pre_cross_self_attention, policy.model.decoder):
        _assert_gradients(module)
    assert (batch["obs"]["tau_ext"].grad.abs().sum(dim=(0, 1, 3)) > 0).all()
    assert all(p.grad is None for p in policy.obs_encoder.backbone.parameters())
    with torch.no_grad():
        prediction = policy.eval().predict_action(
            batch["obs"], generator=torch.Generator().manual_seed(9),
        )
    assert prediction["action"].shape == (2, 8, 7)
    assert torch.isfinite(prediction["action"]).all()


@pytest.mark.parametrize("head", ["act", "dit"])
@pytest.mark.parametrize("mode,upsample,context_count", [
    (None, None, 32), ("visual", True, 16), ("visual", False, 4),
])
def test_serialized_config_and_state_preserve_legacy_and_visual_predictions(head, mode, upsample, context_count, tiny_backbone):
    config = _policy_config(head, mode, upsample)
    original = _policy(config).eval()
    observations = _batch()["obs"]
    with torch.no_grad():
        expected_context = original._encode_context(observations, apply_curriculum_mask=False)
        expected = original.predict_action(observations, generator=torch.Generator().manual_seed(13))
    buffer = io.BytesIO()
    torch.save({"config": OmegaConf.to_container(config, resolve=True),
                "state_dict": original.state_dict()}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    if mode is None:
        assert "fusion_output_mode" not in saved["config"]
        assert "fusion_upsample" not in saved["config"]
    restored = _policy(OmegaConf.create(saved["config"])).eval()
    restored.load_state_dict(saved["state_dict"], strict=True)
    assert restored.obs_encoder.fusion_output_mode == (mode or "all")
    assert restored.obs_encoder.fusion_upsample is (True if upsample is None else upsample)
    with torch.no_grad():
        actual_context = restored._encode_context(observations, apply_curriculum_mask=False)
        actual = restored.predict_action(observations, generator=torch.Generator().manual_seed(13))
    assert actual_context.shape == (2, context_count, 16)
    torch.testing.assert_close(actual_context, expected_context, rtol=0, atol=0)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [
    {"fusion_output_mode": "force"},
    {"fusion_output_mode": True},
    {"fusion_output_mode": "visual", "factor": 1, "pre_cross_self_attn_layers": 0},
    {"fusion_output_mode": "visual", "factor": 1, "cross_attn_layers": 1},
])
def test_invalid_visual_output_combinations_are_rejected(kwargs):
    with pytest.raises(ValueError, match="fusion_output_mode"):
        _encoder(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"fusion_upsample": "false"},
    {"fusion_upsample": False, "fusion_highres_skip": True},
    {"factor": 1, "fusion_upsample": False, "fusion_highres_skip": False},
])
def test_invalid_coarse_output_combinations_are_rejected(kwargs):
    with pytest.raises(ValueError, match="fusion_upsample|fusion_highres_skip"):
        _encoder(fusion_output_mode="visual", **kwargs)
