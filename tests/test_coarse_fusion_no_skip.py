"""The no-skip ablation removes only the high-resolution visual bypass."""

import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from diffusion_policy.model.vision.force_aware_obs_encoder import ForceAwareObsEncoder
from diffusion_policy.policy import force_aware_diffusion_transformer_policy as policy_module
from test_civi_action_heads import _batch, _make_policy
from test_civi_selfattn_config import compose_teacher
from test_coarse_residual_fusion import _encoder, _GridBackbone, _observations


@pytest.fixture(autouse=True, scope="module")
def _cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _restore_by_camera(encoder, coarse):
    restored = []
    offset = 0
    for key in encoder.image_keys:
        height, width = encoder.patch_grid_shapes[key]
        count = height * width // encoder.fusion_downsample_factor ** 2
        grid = coarse[:, offset:offset + count].transpose(1, 2).reshape(
            coarse.shape[0], encoder.n_emb,
            height // encoder.fusion_downsample_factor,
            width // encoder.fusion_downsample_factor,
        )
        restored.append(F.interpolate(
            grid, size=(height, width), mode="bilinear", align_corners=False,
        ).flatten(2).transpose(1, 2))
        offset += count
    return torch.cat(restored, dim=1)


def test_no_skip_upsamples_full_fused_output_and_keeps_torque_tokens():
    encoder = _encoder(fusion_highres_skip=False)
    images = torch.randn(4, 48, 16)
    torque = torch.randn(4, 8, 16)
    captured = {}
    hook = encoder.pre_cross_self_attention.register_forward_hook(
        lambda module, args, output: captured.update(input=args[0], output=output)
    )
    try:
        actual_images, actual_torque = encoder._coarse_residual_self_attention(
            images, torque, encoder.image_keys, [24, 24],
        )
    finally:
        hook.remove()
    # Each rectangular camera has six pooled tokens, followed by eight torque
    # tokens. Upsampling must not mix cameras or subtract the pooled base.
    expected = _restore_by_camera(encoder, captured["output"][:, :12])
    torch.testing.assert_close(actual_images, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_torque, captured["output"][:, 12:], rtol=0, atol=0)
    delta_only = _restore_by_camera(
        encoder, captured["output"][:, :12] - captured["input"][:, :12],
    )
    assert not torch.allclose(actual_images, delta_only)
    assert actual_images.shape == images.shape
    assert actual_torque.shape == torque.shape


def test_identity_fusion_without_skip_retains_the_coarse_visual_base():
    encoder = _encoder(fusion_highres_skip=False)
    encoder.pre_cross_self_attention = nn.Identity()
    images = torch.randn(2, 48, 16)
    torque = torch.randn(2, 8, 16)
    pooled = []
    for key, camera in zip(encoder.image_keys, images.split([24, 24], dim=1)):
        grid = camera.transpose(1, 2).reshape(2, 16, *encoder.patch_grid_shapes[key])
        pooled.append(F.avg_pool2d(grid, 2, 2).flatten(2).transpose(1, 2))
    actual, actual_torque = encoder._coarse_residual_self_attention(
        images, torque, encoder.image_keys, [24, 24],
    )
    expected = _restore_by_camera(encoder, torch.cat(pooled, dim=1))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_torque, torque, rtol=0, atol=0)
    assert actual.abs().sum() > 0


def test_default_and_explicit_skip_have_identical_outputs_and_parameters():
    default = _encoder()
    enabled = _encoder(fusion_highres_skip=True)
    disabled = _encoder(fusion_highres_skip=False)
    assert default.fusion_highres_skip is True
    assert default.state_dict().keys() == disabled.state_dict().keys()
    disabled.load_state_dict(default.state_dict(), strict=True)
    images, torque = _observations(default)
    torch.testing.assert_close(
        default(images=images, wrench_history=torque),
        enabled(images=images, wrench_history=torque), rtol=0, atol=0,
    )


def test_no_skip_preserves_full_action_context_shape():
    encoder = _encoder(
        fusion_highres_skip=False,
        image_shapes={"side": (3, 224, 224), "wrist": (3, 224, 224)},
        patch_size=16, n_emb=256,
    )
    images, torque = _observations(encoder, batch_size=1)
    with torch.no_grad():
        context = encoder(images=images, wrench_history=torque)
    assert encoder.fusion_tokens_per_observation == 106
    assert context.shape == (1, 800, 256)


@pytest.mark.parametrize("head", ["act", "dit"])
def test_both_heads_backpropagate_through_no_skip_fusion(head):
    policy = _make_policy(head, fusion_downsample_factor=2).train()
    policy.obs_encoder.fusion_highres_skip = False
    batch = _batch()
    batch["obs"]["tau_ext"].requires_grad_()
    loss = policy.compute_loss(batch, generator=torch.Generator().manual_seed(61))
    loss.backward()
    assert torch.isfinite(loss)
    for module in (
        policy.obs_encoder.image_projection, policy.obs_encoder.wrench_projection,
        policy.obs_encoder.force_encoder, policy.obs_encoder.pre_cross_self_attention,
        policy.model.decoder,
    ):
        gradients = [p.grad for p in module.parameters() if p.requires_grad]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum().item() for g in gradients) > 0
    assert (batch["obs"]["tau_ext"].grad.abs().sum(dim=(0, 1, 3)) > 0).all()


def test_policy_constructor_forwards_no_skip_setting(monkeypatch):
    def create_encoder(**kwargs):
        return ForceAwareObsEncoder(**kwargs, backbone=_GridBackbone())

    monkeypatch.setattr(policy_module, "ForceAwareObsEncoder", create_encoder)
    policy = policy_module.ForceAwareDiffusionTransformerPolicy(
        shape_meta={"obs": {
            "side": {"shape": [3, 16, 16], "type": "rgb"},
            "wrist": {"shape": [3, 16, 16], "type": "rgb"},
            "tau_ext": {"shape": [8, 7], "type": "low_dim"},
        }, "action": {"shape": [7]}},
        noise_scheduler=None, horizon=9, n_action_steps=8, n_obs_steps=2,
        wrench_key="tau_ext", n_emb=16, n_head=4, n_layer=1, n_cond_layers=0,
        action_head_backbone="act", action_representation="joint_position",
        pre_cross_self_attn_layers=1, cross_attn_layers=0,
        fusion_downsample_factor=2, fusion_highres_skip=False,
    )
    assert policy.obs_encoder.fusion_highres_skip is False


def test_no_skip_checkpoint_round_trip_preserves_predictions():
    policy = _make_policy(fusion_downsample_factor=2).eval()
    policy.obs_encoder.fusion_highres_skip = False
    obs = _batch()["obs"]
    expected = policy.predict_action(obs)
    buffer = io.BytesIO()
    torch.save(policy.state_dict(), buffer)
    buffer.seek(0)
    restored = _make_policy(fusion_downsample_factor=2).eval()
    restored.obs_encoder.fusion_highres_skip = False
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    actual = restored.predict_action(obs)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [
    {"factor": 1, "fusion_highres_skip": False},
    {"fusion_highres_skip": "false"},
    {"fusion_highres_skip": 0},
])
def test_invalid_no_skip_configuration_fails_explicitly(kwargs):
    with pytest.raises(ValueError, match="fusion_highres_skip"):
        _encoder(**kwargs)


@pytest.mark.parametrize("name", ["train_civi_teacher_selfattn", "train_civi_teacher_board_selfattn"])
def test_coarse_visual_default_and_explicit_legacy_upsampled_skip(name):
    default = compose_teacher(name)
    changed = compose_teacher(name, [
        "policy.fusion_highres_skip=true", "policy.fusion_upsample=true",
    ])
    assert default.policy.fusion_highres_skip is False
    assert default.policy.fusion_upsample is False
    assert default.policy.fusion_output_mode == "visual"
    assert changed.policy.fusion_highres_skip is True
    assert changed.policy.fusion_upsample is True
    assert changed.policy.fusion_downsample_factor == default.policy.fusion_downsample_factor == 2
