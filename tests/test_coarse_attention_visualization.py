"""Report unsupported coarse fusion maps without disabling backbone attention."""

from types import SimpleNamespace

import pytest
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from diffusion_policy.model.vision.force_aware_obs_encoder import ForceAwareObsEncoder
from diffusion_policy.scripts.checkpoint_attention_capture import AttentionCapture
from diffusion_policy.scripts.fusion_visual_attention_trace import FusionVisualAttentionTrace


def _coarse_policy():
    torch.manual_seed(29)
    backbone = DINOv3ViTModel(DINOv3ViTConfig(
        image_size=16, patch_size=8, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=4, num_register_tokens=2,
    ))
    shapes = {key: (3, 16, 16) for key in ("side", "wrist")}
    encoder = ForceAwareObsEncoder(
        "unused", image_shapes=shapes, backbone=backbone,
        n_emb=16, n_head=4, max_obs_steps=2, wrench_dim=7,
        wrench_history_steps=8, force_temporal_encoder="lstm",
        pre_cross_self_attn_layers=1, cross_attn_layers=0,
        fusion_downsample_factor=2, dropout=0.0,
    ).eval()
    policy = SimpleNamespace(obs_encoder=encoder, n_obs_steps=2)
    sample = {
        "images": {key: torch.rand(1, 2, *shape) for key, shape in shapes.items()},
        "wrench_history": torch.rand(1, 2, 8, 7),
    }
    return policy, sample


@pytest.mark.parametrize("mode", ["auto", "self-attn"])
def test_coarse_fusion_capture_fails_before_registering_hooks(mode):
    policy, _ = _coarse_policy()
    with pytest.raises(ValueError, match="Coarse residual fusion.*full-resolution"):
        AttentionCapture(policy, mode=mode)
    assert not policy.obs_encoder._forward_pre_hooks
    assert not policy.obs_encoder.pre_cross_self_attention.layers[0].self_attn._forward_hooks


def test_coarse_fusion_rollout_reports_unsupported_residual_routes():
    policy, _ = _coarse_policy()
    with pytest.raises(ValueError, match="Coarse residual fusion rollout.*full-resolution"):
        FusionVisualAttentionTrace(policy)
    assert not policy.obs_encoder._forward_pre_hooks


def test_coarse_policy_still_captures_dino_attention_without_changing_output():
    policy, sample = _coarse_policy()
    with torch.inference_mode():
        expected = policy.obs_encoder(**sample)
        with AttentionCapture(policy, mode="dino") as capture:
            actual = policy.obs_encoder(**sample)
            maps = capture.attention_by_camera()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert set(maps) == {"side", "wrist"}
    assert all(value.shape == (1, 4) for value in maps.values())
    assert not policy.obs_encoder._forward_pre_hooks
