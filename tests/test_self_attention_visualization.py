from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from diffusion_policy.model.vision.dino_token_baseline_obs_encoder import DinoTokenBaselineObsEncoder
from diffusion_policy.scripts import visualize_cross_attention_episode as cross
from diffusion_policy.scripts import visualize_image_self_attn_episode as image
from diffusion_policy.scripts import visualize_wrench_image_self_attn_episode as wrench
from diffusion_policy.scripts.self_attention_visualization import (
    ImageSelfAttentionCapture,
    WrenchImageSelfAttentionCapture,
)


def make_encoder(mode, *, mixed_shapes=False, force_temporal_encoder="none"):
    config = DINOv3ViTConfig(
        image_size=24, patch_size=8, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_register_tokens=2,
    )
    backbone = DINOv3ViTModel(config).eval()
    shapes = {"side": [3, 16, 24], "wrist": [3, 16, 16 if mixed_shapes else 24]}
    encoder = DinoTokenBaselineObsEncoder(
        "unused", shapes, mode=mode, n_emb=16, n_head=4,
        wrench_history_steps=3, self_attn_layers=2, dropout=0.0, backbone=backbone,
        force_temporal_encoder=force_temporal_encoder,
        force_encoder_layers=2 if force_temporal_encoder == "gru" else 0,
    ).eval()
    policy = SimpleNamespace(obs_encoder=encoder, n_obs_steps=2, baseline_mode=mode)
    images = {key: torch.rand(1, 2, *shape) for key, shape in shapes.items()}
    return policy, images


@pytest.mark.parametrize("query_mode", ["all", "last"])
@pytest.mark.parametrize("force_temporal_encoder", ["none", "gru"])
def test_joint_capture_selects_latest_wrench_rows_and_correct_camera_columns(query_mode, force_temporal_encoder):
    policy, images = make_encoder("self_attn", force_temporal_encoder=force_temporal_encoder)
    encoder = policy.obs_encoder
    force = torch.randn(1, 2, 3, 6)
    with torch.inference_mode():
        original = encoder(images=images, wrench_history=force)
    attention = encoder.fusion.layers[-1].self_attn
    captured = []
    capture = WrenchImageSelfAttentionCapture(policy, query_mode=query_mode)
    with capture:
        handle = attention.register_forward_hook(
            lambda module, args, output: captured.append(output[1].detach().clone())
        )
        try:
            with torch.inference_mode():
                actual = encoder(images=images, wrench_history=force)
        finally:
            handle.remove()
        maps = capture.attention_by_camera()
    # Two observations of [side x6, wrist x6, wrench x3]. Only the
    # final fusion layer, latest wrench queries and latest camera keys count.
    full = captured[0]
    query = slice(27, 30) if query_mode == "all" else slice(29, 30)
    torch.testing.assert_close(maps["side"], full[0, :, query, 15:21].mean(dim=(0, 1)))
    torch.testing.assert_close(maps["wrist"], full[0, :, query, 21:27].mean(dim=(0, 1)))
    # Do not renormalize selected keys: some attention goes to other times/force.
    assert 0 < sum(values.sum() for values in maps.values()) < 1
    torch.testing.assert_close(actual, original, rtol=1e-5, atol=1e-6)
    assert not attention._forward_hooks
    assert not attention._forward_pre_hooks
    assert not encoder._forward_pre_hooks


@pytest.mark.parametrize("query_mode", ["all", "last"])
@pytest.mark.parametrize("mixed_shapes", [False, True])
def test_image_capture_uses_real_dino_attention_and_preserves_camera_time_mapping(query_mode, mixed_shapes):
    policy, images = make_encoder("dino", mixed_shapes=mixed_shapes)
    encoder = policy.obs_encoder
    attention = encoder.backbone.layer[-1].attention
    backend = attention.config._attn_implementation
    with torch.inference_mode():
        original = encoder(images=images)
    captured = []
    with ImageSelfAttentionCapture(policy, query_mode=query_mode) as capture:
        handle = attention.register_forward_hook(
            lambda module, args, output: captured.append(output[1].detach().clone())
        )
        try:
            with torch.inference_mode():
                actual = encoder(images=images)
        finally:
            handle.remove()
        maps = capture.attention_by_camera()
        assert attention.config._attn_implementation == backend
    assert len(captured) == (2 if mixed_shapes else 1)
    # One CLS and two registers precede spatial patches. Shape groups batch
    # cameras, each with observation 0 followed by observation 1.
    query = slice(3, None) if query_mode == "all" else slice(-1, None)
    expected_side = captured[0][1, :, query, 3:].mean(dim=(0, 1))
    wrist_weights = captured[1][1] if mixed_shapes else captured[0][3]
    expected_wrist = wrist_weights[:, query, 3:].mean(dim=(0, 1))
    torch.testing.assert_close(maps["side"], expected_side)
    torch.testing.assert_close(maps["wrist"], expected_wrist)
    assert maps["side"].shape == (6,)
    assert maps["wrist"].shape == ((4,) if mixed_shapes else (6,))
    assert all(0 < values.sum() < 1 for values in maps.values())
    torch.testing.assert_close(actual, original, rtol=1e-5, atol=1e-6)
    assert attention.config._attn_implementation == backend
    assert not attention._forward_hooks
    assert not attention._forward_pre_hooks
    assert not encoder._forward_pre_hooks


def test_dino_backend_and_hooks_restore_after_inference_failure(monkeypatch):
    policy, images = make_encoder("dino")
    attention = policy.obs_encoder.backbone.layer[-1].attention
    backend = attention.config._attn_implementation

    def fail(*args, **kwargs):
        assert attention.config._attn_implementation == "eager"
        raise RuntimeError("inference failed")

    monkeypatch.setattr(attention, "forward", fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        with ImageSelfAttentionCapture(policy):
            with torch.inference_mode():
                policy.obs_encoder(images=images)
    assert attention.config._attn_implementation == backend
    assert not attention._forward_hooks
    assert not attention._forward_pre_hooks
    assert not policy.obs_encoder._forward_pre_hooks


def test_cubic_heatmap_overshoot_does_not_wrap_low_and_high_colors():
    rendered = cross.overlay_attention(
        np.zeros((64, 64, 3), dtype=np.uint8),
        np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
        grid=(2, 2), alpha=1.0, normalization_bounds=(0.0, 1.0),
    )
    # Cubic interpolation overshoots at both edges of this step. The minimum
    # must remain JET blue and the maximum JET red, rather than wrapping uint8.
    np.testing.assert_array_equal(rendered[32, 0], [0, 0, 128])
    np.testing.assert_array_equal(rendered[32, -1], [128, 0, 0])


@pytest.mark.parametrize("module", [image, wrench])
def test_baseline_cli_accepts_the_same_options_as_cross_attention(module):
    args = [
        "--checkpoint", "some/model.ckpt", "--episode", "3", "--camera", "both",
        "--dataset-path", "some/dataset", "--video-backend", "pyav",
        "--output", "some/video.mp4", "--fps", "25", "--video-codec", "h264",
        "--stride", "2", "--max-frames", "10", "--alpha", "0.3",
        "--attention-query", "last", "--attention-normalization", "frame",
        "--seed", "17", "--device", "cpu", "--save-frames", "--overwrite",
    ]
    assert vars(module.parse_args(args)) == vars(cross.parse_args(args))
    default_args = ["--checkpoint-dir", "some/checkpoints", "--episode", "0"]
    assert vars(module.parse_args(default_args)) == vars(cross.parse_args(default_args))


@pytest.mark.parametrize("capture_class,mode", [
    (ImageSelfAttentionCapture, "self_attn"),
    (WrenchImageSelfAttentionCapture, "dino"),
    (ImageSelfAttentionCapture, None),
    (WrenchImageSelfAttentionCapture, None),
])
def test_wrong_model_is_rejected_with_matching_checkpoint_hint(capture_class, mode):
    with pytest.raises(ValueError, match="requires a train_dp_"):
        capture_class(SimpleNamespace(baseline_mode=mode))
