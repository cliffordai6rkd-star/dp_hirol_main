from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.vision.force_aware_obs_encoder import ForceAwareObsEncoder
from diffusion_policy.scripts.image_query_attention_capture import ImageQueryAttentionCapture


class PatchBackbone(nn.Module):
    """Small image-dependent patch encoder with real variable-size grids."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, patch_size=2, num_register_tokens=0)
        self.projection = nn.Linear(3, 4)

    def forward(self, pixel_values):
        patches = F.avg_pool2d(pixel_values, 2).flatten(2).transpose(1, 2)
        patches = self.projection(patches)
        return SimpleNamespace(last_hidden_state=torch.cat([patches.mean(1, keepdim=True), patches], 1))


def make_policy(camera_mode="joint", layers=2, query="image"):
    torch.manual_seed(9)
    encoder = ForceAwareObsEncoder(
        pretrained_model_name_or_path="unused",
        n_emb=4,
        n_head=2,
        max_obs_steps=2,
        wrench_history_steps=3,
        force_temporal_encoder="lstm",
        dropout=0.0,
        image_shapes={"side": (3, 2, 4), "wrist": (3, 4, 4)},
        camera_attention_mode=camera_mode,
        backbone=PatchBackbone(),
        cross_attn_layers=layers,
        cross_attn_query=query,
    ).eval()
    return SimpleNamespace(obs_encoder=encoder, n_obs_steps=2)


def inputs():
    return {
        "images": {"side": torch.rand(1, 2, 3, 2, 4), "wrist": torch.rand(1, 2, 3, 4, 4)},
        "wrench_history": torch.randn(1, 2, 3, 6),
    }


@pytest.mark.parametrize("camera_mode", ["joint", "per_camera"])
@pytest.mark.parametrize("layer_index", [0, 1, -1, -2])
def test_capture_matches_real_mha_latest_observation_and_preserves_model(camera_mode, layer_index):
    policy = make_policy(camera_mode)
    encoder = policy.obs_encoder
    sample = inputs()
    before = {key: value.clone() for key, value in encoder.state_dict().items()}
    with torch.no_grad():
        original_output = encoder(**sample)
        with ImageQueryAttentionCapture(policy, layer_index=layer_index) as capture:
            records = []
            handle = capture.attention.register_forward_hook(
                lambda module, args, output: records.append(tuple(x.detach().clone() for x in output))
            )
            try:
                captured_output = encoder(**sample)
            finally:
                handle.remove()
            attention = capture.attention_by_camera()
            updates = capture.update_by_camera()
            assert not encoder.capture_cross_attention
    torch.testing.assert_close(captured_output, original_output, rtol=1e-5, atol=1e-6)
    assert encoder.state_dict().keys() == before.keys()
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert not capture.attention._forward_hooks
    assert not capture.attention._forward_pre_hooks
    assert not encoder._forward_pre_hooks
    assert len(records) == (1 if camera_mode == "joint" else 2)
    offset = 0
    for index, key in enumerate(encoder.image_keys):
        output, weights = records[0 if camera_mode == "joint" else index]
        patches = encoder.patch_counts[key]
        start = offset if camera_mode == "joint" else 0
        expected_attention = weights[1, :, start:start + patches].mean(0).T
        expected_update = output[1, start:start + patches].square().sum(-1).sqrt()
        torch.testing.assert_close(attention[key], expected_attention)
        torch.testing.assert_close(updates[key], expected_update)
        torch.testing.assert_close(attention[key].sum(0), torch.ones(patches))
        assert attention[key].shape == (3, patches)
        assert updates[key].shape == (patches,)
        assert attention[key].device.type == updates[key].device.type == "cpu"
        assert attention[key].dtype == updates[key].dtype == torch.float32
        assert not attention[key].requires_grad and not updates[key].requires_grad
        offset += patches


def test_update_is_projected_attention_channel_norm_before_residual_and_ffn():
    policy = make_policy(layers=1)
    encoder = policy.obs_encoder
    # Make the downstream FFN contribution enormous; it must not enter DeltaV.
    with torch.no_grad():
        encoder.cross_ffn[-2].bias.fill_(10000)

    def known_attention(module, query, key, value, **kwargs):
        assert kwargs["need_weights"] is True
        assert kwargs["average_attn_weights"] is False
        batch, patches, _ = query.shape
        update = query.new_zeros(batch, patches, 4)
        update[0] = 1000  # Earlier observation must be excluded.
        update[1, :, :2] = query.new_tensor([3.0, 4.0])
        update[1] *= torch.arange(1, patches + 1).view(-1, 1)
        head_weights = query.new_tensor([[0.1, 0.3, 0.6], [0.5, 0.2, 0.3]])
        weights = head_weights[None, :, None, :].expand(batch, 2, patches, 3).clone()
        weights[0] = query.new_tensor([0.0, 0.0, 1.0])
        return update, weights

    encoder.cross_attention.forward = MethodType(known_attention, encoder.cross_attention)
    with ImageQueryAttentionCapture(policy) as capture:
        encoder(**inputs())
        attention = capture.attention_by_camera()
        updates = capture.update_by_camera()
    torch.testing.assert_close(updates["side"], torch.tensor([5.0, 10.0]))
    torch.testing.assert_close(updates["wrist"], torch.tensor([15.0, 20.0, 25.0, 30.0]))
    for value in attention.values():
        torch.testing.assert_close(value, torch.tensor([0.3, 0.25, 0.45])[:, None].expand_as(value))


@pytest.mark.parametrize("layers,query,message", [
    (0, "image", "at least one"),
    (1, "wrench", "cross_attn_query"),
])
def test_rejects_architectures_without_image_query_cross_attention(layers, query, message):
    with pytest.raises(ValueError, match=message):
        ImageQueryAttentionCapture(make_policy(layers=layers, query=query))


@pytest.mark.parametrize("layer_index", [-3, 2, 1.5])
def test_rejects_invalid_layer_index(layer_index):
    with pytest.raises(ValueError, match="layer_index"):
        ImageQueryAttentionCapture(make_policy(), layer_index=layer_index)


def test_capture_resets_between_forwards_and_removes_hooks_on_error():
    policy = make_policy()
    capture = ImageQueryAttentionCapture(policy)
    with pytest.raises(ValueError, match="one sample"):
        with capture:
            with pytest.raises(RuntimeError, match="No complete"):
                capture.attention_by_camera()
            policy.obs_encoder(**inputs())
            sample = inputs()
            sample["images"]["side"] = sample["images"]["side"].repeat(2, 1, 1, 1, 1)
            policy.obs_encoder(**sample)
    assert not capture.active
    assert not policy.obs_encoder._forward_pre_hooks
    assert not capture.attention._forward_pre_hooks
    assert not capture.attention._forward_hooks
    with pytest.raises(RuntimeError, match="No complete"):
        capture.update_by_camera()
    # The same capture object can be reused without stale maps.
    with capture:
        with pytest.raises(RuntimeError, match="No complete"):
            capture.update_by_camera()
        policy.obs_encoder(**inputs())
        assert set(capture.attention_by_camera()) == {"side", "wrist"}


def test_existing_capture_flag_is_not_modified():
    policy = make_policy()
    policy.obs_encoder.set_attention_capture(True)
    with ImageQueryAttentionCapture(policy) as capture:
        policy.obs_encoder(**inputs())
        capture.attention_by_camera()
        assert policy.obs_encoder.capture_cross_attention is True
    assert policy.obs_encoder.capture_cross_attention is True
