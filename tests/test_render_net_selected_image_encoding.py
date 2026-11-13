from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch
from torch import nn

from diffusion_policy.model.vision.force_aware_obs_encoder import ForceAwareObsEncoder


class BackboneOutput(NamedTuple):
    last_hidden_state: torch.Tensor


class PatchBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=6, patch_size=4, num_register_tokens=1)
        self.patch = nn.Conv2d(3, 6, kernel_size=4, stride=4)
        self.batch_sizes = []
        self.record_batch_sizes = True

    def forward(self, pixel_values):
        if self.record_batch_sizes:
            self.batch_sizes.append(pixel_values.shape[0])
        patches = self.patch(pixel_values).flatten(2).transpose(1, 2)
        prefix = patches.new_zeros(patches.shape[0], 2, 6)
        return BackboneOutput(last_hidden_state=torch.cat([prefix, patches], dim=1))


def make_encoder(*, different_camera_shapes=False):
    shapes = {"side": (3, 8, 12),
              "wrist": (3, 12, 8) if different_camera_shapes else (3, 8, 12)}
    encoder = ForceAwareObsEncoder(
        "unused", backbone=PatchBackbone(), n_emb=8, n_head=2, max_obs_steps=3,
        image_shapes=shapes, cross_attn_query="image", cross_attn_layers=2,
        dropout=0.0,
    ).eval().requires_grad_(False)
    return encoder, {key: torch.rand(2, 3, *shape) for key, shape in shapes.items()}


@pytest.mark.parametrize("apply_fusion_ffn", [False, True])
@pytest.mark.parametrize("different_camera_shapes", [False, True])
@pytest.mark.parametrize("pass_indices", [False, True])
def test_selected_observations_match_full_tokens_and_input_gradients(
    apply_fusion_ffn, different_camera_shapes, pass_indices,
):
    torch.manual_seed(31)
    encoder, images = make_encoder(different_camera_shapes=different_camera_shapes)
    full_images = {key: value.clone().requires_grad_() for key, value in images.items()}
    selected_images = {key: value.clone().requires_grad_() for key, value in images.items()}
    # Time 1 of batch 0 must retain its original time/camera embeddings.
    mask = torch.tensor([[False, True, False], [True, False, True]])
    indices = mask.flatten().nonzero().flatten()
    full = encoder.encode_image_context(
        full_images, input_grad=True, apply_fusion_ffn=apply_fusion_ffn,
    )
    encoder.backbone.batch_sizes.clear()
    selection = ({"selected_observation_indices": indices} if pass_indices
                 else {"observation_mask": mask})
    selected = encoder.encode_image_context(
        selected_images, input_grad=True, apply_fusion_ffn=apply_fusion_ffn, **selection,
    )
    full = full.reshape(2, 3, -1, 8)
    selected = selected.reshape_as(full)
    torch.testing.assert_close(selected[mask], full[mask], rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(selected[~mask]) == 0
    assert encoder.backbone.batch_sizes == ([3, 3] if different_camera_shapes else [6])

    weights = torch.randn_like(full)
    (full[mask] * weights[mask]).sum().backward()
    (selected * weights).sum().backward()
    for key in images:
        assert selected_images[key].grad[mask].abs().sum() > 0
        assert torch.count_nonzero(selected_images[key].grad[~mask]) == 0
        torch.testing.assert_close(selected_images[key].grad, full_images[key].grad,
                                   rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("apply_fusion_ffn", [False, True])
def test_empty_selection_skips_backbone_and_keeps_zero_input_gradients(apply_fusion_ffn):
    encoder, images = make_encoder()
    images = {key: value.requires_grad_() for key, value in images.items()}
    # Nonzero norm bias must not appear at unmeasured positions.
    with torch.no_grad():
        encoder.context_norm.bias.fill_(0.75)
    tokens = encoder.encode_image_context(
        images, input_grad=True, apply_fusion_ffn=apply_fusion_ffn,
        observation_mask=torch.zeros(2, 3, dtype=torch.bool),
    )
    assert tokens.shape == (2, 36, 8)
    assert torch.count_nonzero(tokens) == 0
    assert encoder.backbone.batch_sizes == []
    tokens.sum().backward()
    for value in images.values():
        assert value.grad is not None
        assert torch.count_nonzero(value.grad) == 0


def test_all_selected_observations_preserve_legacy_output():
    encoder, images = make_encoder()
    expected = encoder.encode_image_context(images, apply_fusion_ffn=True)
    actual = encoder.encode_image_context(
        images, apply_fusion_ffn=True, observation_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_compiled_selection_accepts_variable_counts_and_preserves_input_gradients():
    encoder, images = make_encoder()
    encoder.backbone.record_batch_sizes = False
    compiled = torch.compile(encoder.encode_image_context, backend="aot_eager",
                             fullgraph=True, dynamic=True)
    for count in (0, 1, 2, 3, 4):
        full_images = {key: value.clone().requires_grad_() for key, value in images.items()}
        selected_images = {key: value.clone().requires_grad_() for key, value in images.items()}
        indices = torch.arange(count)
        full = encoder.encode_image_context(full_images, input_grad=True, apply_fusion_ffn=True)
        expected = full.reshape(6, -1, 8).index_select(0, indices)
        actual = compiled(selected_images, input_grad=True, apply_fusion_ffn=True,
                          selected_observation_indices=indices).reshape(6, -1, 8)
        torch.testing.assert_close(actual.index_select(0, indices), expected)
        weights = torch.randn_like(actual)
        (expected * weights.index_select(0, indices)).sum().backward()
        (actual * weights).sum().backward()
        for key in images:
            torch.testing.assert_close(selected_images[key].grad, full_images[key].grad,
                                       rtol=2e-5, atol=2e-6)
