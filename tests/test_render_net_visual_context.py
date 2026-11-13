"""RGB-only distillation preserves the visual teacher's action-context layout."""

import pytest
import torch
from torch import nn

from test_coarse_residual_fusion import _encoder, _observations


@pytest.fixture(autouse=True, scope="module")
def _cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _visual_encoder(*, upsample, **kwargs):
    return _encoder(
        fusion_output_mode="visual", fusion_upsample=upsample,
        fusion_highres_skip=upsample, **kwargs,
    ).eval().requires_grad_(False)


@pytest.mark.parametrize("upsample,token_count", [(True, 784), (False, 196)])
def test_student_context_matches_real_board_layout(upsample, token_count):
    encoder = _visual_encoder(
        upsample=upsample, n_emb=256, patch_size=16,
        image_shapes={"side": (3, 224, 224), "wrist": (3, 224, 224)},
    )
    images, _ = _observations(encoder, batch_size=1)
    tokens = encoder.encode_image_context(images)
    assert tokens.shape == (1, token_count, 256)
    assert token_count == 2 * encoder.context_tokens_per_observation


@pytest.mark.parametrize("upsample", [True, False])
def test_student_matches_teacher_without_fusion_updates_and_never_reads_torque(upsample):
    # Different rectangular grids and reversed input insertion order expose
    # accidental camera/time flattening before spatial pooling.
    encoder = _visual_encoder(
        upsample=upsample,
        image_shapes={"wrist": (3, 48, 32), "side": (3, 32, 64)},
    )
    images, torque = _observations(encoder)
    # With no joint-attention update, the actual visual teacher should reduce
    # exactly to the student's frozen image-only path at either resolution.
    encoder.pre_cross_self_attention = nn.Identity()
    with torch.no_grad():
        encoder.context_norm.weight.copy_(torch.linspace(0.7, 1.3, encoder.n_emb))
        encoder.context_norm.bias.copy_(torch.linspace(-0.2, 0.3, encoder.n_emb))
        expected = encoder(images=images, wrench_history=torque)

    def forbid_torque_or_fusion(*args):
        pytest.fail("student image encoding must not read torque or rerun fusion")

    handles = [module.register_forward_pre_hook(forbid_torque_or_fusion) for module in (
        encoder.wrench_projection, encoder.force_encoder,
        encoder.pre_cross_self_attention,
    )]
    try:
        actual = encoder.encode_image_context(images)
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("upsample,pixel_block", [(True, 8), (False, 16)])
def test_frozen_image_path_backpropagates_into_matching_camera_time_and_patch(upsample, pixel_block):
    encoder = _visual_encoder(upsample=upsample)
    images, _ = _observations(encoder)
    images = {key: value.requires_grad_() for key, value in images.items()}
    tokens = encoder.encode_image_context(images, input_grad=True)
    per_observation = tokens.reshape(2, 2, -1, encoder.n_emb)
    weight = torch.randn(encoder.n_emb, generator=torch.Generator().manual_seed(83))
    # The test backbone is patch-local: first side token at sample 1/time 1
    # must backpropagate only into that camera/time's matching image block.
    (per_observation[1, 1, 0] * weight).sum().backward()
    allowed = torch.zeros_like(images["side"], dtype=torch.bool)
    allowed[1, 1, :, :pixel_block, :pixel_block] = True
    for key, value in images.items():
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        if key == "side":
            assert value.grad[allowed].abs().sum() > 0
            assert torch.count_nonzero(value.grad[~allowed]) == 0
        else:
            assert torch.count_nonzero(value.grad) == 0
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in encoder.parameters())


@pytest.mark.parametrize("empty", [False, True])
def test_coarse_student_selection_keeps_layout_and_zeroes_unselected_observations(empty):
    encoder = _visual_encoder(upsample=False)
    images, _ = _observations(encoder)
    images = {key: value.requires_grad_() for key, value in images.items()}
    with torch.no_grad():
        encoder.context_norm.bias.fill_(0.7)
        expected = encoder.encode_image_context(images).reshape(2, 2, -1, encoder.n_emb)
    mask = torch.zeros(2, 2, dtype=torch.bool)
    if not empty:
        mask[0, 1] = True
        mask[1, 0] = True
    actual = encoder.encode_image_context(
        images, input_grad=True, observation_mask=mask,
    ).reshape_as(expected)
    torch.testing.assert_close(actual[mask], expected[mask], rtol=0, atol=0)
    assert torch.count_nonzero(actual[~mask]) == 0
    (actual * torch.randn_like(actual)).sum().backward()
    for image in images.values():
        assert image.grad is not None
        assert torch.count_nonzero(image.grad[~mask]) == 0
        if not empty:
            assert image.grad[mask].abs().sum() > 0


def test_all_output_teachers_retain_legacy_dense_image_encoding():
    dense = _encoder(fusion_output_mode="all", fusion_highres_skip=False)
    coarse = _encoder(fusion_output_mode="all", fusion_highres_skip=False,
                      fusion_upsample=False)
    coarse.load_state_dict(dense.state_dict(), strict=True)
    images, _ = _observations(dense)
    with torch.no_grad():
        expected = dense.encode_image_context(images)
        actual = coarse.encode_image_context(images)
    assert actual.shape == (2, 2 * sum(coarse.patch_counts.values()), coarse.n_emb)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("upsample", [True, False])
def test_visual_student_does_not_replay_legacy_cross_attention_ffn(upsample):
    encoder = _visual_encoder(upsample=upsample)
    images, _ = _observations(encoder)
    with pytest.raises(ValueError, match="Fusion FFN image supervision"):
        encoder.encode_image_context(images, apply_fusion_ffn=True)
