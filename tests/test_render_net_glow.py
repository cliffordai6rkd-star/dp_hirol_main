"""Behavioral checks for the restricted Gaussian input-rendering channel."""

import copy

import pytest
import torch

from diffusion_policy.model.vision.render_net import GaussianGlowBackbone, RenderNet


def make_renderer(**kwargs):
    return RenderNet(
        6, base_channels=4, conditioning_dim=8, wrench_history_steps=3,
        embed_dim=16, image_self_attn_layers=1, cross_attn_layers=1, patch_size=8, num_heads=4,
        high_attention_render={"enabled": True}, rendering_mode="gaussian_glow",
        glow={"num_blobs": 3, "sigma_min": 0.04, "sigma_max": 0.2}, **kwargs,
    )


@pytest.mark.parametrize("backbone", ["unet", "transformer"])
@pytest.mark.parametrize("conv_block", ["standard", "bottleneck"])
def test_glow_handles_multiple_camera_sizes_and_only_brightens(backbone, conv_block):
    torch.manual_seed(2)
    model = make_renderer(backbone=backbone, conv_block=conv_block).eval()
    images = {"wrist": torch.rand(1, 2, 3, 17, 25), "side": torch.ones(1, 2, 3, 20, 28)}
    wrench = torch.randn(1, 2, 3, 6)
    with torch.no_grad():
        rendered, masks = model(images, wrench, return_aux=True)
        reversed_images = model(dict(reversed(list(images.items()))), wrench)
    assert isinstance(model.backbone, GaussianGlowBackbone)
    assert len(model.backbone.decoders) == 0
    for key, original in images.items():
        assert rendered[key].shape == original.shape
        assert masks[key].shape == (*original.shape[:2], 1, *original.shape[-2:])
        assert torch.isfinite(rendered[key]).all()
        assert (rendered[key] >= original).all() and (rendered[key] <= 1).all()
        assert (masks[key] >= 0).all() and (masks[key] <= 1).all()
        torch.testing.assert_close(rendered[key], reversed_images[key], atol=0, rtol=0)
    assert (rendered["wrist"] > images["wrist"]).any()
    torch.testing.assert_close(rendered["side"], images["side"], atol=0, rtol=0)


def test_student_region_controls_actual_rendering_and_can_be_bypassed():
    model = make_renderer().eval()
    with torch.no_grad():
        model.mask_head.weight.zero_()
        model.mask_head.bias.fill_(-1000)
        images = {"wrist": torch.full((1, 2, 3, 17, 25), 0.25)}
        wrench = torch.randn(1, 2, 3, 6)
        gated, masks = model(images, wrench, return_aux=True)
        unrestricted = model(images, wrench, use_attention_mask=False)
    assert masks["wrist"].count_nonzero() == 0
    torch.testing.assert_close(gated["wrist"], images["wrist"], atol=0, rtol=0)
    assert (unrestricted["wrist"] > images["wrist"]).any()


def test_glow_loss_reaches_spatial_head_region_and_full_wrench_sequence_on_first_step():
    torch.manual_seed(4)
    model = make_renderer()
    images = {key: torch.rand(1, 2, 3, 17, 25) * 0.5 for key in ("wrist", "side")}
    wrench = torch.randn(1, 2, 3, 6, requires_grad=True)
    rendered = model(images, wrench)
    sum(result.square().mean() for result in rendered.values()).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for parameter in (model.output.weight, model.backbone.blob_queries,
                      model.mask_head.weight, model.wrench_encoder.weight_ih_l0,
                      model.backbone.blocks[0].cross_attention.out_proj.weight):
        assert parameter.grad.abs().sum() > 0
    assert (wrench.grad.abs().sum(dim=(0, 1, 3)) > 0).all()


def test_one_gaussian_is_smooth_localized_and_has_differentiable_geometry():
    backbone = GaussianGlowBackbone(
        4, 8, self_attention=True, patch_size=8, embed_dim=16, depth=1,
        glow={"num_blobs": 1, "sigma_min": 0.04, "sigma_max": 0.20},
    )
    parameters = torch.zeros(1, 1, 9, requires_grad=True)
    with torch.no_grad():
        parameters[..., 2] = -1
    field = backbone.render_blobs(parameters, 65, 65)
    middle = field[0, 0, 32]
    assert (middle[1:33] >= middle[:32]).all()
    assert (middle[33:] <= middle[32:-1]).all()
    assert field[0, 0, 32, 32] > 100 * field[0, 0, 0, 0]
    torch.testing.assert_close(middle, middle.flip(0), atol=1e-7, rtol=1e-6)
    y, x = torch.meshgrid(torch.arange(65), torch.arange(65), indexing="ij")
    weight = x.float() + 0.01 * x * y
    (field * weight).mean().backward()
    assert torch.isfinite(parameters.grad).all()
    assert (parameters.grad[..., :4].abs() > 0).all()
    assert parameters.grad[..., 4].abs().sum() > 0


def test_compiling_glow_backbone_preserves_outputs_gradients_and_checkpoint_keys():
    torch.manual_seed(8)
    eager = make_renderer()
    compiled = copy.deepcopy(eager)
    compiled.compile_backbone(backend="aot_eager", fullgraph=True, dynamic=False)
    images = {"wrist": torch.rand(1, 2, 3, 17, 25)}
    wrench = torch.randn(1, 2, 3, 6)
    expected = eager(images, wrench)["wrist"]
    actual = compiled(images, wrench)["wrist"]
    torch.testing.assert_close(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    assert compiled.state_dict().keys() == eager.state_dict().keys()
    for name, parameter in compiled.named_parameters():
        reference = dict(eager.named_parameters())[name]
        if reference.grad is not None:
            torch.testing.assert_close(parameter.grad, reference.grad, atol=1e-7, rtol=2e-4)


@pytest.mark.parametrize("glow", [
    {"num_blobs": 0}, {"num_blobs": 1.5}, {"num_blobs": True},
    {"sigma_min": 0}, {"sigma_max": 2}, {"sigma_min": 0.3, "sigma_max": 0.2},
    {"max_intensity": float("inf")}, {"initial_intensity": 0}, {"unknown": 1},
])
def test_invalid_glow_config_is_rejected(glow):
    with pytest.raises(ValueError):
        RenderNet(6, rendering_mode="gaussian_glow", glow=glow)


def test_missing_rendering_mode_restores_existing_residual_checkpoint_exactly():
    options = {"base_channels": 4, "conditioning_dim": 8,
               "transformer": {"embed_dim": 16, "depth": 1, "patch_size": 8}}
    old = RenderNet(6, **options).eval()
    restored = RenderNet(6, rendering_mode="residual", **options).eval()
    restored.load_state_dict(old.state_dict(), strict=True)
    assert old.state_dict().keys() == restored.state_dict().keys()
    images = {"wrist": torch.rand(1, 2, 3, 17, 25)}
    wrench = torch.randn(1, 2, 8, 6)
    with torch.no_grad():
        torch.testing.assert_close(old(images, wrench)["wrist"], restored(images, wrench)["wrist"],
                                   atol=0, rtol=0)
