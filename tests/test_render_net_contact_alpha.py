"""A shared contact alpha continuously scales the student's RGB residual."""

import pytest
import torch

from diffusion_policy.model.vision.render_net import RenderNet
from diffusion_policy.model.vision.wrench_preprocessing import compute_contact_alpha


@pytest.mark.parametrize("mode", ["residual", "gaussian_glow"])
@pytest.mark.parametrize("use_mask", [False, True])
def test_contact_count_scales_actual_rgb_change_and_preserves_spatial_field(mode, use_mask):
    torch.manual_seed(19)
    renderer = RenderNet(
        6, base_channels=4, conditioning_dim=8, wrench_history_steps=8,
        embed_dim=16, image_self_attn_layers=1, cross_attn_layers=1,
        decoder_layers=3, patch_size=8, num_heads=4, residual_scale=.1,
        rendering_mode=mode, high_attention_render={"enabled": use_mask, "threshold": 0.},
        glow={"num_blobs": 2},
    ).eval()
    with torch.no_grad():
        renderer.output.bias.fill_(.4)
    images = {key: torch.full((1, 2, 3, 16, 24), .4) for key in ("side", "wrist")}
    wrench = torch.randn(1, 2, 8, 6)
    with torch.no_grad():
        fully_open, full_masks = renderer(images, wrench, return_aux=True)
        for count in (0, 1, 2, 4, 8):
            mask = torch.zeros(1, 2, 8, dtype=torch.bool)
            mask[..., :count] = True
            alpha = compute_contact_alpha(mask)[..., None, None, None]
            rendered, fields = renderer(images, wrench, future_contact=mask, return_aux=True)
            for camera in images:
                expected = images[camera] + alpha * (fully_open[camera] - images[camera])
                torch.testing.assert_close(rendered[camera], expected, atol=1e-7, rtol=1e-6)
                if count <= 1:
                    torch.testing.assert_close(rendered[camera], images[camera], rtol=0, atol=0)
                else:
                    assert (rendered[camera] - images[camera]).abs().max() > 0
                if use_mask:
                    torch.testing.assert_close(fields[camera], full_masks[camera], rtol=0, atol=0)


def test_dataloader_alpha_is_used_directly_and_closed_images_have_no_wrench_gradient():
    torch.manual_seed(20)
    renderer = RenderNet(
        6, base_channels=4, conditioning_dim=8, wrench_history_steps=8,
        embed_dim=16, image_self_attn_layers=1, cross_attn_layers=1,
        decoder_layers=3, patch_size=8, num_heads=4, residual_scale=.1,
        high_attention_render={"enabled": False},
    ).eval()
    with torch.no_grad():
        renderer.output.weight.normal_(std=.1)
    images = {"side": torch.full((1, 2, 3, 16, 24), .5)}
    wrench = torch.randn(1, 2, 8, 6, requires_grad=True)
    alpha = torch.tensor([[0., .5]])
    rendered = renderer(images, wrench, contact_alpha=alpha,
                        future_contact=torch.ones(1, 2, 8, dtype=torch.bool))["side"]
    torch.testing.assert_close(rendered[:, 0], images["side"][:, 0], rtol=0, atol=0)
    rendered.sum().backward()
    assert wrench.grad[:, 0].count_nonzero() == 0
    assert wrench.grad[:, 1].abs().sum() > 0
