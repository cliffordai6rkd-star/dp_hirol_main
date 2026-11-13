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


def _unconstrained_renderer(shared):
    renderer = RenderNet(
        6, base_channels=4, conditioning_dim=8, wrench_history_steps=8,
        embed_dim=16, image_self_attn_layers=1, cross_attn_layers=1,
        decoder_layers=3, patch_size=8, num_heads=4, residual_scale=.1,
        shared_condition_backbone=shared, separate_gate_path=not shared,
        condition_backbone=None if shared else {
            "base_channels": 2, "embed_dim": 8, "image_self_attn_layers": 0,
            "cross_attn_layers": 1, "decoder_layers": 0,
        },
        high_attention_render={"enabled": True, "threshold": 0.},
        phase_prediction={"enabled": True, "hidden_dim": 8},
        use_contact_alpha=False, use_region_gate=False,
    ).eval()
    with torch.no_grad():
        renderer.output.weight.normal_(std=.025)
        renderer.output.bias.fill_(.2)
    images = {"side": torch.full((1, 2, 3, 16, 24), .4)}
    wrench = torch.randn(1, 2, 8, 6)
    return renderer, images, wrench


@pytest.mark.parametrize("shared", [False, True])
def test_unconstrained_rgb_ignores_external_alpha_and_contact_labels(shared):
    torch.manual_seed(21)
    renderer, images, wrench = _unconstrained_renderer(shared)
    with torch.no_grad():
        reference = renderer(images, wrench)["side"]
        assert not torch.equal(reference, images["side"])
        for value in (0., .3, 1.):
            actual = renderer(
                images, wrench, contact_alpha=torch.full((1, 2), value),
                future_contact=torch.full((1, 2, 8), value > 0, dtype=torch.bool),
            )["side"]
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        # Re-enabling the default alpha behavior closes exactly the same RGB
        # model, demonstrating the ablation rather than an unused input path.
        renderer.use_contact_alpha = True
        closed = renderer(images, wrench, contact_alpha=torch.zeros(1, 2))["side"]
        torch.testing.assert_close(closed, images["side"], rtol=0, atol=0)


@pytest.mark.parametrize("shared", [False, True])
def test_unconstrained_rgb_ignores_predicted_strength_but_keeps_rgb_gradients(shared):
    torch.manual_seed(22)
    renderer, images, wrench = _unconstrained_renderer(shared)
    snapshots = []
    with torch.no_grad():
        for bias in (-12., 12.):
            renderer.mask_head.weight[[0, 4]] = 0
            renderer.mask_head.bias[[0, 4]] = bias
            output, masks, prediction = renderer(
                images, wrench, return_aux=True, return_phase=True,
            )
            snapshots.append((output["side"].clone(), masks["side"].clone(),
                              prediction["contact_alpha"].clone()))
    torch.testing.assert_close(snapshots[0][0], snapshots[1][0], rtol=0, atol=0)
    assert snapshots[0][1].max() < .001 and snapshots[1][1].min() > .999
    assert snapshots[0][2].max() < .001 and snapshots[1][2].min() > .999

    renderer.freeze_wrench_encoder()
    renderer.set_training_stage("rgb")
    renderer.set_stage_mode(True)
    output = renderer(images, wrench, contact_alpha=torch.zeros(1, 2))["side"]
    (output - images["side"] - .05).square().mean().backward()
    assert renderer.output.weight.grad is not None
    assert renderer.output.weight.grad.abs().sum() > 0
    if not shared:
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in renderer.backbone.stem.parameters())
    assert all(p.grad is None for p in renderer.gate_parameters())


@pytest.mark.parametrize("shared", [False, True])
def test_region_gate_ablation_preserves_alpha_scaling_and_condition_predictions(shared):
    torch.manual_seed(23)
    renderer, images, wrench = _unconstrained_renderer(shared)
    renderer.use_contact_alpha = True
    with torch.no_grad():
        fully_open, masks, predictions = renderer(
            images, wrench, contact_alpha=torch.ones(1, 2),
            return_aux=True, return_phase=True,
        )
        for alpha in (0., .3, 1.):
            actual = renderer(images, wrench, contact_alpha=torch.full((1, 2), alpha))
            expected = images["side"] + alpha * (fully_open["side"] - images["side"])
            torch.testing.assert_close(actual["side"], expected, rtol=1e-6, atol=1e-7)
        # Region gating is downstream of the condition predictions. Removing
        # it must not alter either alpha/phase or the diagnostic spatial map.
        renderer.use_region_gate = True
        _, gated_masks, gated_predictions = renderer(
            images, wrench, return_aux=True, return_phase=True,
        )
        torch.testing.assert_close(gated_masks["side"], masks["side"], rtol=0, atol=0)
        for key in predictions:
            torch.testing.assert_close(gated_predictions[key], predictions[key], rtol=0, atol=0)
