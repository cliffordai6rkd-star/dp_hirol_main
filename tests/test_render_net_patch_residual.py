"""Pixel layout, zero-strength behavior and gradients for patch RGB rendering."""

import copy

import pytest
import torch

from diffusion_policy.model.vision.render_net import PatchResidualBackbone, RenderNet


def make_patch_renderer(**kwargs):
    return RenderNet(
        6, base_channels=4, conditioning_dim=8, wrench_history_steps=3,
        backbone="transformer", rendering_mode="patch_residual", residual_scale=0.1,
        embed_dim=16, image_self_attn_layers=1, cross_attn_layers=1, patch_size=8, num_heads=4,
        high_attention_render={"enabled": True, "source": "context",
                               "selection": "soft", "activation": "relu"},
        **kwargs,
    )


def test_zero_strength_blocks_learned_rgb_and_refinement_bias_exactly():
    """Identity must result from the field, even after the RGB head has learned."""
    torch.manual_seed(15)
    renderer = make_patch_renderer().eval()
    with torch.no_grad():
        renderer.output.weight.normal_(std=0.04)
        renderer.output.bias.fill_(0.3)
        renderer.backbone.refine.bias.copy_(torch.tensor([0.2, -0.15, 0.1]))
        renderer.mask_head.weight.zero_()
        renderer.mask_head.bias.fill_(-0.4)
    images = {"side": torch.rand(2, 2, 3, 17, 25) * 0.6 + 0.2,
              "wrist": torch.rand(2, 2, 3, 20, 28) * 0.6 + 0.2}
    wrench = torch.randn(2, 2, 3, 6)
    with torch.no_grad():
        gated, fields = renderer(images, wrench, return_aux=True)
        unrestricted = renderer(images, wrench, use_attention_mask=False)
    for camera, original in images.items():
        assert fields[camera].shape == (*original.shape[:2], 1, *original.shape[-2:])
        assert fields[camera].count_nonzero() == 0
        torch.testing.assert_close(gated[camera], original, rtol=0, atol=0)
        difference = (unrestricted[camera] - original).abs()
        assert difference.max() > 0.01
        assert difference.max() <= renderer.residual_scale + 1e-7


def test_patch_rgb_layout_preserves_channel_pixel_and_patch_order_when_cropped(monkeypatch):
    """Distinct patch/pixel values expose swaps that ordinary shape tests miss."""
    patch, grid_h, grid_w, batch = 4, 3, 4, 2
    height, width = 9, 13
    backbone = PatchResidualBackbone(
        4, 8, self_attention=True, patch_size=patch, embed_dim=8, depth=1,
        num_heads=2, residual_refine=False,
    )
    features = torch.zeros(batch, 8, grid_h, grid_w)
    patch_ids = torch.arange(batch * grid_h * grid_w).reshape(batch, grid_h, grid_w)
    features[:, 0] = patch_ids
    with torch.no_grad():
        backbone.output.weight.zero_()
        backbone.output.weight[:, 0] = 1000
        for channel in range(3):
            for row in range(patch):
                for column in range(patch):
                    index = channel * patch * patch + row * patch + column
                    backbone.output.bias[index] = channel * 100 + row * 10 + column
    monkeypatch.setattr(backbone, "encode", lambda *args: (features, [], height, width))
    actual, returned_features = backbone(
        torch.zeros(batch, 3, height, width), torch.zeros(batch, 3, 8), torch.zeros(1, 1, 8),
    )
    expected = torch.empty(batch, 3, height, width)
    for row in range(height):
        for column in range(width):
            for channel in range(3):
                expected[:, channel, row, column] = (
                    patch_ids[:, row // patch, column // patch] * 1000
                    + channel * 100 + (row % patch) * 10 + column % patch
                )
    assert actual.shape == (batch, 3, height, width)
    assert returned_features is features
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("refine", [False, True])
def test_multiple_cameras_keep_shapes_and_camera_identity_independent_of_mapping_order(refine):
    torch.manual_seed(16)
    renderer = make_patch_renderer(patch_residual_refine=refine).eval()
    with torch.no_grad():
        renderer.output.weight.normal_(std=0.03)
        renderer.output.bias.copy_(torch.linspace(-0.3, 0.3, renderer.output.out_features))
        renderer.mask_head.weight.zero_()
        renderer.mask_head.bias.fill_(0.5)
    images = {"wrist": torch.full((1, 2, 3, 17, 25), 0.5),
              "side": torch.full((1, 2, 3, 20, 28), 0.5)}
    wrench = torch.randn(1, 2, 3, 6)
    with torch.no_grad():
        rendered, fields = renderer(images, wrench, return_aux=True)
        reverse = renderer(dict(reversed(list(images.items()))), wrench)
    for camera, original in images.items():
        assert rendered[camera].shape == original.shape
        torch.testing.assert_close(rendered[camera], reverse[camera], rtol=0, atol=0)
        torch.testing.assert_close(fields[camera], torch.full_like(fields[camera], 0.5))
        delta = rendered[camera] - original
        assert delta.min() < 0 and delta.max() > 0
        assert delta.abs().max() <= 0.5 * renderer.residual_scale + 1e-7


def test_patch_renderer_compile_preserves_gradients_and_portable_state():
    torch.manual_seed(19)
    eager = make_patch_renderer().train()
    with torch.no_grad():
        eager.output.weight.normal_(std=0.03)
        eager.mask_head.weight.zero_()
        eager.mask_head.bias.fill_(0.4)
    compiled = copy.deepcopy(eager)
    compiled.compile_backbone(backend="aot_eager", fullgraph=True, dynamic=False)
    images = {"wrist": torch.rand(1, 2, 3, 17, 25) * 0.6 + 0.2}
    wrench = torch.randn(1, 2, 3, 6)
    expected = eager(images, wrench)["wrist"]
    actual = compiled(images, wrench)["wrist"]
    torch.testing.assert_close(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    assert compiled.state_dict().keys() == eager.state_dict().keys()
    for name, parameter in compiled.named_parameters():
        reference = dict(eager.named_parameters())[name]
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, reference.grad, atol=1e-7, rtol=2e-4)
    assert compiled.wrench_encoder.weight_ih_l0.grad.abs().sum() > 0

    def forbid_compilation(*args):
        raise AssertionError("Evaluation must use the original backbone")

    restored = make_patch_renderer().eval()
    restored.load_state_dict(compiled.state_dict(), strict=True)
    compiled._compiled_backbone_forward = forbid_compilation
    compiled.eval()
    with torch.no_grad():
        torch.testing.assert_close(compiled(images, wrench)["wrist"],
                                   restored(images, wrench)["wrist"], rtol=0, atol=0)


def test_patch_renderer_two_steps_reach_lstm_and_attention_through_frozen_dit():
    from test_render_net_denoiser_distillation import make_workspace
    from test_render_net_distillation import make_attention_teacher

    workspace, _ = make_workspace()
    teacher, obs, _ = make_attention_teacher()
    teacher.normalizer.fit({"action": torch.randn(32, 7)})
    workspace.teacher_policy = teacher.eval().requires_grad_(False)
    workspace.model = RenderNet(
        6, base_channels=4, conditioning_dim=8, wrench_history_steps=7,
        rendering_mode="patch_residual", backbone="transformer", residual_scale=0.1,
        patch_size=8, embed_dim=16, image_self_attn_layers=1, cross_attn_layers=1, num_heads=4,
        high_attention_render={"enabled": True, "selection": "soft", "activation": "relu"},
    )
    with torch.no_grad():
        workspace.model.mask_head.weight.zero_()
        workspace.model.mask_head.bias.fill_(0.4)
    teacher_before = {key: value.clone() for key, value in teacher.state_dict().items()}
    batch = {"obs": obs, "action": torch.randn(2, 4, 7)}
    optimizer = torch.optim.AdamW(workspace.model.parameters(), lr=1e-3)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = workspace._batch_loss(batch)
        assert torch.isfinite(loss)
        loss.backward()
        renderer = workspace.model
        assert renderer.output.weight.grad.abs().sum() > 0
        if step:
            parameters = [renderer.wrench_encoder.weight_ih_l0,
                          renderer.backbone.blocks[0].self_attention.out_proj.weight,
                          renderer.backbone.blocks[0].cross_attention.out_proj.weight,
                          renderer.mask_head.weight, renderer.camera_embedding.weight,
                          renderer.backbone.refine.weight]
            for parameter in parameters:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                assert parameter.grad.abs().sum() > 0
        assert all(not parameter.requires_grad and parameter.grad is None
                   for parameter in teacher.parameters())
        optimizer.step()
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[key], rtol=0, atol=0)


@pytest.mark.parametrize("autocast", [False, True])
def test_sigmoid_region_recovers_from_negative_logits_with_positive_supervision(autocast):
    """A missed positive region must still be able to turn rendering back on."""
    torch.manual_seed(25)
    renderer = RenderNet(
        6, base_channels=4, conditioning_dim=8, wrench_history_steps=3,
        rendering_mode="patch_residual",
        embed_dim=16, image_self_attn_layers=1, cross_attn_layers=1, patch_size=8, num_heads=4,
        high_attention_render={"enabled": True, "selection": "soft", "activation": "sigmoid",
                               "threshold": .1},
    ).requires_grad_(False)
    with torch.no_grad():
        renderer.mask_head.bias.fill_(-4.)
        renderer.output.bias.fill_(.3)
    renderer.mask_head.bias.requires_grad_(True)
    optimizer = torch.optim.AdamW([renderer.mask_head.bias], lr=0.5, weight_decay=0.)
    images = {"wrist": torch.full((1, 2, 3, 17, 25), .5)}
    wrench = torch.randn(1, 2, 3, 6)
    strengths, losses, rendered_deltas = [], [], []
    # oneDNN BF16 backward is unavailable on some CPU instruction sets.
    with torch.backends.mkldnn.flags(enabled=False):
        for _ in range(6):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                rendered, masks = renderer(images, wrench, return_aux=True)
                field = masks["wrist"]
                loss = (field - .7).square().mean()
            assert field.dtype == torch.float32
            strengths.append(float(field.detach().mean()))
            losses.append(float(loss.detach()))
            rendered_deltas.append(float((rendered["wrist"] - images["wrist"]).abs().max()))
            loss.backward()
            assert torch.isfinite(renderer.mask_head.bias.grad).all()
            assert renderer.mask_head.bias.grad.item() < 0
            optimizer.step()
    assert strengths[-1] > 5 * strengths[0]
    assert losses[-1] < losses[0]
    assert rendered_deltas[0] == 0
    assert rendered_deltas[-1] > 0


@pytest.mark.parametrize("initial_strength, threshold", [(None, 0.), (.2, .1)])
def test_sigmoid_region_and_zero_rgb_head_bootstrap_frozen_denoiser_gradients(initial_strength, threshold):
    """The weak positive gate must train both heads before the shared backbone."""
    from test_render_net_denoiser_distillation import make_workspace
    from test_render_net_distillation import make_attention_teacher

    workspace, _ = make_workspace()
    teacher, obs, _ = make_attention_teacher()
    teacher.normalizer.fit({"action": torch.randn(32, 7)})
    workspace.teacher_policy = teacher.eval().requires_grad_(False)
    region_config = {"enabled": True, "source": "context", "selection": "soft",
                     "activation": "sigmoid", "strength_scale": 5., "threshold": threshold}
    if initial_strength is not None:
        region_config["initial_strength"] = initial_strength
    workspace.model = RenderNet(
        6, base_channels=4, conditioning_dim=8, wrench_history_steps=7,
        rendering_mode="patch_residual", residual_scale=.1,
        patch_size=8, embed_dim=16, image_self_attn_layers=1, cross_attn_layers=1, num_heads=4,
        high_attention_render=region_config,
    )
    workspace.attention_config = region_config
    workspace.attention_enabled = True
    workspace.region_source = "context"
    with torch.no_grad():
        teacher_tokens = teacher._encode_context(obs, apply_curriculum_mask=False)
        images, wrench, _, _ = teacher.prepare_context_inputs(obs)
        rendered, fields = workspace.model(images, wrench, return_aux=True)
        for key, image in images.items():
            torch.testing.assert_close(rendered[key], image, rtol=0, atol=0)
            torch.testing.assert_close(fields[key], torch.full_like(fields[key], initial_strength or .1))
    teacher_before = {key: value.clone() for key, value in teacher.state_dict().items()}
    batch = {"obs": obs, "action": torch.randn(2, 4, 7), "teacher_targets": {
        "tokens": teacher_tokens,
        "patch_attention": {key: torch.full((2, 2, patches), 5.)
                            for key, patches in teacher.obs_encoder.patch_counts.items()},
    }}
    optimizer = torch.optim.AdamW(workspace.model.parameters(), lr=1e-3)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = workspace._batch_loss(batch)
        assert torch.isfinite(loss)
        loss.backward()
        renderer = workspace.model
        assert renderer.output.weight.grad.abs().sum() > 0
        assert renderer.mask_head.weight.grad.abs().sum() > 0
        if step:
            for parameter in (renderer.wrench_encoder.weight_ih_l0,
                              renderer.backbone.blocks[0].self_attention.out_proj.weight,
                              renderer.backbone.blocks[0].cross_attention.out_proj.weight,
                              renderer.camera_embedding.weight):
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                assert parameter.grad.abs().sum() > 0
        assert all(not parameter.requires_grad and parameter.grad is None
                   for parameter in teacher.parameters())
        optimizer.step()
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[key], rtol=0, atol=0)


@pytest.mark.parametrize("strength", [.25, .5, .75])
def test_strength_threshold_is_strict_and_preserves_open_intensity(strength):
    """An equal/below-threshold field must leave RGB exactly unchanged."""
    renderer = RenderNet(
        6, base_channels=4, conditioning_dim=8, wrench_history_steps=3,
        rendering_mode="patch_residual", residual_scale=.1,
        embed_dim=16, image_self_attn_layers=1, cross_attn_layers=1, patch_size=8, num_heads=4,
        high_attention_render={"enabled": True, "selection": "soft", "activation": "sigmoid",
                               "initial_strength": strength, "threshold": .5},
    )
    with torch.no_grad():
        renderer.output.bias.fill_(.3)
        renderer.backbone.refine.bias.fill_(.2)
    images = {"wrist": torch.full((1, 2, 3, 17, 25), .5)}
    wrench = torch.randn(1, 2, 3, 6)
    rendered, masks = renderer(images, wrench, return_aux=True)
    torch.testing.assert_close(masks["wrist"], torch.full_like(masks["wrist"], strength))
    if strength <= .5:
        torch.testing.assert_close(rendered["wrist"], images["wrist"], rtol=0, atol=0)
    else:
        expected_delta = strength * .1 * torch.tensor(.5).tanh()
        torch.testing.assert_close(rendered["wrist"], images["wrist"] + expected_delta)
    rendered["wrist"].sum().backward()
    if strength <= .5:
        assert renderer.output.bias.grad.count_nonzero() == 0
        assert renderer.mask_head.bias.grad.count_nonzero() == 0
    else:
        assert renderer.output.bias.grad.abs().sum() > 0
        assert renderer.mask_head.bias.grad.abs().sum() > 0


@pytest.mark.parametrize("training", [False, True])
def test_strength_threshold_runs_after_upsampling_without_edge_leakage(monkeypatch, training):
    """Thresholding patch scores before bilinear resize would leak at boundaries."""
    renderer = RenderNet(
        6, base_channels=4, conditioning_dim=8, wrench_history_steps=3,
        rendering_mode="patch_residual", residual_scale=.1,
        embed_dim=16, image_self_attn_layers=1, cross_attn_layers=1, patch_size=8, num_heads=4,
        high_attention_render={"enabled": True, "selection": "soft", "activation": "sigmoid",
                               "threshold": .1},
    ).train(training)
    with torch.no_grad():
        renderer.output.bias.fill_(.3)
        renderer.backbone.refine.bias.fill_(.2)

    def mask_logits(features):
        width = features.shape[-1]
        values = torch.linspace(.02, .18, width, device=features.device)
        return torch.logit(values).reshape(1, 1, 1, width).expand(
            features.shape[0], 1, features.shape[-2], width)

    monkeypatch.setattr(renderer.mask_head, "forward", mask_logits)
    images = {"wrist": torch.full((1, 2, 3, 17, 25), .5)}
    wrench = torch.randn(1, 2, 3, 6)
    rendered, masks = renderer(images, wrench, return_aux=True)
    mask = masks["wrist"].expand_as(images["wrist"])
    closed, opened = mask <= .1, mask > .1
    assert closed.any() and opened.any()
    torch.testing.assert_close(rendered["wrist"][closed], images["wrist"][closed], rtol=0, atol=0)
    expected = images["wrist"] + mask * .1 * torch.tensor(.5).tanh()
    torch.testing.assert_close(rendered["wrist"][opened], expected[opened])


@pytest.mark.parametrize("threshold", [-.1, 1.1, float("nan")])
def test_invalid_strength_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        RenderNet(6, high_attention_render={"enabled": True, "threshold": threshold})
