"""Camera fusion exchanges current observations without RGB condition gates."""

import pytest
import torch

from diffusion_policy.model.vision.render_net import RenderNet


@pytest.fixture(autouse=True, scope="module")
def _cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _renderer(**overrides):
    options = dict(
        wrench_dim=7, base_channels=4, conditioning_dim=8, embed_dim=16,
        patch_size=8, num_heads=4, image_self_attn_layers=1,
        cross_attn_layers=1, cross_camera_attn_layers=1, wrench_history_steps=4,
        max_obs_steps=2, image_keys=("side", "wrist"), dropout=0.0,
        use_contact_alpha=False, use_region_gate=False,
        high_attention_render={"enabled": False}, phase_prediction={"enabled": False},
    )
    torch.manual_seed(205)
    model = RenderNet(**{**options, **overrides}).eval()
    # Production starts with an identity render. A nonzero RGB projection
    # tests information flow after that projection has begun learning.
    with torch.no_grad():
        model.output.weight.normal_(std=0.04)
    return model


def _observations():
    generator = torch.Generator().manual_seed(207)
    images = {
        "side": .2 + .6 * torch.rand(2, 2, 3, 17, 25, generator=generator),
        "wrist": .2 + .6 * torch.rand(2, 2, 3, 24, 16, generator=generator),
    }
    wrench = torch.randn(2, 2, 4, 7, generator=generator)
    return images, wrench


@pytest.mark.parametrize("target,other", [("side", "wrist"), ("wrist", "side")])
def test_camera_rgb_depends_on_other_camera_and_wrench(target, other):
    model = _renderer()
    images, wrench = _observations()
    images[other].requires_grad_()
    wrench.requires_grad_()
    output = model(images, wrench)[target]
    assert output.shape == images[target].shape
    changed_images = {**images, other: 1 - images[other]}
    with torch.no_grad():
        changed_camera = model(changed_images, wrench)[target]
        changed_wrench = model(images, wrench.flip(-1) * 3)[target]
    assert (changed_camera - output).abs().max() > 1e-6
    assert (changed_wrench - output).abs().max() > 1e-6
    output.square().mean().backward()
    assert torch.isfinite(images[other].grad).all()
    assert images[other].grad.abs().sum() > 0
    assert torch.isfinite(wrench.grad).all()
    assert wrench.grad.abs().sum() > 0
    for module in (model.cross_camera_attention, model.wrench_encoder):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(gradient is not None and torch.isfinite(gradient).all()
                   for gradient in gradients)
        assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_camera_fusion_does_not_read_future_observations_or_other_batch_items():
    model = _renderer()
    images, wrench = _observations()
    images = {key: value.requires_grad_() for key, value in images.items()}
    wrench.requires_grad_()
    output = model(images, wrench)["side"][0, 0]
    changed_images = {key: value.detach().clone() for key, value in images.items()}
    changed_wrench = wrench.detach().clone()
    for value in (*changed_images.values(), changed_wrench):
        value[:, 1] = 1 - value[:, 1]
        value[1] = value[1] * 2
    changed = model(changed_images, changed_wrench)["side"][0, 0]
    torch.testing.assert_close(changed, output, rtol=0, atol=0)
    output.square().mean().backward()
    for value in (*images.values(), wrench):
        assert value.grad[0, 0].abs().sum() > 0
        assert value.grad[:, 1].count_nonzero() == 0
        assert value.grad[1].count_nonzero() == 0


@pytest.mark.parametrize("stage", ["joint", "rgb"])
def test_complete_renderer_is_trainable_without_condition_parameters(stage):
    model = _renderer()
    model.set_training_stage(stage)
    model.set_stage_mode(True)
    assert model.mask_head is None
    assert model.phase_temporal_encoder is None
    assert model.gate_backbone is None
    assert model.gate_wrench_encoder is None
    assert model.gate_parameters() == []
    assert {id(parameter) for parameter in model.rgb_parameters()} == {
        id(parameter) for parameter in model.parameters()
    }
    assert all(parameter.requires_grad for parameter in model.parameters())
    images, wrench = _observations()
    output = model(images, wrench)
    sum(value.square().mean() for value in output.values()).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_rgb_is_plain_bounded_residual_and_ignores_phase_conditions(monkeypatch):
    model = _renderer()
    images, wrench = _observations()
    raw_outputs = []
    handle = model.output.register_forward_hook(
        lambda module, args, output: raw_outputs.append(output.detach().clone())
    )

    def forbidden(*args, **kwargs):
        pytest.fail("camera fusion must not execute a condition or gate path")

    for method in ("_gate_head_outputs", "_condition_head_outputs", "_mask_gate"):
        monkeypatch.setattr(model, method, forbidden)
    try:
        with torch.no_grad():
            output, masks, phase = model(
                images, wrench, contact_alpha=torch.zeros(2, 2),
                future_contact=torch.zeros(2, 2, 4), return_aux=True, return_phase=True,
            )
        assert masks == {} and phase is None
        for key, raw in zip(model.image_keys, raw_outputs):
            expected = (images[key] + model.residual_scale * raw.reshape_as(images[key]).tanh()).clamp(0, 1)
            torch.testing.assert_close(output[key], expected, rtol=0, atol=0)
    finally:
        handle.remove()


def test_sparse_rendering_and_camera_order_preserve_correspondence():
    model = _renderer()
    images, wrench = _observations()
    selected = torch.tensor([[False, True], [True, False]])
    with torch.no_grad():
        full = model(images, wrench)
        sparse = model(dict(reversed(list(images.items()))), wrench,
                       render_observation_mask=selected)
        empty = model(images, wrench, render_observation_mask=torch.zeros_like(selected))
    for key in images:
        torch.testing.assert_close(sparse[key][selected], full[key][selected], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(sparse[key][~selected], images[key][~selected], rtol=0, atol=0)
        torch.testing.assert_close(empty[key], images[key], rtol=0, atol=0)


def test_default_legacy_path_has_no_cross_camera_parameters_or_dependency():
    options = dict(
        wrench_dim=7, base_channels=4, conditioning_dim=8, embed_dim=16,
        patch_size=8, num_heads=4, image_self_attn_layers=1, cross_attn_layers=1,
        wrench_history_steps=4, max_obs_steps=2,
    )
    torch.manual_seed(209)
    legacy = RenderNet(**options).eval()
    torch.manual_seed(209)
    explicit = RenderNet(**options, cross_camera_attn_layers=0).eval()
    assert not any("cross_camera" in name for name in legacy.state_dict())
    assert legacy.state_dict().keys() == explicit.state_dict().keys()
    for name, tensor in legacy.state_dict().items():
        torch.testing.assert_close(tensor, explicit.state_dict()[name], rtol=0, atol=0)
    with torch.no_grad():
        legacy.output.weight.normal_(std=.04)
        explicit.load_state_dict(legacy.state_dict(), strict=True)
        images, wrench = _observations()
        expected = legacy(images, wrench)
        restored = explicit(images, wrench)
        changed = legacy({**images, "wrist": 1 - images["wrist"]}, wrench)
    for key in images:
        torch.testing.assert_close(expected[key], restored[key], rtol=0, atol=0)
    torch.testing.assert_close(expected["side"], changed["side"], rtol=0, atol=0)


@pytest.mark.parametrize("conditions", [
    {"use_contact_alpha": True}, {"use_region_gate": True},
    {"phase_prediction": {"enabled": True}},
    {"high_attention_render": {"enabled": True}},
])
def test_new_path_rejects_enabled_conditions(conditions):
    with pytest.raises(ValueError, match="cross_camera_attn_layers"):
        _renderer(**conditions)
