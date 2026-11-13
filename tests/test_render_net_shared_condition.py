"""The shared condition path encodes once and decodes selected observations."""

import copy

import pytest
import torch
from torch.nn import functional as F

from diffusion_policy.model.vision.render_net import RenderNet


@pytest.fixture(autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _renderer(*, shared=True):
    model = RenderNet(
        7, base_channels=4, conditioning_dim=8, wrench_history_steps=8,
        embed_dim=16, image_self_attn_layers=1, cross_attn_layers=1,
        decoder_layers=3, patch_size=8, num_heads=4, conv_block="bottleneck",
        shared_condition_backbone=shared, separate_gate_path=not shared,
        phase_prediction={"enabled": True, "hidden_dim": 8,
                          "freeze_wrench_encoder": True},
        high_attention_render={"enabled": True, "initial_strength": .4,
                               "threshold": 0., "detach_for_rgb": False},
        residual_scale=.5,
    )
    model.freeze_wrench_encoder()
    return model


def _observations():
    images = {key: .2 + .6 * torch.rand(2, 2, 3, 16, 24)
              for key in ("side", "wrist")}
    return images, torch.randn(2, 2, 8, 7)


def _condition_loss(masks, prediction):
    phase = torch.tensor([[0, 1], [1, 2]])
    alpha = torch.tensor([[0., .25], [.75, 1.]])
    return (F.cross_entropy(prediction["phase_logits"].flatten(0, 1), phase.flatten())
            + F.mse_loss(prediction["contact_alpha"], alpha)
            + sum((mask - .7).square().mean() for mask in masks.values()))


def _rgb_loss(rendered, images):
    return sum((rendered[key] - (image + .08)).square().mean()
               for key, image in images.items())


def _has_gradient(module):
    return any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in module.parameters())


def test_shared_encoder_runs_once_per_camera_and_warmup_bypasses_decoder(monkeypatch):
    torch.manual_seed(1201)
    model = _renderer()
    images, wrench = _observations()
    calls = []
    encode = model.backbone.encode

    def counted_encode(image, *args):
        calls.append(tuple(image.shape))
        return encode(image, *args)

    monkeypatch.setattr(model.backbone, "encode", counted_encode)
    decoder_calls = []
    hook = model.backbone.decoders[0].register_forward_hook(
        lambda *args: decoder_calls.append(1))
    try:
        assert model.gate_backbone is None
        assert model.gate_wrench_encoder is None
        model.set_training_stage("condition_warmup")
        model.set_stage_mode(training=True)
        rendered, masks, prediction = model(
            images, wrench, condition_only=True, return_aux=True, return_phase=True,
        )
        assert rendered is None
        assert calls == [(4, 3, 16, 24)] * 2
        assert not decoder_calls
        assert masks["side"].shape == (2, 2, 1, 16, 24)
        _condition_loss(masks, prediction).backward()
        assert _has_gradient(model.backbone.stem)
        assert _has_gradient(model.mask_head)
        assert _has_gradient(model.phase_temporal_encoder)
        assert all(not parameter.requires_grad for parameter in model.backbone.decoders.parameters())
        assert all(parameter.grad is None for parameter in model.wrench_encoder.parameters())

        calls.clear()
        model.set_training_stage("joint")
        model.set_stage_mode(training=True)
        model(images, wrench, return_aux=True, return_phase=True)
        assert calls == [(4, 3, 16, 24)] * 2
        assert len(decoder_calls) == 2
    finally:
        hook.remove()


def test_joint_loss_reaches_encoder_condition_and_rgb_after_zero_output_update():
    torch.manual_seed(1202)
    model = _renderer()
    images, wrench = _observations()
    model.set_training_stage("joint")
    model.set_stage_mode(training=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        rendered, masks, prediction = model(images, wrench, return_aux=True, return_phase=True)
        (_condition_loss(masks, prediction) + _rgb_loss(rendered, images)).backward()
        assert _has_gradient(model.backbone.stem)
        assert _has_gradient(model.mask_head)
        assert _has_gradient(model.phase_temporal_encoder)
        assert _has_gradient(model.output)
        if update:
            assert _has_gradient(model.backbone.decoders)
        optimizer.step()


def test_render_selection_keeps_all_condition_predictions_and_free_images(monkeypatch):
    torch.manual_seed(1203)
    model = _renderer().train()
    images, wrench = _observations()
    with torch.no_grad():
        model.output.bias.fill_(.4)
    full, full_masks, full_prediction = model(images, wrench, return_aux=True, return_phase=True)
    decoded_counts = []
    decode = model.backbone.decode_features

    def counted_decode(features, *args, **kwargs):
        decoded_counts.append(features.shape[0])
        return decode(features, *args, **kwargs)

    monkeypatch.setattr(model.backbone, "decode_features", counted_decode)
    selected = torch.tensor([[True, False], [False, True]])
    rendered, masks, prediction = model(
        images, wrench, return_aux=True, return_phase=True,
        render_observation_mask=selected,
    )
    assert decoded_counts == [2, 2]
    for key in images:
        torch.testing.assert_close(rendered[key][~selected], images[key][~selected], rtol=0, atol=0)
        torch.testing.assert_close(rendered[key][selected], full[key][selected])
        assert not torch.equal(rendered[key][selected], images[key][selected])
        torch.testing.assert_close(masks[key], full_masks[key], rtol=0, atol=0)
    for key in prediction:
        torch.testing.assert_close(prediction[key], full_prediction[key], rtol=0, atol=0)

    decoded_counts.clear()
    free, free_masks, free_prediction = model(
        images, wrench, return_aux=True, return_phase=True,
        render_observation_mask=torch.zeros_like(selected),
    )
    assert not decoded_counts
    for key in images:
        torch.testing.assert_close(free[key], images[key], rtol=0, atol=0)
        torch.testing.assert_close(free_masks[key], full_masks[key], rtol=0, atol=0)
    for key in prediction:
        torch.testing.assert_close(free_prediction[key], full_prediction[key], rtol=0, atol=0)


def test_compiled_shared_encode_and_decode_match_eager_outputs_and_gradients(monkeypatch):
    torch.manual_seed(1204)
    eager = _renderer().train()
    # A nonzero RGB output exposes gradients through the full decoder.
    with torch.no_grad():
        eager.output.weight.normal_(std=.03)
    compiled = copy.deepcopy(eager)
    compiled.compile_backbone(backend="eager")
    calls = {"encode": 0, "decode": 0}
    for key, attribute in (("encode", "_compiled_shared_encode"),
                           ("decode", "_compiled_shared_decode")):
        implementation = getattr(compiled, attribute)

        def counted(*args, _key=key, _implementation=implementation, **kwargs):
            calls[_key] += 1
            return _implementation(*args, **kwargs)

        monkeypatch.setattr(compiled, attribute, counted)
    images, wrench = _observations()
    selected = torch.tensor([[True, False], [False, True]])
    outputs = []
    for model in (eager, compiled):
        rendered, masks, prediction = model(
            images, wrench, return_aux=True, return_phase=True,
            render_observation_mask=selected,
        )
        (_rgb_loss(rendered, images) + _condition_loss(masks, prediction)).backward()
        outputs.append((rendered, masks, prediction))
        assert _has_gradient(model.backbone.decoders)
        assert _has_gradient(model.backbone.stem)
    assert calls == {"encode": 2, "decode": 2}
    for expected, actual in zip(outputs[0], outputs[1]):
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
    for (name, expected), (actual_name, actual) in zip(eager.named_parameters(),
                                                    compiled.named_parameters()):
        assert name == actual_name
        if expected.grad is None:
            assert actual.grad is None, name
        else:
            torch.testing.assert_close(actual.grad, expected.grad, msg=lambda message: name + ": " + message)


def test_separate_gate_path_uses_compiled_backbone_and_matches_eager(monkeypatch):
    torch.manual_seed(1205)
    eager = _renderer(shared=False).train()
    with torch.no_grad():
        eager.output.weight.normal_(std=.03)
    compiled = copy.deepcopy(eager)
    compiled.compile_backbone(backend="eager")
    implementation = compiled._compiled_backbone_forward
    calls = []

    def counted(*args, **kwargs):
        calls.append(tuple(args[0].shape))
        return implementation(*args, **kwargs)

    monkeypatch.setattr(compiled, "_compiled_backbone_forward", counted)
    images, wrench = _observations()
    outputs = []
    for model in (eager, compiled):
        rendered, masks, prediction = model(images, wrench, return_aux=True, return_phase=True)
        (_rgb_loss(rendered, images) + _condition_loss(masks, prediction)).backward()
        outputs.append((rendered, masks, prediction))
        assert _has_gradient(model.backbone.decoders)
        assert _has_gradient(model.backbone.stem)
    assert len(calls) == len(images)
    assert calls == [(4, 3, 16, 24)] * len(images)
    for expected, actual in zip(outputs[0], outputs[1]):
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
    for (name, expected), (actual_name, actual) in zip(eager.named_parameters(),
                                                    compiled.named_parameters()):
        assert name == actual_name
        if expected.grad is None:
            assert actual.grad is None, name
        else:
            torch.testing.assert_close(actual.grad, expected.grad, msg=lambda message: name + ": " + message)
