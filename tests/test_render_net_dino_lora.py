"""Exercise the LoRA DINO renderer with real, locally saved tiny DINO weights."""

import copy

import pytest
import torch
from torch.nn import functional as F
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from test_render_net_distillation import make_attention_teacher
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.vision.render_net import RenderNet


@pytest.fixture(autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def pretrained_dino(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny_dinov3_renderer")
    config = DINOv3ViTConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, patch_size=16, image_size=32,
        num_register_tokens=2, attention_dropout=0., drop_path_rate=0.,
    )
    DINOv3ViTModel(config).save_pretrained(directory)
    return directory


def _renderer(pretrained_dino):
    model = RenderNet(
        6, image_encoder="dino_lora", image_self_attn_layers=0,
        base_channels=4, conditioning_dim=8, wrench_history_steps=7,
        embed_dim=16, cross_attn_layers=1, patch_size=16,
        decoder_layers=3, num_heads=4, conv_block="bottleneck",
        dino={"pretrained_model_name_or_path": str(pretrained_dino),
              "local_files_only": True, "intermediate_layers": None,
              "lora": {"rank": 2, "alpha": 4., "dropout": 0.,
                       "target_modules": ["q_proj", "v_proj"]}},
        shared_condition_backbone=True, separate_gate_path=False,
        phase_prediction={"enabled": True, "hidden_dim": 8,
                          "freeze_wrench_encoder": True},
        high_attention_render={"enabled": True, "initial_strength": .4,
                               "threshold": .1, "detach_for_rgb": False},
        spatial_modulation={"enabled": True}, residual_scale=.5,
    )
    model.freeze_wrench_encoder()
    model.set_training_stage("joint")
    model.set_stage_mode(True)
    return model


def _observations():
    images = {"side": .2 + .6 * torch.rand(2, 2, 3, 24, 40),
              "wrist": .2 + .6 * torch.rand(2, 2, 3, 16, 32)}
    return images, torch.randn(2, 2, 7, 6)


def _assert_base_frozen(model):
    encoder = model.backbone.dino_encoder
    lora_ids = {id(parameter) for parameter in encoder.lora_parameters()}
    assert lora_ids
    base = {name: parameter for name, parameter in encoder.backbone.named_parameters()
            if id(parameter) not in lora_ids}
    assert base
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in base.values())
    assert not encoder.backbone.training
    return base


def _has_gradient(parameters):
    return any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in parameters)


def _condition_loss(masks, predictions):
    return (F.cross_entropy(predictions["phase_logits"].flatten(0, 1),
                            torch.tensor([0, 1, 1, 2]))
            + F.mse_loss(predictions["contact_alpha"], torch.tensor([[0., .3], [.7, 1.]]))
            + sum((mask - .6).square().mean() for mask in masks.values()))


def test_dino_renderer_identity_shapes_and_selected_rgb_decoding(pretrained_dino, monkeypatch):
    torch.manual_seed(1601)
    model = _renderer(pretrained_dino)
    images, wrench = _observations()
    assert not hasattr(model.backbone, "stem")
    assert not hasattr(model.backbone, "encoders")
    assert not model.backbone.image_self_attention
    rendered, masks, predictions = model(images, wrench, return_aux=True, return_phase=True)
    for key, image in images.items():
        torch.testing.assert_close(rendered[key], image, rtol=0, atol=0)
        assert masks[key].shape == (*image.shape[:2], 1, *image.shape[-2:])
    assert predictions["phase_logits"].shape == (2, 2, 3)
    assert predictions["contact_alpha"].shape == (2, 2)
    _assert_base_frozen(model)

    with torch.no_grad():
        model.output.bias.fill_(.3)
    full, full_masks, full_predictions = model(images, wrench, return_aux=True, return_phase=True)
    counts = []
    decode = model.backbone.decode_features

    def counted_decode(features, *args, **kwargs):
        counts.append(features.shape[0])
        return decode(features, *args, **kwargs)

    monkeypatch.setattr(model.backbone, "decode_features", counted_decode)
    selected = torch.tensor([[True, False], [False, True]])
    sparse, sparse_masks, sparse_predictions = model(
        images, wrench, return_aux=True, return_phase=True, render_observation_mask=selected,
    )
    assert counts == [2, 2]
    for key, image in images.items():
        torch.testing.assert_close(sparse[key][selected], full[key][selected])
        torch.testing.assert_close(sparse[key][~selected], image[~selected], rtol=0, atol=0)
        torch.testing.assert_close(sparse_masks[key], full_masks[key], rtol=0, atol=0)
        assert not torch.equal(sparse[key][selected], image[selected])
    for key in predictions:
        torch.testing.assert_close(sparse_predictions[key], full_predictions[key], rtol=0, atol=0)
    counts.clear()
    free = model(images, wrench, render_observation_mask=torch.zeros_like(selected))
    assert not counts
    for key in images:
        torch.testing.assert_close(free[key], images[key], rtol=0, atol=0)


def test_warmup_joint_and_ema_keep_base_frozen_while_rgb_token_loss_updates_lora(pretrained_dino):
    torch.manual_seed(1602)
    model = _renderer(pretrained_dino)
    base_snapshot = {name: parameter.detach().clone()
                     for name, parameter in _assert_base_frozen(model).items()}
    ema_model = copy.deepcopy(model)
    ema = EMAModel(ema_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    images, wrench = _observations()
    decoder_calls = []
    hook = model.backbone.decoders[0].register_forward_hook(lambda *args: decoder_calls.append(1))
    try:
        model.set_training_stage("condition_warmup")
        model.set_stage_mode(True)
        rendered, masks, predictions = model(
            images, wrench, condition_only=True, return_aux=True, return_phase=True,
        )
        assert rendered is None and not decoder_calls
        assert all(not parameter.requires_grad for parameter in model.rgb_parameters())
        _condition_loss(masks, predictions).backward()
        assert _has_gradient(model.backbone.dino_encoder.lora_parameters())
        _assert_base_frozen(model)
        optimizer.step()
        ema.step(model)

        model.set_training_stage("joint")
        model.set_stage_mode(True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.output.weight.normal_(std=.03)
        teacher, observations, _ = make_attention_teacher()
        images = {key: observations[key] for key in teacher.image_keys}
        wrench = teacher.prepare_context_inputs(observations)[1]
        with torch.no_grad():
            target = teacher._encode_context(observations, apply_curriculum_mask=False)
        rendered = model(images, wrench)
        encoded = teacher.obs_encoder.encode_image_context(
            rendered, input_grad=True, apply_fusion_ffn=True,
        )
        F.mse_loss(encoded, target).backward()
        assert _has_gradient(model.backbone.dino_encoder.lora_parameters())
        assert _has_gradient(model.backbone.decoders.parameters())
        assert _has_gradient(model.output.parameters())
        _assert_base_frozen(model)
        assert all(parameter.grad is None for parameter in teacher.parameters())
        optimizer.step()
        ema.step(model)
    finally:
        hook.remove()

    for stage in ("condition_warmup", "joint"):
        model.set_training_stage(stage)
        model.set_stage_mode(True)
        for name, parameter in _assert_base_frozen(model).items():
            torch.testing.assert_close(parameter, base_snapshot[name], rtol=0, atol=0)
    ema_model.set_stage_mode(False)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in ema_model.parameters())
    for name, parameter in _assert_base_frozen(ema_model).items():
        torch.testing.assert_close(parameter, base_snapshot[name], rtol=0, atol=0)


def test_dino_renderer_checkpoint_roundtrip_loads_strictly(pretrained_dino, tmp_path):
    model = _renderer(pretrained_dino).eval()
    with torch.no_grad():
        model.output.weight.normal_(std=.03)
        for parameter in model.backbone.dino_encoder.lora_parameters():
            parameter.add_(.01)
    checkpoint = tmp_path / "dino_renderer.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = _renderer(pretrained_dino).eval()
    result = restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    images, wrench = _observations()
    with torch.no_grad():
        expected = model(images, wrench, return_aux=True, return_phase=True)
        actual = restored(images, wrench, return_aux=True, return_phase=True)
    for expected_group, actual_group in zip(expected, actual):
        for key in expected_group:
            torch.testing.assert_close(actual_group[key], expected_group[key], rtol=0, atol=0)
    assert next(model.backbone.dino_encoder.backbone.parameters()).data_ptr() != next(
        restored.backbone.dino_encoder.backbone.parameters()).data_ptr()
    _assert_base_frozen(restored)


def test_compiled_dino_and_film_fullgraph_match_eager_outputs_and_gradients(pretrained_dino, monkeypatch):
    torch.manual_seed(1603)
    eager = _renderer(pretrained_dino)
    with torch.no_grad():
        eager.output.weight.normal_(std=.03)
        for modulation in eager.spatial_modulations:
            modulation.affine[-1].weight.normal_(std=.02)
        for parameter in eager.backbone.dino_encoder.lora_parameters():
            parameter.add_(.01)
    compiled = copy.deepcopy(eager)
    keys_before = tuple(compiled.state_dict())
    compiled.compile_backbone(backend="aot_eager", fullgraph=True)
    assert tuple(compiled.state_dict()) == keys_before
    calls = {"encode": 0, "decode": 0}
    for name, attribute in (("encode", "_compiled_shared_encode"),
                            ("decode", "_compiled_shared_decode")):
        implementation = getattr(compiled, attribute)

        def counted(*args, _name=name, _implementation=implementation, **kwargs):
            calls[_name] += 1
            return _implementation(*args, **kwargs)

        monkeypatch.setattr(compiled, attribute, counted)
    images, wrench = _observations()
    selected = torch.tensor([[True, False], [False, True]])
    outputs = []
    for model in (eager, compiled):
        rendered, masks, predictions = model(
            images, wrench, return_aux=True, return_phase=True, render_observation_mask=selected,
        )
        loss = _condition_loss(masks, predictions) + sum(
            (rendered[key] - (image + .05)).square().mean() for key, image in images.items()
        )
        loss.backward()
        outputs.append((rendered, masks, predictions))
        _assert_base_frozen(model)
        assert _has_gradient(model.backbone.dino_encoder.lora_parameters())
    assert calls == {"encode": 2, "decode": 2}
    for expected_group, actual_group in zip(outputs[0], outputs[1]):
        for key in expected_group:
            torch.testing.assert_close(actual_group[key], expected_group[key], rtol=2e-4, atol=2e-6)
    for name, parameter in eager.named_parameters():
        actual = dict(compiled.named_parameters())[name]
        assert (parameter.grad is None) == (actual.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, actual.grad, rtol=3e-3, atol=2e-5,
                                       msg=lambda message: f"{name}: {message}")
    calls.update(encode=0, decode=0)
    with torch.no_grad():
        compiled.eval()(images, wrench)
    assert calls == {"encode": 0, "decode": 0}
