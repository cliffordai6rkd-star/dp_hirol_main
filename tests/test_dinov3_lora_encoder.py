import copy

import pytest
import torch
from transformers import AutoModel, DINOv3ViTConfig, DINOv3ViTModel

from diffusion_policy.model.vision.dinov3_lora_encoder import DINOv3LoRAEncoder, LoRALinear


def make_encoder(monkeypatch, *, depth=4, **kwargs):
    torch.manual_seed(41)
    reference = DINOv3ViTModel(DINOv3ViTConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=depth,
        num_attention_heads=4, patch_size=4, image_size=8, num_register_tokens=2,
        pos_embed_jitter=0.2, attention_dropout=0.1, drop_path_rate=0.2,
    )).eval()
    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *args, **options: copy.deepcopy(reference))
    return DINOv3LoRAEncoder("unused", rank=2, alpha=4, **kwargs), reference


def test_zero_initialized_lora_matches_pretrained_features_and_intermediate_layers(monkeypatch):
    encoder, reference = make_encoder(monkeypatch)
    image = torch.rand(2, 3, 8, 12)
    with torch.no_grad():
        expected = reference(
            pixel_values=(image - encoder.image_mean) / encoder.image_std,
            output_hidden_states=True,
        )
        final, intermediate = encoder(image)
    assert encoder.intermediate_layers == (0, 1, 2)
    assert final.shape == (2, 6, 16)
    assert len(intermediate) == 3
    torch.testing.assert_close(final, expected.last_hidden_state[:, 3:], rtol=0, atol=0)
    for index, tokens in zip(encoder.intermediate_layers, intermediate):
        torch.testing.assert_close(tokens, reference.norm(expected.hidden_states[index + 1])[:, 3:],
                                   rtol=0, atol=0)
    assert encoder.hidden_size == 16
    assert encoder.patch_size == 4
    assert encoder.num_register_tokens == 2
    assert len(list(encoder.lora_parameters())) == 4 * 2 * 2
    for layer in encoder.backbone.layer:
        assert isinstance(layer.attention.q_proj, LoRALinear)
        assert isinstance(layer.attention.v_proj, LoRALinear)
        assert isinstance(layer.attention.k_proj, torch.nn.Linear)


def test_lora_uses_independent_weights_and_backpropagates_without_changing_the_base(monkeypatch):
    encoder, teacher = make_encoder(monkeypatch)
    original = {name: parameter.detach().clone() for name, parameter in teacher.named_parameters()}
    assert next(encoder.backbone.parameters()).data_ptr() != next(teacher.parameters()).data_ptr()
    image = torch.rand(2, 3, 8, 12, requires_grad=True)
    optimizer = torch.optim.SGD(encoder.lora_parameters(), lr=0.2)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        final, intermediate = encoder(image)
        loss = (final * torch.randn_like(final)).sum()
        loss = loss + sum((tokens * torch.randn_like(tokens)).sum() for tokens in intermediate)
        loss.backward()
        assert image.grad is not None and image.grad.abs().sum() > 0
        for layer in encoder.backbone.layer:
            adapter = layer.attention.v_proj
            assert adapter.lora_B.weight.grad.abs().sum() > 0
            if step:
                assert adapter.lora_A.weight.grad.abs().sum() > 0
        assert all(parameter.grad is None for name, parameter in encoder.named_parameters()
                   if "lora_" not in name)
        optimizer.step()
    for name, parameter in teacher.named_parameters():
        torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)


def test_enforce_freeze_preserves_stage_lora_ownership_and_base_eval_mode(monkeypatch):
    encoder, _ = make_encoder(monkeypatch, dropout=0.3)
    encoder.requires_grad_(True)
    encoder.enforce_base_freeze()
    assert all(parameter.requires_grad == ("lora_" in name)
               for name, parameter in encoder.named_parameters())
    encoder.train()
    assert not encoder.backbone.training
    assert not encoder.backbone.rope_embeddings.training
    for layer in encoder.backbone.layer:
        assert not layer.training and not layer.attention.training
        assert layer.attention.q_proj.dropout.training
        assert not layer.attention.q_proj.base.training
    encoder.requires_grad_(False)
    encoder.enforce_base_freeze()
    encoder.train()
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    encoder.eval()
    assert not encoder.backbone.layer[0].attention.q_proj.dropout.training


def test_twelve_layer_default_and_explicit_intermediate_layers(monkeypatch):
    encoder, _ = make_encoder(monkeypatch, depth=12)
    assert encoder.intermediate_layers == (2, 5, 8)
    encoder, _ = make_encoder(monkeypatch, intermediate_layers=(1, 3))
    assert encoder.intermediate_layers == (1, 3)


def test_compiled_encoder_preserves_lora_and_image_gradients(monkeypatch):
    encoder, _ = make_encoder(monkeypatch)
    # Exercise both LoRA factors rather than only the initially nonzero B gradient.
    with torch.no_grad():
        for layer in encoder.backbone.layer:
            layer.attention.q_proj.lora_B.weight.normal_(std=0.02)
            layer.attention.v_proj.lora_B.weight.normal_(std=0.02)
    compiled_encoder = copy.deepcopy(encoder)
    compiled = torch.compile(compiled_encoder, backend="aot_eager", fullgraph=True, dynamic=False)
    image = torch.rand(2, 3, 8, 12, requires_grad=True)
    compiled_image = image.detach().clone().requires_grad_()
    expected = encoder(image)
    actual = compiled(compiled_image)
    for actual_tokens, expected_tokens in zip((actual[0], *actual[1]), (expected[0], *expected[1])):
        torch.testing.assert_close(actual_tokens, expected_tokens)
    weights = [torch.randn_like(tokens) for tokens in (expected[0], *expected[1])]
    sum((tokens * weight).sum() for tokens, weight in zip((expected[0], *expected[1]), weights)).backward()
    sum((tokens * weight).sum() for tokens, weight in zip((actual[0], *actual[1]), weights)).backward()
    torch.testing.assert_close(compiled_image.grad, image.grad)
    for actual_parameter, expected_parameter in zip(compiled_encoder.lora_parameters(), encoder.lora_parameters()):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)


@pytest.mark.parametrize("options", [
    {"rank": 0}, {"alpha": 0}, {"dropout": 1}, {"target_modules": ()},
    {"target_modules": ("not_a_projection",)}, {"intermediate_layers": (3, 1)},
])
def test_invalid_lora_configuration_is_rejected(monkeypatch, options):
    _, reference = make_encoder(monkeypatch)
    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *args, **kwargs: copy.deepcopy(reference))
    with pytest.raises(ValueError):
        DINOv3LoRAEncoder("unused", **options)
