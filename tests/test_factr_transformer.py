"""Regression checks for FACTR's full-memory action decoder semantics."""

import math

import pytest
import torch
from torch.nn import functional as F

from diffusion_policy.model.act.factr_transformer import FactrActionTransformer


@pytest.fixture(autouse=True)
def _two_cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


def _model(**kwargs):
    config = dict(
        n_emb=16, n_head=4, num_encoder_layers=2, num_decoder_layers=2,
        dim_feedforward=24, dropout=0.1, horizon=3, action_dim=7,
    )
    config.update(kwargs)
    return FactrActionTransformer(**config)


def _attention(module, query, key, value):
    """Independent, explicit attention math; does not call MultiheadAttention."""
    weights = module.in_proj_weight.chunk(3)
    biases = module.in_proj_bias.chunk(3)
    batch = query.shape[0]
    heads = module.num_heads
    width = query.shape[-1]
    projected = [
        F.linear(tensor, weight, bias).reshape(batch, -1, heads, width // heads)
        .transpose(1, 2)
        for tensor, weight, bias in zip((query, key, value), weights, biases)
    ]
    q, k, v = projected
    probability = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(width // heads), -1)
    output = (probability @ v).transpose(1, 2).reshape(batch, -1, width)
    return F.linear(output, module.out_proj.weight, module.out_proj.bias)


def _norm(module, value):
    return F.layer_norm(value, module.normalized_shape, module.weight, module.bias, module.eps)


def _feedforward(module, value):
    return F.linear(F.relu(F.linear(value, module.linear1.weight, module.linear1.bias)),
                    module.linear2.weight, module.linear2.bias)


def _reference_actions(model, tokens):
    # Upstream positions are constructed in float32 before any dtype conversion.
    position = torch.arange(tokens.shape[1], dtype=torch.float32)[:, None]
    frequency = torch.exp(torch.arange(0, model.n_emb, 2, dtype=torch.float32)
                          * -(math.log(10000.0) / model.n_emb))
    pos = torch.stack((torch.sin(position * frequency), torch.cos(position * frequency)),
                      dim=-1).flatten(-2).to(tokens)[None]
    memory = tokens
    for layer in model.encoder.layers:
        update = _attention(layer.self_attn, memory + pos, memory + pos, memory)
        memory = _norm(layer.norm1, memory + update)
        memory = _norm(layer.norm2, memory + _feedforward(layer, memory))
    action_pos = model.ac_query.weight[None].expand(tokens.shape[0], -1, -1)
    action = torch.zeros_like(action_pos)
    for layer in model.decoder.layers:
        update = _attention(layer.self_attn, action + action_pos, action + action_pos, action)
        action = _norm(layer.norm1, action + update)
        update = _attention(layer.multihead_attn, action + action_pos, memory + pos, memory)
        action = _norm(layer.norm2, action + update)
        action = _norm(layer.norm3, action + _feedforward(layer, action))
    return F.linear(action, model.ac_proj.weight, model.ac_proj.bias)


@pytest.mark.parametrize("encoder_layers", [0, 2])
def test_actions_and_input_gradients_match_postnorm_act_equations(encoder_layers):
    torch.manual_seed(41)
    model = _model(num_encoder_layers=encoder_layers).double().eval()
    tokens = torch.randn(2, 7, 16, dtype=torch.float64, requires_grad=True)
    actual = model(tokens)
    reference = _reference_actions(model, tokens)
    torch.testing.assert_close(actual, reference, rtol=1e-9, atol=1e-9)
    actual_grad, = torch.autograd.grad(actual.square().sum(), tokens, retain_graph=True)
    reference_grad, = torch.autograd.grad(reference.square().sum(), tokens)
    torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-8, atol=1e-9)


def test_decoder_reads_force_rows_and_backpropagates_to_them_without_encoder():
    torch.manual_seed(6)
    model = _model(num_encoder_layers=0).eval()
    memory = torch.randn(2, 6, 16, requires_grad=True)
    seen_memory = []

    def observe_memory(module, args, kwargs):
        seen_memory.append(kwargs["value"])

    hook = model.decoder.layers[0].multihead_attn.register_forward_pre_hook(
        observe_memory, with_kwargs=True
    )
    actions = model.decode(memory)
    hook.remove()
    torch.testing.assert_close(seen_memory[0], memory.transpose(0, 1))
    assert actions.shape == (2, 3, 7)
    actions.square().sum().backward()
    # The last two rows represent force tokens; neither is discarded.
    assert (memory.grad[:, -2:].norm(dim=-1) > 0).all()
    changed = memory.detach().clone()
    changed[:, -2:] = torch.flip(changed[:, -2:], dims=[-1])
    assert not torch.allclose(actions, model.decode(changed))


def test_encoder_retains_complete_context_and_zero_layer_is_identity():
    torch.manual_seed(12)
    tokens = torch.randn(2, 11, 16)
    model = _model().eval()
    memory = model.encode(tokens)
    assert memory.shape == tokens.shape
    torch.testing.assert_close(model(tokens), model.decode(memory), rtol=0, atol=0)
    passthrough = _model(num_encoder_layers=0).eval()
    torch.testing.assert_close(passthrough.encode(tokens), tokens, rtol=0, atol=0)


def test_eval_actions_are_deterministic_and_need_no_diffusion_noise():
    torch.manual_seed(17)
    model = _model().eval()
    tokens = torch.randn(2, 5, 16)
    before = torch.get_rng_state()
    first = model(tokens)
    assert torch.equal(torch.get_rng_state(), before)
    torch.manual_seed(834)
    second = model(tokens)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_frozen_decoder_preserves_gradient_to_rendered_context():
    torch.manual_seed(5)
    model = _model(num_encoder_layers=0).eval().requires_grad_(False)
    memory = torch.randn(2, 5, 16, requires_grad=True)
    model.decode(memory).square().sum().backward()
    assert memory.grad is not None and memory.grad.norm() > 0
    assert all(parameter.grad is None for parameter in model.parameters())
