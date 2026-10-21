from types import MethodType

import pytest
import torch
import torch.nn as nn

from diffusion_policy.model.diffusion.context_transformer_for_diffusion import ContextTransformerForDiffusion
from diffusion_policy.scripts.dit_action_attention_capture import DitActionAttentionCapture


class TinyPolicy(nn.Module):
    def __init__(self, decoder_layers=2, condition_layers=1, horizon=4):
        super().__init__()
        self.obs_encoder = nn.Identity()
        self.model = ContextTransformerForDiffusion(
            input_dim=2, output_dim=2, horizon=horizon, n_emb=4, n_head=2,
            n_layer=decoder_layers, n_cond_layers=condition_layers,
            max_context_tokens=3, p_drop_attn=0.0,
        )
        self.n_obs_steps = 2
        self.action_start_index = 2
        self.n_action_steps = 2
        self.horizon = horizon

    def predict_action(self, context):
        context = self.obs_encoder(context)
        sample = torch.ones(context.shape[0], self.horizon, 2)
        for timestep in (9, 4, 0):
            sample = self.model(sample, timestep, context)
        return sample


def make_policy(**kwargs):
    torch.manual_seed(11)
    return TinyPolicy(**kwargs).eval()


def context():
    return torch.randn(1, 2, 4)


def prescribe(module, matrix):
    matrix = torch.as_tensor(matrix, dtype=torch.float32)
    if matrix.ndim == 2:
        matrix = matrix[None].repeat(2, 1, 1)

    def known_attention(self, query, key, value, **kwargs):
        assert kwargs["need_weights"] is True
        assert kwargs["average_attn_weights"] is False
        assert matrix.shape == (2, query.shape[1], key.shape[1])
        return torch.zeros_like(query), matrix[None].to(query)

    module.forward = MethodType(known_attention, module)


def test_rollout_reverses_ca_then_sa_and_keeps_timestep_path():
    policy = make_policy(horizon=2)
    prescribe(policy.model.condition_encoder.layers[0].self_attn,
              [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    first, last = policy.model.decoder.layers
    prescribe(first.self_attn, [[1, 0], [0, 1]])
    prescribe(first.multihead_attn, [[1, 0, 0], [0, 0, 1]])
    prescribe(last.self_attn, [[0, 1], [1, 0]])
    prescribe(last.multihead_attn, [[0, 0, 1], [0, 1, 0]])
    with DitActionAttentionCapture(policy, action_query="last") as capture:
        policy.predict_action(context())
        torch.testing.assert_close(capture.context_relevance(), torch.tensor([0.3125, 0.125]))
        metadata = capture.diagnostics()
        assert metadata["time_mass"] == pytest.approx(0.3125)
        assert metadata["input_action_mass"] == pytest.approx(0.25)
        assert metadata["context_mass"] == pytest.approx(0.4375)
        assert sum(metadata[key] for key in ("time_mass", "input_action_mass", "context_mass")) == pytest.approx(1)


def test_executed_chunk_uses_policy_action_start_and_head_average():
    policy = make_policy(decoder_layers=1, condition_layers=0)
    layer = policy.model.decoder.layers[0]
    prescribe(layer.self_attn, torch.eye(4))
    prescribe(layer.multihead_attn, [
        [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]],
    ])
    with DitActionAttentionCapture(policy) as capture:
        policy.predict_action(context())
        torch.testing.assert_close(capture.context_relevance(), torch.tensor([0.125, 0.375]))
        assert capture.diagnostics()["selected_action_indices"] == [2, 3]


@pytest.mark.parametrize("query,index,expected", [
    ("first", None, [0]), ("last", None, [3]), ("all", None, [0, 1, 2, 3]),
    ("executed", None, [2, 3]), ("all", -2, [2]),
])
def test_action_query_options(query, index, expected):
    policy = make_policy()
    with DitActionAttentionCapture(policy, action_query=query, action_index=index) as capture:
        policy.predict_action(context())
        assert capture.diagnostics()["selected_action_indices"] == expected


@pytest.mark.parametrize("selection,indices", [("first", [0]), ("last", [2]), (1, [1]), ("-2", [1])])
def test_denoising_selection_keeps_calls_separate(selection, indices):
    policy = make_policy()
    with DitActionAttentionCapture(policy, denoising_step=selection) as capture:
        policy.predict_action(context())
        all_context = capture.step_context_relevance()
        torch.testing.assert_close(capture.context_relevance(), all_context[indices[0]])
        assert all_context.shape == (3, 2)
        assert capture.diagnostics()["selected_denoising_indices"] == indices


def test_default_selects_last_call_without_cross_step_average():
    policy = make_policy()
    with DitActionAttentionCapture(policy) as capture:
        policy.predict_action(context())
        torch.testing.assert_close(capture.context_relevance(), capture.step_context_relevance()[-1])
        assert capture.diagnostics()["denoising_step"] == "last"


def test_real_capture_preserves_model_output_parameters_and_hooks():
    policy = make_policy(condition_layers=2)
    values = context()
    before = {key: value.clone() for key, value in policy.state_dict().items()}
    original = policy.predict_action(values)
    with DitActionAttentionCapture(policy) as capture:
        captured = policy.predict_action(values)
        relevance = capture.context_relevance()
        metadata = capture.diagnostics()
        assert relevance.shape == (2,) and (relevance >= 0).all()
        assert metadata["time_mass"] + metadata["input_action_mass"] + metadata["context_mass"] == pytest.approx(1, abs=1e-6)
    torch.testing.assert_close(captured, original, rtol=1e-5, atol=1e-6)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert not capture.active
    for module in policy.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks


def test_capture_clears_between_predictions_and_restores_hooks_on_error():
    policy = make_policy()
    capture = DitActionAttentionCapture(policy)
    with capture:
        policy.predict_action(context())
        assert len(capture.diagnostics()["timesteps"]) == 3
        policy.predict_action(context())
        assert len(capture.diagnostics()["timesteps"]) == 3
    with pytest.raises(ValueError, match="one sample"):
        with capture:
            policy.predict_action(context().repeat(2, 1, 1))
    assert not capture.active
    for module in policy.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks


@pytest.mark.parametrize("kwargs,match", [
    ({"layer_index": -3}, "layer_index"), ({"layer_index": 2}, "layer_index"),
    ({"layer_index": 0.5}, "layer_index"), ({"action_query": "unknown"}, "action_query"),
    ({"denoising_step": "mean"}, "single sampling-call index"),
    ({"denoising_step": "unknown"}, "denoising_step"), ({"denoising_step": 1.5}, "denoising_step"),
])
def test_invalid_options_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DitActionAttentionCapture(make_policy(), **kwargs)


def test_invalid_action_and_training_mode():
    policy = make_policy()
    with pytest.raises(ValueError, match="action_index"):
        with DitActionAttentionCapture(policy, action_index=4):
            policy.predict_action(context())
    policy.train()
    with pytest.raises(ValueError, match="eval-mode"):
        with DitActionAttentionCapture(policy):
            pass


def test_unsupported_model_rejected():
    policy = make_policy()
    policy.model = nn.Identity()
    with pytest.raises(ValueError, match="ContextTransformerForDiffusion"):
        DitActionAttentionCapture(policy)
