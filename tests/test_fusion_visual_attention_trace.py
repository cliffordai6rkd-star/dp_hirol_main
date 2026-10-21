"""Unit tests for the residual-aware fusion rollout helper."""

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from diffusion_policy.model.vision.dino_token_baseline_obs_encoder import DinoTokenBaselineObsEncoder
from diffusion_policy.model.vision.force_aware_obs_encoder import ForceAwareObsEncoder
from diffusion_policy.scripts.fusion_visual_attention_trace import FusionVisualAttentionTrace


class PatchBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, patch_size=2, num_register_tokens=0)
        self.projection = nn.Linear(3, 4)

    def forward(self, pixel_values):
        patches = self.projection(F.avg_pool2d(pixel_values, 2).flatten(2).transpose(1, 2))
        return SimpleNamespace(last_hidden_state=torch.cat((patches.mean(1, keepdim=True), patches), 1))


def make_policy(mode="force", query="image", camera_mode="joint", self_layers=1,
                cross_layers=2, steps=2, shapes=None, wrench=2):
    torch.manual_seed(84)
    shapes = shapes or {"side": (3, 2, 2), "wrist": (3, 2, 4)}
    common = dict(
        pretrained_model_name_or_path="unused", image_shapes=shapes,
        n_emb=4, n_head=2, max_obs_steps=steps, wrench_history_steps=wrench,
        dropout=0.0, backbone=PatchBackbone(),
    )
    if mode == "force":
        encoder = ForceAwareObsEncoder(
            **common, pre_cross_self_attn_layers=self_layers,
            cross_attn_layers=cross_layers, cross_attn_query=query,
            camera_attention_mode=camera_mode, force_temporal_encoder="lstm",
        )
    else:
        encoder = DinoTokenBaselineObsEncoder(
            **common, mode=mode, self_attn_layers=self_layers,
            force_temporal_encoder="gru", force_encoder_layers=1,
        )
    policy = SimpleNamespace(obs_encoder=encoder.eval(), n_obs_steps=steps)
    sample = {"images": {key: torch.rand(1, steps, *shape) for key, shape in shapes.items()},
              "wrench_history": torch.randn(1, steps, wrench, 6)}
    return policy, sample


def prescribe_attention(module, matrices):
    matrices = [torch.as_tensor(value, dtype=torch.float32) for value in matrices]
    calls = []

    def forward(self, query, key, value, **kwargs):
        assert kwargs["need_weights"] is True
        assert kwargs["average_attn_weights"] is False
        matrix = matrices[len(calls)]
        calls.append(matrix)
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0).expand(query.shape[0], -1, -1)
        assert matrix.shape == (query.shape[0], query.shape[1], key.shape[1])
        weights = matrix[:, None].expand(-1, self.num_heads, -1, -1).clone()
        return torch.zeros_like(query), weights.to(query)

    module.forward = MethodType(forward, module)


def residual_sa(matrix):
    matrix = torch.as_tensor(matrix, dtype=torch.float32)
    matrix = matrix + torch.eye(matrix.shape[-1])
    return matrix / matrix.sum(-1, keepdim=True)


@pytest.mark.parametrize("query", ["image", "wrench"])
@pytest.mark.parametrize("camera_mode", ["joint", "per_camera"])
@pytest.mark.parametrize("self_layers,cross_layers", [(0, 0), (0, 2), (2, 0), (2, 2)])
def test_force_variants_conserve_mass(query, camera_mode, self_layers, cross_layers):
    policy, sample = make_policy(query=query, camera_mode=camera_mode,
                                 self_layers=self_layers, cross_layers=cross_layers)
    trace = FusionVisualAttentionTrace(policy, camera_fusion_approximation="equal")
    with torch.inference_mode(), trace:
        expected = policy.obs_encoder(**sample)
        actual = policy.obs_encoder(**sample)
        maps, diagnostics = trace.project(torch.rand(2, 3, trace.context_tokens))
    torch.testing.assert_close(actual, expected)
    assert maps["side"].shape == (2, 3, 2, 1)
    assert maps["wrist"].shape == (2, 3, 2, 2)
    torch.testing.assert_close(
        diagnostics["visual_mass"] + diagnostics["wrench_mass"],
        diagnostics["input_mass"], atol=2e-5, rtol=0,
    )
    assert diagnostics["mass_conservation_error"].max() < 2e-5


@pytest.mark.parametrize("query", ["image", "wrench"])
def test_multiple_ca_layers_accumulate_fixed_memory(query):
    policy, sample = make_policy(query=query, self_layers=0, steps=1,
                                 shapes={"wrist": (3, 2, 4)})
    prescribe_attention(policy.obs_encoder.cross_attention, [[[1, 0], [0, 1]]])
    prescribe_attention(policy.obs_encoder.extra_cross_attention_layers[0].cross_attention,
                        [[[0, 1], [1, 0]]])
    with FusionVisualAttentionTrace(policy) as trace:
        policy.obs_encoder(**sample)
        maps, diagnostics = trace.project(torch.tensor([1.0, 0.0]))
    if query == "image":
        torch.testing.assert_close(maps["wrist"], torch.tensor([[0.25, 0.0]]))
        torch.testing.assert_close(diagnostics["wrench_mass"], torch.tensor(0.75))
    else:
        torch.testing.assert_close(maps["wrist"], torch.tensor([[0.25, 0.5]]))
        torch.testing.assert_close(diagnostics["wrench_mass"], torch.tensor(0.25))


def test_pre_self_attention_propagates_full_token_paths():
    policy, sample = make_policy(query="wrench", self_layers=2, cross_layers=1,
                                 steps=1, shapes={"wrist": (3, 2, 4)}, wrench=1)
    matrices = (
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    )
    for layer, matrix in zip(policy.obs_encoder.pre_cross_self_attention.layers, matrices):
        prescribe_attention(layer.self_attn, [matrix])
    prescribe_attention(policy.obs_encoder.cross_attention, [[[0, 1]]])
    with FusionVisualAttentionTrace(policy) as trace:
        policy.obs_encoder(**sample)
        maps, diagnostics = trace.project(torch.ones(1))
    expected = torch.tensor([0.0, 0.5, 0.5]) @ residual_sa(matrices[1]) @ residual_sa(matrices[0])
    torch.testing.assert_close(maps["wrist"], expected[:2].unsqueeze(0))
    torch.testing.assert_close(diagnostics["wrench_mass"], expected[2])


@pytest.mark.parametrize("mode", ["dino", "image_self_attn", "self_attn"])
def test_baseline_modes(mode):
    policy, sample = make_policy(mode=mode, self_layers=2)
    with FusionVisualAttentionTrace(policy) as trace:
        policy.obs_encoder(**sample)
        maps, diagnostics = trace.project(torch.ones(trace.context_tokens))
    assert maps["wrist"].shape == (2, 2)
    torch.testing.assert_close(diagnostics["input_mass"], torch.tensor(float(trace.context_tokens)))
    torch.testing.assert_close(diagnostics["visual_mass"] + diagnostics["wrench_mass"], diagnostics["input_mass"])
    if mode != "self_attn":
        torch.testing.assert_close(diagnostics["wrench_mass"], torch.tensor(0.0))


def test_flattened_baseline_self_attention_tracks_earlier_observation():
    policy, sample = make_policy(mode="self_attn", self_layers=1, steps=2,
                                 shapes={"wrist": (3, 2, 2)}, wrench=1)
    prescribe_attention(policy.obs_encoder.fusion.layers[0].self_attn,
                        [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]])
    with FusionVisualAttentionTrace(policy) as trace:
        policy.obs_encoder(**sample)
        maps, _ = trace.project(torch.tensor([0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(maps["wrist"], torch.tensor([[0.5], [0.0]]))


def test_image_self_attention_preserves_batch_and_head_average():
    policy, sample = make_policy(mode="image_self_attn", self_layers=1,
                                 shapes={"wrist": (3, 2, 4)})
    module = policy.obs_encoder.fusion.layers[0].self_attn
    weights = torch.tensor([
        [[[1, 0], [1, 0]], [[0, 1], [0, 1]]],
        [[[0, 1], [0, 1]], [[0, 1], [0, 1]]],
    ], dtype=torch.float32)

    def attention(self, query, key, value, **kwargs):
        assert kwargs["need_weights"] and not kwargs["average_attn_weights"]
        return torch.zeros_like(query), weights.to(query)

    module.forward = MethodType(attention, module)
    with FusionVisualAttentionTrace(policy) as trace:
        policy.obs_encoder(**sample)
        maps, diagnostics = trace.project(torch.tensor([0.0, 1.0, 1.0, 0.0]))
    torch.testing.assert_close(maps["wrist"], torch.tensor([[0.25, 0.75], [0.5, 0.5]]))
    torch.testing.assert_close(diagnostics["visual_mass"], torch.tensor(2.0))


def test_per_camera_image_query_concatenates_camera_rows():
    policy, sample = make_policy(camera_mode="per_camera", self_layers=0,
                                 cross_layers=1, steps=1)
    prescribe_attention(policy.obs_encoder.cross_attention, [[[1, 0]], [[0, 1], [1, 0]]])
    with FusionVisualAttentionTrace(policy) as trace:
        policy.obs_encoder(**sample)
        maps, diagnostics = trace.project(torch.tensor([2.0, 4.0, 6.0]))
    torch.testing.assert_close(maps["side"], torch.tensor([[1.0]]))
    torch.testing.assert_close(maps["wrist"], torch.tensor([[2.0, 3.0]]))
    torch.testing.assert_close(diagnostics["wrench_mass"], torch.tensor(6.0))


def test_per_camera_wrench_query_requires_explicit_approximation():
    policy, sample = make_policy(query="wrench", camera_mode="per_camera",
                                 self_layers=0, cross_layers=1, steps=1)
    with pytest.raises(ValueError, match="attention alone cannot recover"):
        FusionVisualAttentionTrace(policy)
    prescribe_attention(policy.obs_encoder.cross_attention, [[[1], [1]], [[1, 0], [0, 1]]])
    with FusionVisualAttentionTrace(policy, camera_fusion_approximation="equal") as trace:
        policy.obs_encoder(**sample)
        maps, diagnostics = trace.project(torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(maps["side"], torch.tensor([[0.25]]))
    torch.testing.assert_close(maps["wrist"], torch.tensor([[0.25, 0.0]]))
    assert diagnostics["equal_camera_fusion_approximation"]


def test_no_attention_keeps_identity_and_nonvisual_mass():
    policy, sample = make_policy(self_layers=0, cross_layers=0, steps=1)
    with FusionVisualAttentionTrace(policy) as trace:
        policy.obs_encoder(**sample)
        maps, diagnostics = trace.project(torch.arange(5, dtype=torch.float32))
    torch.testing.assert_close(maps["side"], torch.tensor([[0.0]]))
    torch.testing.assert_close(maps["wrist"], torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(diagnostics["wrench_mass"], torch.tensor(7.0))


@pytest.mark.parametrize("bad", [torch.ones(3), torch.full((6,), -1.0), torch.full((6,), float("nan"))])
def test_invalid_relevance_rejected(bad):
    policy, sample = make_policy()
    with FusionVisualAttentionTrace(policy) as trace:
        policy.obs_encoder(**sample)
        with pytest.raises(ValueError, match="context_relevance"):
            trace.project(bad)


def test_cleanup_and_reuse():
    policy, sample = make_policy()
    trace = FusionVisualAttentionTrace(policy)
    with pytest.raises(ValueError, match="one sample"):
        with trace:
            with pytest.raises(RuntimeError, match="No complete"):
                trace.project(torch.ones(trace.context_tokens))
            policy.obs_encoder(**sample)
            trace.project(torch.ones(trace.context_tokens))
            sample["images"]["side"] = sample["images"]["side"].repeat(2, 1, 1, 1, 1)
            policy.obs_encoder(**sample)
    assert not trace.active and not trace.handles
    for module in (*trace.self_attention, *trace.cross_attention):
        assert not module._forward_pre_hooks and not module._forward_hooks


def test_unsupported_and_training_encoder_rejected():
    policy, _ = make_policy()
    policy.obs_encoder.train()
    with pytest.raises(ValueError, match="eval-mode"):
        with FusionVisualAttentionTrace(policy):
            pass
    with pytest.raises(ValueError, match="Fusion rollout supports"):
        FusionVisualAttentionTrace(SimpleNamespace(obs_encoder=nn.Identity(), n_obs_steps=2))
