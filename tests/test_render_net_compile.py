"""Compilation follows the shapes used by sparse RGB rendering."""

from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from diffusion_policy.model.vision.render_net import RenderNet
from diffusion_policy.workspace.train_distill_render_net_workspace import TrainDistillRenderNetWorkspace


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("selected", [False, True])
def test_sparse_independent_rgb_compiles_with_dynamic_batch(monkeypatch, shared, selected):
    workspace = object.__new__(TrainDistillRenderNetWorkspace)
    workspace.cfg = OmegaConf.create({
        "training": {"compile": {"enabled": True}},
        "distillation": {"skip_unselected_observations": selected},
    })
    # Only inspect compiler dispatch; this test does not allocate CUDA tensors.
    workspace.device = torch.device("cuda")
    workspace.model = RenderNet(
        7, base_channels=4, embed_dim=16, conditioning_dim=8, num_heads=4,
        image_self_attn_layers=1, cross_attn_layers=1,
        shared_condition_backbone=shared, separate_gate_path=not shared,
        phase_prediction={"enabled": True, "hidden_dim": 8},
    )
    workspace.teacher_policy = SimpleNamespace(
        obs_encoder=SimpleNamespace(encode_image_context=lambda images: images))
    calls = []

    def compile_spy(function, **options):
        calls.append(options)
        return function

    monkeypatch.setattr(torch, "compile", compile_spy)
    workspace._setup_acceleration()
    expected = [False, True, selected] if shared else [selected, selected]
    assert [call["dynamic"] for call in calls] == expected
    assert all(call["fullgraph"] for call in calls)
