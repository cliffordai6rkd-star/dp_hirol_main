"""Causal histories and phase supervision in the actual renderer training path."""

import contextlib
import copy

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

from test_phase_binary_fusion import teacher_and_obs, IMAGE_SHAPES
from diffusion_policy.dataset.lerobot_v3_dataset import LeRobotV3ImageDataset
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.vision.causal_phase_gate import CausalPhaseGate
from diffusion_policy.model.vision.render_net import RenderNet
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor
from diffusion_policy.scripts.visualize_render_net_episode import render_observations
from diffusion_policy.workspace.train_distill_render_net_workspace import (
    TrainDistillRenderNetWorkspace, student_context_tokens,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(batch=2):
    return {"state": torch.randn(batch, 2, 16, 21),
            "images": {key: torch.randint(0, 256, (batch, 2, 4, 3, 32, 32), dtype=torch.uint8)
                       for key in ("side", "wrist")}}


def renderer():
    model = RenderNet(
        6, base_channels=4, conditioning_dim=16, embed_dim=16, num_heads=4,
        image_self_attn_layers=1, cross_attn_layers=1, decoder_layers=4, patch_size=16,
        image_keys=tuple(IMAGE_SHAPES), separate_gate_path=True,
        high_attention_render={"enabled": True, "threshold": 0.0},
        phase_predictor={"enabled": True, "hidden_dim": 8, "image_size": 32},
    )
    model.phase_predictor.fit_normalizer(torch.randn(40, 21))
    return model


def test_each_observation_has_its_own_past_without_later_observation_features():
    model = CausalPhaseGate(("side", "wrist"), hidden_dim=8, image_size=32).eval()
    model.fit_normalizer(torch.randn(40, 21))
    original = inputs()
    changed = copy.deepcopy(original)
    changed["state"][:, 1] += 100
    for pixels in changed["images"].values():
        pixels[:, 1] = 0
    first, second = model(original), model(changed)
    torch.testing.assert_close(first["p_open"][:, 0], second["p_open"][:, 0], rtol=0, atol=0)
    assert not torch.equal(first["p_open"][:, 1], second["p_open"][:, 1])
    assert first["p_open"].requires_grad and not first["contact_alpha"].requires_grad


def test_dataset_histories_never_cross_episode_or_observation_boundaries():
    dataset = LeRobotV3ImageDataset.__new__(LeRobotV3ImageDataset)
    dataset.phase_predictor_config = {"history_steps": 16, "image_offsets": [-12, -8, -4, 0]}
    dataset.episode_ends = np.array([20, 40])
    dataset.episode_ranges = [range(20), range(20, 40)]
    state, images = dataset._phase_history_indices(np.array([19, 20, 21, 39]))
    np.testing.assert_array_equal(state[0], np.arange(4, 20))
    np.testing.assert_array_equal(images[0], [7, 11, 15, 19])
    np.testing.assert_array_equal(state[1], np.full(16, 20))
    np.testing.assert_array_equal(images[2], [20, 20, 20, 21])
    dataset.phase_state_data = np.arange(40 * 21, dtype=np.float32).reshape(40, 21)
    dataset.train_mask = np.array([True, False])
    np.testing.assert_array_equal(dataset.phase_training_states(), dataset.phase_state_data[:20])


def test_predicted_switch_overrides_offline_labels_and_inference_needs_no_labels():
    teacher, obs = teacher_and_obs()
    teacher.eval().requires_grad_(False)
    model = renderer().eval()
    obs["phase_inputs"] = inputs()
    with torch.no_grad():
        model.phase_predictor.head[-1].weight.zero_()
        model.phase_predictor.head[-1].bias.fill_(10)
        model.output.bias.fill_(0.25)
    tokens, masks, contact, condition = student_context_tokens(
        teacher, model, obs, return_condition=True,
    )
    assert contact.all()  # Includes frames labeled free by the teacher.
    inference_obs = {key: value for key, value in obs.items()
                     if key not in {"contact_phase", "contact_alpha"}}
    preprocessor = WrenchPreprocessor.from_policy(teacher)
    images, rendered = render_observations(model, preprocessor, inference_obs, tuple(IMAGE_SHAPES), 2)
    expected = teacher.obs_encoder.encode_image_context(rendered, input_grad=True)
    torch.testing.assert_close(tokens, expected, rtol=0, atol=0)
    with torch.no_grad():
        model.phase_predictor.head[-1].bias.fill_(-10)
    _, closed = render_observations(model, preprocessor, inference_obs, tuple(IMAGE_SHAPES), 2)
    for key in images:
        torch.testing.assert_close(closed[key], images[key], rtol=0, atol=0)


@pytest.mark.parametrize("stage", ["intensity", "rgb"])
def test_actual_workspace_loss_trains_phase_in_both_stages_with_separate_rgb_gradient(stage):
    teacher, obs = teacher_and_obs()
    teacher.eval().requires_grad_(False)
    model = renderer()
    model.set_training_stage(stage)
    model.set_stage_mode(True)
    obs["phase_inputs"] = inputs()
    with torch.no_grad():
        model.phase_predictor.head[-1].bias.fill_(5)
        model.output.bias.fill_(0.2)
    workspace = TrainDistillRenderNetWorkspace.__new__(TrainDistillRenderNetWorkspace)
    workspace.cfg = OmegaConf.create({"distillation": {"objective": "token_mse"}})
    workspace.model, workspace.teacher_policy = model, teacher
    workspace.device = torch.device("cpu")
    workspace.training_stage, workspace.render_stages_enabled = stage, True
    workspace.attention_enabled = True
    workspace.region_source = "context"
    workspace.attention_config = {"selection": "soft", "normalization": "image_minmax", "pixel_loss_weight": 0.0}
    workspace.context_patch_loss_weight = workspace.intensity_loss_weight = 1.0
    workspace.smoothness_loss_weight = 0.02
    workspace.shape_info = {}
    workspace._autocast = contextlib.nullcontext
    loss = workspace._loss(obs)
    expected_phase = F.binary_cross_entropy_with_logits(
        model.phase_predictor(obs["phase_inputs"])["phase_logits"], (obs["contact_phase"] > 0).float(),
    )
    assert workspace.loss_metrics["phase_bce"] == pytest.approx(expected_phase.item())
    if stage == "rgb":
        assert loss.item() == pytest.approx(workspace.loss_metrics["token_mse"] + expected_phase.item())
        phase_gradient = torch.autograd.grad(expected_phase, tuple(model.phase_predictor.parameters()), retain_graph=True)
        loss.backward()
        for parameter, expected in zip(model.phase_predictor.parameters(), phase_gradient):
            torch.testing.assert_close(parameter.grad, expected)
        assert all(p.grad is None for p in model.gate_parameters())
        assert model.output.bias.grad.abs().sum() > 0
    else:
        assert loss.item() == pytest.approx(workspace.loss_metrics["intensity_loss"] + expected_phase.item())
        loss.backward()
        assert all(p.grad is None for p in model.rgb_parameters())
        assert any(p.grad is not None for p in model.gate_parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.phase_predictor.parameters())
    assert all(p.grad is None for p in teacher.parameters())


def test_stage_transition_keeps_phase_trainable_and_checkpoint_preserves_normalization(tmp_path):
    model = renderer()
    ema_model = copy.deepcopy(model)
    ema = EMAModel(ema_model)
    model.set_training_stage("intensity")
    assert all(p.requires_grad for p in model.phase_predictor.parameters())
    model.initialize_rgb_from_gate()
    model.set_training_stage("rgb")
    ema.sync_parameters(model)
    assert all(p.requires_grad for p in model.phase_predictor.parameters())
    assert all(not p.requires_grad for p in model.gate_parameters())
    sample = inputs()
    torch.save(ema_model.state_dict(), tmp_path / "phase.ckpt")
    restored = renderer().eval()
    restored.load_state_dict(torch.load(tmp_path / "phase.ckpt", weights_only=True), strict=True)
    torch.testing.assert_close(restored.phase_predictor(sample)["p_open"], ema_model.phase_predictor(sample)["p_open"])


def test_validation_phase_rates_use_counts_and_absent_phases_are_explicit():
    metrics = TrainDistillRenderNetWorkspace._phase_rates([2, 8, 3, 4, 0, 0])
    assert metrics["phase_free_false_open"] == 0.25
    assert metrics["phase_precontact_recall"] == 0.75
    assert metrics["phase_contact_recall_count"] == 0
