"""The canonical student's three heads have separate optimization lifetimes."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from diffusion_policy.model.vision.causal_phase_gate import CausalPhaseGate
from diffusion_policy.model.vision.render_net import RenderNet


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def phase_inputs():
    return {"state": torch.randn(2, 2, 50, 21),
            "images": {key: torch.rand(2, 2, 4, 3, 32, 32)
                       for key in ("side", "wrist")}}


def renderer(num_classes=3, freeze_phase_after_convergence=False):
    model = RenderNet(
        7, base_channels=4, conditioning_dim=16, embed_dim=16, num_heads=4,
        image_self_attn_layers=1, cross_attn_layers=1, decoder_layers=4, patch_size=16,
        wrench_history_steps=50, separate_gate_path=True,
        high_attention_render={"enabled": True, "threshold": 0.0},
        phase_predictor={"enabled": True, "num_classes": num_classes, "history_steps": 50,
                         "hidden_dim": 8, "image_size": 32,
                         "freeze_after_convergence": freeze_phase_after_convergence},
    )
    model.phase_predictor.fit_normalizer(torch.randn(100, 21))
    return model


def test_three_class_predictions_keep_precontact_and_contact_distinct():
    model = CausalPhaseGate(("side", "wrist"), num_classes=3, history_steps=50,
                            hidden_dim=8, image_size=32)
    model.fit_normalizer(torch.randn(100, 21))
    sample = phase_inputs()
    for phase_id in range(3):
        with torch.no_grad():
            model.head[-1].weight.zero_()
            model.head[-1].bias.fill_(-8)
            model.head[-1].bias[phase_id] = 8
        result = model(sample)
        assert result["phase_logits"].shape == (2, 2, 3)
        assert (result["phase_prediction"] == phase_id).all()
        assert (result["contact_alpha"] == float(phase_id != 0)).all()
        assert not result["contact_alpha"].requires_grad
        torch.testing.assert_close(result["p_open"], 1 - result["phase_probabilities"][..., 0])


def test_phase_stage_does_not_execute_renderer_and_only_changes_phase_weights():
    model = renderer()
    model.set_training_stage("phase")
    model.train()
    assert model.phase_predictor.training
    assert not model.backbone.training and not model.gate_backbone.training
    before = copy.deepcopy(model.state_dict())
    handle = model.backbone.register_forward_pre_hook(
        lambda *_: pytest.fail("Phase training must not run RGB"))
    gate_handle = model.gate_backbone.register_forward_pre_hook(
        lambda *_: pytest.fail("Phase training must not run intensity"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    result = model(phase_only=True, phase_inputs=phase_inputs())
    F.cross_entropy(result["phase_logits"].flatten(0, 1), torch.tensor([0, 1, 2, 1])).backward()
    optimizer.step()
    handle.remove()
    gate_handle.remove()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.phase_predictor.parameters())
    assert all(p.grad is None for p in model.gate_parameters() + model.rgb_parameters())
    for name, value in model.state_dict().items():
        if not name.startswith("phase_predictor."):
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_intensity_then_rgb_keep_completed_heads_frozen_and_deterministic():
    model = renderer()
    sample = phase_inputs()
    images = {key: torch.rand(2, 2, 3, 32, 32) for key in model.image_keys}
    wrench = torch.randn(2, 2, 50, 7)
    with torch.no_grad():
        model.phase_predictor.head[-1].weight.zero_()
        model.phase_predictor.head[-1].bias.copy_(torch.tensor([-8., -8., 8.]))
    model.set_training_stage("intensity")
    model.train()
    assert not model.phase_predictor.training
    assert not model.backbone.training and model.gate_backbone.training
    frozen_phase = copy.deepcopy(model.phase_predictor.state_dict())
    rgb_handle = model.backbone.register_forward_pre_hook(
        lambda *_: pytest.fail("Intensity training must not run RGB"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    rendered, masks, prediction = model(images, wrench, intensity_only=True, phase_inputs=sample)
    assert rendered is None and not prediction["phase_logits"].requires_grad
    sum(F.mse_loss(mask, torch.full_like(mask, 0.8)) for mask in masks.values()).backward()
    optimizer.step()
    rgb_handle.remove()
    assert all(p.grad is None for p in model.phase_predictor.parameters())
    assert all(p.grad is None for p in model.rgb_parameters())
    for name, value in model.phase_predictor.state_dict().items():
        torch.testing.assert_close(value, frozen_phase[name], rtol=0, atol=0)

    model.zero_grad(set_to_none=True)
    model.initialize_rgb_from_gate()
    model.set_training_stage("rgb")
    model.train()
    assert not model.phase_predictor.training and not model.gate_backbone.training
    assert model.backbone.training
    frozen_gate = [parameter.detach().clone() for parameter in model.gate_parameters()]
    rendered, _, condition = model(images, wrench, phase_inputs=sample, return_condition=True,
                                   future_contact=torch.zeros(2, 2, 50, dtype=torch.bool))
    sum(image.mean() for image in rendered.values()).backward()
    optimizer.step()
    assert condition["contact_alpha"].all()
    assert all(p.grad is None for p in list(model.phase_predictor.parameters()) + model.gate_parameters())
    assert model.output.bias.grad.abs().sum() > 0
    for actual, expected in zip(model.gate_parameters(), frozen_gate):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, value in model.phase_predictor.state_dict().items():
        torch.testing.assert_close(value, frozen_phase[name], rtol=0, atol=0)


def test_binary_checkpoint_layout_is_unchanged():
    model = CausalPhaseGate(("side", "wrist"), hidden_dim=8)
    assert model.num_classes == 2 and model.head[-1].out_features == 1
    model.fit_normalizer(torch.randn(20, 21))
    restored = CausalPhaseGate(("side", "wrist"), hidden_dim=8)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.normalizer_fitted


def test_zero_padded_states_stay_zero_after_nonzero_mean_normalization():
    model = CausalPhaseGate(("side", "wrist"), num_classes=3, history_steps=50,
                            hidden_dim=8, image_size=32, pad_mode="zero")
    model.fit_normalizer(torch.randn(100, 21) + 3)
    sample = phase_inputs()
    sample["state"][:, :, :10] = 0
    normalized = []
    handle = model.state_projection.register_forward_pre_hook(
        lambda module, arguments: normalized.append(arguments[0].detach().clone()))
    model(sample)
    handle.remove()
    projected_input = normalized[0].reshape(2, 2, 50, 21)
    assert torch.count_nonzero(projected_input[:, :, :10]) == 0
    torch.testing.assert_close(projected_input[:, :, 10:],
                               (sample["state"][:, :, 10:] - model.state_mean) / model.state_std)


@pytest.mark.parametrize("num_classes", [2, 3])
def test_thresholded_zero_torque_is_preserved_without_erasing_measured_joint_states(num_classes):
    model = CausalPhaseGate(("side", "wrist"), num_classes=num_classes, history_steps=50,
                            hidden_dim=8, image_size=32, pad_mode="zero")
    model.fit_normalizer(torch.randn(100, 21) + 3)
    sample = phase_inputs()
    sample["state"][:, :, :10] = 0  # Missing history.
    sample["state"][:, :, 10:20, 14:] = 0  # Measured state with sub-threshold torque.
    normalized = []
    handle = model.state_projection.register_forward_pre_hook(
        lambda module, arguments: normalized.append(arguments[0].detach().clone()))
    model(sample)
    handle.remove()
    projected_input = normalized[0].reshape(2, 2, 50, 21)
    expected = (sample["state"] - model.state_mean) / model.state_std
    assert torch.count_nonzero(projected_input[:, :, :10]) == 0
    torch.testing.assert_close(projected_input[:, :, 10:, :14], expected[:, :, 10:, :14])
    torch.testing.assert_close(projected_input[:, :, 20:, 14:], expected[:, :, 20:, 14:])
    if num_classes == 3:
        assert torch.count_nonzero(projected_input[:, :, 10:20, 14:]) == 0
    else:
        torch.testing.assert_close(projected_input[:, :, 10:20, 14:], expected[:, :, 10:20, 14:])


def test_binary_phase_gate_preserves_thresholded_tau_ext_zeros():
    model = CausalPhaseGate(
        ("side", "wrist"), num_classes=2, history_steps=50, hidden_dim=8,
        image_size=32, pad_mode="zero",
        state_source_keys=("observation.joint", "observation.velocity", "observation.tau_ext"),
    )
    model.fit_normalizer(torch.randn(100, 21) + 3)
    sample = phase_inputs()
    sample["state"][..., 14:] = 0
    normalized = []
    handle = model.state_projection.register_forward_pre_hook(
        lambda module, arguments: normalized.append(arguments[0].detach().clone()))
    model(sample)
    handle.remove()
    projected_input = normalized[0].reshape(2, 2, 50, 21)
    assert torch.count_nonzero(projected_input[..., 14:]) == 0
    torch.testing.assert_close(projected_input[..., :14],
                               (sample["state"][..., :14] - model.state_mean[:14])
                               / model.state_std[:14])


def test_binary_phase_head_freezes_after_convergence():
    model = renderer(num_classes=2, freeze_phase_after_convergence=True)
    model.set_training_stage("phase")
    assert all(parameter.requires_grad for parameter in model.phase_predictor.parameters())
    model.set_training_stage("intensity")
    assert all(not parameter.requires_grad for parameter in model.phase_predictor.parameters())
    model.set_training_stage("rgb")
    assert all(not parameter.requires_grad for parameter in model.phase_predictor.parameters())


def test_binary_phase_validation_uses_bce_labels_free_and_contact():
    from diffusion_policy.workspace.train_distill_render_net_workspace import (
        TrainDistillRenderNetWorkspace,
    )

    workspace = object.__new__(TrainDistillRenderNetWorkspace)
    workspace.model = SimpleNamespace(phase_predictor=SimpleNamespace(num_classes=2))
    workspace.teacher_policy = SimpleNamespace(n_obs_steps=2)
    labels = torch.tensor([[0, 2]])
    logits = torch.tensor([[8.0, 8.0]])
    open_prediction = torch.tensor([[0.0, 1.0]])
    loss, metrics = workspace._phase_loss_metrics(
        {"contact_phase": labels},
        {"phase_logits": logits, "contact_alpha": open_prediction},
    )
    torch.testing.assert_close(loss, torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([[0.0, 1.0]]),
    ))
    assert metrics["phase_accuracy"] == 1.0
    assert metrics["phase_free_false_open_count"] == 1
    assert metrics["phase_contact_recall_count"] == 1
    assert metrics["phase_precontact_recall_count"] == 0
