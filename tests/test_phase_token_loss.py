"""Token supervision follows offline phases, independently of rendering gates."""

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from diffusion_policy.model.vision.response_distillation import (
    masked_token_mse, phase_token_mask, response_weighted_token_mse,
)
from diffusion_policy.workspace import train_distill_render_net_workspace as distill
from test_render_net_distillation import make_attention_teacher


def test_precontact_contact_normalization_and_free_gradients():
    phases = torch.tensor([[0, 1, 2], [2, 0, 0]])
    shape = (2, 6, 2)
    mask = phase_token_mask(torch.empty(shape), phases)
    student = torch.where(mask[..., None], 2., 100.).expand(shape).clone().requires_grad_()
    teacher = torch.zeros(shape, requires_grad=True)
    loss, numerator, count = masked_token_mse(student, teacher, mask)
    assert loss.item() == 4.
    assert numerator.item() == 24. and count.item() == 6
    loss.backward()
    assert not student.grad[~mask].any()
    assert student.grad[mask].abs().sum() > 0
    assert teacher.grad is None


def test_all_free_returns_differentiable_zero():
    student = torch.randn(2, 8, 3, requires_grad=True)
    mask = phase_token_mask(student, torch.zeros(2, 2, dtype=torch.long))
    loss, _, count = masked_token_mse(student, torch.randn_like(student), mask)
    assert count.item() == 0 and loss.item() == 0.
    loss.backward()
    assert not student.grad.any()


def test_response_weighting_also_excludes_free_tokens():
    student = torch.tensor([[[9.], [1.], [2.]]], requires_grad=True)
    teacher = torch.zeros_like(student)
    mask = phase_token_mask(student, torch.tensor([[0, 1, 2]]))
    mse, weighted, numerator, weight = response_weighted_token_mse(
        student, teacher, torch.tensor([[1e9, 1., 3.]]), noise_floor=0., token_mask=mask,
    )
    assert mse.item() == 2.5
    assert weighted.item() == 3.25
    assert numerator.item() == 13. and weight.item() == 4.
    weighted.backward()
    assert student.grad[0, 0].item() == 0.


def workspace():
    result = object.__new__(distill.TrainDistillRenderNetWorkspace)
    result.cfg = OmegaConf.create({"training": {"render_stages": {"transition": "epoch"}}})
    result.model = nn.Identity()
    result.device = torch.device("cpu")
    result.max_val_batches = None
    result.global_step = result.epoch = 0
    result.training_stage = "rgb"
    return result


def test_validation_uses_selected_token_counts_across_mixed_and_free_batches():
    instance = workspace()
    values = iter([(4., 1), (1., 3), (0., 0)])

    def loss(obs):
        mean, count = next(values)
        instance.loss_metrics = {
            "token_mse": mean, "token_numerator": mean * count, "token_count": count,
        }
        return torch.tensor(mean)

    instance._loss = loss
    batches = [{"obs": {"value": torch.zeros(1, 1)}} for _ in range(3)]
    assert instance._validate(batches) == pytest.approx(7 / 4)
    assert instance.validation_metrics["token_count"] == 4
    assert instance.validation_metrics["token_mse"] == pytest.approx(7 / 4)


def test_rgb_workspace_loss_uses_labels_even_when_render_gate_is_closed(monkeypatch):
    instance = workspace()
    teacher, obs, _ = make_attention_teacher()
    instance.teacher_policy = teacher
    instance.attention_enabled = False
    instance.attention_config = {}
    instance.amp_enabled = False
    instance.shape_info = {}
    obs["contact_phase"] = torch.tensor([[0, 1], [2, 0]])
    target = torch.zeros(2, 20, 16)
    mask = phase_token_mask(target, obs["contact_phase"])
    student = torch.where(mask[..., None], 2., 100.).expand_as(target).clone().requires_grad_()
    # Predicted-closed gates must not remove true precontact/contact supervision.
    monkeypatch.setattr(distill, "student_context_tokens", lambda *args, **kwargs: (
        student, {}, torch.zeros(2, 2, 1, dtype=torch.bool),
    ))
    loss = instance._loss(obs, cached_targets={"tokens": target})
    assert loss.item() == 4.
    assert instance.loss_metrics["token_count"] == 20
    loss.backward()
    assert student.grad[mask].abs().sum() > 0
    assert not student.grad[~mask].any()
