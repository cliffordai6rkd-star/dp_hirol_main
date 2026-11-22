import pytest
import torch

from tools.fit_paired_torque_response import paired_losses


def test_paired_loss_distinguishes_shared_offset_from_wrong_force_response():
    target = torch.tensor([[[[1.]]], [[[0.]]], [[[-1.]]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    offset = (target + 2).requires_grad_()
    terms = paired_losses(offset, target, mask)
    assert terms["absolute"].item() == 4.
    assert terms["response"].item() == 0.
    insensitive = torch.zeros_like(target, requires_grad=True)
    terms = paired_losses(insensitive, target, mask)
    assert terms["absolute"].item() == pytest.approx(2 / 3)
    assert terms["zero_response"].item() == 1.
    assert terms["contact_response"].item() == 4.
    assert terms["response"].item() == 2.5
    (terms["absolute"] + terms["response"]).backward()
    assert insensitive.grad.abs().sum() > 0


def test_all_paired_terms_exclude_free_and_detach_teacher():
    target = torch.zeros(3, 1, 2, 1, requires_grad=True)
    prediction = torch.tensor([[[[1.], [100.]]], [[[0.], [200.]]], [[[-1.], [300.]]]], requires_grad=True)
    terms = paired_losses(prediction, target, torch.tensor([[True, False]]))
    assert terms["absolute"].item() == pytest.approx(2 / 3)
    assert terms["response"].item() == 2.5
    (terms["absolute"] + terms["response"]).backward()
    assert not prediction.grad[:, :, 1].any()
    assert target.grad is None
