"""Scientific loss contracts for the independent paired-response experiment."""
import torch
from run_paired_full_response import masked_response_weights


def test_teacher_weights_are_frozen_and_quiet_patches_are_excluded():
    teacher = torch.tensor([[[0.,0.],[1.,1.],[2.,2.]]], requires_grad=True)
    weights, count = masked_response_weights(teacher,.1)
    assert not weights.requires_grad and not count.requires_grad
    assert count == 2 and weights[0,0] == 0
    assert torch.allclose(weights.sum()/count,torch.tensor(1.))
    student = torch.ones_like(teacher,requires_grad=True)
    loss = (student.square().mean(-1)*weights).sum()/count
    loss.backward()
    assert torch.count_nonzero(student.grad[0,0]) == 0
    assert torch.count_nonzero(student.grad[0,1:]) > 0
    assert teacher.grad is None


def test_microbatch_preserves_effective_patch_normalization_and_gradients():
    torch.manual_seed(3)
    teacher = torch.randn(7,5,4)
    teacher[:2,:3] *= 1e-6
    student = torch.randn_like(teacher,requires_grad=True)
    weights,count = masked_response_weights(teacher,.2)
    errors = (student-teacher).square().mean(-1)
    full = (weights*errors).sum()/count
    grad = torch.autograd.grad(full,student,retain_graph=True)[0]
    split = sum((weights[i:i+3]*errors[i:i+3]).sum()/count for i in range(0,len(teacher),3))
    split_grad = torch.autograd.grad(split,student)[0]
    assert torch.allclose(full,split,atol=1e-7)
    assert torch.allclose(grad,split_grad,atol=1e-7)


def test_no_response_has_zero_weight_and_finite_normalization():
    weights,count = masked_response_weights(torch.zeros(2,3,4),.01)
    assert count == 0 and torch.isfinite(weights).all() and torch.count_nonzero(weights) == 0
