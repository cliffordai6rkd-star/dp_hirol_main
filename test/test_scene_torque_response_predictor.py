import torch
from run_scene_torque_response_predictor import SceneTorquePredictor


def test_physical_zero_and_both_inputs_have_gradient_after_head_opens():
    torch.manual_seed(42)
    net = SceneTorquePredictor(torch.ones(256), torch.ones(7), hidden=8, rank=8)
    torch.nn.init.normal_(net.head[-1].weight, std=.1)
    local = torch.randn(2, 4, 256, requires_grad=True)
    pool = torch.randn(2, 2, 256)
    tau = torch.randn(2, 2, 8, 7, requires_grad=True)
    patches = torch.tensor([0, 195, 392, 783])
    assert net(local, pool, torch.zeros_like(tau), patches).abs().max() < 1e-7
    net(local, pool, tau, patches).square().mean().backward()
    assert local.grad.norm() > 0 and tau.grad.norm() > 0
    assert net.torque.weight_ih_l0.grad.norm() > 0
