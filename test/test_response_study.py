"""Numerical contracts used by the independent response/reachability diagnostics."""
import torch
from diagnose_rgb_local_jacobian import conjugate_gradient
from run_shared_rgb_oracle import TinySharedGenerator


def test_matrix_free_ridge_cg_matches_small_exact_solution():
    torch.manual_seed(3)
    jacobian=torch.randn(11,7,dtype=torch.float64)
    target=torch.randn(11,dtype=torch.float64)
    ridge=.5
    operator=lambda v:jacobian.T@(jacobian@v)+ridge*v
    solution,history=conjugate_gradient(operator,jacobian.T@target,iterations=20,tolerance=1e-10)
    expected=torch.linalg.solve(jacobian.T@jacobian+ridge*torch.eye(7,dtype=torch.float64),jacobian.T@target)
    assert torch.allclose(solution,expected,atol=1e-8)
    assert history[-1]<history[0]*1e-8


def test_tiny_generator_is_shared_identity_initialized_and_learns_torque():
    torch.manual_seed(1)
    generator=TinySharedGenerator(8)
    images={k:torch.rand(2,2,3,32,32) for k in ['side','wrist']}
    torque=torch.randn(2,2,8,7)
    initial=generator(images,torque)
    assert all(torch.count_nonzero(v)==0 for v in initial.values())
    # After opening the last layer, both image and torque paths get gradients.
    torch.nn.init.normal_(generator.decoder[-1].weight,std=.02)
    output=generator(images,torque)
    sum(v.square().mean() for v in output.values()).backward()
    assert generator.encoder[0].weight.grad.norm()>0
    assert generator.torque[0].weight.grad.norm()>0
    assert all(v.shape==images[k].shape for k,v in output.items())
