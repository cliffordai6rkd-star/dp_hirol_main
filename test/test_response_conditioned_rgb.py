import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion_policy.model.vision.render_net import build_render_net
from response_conditioned_rgb_common import ResponseConditionedDecoder


def test_response_projection_gradient_and_exact_closed_alpha_without_torque_path():
    torch.set_num_threads(1)
    owner = build_render_net(7, dict(base_channels=4, conditioning_dim=256,
        patch_size=8, embed_dim=16, num_heads=4, mlp_ratio=2, dropout=0,
        image_self_attn_layers=4, cross_attn_layers=2, decoder_layers=3,
        conv_block='bottleneck', wrench_layers=2, spatial_modulation=dict(enabled=True),
        dual_path=dict(enabled=True, decoder_film=True)))
    nn.init.normal_(owner.output.weight, std=.01)
    net = ResponseConditionedDecoder(owner, .1).eval()
    images = {key: torch.rand(1,2,3,32,32) for key in owner.image_keys}
    gates = {key: torch.ones(1,2,1,32,32) for key in images}
    response = torch.randn(1,784,256)
    closed = net(images,response,gates,torch.zeros(1,2))
    assert all(torch.equal(closed[k],images[k]) for k in images)
    rendered = net(images,response,gates,torch.ones(1,2))
    sum((rendered[k]-images[k]).square().mean() for k in images).backward()
    assert net.response_projection.weight.grad.norm() > 0
    assert owner.backbone.cross_attention[0].cross_attention.q_proj_weight.grad.norm() > 0
    assert owner.backbone.decoder_film.affines[0][-1].weight.grad.norm() > 0
    assert all(p.grad is None and not p.requires_grad for p in owner.wrench_encoder.parameters())
    assert all((rendered[k]-images[k]).abs().max() <= .100001 for k in images)
