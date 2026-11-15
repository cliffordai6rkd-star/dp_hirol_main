import copy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diffusion_policy.model.vision.render_net import build_render_net
from response_conditioned_rgb_common import ResponseConditionedDecoder
from response_decoder_ablation import DiagnosticResponseDecoder


def owner():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model=build_render_net(7,dict(base_channels=4,conditioning_dim=256,patch_size=8,
        embed_dim=16,num_heads=4,mlp_ratio=2,dropout=0,image_self_attn_layers=4,
        cross_attn_layers=2,decoder_layers=3,conv_block='bottleneck',wrench_layers=2,
        spatial_modulation=dict(enabled=True),dual_path=dict(enabled=True,decoder_film=True)))
    nn.init.normal_(model.output.weight,std=.01)
    for affine in model.backbone.decoder_film.affines:nn.init.normal_(affine[-1].weight,std=.01)
    return model


def inputs():
    return ({k:torch.rand(1,2,3,32,32) for k in ['side','wrist']},torch.randn(1,784,256),
        {k:torch.ones(1,2,1,32,32) for k in ['side','wrist']},torch.full((1,2),.5))


@pytest.mark.parametrize('amp',[False,True])
def test_A_is_exact_original_forward_and_trace_preserves_actual_values(amp):
    original=ResponseConditionedDecoder(owner(),.1).eval()
    diagnostic=DiagnosticResponseDecoder(owner(),.1).eval()
    diagnostic.load_state_dict(copy.deepcopy(original.state_dict()),strict=True)
    args=inputs()
    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):
        expected=original(*args);actual,stats=diagnostic(*args,trace=True)
    for key in expected:torch.testing.assert_close(expected[key],actual[key],rtol=0,atol=0)
    torch.testing.assert_close(stats['response_input_mse'],stats['response_raw_mse']*.25)
    for i in range(2):torch.testing.assert_close(stats[f'film{i}_after_alpha_mse'],stats[f'film{i}_after_gate_mse']*.25)


@pytest.mark.parametrize('projection,ca_response',[(True,True),(True,False),(False,True)])
def test_single_alpha_and_route_gradient_ownership(projection,ca_response):
    net=DiagnosticResponseDecoder(owner(),.1,input_alpha=False,film_alpha=False,
        projection=projection,ca_response=ca_response).eval()
    image,response,gate,alpha=inputs()
    rendered,stats=net(image,response,gate,alpha,trace=True)
    torch.testing.assert_close(stats['response_input_mse'],stats['response_raw_mse'],rtol=0,atol=0)
    assert torch.all(stats['alpha_input']==1) and torch.all(stats['alpha_film']==1)
    for i in range(2):torch.testing.assert_close(stats[f'film{i}_after_alpha_mse'],stats[f'film{i}_after_gate_mse'],rtol=0,atol=0)
    sum((rendered[k]-image[k]).square().mean() for k in image).backward()
    grad=net.response_projection.weight.grad
    assert bool(grad is not None and grad.norm()>0)==projection
    attention=net.owner.backbone.cross_attention[0].cross_attention
    assert any(p.grad is not None and p.grad.norm()>0 for p in attention.parameters())==ca_response
    assert net.owner.backbone.cross_attention[0].ffn[0].weight.grad.norm()>0
    assert all(p.grad is None and not p.requires_grad for p in net.owner.wrench_encoder.parameters())
    closed=net(image,response,gate,torch.zeros_like(alpha))
    assert all(torch.equal(closed[k],image[k]) for k in image)


def test_smooth_same_pixel_bound_preserves_gradient_past_old_hard_cap():
    for mode in ['hard','smooth']:
        net=DiagnosticResponseDecoder(owner(),.1,parameterization=mode).eval()
        net.owner.residual_scale=1.
        with torch.no_grad():
            net.owner.backbone.output.weight.zero_()
            net.owner.backbone.output.bias.fill_(.4)
        net.capture_pixels=True
        image,response,gate,alpha=inputs()
        image={k:torch.full_like(v,.5) for k,v in image.items()}
        rendered,stats=net(image,response,gate,alpha,trace=True)
        for k in image:
            assert (rendered[k]-image[k]).abs().max()<=.05+1e-7
        sum(v.sum() for v in rendered.values()).backward()
        assert bool(net.owner.backbone.output.bias.grad.abs().sum()>0)==(mode=='smooth')
        assert bool(stats['hard_clipped_channels'].sum()>0)==(mode=='hard')
        assert bool(stats['effective_channels'].sum()>0)==(mode=='smooth')
