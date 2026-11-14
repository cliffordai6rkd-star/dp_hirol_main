"""Requested structure, frozen branches, exact gates and state compatibility."""
from pathlib import Path
import copy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "test")]
import pytest
import torch
from torch import nn
from diffusion_policy.model.vision.render_net import build_render_net
from diffusion_policy.model.vision.dual_path_render_net import load_common_initialization
from test_response_weighted_distillation import renderer_config


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(dual=False, film=False, kind="linear"):
    c = renderer_config()
    c.update(image_self_attn_layers=4 if dual else 6, cross_attn_layers=2,
             separate_gate_path=True, conv_block="bottleneck", wrench_layers=2,
             context_adapter=dict(enabled=True, kind=kind, hidden_dim=8))
    if dual:
        c["dual_path"] = dict(enabled=True, decoder_film=film)
    return c


def test_old_layout_and_opt_in_false_strictly_compatible():
    original = build_render_net(6, config())
    c = config(); c["dual_path"] = {"enabled": False}
    restored = build_render_net(6, c)
    restored.load_state_dict(original.state_dict(), strict=True)
    assert not any("film" in key for key in original.state_dict())


def test_execution_order_decoder_sizes_and_zero_film_equivalence():
    old = build_render_net(6, config())
    b = build_render_net(6, config(True)).eval()
    c = build_render_net(6, config(True, True)).eval()
    for m in (b, c): load_common_initialization(m, old.state_dict())
    order = []
    handles = [block.register_forward_hook(lambda m,a,o,s=f"sa{i}":order.append(s))
               for i,block in enumerate(c.backbone.image_self_attention)]
    handles += [block.register_forward_hook(lambda m,a,o,s=f"ca{i}":order.append(s))
                for i,block in enumerate(c.backbone.cross_attention)]
    x, tau, camera = torch.rand(1,3,224,224), torch.randn(1,7,16), torch.randn(1,1,16)
    kw=dict(return_decoder_features=True, modulation_gate=torch.ones(1,1,224,224),
            contact_alpha=torch.ones(1,1,1,1))
    out = c.backbone(x,tau,camera,**kw)
    for h in handles: h.remove()
    assert order == ["sa0","sa1","ca0","sa2","sa3","ca1"]
    assert [v.shape[-2:] for v in out[2]] == [(56,56),(112,112),(224,224)]
    assert len(c.backbone.decoder_film.affines) == 2
    torch.testing.assert_close(out[0],b.backbone(x,tau,camera,**kw)[0],rtol=0,atol=0)
    # The production random output head, rather than a zero RGB head, is also shared.
    torch.nn.init.normal_(old.output.weight,std=.02)
    for m in (b,c): load_common_initialization(m,old.state_dict())
    torch.testing.assert_close(c.backbone(x,tau,camera,**kw)[0],
                               b.backbone(x,tau,camera,**kw)[0],rtol=0,atol=0)


class ClosedCondition(nn.Module):
    def forward(self,images,wrench):
        return {k:torch.ones_like(v[:,:,:1]) for k,v in images.items()}, {
            "contact_alpha":wrench.new_zeros(wrench.shape[:2])}


@pytest.mark.parametrize("film", [False, True])
def test_alpha_zero_exact_identity_after_nonzero_updates(film):
    m=build_render_net(6,config(True,film)).eval()
    torch.nn.init.normal_(m.output.weight,std=.1)
    if film:
        for affine in m.backbone.decoder_film.affines:
            nn.init.normal_(affine[-1].weight,std=.1)
    m.condition_network=ClosedCondition()
    images={k:torch.rand(2,2,3,32,32) for k in m.image_keys}
    wrench=torch.randn(2,2,7,6)
    for k,v in m(images,wrench).items(): assert torch.equal(v,images[k])


@pytest.mark.parametrize("kind", ["linear","mlp"])
def test_strict_round_trip_and_rgb_stage_gradient_ownership(kind):
    c=config(True,True,kind); m=build_render_net(6,c)
    m.set_training_stage("rgb"); m.set_stage_mode(True)
    torch.nn.init.normal_(m.output.weight,std=.03)
    images={k:torch.rand(2,2,3,32,32) for k in m.image_keys}
    wrench=torch.randn(2,2,7,6)
    optim=torch.optim.AdamW(m.rgb_parameters(),lr=.01)
    for step in range(2):
        optim.zero_grad(set_to_none=True)
        result=m(images,wrench)
        sum((v*torch.randn_like(v)).mean() for v in result.values()).backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0
                   for p in m.backbone.cross_attention[0].parameters())
        assert m.wrench_encoder.weight_ih_l0.grad.abs().sum()>0
        assert all(a[-1].weight.grad.abs().sum()>0 for a in m.backbone.decoder_film.affines)
        if step: assert m.backbone.decoder_film.pool_score.weight.grad.abs().sum()>0
        assert all(p.grad is None for p in m.context_adapter.parameters())
        assert all(p.grad is None for p in m.condition_network.parameters())
        optim.step()
    m.eval().requires_grad_(False); restored=build_render_net(6,c).eval().requires_grad_(False)
    restored.load_state_dict(copy.deepcopy(m.state_dict()),strict=True)
    for k,v in m(images,wrench).items():
        torch.testing.assert_close(v,restored(images,wrench)[k],rtol=0,atol=0)


def test_transfer_rejects_unrelated_missing_weights():
    old=build_render_net(6,config()).state_dict(); del old["backbone.stem.body.0.weight"]
    with pytest.raises(ValueError,match="Unexpected new"):
        load_common_initialization(build_render_net(6,config(True,True)),old)


def test_decoder_film_gate_and_alpha_close_only_its_residual():
    from diffusion_policy.model.vision.dual_path_render_net import TorqueDecoderFiLM
    module=TorqueDecoderFiLM(16,[8,4])
    nn.init.normal_(module.affines[0][-1].weight,std=.1)
    feature=torch.randn(2,8,16,16);pooled=module.pool(torch.randn(2,7,16))
    gate=torch.ones(2,1,32,32);alpha=torch.ones(2,1,1,1)
    torch.testing.assert_close(module(feature,pooled,0,gate*0,alpha),feature,rtol=0,atol=0)
    torch.testing.assert_close(module(feature,pooled,0,gate,alpha*0),feature,rtol=0,atol=0)
    assert not torch.equal(module(feature,pooled,0,gate,alpha),feature)


def test_independent_configs_preserve_loss_teacher_and_region_settings():
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    for suffix,sa,film,kind in [('a_baseline',6,False,'linear'),('b_interleaved',4,False,'linear'),
                                ('c_film_linear',4,True,'linear'),('c_film_mlp',4,True,'mlp')]:
        with initialize_config_dir(version_base=None,config_dir=str(ROOT/'diffusion_policy/config')):
            cfg=compose(config_name='student_dual_path_'+suffix)
        assert cfg.render_net.backbone.image_self_attn_layers==sa
        assert cfg.render_net.dual_path.decoder_film==film
        assert cfg.render_net.context_adapter.kind==kind
        assert cfg.render_net.high_attention_render.threshold==.2
        assert cfg.distillation.response_loss_weight==1 and not cfg.distillation.use_teacher_ffn
        assert 'optimizer_step=00180000' in cfg.teacher_checkpoint
        assert cfg.ablation.optimizer_steps==5000 and cfg.ablation.calibration_steps==2000
