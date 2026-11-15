"""Semantic regression tests for the formal direct-torque DiT path."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from diffusion_policy.distillation.dit_rgb import MixedPlan, diffusion_pair, element_mse, fixed_probe, state_sha
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.render_net import RenderNet, build_render_net
from diffusion_policy.dataset.render_net_teacher_cache import TeacherTargetCache


@pytest.fixture(autouse=True)
def small_threads():
    previous = torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny():
    return RenderNet(7,base_channels=4,conditioning_dim=8,wrench_layers=2,
        image_self_attn_layers=1,cross_attn_layers=1,decoder_layers=1,patch_size=2,embed_dim=8,
        num_heads=2,dropout=0.,wrench_history_steps=8,max_obs_steps=2,image_keys=['side','wrist'],
        rgb_epsilon=.1,residual_scale=1.,separate_gate_path=True,
        condition_prediction={'enabled':True},
        condition_backbone=dict(base_channels=4,patch_size=2,embed_dim=8,num_heads=2,
                               image_self_attn_layers=1,cross_attn_layers=1,decoder_layers=0),
        high_attention_render=dict(enabled=True,threshold=.2),spatial_modulation={'enabled':False})


def test_direct_tau_kv_and_two_updates_frozen_and_alpha0():
    torch.manual_seed(1)
    model = tiny();model.set_training_stage('rgb');model.set_stage_mode(True)
    images = {k:torch.rand(1,2,3,8,8) for k in ['side','wrist']}
    torque = torch.randn(1,2,8,7)
    masks = {k:torch.ones(1,2,1,8,8) for k in images}
    zero = (masks,{'contact_alpha':torch.zeros(1,2)})
    frozen = state_sha(model.condition_network.state_dict())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    kv = []
    handle = model.backbone.cross_attention[0].register_forward_pre_hook(lambda m,args: kv.append(args[1].shape))
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(images,torque,cached_condition=(masks,{'contact_alpha':torch.ones(1,2)}))
        loss = sum((v-.1).square().mean() for v in output.values());loss.backward()
        if step == 1:
            assert model.wrench_encoder.weight_ih_l0.grad.abs().sum()>0
            assert model.backbone.stem.body[0].weight.grad.abs().sum()>0
        assert model.output.weight.grad.abs().sum()>0
        optimizer.step()
    for k,v in model(images,torque,cached_condition=zero).items():
        assert torch.equal(v,images[k])
    assert all(shape[1]==8 for shape in kv) # eight LSTM steps, never196 response tokens
    assert state_sha(model.condition_network.state_dict()) == frozen
    assert all(p.grad is None for p in model.condition_network.parameters())
    with torch.no_grad():
        output = model(images,torque,cached_condition=(masks,{'contact_alpha':torch.ones(1,2)}))
    assert all((output[k]-images[k]).abs().max()<=.1000001 for k in images)
    handle.remove()


def test_probes_independent_of_order_batch_and_rank():
    action = torch.zeros(4,9,7)
    ids = [19,4,12,67]
    n,t = fixed_probe(action,ids,3,100)
    for subset in [[2,0],[3],[1]]:
        a,b = fixed_probe(action[subset],[ids[i] for i in subset],3,100)
        torch.testing.assert_close(a,n[subset],rtol=0,atol=0)
        torch.testing.assert_close(b,t[subset],rtol=0,atol=0)
    n2,t2 = fixed_probe(action,ids,4,100)
    assert not torch.equal(n,n2)


def test_same_diffusion_pair_and_mask():
    from diffusers import DDIMScheduler
    normalizer = LinearNormalizer();normalizer.fit({'action':torch.randn(30,7)})
    teacher = SimpleNamespace(normalizer=normalizer,horizon=9,action_dim=7,noise_scheduler=DDIMScheduler(num_train_timesteps=100))
    noisy,t,demo = diffusion_pair(teacher,torch.randn(2,9,7))
    assert noisy.shape==demo.shape==(2,9,7) and t.shape==(2,)
    prediction = torch.ones(2,9,7)
    target = torch.zeros_like(prediction);prediction[:,5:]=100
    mask = torch.arange(9)[None,:].expand(2,-1)<5
    assert torch.equal(element_mse(prediction,target,mask),torch.ones(2))
    with pytest.raises(ValueError):
        element_mse(prediction,target,torch.zeros_like(mask))


def test_mixed_plan_exact_resume_and_coverage():
    phases = np.repeat(np.arange(3),12);episodes = np.tile(np.repeat([0,1,2],4),3)
    plan = MixedPlan(phases,episodes)
    visited = set()
    for i in range(90):
        batch = plan.next();visited.update(batch)
        assert np.bincount(phases[batch],minlength=3).tolist()==[2,2,4]
    assert len(visited)==len(phases)
    restored = MixedPlan(phases,episodes);restored.load_state_dict(copy.deepcopy(plan.state_dict()))
    assert restored.next()==plan.next()


def test_static_cache_shuffled_alignment(tmp_path):
    cache = TeacherTargetCache(tmp_path,{'teacher':'hash','layout':'two_cameras'},3,token_dtype='float32')
    values = torch.arange(24.).reshape(3,4,2)
    cache.append(values,{'side':torch.ones(3,2,1,8,8),'alpha':torch.zeros(3,2)})
    for i in [2,0,1]:
        assert torch.equal(cache.read(i)['tokens'],values[i])
    with pytest.raises(ValueError):
        TeacherTargetCache(tmp_path,{'teacher':'changed'},3)


def test_formal_config_has_no_response_and_base_shape():
    cfg = OmegaConf.load('diffusion_policy/config/train_civi_student_dit_minimal.yaml')
    options = OmegaConf.to_container(cfg.render_net)
    model = build_render_net(7,options)
    assert len(model.backbone.image_self_attention)==6
    assert len(model.backbone.cross_attention)==2
    assert len(model.backbone.decoders)==3
    assert not model.spatial_modulation_enabled and not model.dual_path_enabled
    assert model.context_adapter is None and model.condition_prediction_enabled
    assert cfg.teacher.step==150000 and '180000' not in cfg.teacher.checkpoint
    assert not any('predictor' in n for n in model.state_dict())


def test_foreach_ema_matches_existing_mul_add_and_skips_frozen():
    from diffusion_policy.model.diffusion.ema_model import EMAModel
    from diffusion_policy.distillation.dit_rgb import foreach_ema_step
    online = torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Linear(8,2))
    online[0].requires_grad_(False)
    one,two = EMAModel(copy.deepcopy(online)),EMAModel(copy.deepcopy(online))
    for i in range(5):
        with torch.no_grad():online[1].weight.add_(.01)
        one.step(online);foreach_ema_step(two,online)
        for a,b in zip(one.averaged_model.parameters(),two.averaged_model.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert one.state_dict()==two.state_dict()


def test_fixed_visual_shape_is_independent_of_renderer_microbatch():
    import contextlib
    from diffusion_policy.workspace.train_civi_student_dit_workspace import TrainCiviStudentDiTWorkspace
    ws=object.__new__(TrainCiviStudentDiTWorkspace)
    ws.cfg=OmegaConf.create({'validation':{'encode_batch':2}})
    ws.adapter=torch.nn.Identity();ws.amp=lambda *args:contextlib.nullcontext()
    shapes=[]
    def visual(images, input_grad, context_adapter):
        shapes.append(images['side'].shape[0])
        return images['side'].flatten(1).unsqueeze(1)
    ws.visual=visual
    image=torch.randn(8,2,3,4,4,requires_grad=True)
    out=ws.encode({'side':image},True)
    assert shapes==[2,2,2,2]
    out.sum().backward()
    assert torch.equal(image.grad,torch.ones_like(image))


def test_independent_formal_budget_does_not_change_exploration(tmp_path):
    from diffusion_policy.distillation.budget import TrainingBudget
    history=tmp_path/'old';history.mkdir()
    (history/'state.json').write_text('{"exploration_rgb_updates":19712}')
    original=(history/'state.json').read_bytes()
    budget=TrainingBudget(tmp_path/'new','formal',2,tmp_path/'run','User:100epoch patience5 batch64')
    budget.charge(1,[1,2]);budget.charge(2,[3,4])
    with pytest.raises(RuntimeError):budget.charge(3,[5,6])
    assert (history/'state.json').read_bytes()==original
    with pytest.raises(ValueError):
        TrainingBudget(tmp_path/'new','formal',3,tmp_path/'run','User:100epoch patience5 batch64')


def test_formal100epoch_config_keeps_mixed_and_patience5():
    from hydra import compose,initialize_config_dir
    from pathlib import Path
    with initialize_config_dir(version_base=None,config_dir=str(Path('diffusion_policy/config').resolve())):
        cfg=compose(config_name='train_civi_student_dit_minimal_100epoch')
    assert cfg.training.global_batch==64 and list(cfg.data.phase_counts)==[16,16,32]
    assert cfg.training.num_epochs==100 and cfg.training.early_stopping.patience==5
    assert cfg.budget.maximum_new_updates==27800 and cfg.budget.scope=='formal'
    assert cfg.validation.encode_batch==2
