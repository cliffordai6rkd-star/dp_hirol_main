"""Context-effect targets and hard, patch-limited RenderNet rendering."""

import pytest
import torch
from omegaconf import OmegaConf

from test_render_net_distillation import make_attention_teacher
from diffusion_policy.model.vision.render_net import RenderNet, top_percentile_mask
from diffusion_policy.model.vision.render_net_attention import context_effect_regions, teacher_context_effect_targets
from diffusion_policy.scripts.visualize_wrench_effect_episode import make_zero_wrench_obs
from diffusion_policy.workspace.train_distill_render_net_workspace import TrainDistillRenderNetWorkspace


def test_context_targets_match_full_fp32_encoder_difference_and_do_not_call_dit():
    teacher, obs, _ = make_attention_teacher()
    original = {key: value.clone() for key, value in obs.items()}
    handle = teacher.model.register_forward_pre_hook(
        lambda *args: (_ for _ in ()).throw(AssertionError("Context targets must not sample DiT")))
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            normal, maps = teacher_context_effect_targets(teacher, obs)
    finally:
        handle.remove()
    with torch.no_grad():
        expected = teacher._encode_context(obs, apply_curriculum_mask=False)
        zero = teacher._encode_context(make_zero_wrench_obs(teacher, obs), apply_curriculum_mask=False)
    torch.testing.assert_close(normal, expected, rtol=0, atol=0)
    delta = (expected - zero).reshape(2, 2, 10, 16).norm(dim=-1)
    torch.testing.assert_close(maps["side"], delta[..., :4], rtol=0, atol=0)
    torch.testing.assert_close(maps["wrist"], delta[..., 4:], rtol=0, atol=0)
    assert normal.dtype == torch.float32 and not normal.requires_grad
    for key in obs:
        torch.testing.assert_close(obs[key], original[key], rtol=0, atol=0)


def test_binary_top90_has_only_twenty_of_192_patches_and_empty_zero_response():
    values = torch.stack([torch.arange(192).float(), torch.zeros(192), torch.ones(192)])
    mask = top_percentile_mask(values, 90)
    assert mask.sum(-1).tolist() == [20, 0, 20]
    assert mask[0, :172].count_nonzero() == 0 and mask[0, 172:].all()
    teacher, obs, _ = make_attention_teacher()
    maps = {"side": torch.tensor([[[0., 1., 2., 3.]]]).expand(2, 2, -1),
            "wrist": torch.zeros(2, 2, 6)}
    masks = context_effect_regions(maps, obs, teacher.obs_encoder.backbone, percentile=90)
    assert set(masks["side"].unique().tolist()) == {0., 1.}
    assert (masks["side"].sum((-1, -2, -3)) == 64).all()
    assert masks["wrist"].count_nonzero() == 0


@pytest.mark.parametrize("rendering_mode", ["gaussian_glow", "residual"])
def test_binary_student_gate_has_no_outside_rendering_and_keeps_region_gradients(rendering_mode):
    torch.manual_seed(5)
    model = RenderNet(
        6, base_channels=4, conditioning_dim=8, wrench_history_steps=3,
        rendering_mode=rendering_mode,
        transformer={"embed_dim": 16, "depth": 1, "patch_size": 8, "num_heads": 4},
        high_attention_render={"enabled": True, "selection": "binary_top_percentile", "percentile": 90},
    )
    with torch.no_grad():
        model.mask_head.bias.fill_(2.)
        if rendering_mode == "residual":
            model.output.weight.normal_(std=.05)
    image = torch.full((1, 2, 3, 32, 40), .3)
    rendered, masks = model({"wrist": image}, torch.randn(1, 2, 3, 6), return_aux=True)
    difference = (rendered["wrist"] - image).abs().sum(2)
    affected = difference.reshape(1, 2, 4, 8, 5, 8).amax((3, 5)) > 0
    # 4x5 patches: at most two patches in each observation can change.
    assert (affected.sum((-1, -2)) <= 2).all() and affected.any()
    loss = rendered["wrist"].mean() + (masks["wrist"] - .2).square().mean()
    loss.backward()
    assert model.mask_head.weight.grad.abs().sum() > 0
    assert model.output.weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.mask_head.weight.zero_()
        model.mask_head.bias.fill_(-10.)
        empty = model({"wrist": image}, torch.randn(1, 2, 3, 6))["wrist"]
    torch.testing.assert_close(empty, image, rtol=0, atol=0)


@pytest.mark.parametrize("cached", [False, True])
def test_context_region_supervision_integrates_with_cached_and_online_training(cached):
    teacher, obs, _ = make_attention_teacher()
    config = {"enabled": True, "source": "context_effect", "selection": "binary_top_percentile", "percentile": 90}
    workspace = object.__new__(TrainDistillRenderNetWorkspace)
    workspace.teacher_policy = teacher
    workspace.device = torch.device("cpu")
    workspace.cfg = OmegaConf.create({})
    workspace.model = RenderNet(
        6, base_channels=4, conditioning_dim=8, wrench_history_steps=7,
        transformer={"embed_dim": 16, "depth": 1, "patch_size": 8, "num_heads": 4},
        rendering_mode="gaussian_glow", high_attention_render=config,
    ).train()
    workspace.attention_config = config
    workspace.attention_enabled = True
    workspace.mask_loss_weight = 1.
    workspace.amp_enabled = False
    workspace.shape_info = {}
    targets = None
    if cached:
        tokens, patches = workspace._region_targets(obs, return_patch_attention=True)
        targets = {"tokens": tokens, "patch_attention": patches}
    loss = workspace._loss(obs, cached_targets=targets)
    loss.backward()
    assert workspace.loss_metrics["mask_supervised"]
    assert workspace.loss_metrics["mask_mse"] > 0
    assert workspace.model.mask_head.weight.grad.abs().sum() > 0
