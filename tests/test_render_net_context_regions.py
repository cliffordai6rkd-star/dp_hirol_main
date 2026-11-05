"""Context-effect targets and hard, patch-limited RenderNet rendering."""

import pytest
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf

from test_render_net_distillation import make_attention_teacher
from diffusion_policy.model.vision.render_net import RenderNet, top_percentile_mask
from diffusion_policy.model.vision.render_net_attention import context_effect_regions, teacher_context_effect_targets
from diffusion_policy.scripts.visualize_wrench_effect_episode import make_zero_wrench_obs
from diffusion_policy.workspace.train_distill_render_net_workspace import (
    TrainDistillRenderNetWorkspace,
    normalize_region_source,
)


def test_region_source_names_have_canonical_config_and_legacy_aliases():
    assert normalize_region_source("noise_attention") == "noise_attention"
    assert normalize_region_source("context") == "context"
    assert normalize_region_source("attention") == "noise_attention"
    assert normalize_region_source("context_effect") == "context"


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


def test_context_effect_soft_regions_use_continuous_quantile_weights():
    teacher, obs, _ = make_attention_teacher()
    maps = {
        "side": torch.arange(16, dtype=torch.float32).reshape(1, 1, 16).expand(2, 2, -1),
        "wrist": torch.zeros(2, 2, 6),
    }
    masks = context_effect_regions(
        maps, obs, teacher.obs_encoder.backbone,
        selection="soft", percentiles=[25, 75],
    )
    assert masks["side"].shape == (2, 2, 1, *obs["side"].shape[-2:])
    assert float(masks["side"].min()) == 0.0
    assert float(masks["side"].max()) == 1.0
    assert ((masks["side"] > 0) & (masks["side"] < 1)).any()
    # A zero effect means no rendering strength anywhere.
    torch.testing.assert_close(masks["wrist"], torch.zeros_like(masks["wrist"]))


def test_context_effect_absolute_strength_keeps_zero_effect_zero_and_uses_bilinear_field():
    teacher, obs, _ = make_attention_teacher()
    maps = {
        "side": torch.tensor([[[0., 2., 8., 18.]]]).expand(2, 2, -1),
        "wrist": torch.zeros(2, 2, 6),
    }
    masks = context_effect_regions(
        maps, obs, teacher.obs_encoder.backbone,
        selection="soft", strength_scale=8.0,
    )
    assert masks["side"].shape == (2, 2, 1, *obs["side"].shape[-2:])
    assert float(masks["side"].min()) == 0.0
    assert float(masks["side"].max()) < 1.0
    assert ((masks["side"] > 0) & (masks["side"] < 1)).any()
    torch.testing.assert_close(masks["wrist"], torch.zeros_like(masks["wrist"]))


def test_image_minmax_is_independent_per_camera_sample_and_observation_then_bilinear():
    teacher, obs, _ = make_attention_teacher()
    normalized = {
        "side": torch.tensor([[[0., .25, .5, 1.], [.5, 0., .25, 1.]],
                              [[0., .5, .75, 1.], [.5, 1., .75, 0.]]]),
        "wrist": torch.tensor([[[1., 0., .5, .25, .75, .125], [0., 1., .25, .75, .5, .125]],
                               [[.25, .5, 1., 0., .125, .75], [.75, .125, 0., 1., .5, .25]]]),
    }
    scales = torch.tensor([[2., 20.], [200., 2000.]])[..., None]
    offsets = torch.tensor([[100., 500.], [3., 7000.]])[..., None]
    maps = {"side": normalized["side"] * scales + offsets,
            "wrist": normalized["wrist"] * scales.flip(0) + offsets.flip(1)}
    masks = context_effect_regions(
        maps, obs, teacher.obs_encoder.backbone,
        selection="soft", normalization="image_minmax", strength_scale=5.,
    )
    # Changing absolute amplitude/background independently per image must not
    # change the region shape or let a different camera determine its scale.
    shifted = {key: values * 2 + 17 for key, values in maps.items()}
    shifted_masks = context_effect_regions(
        shifted, obs, teacher.obs_encoder.backbone,
        selection="soft", normalization="image_minmax",
    )
    for key, grid in {"side": (2, 2), "wrist": (2, 3)}.items():
        expected = F.interpolate(
            normalized[key].reshape(4, 1, *grid), size=obs[key].shape[-2:],
            mode="bilinear", align_corners=False,
        ).reshape(2, 2, 1, *obs[key].shape[-2:])
        torch.testing.assert_close(masks[key], expected, rtol=0, atol=0)
        torch.testing.assert_close(shifted_masks[key], expected, rtol=0, atol=0)
        assert ((masks[key] > 0) & (masks[key] < 1)).any()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_image_minmax_flat_images_have_zero_strength_without_nan(dtype):
    teacher, obs, _ = make_attention_teacher()
    constants = torch.tensor([[0., 5.], [17., 0.]], dtype=dtype)[..., None]
    maps = {"side": constants.expand(2, 2, 4),
            "wrist": constants.flip(0).expand(2, 2, 6)}
    masks = context_effect_regions(
        maps, obs, teacher.obs_encoder.backbone,
        selection="soft", normalization="image_minmax",
    )
    for value in masks.values():
        assert value.dtype == torch.float32 and torch.isfinite(value).all()
        assert value.count_nonzero() == 0


@pytest.mark.parametrize("rendering_mode", ["gaussian_glow", "residual"])
def test_binary_student_gate_has_no_outside_rendering_and_keeps_region_gradients(rendering_mode):
    torch.manual_seed(5)
    model = RenderNet(
        6, base_channels=4, conditioning_dim=8, wrench_history_steps=3,
        rendering_mode=rendering_mode,
        embed_dim=16, image_self_attn_layers=1, cross_attn_layers=1, patch_size=8, num_heads=4,
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


def test_patch_residual_outputs_pixel_blocks_and_zero_strength_is_identity():
    torch.manual_seed(9)
    model = RenderNet(
        6, base_channels=4, conditioning_dim=8, wrench_history_steps=3,
        rendering_mode="patch_residual", patch_residual_refine=True,
        embed_dim=16, image_self_attn_layers=1, cross_attn_layers=1, patch_size=8, num_heads=4,
        high_attention_render={"enabled": True, "activation": "relu"},
    ).eval()
    image = torch.full((1, 2, 3, 32, 40), .3)
    wrench = torch.randn(1, 2, 3, 6)
    with torch.no_grad():
        model.mask_head.weight.zero_()
        model.mask_head.bias.fill_(-1.)
        rendered, masks = model({"wrist": image}, wrench, return_aux=True)
    assert rendered["wrist"].shape == image.shape
    assert masks["wrist"].count_nonzero() == 0
    torch.testing.assert_close(rendered["wrist"], image, rtol=0, atol=0)
    assert model.output.out_features == 3 * 8 * 8
    with torch.no_grad():
        model.output.weight.normal_(std=.05)
        model.mask_head.bias.fill_(1.)
    rendered = model({"wrist": image}, wrench)["wrist"]
    assert (rendered - image).abs().sum() > 0


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
        embed_dim=16, image_self_attn_layers=1, cross_attn_layers=1, patch_size=8, num_heads=4,
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


@pytest.mark.parametrize("cached", [False, True])
def test_context_region_image_minmax_supervision_integrates_with_cached_and_online_training(cached):
    teacher, obs, _ = make_attention_teacher()
    config = {
        "enabled": True, "source": "context", "selection": "soft",
        "percentiles": [80, 99], "baseline": "raw_zero", "normalization": "image_minmax",
    }
    workspace = object.__new__(TrainDistillRenderNetWorkspace)
    workspace.teacher_policy = teacher
    workspace.device = torch.device("cpu")
    workspace.cfg = OmegaConf.create({})
    workspace.model = RenderNet(
        6, base_channels=4, conditioning_dim=8, wrench_history_steps=7,
        embed_dim=16, image_self_attn_layers=1, cross_attn_layers=1, patch_size=8, num_heads=4,
        rendering_mode="gaussian_glow", high_attention_render=config,
    ).train()
    workspace.attention_config = config
    workspace.region_source = "context"
    workspace.attention_enabled = True
    workspace.mask_loss_weight = 1.
    workspace.amp_enabled = False
    workspace.shape_info = {}
    tokens, patches = workspace._region_targets(obs, return_patch_attention=True)
    expected_targets = context_effect_regions(
        patches, obs, teacher.obs_encoder.backbone,
        selection="soft", normalization="image_minmax",
    )
    _, online_targets = workspace._region_targets(obs)
    cached_targets = workspace._region_masks(patches, obs)
    for key in expected_targets:
        torch.testing.assert_close(online_targets[key], expected_targets[key], rtol=0, atol=0)
        torch.testing.assert_close(cached_targets[key], expected_targets[key], rtol=0, atol=0)
    targets = {"tokens": tokens, "patch_attention": patches} if cached else None
    loss = workspace._loss(obs, cached_targets=targets)
    loss.backward()
    assert workspace.loss_metrics["mask_supervised"]
    assert workspace.loss_metrics["mask_mse"] > 0
    assert workspace.model.mask_head.weight.grad.abs().sum() > 0
