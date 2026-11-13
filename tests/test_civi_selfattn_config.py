from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from diffusion_policy.model.vision.contact_curriculum import FactrBlurScheduler
from diffusion_policy.model.vision.force_aware_obs_encoder import ForceAwareObsEncoder
from diffusion_policy.workspace.train_force_aware_diffusion_workspace import (
    _action_mse_protocol,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "diffusion_policy" / "config"


def compose_teacher(name="train_civi_teacher_selfattn", overrides=()):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=name, overrides=list(overrides))


@pytest.mark.parametrize("head", ["act", "dit"])
def test_selfattn_teacher_preserves_observation_and_action_alignment(head):
    original = compose_teacher("train_civi_teacher")
    cfg = compose_teacher(overrides=[f"action_head_backbone={head}"])

    assert cfg.action_head_backbone == cfg.policy.action_head_backbone == head
    assert cfg.policy.pre_cross_self_attn_layers == 2
    assert cfg.policy.fusion_downsample_factor == 2
    assert cfg.policy.fusion_output_mode == "visual"
    assert cfg.policy.fusion_upsample is False
    assert cfg.policy.fusion_highres_skip is False
    assert cfg.policy.cross_attn_layers == 0
    assert cfg.policy.n_cond_layers == 0
    assert cfg.policy.n_layer == 6
    assert cfg.policy.n_emb == original.policy.n_emb
    assert cfg.policy.n_head == original.policy.n_head
    assert cfg.policy.force_temporal_encoder == original.policy.force_temporal_encoder == "lstm"
    assert cfg.policy.force_encoder_layers == original.policy.force_encoder_layers == 2
    assert cfg.policy.freeze_dino_backbone is True
    assert cfg.policy.input_gate_enabled is False
    assert cfg.shape_meta == original.shape_meta
    assert cfg.n_obs_steps == original.n_obs_steps == 2
    assert cfg.horizon == original.horizon == 9
    assert cfg.n_action_steps == original.n_action_steps == 8
    assert cfg.task.dataset.pad_before == original.task.dataset.pad_before == 1
    assert cfg.task.dataset.windowed_lowdim_feature_groups.tau_ext.steps == 8
    assert cfg.policy.noise_scheduler == original.policy.noise_scheduler
    assert cfg.policy.image_blur_scope == original.policy.image_blur_scope
    assert cfg.policy.image_blur_schedule == original.policy.image_blur_schedule
    assert cfg.training.resume is False
    assert f"selfattn_{head}/teacher" in cfg.output_dir


def test_board_selfattn_config_keeps_dataset_and_selects_act_by_default():
    cfg = compose_teacher("train_civi_teacher_board_selfattn")

    assert cfg.policy.action_head_backbone == "act"
    assert cfg.policy.act_loss_type == "l1"
    assert cfg.policy.fusion_downsample_factor == 2
    assert cfg.policy.fusion_output_mode == "visual"
    assert cfg.policy.fusion_upsample is False
    assert cfg.policy.fusion_highres_skip is False
    assert cfg.dataset_path == "data/board_25hz_3cam_lerobotv3_clean_ma5_l1_12p5"
    assert cfg.task.dataset.dataset_path == cfg.dataset_path
    assert cfg.dataloader.batch_size == cfg.val_dataloader.batch_size == 64
    assert cfg.training.distributed.enabled is False
    assert "board_selfattn_act/teacher" in cfg.output_dir


@pytest.mark.parametrize("name", ["train_civi_teacher_selfattn", "train_civi_teacher_board_selfattn"])
def test_selfattn_outputs_use_model_output_and_hydra_follows_override(name):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=name, return_hydra_config=True)
        overridden = compose(
            config_name=name,
            overrides=["output_dir=/tmp/custom-teacher"],
            return_hydra_config=True,
        )

    assert Path(cfg.output_dir).parts[0] == "model_output"
    assert Path(cfg.output_dir).name == "teacher"
    assert cfg.hydra.run.dir == cfg.output_dir
    assert overridden.output_dir == overridden.hydra.run.dir == "/tmp/custom-teacher"


@pytest.mark.parametrize("name", ["train_civi_teacher", "train_civi_teacher_board_selfattn"])
def test_civi_blur_covers_all_phases_and_uses_training_percentages(name):
    cfg = compose_teacher(name)
    assert cfg.policy.image_blur_scope == "all"
    assert cfg.policy.image_blur_schedule == {
        "schedule_type": "linear",
        "total_steps": None,
        "start_sigma": 5.0,
        "end_sigma": 0.0,
        "start_fraction": 2 / 3,
        "end_fraction": 1.0,
    }


@pytest.mark.parametrize("name", ["train_civi_teacher", "train_civi_teacher_board_selfattn"])
def test_configured_factr_blur_holds_two_thirds_then_opens_linearly(name):
    cfg = compose_teacher(name)
    scheduler = FactrBlurScheduler(**OmegaConf.to_container(
        cfg.policy.image_blur_schedule, resolve=True,
    ))
    assert scheduler.auto_total_steps is True
    scheduler.set_total_steps(300)
    assert [scheduler.sigma(step) for step in (0, 200, 250, 300)] == pytest.approx(
        [5.0, 5.0, 2.5, 0.0],
    )


@pytest.mark.parametrize("factor,upsample,fusion_tokens,visual_tokens", [
    (2, False, 106, 98), (2, True, 106, 392), (1, True, 400, 392),
])
def test_selfattn_fuses_torque_but_outputs_only_camera_tokens(factor, upsample, fusion_tokens, visual_tokens):
    class TinyDino(torch.nn.Module):
        config = SimpleNamespace(hidden_size=12, patch_size=16, num_register_tokens=0)

        def forward(self, pixel_values):
            batch, _, height, width = pixel_values.shape
            patches = (height // 16) * (width // 16)
            return SimpleNamespace(last_hidden_state=pixel_values.new_zeros(batch, patches + 1, 12))

    cfg = compose_teacher(overrides=[
        f"policy.fusion_downsample_factor={factor}",
        f"policy.fusion_upsample={str(upsample).lower()}",
        f"policy.fusion_highres_skip={str(upsample).lower()}",
    ])
    encoder = ForceAwareObsEncoder(
        pretrained_model_name_or_path="unused",
        backbone=TinyDino(),
        image_shapes={key: tuple(meta.shape) for key, meta in cfg.shape_meta.obs.items() if meta.type == "rgb"},
        max_obs_steps=cfg.n_obs_steps,
        wrench_dim=7,
        wrench_history_steps=cfg.policy.wrench_history_steps,
        n_emb=cfg.policy.n_emb,
        n_head=cfg.policy.n_head,
        force_temporal_encoder=cfg.policy.force_temporal_encoder,
        force_encoder_layers=cfg.policy.force_encoder_layers,
        pre_cross_self_attn_layers=cfg.policy.pre_cross_self_attn_layers,
        fusion_downsample_factor=cfg.policy.fusion_downsample_factor,
        fusion_output_mode=cfg.policy.fusion_output_mode,
        fusion_upsample=cfg.policy.fusion_upsample,
        fusion_highres_skip=cfg.policy.fusion_highres_skip,
        cross_attn_layers=cfg.policy.cross_attn_layers,
        cross_attn_query=cfg.policy.cross_attn_query,
        dropout=0.0,
    ).eval()
    images = {key: torch.zeros(1, cfg.n_obs_steps, *shape) for key, shape in encoder.image_shapes.items()}
    seen_fusion_shapes = []
    handle = encoder.pre_cross_self_attention.register_forward_pre_hook(
        lambda module, args: seen_fusion_shapes.append(tuple(args[0].shape))
    )
    try:
        with torch.no_grad():
            context = encoder(images=images, wrench_history=torch.zeros(1, cfg.n_obs_steps, 8, 7))
    finally:
        handle.remove()
    assert seen_fusion_shapes == [(2, fusion_tokens, 256)]
    assert encoder.fusion_tokens_per_observation == fusion_tokens
    # Torque participates in fusion; the action head reads the two visual grids.
    assert encoder.context_tokens_per_observation == visual_tokens
    assert context.shape == (1, 2 * visual_tokens, 256)


def test_act_checkpoint_score_describes_deterministic_action_prediction():
    cfg = compose_teacher(overrides=["checkpoint.action_mse.enabled=true"])
    assert _action_mse_protocol(cfg) == {
        "name": "deterministic_action_chunk_v1",
        "episode_selection": "all",
        "reduction": "action_element_mean",
    }


def test_dit_checkpoint_score_preserves_legacy_noise_protocol():
    cfg = compose_teacher(overrides=["action_head_backbone=dit", "checkpoint.action_mse.enabled=true"])
    original = compose_teacher("train_civi_teacher")
    assert _action_mse_protocol(cfg) == _action_mse_protocol(original) == {
        "name": "all_windows_fixed_noise_v1",
        "episode_selection": "all",
        "seed": 42,
        "noise_seed_scheme": "blake2b_episode_anchor_v1",
        "reduction": "action_element_mean",
        "num_inference_steps": 8,
        "clip_sample": True,
    }


@pytest.mark.parametrize("head", ["act", "dit"])
def test_selfattn_teacher_uses_epoch_validation_and_early_stopping(head):
    cfg = compose_teacher("train_civi_teacher_board_selfattn", [f"action_head_backbone={head}"])
    assert cfg.training.num_epochs == 1000
    assert cfg.training.num_optimizer_steps is None
    assert cfg.val_epoch_every == cfg.training.val_epoch_every == 5
    assert cfg.training.validation_start_optimizer_step == 0
    assert cfg.training.checkpoint_every_optimizer_steps is None
    assert cfg.training.early_stopping == {
        "enabled": True, "monitor": "val_loss", "patience": 100, "min_delta": 0.0,
    }
    assert cfg.task.dataset.val_ratio == 0.1
    assert cfg.checkpoint.topk.monitor_key == "val_loss"
    assert cfg.checkpoint.topk.mode == "min"
    assert cfg.checkpoint.topk.k == 1
    assert cfg.checkpoint.action_mse.enabled is False
    changed = compose_teacher(overrides=["val_epoch_every=10"])
    assert changed.training.val_epoch_every == changed.training.checkpoint_every == 10


@pytest.mark.parametrize("head", ["act", "dit"])
def test_sampled_or_disabled_checkpoint_score_retains_legacy_none(head):
    cfg = OmegaConf.create({
        "policy": {"action_head_backbone": head},
        "checkpoint": {"action_mse": {"enabled": True, "episode_selection": "random"}},
    })
    assert _action_mse_protocol(cfg) is None
    cfg.checkpoint.action_mse.enabled = False
    assert _action_mse_protocol(cfg) is None
