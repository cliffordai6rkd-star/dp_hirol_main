"""Compose the dedicated student preset through the same loader as the CLI."""

from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

from diffusion_policy.common import checkpoint_action_mse
from diffusion_policy.common.config_cli import load_config_reference
from diffusion_policy.model.vision.render_net import build_render_net


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "diffusion_policy/config"
STUDENT = CONFIG_DIR / "train_distill_render_net_board_selfattn.yaml"


def _config(*overrides, include_hydra=False):
    return load_config_reference(str(STUDENT), overrides, include_hydra=include_hydra)


def test_new_student_requires_an_explicit_teacher_checkpoint():
    cfg = _config()
    assert OmegaConf.is_missing(cfg, "teacher_checkpoint")
    assert cfg.teacher_config == "diffusion_policy/config/train_civi_teacher_board_selfattn.yaml"
    with pytest.raises(ValueError, match="Set teacher_checkpoint="):
        checkpoint_action_mse.resolve_student_teacher(cfg)


def test_explicit_checkpoint_is_resolved_without_automatic_teacher_selection(monkeypatch, tmp_path):
    checkpoint = tmp_path / "new_teacher" / "best.ckpt"
    cfg = _config(f"teacher_checkpoint={checkpoint}")
    cfg.teacher_config = str(ROOT / cfg.teacher_config)

    def unexpected_selection(*args, **kwargs):
        pytest.fail("the dedicated student must use its explicitly selected checkpoint")

    monkeypatch.setattr(checkpoint_action_mse, "select_action_mse_teacher", unexpected_selection)
    actual, teacher_cfg = checkpoint_action_mse.resolve_student_teacher(cfg)
    assert actual == checkpoint
    assert cfg.teacher_checkpoint == str(checkpoint)
    assert cfg.teacher_selection is None
    assert teacher_cfg.policy.fusion_output_mode == "visual"
    assert teacher_cfg.policy.fusion_upsample is False
    assert teacher_cfg.policy.action_head_backbone == "act"


def test_force_context_losses_preserve_all_phase_supervision_without_gate_heads():
    cfg = _config()
    assert cfg.distillation.objective == "token_mse"
    assert cfg.distillation.token_region_phases is None
    assert cfg.distillation.skip_unselected_observations is False
    assert cfg.distillation.use_teacher_ffn is False
    assert cfg.distillation.token_loss_weight == 1.0
    assert cfg.distillation.demo_loss_weight == 0.0
    assert cfg.distillation.phase_ce_weight == cfg.distillation.alpha_loss_weight == 0.0
    assert cfg.distillation.force_context == {
        "enabled": True, "difference_weight": 1.0, "region_weight": 1.0,
        "difference_scale": None, "region_scale": None,
    }
    assert cfg.render_net.condition_backbone is None
    assert cfg.render_net.shared_condition_backbone is False
    assert cfg.render_net.separate_gate_path is False
    assert cfg.render_net.use_contact_alpha is False
    assert cfg.render_net.use_region_gate is False
    assert cfg.render_net.phase_prediction.enabled is False
    assert cfg.render_net.phase_prediction.freeze_wrench_encoder is True
    attention = cfg.render_net.high_attention_render
    assert attention.enabled is False
    assert all(attention[key] == 0 for key in (
        "intensity_loss_weight", "context_patch_loss_weight", "pixel_loss_weight",
        "smoothness_loss_weight",
    ))
    assert cfg.training.render_stages.enabled is False
    assert cfg.training.render_stages.condition_early_stopping.enabled is False
    assert cfg.training.render_stages.condition_warmup_epochs == 0
    assert cfg.dataset.phase_resampling_enabled is False
    assert cfg.dataset.contact_label_config is None
    assert cfg.dataset.image_randomer_config is None
    assert "image_blur_schedule" not in cfg.render_net
    assert "image_blur_schedule" not in cfg.distillation


def test_force_pair_cache_is_float32_and_retained_before_training():
    cfg = _config("teacher_cache.only=true", "distillation.force_context.difference_weight=0.5",
                  "distillation.force_context.region_weight=0.2")
    assert cfg.teacher_cache.enabled and cfg.teacher_cache.only
    assert cfg.teacher_cache.dtype == cfg.teacher_cache.precision == "float32"
    assert cfg.teacher_cache.cleanup_on_exit is False
    assert cfg.distillation.force_context.difference_scale is None
    assert cfg.distillation.force_context.region_scale is None
    assert cfg.distillation.force_context.difference_weight == 0.5
    assert cfg.distillation.force_context.region_weight == 0.2


def test_composed_render_config_builds_without_gate_heads_and_freezes_force_encoder():
    cfg = _config()
    options = OmegaConf.to_container(cfg.render_net, resolve=True)
    options.update(options.pop("backbone"))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = build_render_net(7, options)
        # The workspace freezes this path after copying the teacher weights.
        model.freeze_wrench_encoder()
        model.set_training_stage("joint")
        model.set_stage_mode(True)
    finally:
        torch.set_num_threads(previous_threads)
    assert model.cross_camera_attn_layers == len(model.cross_camera_attention) == 1
    assert model.initialize_wrench_from_teacher
    assert model.mask_head is None
    assert model.phase_temporal_encoder is None
    assert model.gate_backbone is None
    assert model.gate_wrench_encoder is None
    assert list(model.gate_parameters()) == []
    assert not model.wrench_encoder.training
    assert all(not parameter.requires_grad for parameter in model.wrench_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.wrench_projection.parameters())
    assert all(parameter.requires_grad for parameter in model.cross_camera_attention.parameters())
    assert {id(parameter) for parameter in model.rgb_parameters()} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }


def test_new_student_artifacts_and_hydra_follow_model_output_override(tmp_path):
    cfg = _config(include_hydra=True)
    assert Path(cfg.output_dir).parts[0] == "model_output"
    assert Path(cfg.teacher_cache.directory).parts[0] == "model_output"
    assert cfg.hydra.run.dir == cfg.output_dir
    assert cfg.logging.enable_plots
    custom = _config(f"output_dir={tmp_path / 'student'}", include_hydra=True)
    assert custom.hydra.run.dir == custom.output_dir == str(tmp_path / "student")


def test_historical_student_preset_retains_its_two_stage_configuration():
    cfg = load_config_reference(str(CONFIG_DIR / "train_distill_render_net_student.yaml"))
    assert not OmegaConf.is_missing(cfg, "teacher_checkpoint")
    assert cfg.teacher_checkpoint is None
    assert cfg.render_net.condition_backbone is not None
    assert cfg.render_net.phase_prediction.enabled
    assert cfg.render_net.high_attention_render.enabled
    assert cfg.training.render_stages.enabled
    assert list(cfg.distillation.token_region_phases) == [1, 2]
