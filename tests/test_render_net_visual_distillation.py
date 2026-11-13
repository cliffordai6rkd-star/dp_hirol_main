"""Full RGB-only workspace integration with the frozen visual ACT teacher."""

import io
from pathlib import Path

import hydra
import pytest
import torch
from omegaconf import OmegaConf, open_dict
from torch.nn import functional as F

from diffusion_policy.model.vision.force_aware_obs_encoder import ForceAwareObsEncoder
from diffusion_policy.model.vision.render_net import build_render_net
from diffusion_policy.policy import force_aware_diffusion_transformer_policy as policy_module
from diffusion_policy.workspace import train_distill_render_net_workspace as distill
from test_civi_action_heads import _PatchBackbone, _batch
from test_visual_only_teacher_context import _policy, _policy_config


@pytest.fixture(autouse=True, scope="module")
def _cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _workspace(monkeypatch, tmp_path, upsample):
    monkeypatch.setattr(
        policy_module, "ForceAwareObsEncoder",
        lambda **kwargs: ForceAwareObsEncoder(**kwargs, backbone=_PatchBackbone()),
    )
    policy_config = _policy_config("act", "visual", upsample)
    with open_dict(policy_config):
        policy_config.cross_attn_query = "image"
    teacher = _policy(policy_config)
    teacher_cfg = OmegaConf.create({
        "policy": policy_config,
        "task": {"dataset": {
            "_target_": "diffusion_policy.dataset.lerobot_v3_dataset.LeRobotV3ImageDataset",
            "dataset_path": str(tmp_path / "data"),
            "shape_meta": policy_config.shape_meta,
            "precontact_seconds": 0.5,
            "contact_label_config": {"stale_teacher_labels": True},
        }},
    })
    monkeypatch.setattr(distill, "resolve_student_teacher",
                        lambda cfg: (tmp_path / "teacher.ckpt", None))
    monkeypatch.setattr(distill, "load_checkpoint_payload",
                        lambda path: ({"state_dicts": {"model": teacher.state_dict()}}, teacher_cfg))
    monkeypatch.setattr(distill, "instantiate_policy", lambda *args, **kwargs: teacher)
    config_dir = Path(__file__).resolve().parents[1] / "diffusion_policy/config"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(config_name="train_distill_render_net_board_selfattn")
    with open_dict(cfg):
        cfg.teacher_checkpoint = str(tmp_path / "teacher.ckpt")
        cfg.training.device = "cpu"
        cfg.training.amp.enabled = False
        cfg.training.compile.enabled = False
        cfg.training.use_ema = False
        cfg.training.resume_from = None
        cfg.teacher_cache.enabled = False
        # This regression keeps the original token-only objective; the
        # paired objective is exercised in test_force_context_workspace.
        cfg.distillation.force_context.enabled = False
        cfg.optimizer.lr = 0.003
        cfg.optimizer.weight_decay = 0.0
        cfg.render_net.cross_camera_attn_layers = 1
        cfg.render_net.backbone.update({
            "image_encoder": "cnn", "patch_size": 4, "decoder_layers": 2,
            "base_channels": 4, "embed_dim": 16, "num_heads": 4,
            "image_self_attn_layers": 1, "cross_attn_layers": 1,
            "mlp_ratio": 2.0, "dropout": 0.0,
        })
    return distill.TrainDistillRenderNetWorkspace(cfg, output_dir=str(tmp_path / "run"))


def _forbidden(*args, **kwargs):
    pytest.fail("token-only distillation must not call ACT or condition/region supervision")


def _assert_nonzero_gradients(module):
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None and torch.isfinite(gradient).all()
                             for gradient in gradients)
    assert sum(gradient.abs().sum().item() for gradient in gradients) > 0


@pytest.mark.parametrize("upsample,patches_per_camera", [(True, 4), (False, 1)])
def test_visual_workspace_trains_only_renderer_on_every_observation(
    monkeypatch, tmp_path, upsample, patches_per_camera,
):
    workspace = _workspace(monkeypatch, tmp_path, upsample)
    teacher, renderer = workspace.teacher_policy, workspace.model
    monkeypatch.setattr(teacher.model, "forward", _forbidden)
    monkeypatch.setattr(workspace, "_region_targets", _forbidden)
    monkeypatch.setattr(workspace, "_denoiser_losses", _forbidden)
    assert workspace.objective == "token_mse"
    assert workspace.token_region_phases is None
    assert not workspace.skip_unselected_observations
    assert not workspace.use_teacher_ffn
    assert not workspace.render_stages_enabled
    assert not workspace.attention_enabled
    assert not renderer.use_contact_alpha and not renderer.use_region_gate
    assert not renderer.phase_prediction_enabled
    assert renderer.mask_head is None and renderer.phase_temporal_encoder is None
    assert renderer.gate_backbone is None and renderer.gate_wrench_encoder is None
    assert len(renderer.cross_camera_attention) == 1
    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    trainable = {id(parameter) for parameter in renderer.parameters() if parameter.requires_grad}
    optimized = {id(parameter) for group in workspace.optimizer.param_groups
                 for parameter in group["params"]}
    # Historical optimizer layout retains frozen entries for checkpoint
    # compatibility; AdamW skips them because they never acquire gradients.
    assert optimized == {id(parameter) for parameter in renderer.parameters()}
    assert trainable < optimized
    assert optimized.isdisjoint(id(parameter) for parameter in teacher.parameters())
    frozen_renderer = {name: parameter.detach().clone()
                       for name, parameter in renderer.named_parameters()
                       if not parameter.requires_grad}
    assert frozen_renderer
    assert all(not parameter.requires_grad for parameter in renderer.wrench_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in renderer.wrench_projection.parameters())
    for name in ("wrench_position", "obs_position", "force_modality"):
        assert not getattr(renderer, name).requires_grad

    # No offline phase/alpha labels are available; all four observations are
    # noncontact and must still contribute to both rendering and token MSE.
    observations = _batch()["obs"]
    observations["tau_ext"] *= 0.001
    with torch.no_grad():
        teacher_tokens = teacher._encode_context(observations, apply_curriculum_mask=False)
    before_teacher = {key: value.clone() for key, value in teacher.state_dict().items()}
    before_renderer = {key: value.clone() for key, value in renderer.state_dict().items()}
    total_tokens = 2 * 2 * patches_per_camera
    assert teacher_tokens.shape == (2, total_tokens, 16)
    renderer.set_stage_mode(True)
    assert not renderer.wrench_encoder.training
    # Same preprocessing and copied frozen weights preserve the teacher's
    # force-token representation throughout student training.
    _, normalized_wrench, _, _ = teacher.prepare_context_inputs(observations)
    with torch.no_grad():
        expected_force = (teacher.obs_encoder.wrench_projection(normalized_wrench)
                          + teacher.obs_encoder.wrench_position
                          + teacher.obs_encoder.obs_position[:, :2]
                          + teacher.obs_encoder.force_modality)
        expected_force, _ = teacher.obs_encoder.force_encoder(expected_force.flatten(0, 1))
        torch.testing.assert_close(renderer.encode_wrench_history(normalized_wrench),
                                   expected_force, rtol=0, atol=0)
    for step in range(3):
        workspace.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            student_tokens = distill.student_context_tokens(teacher, renderer, observations)
            expected = F.mse_loss(student_tokens.float(), teacher_tokens.float())
        loss = workspace._batch_loss({"obs": observations})
        torch.testing.assert_close(loss.detach(), expected, rtol=1e-6, atol=1e-7)
        assert loss.requires_grad and loss.item() > 0
        assert workspace.loss_metrics["token_region_observation_count"] == 4
        assert workspace.loss_metrics["contact_open_ratio"] == 0
        assert workspace.loss_metrics["closed_token_count"] == 2 * total_tokens
        for metric in ("denoiser_mse", "demo_mse", "mask_mse", "context_patch_mse",
                       "intensity_smoothness", "intensity_loss", "phase_ce", "alpha_mse"):
            assert workspace.loss_metrics[metric] == 0
        loss.backward()
        # Identity initialization zeroes the RGB head, so gradients reach its
        # upstream encoders after the first optimizer update.
        if step > 0:
            _assert_nonzero_gradients(renderer.cross_camera_attention)
            _assert_nonzero_gradients(renderer.backbone.cross_attention)
        for name, parameter in renderer.named_parameters():
            if name in frozen_renderer:
                assert parameter.grad is None
        assert all(parameter.grad is None for parameter in teacher.parameters())
        workspace.optimizer.step()
        for name, parameter in renderer.named_parameters():
            if name in frozen_renderer:
                torch.testing.assert_close(parameter, frozen_renderer[name], rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(renderer.encode_wrench_history(normalized_wrench),
                                   expected_force, rtol=0, atol=0)
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, before_teacher[key], rtol=0, atol=0)
    assert any(not torch.equal(value, before_renderer[key])
               for key, value in renderer.state_dict().items())
    assert workspace.shape_info["teacher_token_shape"] == [2, total_tokens, 16]
    assert workspace.shape_info["student_token_shape"] == [2, total_tokens, 16]
    assert workspace.shape_info["patch_tokens_per_camera"] == {
        "side": patches_per_camera, "wrist": patches_per_camera,
    }
    assert workspace.shape_info["patch_tokens_per_observation"] == 2 * patches_per_camera
    dataset_config = workspace._dataset_config()
    assert dataset_config.contact_label_config is None
    assert dataset_config.precontact_seconds == 0.5

    # Cached token supervision must bypass teacher fusion, require no actions
    # or region cache, and match the ordinary loss and renderer gradients.
    workspace.optimizer.zero_grad(set_to_none=True)
    online_loss = workspace._batch_loss({"obs": observations})
    online_loss.backward()
    gradients = {name: parameter.grad.clone() for name, parameter in renderer.named_parameters()
                 if parameter.requires_grad}
    workspace.optimizer.zero_grad(set_to_none=True)
    monkeypatch.setattr(teacher, "_encode_context", _forbidden)
    cached_loss = workspace._batch_loss({
        "obs": observations, "teacher_targets": {"tokens": teacher_tokens},
    })
    torch.testing.assert_close(cached_loss.detach(), online_loss.detach(), rtol=0, atol=0)
    cached_loss.backward()
    for name, parameter in renderer.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
        else:
            assert parameter.grad is None

    # Deployment rebuilds only RenderNet; all frozen visual weights remain
    # external, and no teacher fusion/ACT execution is needed to render RGB.
    renderer.eval()
    render_config = OmegaConf.to_container(workspace.cfg.render_net, resolve=True)
    stream = io.BytesIO()
    torch.save({"config": render_config, "state_dict": renderer.state_dict()}, stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    restored = build_render_net(7, saved["config"]).eval()
    restored.load_state_dict(saved["state_dict"], strict=True)
    images, wrench, _, _ = teacher.prepare_context_inputs(observations)
    with torch.no_grad():
        expected_rgb = renderer(images, wrench)
        actual_rgb = restored(images, wrench)
    for key in ("side", "wrist"):
        torch.testing.assert_close(actual_rgb[key], expected_rgb[key], rtol=0, atol=0)
