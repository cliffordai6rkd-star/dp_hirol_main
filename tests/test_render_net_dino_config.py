"""DINO defaults, independent weights, and architecture-aware student restore."""

from pathlib import Path

import dill
import pytest
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace import train_distill_render_net_workspace as distill
from test_render_net_ddp import TinyPreprocessor, TinyRenderer, TinyTeacher, _config


def _mock_workspace_dependencies(monkeypatch, tmp_path, *, teacher_path="models/dino-teacher"):
    teacher = TinyTeacher()
    if teacher_path is not None:
        teacher.obs_encoder.pretrained_model_name_or_path = teacher_path
    teacher_cfg = OmegaConf.create({"policy": {}, "task": {"dataset": {}}})
    monkeypatch.setattr(distill, "resolve_student_teacher",
                        lambda _: (tmp_path / "teacher.ckpt", teacher_cfg))
    monkeypatch.setattr(distill, "load_checkpoint_payload",
                        lambda _: ({"state_dicts": {}}, teacher_cfg))
    monkeypatch.setattr(distill, "instantiate_policy", lambda *args, **kwargs: teacher)
    monkeypatch.setattr(distill.WrenchPreprocessor, "from_policy", lambda _: TinyPreprocessor())
    builds = []

    def build(wrench_dim, config):
        builds.append((wrench_dim, config))
        return TinyRenderer()

    monkeypatch.setattr(distill, "build_render_net", build)
    return teacher, builds


def _dino_config(tmp_path, *, path=None):
    cfg = _config(tmp_path)
    cfg.render_net.backbone.image_encoder = "dino_lora"
    cfg.render_net.backbone.dino = {
        "pretrained_model_name_or_path": path,
        "local_files_only": True,
        "lora": {"rank": 8, "alpha": 16, "dropout": 0., "target_modules": ["q_proj", "v_proj"]},
        "intermediate_layers": None,
    }
    return cfg


def test_explicit_dino_joint_preset_retains_pretrained_lora_and_fusion_dimensions():
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] /
                         "diffusion_policy/config/train_distill_render_net_student_dino_joint.yaml")
    backbone = cfg.render_net.backbone
    assert backbone.image_encoder == "dino_lora"
    assert backbone.patch_size == 16 and backbone.image_self_attn_layers == 0
    assert backbone.embed_dim == 128 and backbone.cross_attn_layers == 2
    assert backbone.decoder_layers == 3 and backbone.base_channels == 32
    assert backbone.dino.pretrained_model_name_or_path is None
    assert backbone.dino.local_files_only
    assert dict(backbone.dino.lora) == {
        "rank": 8, "alpha": 16, "dropout": 0., "target_modules": ["q_proj", "v_proj"],
    }
    assert backbone.dino.intermediate_layers is None
    assert cfg.render_net.shared_condition_backbone
    assert cfg.render_net.spatial_modulation.enabled
    assert cfg.training.render_stages.condition_warmup_epochs == 1
    assert cfg.dataloader.batch_size == 4 and cfg.val_dataloader.batch_size == 8
    assert list(cfg.distillation.token_region_phases) == [1, 2]


@pytest.mark.parametrize("configured_path,expected", [
    (None, "models/dino-teacher"),
    ("models/dino-explicit", "models/dino-explicit"),
])
def test_runtime_resolves_and_persists_independent_student_dino_path(
        monkeypatch, tmp_path, configured_path, expected):
    teacher, builds = _mock_workspace_dependencies(monkeypatch, tmp_path)
    workspace = distill.TrainDistillRenderNetWorkspace(
        _dino_config(tmp_path, path=configured_path), output_dir=str(tmp_path))
    assert builds[0][0] == 1
    assert builds[0][1]["dino"]["pretrained_model_name_or_path"] == expected
    assert workspace.cfg.render_net.dino.pretrained_model_name_or_path == expected
    assert teacher.obs_encoder.pretrained_model_name_or_path == "models/dino-teacher"
    assert not any(p.requires_grad for p in teacher.parameters())
    checkpoint = workspace.save_checkpoint(use_thread=False)
    payload = torch.load(checkpoint, pickle_module=dill, map_location="cpu")
    assert payload["cfg"].render_net.dino.pretrained_model_name_or_path == expected


def test_dino_requires_an_explicit_or_teacher_pretrained_path(monkeypatch, tmp_path):
    _, builds = _mock_workspace_dependencies(monkeypatch, tmp_path, teacher_path=None)
    with pytest.raises(ValueError, match="DINO RenderNet requires"):
        distill.TrainDistillRenderNetWorkspace(_dino_config(tmp_path), output_dir=str(tmp_path))
    assert not builds


def test_legacy_cnn_does_not_require_a_teacher_dino_path(monkeypatch, tmp_path):
    _mock_workspace_dependencies(monkeypatch, tmp_path, teacher_path=None)
    workspace = distill.TrainDistillRenderNetWorkspace(_config(tmp_path), output_dir=str(tmp_path))
    assert workspace.cfg.render_net.image_encoder == "cnn"


@pytest.mark.parametrize("saved_config_shape", ["flat", "nested", "legacy_missing"])
def test_cnn_resume_cannot_replace_new_dino_architecture_even_with_matching_states(
        monkeypatch, tmp_path, saved_config_shape):
    _mock_workspace_dependencies(monkeypatch, tmp_path)
    original = distill.TrainDistillRenderNetWorkspace(_config(tmp_path), output_dir=str(tmp_path))
    path = original.save_checkpoint(use_thread=False)
    if saved_config_shape != "flat":
        payload = torch.load(path, pickle_module=dill, map_location="cpu")
        del payload["cfg"].render_net.image_encoder
        if saved_config_shape == "nested":
            payload["cfg"].render_net.backbone = {"image_encoder": "cnn"}
        torch.save(payload, path, pickle_module=dill)
    cfg = _dino_config(tmp_path)
    cfg.training.resume_from = path
    with pytest.raises(RuntimeError, match="image_encoder: checkpoint=cnn, current=dino_lora"):
        distill.TrainDistillRenderNetWorkspace(cfg, output_dir=str(tmp_path))
    assert cfg.render_net.image_encoder == "dino_lora"


def test_matching_dino_family_can_resume_with_current_resolved_config(monkeypatch, tmp_path):
    _mock_workspace_dependencies(monkeypatch, tmp_path)
    original = distill.TrainDistillRenderNetWorkspace(_dino_config(tmp_path), output_dir=str(tmp_path))
    with torch.no_grad():
        original.model.offset.weight.fill_(.37)
    path = original.save_checkpoint(use_thread=False)
    cfg = _dino_config(tmp_path)
    cfg.training.resume_from = path
    resumed = distill.TrainDistillRenderNetWorkspace(cfg, output_dir=str(tmp_path))
    assert resumed.cfg.render_net.image_encoder == "dino_lora"
    assert resumed.cfg.render_net.dino.pretrained_model_name_or_path == "models/dino-teacher"
    torch.testing.assert_close(resumed.model.offset.weight, original.model.offset.weight)
