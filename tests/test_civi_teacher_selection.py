"""Unified students select final teacher weights without changing legacy selection."""

from pathlib import Path

from omegaconf import OmegaConf
import pytest

from diffusion_policy.common.checkpoint_action_mse import (
    ACTION_MSE_KEY, record_checkpoint_action_mse, resolve_student_teacher,
)
from diffusion_policy.common.config_cli import load_config_reference


ROOT = Path(__file__).resolve().parents[1]


def student_config(tmp_path, teacher_dir):
    teacher_yaml = tmp_path / "teacher.yaml"
    OmegaConf.save(OmegaConf.create({
        "output_dir": str(teacher_dir),
        "checkpoint": {"action_mse": {"enabled": False},
                       "topk": {"monitor_key": "val_action_mse_raw"}},
        "task": {"dataset": {"dataset_path": "/teacher/data"}},
    }), teacher_yaml)
    return load_config_reference(
        str(ROOT / "diffusion_policy/config/train_distill_render_net_student.yaml"),
        [f"teacher_config={teacher_yaml}"],
    )


def test_unified_student_prefers_latest_even_if_legacy_scores_rank_best_first(tmp_path):
    teacher_dir = tmp_path / "completed teacher"
    checkpoints = teacher_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    latest, best = checkpoints / "latest.ckpt", checkpoints / "best.ckpt"
    latest.touch()
    best.touch()
    record_checkpoint_action_mse(teacher_dir, {
        "optimizer_step": 1, ACTION_MSE_KEY: .01, "episode_ids": [1],
    }, best)
    cfg = student_config(tmp_path, teacher_dir)

    checkpoint, source = resolve_student_teacher(cfg)

    assert checkpoint == latest
    assert cfg.teacher_checkpoint == str(latest)
    assert cfg.teacher_selection is None
    assert source.task.dataset.dataset_path == "/teacher/data"
    assert "hydra" not in source


def test_teacher_run_override_resolves_dynamic_teacher_default_output(tmp_path):
    teacher_dir = tmp_path / "explicit completed teacher"
    latest = teacher_dir / "checkpoints/latest.ckpt"
    latest.parent.mkdir(parents=True)
    latest.touch()
    cfg = student_config(tmp_path, tmp_path / "timestamped default does not exist")
    cfg.teacher_output_dir = str(teacher_dir)

    checkpoint, _ = resolve_student_teacher(cfg)

    assert checkpoint == latest
    assert cfg.teacher_output_dir == str(teacher_dir)


def test_missing_latest_reports_run_path_and_both_cli_fixes_without_using_best(tmp_path):
    teacher_dir = tmp_path / "timestamped new run"
    best = teacher_dir / "checkpoints/best.ckpt"
    best.parent.mkdir(parents=True)
    best.touch()
    cfg = student_config(tmp_path, teacher_dir)

    with pytest.raises(FileNotFoundError) as missing:
        resolve_student_teacher(cfg)

    message = str(missing.value)
    assert str(teacher_dir / "checkpoints/latest.ckpt") in message
    for remedy in ("teacher_output_dir=", "teacher_checkpoint=", "--teacher-run", "--teacher-checkpoint"):
        assert remedy in message


def test_explicit_teacher_checkpoint_overrides_latest_and_missing_default_run(tmp_path):
    checkpoint = tmp_path / "specific teacher.ckpt"
    checkpoint.touch()
    cfg = student_config(tmp_path, tmp_path / "absent teacher")
    cfg.teacher_checkpoint = str(checkpoint)

    selected, source = resolve_student_teacher(cfg)

    assert selected == checkpoint
    assert source.task.dataset.dataset_path == "/teacher/data"
