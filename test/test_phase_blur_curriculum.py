"""Phase-selective RGB blur through the real teacher training/context path."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from diffusion_policy.model.vision.contact_curriculum import (
    FactrBlurScheduler,
    gaussian_2d_smoothing,
)
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor
from test_force_aware_diffusion import make_policy


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_teacher():
    teacher = make_policy(
        blur_scope="precontact_contact",
        image_keys=("side", "wrist"),
        image_blur_schedule={
            "total_steps": 300, "start_sigma": 5., "end_sigma": 0.,
            "start_fraction": 1 / 3, "end_fraction": 1.,
        },
    )
    teacher.precontact_seconds = .5
    return teacher


def observations():
    return {
        **{key: torch.rand(2, 2, 3, 16, 16) for key in ("side", "wrist")},
        # All torque observations indicate contact: phase labels must still
        # preserve each free RGB frame and select pre-contact explicitly.
        "wrench_ext": torch.full((2, 2, 8, 6), 10.),
        "contact_phase": torch.tensor([[0, 1], [2, 0]]),
        # The zero-alpha pre-contact boundary must be blurred too.
        "contact_alpha": torch.tensor([[0., 0.], [1., 0.]]),
    }


def capture_inputs(teacher, seen):
    def capture(module, args, kwargs):
        seen.append({key: value.detach().clone() for key, value in kwargs["images"].items()})
    return teacher.obs_encoder.register_forward_pre_hook(capture, with_kwargs=True)


@pytest.mark.parametrize("step,sigma", [
    (0, 5.), (100, 5.), (150, 3.75), (200, 2.5),
    (250, 1.25), (299, .025), (300, 0.), (400, 0.),
])
def test_training_blurs_only_precontact_and_contact_at_scheduled_strength(step, sigma):
    teacher = make_teacher().train()
    obs = observations()
    original = {key: value.clone() for key, value in obs.items()}
    seen = []
    handle = capture_inputs(teacher, seen)
    try:
        loss = teacher.compute_loss(
            {"obs": obs, "action": torch.randn(2, 9, 7)}, optimizer_step=step,
        )
    finally:
        handle.remove()
    assert torch.isfinite(loss)
    loss.backward()
    assert teacher.model.output_head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.obs_encoder.backbone.parameters())
    selected = (obs["contact_phase"] == 1) | (obs["contact_phase"] == 2)
    for key in ("side", "wrist"):
        expected = obs[key].clone()
        if sigma > 0:
            expected[selected] = gaussian_2d_smoothing(obs[key][selected], sigma)
        torch.testing.assert_close(seen[0][key], expected)
        torch.testing.assert_close(seen[0][key][~selected], obs[key][~selected], rtol=0, atol=0)
    for key in obs:
        torch.testing.assert_close(obs[key], original[key], rtol=0, atol=0)
    assert teacher.last_curriculum_metrics["curriculum/blur_sigma"] == pytest.approx(sigma)
    assert teacher.last_curriculum_metrics["curriculum/blurred_image_fraction"] == (
        .5 if sigma > 0 else 0.
    )


def test_validation_and_distillation_targets_are_clear_without_phase_labels():
    teacher = make_teacher().eval()
    obs = observations()
    del obs["contact_phase"]
    seen = []
    handle = capture_inputs(teacher, seen)
    try:
        teacher.compute_loss({"obs": obs, "action": torch.randn(2, 9, 7)}, optimizer_step=0)
        teacher._encode_context(obs, apply_curriculum_mask=False)
    finally:
        handle.remove()
    assert len(seen) == 2
    for images in seen:
        for key in ("side", "wrist"):
            torch.testing.assert_close(images[key], obs[key], rtol=0, atol=0)
    assert teacher.last_curriculum_metrics["curriculum/blur_sigma"] == 0.


def test_active_phase_blur_requires_phase_labels_instead_of_using_alpha():
    teacher = make_teacher().train()
    obs = observations()
    del obs["contact_phase"]
    with pytest.raises(ValueError, match="requires obs\\['contact_phase'\\]"):
        teacher._encode_context(obs, apply_curriculum_mask=True)


@pytest.mark.parametrize("name", [
    "train_civi_teacher", "train_civi_teacher_selfattn", "train_civi_teacher_board_selfattn",
])
def test_current_teacher_configs_use_phase_blur_and_runtime_optimizer_budget(name):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "diffusion_policy/config")):
        cfg = compose(config_name=name)
    assert cfg.policy.image_blur_scope == "precontact_contact"
    assert cfg.policy.precontact_seconds == cfg.task.dataset.precontact_seconds == .5
    # The policy exports this duration to the dataset, which then returns
    # contact_phase rather than only the older binary alpha labels.
    teacher = make_teacher()
    assert WrenchPreprocessor.from_policy(teacher).get_config()["precontact_seconds"] == .5
    scheduler = FactrBlurScheduler(**OmegaConf.to_container(cfg.policy.image_blur_schedule, resolve=True))
    assert scheduler.auto_total_steps
    with pytest.raises(RuntimeError, match="optimizer-step budget"):
        scheduler.sigma(0)
    scheduler.set_total_steps(300)
    assert [scheduler.sigma(step) for step in (0, 100, 200, 300)] == pytest.approx([5., 5., 2.5, 0.])
