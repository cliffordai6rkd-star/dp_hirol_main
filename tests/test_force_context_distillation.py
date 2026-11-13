"""Force-response distillation matches vectors and softly prices RGB edits."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from diffusion_policy.model.vision.force_context_distillation import (
    compute_force_context_losses,
    raw_zero_wrench_observations,
    teacher_context_pair,
)


def _encoder(grids=None):
    grids = grids or {"side": (1, 1)}
    return SimpleNamespace(
        image_keys=list(grids),
        context_patch_grid_shapes=grids,
        context_patch_counts={key: h * w for key, (h, w) in grids.items()},
    )


def _losses(student, student_zero, teacher, teacher_zero, **kwargs):
    encoder = kwargs.pop("encoder", _encoder())
    steps = kwargs.pop("obs_steps", 1)
    images = kwargs.pop("images", {
        key: torch.zeros(student.shape[0], steps, 3, h * 2, w * 2)
        for key, (h, w) in encoder.context_patch_grid_shapes.items()
    })
    rendered = kwargs.pop("rendered", {key: image + 1 for key, image in images.items()})
    return compute_force_context_losses(
        student, student_zero, teacher, teacher_zero,
        rendered, images, encoder, steps,
        kwargs.pop("difference_scale", 2.0), kwargs.pop("region_scale", 1.0),
        **kwargs,
    )


@pytest.mark.parametrize("student_delta", [[-1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
def test_difference_loss_rejects_wrong_direction_or_amplitude(student_delta):
    teacher = torch.tensor([[[1.0, 0.0]]])
    student = torch.tensor([[student_delta]])
    zero = torch.zeros_like(teacher)
    losses = _losses(student, zero, teacher, zero, difference_scale=4)
    expected = (student - teacher).square().mean()
    assert expected > 0
    torch.testing.assert_close(losses["force_delta_mse"], expected)
    torch.testing.assert_close(losses["force_delta_loss"], expected / 4)


def test_force_ignoring_student_has_opposite_gradients_in_both_branches():
    student = torch.zeros(1, 1, 2, requires_grad=True)
    student_zero = torch.zeros_like(student, requires_grad=True)
    teacher = torch.tensor([[[2.0, -1.0]]], requires_grad=True)
    teacher_zero = torch.zeros_like(teacher, requires_grad=True)
    rendered = {"side": torch.ones(1, 1, 3, 2, 2, requires_grad=True)}
    losses = _losses(student, student_zero, teacher, teacher_zero, rendered=rendered)
    assert losses["force_delta_loss"] > 0
    (losses["force_delta_loss"] + losses["context_region_loss"]).backward()
    assert student.grad.abs().sum() > 0
    torch.testing.assert_close(student.grad, -student_zero.grad)
    assert teacher.grad is None and teacher_zero.grad is None
    assert rendered["side"].grad.abs().sum() > 0


def test_exact_delta_match_is_zero_even_with_shared_context_offset():
    teacher = torch.tensor([[[2.0, -1.0]]])
    teacher_zero = torch.tensor([[[0.5, 0.25]]])
    losses = _losses(teacher + 3, teacher_zero + 3, teacher, teacher_zero)
    assert losses["force_delta_mse"] == 0
    assert losses["force_delta_loss"] == 0
    assert losses["token_mse"] == 9


def test_token_loss_detaches_teacher_targets():
    student = torch.zeros(1, 1, 2, requires_grad=True)
    teacher = torch.ones_like(student, requires_grad=True)
    losses = _losses(student, student, teacher, teacher)
    losses["token_mse"].backward()
    assert student.grad.abs().sum() > 0
    assert teacher.grad is None


def test_region_penalty_preserves_camera_time_rectangular_grid_order_and_token_mean():
    encoder = _encoder({"side": (1, 2), "wrist": (2, 2)})
    # Deliberately reverse input camera insertion order. Each patch has a
    # distinctive residual, with RGB channel factors whose mean is one.
    patch_residuals = {
        "wrist": torch.tensor([[[[3., 4.], [5., 6.]], [[9., 10.], [11., 12.]]]]),
        "side": torch.tensor([[[[1., 2.]], [[7., 8.]]]]),
    }
    rgb_factors = torch.tensor([0., 1., 2.]).reshape(1, 1, 3, 1, 1)
    rendered = {
        key: values[:, :, None].repeat_interleave(2, -2).repeat_interleave(3, -1) * rgb_factors
        for key, values in patch_residuals.items()
    }
    images = {key: torch.zeros_like(value) for key, value in rendered.items()}
    response = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1)
    zeros = torch.zeros_like(response)
    losses = _losses(
        zeros, zeros, response, zeros, encoder=encoder, obs_steps=2,
        images=images, rendered=rendered,
    )
    # The token residuals are [1..12], the response is [0..11]; therefore
    # each weighted residual is exactly one when region_scale is one.
    torch.testing.assert_close(losses["context_region_loss"], torch.tensor(1.0))


def test_region_pooling_uses_absolute_rgb_difference_before_spatial_mean():
    zeros = torch.zeros(1, 1, 2)
    rendered = {"side": torch.tensor([1., -1., -1., 1.]).reshape(1, 1, 1, 2, 2).expand(-1, -1, 3, -1, -1)}
    losses = _losses(zeros, zeros, zeros, zeros, rendered=rendered)
    assert losses["context_region_loss"] == 1


def test_low_response_has_stronger_edit_penalty_and_zero_delta_has_no_hotspots():
    encoder = _encoder({"side": (1, 3)})
    zero = torch.zeros(1, 3, 2)
    response = torch.tensor([0., 1., 9.]).reshape(1, 3, 1).expand(-1, -1, 2)
    rendered = {"side": torch.ones(1, 1, 3, 2, 6, requires_grad=True)}
    losses = _losses(zero, zero, response, zero, encoder=encoder, rendered=rendered)
    torch.testing.assert_close(losses["context_region_loss"], torch.tensor((1 + .5 + .1) / 3))
    losses["context_region_loss"].backward()
    gradients = rendered["side"].grad.reshape(1, 1, 3, 2, 3, 2).sum((0, 1, 2, 3, 5))
    assert gradients[0] > gradients[1] > gradients[2] > 0
    no_response = _losses(zero, zero, zero, zero, encoder=encoder)
    assert no_response["context_region_loss"] == 1


def test_high_response_does_not_require_an_rgb_edit():
    zero = torch.zeros(1, 1, 2)
    rendered = {"side": torch.zeros(1, 1, 3, 2, 2, requires_grad=True)}
    losses = _losses(zero, zero, zero + 100, zero, rendered=rendered)
    assert losses["context_region_loss"] == 0
    losses["context_region_loss"].backward()
    assert torch.count_nonzero(rendered["side"].grad) == 0


@pytest.mark.parametrize("grid,token_count", [(14, 784), (7, 196)])
def test_board_teacher_layouts_and_fp32_losses_under_autocast(grid, token_count):
    encoder = _encoder({"side": (grid, grid), "wrist": (grid, grid)})
    student = torch.zeros(1, token_count, 256, dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.ones_like(student, requires_grad=True)
    images = {key: torch.zeros(1, 2, 3, 28, 28, dtype=torch.bfloat16) for key in encoder.image_keys}
    rendered = {key: torch.ones_like(value, requires_grad=True) for key, value in images.items()}
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        losses = _losses(
            student, student, teacher, torch.zeros_like(teacher), encoder=encoder,
            obs_steps=2, images=images, rendered=rendered,
        )
    for value in losses.values():
        assert value.dtype == torch.float32 and value.ndim == 0
        assert torch.isfinite(value)
    torch.testing.assert_close(losses["context_region_loss"], torch.tensor(.5))
    sum(losses.values()).backward()
    assert student.grad is not None and teacher.grad is None
    assert all(value.grad.abs().sum() > 0 for value in rendered.values())


@pytest.mark.parametrize("scale", [0, -1, float("inf"), float("nan")])
@pytest.mark.parametrize("name", ["difference_scale", "region_scale"])
def test_scales_must_be_positive_and_finite(name, scale):
    zero = torch.zeros(1, 1, 2)
    with pytest.raises(ValueError, match=name):
        _losses(zero, zero, zero, zero, **{name: scale})


def test_mismatched_context_shape_is_rejected():
    zero = torch.zeros(1, 1, 2)
    with pytest.raises(ValueError, match="All context tensors"):
        _losses(zero, zero.expand(1, 2, 2), zero, zero)
    with pytest.raises(ValueError, match="visual patch counts"):
        _losses(zero, zero, zero, zero, obs_steps=2)


class _Teacher(nn.Module):
    wrench_key = "tau_ext"
    phase_wrench_key = "phase_tau"
    future_phase_wrench_key = "future_tau"

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(2), requires_grad=False)
        self.calls = []

    def _encode_context(self, obs, *, apply_curriculum_mask):
        assert not self.training and not torch.is_grad_enabled()
        assert apply_curriculum_mask is False
        assert all(not value.is_floating_point() or value.dtype == torch.float32
                   for value in obs.values() if torch.is_tensor(value))
        self.calls.append(obs)
        # Physical raw zero normalizes to a nonzero value in this example.
        normalized = (obs[self.wrench_key] - 2) / 4
        result = normalized @ self.weight
        assert result.dtype == torch.float32
        return result


def _observations(dtype=torch.float32):
    return {
        "image": torch.ones(1, 2, 3, 4, 4, dtype=dtype),
        "tau_ext": torch.full((1, 2, 2), 6., dtype=dtype, requires_grad=True),
        "phase_tau": torch.ones(1, 2, 2, dtype=dtype),
        "future_tau": torch.ones(1, 4, 2, dtype=dtype),
        "contact_alpha": torch.ones(1, 2, dtype=dtype),
        "contact_phase": torch.ones(1, 2, dtype=torch.long),
        "sample_index": torch.ones(1, dtype=torch.long),
    }


@pytest.mark.parametrize("shared_phase_key", [True, False])
def test_raw_zero_copy_preserves_images_and_original_observations(shared_phase_key):
    teacher = _Teacher().eval()
    if shared_phase_key:
        teacher.phase_wrench_key = teacher.wrench_key
        teacher.future_phase_wrench_key = None
    obs = _observations()
    snapshots = {key: value.clone() for key, value in obs.items()}
    reference = raw_zero_wrench_observations(teacher, obs)
    assert reference is not obs and reference["image"] is obs["image"]
    assert "contact_alpha" not in reference and "contact_phase" not in reference
    for key in (teacher.wrench_key, teacher.phase_wrench_key, teacher.future_phase_wrench_key):
        if key is not None:
            assert torch.count_nonzero(reference[key]) == 0
            assert reference[key] is not obs[key]
    for key in obs:
        torch.testing.assert_close(obs[key], snapshots[key])


def test_teacher_pair_uses_two_clear_fp32_passes_and_raw_zero_normalization():
    teacher = _Teacher().eval()
    obs = _observations(dtype=torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        pair = teacher_context_pair(teacher, obs)
    assert set(pair) == {"tokens", "zero_tokens"}
    assert len(teacher.calls) == 2
    torch.testing.assert_close(pair["tokens"], torch.ones(1, 2, 2))
    torch.testing.assert_close(pair["zero_tokens"], torch.full((1, 2, 2), -.5))
    assert all(value.dtype == torch.float32 and not value.requires_grad for value in pair.values())
    assert "contact_alpha" in teacher.calls[0] and "contact_phase" in teacher.calls[0]
    assert "contact_alpha" not in teacher.calls[1] and "contact_phase" not in teacher.calls[1]
    assert teacher.calls[0]["image"] is teacher.calls[1]["image"]
    assert teacher.calls[0]["sample_index"] is obs["sample_index"]
    assert obs["image"].dtype == torch.bfloat16
    assert obs["tau_ext"].dtype == torch.bfloat16 and torch.all(obs["tau_ext"] == 6)
    assert "contact_alpha" in obs and "contact_phase" in obs
    assert not teacher.training


@pytest.mark.parametrize("training,trainable", [(True, False), (False, True)])
def test_teacher_pair_requires_frozen_eval_teacher(training, trainable):
    teacher = _Teacher().train(training).requires_grad_(trainable)
    with pytest.raises(ValueError, match="frozen eval teacher"):
        teacher_context_pair(teacher, _observations())
    assert len(teacher.calls) == 0 and teacher.training == training
