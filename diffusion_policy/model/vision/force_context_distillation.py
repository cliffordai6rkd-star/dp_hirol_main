"""Distill final visual context and its response to raw torque interventions."""

import math

import torch
import torch.nn.functional as F


def raw_zero_wrench_observations(teacher, obs):
    """Copy observations with physical zero torque and no stale phase labels."""
    reference = dict(obs)
    reference.pop("contact_alpha", None)
    reference.pop("contact_phase", None)
    keys = dict.fromkeys(getattr(teacher, name, None) for name in (
        "wrench_key", "phase_wrench_key", "future_phase_wrench_key",
    ))
    for key in keys:
        if key is not None:
            reference[key] = torch.zeros_like(obs[key])
    return reference


@torch.no_grad()
def teacher_context_pair(teacher, obs):
    """Return clear FP32 targets for real and physically zero raw torque.

    The teacher stays frozen and in eval mode. Both interventions re-enter
    its normal preprocessing, so raw zeros are normalized by the teacher.
    """
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError("Context distillation requires a frozen eval teacher")
    observations = {
        key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
        for key, value in obs.items()
    }
    reference = raw_zero_wrench_observations(teacher, observations)
    device = next(value.device for value in observations.values() if torch.is_tensor(value))
    with torch.autocast(device_type=device.type, enabled=False):
        tokens = teacher._encode_context(observations, apply_curriculum_mask=False)
        zero_tokens = teacher._encode_context(reference, apply_curriculum_mask=False)
    return {"tokens": tokens.detach().float(), "zero_tokens": zero_tokens.detach().float()}


def compute_force_context_losses(
    student_tokens, student_zero_tokens,
    teacher_tokens, teacher_zero_tokens,
    rendered, images, encoder, obs_steps,
    difference_scale, region_scale,
):
    """Match context vectors and penalize RGB edits in low-response regions.

    Teacher response magnitudes supply a soft prior on final output token
    positions; they do not identify exact causal input pixels. Only edits in
    the real-torque rendered branch incur the region penalty.
    """
    difference_scale = float(difference_scale)
    region_scale = float(region_scale)
    if not math.isfinite(difference_scale) or difference_scale <= 0:
        raise ValueError("difference_scale must be positive and finite")
    if not math.isfinite(region_scale) or region_scale <= 0:
        raise ValueError("region_scale must be positive and finite")
    if student_tokens.ndim != 3 or any(tokens.shape != student_tokens.shape for tokens in (
        student_zero_tokens, teacher_tokens, teacher_zero_tokens,
    )):
        raise ValueError("All context tensors must share shape [B, tokens, D]")

    obs_steps = int(obs_steps)
    counts = encoder.context_patch_counts
    grids = encoder.context_patch_grid_shapes
    patches = sum(int(counts[key]) for key in encoder.image_keys)
    batch_size = student_tokens.shape[0]
    if obs_steps <= 0 or student_tokens.shape[1] != obs_steps * patches:
        raise ValueError("Context tokens do not match observation steps and visual patch counts")

    with torch.autocast(device_type=student_tokens.device.type, enabled=False):
        student = student_tokens.float()
        student_zero = student_zero_tokens.float()
        teacher = teacher_tokens.detach().float()
        teacher_zero = teacher_zero_tokens.detach().float()
        delta_teacher = teacher - teacher_zero
        delta_student = student - student_zero
        token_mse = (student - teacher).square().mean()
        force_delta_mse = (delta_student - delta_teacher).square().mean()
        force_delta_loss = force_delta_mse / difference_scale

        response = delta_teacher.square().mean(dim=-1).sqrt()
        weights = (response / (response + region_scale)).detach()
        camera_residuals = []
        for key in encoder.image_keys:
            edited, original = rendered[key], images[key]
            if (edited.ndim != 5 or edited.shape != original.shape
                    or edited.shape[:3] != (batch_size, obs_steps, 3)):
                raise ValueError(f"Images for {key!r} must share shape [B, obs_steps, 3, H, W]")
            height, width = grids[key]
            count = int(counts[key])
            if height <= 0 or width <= 0 or height * width != count:
                raise ValueError(f"Context patch grid and count disagree for {key!r}")
            residual = (edited.float() - original.detach().float()).abs()
            pooled = F.adaptive_avg_pool2d(
                residual.flatten(0, 1), (height, width),
            ).mean(dim=1)
            camera_residuals.append(pooled.reshape(batch_size, obs_steps, count))

        # Preserve observation -> camera -> row-major patch ordering. Taking
        # one mean over concatenated patches also handles unequal camera grids.
        residual_tokens = torch.cat(camera_residuals, dim=-1).reshape(batch_size, -1)
        context_region_loss = ((1 - weights) * residual_tokens).mean()

    return {
        "token_mse": token_mse,
        "force_delta_mse": force_delta_mse,
        "force_delta_loss": force_delta_loss,
        "context_region_loss": context_region_loss,
    }
