"""Frozen-teacher spatial supervision for RenderNet's optional region head.

Reuses the video's equal-residual attention rollout. This is a spatial prior
at DINO patch resolution, not a pixel attribution or a causal importance map.
"""

import math
import time

import torch
from torch.nn import functional as F

from diffusion_policy.scripts.dit_action_attention_capture import DitActionAttentionCapture
from diffusion_policy.scripts.fusion_visual_attention_trace import FusionVisualAttentionTrace
from diffusion_policy.scripts.visualize_cross_attention_episode import patch_grid


def validate_percentiles(percentiles):
    if len(percentiles) != 2:
        raise ValueError("percentiles must contain [lower, upper]")
    low, high = map(float, percentiles)
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high <= 100):
        raise ValueError("percentiles require 0 <= lower < upper <= 100")
    return low, high


def percentile_region(patch_attention, percentiles):
    """Soft weights per camera/observation, quantiles over the last (patch) axis.

    Below q_low => 0; above q_high => 1; linear between. If the quantile
    interval collapses on a sparse map, use its actual range. A truly constant
    map carries no spatial preference and leaves the whole image available.
    """
    low, high = validate_percentiles(percentiles)
    values = patch_attention.detach().float()
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("Teacher patch attention must be finite and nonnegative")
    bounds = torch.quantile(values, values.new_tensor([low, high]) / 100, dim=-1, keepdim=True)
    lower, upper = bounds[0], bounds[1]
    collapsed = upper <= lower
    lower = torch.where(collapsed, values.amin(-1, keepdim=True), lower)
    upper = torch.where(collapsed, values.amax(-1, keepdim=True), upper)
    span = upper - lower
    weights = ((values - lower) / span.clamp_min(torch.finfo(values.dtype).tiny)).clamp(0, 1)
    return torch.where(span > 0, weights, torch.ones_like(weights))


def attention_regions(patch_attention, images, backbone, percentiles):
    """Convert raw cached [B,To,P] patch percentages to RGB-sized soft masks."""
    masks = {}
    for key, attention in patch_attention.items():
        batch, steps, patches = attention.shape
        height, width = images[key].shape[-2:]
        gh, gw = patch_grid(patches, height, width, backbone)
        weights = percentile_region(attention, percentiles)
        masks[key] = F.interpolate(weights.reshape(batch * steps, 1, gh, gw),
                                   size=(height, width), mode="bilinear", align_corners=False
                                   ).reshape(batch, steps, 1, height, width).to(images[key].device)
    return masks


@torch.no_grad()
def teacher_attention_targets(teacher, obs, *, percentiles, seed=42, progress_callback=None,
                              return_patch_attention=False):
    """Return encoder token targets and teacher-derived pixel masks for a batch.

    Capture all samples and denoising calls together, retaining attention on
    the teacher device until the batch is complete. Generated actions are
    discarded. The student path never executes this function at inference.
    """
    validate_percentiles(percentiles)
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError("Attention supervision requires a frozen eval teacher")
    batch = next(iter(obs.values())).shape[0]
    started = time.monotonic()
    if progress_callback is not None:
        progress_callback(0, batch, 0.0)
    # Keep the exact ordinary full-batch teacher target. Requesting attention
    # weights changes the MHA kernel (noticeably under AMP), so the captured
    # encoder output must not replace the original distillation target.
    targets = teacher._encode_context(obs, apply_curriculum_mask=False).detach()
    with FusionVisualAttentionTrace(teacher, batched=True) as fusion, \
            DitActionAttentionCapture(teacher, action_query="executed", batched=True) as dit:
        tokens = teacher._encode_context(obs, apply_curriculum_mask=False).detach()
        # The serial implementation restarted the same seed for every sample.
        # Independent generators retain that noise, including stochastic steps.
        generators = [torch.Generator(device=tokens.device).manual_seed(int(seed)) for _ in range(batch)]
        teacher.conditional_sample(
            (batch, teacher.horizon, teacher.action_dim), context=tokens, generator=generators,
        )
        maps, diagnostics = fusion.project(dit.step_context_relevance())
    total = diagnostics["input_mass"]
    if not torch.isfinite(total).all() or (total <= 0).any():
        raise RuntimeError("Teacher rollout has invalid context mass")
    # Normalize each denoising call before averaging, independently per sample.
    patches = {key: (100 * values / total[..., None, None]).mean(0).detach()
               for key, values in maps.items()}
    if progress_callback is not None:
        progress_callback(batch, batch, time.monotonic() - started)
    if return_patch_attention:
        return targets, patches
    return targets, attention_regions(patches, obs, teacher.obs_encoder.backbone, percentiles)
