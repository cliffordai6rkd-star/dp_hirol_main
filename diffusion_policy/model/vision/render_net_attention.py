"""Frozen-teacher spatial supervision for RenderNet's optional region head.

Reuses the video's equal-residual attention rollout. This is a spatial prior
at DINO patch resolution, not a pixel attribution or a causal importance map.
"""

import math
import time

import torch
from torch.nn import functional as F

from diffusion_policy.model.vision.render_net import top_percentile_mask
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


def percentile_region(patch_attention, percentiles, *, zero_is_zero=False):
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
    result = torch.where(span > 0, weights, torch.ones_like(weights))
    if zero_is_zero:
        result = torch.where(values.amax(-1, keepdim=True) > 0, result, torch.zeros_like(result))
    return result


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


def context_effect_regions(
    patch_effect,
    images,
    backbone,
    *,
    percentile=90.0,
    percentiles=(80.0, 99.0),
    selection="binary_top_percentile",
    strength_scale=None,
    normalization=None,
):
    """Map final-context effects to either soft or binary RGB region targets.

    ``normalization=image_minmax`` maps the weakest/strongest patches to 0/1
    independently for each sample, camera and observation. Constant maps have
    no spatial contrast and produce zero strength. Normalization precedes
    bilinear upsampling. With no normalization specified, preserve the saved
    experiment's absolute strength_scale or percentile mapping.
    """
    if normalization not in (None, "image_minmax"):
        raise ValueError(f"Unknown context normalization: {normalization!r}")
    masks = {}
    weights_by_camera = context_effect_patch_weights(
        patch_effect, selection=selection, percentile=percentile,
        percentiles=percentiles, strength_scale=strength_scale,
        normalization=normalization,
    )
    for key, values in weights_by_camera.items():
        height, width = images[key].shape[-2:]
        batch, steps, patches = values.shape
        gh, gw = patch_grid(patches, height, width, backbone)
        mode = "bilinear" if selection == "soft" else "nearest"
        masks[key] = F.interpolate(
            values.reshape(batch * steps, 1, gh, gw), size=(height, width),
            mode=mode, align_corners=False if mode == "bilinear" else None,
        ).reshape(batch, steps, 1, height, width).to(images[key].device)
    return masks


def context_effect_patch_weights(
    patch_effect,
    *,
    percentile=90.0,
    percentiles=(80.0, 99.0),
    selection="binary_top_percentile",
    strength_scale=None,
    normalization=None,
):
    """Map raw patch context effects to continuous or binary patch weights."""
    if normalization not in (None, "image_minmax"):
        raise ValueError(f"Unknown context normalization: {normalization!r}")
    weights_by_camera = {}
    for key, effect in patch_effect.items():
        values = effect.detach().float()
        if not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError("Context effects must be finite and nonnegative")
        if selection == "soft":
            if normalization == "image_minmax":
                lower = values.amin(dim=-1, keepdim=True)
                span = values.amax(dim=-1, keepdim=True) - lower
                # A constant image has zero numerator; use a safe denominator.
                weights = (values - lower) / torch.where(span > 0, span, torch.ones_like(span))
            elif strength_scale is not None:
                scale = float(strength_scale)
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError("strength_scale must be positive and finite")
                weights = values / (values + scale)
            else:
                weights = percentile_region(values, percentiles, zero_is_zero=True)
            mode = "bilinear"
        elif selection == "binary_top_percentile":
            weights = top_percentile_mask(values, percentile)
        else:
            raise ValueError(f"Unknown context region selection: {selection!r}")
        weights_by_camera[key] = weights
    return weights_by_camera


@torch.no_grad()
def teacher_context_effect_targets(teacher, obs, *, baseline="raw_zero"):
    """Exact final-fusion delta-Z magnitudes, with paired FP32 encoder calls.

    Returns ordinary normal tokens and raw [B, To, P] effects per camera. No
    DiT rollout or attention-kernel changes are needed. Keep raw magnitudes in
    the cache so selection thresholds can change without recomputation.
    """
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError("Context-effect supervision requires a frozen eval teacher")
    if baseline != "raw_zero":
        raise ValueError("Context-effect region supervision currently supports raw_zero")
    reference = dict(obs)
    # The Dataset label describes the original sample. Recompute it from the
    # zero-wrench counterfactual so its force-attention update is disabled.
    reference.pop("contact_alpha", None)
    keys = dict.fromkeys(getattr(teacher, name, None) for name in
                         ("wrench_key", "phase_wrench_key", "future_phase_wrench_key"))
    for key in keys:
        if key is not None:
            reference[key] = torch.zeros_like(obs[key])
    device = next(iter(obs.values())).device
    with torch.autocast(device_type=device.type, enabled=False):
        normal = teacher._encode_context(obs, apply_curriculum_mask=False).detach()
        # Re-enter shared preprocessing on raw zeros: both future labels and
        # the complete fusion gate must be recomputed for this counterfactual.
        zero = teacher._encode_context(reference, apply_curriculum_mask=False).detach()
        encoder = teacher.obs_encoder
        steps = int(teacher.n_obs_steps)
        patches = sum(encoder.patch_counts.values())
        if normal.ndim != 3 or normal.shape[1] != steps * patches or normal.shape != zero.shape:
            raise ValueError("Context-effect regions require an image-patch context layout")
        delta = torch.linalg.vector_norm((normal.float() - zero.float()).reshape(
            normal.shape[0], steps, patches, normal.shape[-1]), dim=-1)
        if not torch.isfinite(delta).all():
            raise ValueError("Non-finite context effects")
    maps, offset = {}, 0
    for key in encoder.image_keys:
        end = offset + int(encoder.patch_counts[key])
        maps[key] = delta[..., offset:end].detach()
        offset = end
    return normal, maps


def _query_slot_attention_maps(context_relevance, fusion):
    """Map residual-free image-query relevance to its image output slots.

    When the fusion CA queries image tokens and has no query residual, the
    ordinary value-path rollout intentionally reports zero image mass: image
    information enters through the query-dependent attention weights.  For
    RenderNet region supervision, the corresponding image-query slots are the
    only available spatial teacher signal, so assign each slot to its matching
    DINO patch before averaging denoising calls.
    """
    if not (fusion.cross_attention and fusion.cross_query == "image"
            and not fusion.cross_query_residual and not fusion.self_attention):
        raise ValueError("Query-slot maps require residual-free image-query cross-attention")
    relevance = torch.as_tensor(context_relevance).detach().float()
    if relevance.ndim < 2 or relevance.shape[-1] != fusion.context_tokens:
        raise ValueError(
            f"context_relevance must end in {fusion.context_tokens} tokens and include denoising calls"
        )
    if not fusion.batched:
        # Batched capture is required by teacher caching; retain a clear error
        # if this helper is accidentally used with a serial trace.
        raise ValueError("Query-slot teacher maps require batched attention capture")
    if relevance.shape[-2] != fusion._batch_size:
        raise ValueError("context_relevance batch does not match fusion trace")
    if not torch.isfinite(relevance).all() or (relevance < 0).any():
        raise ValueError("context_relevance must contain finite nonnegative values")
    total = relevance.sum(-1)
    if not torch.isfinite(total).all() or (total <= 0).any():
        raise ValueError("DiT context relevance must have positive finite mass")
    query = relevance.reshape(
        *relevance.shape[:-1], fusion.observation_steps, fusion.image_token_count,
    )
    scale = 100.0 / total[..., None, None]
    maps = {}
    offset = 0
    for key in fusion.image_keys:
        end = offset + fusion.patch_counts[key]
        maps[key] = query[..., offset:end] * scale
        offset = end
    return maps


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
        context_relevance = dit.step_context_relevance()
        query_slot_mode = bool(
            fusion.cross_attention and fusion.cross_query == "image"
            and not fusion.cross_query_residual and not fusion.self_attention
        )
        if query_slot_mode:
            maps = _query_slot_attention_maps(context_relevance, fusion)
        else:
            maps, diagnostics = fusion.project(context_relevance)
            total = diagnostics["input_mass"]
            if not torch.isfinite(total).all() or (total <= 0).any():
                raise RuntimeError("Teacher rollout has invalid context mass")
            # Normalize each denoising call before averaging, independently per sample.
            maps = {key: 100 * values / total[..., None, None] for key, values in maps.items()}
        # Normalize each denoising call before averaging, independently per sample.
        patches = {key: values.mean(0).detach() for key, values in maps.items()}
    if progress_callback is not None:
        progress_callback(batch, batch, time.monotonic() - started)
    if return_patch_attention:
        return targets, patches
    return targets, attention_regions(patches, obs, teacher.obs_encoder.backbone, percentiles)
