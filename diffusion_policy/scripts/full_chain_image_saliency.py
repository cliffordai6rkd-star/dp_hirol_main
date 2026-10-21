"""Local input-image sensitivity of a final diffusion-policy output.

One differentiable ``predict_action`` forward builds the complete sampling
chain. VJPs then measure its final selected output's local Jacobian with respect
to raw input RGB values. This is sensitivity, not attention or causal attribution.
"""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import operator
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import torch.nn as nn


@dataclass
class SaliencyResult:
    """Detached prediction, CPU pixel maps, and JSON-compatible diagnostics."""

    prediction: Any
    maps: dict
    diagnostics: dict


def _detach_tree(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, Mapping):
        return {key: _detach_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    return value


@contextmanager
def _input_gradient_model(policy):
    """Freeze weights but enable the image path; restore even mixed flags."""
    modules = tuple(policy.modules())
    training = tuple(module.training for module in modules)
    parameters = tuple(policy.parameters())
    requires_grad = tuple(parameter.requires_grad for parameter in parameters)
    encoder = getattr(policy, "obs_encoder", None)
    frozen = [(module, module.freeze_backbone) for module in encoder.modules()
              if hasattr(module, "freeze_backbone")] if isinstance(encoder, nn.Module) else []
    try:
        policy.eval()
        for parameter in parameters:
            parameter.requires_grad_(False)
        for module, _ in frozen:
            # These flags normally wrap a frozen DINO forward in no_grad.
            # The weights stay frozen; only gradients to images are enabled.
            module.freeze_backbone = False
        yield
    finally:
        for module, value in frozen:
            module.freeze_backbone = value
        for parameter, value in zip(parameters, requires_grad):
            parameter.requires_grad_(value)
        # Setting individual flags preserves intentionally mixed train/eval
        # state without invoking custom train() methods or changing buffers.
        for module, value in zip(modules, training):
            module.training = value


def full_chain_image_saliency(
    policy: nn.Module,
    obs: Mapping,
    output_selector: Callable,
    *,
    image_keys: Optional[Sequence[str]] = None,
    estimator: str = "hutchinson",
    num_probes: int = 8,
    probe_seed: int = 0,
    generator: Optional[torch.Generator] = None,
    autocast_dtype: Optional[torch.dtype] = None,
) -> SaliencyResult:
    """Return local final-output sensitivity for each raw RGB input pixel.

    ``output_selector(policy, prediction, differentiable_obs)`` must return the
    differentiable final target, with any output-coordinate scaling made
    explicit by the caller. No target/pose-coordinate assumptions are made.

    For flattened target ``y`` with D coordinates, the camera map is

        map[b,t,h,w] = sqrt(sum_d sum_rgb (dy[d]/dI[b,t,rgb,h,w])**2 / D).

    ``exact`` computes every output-coordinate VJP. ``hutchinson`` estimates
    the squared norm using independent Rademacher probes: E[(J^T z)^2] sums
    squared Jacobian entries. The square root is applied after that estimate;
    the squared estimate is unbiased, its square root need not be. No averaging
    across denoising calls occurs. All calls remain in the same gradient graph.

    Probe randomness uses a fresh CPU generator, independently of the sampler
    ``generator`` and global RNG. Reusing ``probe_seed`` gives identical probes
    for identically shaped targets. Selected inputs are cloned leaves; all
    other observations are detached constants. Maps keep every observation
    time and are detached CPU float32 ``[B, To, H, W]`` tensors.

    If ``autocast_dtype`` is supplied, only the differentiable forward is run
    under device autocast (for example ``torch.bfloat16`` on CUDA). The VJPs
    and accumulated maps remain float32. This is an opt-in speed/precision
    tradeoff and is recorded in diagnostics.

    Weights, existing parameter gradients, train/eval flags, and frozen-backbone
    flags are preserved. ``no_grad`` callers are supported; ``inference_mode``
    callers must leave that context before requesting gradients.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Full-chain saliency requires gradients; leave torch.inference_mode() first")
    if not isinstance(policy, nn.Module):
        raise ValueError("policy must be an nn.Module")
    if not callable(output_selector):
        raise ValueError("output_selector must be callable")
    if estimator not in {"exact", "hutchinson"}:
        raise ValueError("estimator must be exact or hutchinson")
    if autocast_dtype is not None and not isinstance(autocast_dtype, torch.dtype):
        raise ValueError("autocast_dtype must be a torch.dtype or None")
    try:
        num_probes = operator.index(num_probes)
        probe_seed = operator.index(probe_seed)
    except TypeError as exc:
        raise ValueError("num_probes and probe_seed must be integers") from exc
    if num_probes < 1:
        raise ValueError("num_probes must be positive")
    image_keys = getattr(policy, "image_keys", ()) if image_keys is None else image_keys
    if isinstance(image_keys, str):
        raise ValueError("image_keys must be a sequence of camera keys")
    image_keys = tuple(image_keys)
    if not image_keys or len(set(image_keys)) != len(image_keys):
        raise ValueError("image_keys must contain distinct camera keys")
    inputs = _detach_tree(obs)
    images = []
    for key in image_keys:
        if key not in inputs:
            raise ValueError(f"Missing camera observation {key!r}")
        image = inputs[key]
        if (not torch.is_tensor(image) or not image.is_floating_point() or image.ndim != 5
                or image.shape[2] != 3 or any(size < 1 for size in image.shape)):
            raise ValueError(f"Camera {key!r} must be floating RGB [B, To, 3, H, W]")
        inputs[key] = image.detach().clone().requires_grad_(True)
        images.append(inputs[key])
    model_calls = []

    def record_call(module, args, kwargs):
        timestep = kwargs.get("timestep", args[1] if len(args) > 1 else None)
        if torch.is_tensor(timestep):
            values = timestep.detach().cpu().reshape(-1).tolist()
            timestep = values[0] if len(values) == 1 else values
        model_calls.append(timestep)

    model = getattr(policy, "model", None)
    handle = None
    connected = {key: False for key in image_keys}
    accumulated = [torch.zeros(image.shape[:2] + image.shape[-2:], device=image.device,
                               dtype=torch.float32) for image in images]
    # cuDNN RNN backward rejects eval-mode GRU/LSTM forwards. Native RNNs allow
    # input gradients while preserving the checkpoint's evaluation behavior.
    disable_cudnn_rnn = any(isinstance(module, nn.RNNBase) for module in policy.modules())
    try:
        autocast_context = (
            torch.autocast(device_type=images[0].device.type, dtype=autocast_dtype)
            if autocast_dtype is not None else nullcontext()
        )
        with torch.enable_grad(), _input_gradient_model(policy), torch.backends.cudnn.flags(
            enabled=False if disable_cudnn_rnn else torch.backends.cudnn.enabled
        ):
            if isinstance(model, nn.Module):
                handle = model.register_forward_pre_hook(record_call, with_kwargs=True)
            with autocast_context:
                prediction = policy.predict_action(inputs, generator=generator)
                target = output_selector(policy, prediction, inputs)
            if not torch.is_tensor(target) or not target.is_floating_point() or target.numel() < 1:
                raise ValueError("output_selector must return a nonempty floating tensor")
            if not target.requires_grad:
                raise RuntimeError("Selected output has no gradient graph; check detached sampling or no_grad inside the policy")
            if not torch.isfinite(target).all():
                raise RuntimeError("Selected output contains nonfinite values")
            target_shape = list(target.shape)
            flat = target.reshape(-1)
            target_count = flat.numel()
            iterations = target_count if estimator == "exact" else num_probes
            probe_rng = torch.Generator(device="cpu").manual_seed(probe_seed)
            for index in range(iterations):
                if estimator == "exact":
                    gradients = torch.autograd.grad(flat[index], images, retain_graph=index + 1 < iterations,
                                                    allow_unused=True)
                else:
                    probe = torch.randint(0, 2, (target_count,), generator=probe_rng, dtype=torch.int64)
                    probe = (probe * 2 - 1).to(device=flat.device, dtype=flat.dtype)
                    gradients = torch.autograd.grad(flat, images, grad_outputs=probe,
                                                    retain_graph=index + 1 < iterations, allow_unused=True)
                for camera_index, (key, gradient) in enumerate(zip(image_keys, gradients)):
                    if gradient is None:
                        continue
                    connected[key] = True
                    squared = gradient.detach().float().square().sum(dim=2)
                    if not torch.isfinite(squared).all():
                        raise RuntimeError(f"Nonfinite image gradients for camera {key!r}")
                    accumulated[camera_index] += squared
            denominator = target_count * (num_probes if estimator == "hutchinson" else 1)
            maps = {key: (value / denominator).sqrt().cpu()
                    for key, value in zip(image_keys, accumulated)}
            detached_prediction = _detach_tree(prediction)
    finally:
        if handle is not None:
            handle.remove()
    return SaliencyResult(
        prediction=detached_prediction,
        maps=maps,
        diagnostics={
            "method": "full_chain_local_image_jacobian_rms",
            "interpretation": "Local final-output sensitivity through the complete differentiable sampling chain; not attention or causal attribution.",
            "estimator": estimator,
            "num_probes": num_probes if estimator == "hutchinson" else None,
            "num_vjps": iterations,
            "probe_seed": probe_seed if estimator == "hutchinson" else None,
            "target_shape": target_shape,
            "target_count": target_count,
            "normalization": "sqrt(sum_output(sum_RGB(dtarget/dpixel)^2) / target_count)",
            "num_model_calls": len(model_calls),
            "timesteps": model_calls,
            "image_keys": list(image_keys),
            "connected_cameras": connected,
            "cudnn_disabled_for_eval_rnn_gradients": disable_cudnn_rnn,
            "autocast_dtype": None if autocast_dtype is None else str(autocast_dtype),
        },
    )
