"""Trace condition-token attention back to the fusion input's DINO patches.

This is residual-aware attention rollout, not a gradient or causal attribution.
Heads are averaged; SA uses RowNormalize(I + A), and each CA block assigns
half of its relevance to its query residual and half to its attention memory.
Position-wise projections, layer norms and FFNs preserve token positions in
this approximation. DINO's internal mixing and the force temporal encoder are
outside the trace: visual maps stop at DINO output patches, and unresolved
wrench relevance is reported separately rather than assigned to image patches.
"""

import inspect

import torch
from torch import nn

from diffusion_policy.model.vision.dino_token_baseline_obs_encoder import DinoTokenBaselineObsEncoder
from diffusion_policy.model.vision.force_aware_obs_encoder import ForceAwareObsEncoder
from diffusion_policy.scripts.image_query_attention_capture import ImageQueryAttentionCapture


class FusionVisualAttentionTrace:
    """Capture all fusion attention during one ordinary encoder forward pass.

    ``project(relevance)`` accepts nonnegative CPU/GPU tensors ``[..., S]`` for
    the S output condition tokens. It returns ``(maps, diagnostics)`` where
    each camera map has shape ``[..., n_obs_steps, camera_patch_count]``.
    No spatial renormalization is applied: maps plus unresolved wrench mass
    conserve input relevance. Use an eval-mode encoder and one policy sample.

    Legacy per-camera wrench-query fusion uses a learned channel projection.
    Its cross-camera relevance split cannot be recovered from attention alone;
    this utility rejects that case by default. Explicitly request
    ``camera_fusion_approximation='equal'`` to use equal camera weights.
    """

    def __init__(self, policy, *, camera_fusion_approximation="error"):
        if camera_fusion_approximation not in {"error", "equal"}:
            raise ValueError("camera_fusion_approximation must be 'error' or 'equal'")
        self.encoder = policy.obs_encoder
        self.observation_steps = int(policy.n_obs_steps)
        if self.observation_steps < 1:
            raise ValueError("n_obs_steps must be positive")
        self._is_force = isinstance(self.encoder, ForceAwareObsEncoder)
        if not self._is_force and not isinstance(self.encoder, DinoTokenBaselineObsEncoder):
            raise ValueError(
                "Fusion rollout supports ForceAwareObsEncoder and DinoTokenBaselineObsEncoder; "
                f"got {type(self.encoder).__name__}"
            )
        self.image_keys, self.patch_counts = self._camera_layout(policy)
        self.image_token_count = sum(self.patch_counts.values())
        self.baseline_mode = getattr(self.encoder, "mode", None)
        self.wrench_count = int(self.encoder.wrench_history_steps) if (
            self._is_force or self.baseline_mode == "self_attn"
        ) else 0
        self.cross_query = getattr(self.encoder, "cross_attn_query", None)
        self.camera_mode = getattr(self.encoder, "camera_attention_mode", "joint")
        fusion = self.encoder.pre_cross_self_attention if self._is_force else self.encoder.fusion
        self.self_attention = [layer.self_attn for layer in getattr(fusion, "layers", ())]
        blocks = (self.encoder, *self.encoder.extra_cross_attention_layers) if self._is_force else ()
        self.cross_attention = [block.cross_attention for block in blocks] if (
            self._is_force and self.encoder.cross_attn_layers
        ) else []
        for attention in (*self.self_attention, *self.cross_attention):
            if not isinstance(attention, nn.MultiheadAttention):
                raise ValueError("Fusion rollout requires torch.nn.MultiheadAttention modules")
        if self.cross_attention:
            per_step = self.image_token_count if self.cross_query == "image" else self.wrench_count
        else:
            per_step = self.image_token_count + self.wrench_count
        self.context_tokens = self.observation_steps * per_step
        self.assumptions = [
            "Attention heads are averaged; SA uses RowNormalize(I+A).",
            "Each CA block splits relevance equally between query residual and attention memory.",
            "Tokenwise projections, normalization and FFNs do not change token routing in this approximation.",
            "The spatial boundary is DINO output patches, not raw input pixels or DINO internal attention.",
            "Remaining force-encoder token relevance is reported as nonvisual wrench mass.",
        ]
        self.equal_camera_approximation = bool(
            self.cross_attention and self.cross_query == "wrench" and self.camera_mode == "per_camera"
            and len(self.image_keys) > 1
        )
        if self.equal_camera_approximation:
            if camera_fusion_approximation != "equal":
                raise ValueError(
                    "Per-camera wrench-query fusion uses a learned camera_fusion Linear: "
                    "attention alone cannot recover its camera contributions. "
                    "Explicitly select camera_fusion_approximation='equal' to use an equal camera split."
                )
            self.assumptions.append(
                "Per-camera wrench-query outputs use a learned channel fusion; "
                "attention rollout approximates this projection by an equal camera split."
            )
        self.architecture = {
            "encoder": type(self.encoder).__name__,
            "baseline_mode": self.baseline_mode,
            "cross_attn_query": self.cross_query,
            "camera_attention_mode": self.camera_mode,
            "fusion_self_attn_layers": len(self.self_attention),
            "cross_attn_layers": len(self.cross_attention),
            "n_obs_steps": self.observation_steps,
            "image_patch_counts": dict(self.patch_counts),
            "wrench_history_steps": self.wrench_count,
            "context_tokens": self.context_tokens,
            "spatial_boundary": "DINO output patch tokens at the fusion input",
            "equal_camera_fusion_approximation": self.equal_camera_approximation,
            "assumptions": list(self.assumptions),
        }
        self.handles = []
        self.active = False
        self._clear()

    def _camera_layout(self, policy):
        shapes = dict(getattr(self.encoder, "image_shapes", {}))
        for key, spec in (getattr(policy, "shape_meta", {}) or {}).get("obs", {}).items():
            if spec.get("type", "low_dim") == "rgb":
                shapes.setdefault(key, tuple(spec["shape"]))
        keys = tuple(self.encoder.image_keys or getattr(policy, "image_keys", ()) or sorted(shapes))
        counts = dict(self.encoder.patch_counts)
        patch_size = getattr(self.encoder.backbone.config, "patch_size", None)
        for key in keys:
            if key in counts:
                continue
            if key not in shapes or patch_size is None:
                raise ValueError(f"Cannot derive DINO patch count for camera {key!r}")
            ph, pw = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
            shape = shapes[key]
            if len(shape) != 3 or ph < 1 or pw < 1 or shape[-2] % ph or shape[-1] % pw:
                raise ValueError(f"Invalid DINO patch grid for camera {key!r}")
            counts[key] = shape[-2] // ph * (shape[-1] // pw)
        if not keys or any(int(counts[key]) < 1 for key in keys):
            raise ValueError("Fusion rollout requires configured camera patch counts")
        return keys, {key: int(counts[key]) for key in keys}

    def _clear(self):
        self._self_weights = [[] for _ in self.self_attention]
        self._cross_weights = [[] for _ in self.cross_attention]
        self._complete = False

    def __enter__(self):
        if self.active:
            raise RuntimeError("Fusion attention trace is already active")
        if self.encoder.training:
            raise ValueError("Fusion attention rollout requires an eval-mode encoder")
        self.active = True
        self._clear()
        try:
            self.handles.append(self.encoder.register_forward_pre_hook(self._reset, with_kwargs=True))
            for kind, modules in (("self", self.self_attention), ("cross", self.cross_attention)):
                for index, module in enumerate(modules):
                    self.handles.append(module.register_forward_pre_hook(
                        ImageQueryAttentionCapture._request_weights, with_kwargs=True
                    ))
                    self.handles.append(module.register_forward_hook(self._capture_hook(kind, index)))
            self.handles.append(self.encoder.register_forward_hook(self._finish, always_call=True))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc_info):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.active = False

    def _reset(self, module, args, kwargs):
        self._clear()
        bound = inspect.signature(module.forward).bind_partial(*args, **kwargs).arguments
        images = bound.get("images")
        if images is None:
            image = bound.get("image")
            if image is None or len(self.image_keys) != 1:
                raise ValueError("Fusion rollout requires configured image observations")
            images = {self.image_keys[0]: image}
        for key in self.image_keys:
            image = images[key]
            if image.ndim != 5 or tuple(image.shape[:2]) != (1, self.observation_steps):
                raise ValueError("Fusion rollout requires one sample with n_obs_steps images")

    def _capture_hook(self, kind, index):
        def capture(module, args, output):
            if not isinstance(output, tuple) or len(output) < 2:
                raise RuntimeError("Fusion attention did not return attention probabilities")
            weights = output[1]
            if weights is None or weights.ndim != 4:
                raise RuntimeError("Expected fusion attention [batch, heads, queries, keys]")
            if not torch.isfinite(weights).all() or (weights < 0).any():
                raise RuntimeError("Fusion attention contains invalid probabilities")
            store = self._self_weights if kind == "self" else self._cross_weights
            store[index].append(weights.detach().float().mean(1).cpu().clone())
        return capture

    def _finish(self, module, args, output):
        if output is None:
            return
        if output.ndim != 3 or tuple(output.shape[:2]) != (1, self.context_tokens):
            raise RuntimeError("Unexpected fusion output token layout")
        for calls in self._self_weights:
            if len(calls) != 1:
                raise RuntimeError("No complete fusion self-attention forward was captured")
        expected_calls = len(self.image_keys) if self.camera_mode == "per_camera" else 1
        for calls in self._cross_weights:
            if len(calls) != expected_calls:
                raise RuntimeError("No complete fusion cross-attention forward was captured")
        self._complete = True

    @staticmethod
    def _backward_self(relevance, weights):
        count = weights.shape[-1]
        if weights.shape[-2] != count or relevance.shape[-2:] != (weights.shape[0], count):
            raise RuntimeError("Self-attention and relevance token layouts do not match")
        transition = weights + torch.eye(count, dtype=weights.dtype).unsqueeze(0)
        transition = transition / transition.sum(-1, keepdim=True)
        return torch.einsum("...ti,tij->...tj", relevance, transition)

    def _cross_matrix(self, calls):
        if self.camera_mode == "joint":
            matrix = calls[0]
        elif self.cross_query == "image":
            matrix = torch.cat(calls, dim=1)
        else:
            matrix = torch.cat(calls, dim=2) / len(calls)
        expected = ((self.image_token_count, self.wrench_count) if self.cross_query == "image"
                    else (self.wrench_count, self.image_token_count))
        if matrix.shape != (self.observation_steps, *expected):
            raise RuntimeError("Unexpected cross-attention token layout")
        return matrix

    def project(self, context_relevance):
        """Propagate ``[..., context_tokens]`` row weights to camera patches."""
        if not self._complete:
            raise RuntimeError("No complete fusion forward pass was captured")
        relevance = torch.as_tensor(context_relevance).detach().to(device="cpu", dtype=torch.float32)
        if relevance.ndim < 1 or relevance.shape[-1] != self.context_tokens:
            raise ValueError(f"context_relevance must end in {self.context_tokens} condition tokens")
        if not torch.isfinite(relevance).all() or (relevance < 0).any():
            raise ValueError("context_relevance must contain finite nonnegative values")
        leading = relevance.shape[:-1]
        input_mass = relevance.sum(-1)
        steps, patches, wrench = self.observation_steps, self.image_token_count, self.wrench_count
        if self.cross_attention:
            query_count = patches if self.cross_query == "image" else wrench
            memory_count = wrench if self.cross_query == "image" else patches
            query = relevance.reshape(*leading, steps, query_count)
            memory = relevance.new_zeros(*leading, steps, memory_count)
            for calls in reversed(self._cross_weights):
                matrix = self._cross_matrix(calls)
                memory = memory + 0.5 * torch.einsum("...tq,tqk->...tk", query, matrix)
                query = query * 0.5
            image, force = (query, memory) if self.cross_query == "image" else (memory, query)
            joint = torch.cat((image, force), dim=-1)
        else:
            joint = relevance.reshape(*leading, steps, patches + wrench)
        if self.baseline_mode == "self_attn":
            joint = joint.reshape(*leading, 1, steps * (patches + wrench))
        for calls in reversed(self._self_weights):
            joint = self._backward_self(joint, calls[0])
        joint = joint.reshape(*leading, steps, patches + wrench)
        image, force = joint[..., :patches], joint[..., patches:]
        maps = {}
        offset = 0
        for key in self.image_keys:
            end = offset + self.patch_counts[key]
            maps[key] = image[..., offset:end].contiguous()
            offset = end
        visual_mass = image.sum((-2, -1))
        wrench_mass = force.sum((-2, -1))
        diagnostics = {
            "input_mass": input_mass,
            "visual_mass": visual_mass,
            "wrench_mass": wrench_mass,
            "nonvisual_wrench_mass": wrench_mass,
            "per_observation_visual_mass": image.sum(-1),
            "per_observation_wrench_mass": force.sum(-1),
            "mass_conservation_error": (visual_mass + wrench_mass - input_mass).abs(),
            "equal_camera_fusion_approximation": self.equal_camera_approximation,
            "assumptions": list(self.assumptions),
        }
        return maps, diagnostics
