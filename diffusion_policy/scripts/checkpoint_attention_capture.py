"""Capture real attention from the architecture stored in a policy checkpoint.

Maps retain the model's softmax mass (no spatial renormalization). Wrench/image
fusion returns [wrench positions, camera patches]; pure visual attention returns
[1, camera patches]. Only the latest observation of one sample is displayed.
"""

import inspect
import operator

import torch
from torch import nn

from diffusion_policy.scripts.image_query_attention_capture import ImageQueryAttentionCapture


class AttentionCapture:
    """Temporarily request attention probabilities without changing parameters.

    ``mode`` is ``auto``, ``cross-attn``, ``self-attn`` (the fusion SA), or
    ``dino`` (backbone SA). Auto prefers CA, then fusion SA, then DINO. A pure
    DINO baseline also accepts self-attn as its only self-attention source.
    Layer indices are zero based, with -1 selecting the final available layer.
    ``query_mode='last'`` selects the last wrench position, or the last patch
    query for image-only SA. It never changes the original softmax denominator.
    """

    def __init__(self, policy, mode="auto", layer_index=-1, query_mode="all"):
        if query_mode not in {"all", "last"}:
            raise ValueError("query_mode must be 'all' or 'last'")
        self.encoder = policy.obs_encoder
        self.observation_steps = int(policy.n_obs_steps)
        self.query_mode = query_mode
        self.image_keys, self.patch_counts = self._camera_layout(policy)
        self.camera_attention_mode = getattr(self.encoder, "camera_attention_mode", "joint")
        self.baseline_mode = getattr(self.encoder, "mode", None)
        self._is_force = hasattr(self.encoder, "pre_cross_self_attention")
        cross_count = int(getattr(self.encoder, "cross_attn_layers", 0))
        fusion = (getattr(self.encoder, "pre_cross_self_attention", None)
                  if self._is_force else getattr(self.encoder, "fusion", None))
        fusion_layers = getattr(fusion, "layers", ())
        dino_layers = getattr(self.encoder.backbone, "layer", ())
        dino_available = bool(len(dino_layers)) and all(
            hasattr(layer, "attention") and hasattr(layer.attention, "config")
            for layer in dino_layers
        )
        self.available_modes = {}
        if cross_count:
            self.available_modes["cross-attn"] = cross_count
        if len(fusion_layers):
            self.available_modes["self-attn"] = len(fusion_layers)
        if dino_available:
            self.available_modes["dino"] = len(dino_layers)
        if mode == "auto":
            mode = next(iter(self.available_modes), None)
        elif mode == "self-attn" and self.baseline_mode == "dino":
            mode = "dino"
        if mode not in self.available_modes:
            available = ", ".join(f"{name} ({count} layers)" for name, count in self.available_modes.items())
            raise ValueError(f"Attention mode {mode!r} is unavailable; available: {available or 'none'}")
        self.mode = mode
        self.layer_count = self.available_modes[mode]
        try:
            layer_index = operator.index(layer_index)
        except TypeError as exc:
            raise ValueError("layer_index must be an integer") from exc
        if not -self.layer_count <= layer_index < self.layer_count:
            raise ValueError(
                f"layer_index {layer_index} is unavailable for {mode}; available layers: "
                f"0..{self.layer_count - 1} (or {-self.layer_count}..-1)"
            )
        self.layer_index = layer_index % self.layer_count
        if mode == "cross-attn":
            blocks = (self.encoder, *self.encoder.extra_cross_attention_layers)
            self.attention = blocks[self.layer_index].cross_attention
            self.query_source = self.encoder.cross_attn_query
            self.temporal_layout = "per_observation"
        elif mode == "self-attn":
            self.attention = fusion_layers[self.layer_index].self_attn
            self.query_source = "wrench" if self._is_force or self.baseline_mode == "self_attn" else "image"
            self.temporal_layout = "flattened_observations" if self.baseline_mode == "self_attn" else "per_observation"
        else:
            self.attention = dino_layers[self.layer_index].attention
            self.query_source = "image"
            self.temporal_layout = "per_camera_observation"
        if mode != "dino" and not isinstance(self.attention, nn.MultiheadAttention):
            raise ValueError("Selected attention must be torch.nn.MultiheadAttention")
        self.has_updates = mode == "cross-attn" and self.query_source == "image"
        force_encoder = getattr(self.encoder, "force_encoder", None)
        force_layers = (int(force_encoder.num_layers)
                        if isinstance(force_encoder, (nn.GRU, nn.LSTM)) else
                        len(getattr(force_encoder, "layers", ())))
        self.architecture = {
            "encoder": type(self.encoder).__name__,
            "baseline_mode": self.baseline_mode,
            "force_encoder": getattr(self.encoder, "force_temporal_encoder_type", None),
            "cross_attn_query": getattr(self.encoder, "cross_attn_query", None),
            "force_encoder_layers": force_layers,
            "pre_cross_self_attn_layers": len(fusion_layers) if self._is_force else 0,
            "cross_attn_layers": cross_count,
            "fusion_self_attn_layers": len(fusion_layers),
            "dino_self_attn_layers": len(dino_layers) if dino_available else 0,
            "wrench_history_steps": int(self.encoder.wrench_history_steps) if self.query_source == "wrench" or self._is_force or self.baseline_mode == "self_attn" else 0,
            "image_patch_counts": dict(self.patch_counts),
            "n_obs_steps": self.observation_steps,
            "selected_heads": int(self.attention.num_heads),
            "camera_attention_mode": self.camera_attention_mode,
            "available_modes": dict(self.available_modes),
            "mode": self.mode,
            "layer_index": self.layer_index,
            "layer_count": self.layer_count,
            "query_source": self.query_source,
            "temporal_layout": self.temporal_layout,
        }
        self.handles = []
        self.active = False
        self._backend = None
        self._backend_changed = False
        self._attention_maps = {}
        self._update_maps = {}
        self._call_index = 0
        self._groups = []

    def _camera_layout(self, policy):
        shapes = dict(getattr(self.encoder, "image_shapes", {}))
        shape_meta = getattr(policy, "shape_meta", {}) or {}
        for key, spec in shape_meta.get("obs", {}).items():
            if spec.get("type", "low_dim") == "rgb":
                shapes.setdefault(key, tuple(spec["shape"]))
        keys = tuple(getattr(self.encoder, "image_keys", ()) or getattr(policy, "image_keys", ()) or sorted(shapes))
        counts = dict(getattr(self.encoder, "patch_counts", {}))
        patch_size = getattr(getattr(self.encoder.backbone, "config", None), "patch_size", None)
        for key in keys:
            if key in counts:
                continue
            if key not in shapes or patch_size is None:
                raise ValueError(f"Cannot derive DINO patch count for camera {key!r} from checkpoint shape_meta")
            patch_h, patch_w = ((int(patch_size), int(patch_size)) if isinstance(patch_size, int) else map(int, patch_size))
            shape = shapes[key]
            if len(shape) != 3 or patch_h < 1 or patch_w < 1 or shape[-2] % patch_h or shape[-1] % patch_w:
                raise ValueError(f"Invalid DINO patch grid for camera {key!r}")
            counts[key] = (shape[-2] // patch_h) * (shape[-1] // patch_w)
        if not keys or any(int(counts[key]) < 1 for key in keys):
            raise ValueError("Attention visualization requires configured image cameras and patch counts")
        return keys, {key: int(counts[key]) for key in keys}

    def __enter__(self):
        if self.active:
            raise RuntimeError("Attention capture is already active")
        self.active = True
        self._attention_maps, self._update_maps = {}, {}
        self._call_index = 0
        try:
            self.handles.append(self.encoder.register_forward_pre_hook(self._reset, with_kwargs=True))
            request = self._request_dino if self.mode == "dino" else ImageQueryAttentionCapture._request_weights
            self.handles.append(self.attention.register_forward_pre_hook(request, with_kwargs=True))
            self.handles.append(self.attention.register_forward_hook(self._capture, always_call=True))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc_info):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self._restore_backend()
        self.active = False

    def _reset(self, module, args, kwargs):
        self._attention_maps, self._update_maps = {}, {}
        self._call_index = 0
        inputs = inspect.signature(module.forward).bind_partial(*args, **kwargs).arguments
        images = inputs.get("images")
        if images is None:
            image = inputs.get("image")
            if image is None or len(self.image_keys) != 1:
                raise ValueError("Episode visualization requires configured image observations")
            images = {self.image_keys[0]: image}
        groups = {}
        for key in self.image_keys:
            image = images[key]
            if image.ndim != 5 or image.shape[:2] != (1, self.observation_steps):
                raise ValueError("Episode visualization requires one sample with n_obs_steps images")
            signature = ((tuple(image.shape[2:]), image.dtype, image.device)
                         if self._is_force else tuple(image.shape[2:]))
            groups.setdefault(signature, []).append(key)
        self._groups = list(groups.values())

    def _request_dino(self, module, args, kwargs):
        self._backend = module.config._attn_implementation
        self._backend_changed = True
        module.config._attn_implementation = "eager"

    def _restore_backend(self):
        if self._backend_changed:
            self.attention.config._attn_implementation = self._backend
            self._backend = None
            self._backend_changed = False

    def _capture(self, module, args, output):
        if self.mode == "dino":
            self._restore_backend()
        if output is None:
            return
        if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
            raise RuntimeError("Selected attention layer did not return attention probabilities")
        attended, weights = output[:2]
        if weights.ndim != 4:
            raise RuntimeError("Expected attention probabilities [batch, heads, queries, keys]")
        if self.mode == "dino":
            self._capture_dino(weights)
        elif self.mode == "cross-attn":
            self._capture_cross(attended, weights)
        else:
            self._capture_self(weights)

    def _wrench_maps(self, value):
        # Input [heads, wrench_positions, camera_patches]. Keep each temporal
        # wrench position and average only heads, preserving the softmax mass.
        if self.query_mode == "last":
            value = value[:, -1:, :]
        return value.detach().float().mean(0).cpu().contiguous()

    def _image_map(self, value):
        if self.query_mode == "last":
            value = value[:, -1:, :]
        return value.detach().float().mean((0, 1)).cpu().unsqueeze(0).contiguous()

    def _capture_cross(self, attended, weights):
        if weights.shape[0] != self.observation_steps or attended.ndim != 3:
            raise RuntimeError("Unexpected cross-attention observation layout")
        if self.camera_attention_mode == "joint":
            if self._call_index:
                raise RuntimeError("Unexpected extra joint cross-attention call")
            cameras = self.image_keys
        elif self.camera_attention_mode == "per_camera":
            if self._call_index >= len(self.image_keys):
                raise RuntimeError("Unexpected extra per-camera cross-attention call")
            cameras = (self.image_keys[self._call_index],)
        else:
            raise ValueError("Unsupported camera_attention_mode")
        self._call_index += 1
        patches = sum(self.patch_counts[key] for key in cameras)
        wrench = int(self.encoder.wrench_history_steps)
        expected = (patches, wrench) if self.query_source == "image" else (wrench, patches)
        if weights.shape[-2:] != expected:
            raise RuntimeError(f"Cross-attention shape {tuple(weights.shape[-2:])} does not match {expected}")
        latest = weights[-1]
        offset = 0
        for key in cameras:
            end = offset + self.patch_counts[key]
            value = latest[:, offset:end].transpose(1, 2) if self.query_source == "image" else latest[:, :, offset:end]
            self._attention_maps[key] = self._wrench_maps(value)
            if self.has_updates:
                self._update_maps[key] = torch.linalg.vector_norm(
                    attended[-1, offset:end].detach().float(), ord=2, dim=-1
                ).cpu().contiguous()
            offset = end

    def _capture_self(self, weights):
        if self._call_index:
            raise RuntimeError("Unexpected extra fusion self-attention call")
        self._call_index += 1
        patches = sum(self.patch_counts.values())
        wrench = int(self.encoder.wrench_history_steps) if self.query_source == "wrench" else 0
        per_step = patches + wrench
        if self.temporal_layout == "flattened_observations":
            expected_batch, expected_tokens = 1, self.observation_steps * per_step
            start = (self.observation_steps - 1) * per_step
        else:
            expected_batch, expected_tokens = self.observation_steps, per_step
            start = 0
        if weights.shape[0] != expected_batch or weights.shape[-2:] != (expected_tokens, expected_tokens):
            raise RuntimeError("Unexpected fusion self-attention token layout")
        latest = weights[-1]
        queries = slice(start + patches, start + per_step) if wrench else slice(start, start + patches)
        offset = start
        for key in self.image_keys:
            end = offset + self.patch_counts[key]
            value = latest[:, queries, offset:end]
            self._attention_maps[key] = self._wrench_maps(value) if wrench else self._image_map(value)
            offset = end

    def _capture_dino(self, weights):
        if self._call_index >= len(self._groups):
            raise RuntimeError("Unexpected extra DINO self-attention call")
        cameras = self._groups[self._call_index]
        self._call_index += 1
        if weights.shape[0] != len(cameras) * self.observation_steps:
            raise RuntimeError("Unexpected DINO camera/observation batch layout")
        prefix = 1 + self.encoder.num_register_tokens
        for camera_index, key in enumerate(cameras):
            patches = self.patch_counts[key]
            if weights.shape[-2:] != (prefix + patches, prefix + patches):
                raise RuntimeError(f"DINO attention does not match patch grid for camera {key!r}")
            index = camera_index * self.observation_steps + self.observation_steps - 1
            self._attention_maps[key] = self._image_map(weights[index, :, prefix:, prefix:])

    def _require_complete(self):
        expected_calls = (len(self._groups) if self.mode == "dino" else
                          len(self.image_keys) if self.mode == "cross-attn" and self.camera_attention_mode == "per_camera" else 1)
        if self._call_index != expected_calls or set(self._attention_maps) != set(self.image_keys):
            raise RuntimeError("No complete attention forward pass was captured")

    def attention_by_camera(self):
        self._require_complete()
        return dict(self._attention_maps)

    def update_by_camera(self):
        if not self.has_updates:
            raise ValueError("Spatial CA update magnitude requires image-query cross-attention")
        self._require_complete()
        return dict(self._update_maps)
