"""Capture image-query/wrench-key attention without changing model parameters.

The returned attention maps keep the original softmax over wrench keys.  The
update maps are channel-wise L2 norms of MultiheadAttention's projected output,
before the encoder adds the image residual or applies its feed-forward block.
Both maps use the latest observation of a single episode sample.
"""

import inspect
import operator

import torch
import torch.nn as nn


class ImageQueryAttentionCapture:
    """Temporarily capture one image-query cross-attention layer.

    ``attention_by_camera()`` returns ``{camera: [wrench_keys, patches]}`` and
    ``update_by_camera()`` returns ``{camera: [patches]}``.  All results are
    detached CPU float32 tensors.  ``layer_index`` follows Python indexing,
    with ``-1`` selecting the final cross-attention layer.

    Use an eval-mode policy for inference.  Hooks only request the existing
    attention probabilities; they do not alter parameters or the encoder's
    legacy attention-capture flag.
    """

    def __init__(self, policy, layer_index=-1):
        self.encoder = policy.obs_encoder
        if getattr(self.encoder, "cross_attn_query", None) != "image":
            raise ValueError("Image-query visualization requires cross_attn_query='image'")
        layer_count = int(getattr(self.encoder, "cross_attn_layers", 0))
        if layer_count < 1:
            raise ValueError("Image-query visualization requires at least one cross-attention layer")
        try:
            layer_index = operator.index(layer_index)
        except TypeError as exc:
            raise ValueError("layer_index must be an integer") from exc
        if not -layer_count <= layer_index < layer_count:
            raise ValueError(f"layer_index must be between {-layer_count} and {layer_count - 1}")
        self.layer_index = layer_index % layer_count
        self.observation_steps = int(policy.n_obs_steps)
        self.image_keys = tuple(self.encoder.image_keys)
        self.patch_counts = {key: int(self.encoder.patch_counts[key]) for key in self.image_keys}
        if not self.image_keys or any(count < 1 for count in self.patch_counts.values()):
            raise ValueError("Image-query visualization requires configured camera patch counts")
        self.camera_attention_mode = self.encoder.camera_attention_mode
        if self.camera_attention_mode not in {"joint", "per_camera"}:
            raise ValueError("Unsupported camera_attention_mode")
        blocks = (self.encoder, *self.encoder.extra_cross_attention_layers)
        self.attention = blocks[self.layer_index].cross_attention
        if not isinstance(self.attention, nn.MultiheadAttention):
            raise ValueError("Cross-attention must be torch.nn.MultiheadAttention")
        self.handles = []
        self.active = False
        self._attention_maps = {}
        self._update_maps = {}
        self._call_index = 0

    def __enter__(self):
        if self.active:
            raise RuntimeError("Image-query attention capture is already active")
        self.active = True
        self._attention_maps = {}
        self._update_maps = {}
        self._call_index = 0
        try:
            self.handles.append(
                self.encoder.register_forward_pre_hook(self._reset, with_kwargs=True)
            )
            self.handles.append(
                self.attention.register_forward_pre_hook(self._request_weights, with_kwargs=True)
            )
            self.handles.append(self.attention.register_forward_hook(self._capture))
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
        self._attention_maps = {}
        self._update_maps = {}
        self._call_index = 0
        inputs = inspect.signature(module.forward).bind_partial(*args, **kwargs).arguments
        images = inputs.get("images")
        if images is None:
            image = inputs.get("image")
            if image is None or len(self.image_keys) != 1:
                raise ValueError("Episode visualization requires configured image observations")
            images = {self.image_keys[0]: image}
        for key in self.image_keys:
            image = images[key]
            if image.ndim != 5 or image.shape[:2] != (1, self.observation_steps):
                raise ValueError("Episode visualization requires one sample with n_obs_steps images")

    @staticmethod
    def _request_weights(module, args, kwargs):
        # MultiheadAttention permits these options both positionally and by
        # keyword.  Preserve the caller's other arguments in either form.
        args = list(args)
        kwargs = dict(kwargs)
        for position, name, value in (
            (4, "need_weights", True), (6, "average_attn_weights", False)
        ):
            if len(args) > position:
                args[position] = value
                kwargs.pop(name, None)
            else:
                kwargs[name] = value
        return tuple(args), kwargs

    def _capture(self, module, args, output):
        attended, weights = output
        if (
            weights is None or weights.ndim != 4 or attended.ndim != 3
            or weights.shape[0] != self.observation_steps
            or attended.shape[0] != self.observation_steps
            or weights.shape[-1] != self.encoder.wrench_history_steps
            or weights.shape[-2] != attended.shape[-2]
        ):
            raise RuntimeError("Unexpected image-query cross-attention shape")
        if self.camera_attention_mode == "joint":
            if self._call_index:
                raise RuntimeError("Unexpected extra joint cross-attention call")
            cameras = self.image_keys
        else:
            if self._call_index >= len(self.image_keys):
                raise RuntimeError("Unexpected extra per-camera cross-attention call")
            cameras = (self.image_keys[self._call_index],)
        self._call_index += 1
        if attended.shape[1] != sum(self.patch_counts[key] for key in cameras):
            raise RuntimeError("Cross-attention query count does not match camera patch counts")

        # Average only heads, retaining every wrench key and image position.
        # The MHA output has already concatenated heads and applied out_proj;
        # its norm must be taken across channels rather than attention heads.
        last_weights = weights[-1].detach().float().mean(dim=0).cpu()
        last_updates = torch.linalg.vector_norm(
            attended[-1].detach().float(), ord=2, dim=-1
        ).cpu()
        offset = 0
        for key in cameras:
            end = offset + self.patch_counts[key]
            self._attention_maps[key] = last_weights[offset:end].transpose(0, 1).contiguous()
            self._update_maps[key] = last_updates[offset:end].contiguous()
            offset = end

    def _require_complete(self):
        expected_calls = 1 if self.camera_attention_mode == "joint" else len(self.image_keys)
        if self._call_index != expected_calls or set(self._attention_maps) != set(self.image_keys):
            raise RuntimeError("No complete image-query cross-attention forward pass was captured")

    def attention_by_camera(self):
        self._require_complete()
        return dict(self._attention_maps)

    def update_by_camera(self):
        self._require_complete()
        return dict(self._update_maps)
