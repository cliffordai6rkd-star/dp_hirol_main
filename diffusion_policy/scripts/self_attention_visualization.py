"""Capture baseline self-attention during the normal policy forward pass.

No trainable modules or checkpoint keys are added. Only the final attention
layer is captured, and all hooks/backend changes are removed on context exit.
Spatial slices retain their original softmax mass; they are never renormalized.
"""

class _SelfAttentionCapture:
    def __init__(self, policy, *, query_mode, baseline_mode):
        if getattr(policy, "baseline_mode", None) != baseline_mode:
            config = (
                "train_dp_selfattn_insert_usb"
                if baseline_mode == "self_attn"
                else "train_dp_dinobase_insert_usb"
            )
            raise ValueError(
                f"This script requires a {config} checkpoint "
                f"(baseline_mode={baseline_mode!r}); use the matching visualization script "
                "for a cross-attention or other baseline checkpoint."
            )
        if query_mode not in {"all", "last"}:
            raise ValueError("attention query must be 'all' or 'last'")
        self.encoder = policy.obs_encoder
        self.observation_steps = int(policy.n_obs_steps)
        self.query_mode = query_mode
        self.maps = {}
        self.handles = []
        self.active = False

    def __enter__(self):
        if self.active:
            raise RuntimeError("Attention capture is already active")
        self.active = True
        try:
            self._install_hooks()
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

    def _restore_backend(self):
        pass

    def _reset(self, module, args, kwargs):
        self.maps = {}
        images = kwargs["images"]
        first = images[self.encoder.image_keys[0]]
        if first.shape[:2] != (1, self.observation_steps):
            raise ValueError("Episode visualization requires one sample with n_obs_steps images")

    def attention_by_camera(self):
        missing = set(self.encoder.image_keys) - self.maps.keys()
        if missing:
            raise RuntimeError(f"No self-attention captured for cameras {sorted(missing)}")
        return self.maps

    def _reduce(self, weights):
        # Input is [heads, selected_queries, image_patches].
        if self.query_mode == "last":
            weights = weights[:, -1:, :]
        return weights.detach().float().mean(dim=(0, 1)).cpu()


class WrenchImageSelfAttentionCapture(_SelfAttentionCapture):
    """Wrench-query/image-key block of the final joint fusion self-attention.

The baseline concatenates [camera_0 patches, camera_1 patches, wrench] for
each observation, then flattens observation/time into one token sequence.
Use wrench queries and image keys at the latest observation. The complete
softmax still includes both times, all cameras and wrench keys.
"""

    def __init__(self, policy, *, query_mode="all"):
        super().__init__(policy, query_mode=query_mode, baseline_mode="self_attn")
        self.attention = self.encoder.fusion.layers[-1].self_attn

    def _install_hooks(self):
        self.handles.append(
            self.encoder.register_forward_pre_hook(self._reset, with_kwargs=True)
        )
        # A hook on the layer's child also disables TransformerEncoderLayer's
        # fused eval path, which would otherwise bypass self_attn.forward().
        self.handles.append(
            self.attention.register_forward_pre_hook(self._request_weights, with_kwargs=True)
        )
        self.handles.append(self.attention.register_forward_hook(self._capture))

    @staticmethod
    def _request_weights(module, args, kwargs):
        kwargs = dict(kwargs, need_weights=True, average_attn_weights=False)
        return args, kwargs

    def _capture(self, module, args, output):
        weights = output[1]
        patches = sum(self.encoder.patch_counts.values())
        tokens_per_step = patches + self.encoder.wrench_history_steps
        total = self.observation_steps * tokens_per_step
        if (
            weights is None or weights.ndim != 4 or weights.shape[0] != 1
            or weights.shape[-2:] != (total, total)
        ):
            raise RuntimeError("Unexpected joint image/wrench self-attention shape")
        start = (self.observation_steps - 1) * tokens_per_step
        query = slice(start + patches, start + tokens_per_step)
        key_start = start
        for key in self.encoder.image_keys:
            key_end = key_start + self.encoder.patch_counts[key]
            self.maps[key] = self._reduce(weights[0, :, query, key_start:key_end])
            key_start = key_end


class ImageSelfAttentionCapture(_SelfAttentionCapture):
    """Patch-query/patch-key attention in the DINOv3 backbone's last layer.

The image-only baseline has no extra fusion self-attention. Capture the real
DINO attention, including its rotary positions, instead of inventing a new
patch-similarity layer. CLS/register tokens are excluded from the displayed
rows/columns without changing the original attention normalization.
"""

    def __init__(self, policy, *, query_mode="all"):
        super().__init__(policy, query_mode=query_mode, baseline_mode="dino")
        layers = getattr(self.encoder.backbone, "layer", None)
        if layers is None or not len(layers) or not hasattr(layers[-1], "attention"):
            raise ValueError("Image self-attention visualization requires a DINOv3 ViT backbone")
        self.attention = layers[-1].attention
        if not hasattr(self.attention, "config"):
            raise ValueError("DINO attention must expose its attention backend config")
        self.original_backend = None
        self.groups = []
        self.group_index = 0

    def _install_hooks(self):
        self.handles.append(
            self.encoder.register_forward_pre_hook(self._reset, with_kwargs=True)
        )
        self.handles.append(self.attention.register_forward_pre_hook(self._request_weights))
        self.handles.append(
            self.attention.register_forward_hook(self._capture, always_call=True)
        )

    def _reset(self, module, args, kwargs):
        super()._reset(module, args, kwargs)
        # Match the encoder's batching: shape groups, sorted camera order,
        # then batch/observation steps within each camera.
        groups = {}
        for key in self.encoder.image_keys:
            groups.setdefault(tuple(kwargs["images"][key].shape[2:]), []).append(key)
        self.groups = list(groups.values())
        self.group_index = 0

    def _request_weights(self, module, args):
        self.original_backend = module.config._attn_implementation
        # SDPA/FlashAttention do not return the probability matrix. Switch
        # only while this last attention layer runs, preserving its Q/K, RoPE,
        # scaling and value/output projections in the real model forward.
        module.config._attn_implementation = "eager"

    def _restore_backend(self):
        if self.original_backend is not None:
            self.attention.config._attn_implementation = self.original_backend
            self.original_backend = None

    def _capture(self, module, args, output):
        self._restore_backend()
        if output is None:  # always_call hook also runs when forward raises
            return
        weights = output[1]
        if self.group_index >= len(self.groups):
            raise RuntimeError("Unexpected extra DINO self-attention call")
        cameras = self.groups[self.group_index]
        self.group_index += 1
        prefix = 1 + self.encoder.num_register_tokens
        if (
            weights is None or weights.ndim != 4
            or weights.shape[0] != len(cameras) * self.observation_steps
        ):
            raise RuntimeError("Unexpected DINO image self-attention shape")
        for camera_index, key in enumerate(cameras):
            patches = self.encoder.patch_counts[key]
            if weights.shape[-2:] != (prefix + patches, prefix + patches):
                raise RuntimeError(f"DINO attention does not match patch grid for camera {key!r}")
            index = camera_index * self.observation_steps + self.observation_steps - 1
            self.maps[key] = self._reduce(weights[index, :, prefix:, prefix:])
