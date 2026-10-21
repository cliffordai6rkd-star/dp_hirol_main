import inspect
from collections.abc import Mapping
from typing import Optional, Sequence

import torch
import torch.nn as nn


class _CrossAttentionBlock(nn.Module):
    """Parameters for one cross-attention + residual + FFN block."""

    def __init__(self, n_emb, n_head, dropout, camera_count=0):
        super().__init__()
        self.force_norm = nn.LayerNorm(n_emb)
        self.image_norm = nn.LayerNorm(n_emb)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=n_emb, num_heads=n_head, dropout=dropout, batch_first=True,
        )
        if camera_count > 1:
            self.camera_fusion = nn.Linear(n_emb * camera_count, n_emb)
            with torch.no_grad():
                self.camera_fusion.weight.zero_()
                for index in range(camera_count):
                    self.camera_fusion.weight[:, index * n_emb:(index + 1) * n_emb].copy_(
                        torch.eye(n_emb) / camera_count
                    )
                self.camera_fusion.bias.zero_()
        else:
            self.camera_fusion = nn.Identity()
        self.cross_dropout = nn.Dropout(dropout)
        self.cross_output_norm = nn.LayerNorm(n_emb)
        self.cross_ffn = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * n_emb, n_emb), nn.Dropout(dropout),
        )


class ForceAwareObsEncoder(nn.Module):
    """Fuse frozen DINOv3 patch tokens with high-rate wrench history tokens."""

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        n_emb: int = 256,
        n_head: int = 4,
        max_obs_steps: int = 4,
        wrench_dim: int = 6,
        wrench_history_steps: int = 8,
        force_temporal_encoder: str = "gru",
        force_encoder_layers: int = 1,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        local_files_only: bool = True,
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.224, 0.225),
        image_keys: Optional[Sequence[str]] = None,
        camera_attention_mode: str = "joint",
        backbone: Optional[nn.Module] = None,
        pre_cross_self_attn_layers: int = 0,
        cross_attn_layers: int = 1,
        cross_attn_query: str = "wrench",
        image_shapes: Optional[Mapping[str, Sequence[int]]] = None,
    ) -> None:
        super().__init__()
        if n_emb % n_head != 0:
            raise ValueError("n_emb must be divisible by n_head")
        if wrench_history_steps < 1 or max_obs_steps < 1:
            raise ValueError("history and observation lengths must be positive")
        if force_temporal_encoder not in {"gru", "lstm", "transformer", "none"}:
            raise ValueError(
                "force_temporal_encoder must be 'gru', 'lstm', 'transformer', or 'none'"
            )
        if force_encoder_layers < 0:
            raise ValueError("force_encoder_layers must be non-negative")
        if pre_cross_self_attn_layers < 0:
            raise ValueError("pre_cross_self_attn_layers must be non-negative")
        if cross_attn_layers < 0:
            raise ValueError("cross_attn_layers must be non-negative")
        if cross_attn_query not in {"wrench", "image"}:
            raise ValueError("cross_attn_query must be 'wrench' or 'image'")
        if camera_attention_mode not in {"joint", "per_camera"}:
            raise ValueError(
                "camera_attention_mode must be 'joint' or 'per_camera'"
            )

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.n_emb = int(n_emb)
        self.max_obs_steps = int(max_obs_steps)
        self.wrench_history_steps = int(wrench_history_steps)
        self.freeze_backbone = bool(freeze_backbone)
        self.force_temporal_encoder_type = force_temporal_encoder
        self.image_keys = tuple(sorted(str(key) for key in (image_keys or image_shapes or ())))
        self.camera_attention_mode = str(camera_attention_mode)
        self.pre_cross_self_attn_layers = int(pre_cross_self_attn_layers)
        self.cross_attn_layers = int(cross_attn_layers)
        self.cross_attn_query = str(cross_attn_query)
        self.image_shapes = {
            str(key): tuple(shape) for key, shape in (image_shapes or {}).items()
        }
        if self.image_shapes and tuple(sorted(self.image_shapes)) != self.image_keys:
            raise ValueError("image_shapes must match image_keys")
        if (not self.cross_attn_layers or self.cross_attn_query == "image") and not self.image_shapes:
            raise ValueError("image_shapes is required for image queries or zero cross-attention layers")
        if len(set(self.image_keys)) != len(self.image_keys):
            raise ValueError("image_keys must not contain duplicates")

        if backbone is None:
            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise ImportError(
                    "DINOv3 requires transformers>=4.56. Install the Nero-aligned "
                    "Python 3.10 training environment."
                ) from exc
            backbone = AutoModel.from_pretrained(
                pretrained_model_name_or_path,
                local_files_only=local_files_only,
            )
        self.backbone = backbone
        forward_parameters = inspect.signature(self.backbone.forward).parameters
        self._interpolate_dino_positions = "interpolate_pos_encoding" in forward_parameters
        hidden_size = getattr(getattr(backbone, "config", None), "hidden_size", None)
        if hidden_size is None:
            raise ValueError("DINOv3 backbone config must expose hidden_size")
        self.num_register_tokens = int(
            getattr(getattr(backbone, "config", None), "num_register_tokens", 0)
        )
        self.patch_counts = {}
        if self.image_shapes:
            patch_size = getattr(backbone.config, "patch_size", None)
            if patch_size is None:
                raise ValueError("DINOv3 config must expose patch_size when image_shapes is provided")
            patch_h, patch_w = (
                (int(patch_size), int(patch_size))
                if isinstance(patch_size, int) else map(int, patch_size)
            )
            for key, shape in self.image_shapes.items():
                if (len(shape) != 3 or shape[0] != 3 or patch_h < 1 or patch_w < 1
                        or shape[1] % patch_h or shape[2] % patch_w):
                    raise ValueError(f"camera {key!r} shape must be RGB and divisible by DINO patch size")
                self.patch_counts[key] = (shape[1] // patch_h) * (shape[2] // patch_w)

        self.image_projection = nn.Linear(int(hidden_size), self.n_emb)
        self.wrench_projection = nn.Linear(int(wrench_dim), self.n_emb)
        self.wrench_position = nn.Parameter(
            torch.zeros(1, 1, self.wrench_history_steps, self.n_emb)
        )
        self.obs_position = nn.Parameter(
            torch.zeros(1, self.max_obs_steps, 1, self.n_emb)
        )
        self.image_modality = nn.Parameter(torch.zeros(1, 1, 1, self.n_emb))
        self.force_modality = nn.Parameter(torch.zeros(1, 1, 1, self.n_emb))
        self.image_mask_token = nn.Parameter(torch.zeros(1, 1, self.n_emb))
        self.camera_embedding = (
            nn.Parameter(
                torch.zeros(1, 1, len(self.image_keys), 1, self.n_emb)
            )
            if len(self.image_keys) > 1
            else None
        )

        if force_temporal_encoder == "gru":
            self.force_encoder = nn.GRU(
                input_size=self.n_emb,
                hidden_size=self.n_emb,
                num_layers=max(1, int(force_encoder_layers)),
                dropout=dropout if force_encoder_layers > 1 else 0.0,
                batch_first=True,
            )
        elif force_temporal_encoder == "lstm":
            self.force_encoder = nn.LSTM(
                input_size=self.n_emb,
                hidden_size=self.n_emb,
                num_layers=max(1, int(force_encoder_layers)),
                dropout=dropout if force_encoder_layers > 1 else 0.0,
                batch_first=True,
            )
        elif force_temporal_encoder == "transformer" and force_encoder_layers > 0:
            force_layer = nn.TransformerEncoderLayer(
                d_model=self.n_emb,
                nhead=n_head,
                dim_feedforward=4 * self.n_emb,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.force_encoder = nn.TransformerEncoder(
                force_layer,
                num_layers=int(force_encoder_layers),
            )
        else:
            self.force_encoder = nn.Identity()

        if self.pre_cross_self_attn_layers:
            joint_layer = nn.TransformerEncoderLayer(
                d_model=self.n_emb,
                nhead=n_head,
                dim_feedforward=4 * self.n_emb,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.pre_cross_self_attention = nn.TransformerEncoder(
                joint_layer, num_layers=self.pre_cross_self_attn_layers
            )
        else:
            # No new checkpoint parameters when loading the original architecture.
            self.pre_cross_self_attention = nn.Identity()

        camera_count = (
            len(self.image_keys)
            if self.camera_attention_mode == "per_camera" and self.cross_attn_query == "wrench"
            else 0
        )
        # Keep the first block's original parameter names so legacy one-layer
        # wrench-query checkpoints still load strictly and produce the same output.
        block_names = (
            "force_norm", "image_norm", "cross_attention", "camera_fusion",
            "cross_dropout", "cross_output_norm", "cross_ffn",
        )
        if self.cross_attn_layers:
            first_block = _CrossAttentionBlock(self.n_emb, n_head, dropout, camera_count)
            for name in block_names:
                setattr(self, name, getattr(first_block, name))
        else:
            for name in block_names:
                setattr(self, name, nn.Identity())
        self.extra_cross_attention_layers = nn.ModuleList([
            _CrossAttentionBlock(self.n_emb, n_head, dropout, camera_count)
            for _ in range(max(0, self.cross_attn_layers - 1))
        ])
        self.context_norm = nn.LayerNorm(self.n_emb)
        # Attention weights are disabled by default because returning them
        # allocates [batch, heads, force_tokens, image_patches] tensors.  The
        # episode visualization utility enables this flag for inference.
        self.capture_cross_attention = False
        self.last_cross_attention_weights = None
        self.last_cross_attention_weights_by_camera = None
        self.last_cross_attention_weights_by_layer = []

        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        nn.init.normal_(self.wrench_position, std=0.02)
        nn.init.normal_(self.obs_position, std=0.02)
        nn.init.normal_(self.image_modality, std=0.02)
        nn.init.normal_(self.force_modality, std=0.02)
        nn.init.normal_(self.image_mask_token, std=0.02)
        if self.camera_embedding is not None:
            nn.init.normal_(self.camera_embedding, std=0.02)
        self._set_backbone_frozen()

    @property
    def output_dim(self) -> int:
        return self.n_emb

    @property
    def context_tokens_per_observation(self) -> int:
        if not self.cross_attn_layers:
            return sum(self.patch_counts.values()) + self.wrench_history_steps
        if self.cross_attn_query == "image":
            return sum(self.patch_counts.values())
        return self.wrench_history_steps

    def _set_backbone_frozen(self) -> None:
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def set_attention_capture(self, enabled: bool = True) -> None:
        """Capture each cross-attention layer; legacy attributes expose the last."""
        self.capture_cross_attention = bool(enabled)
        if not self.capture_cross_attention:
            self.last_cross_attention_weights = None
            self.last_cross_attention_weights_by_camera = None
            self.last_cross_attention_weights_by_layer = []

    def _apply_cross_attention(self, block, query_tokens, memory_tokens, image_keys, patch_counts):
        if self.cross_attn_query == "wrench":
            query = block.force_norm(query_tokens)
            memory_norm = block.image_norm
        else:
            query = block.image_norm(query_tokens)
            memory_norm = block.force_norm
        if self.camera_attention_mode == "joint":
            attended, weights = block.cross_attention(
                query=query, key=memory_norm(memory_tokens), value=memory_tokens,
                need_weights=self.capture_cross_attention, average_attn_weights=False,
            )
        else:
            # Wrench queries read each camera separately and fuse the results;
            # image queries keep all camera positions and share the wrench K/V.
            camera_tokens = memory_tokens if self.cross_attn_query == "wrench" else query
            attended_by_camera, weights_by_camera = [], {}
            for key, tokens in zip(image_keys, camera_tokens.split(patch_counts, dim=1)):
                camera_query = query if self.cross_attn_query == "wrench" else tokens
                camera_memory = tokens if self.cross_attn_query == "wrench" else memory_tokens
                attended_camera, camera_weights = block.cross_attention(
                    query=camera_query, key=memory_norm(camera_memory), value=camera_memory,
                    need_weights=self.capture_cross_attention, average_attn_weights=False,
                )
                attended_by_camera.append(attended_camera)
                if self.capture_cross_attention:
                    weights_by_camera[key] = camera_weights
            attended = (
                block.camera_fusion(torch.cat(attended_by_camera, dim=-1))
                if self.cross_attn_query == "wrench"
                else torch.cat(attended_by_camera, dim=1)
            )
            weights = weights_by_camera
        fused = query_tokens + block.cross_dropout(attended)
        fused = fused + block.cross_ffn(block.cross_output_norm(fused))
        return fused, weights

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _encode_dino(self, image: torch.Tensor):
        image = (image - self.image_mean.to(image)) / self.image_std.to(image)
        backbone_kwargs = {"pixel_values": image}
        if self._interpolate_dino_positions:
            # Nero records 192x256 images instead of DINOv3's square pretraining crop.
            backbone_kwargs["interpolate_pos_encoding"] = True
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(**backbone_kwargs)
        else:
            outputs = self.backbone(**backbone_kwargs)
        tokens = getattr(outputs, "last_hidden_state", None)
        if tokens is None or tokens.ndim != 3:
            raise RuntimeError("DINOv3 backbone must return last_hidden_state [B, N, D]")
        patch_start = 1 + self.num_register_tokens
        if tokens.shape[1] <= patch_start:
            raise RuntimeError(
                f"DINOv3 returned {tokens.shape[1]} tokens; cannot remove CLS/register tokens"
            )
        return tokens[:, 0], tokens[:, patch_start:]

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        wrench_history: Optional[torch.Tensor] = None,
        image_token_mask: Optional[torch.Tensor] = None,
        images: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if images is not None and image is not None:
            raise ValueError("provide either images or the legacy image argument, not both")
        if images is None:
            if image is None:
                raise ValueError("at least one RGB image observation is required")
            legacy_key = self.image_keys[0] if len(self.image_keys) == 1 else "image"
            images = {legacy_key: image}
        if not isinstance(images, Mapping) or not images:
            raise ValueError("images must be a non-empty mapping of RGB observation tensors")

        runtime_image_keys = tuple(sorted(str(key) for key in images))
        if self.image_keys:
            if runtime_image_keys != self.image_keys:
                missing = sorted(set(self.image_keys) - set(runtime_image_keys))
                extra = sorted(set(runtime_image_keys) - set(self.image_keys))
                raise KeyError(
                    f"image observations do not match configured cameras; "
                    f"missing={missing}, extra={extra}"
                )
            image_keys = self.image_keys
        else:
            if len(runtime_image_keys) != 1:
                raise ValueError(
                    "a legacy encoder without image_keys supports exactly one camera"
                )
            image_keys = runtime_image_keys

        first_image = images[image_keys[0]]
        if first_image.ndim != 5:
            raise ValueError(
                f"image {image_keys[0]!r} must be [B, To, C, H, W], "
                f"got {tuple(first_image.shape)}"
            )
        batch_size, obs_steps = first_image.shape[:2]
        for key in image_keys:
            camera_image = images[key]
            if camera_image.ndim != 5:
                raise ValueError(
                    f"image {key!r} must be [B, To, C, H, W], "
                    f"got {tuple(camera_image.shape)}"
                )
            if camera_image.shape[:2] != (batch_size, obs_steps):
                raise ValueError(
                    f"image {key!r} batch/observation dimensions "
                    f"{tuple(camera_image.shape[:2])} do not match "
                    f"{(batch_size, obs_steps)}"
                )
            if camera_image.shape[2] != 3:
                raise ValueError(
                    f"image {key!r} must have 3 channels, got {camera_image.shape[2]}"
                )
            if key in self.image_shapes and tuple(camera_image.shape[2:]) != self.image_shapes[key]:
                raise ValueError(f"image {key!r} must have shape {self.image_shapes[key]}")

        if wrench_history is None:
            raise ValueError("wrench_history is required")
        if wrench_history.ndim != 4:
            raise ValueError(
                "wrench_history must be [B, To, K, D], "
                f"got {tuple(wrench_history.shape)}"
            )
        if wrench_history.shape[:2] != (batch_size, obs_steps):
            raise ValueError("images and wrench batch/observation dimensions must match")
        history_steps = wrench_history.shape[2]
        if obs_steps > self.max_obs_steps:
            raise ValueError(f"obs_steps={obs_steps} exceeds max_obs_steps={self.max_obs_steps}")
        if history_steps != self.wrench_history_steps:
            raise ValueError(
                f"wrench history has {history_steps} steps, expected {self.wrench_history_steps}"
            )

        obs_position = self.obs_position[:, :obs_steps]
        if image_token_mask is not None and image_token_mask.shape != (
            batch_size,
            obs_steps,
        ):
            raise ValueError(
                "image_token_mask must be [B, To], "
                f"got {tuple(image_token_mask.shape)}"
            )

        # Match native DP's shared-RGB-model path: cameras with the same tensor
        # shape are concatenated along the batch axis and encoded in one call.
        camera_groups = {}
        for key in image_keys:
            camera_image = images[key]
            signature = (
                tuple(camera_image.shape[2:]),
                camera_image.dtype,
                camera_image.device,
            )
            camera_groups.setdefault(signature, []).append(key)

        encoded_camera_patches = {}
        flat_camera_batch = batch_size * obs_steps
        for grouped_keys in camera_groups.values():
            flat_images = torch.cat(
                [
                    images[key].reshape(-1, *images[key].shape[2:])
                    for key in grouped_keys
                ],
                dim=0,
            )
            _, grouped_patches = self._encode_dino(flat_images)
            grouped_patches = self.image_projection(grouped_patches)
            for key, patch_image in zip(
                grouped_keys,
                grouped_patches.split(flat_camera_batch, dim=0),
            ):
                if key in self.patch_counts and patch_image.shape[1] != self.patch_counts[key]:
                    raise ValueError(f"DINO patch count does not match configured camera {key!r}")
                encoded_camera_patches[key] = patch_image.reshape(
                    batch_size, obs_steps, patch_image.shape[1], self.n_emb
                )

        camera_patch_tokens = []
        for camera_index, key in enumerate(image_keys):
            patch_image = encoded_camera_patches[key]
            patch_image = patch_image + obs_position + self.image_modality
            if self.camera_embedding is not None:
                patch_image = (
                    patch_image + self.camera_embedding[:, :, camera_index]
                )
            if image_token_mask is not None:
                mask = image_token_mask[:, :, None, None]
                mask_token = self.image_mask_token.reshape(1, 1, 1, self.n_emb)
                patch_image = torch.where(mask, mask_token, patch_image)
            camera_patch_tokens.append(patch_image)

        force = self.wrench_projection(wrench_history)
        force = (
            force
            + self.wrench_position[:, :, :history_steps]
            + obs_position
            + self.force_modality
        )
        flat_force = force.reshape(batch_size * obs_steps, history_steps, self.n_emb)
        if self.force_temporal_encoder_type in {"gru", "lstm"}:
            flat_force, _ = self.force_encoder(flat_force)
        else:
            flat_force = self.force_encoder(flat_force)

        camera_patch_counts = [tokens.shape[2] for tokens in camera_patch_tokens]
        image_token_count = sum(camera_patch_counts)
        flat_images = torch.cat(camera_patch_tokens, dim=2).reshape(
            batch_size * obs_steps, image_token_count, self.n_emb
        )
        if self.pre_cross_self_attn_layers:
            # Per observation: [all camera patches, all temporal wrench outputs].
            # Self-attention changes features, preserving token order/length.
            mixed = self.pre_cross_self_attention(
                torch.cat([flat_images, flat_force], dim=1)
            )
            flat_images = mixed[:, :image_token_count]
            flat_force = mixed[:, image_token_count:]

        self.last_cross_attention_weights = None
        self.last_cross_attention_weights_by_camera = None
        self.last_cross_attention_weights_by_layer = []
        if not self.cross_attn_layers:
            # No cross-attention: retain both streams, including any preceding
            # joint self-attention updates, rather than discarding the images.
            context = torch.cat([flat_images, flat_force], dim=1)
        else:
            if self.cross_attn_query == "wrench":
                context, memory = flat_force, flat_images
                if self.camera_attention_mode == "per_camera" and not self.pre_cross_self_attn_layers:
                    # Preserve the original per-camera checkpoint's K/V path.
                    memory = torch.cat(
                        [encoded_camera_patches[key] for key in image_keys], dim=2
                    ).reshape(batch_size * obs_steps, image_token_count, self.n_emb)
            else:
                context, memory = flat_images, flat_force
            # Each layer updates the query stream; K/V remain the output of
            # the temporal encoder and optional joint self-attention.
            for block in (self, *self.extra_cross_attention_layers):
                context, weights = self._apply_cross_attention(
                    block, context, memory, image_keys, camera_patch_counts
                )
                if self.capture_cross_attention:
                    detached = (
                        {key: value.detach() for key, value in weights.items()}
                        if isinstance(weights, dict) else weights.detach()
                    )
                    self.last_cross_attention_weights_by_layer.append(detached)
            if self.capture_cross_attention:
                self.last_cross_attention_weights = self.last_cross_attention_weights_by_layer[-1]
                if self.camera_attention_mode == "per_camera":
                    self.last_cross_attention_weights_by_camera = self.last_cross_attention_weights

        context = context.reshape(batch_size, -1, self.n_emb)
        return self.context_norm(context)
