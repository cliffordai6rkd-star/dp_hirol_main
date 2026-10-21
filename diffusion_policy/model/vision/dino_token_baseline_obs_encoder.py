"""DINO patch-token baselines with optional joint image/wrench self-attention."""

import inspect
from collections.abc import Mapping
from typing import Optional

import torch
from torch import nn


class DinoTokenBaselineObsEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        image_shapes: Mapping,
        mode: str = "dino",
        n_emb: int = 256,
        n_head: int = 4,
        max_obs_steps: int = 2,
        wrench_dim: int = 6,
        wrench_history_steps: int = 8,
        self_attn_layers: int = 1,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        local_files_only: bool = True,
        backbone: Optional[nn.Module] = None,
        force_temporal_encoder: str = "none",
        force_encoder_layers: int = 0,
    ):
        super().__init__()
        if mode not in {"dino", "image_self_attn", "self_attn"}:
            raise ValueError("baseline mode must be 'dino', 'image_self_attn', or 'self_attn'")
        if not image_shapes:
            raise ValueError("at least one camera is required")
        if n_emb % n_head or max_obs_steps < 1 or wrench_history_steps < 1:
            raise ValueError("invalid token embedding or history dimensions")
        if mode in {"image_self_attn", "self_attn"} and self_attn_layers < 1:
            raise ValueError("self-attention baselines require at least one attention layer")
        if force_temporal_encoder not in {"none", "gru", "lstm", "transformer"}:
            raise ValueError("baseline force_temporal_encoder must be 'none', 'gru', 'lstm', or 'transformer'")
        if force_encoder_layers < 0 or (
            force_temporal_encoder in {"gru", "lstm"} and force_encoder_layers < 1
        ):
            raise ValueError("GRU/LSTM require force_encoder_layers >= 1; use 'none' to disable it")
        if backbone is None:
            from transformers import AutoModel

            backbone = AutoModel.from_pretrained(
                pretrained_model_name_or_path, local_files_only=local_files_only
            )
        self.backbone = backbone
        self.freeze_backbone = bool(freeze_backbone)
        self.mode = mode
        self.n_emb = int(n_emb)
        self.max_obs_steps = int(max_obs_steps)
        self.wrench_history_steps = int(wrench_history_steps)
        self.wrench_dim = int(wrench_dim)
        self.force_temporal_encoder_type = force_temporal_encoder
        self.image_keys = tuple(sorted(image_shapes))
        self.image_shapes = {key: tuple(image_shapes[key]) for key in self.image_keys}
        config = backbone.config
        patch_size = config.patch_size
        patch_h, patch_w = (
            (int(patch_size), int(patch_size))
            if isinstance(patch_size, int)
            else map(int, patch_size)
        )
        self.patch_counts = {}
        for key, shape in self.image_shapes.items():
            if len(shape) != 3 or shape[0] != 3 or shape[1] % patch_h or shape[2] % patch_w:
                raise ValueError(f"camera {key!r} shape must be RGB and divisible by DINO patch size")
            self.patch_counts[key] = (shape[1] // patch_h) * (shape[2] // patch_w)
        self.num_register_tokens = int(getattr(config, "num_register_tokens", 0))
        self._interpolate_positions = (
            "interpolate_pos_encoding" in inspect.signature(backbone.forward).parameters
        )
        self.image_projection = nn.Linear(int(config.hidden_size), n_emb)
        self.obs_position = nn.Parameter(torch.empty(1, max_obs_steps, 1, n_emb))
        self.camera_embedding = nn.Parameter(torch.empty(1, 1, len(self.image_keys), 1, n_emb))
        self.image_modality = nn.Parameter(torch.empty(1, 1, 1, n_emb))
        if mode == "self_attn":
            self.wrench_projection = nn.Linear(wrench_dim, n_emb)
            self.wrench_position = nn.Parameter(torch.empty(1, 1, wrench_history_steps, n_emb))
            self.force_modality = nn.Parameter(torch.empty(1, 1, 1, n_emb))
            if force_temporal_encoder in {"gru", "lstm"}:
                recurrent = nn.GRU if force_temporal_encoder == "gru" else nn.LSTM
                self.force_encoder = recurrent(
                    input_size=n_emb,
                    hidden_size=n_emb,
                    num_layers=int(force_encoder_layers),
                    dropout=dropout if force_encoder_layers > 1 else 0.0,
                    batch_first=True,
                )
            elif force_temporal_encoder == "transformer" and force_encoder_layers > 0:
                force_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb, nhead=n_head, dim_feedforward=4 * n_emb,
                    dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
                )
                self.force_encoder = nn.TransformerEncoder(force_layer, num_layers=force_encoder_layers)
            else:
                # Preserve the direct-projection architecture of old checkpoints.
                self.force_encoder = nn.Identity()
            layer = nn.TransformerEncoderLayer(
                d_model=n_emb, nhead=n_head, dim_feedforward=4 * n_emb,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            self.fusion = nn.TransformerEncoder(layer, num_layers=self_attn_layers)
            nn.init.normal_(self.wrench_position, std=0.02)
            nn.init.normal_(self.force_modality, std=0.02)
        elif mode == "image_self_attn":
            layer = nn.TransformerEncoderLayer(
                d_model=n_emb, nhead=n_head, dim_feedforward=4 * n_emb,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            self.fusion = nn.TransformerEncoder(layer, num_layers=self_attn_layers)
        else:
            self.fusion = nn.Identity()
        self.context_norm = nn.LayerNorm(n_emb)
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        for parameter in (self.obs_position, self.camera_embedding, self.image_modality):
            nn.init.normal_(parameter, std=0.02)
        self._set_backbone_frozen()

    @property
    def context_tokens_per_observation(self):
        return sum(self.patch_counts.values()) + (
            self.wrench_history_steps if self.mode == "self_attn" else 0
        )

    def _set_backbone_frozen(self):
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self._set_backbone_frozen()
        return self

    def _encode_dino(self, image):
        kwargs = {"pixel_values": (image - self.image_mean.to(image)) / self.image_std.to(image)}
        if self._interpolate_positions:
            kwargs["interpolate_pos_encoding"] = True
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_backbone):
            output = self.backbone(**kwargs).last_hidden_state
        # Preserve every spatial patch; CLS/register tokens are not patch tokens.
        return output[:, 1 + self.num_register_tokens :]

    def forward(self, *, images: Mapping, wrench_history=None):
        if tuple(sorted(images)) != self.image_keys:
            raise ValueError(f"expected image keys {self.image_keys}, got {tuple(sorted(images))}")
        first = images[self.image_keys[0]]
        if first.ndim != 5:
            raise ValueError("images must have shape [B, To, 3, H, W]")
        batch, steps = first.shape[:2]
        if not 1 <= steps <= self.max_obs_steps:
            raise ValueError("observation history exceeds configured max_obs_steps")
        groups = {}
        for key in self.image_keys:
            value = images[key]
            if tuple(value.shape) != (batch, steps, *self.image_shapes[key]):
                raise ValueError(f"incorrect image shape for camera {key!r}: {tuple(value.shape)}")
            groups.setdefault(tuple(value.shape[2:]), []).append(key)
        encoded = {}
        for keys in groups.values():
            flat = torch.cat([images[key].flatten(0, 1) for key in keys], dim=0)
            patches = self.image_projection(self._encode_dino(flat))
            for key, tokens in zip(keys, patches.split(batch * steps)):
                if tokens.shape[1] != self.patch_counts[key]:
                    raise ValueError(f"DINO patch count does not match configured camera {key!r}")
                encoded[key] = tokens.reshape(batch, steps, self.patch_counts[key], self.n_emb)
        tokens = [
            encoded[key] + self.obs_position[:, :steps] + self.image_modality
            + self.camera_embedding[:, :, camera]
            for camera, key in enumerate(self.image_keys)
        ]
        if self.mode == "self_attn":
            expected = (batch, steps, self.wrench_history_steps, self.wrench_dim)
            if wrench_history is None or tuple(wrench_history.shape) != expected:
                raise ValueError(f"self_attn baseline requires wrench_history {expected}")
            force = (
                self.wrench_projection(wrench_history) + self.wrench_position
                + self.obs_position[:, :steps] + self.force_modality
            )
            if self.force_temporal_encoder_type in {"gru", "lstm", "transformer"}:
                # Each observation's K wrench rows form an independent sequence.
                # Keep every temporal output token, not only the final hidden state.
                force = self.force_encoder(
                    force.reshape(batch * steps, self.wrench_history_steps, self.n_emb)
                )
                if self.force_temporal_encoder_type in {"gru", "lstm"}:
                    force, _ = force
                force = force.reshape(batch, steps, self.wrench_history_steps, self.n_emb)
            tokens.append(force)
        context = torch.cat(tokens, dim=2)
        if self.mode == "image_self_attn":
            # Match force-aware joint SA: all cameras interact independently
            # inside each observation, preserving every image patch position.
            context = self.fusion(context.flatten(0, 1)).reshape(batch, -1, self.n_emb)
            return self.context_norm(context)
        context = context.flatten(1, 2)
        # In self_attn mode Q, K and V all come from the complete concatenation.
        return self.context_norm(self.fusion(context))
