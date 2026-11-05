"""Render RGB using full LSTM wrench sequences and spatial cross-attention."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def top_percentile_mask(values, percentile, *, minimum=0.0):
    """Select the highest ceil((100-p)% * patches) independently per image.

    Zero responses remain empty. Stable sorting resolves ties without ever
    expanding the support beyond the requested patch count.
    """
    percentile = float(percentile)
    if not math.isfinite(percentile) or not 0 <= percentile < 100:
        raise ValueError("Region percentile must satisfy 0 <= percentile < 100")
    count = max(1, math.ceil(values.shape[-1] * (100 - percentile) / 100))
    indices = values.detach().argsort(dim=-1, descending=True, stable=True)[..., :count]
    mask = torch.zeros_like(values).scatter_(-1, indices, 1)
    return mask * (values.detach() > minimum).to(values.dtype)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(math.gcd(8, out_channels), out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(math.gcd(8, out_channels), out_channels),
        )
        self.skip = (nn.Identity() if in_channels == out_channels else
                     nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, image):
        return F.silu(self.body(image) + self.skip(image))


class BottleneckResidualBlock(nn.Module):
    """Mix channels at 1x1, reserving the spatial convolution for a narrow tensor."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        hidden = max(8, out_channels // 2)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.GroupNorm(math.gcd(8, hidden), hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(math.gcd(8, hidden), hidden), nn.SiLU(),
            nn.Conv2d(hidden, out_channels, 1),
            nn.GroupNorm(math.gcd(8, out_channels), out_channels),
        )
        self.skip = (nn.Identity() if in_channels == out_channels else
                     nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, image):
        return F.silu(self.body(image) + self.skip(image))


class BottleneckDecoderBlock(nn.Module):
    """Reduce channels before resizing; retain the full-resolution skip."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        hidden = max(8, out_channels // 2)
        self.up_projection = nn.Conv2d(in_channels, hidden, 1)
        self.body = nn.Sequential(
            nn.Conv2d(hidden + out_channels, hidden, 1),
            nn.GroupNorm(math.gcd(8, hidden), hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(math.gcd(8, hidden), hidden), nn.SiLU(),
            nn.Conv2d(hidden, out_channels, 1),
            nn.GroupNorm(math.gcd(8, out_channels), out_channels),
        )

    def forward(self, features, skip):
        features = F.interpolate(self.up_projection(features), size=skip.shape[-2:],
                                 mode="bilinear", align_corners=False)
        return F.silu(self.body(torch.cat([features, skip], dim=1)) + skip)


class ImageSelfAttentionBlock(nn.Module):
    """One image-token self-attention block."""

    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        query = self.norm(tokens)
        update, _ = self.attention(query, query, query, need_weights=False)
        tokens = tokens + self.dropout(update)
        return tokens + self.ffn(self.output_norm(tokens))


class WrenchCrossAttentionBlock(nn.Module):
    """Image queries read every LSTM timestep; residuals retain image features."""

    def __init__(self, embed_dim, conditioning_dim, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.image_norm = nn.LayerNorm(embed_dim)
        self.wrench_norm = nn.LayerNorm(conditioning_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
            kdim=conditioning_dim, vdim=conditioning_dim,
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, wrench_tokens):
        update, _ = self.cross_attention(
            query=self.image_norm(tokens), key=self.wrench_norm(wrench_tokens),
            value=wrench_tokens, need_weights=False,
        )
        tokens = tokens + self.dropout(update)
        return tokens + self.ffn(self.output_norm(tokens))


class ConvAttentionBackbone(nn.Module):
    """CNN image encoder with configurable image/fusion/decoder depth."""

    def __init__(self, base_channels, conditioning_dim, *, self_attention,
                 patch_size=16, embed_dim=128, depth=4, num_heads=4,
                 mlp_ratio=4.0, dropout=0.0, conv_block="standard", decode=True,
                 image_self_attn_layers=None, cross_attn_layers=None, decoder_layers=None):
        super().__init__()
        if (patch_size < 2 or patch_size & (patch_size - 1) or depth < 1
                or num_heads < 1 or embed_dim < 4 or embed_dim % 4
                or embed_dim % num_heads or base_channels < 1 or mlp_ratio <= 0):
            raise ValueError("Render backbone needs a power-of-two patch_size >= 2, positive "
                             "channels/depth/heads/MLP ratio, and embed_dim divisible by 4 and heads")
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.decode = bool(decode)
        self.conv_block = conv_block
        if conv_block not in {"standard", "bottleneck"}:
            raise ValueError("conv_block must be standard or bottleneck")
        block_type = ResidualBlock if conv_block == "standard" else BottleneckResidualBlock
        levels = int(math.log2(patch_size))
        channels = [min(base_channels * 2 ** level, embed_dim) for level in range(levels)]
        channels.append(embed_dim)
        image_self_attn_layers = (depth if image_self_attn_layers is None
                                  else int(image_self_attn_layers))
        cross_attn_layers = (depth if cross_attn_layers is None
                             else int(cross_attn_layers))
        decoder_layers = (levels if decoder_layers is None else int(decoder_layers))
        if min(image_self_attn_layers, cross_attn_layers, decoder_layers) < 0:
            raise ValueError("attention and decoder layer counts must be non-negative")
        if decoder_layers > levels:
            raise ValueError("decoder_layers cannot exceed log2(patch_size) for the CNN encoder")
        self.decoder_layers = decoder_layers
        self.feature_dim = (channels[levels - decoder_layers]
                            if self.decode and decoder_layers else embed_dim)
        self.stem = ResidualBlock(3, channels[0])
        self.encoders = nn.ModuleList([
            block_type(previous, current) for previous, current in zip(channels, channels[1:])
        ])
        if not self_attention:
            image_self_attn_layers = 0
        self.image_self_attention = nn.ModuleList([
            ImageSelfAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(image_self_attn_layers)
        ])
        self.cross_attention = nn.ModuleList([
            WrenchCrossAttentionBlock(embed_dim, conditioning_dim, num_heads, mlp_ratio,
                                      dropout)
            for _ in range(cross_attn_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.decoders = nn.ModuleList([
            (ResidualBlock(channels[level + 1] + channels[level], channels[level])
             if conv_block == "standard" else
             BottleneckDecoderBlock(channels[level + 1], channels[level]))
            for level in reversed(range(levels - decoder_layers, levels)) if self.decode
        ])
        self.output = nn.Conv2d(self.feature_dim, 3, 3, padding=1)

    def _position(self, height, width, reference):
        y, x = torch.meshgrid(torch.arange(height, device=reference.device),
                              torch.arange(width, device=reference.device), indexing="ij")
        frequency = 10000.0 ** (-torch.arange(self.embed_dim // 4, device=reference.device).float()
                               / (self.embed_dim // 4))
        x, y = x.flatten()[:, None] * frequency, y.flatten()[:, None] * frequency
        return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=-1).to(reference.dtype)

    def encode(self, image, wrench_tokens, camera_embedding):
        height, width = image.shape[-2:]
        p = self.patch_size
        image = F.pad(image, (0, (-width) % p, 0, (-height) % p), mode="replicate")
        features = self.stem(image)
        skips = []
        for encoder in self.encoders:
            if self.decode:
                skips.append(features)
            features = encoder(F.avg_pool2d(features, 2))
        gh, gw = features.shape[-2:]
        tokens = features.flatten(2).transpose(1, 2)
        tokens = tokens + self._position(gh, gw, tokens) + camera_embedding.to(tokens)
        for block in self.image_self_attention:
            tokens = block(tokens)
        for block in self.cross_attention:
            tokens = block(tokens, wrench_tokens)
        features = self.norm(tokens).transpose(1, 2).reshape(image.shape[0], self.embed_dim, gh, gw)
        return features, skips, height, width

    def forward(self, image, wrench_tokens, camera_embedding):
        features, skips, height, width = self.encode(image, wrench_tokens, camera_embedding)
        for decoder, skip in zip(self.decoders, reversed(skips[-self.decoder_layers:])):
            if self.conv_block == "bottleneck":
                features = decoder(features, skip)
            else:
                features = decoder(torch.cat([
                    F.interpolate(features, size=skip.shape[-2:], mode="bilinear", align_corners=False),
                    skip,
                ], dim=1))
        if features.shape[-2:] != (height, width):
            features = F.interpolate(features, size=(height, width), mode="bilinear", align_corners=False)
        features = features[:, :, :height, :width]
        return self.output(features), features


class PatchResidualBackbone(ConvAttentionBackbone):
    """Predict one RGB residual block for every spatial feature token.

    The feature grid is aligned with the DINO patch grid.  Each token emits a
    ``patch_size x patch_size`` RGB block, which is tiled back to the padded
    image and cropped to the original resolution.  The image path still uses
    the same CNN, optional image self-attention and image-query/wrench-key
    cross-attention as :class:`ConvAttentionBackbone`.
    """

    def __init__(self, base_channels, conditioning_dim, *, patch_size=16,
                 embed_dim=128, residual_refine=True, **kwargs):
        super().__init__(base_channels, conditioning_dim, patch_size=patch_size,
                         embed_dim=embed_dim, decode=False, **kwargs)
        self.output = nn.Linear(self.embed_dim, 3 * self.patch_size * self.patch_size)
        self.residual_refine = bool(residual_refine)
        if self.residual_refine:
            self.refine = nn.Conv2d(3, 3, kernel_size=3, padding=1)
            with torch.no_grad():
                self.refine.weight.zero_()
                self.refine.bias.zero_()
                self.refine.weight[:, :, 1, 1] = torch.eye(3)

    def forward(self, image, wrench_tokens, camera_embedding):
        features, _, height, width = self.encode(image, wrench_tokens, camera_embedding)
        batch, _, grid_h, grid_w = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        residual = self.output(tokens).reshape(
            batch, grid_h, grid_w, 3, self.patch_size, self.patch_size,
        )
        residual = residual.permute(0, 3, 1, 4, 2, 5).reshape(
            batch, 3, grid_h * self.patch_size, grid_w * self.patch_size,
        )
        if self.residual_refine:
            residual = self.refine(residual)
        return residual[:, :, :height, :width], features


class GaussianGlowBackbone(ConvAttentionBackbone):
    """Learn blob parameters from fused RGB/wrench tokens, without force/color rules."""

    def __init__(self, base_channels, conditioning_dim, *, glow=None, **kwargs):
        options = dict(glow or {})
        unknown = set(options) - {"num_blobs", "sigma_min", "sigma_max", "max_intensity",
                                  "initial_intensity"}
        if unknown:
            raise ValueError(f"Unknown Gaussian glow options: {sorted(unknown)}")
        num_blobs = options.get("num_blobs", 8)
        if isinstance(num_blobs, bool) or int(num_blobs) != num_blobs or num_blobs < 1:
            raise ValueError("glow.num_blobs must be a positive integer")
        self.num_blobs = int(num_blobs)
        self.sigma_min = float(options.get("sigma_min", 0.03))
        self.sigma_max = float(options.get("sigma_max", 0.20))
        self.max_intensity = float(options.get("max_intensity", 1.0))
        initial_intensity = float(options.get("initial_intensity", 0.05))
        if not (math.isfinite(self.sigma_min) and math.isfinite(self.sigma_max)
                and 0 < self.sigma_min < self.sigma_max <= 1):
            raise ValueError("glow radii require 0 < sigma_min < sigma_max <= 1")
        if not (math.isfinite(self.max_intensity) and math.isfinite(initial_intensity)
                and 0 < initial_intensity < self.max_intensity):
            raise ValueError("glow intensity requires 0 < initial_intensity < max_intensity")
        super().__init__(base_channels, conditioning_dim, decode=False, **kwargs)
        self.blob_queries = nn.Parameter(torch.empty(1, self.num_blobs, self.embed_dim))
        nn.init.normal_(self.blob_queries, std=0.02)
        self.blob_attention = nn.MultiheadAttention(
            self.embed_dim, kwargs.get("num_heads", 4), batch_first=True,
        )
        self.blob_norm = nn.LayerNorm(self.embed_dim)
        # center xy, radii xy, rotation, intensity, RGB; all are bounded below.
        self.output = nn.Linear(self.embed_dim, 9)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.output.bias[2:4].fill_(-1.0)
            fraction = initial_intensity / self.max_intensity
            self.output.bias[5] = math.log(fraction / (1 - fraction))
        side = math.ceil(math.sqrt(self.num_blobs))
        indices = torch.arange(self.num_blobs)
        centers = torch.stack(((indices % side + 0.5) / side,
                               (indices // side + 0.5) / side), dim=-1)
        self.register_buffer("center_logits", torch.logit(centers).unsqueeze(0))

    def render_blobs(self, parameters, height, width):
        """Evaluate a small positive Gaussian mixture in normalized image coordinates."""
        parameters = parameters.float()
        centers = (parameters[..., :2] + self.center_logits).sigmoid()
        radii = self.sigma_min + (self.sigma_max - self.sigma_min) * parameters[..., 2:4].sigmoid()
        angles = math.pi * parameters[..., 4].tanh()
        intensity = self.max_intensity * parameters[..., 5].sigmoid()
        colors = parameters[..., 6:9].sigmoid()
        y, x = torch.meshgrid(
            (torch.arange(height, device=parameters.device).float() + 0.5) / height,
            (torch.arange(width, device=parameters.device).float() + 0.5) / width,
            indexing="ij",
        )
        dx = x - centers[..., 0, None, None]
        dy = y - centers[..., 1, None, None]
        cosine, sine = angles.cos()[..., None, None], angles.sin()[..., None, None]
        local_x = (cosine * dx + sine * dy) / radii[..., 0, None, None]
        local_y = (-sine * dx + cosine * dy) / radii[..., 1, None, None]
        gaussian = torch.exp(-0.5 * (local_x.square() + local_y.square()))
        weights = intensity[..., None] * colors
        return torch.einsum("bkhw,bkc->bchw", gaussian, weights)

    def forward(self, image, wrench_tokens, camera_embedding):
        features, _, height, width = self.encode(image, wrench_tokens, camera_embedding)
        memory = features.flatten(2).transpose(1, 2)
        queries = self.blob_queries.expand(image.shape[0], -1, -1)
        update, _ = self.blob_attention(queries, memory, memory, need_weights=False)
        parameters = self.output(self.blob_norm(queries + update))
        return self.render_blobs(parameters, height, width), features


class RenderNet(nn.Module):
    """Wrench-conditioned renderer with one structured backbone configuration."""

    def __init__(self, wrench_dim, base_channels=16, conditioning_dim=256, residual_scale=0.1,
                 image_encoder="cnn", wrench_encoder="lstm", image_self_attn_layers=4,
                 cross_attn_layers=4, decoder_layers=4, patch_size=16, embed_dim=128,
                 num_heads=4, mlp_ratio=4.0, dropout=0.0, high_attention_render=None,
                 wrench_history_steps=8, max_obs_steps=2, wrench_layers=1,
                 wrench_dropout=0.0, image_keys=("side", "wrist"),
                 initialize_wrench_from_teacher=True, conv_block="standard",
                 rendering_mode="residual", glow=None, patch_residual_refine=True):
        super().__init__()
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("residual_scale must be positive and finite")
        if min(conditioning_dim, wrench_dim, wrench_history_steps, max_obs_steps, wrench_layers) < 1:
            raise ValueError("Wrench dimensions, sequence lengths and LSTM layers must be positive")
        if image_encoder != "cnn":
            raise ValueError("The active RenderNet image_encoder is 'cnn'")
        if wrench_encoder != "lstm":
            raise ValueError("The active RenderNet wrench_encoder is 'lstm'")
        if rendering_mode not in {"residual", "patch_residual", "gaussian_glow"}:
            raise ValueError("rendering_mode must be residual, patch_residual or gaussian_glow")
        self.rendering_mode = rendering_mode
        self.image_keys = tuple(sorted(str(key) for key in image_keys))
        if not self.image_keys or len(set(self.image_keys)) != len(self.image_keys):
            raise ValueError("image_keys must be nonempty and unique")
        self.residual_scale = float(residual_scale)
        self.image_encoder_name = str(image_encoder)
        self.wrench_encoder_name = str(wrench_encoder)
        self.conv_block = conv_block
        self.initialize_wrench_from_teacher = bool(initialize_wrench_from_teacher)
        self.wrench_history_steps = int(wrench_history_steps)
        self.max_obs_steps = int(max_obs_steps)
        self.wrench_projection = nn.Linear(wrench_dim, conditioning_dim)
        self.wrench_position = nn.Parameter(torch.zeros(1, 1, wrench_history_steps, conditioning_dim))
        self.obs_position = nn.Parameter(torch.zeros(1, max_obs_steps, 1, conditioning_dim))
        self.force_modality = nn.Parameter(torch.zeros(1, 1, 1, conditioning_dim))
        self.wrench_encoder = nn.LSTM(
            conditioning_dim, conditioning_dim, num_layers=wrench_layers, batch_first=True,
            dropout=wrench_dropout if wrench_layers > 1 else 0.0,
        )
        backbone_options = dict(
            base_channels=base_channels,
            conditioning_dim=conditioning_dim,
            self_attention=True,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=max(1, int(image_self_attn_layers), int(cross_attn_layers)),
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            conv_block=conv_block,
            image_self_attn_layers=int(image_self_attn_layers),
            cross_attn_layers=int(cross_attn_layers),
            decoder_layers=int(decoder_layers),
        )
        if rendering_mode == "gaussian_glow":
            self.backbone = GaussianGlowBackbone(glow=glow, **backbone_options)
        elif rendering_mode == "patch_residual":
            self.backbone = PatchResidualBackbone(
                residual_refine=patch_residual_refine, **backbone_options,
            )
        else:
            self.backbone = ConvAttentionBackbone(**backbone_options)
        self.camera_embedding = nn.Embedding(len(self.image_keys), self.backbone.embed_dim)
        for parameter in (self.wrench_position, self.obs_position, self.force_modality,
                          self.camera_embedding.weight):
            nn.init.normal_(parameter, std=0.02)
        if rendering_mode in {"residual", "patch_residual"}:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
        self.high_attention_enabled = bool((high_attention_render or {}).get("enabled", False))
        region_config = high_attention_render or {}
        self.region_selection = region_config.get("selection", "soft")
        self.region_activation = str(region_config.get("activation", "sigmoid"))
        self.region_threshold = float(region_config.get("threshold", 0.0))
        self.detach_intensity_for_rgb = bool(region_config.get("detach_for_rgb", True))
        if not math.isfinite(self.region_threshold) or not 0 <= self.region_threshold <= 1:
            raise ValueError("high_attention_render.threshold must be finite and between 0 and 1")
        self.region_percentile = float(region_config.get("percentile", 90.0))
        self.region_prediction_threshold = float(region_config.get("prediction_threshold", 0.5))
        if self.region_selection not in {"soft", "binary_top_percentile"}:
            raise ValueError("Unknown high_attention_render.selection")
        if self.region_activation not in {"sigmoid", "relu"}:
            raise ValueError("Unknown high_attention_render.activation")
        if self.region_selection == "binary_top_percentile":
            if not 0 <= self.region_percentile < 100 or not 0 <= self.region_prediction_threshold < 1:
                raise ValueError("Invalid binary region percentile or prediction threshold")
        self.mask_head = (nn.Conv2d(self.backbone.feature_dim, 1, 1)
                          if self.high_attention_enabled else None)
        if self.mask_head is not None:
            if self.region_activation == "sigmoid":
                initial_strength = float(region_config.get("initial_strength", 0.1))
                if not math.isfinite(initial_strength) or not 0 < initial_strength < 1:
                    raise ValueError("Sigmoid region initial_strength must be finite and between 0 and 1")
                nn.init.zeros_(self.mask_head.weight)
                nn.init.constant_(self.mask_head.bias, math.log(initial_strength / (1 - initial_strength)))
            else:
                nn.init.constant_(self.mask_head.bias, 0.1)
        self._compiled_backbone_forward = None

    def compile_backbone(self, **kwargs):
        # Compile only the image path. LSTM stays in cuDNN; state_dict keys stay
        # unchanged, and evaluation does not compile a second forward-only graph.
        self._compiled_backbone_forward = torch.compile(self.backbone.forward, **kwargs)

    @property
    def output(self):
        return self.backbone.output

    @torch.no_grad()
    def initialize_wrench_encoder(self, teacher_encoder):
        """Copy the complete teacher force path without keeping a teacher reference."""
        if self.wrench_encoder_name != "lstm" or not isinstance(teacher_encoder.force_encoder, nn.LSTM):
            raise ValueError("RenderNet and teacher wrench encoders must both be LSTM")
        self.wrench_projection.load_state_dict(teacher_encoder.wrench_projection.state_dict())
        self.wrench_encoder.load_state_dict(teacher_encoder.force_encoder.state_dict())
        for name in ("wrench_position", "obs_position", "force_modality"):
            target, source = getattr(self, name), getattr(teacher_encoder, name)
            if target.shape != source.shape:
                raise ValueError(f"Teacher {name} shape {source.shape} does not match RenderNet {target.shape}")
            target.copy_(source)

    def encode_wrench_history(self, wrench_history):
        """Return [B*To,K,D] LSTM outputs, retaining every history timestep."""
        if wrench_history.ndim != 4:
            raise ValueError("wrench_history must have shape [B, To, K, D]")
        batch, steps, history, dims = wrench_history.shape
        if not 0 < steps <= self.max_obs_steps or history != self.wrench_history_steps:
            raise ValueError("Wrench observation/history lengths do not match RenderNet configuration")
        force = (self.wrench_projection(wrench_history) + self.wrench_position
                 + self.obs_position[:, :steps] + self.force_modality)
        sequence, _ = self.wrench_encoder(force.reshape(batch * steps, history, -1))
        return sequence

    def forward(self, images, wrench_history, *, return_aux=False, use_attention_mask=None):
        """Output RGB only; downstream DINO receives no student token adapter."""
        use_mask = self.high_attention_enabled if use_attention_mask is None else bool(use_attention_mask)
        if use_mask and self.mask_head is None:
            raise ValueError("Attention rendering was disabled during training; no trained mask head exists")
        wrench_tokens = self.encode_wrench_history(wrench_history)
        batch, steps = wrench_history.shape[:2]
        rendered, masks = {}, {}
        for key, image in images.items():
            if key not in self.image_keys:
                raise KeyError(f"Camera {key!r} is absent from configured image_keys {self.image_keys}")
            if image.ndim != 5 or image.shape[:2] != (batch, steps) or image.shape[2] != 3:
                raise ValueError("RenderNet requires matching [B, To, 3, H, W] images")
            camera = self.camera_embedding.weight[self.image_keys.index(key)].reshape(1, 1, -1)
            backbone_forward = (self._compiled_backbone_forward
                                if self._compiled_backbone_forward is not None
                                and self.training and torch.is_grad_enabled() else self.backbone)
            raw_residual, features = backbone_forward(image.flatten(0, 1), wrench_tokens, camera)
            residual = (raw_residual if self.rendering_mode == "gaussian_glow"
                        else raw_residual.tanh()).reshape_as(image)
            if use_mask:
                logits = self.mask_head(features)
                if self.region_activation == "relu":
                    mask = F.relu(logits).clamp(max=1.0)
                else:
                    mask = logits.float().sigmoid()
                gate = mask
                if self.region_selection == "binary_top_percentile":
                    height, width = image.shape[-2:]
                    patch = self.backbone.patch_size
                    grid = (math.ceil(height / patch), math.ceil(width / patch))
                    scores = F.adaptive_avg_pool2d(mask, grid)
                    hard = top_percentile_mask(scores.flatten(1), self.region_percentile,
                                               minimum=self.region_prediction_threshold).reshape_as(scores)
                    # Hard support in the forward pass; soft surrogate gradients
                    # train the region head through the denoiser objective too.
                    gate = scores + (hard - scores).detach()
                    padded_size = (grid[0] * patch, grid[1] * patch)
                    gate = F.interpolate(gate, size=padded_size, mode="nearest")[:, :, :height, :width]
                if self.rendering_mode in {"gaussian_glow", "patch_residual"}:
                    height, width = image.shape[-2:]
                    patch = self.backbone.patch_size
                    padded_size = (height + (-height) % patch, width + (-width) % patch)
                    mask = F.interpolate(mask, size=padded_size, mode="bilinear", align_corners=False)
                    mask = mask[:, :, :height, :width]
                mask = mask.reshape(batch, steps, 1, *image.shape[-2:])
                masks[key] = mask
                if self.region_selection == "soft":
                    # Threshold after interpolation so closed pixels stay exactly unchanged.
                    # Region MSE uses the unthresholded mask, including closed regions.
                    gate = torch.where(mask > self.region_threshold, mask, torch.zeros_like(mask))
                render_gate = gate.detach() if self.detach_intensity_for_rgb else gate
                residual = residual * render_gate.reshape(batch, steps, 1, *image.shape[-2:])
            if self.rendering_mode == "gaussian_glow":
                # Positive optical exposure gives smooth bounded brightening without clamp saturation.
                strength = -torch.expm1(-self.residual_scale * residual)
                rendered[key] = (image.float() + (1 - image.float()) * strength).to(image.dtype)
            else:
                rendered[key] = (image + self.residual_scale * residual).clamp(0, 1)
        return (rendered, masks) if return_aux else rendered


def build_render_net(wrench_dim, config):
    return RenderNet(wrench_dim=wrench_dim, **dict(config))
