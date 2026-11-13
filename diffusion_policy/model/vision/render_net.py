"""Render RGB using full LSTM wrench sequences and spatial cross-attention."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from diffusion_policy.model.vision.wrench_preprocessing import compute_contact_alpha


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


class _AttentionBlockCompatibilityView:
    """Expose the pre-v2 ``backbone.blocks`` inspection interface.

    Older checkpoints and diagnostic scripts referred to the two attention
    stacks as ``blocks[i].self_attention`` and ``blocks[i].cross_attention``.
    The active CNN implementation stores those stacks separately.  This view
    keeps the old read-only access path without registering duplicate modules
    in the state dict.
    """

    def __init__(self, backbone, index):
        self._backbone = backbone
        self._index = int(index)

    @property
    def self_attention(self):
        block = self._backbone.image_self_attention[self._index]
        return block.attention

    @property
    def cross_attention(self):
        block = self._backbone.cross_attention[self._index]
        return block.cross_attention


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

    @property
    def blocks(self):
        """Compatibility view for legacy attention diagnostics."""
        count = min(len(self.image_self_attention), len(self.cross_attention))
        return tuple(_AttentionBlockCompatibilityView(self, index) for index in range(count))

    def encoder_modules(self):
        return (self.stem, self.encoders, self.image_self_attention,
                self.cross_attention, self.norm)

    def rgb_modules(self):
        return (self.decoders, self.output)

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
        """Encode, fuse and decode one camera image."""
        features, skips, height, width = self.encode(image, wrench_tokens, camera_embedding)
        return self.decode_features(features, skips, height, width)

    def decode_features(self, features, skips, height, width):
        """Decode an existing fused representation without repeating its encoder."""
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


class DINOAttentionBackbone(ConvAttentionBackbone):
    """LoRA DINO image features with wrench fusion and an RGB decoder.

    DINO replaces the CNN encoder and its additional self-attention stack.
    Intermediate DINO features supply the decoder skips through channel
    projections and resizing; no second image encoder is trained from scratch.
    """

    def __init__(self, base_channels, conditioning_dim, *, dino, patch_size=16,
                 embed_dim=128, num_heads=4, mlp_ratio=4.0, dropout=0.0,
                 conv_block="bottleneck", cross_attn_layers=2, decoder_layers=3):
        nn.Module.__init__(self)
        from diffusion_policy.model.vision.dinov3_lora_encoder import DINOv3LoRAEncoder

        if (base_channels < 1 or embed_dim < 4 or embed_dim % 4 or num_heads < 1
                or embed_dim % num_heads or mlp_ratio <= 0 or cross_attn_layers < 0
                or decoder_layers < 1):
            raise ValueError("Invalid DINO RenderNet channels, heads, or layer counts")
        if conv_block not in {"standard", "bottleneck"}:
            raise ValueError("conv_block must be standard or bottleneck")
        options = dict(dino or {})
        lora = dict(options.pop("lora", {}))
        self.dino_encoder = DINOv3LoRAEncoder(**options, **lora)
        self.patch_size = int(self.dino_encoder.patch_size)
        if int(patch_size) != self.patch_size:
            raise ValueError("RenderNet patch_size must match the pretrained DINO patch size")
        if len(self.dino_encoder.intermediate_layers) != decoder_layers:
            raise ValueError("DINO intermediate_layers must supply one feature per RGB decoder layer")
        self.embed_dim = int(embed_dim)
        self.decoder_layers = int(decoder_layers)
        self.decode = True
        self.conv_block = conv_block
        self.image_projection = nn.Linear(self.dino_encoder.hidden_size, embed_dim)
        # Preserve the inspection interface: DINO supplies all image
        # self-attention, with no separately initialized attention blocks.
        self.image_self_attention = nn.ModuleList()
        self.cross_attention = nn.ModuleList([
            WrenchCrossAttentionBlock(embed_dim, conditioning_dim, num_heads,
                                      mlp_ratio, dropout)
            for _ in range(cross_attn_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        channels = [min(base_channels * 2 ** level, embed_dim)
                    for level in range(decoder_layers)]
        self.skip_projections = nn.ModuleList([
            nn.Conv2d(self.dino_encoder.hidden_size, width, 1) for width in channels
        ])
        decoder_inputs = [embed_dim, *reversed(channels[1:])]
        self.decoders = nn.ModuleList([
            (BottleneckDecoderBlock(previous, width) if conv_block == "bottleneck"
             else ResidualBlock(previous + width, width))
            for previous, width in zip(decoder_inputs, reversed(channels))
        ])
        self.feature_dim = channels[0]
        self.output = nn.Conv2d(self.feature_dim, 3, 3, padding=1)

    def encoder_modules(self):
        return (self.dino_encoder, self.image_projection, self.cross_attention, self.norm)

    def rgb_modules(self):
        return (self.skip_projections, self.decoders, self.output)

    def encode(self, image, wrench_tokens, camera_embedding):
        height, width = image.shape[-2:]
        patch = self.patch_size
        padded = F.pad(image, (0, (-width) % patch, 0, (-height) % patch), mode="replicate")
        grid_h, grid_w = padded.shape[-2] // patch, padded.shape[-1] // patch
        final_tokens, intermediate_tokens = self.dino_encoder(padded)
        tokens = self.image_projection(final_tokens)
        tokens = tokens + self._position(grid_h, grid_w, tokens) + camera_embedding.to(tokens)
        for block in self.cross_attention:
            tokens = block(tokens, wrench_tokens)
        features = self.norm(tokens).transpose(1, 2).reshape(
            image.shape[0], self.embed_dim, grid_h, grid_w,
        )
        skips = []
        for level, (projection, hidden) in enumerate(zip(self.skip_projections, intermediate_tokens)):
            hidden = hidden.transpose(1, 2).reshape(
                image.shape[0], self.dino_encoder.hidden_size, grid_h, grid_w,
            )
            # Early DINO layers provide the finer decoder skips. With three
            # decoder blocks these are full, half and quarter image resolution.
            skip_size = (math.ceil(height / 2 ** level), math.ceil(width / 2 ** level))
            skips.append(F.interpolate(projection(hidden), size=skip_size,
                                       mode="bilinear", align_corners=False))
        return features, skips, height, width


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
    """Wrench-conditioned renderer with condition and RGB paths.

    The condition path predicts region maps, phase logits, and alpha. With
    ``separate_gate_path`` and ``condition_backbone``, it has its own compact
    visual encoder; freezing conditions leaves the entire RGB path trainable.
    Shared-backbone and legacy joint/intensity modes remain available.
    ``cross_camera_attn_layers > 0`` selects plain RGB residual rendering:
    camera bottlenecks exchange information without region or phase heads.
    """

    def __init__(self, wrench_dim, base_channels=16, conditioning_dim=256, residual_scale=0.1,
                 image_encoder="cnn", wrench_encoder="lstm", image_self_attn_layers=4,
                 cross_attn_layers=4, decoder_layers=None, patch_size=16, embed_dim=128,
                 num_heads=4, mlp_ratio=4.0, dropout=0.0, high_attention_render=None,
                 wrench_history_steps=8, max_obs_steps=2, wrench_layers=1,
                 wrench_dropout=0.0, image_keys=("side", "wrist"),
                 initialize_wrench_from_teacher=True, conv_block="standard",
                 rendering_mode="residual", glow=None, patch_residual_refine=True,
                 backbone=None, architecture_version=None,
                 separate_gate_path=False, transformer=None, phase_prediction=None,
                 shared_condition_backbone=False, dino=None, condition_backbone=None,
                 use_contact_alpha=True, use_region_gate=True, cross_camera_attn_layers=0):
        super().__init__()
        # ``backbone``/``transformer`` were the public configuration shape in
        # older RenderNet checkpoints. Accepting those fields keeps old CNN
        # configs and visualization checkpoints loadable with the same semantics.
        legacy_transformer = dict(transformer or {})
        if legacy_transformer:
            patch_size = legacy_transformer.get("patch_size", patch_size)
            embed_dim = legacy_transformer.get("embed_dim", embed_dim)
            num_heads = legacy_transformer.get("num_heads", num_heads)
            depth = legacy_transformer.get("depth")
            if depth is not None:
                image_self_attn_layers = depth
                cross_attn_layers = depth
            decoder_layers = legacy_transformer.get("decoder_layers", decoder_layers)
        # The removed v1 entry point consumed seven wrench samples by default.
        # Keep that default for its serialized config shape; v2 and the active
        # training config always provide the resolved history length.
        if architecture_version is not None and int(architecture_version) == 1 and wrench_history_steps == 8:
            wrench_history_steps = 7
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("residual_scale must be positive and finite")
        if min(conditioning_dim, wrench_dim, wrench_history_steps, max_obs_steps, wrench_layers) < 1:
            raise ValueError("Wrench dimensions, sequence lengths and LSTM layers must be positive")
        if image_encoder not in {"cnn", "dino_lora"}:
            raise ValueError("RenderNet image_encoder must be cnn or dino_lora")
        if image_encoder == "dino_lora" and (
            image_self_attn_layers != 0 or rendering_mode != "residual"
            or not shared_condition_backbone or separate_gate_path
        ):
            raise ValueError("DINO LoRA requires shared residual rendering, no separate gate path, "
                             "and image_self_attn_layers=0")
        if wrench_encoder != "lstm":
            raise ValueError("The active RenderNet wrench_encoder is 'lstm'")
        if backbone is not None and str(backbone) not in {"unet", "transformer", "cnn"}:
            raise ValueError("backbone must be unet, transformer, cnn or omitted")
        if rendering_mode not in {"residual", "patch_residual", "gaussian_glow"}:
            raise ValueError("rendering_mode must be residual, patch_residual or gaussian_glow")
        effective_decoder_layers = ((3 if image_encoder == "dino_lora" else int(math.log2(patch_size)))
                                    if decoder_layers is None else int(decoder_layers))
        self.rendering_mode = rendering_mode
        self.image_keys = tuple(sorted(str(key) for key in image_keys))
        if not self.image_keys or len(set(self.image_keys)) != len(self.image_keys):
            raise ValueError("image_keys must be nonempty and unique")
        self.residual_scale = float(residual_scale)
        # These switches affect rendering only. Keep the condition network and
        # checkpoint tensors intact for paired rendering-gate ablations.
        self.use_contact_alpha = bool(use_contact_alpha)
        self.use_region_gate = bool(use_region_gate)
        self.image_encoder_name = str(image_encoder)
        self.wrench_encoder_name = str(wrench_encoder)
        self.conv_block = conv_block
        self.initialize_wrench_from_teacher = bool(initialize_wrench_from_teacher)
        self.wrench_history_steps = int(wrench_history_steps)
        self.max_obs_steps = int(max_obs_steps)
        phase_config = dict(phase_prediction or {})
        unknown_phase = set(phase_config) - {"enabled", "hidden_dim", "freeze_wrench_encoder"}
        if unknown_phase:
            raise ValueError(f"Unknown phase_prediction options: {sorted(unknown_phase)}")
        self.phase_prediction_enabled = bool(phase_config.get("enabled", False))
        self.shared_condition_backbone = bool(shared_condition_backbone)
        if int(cross_camera_attn_layers) != cross_camera_attn_layers or cross_camera_attn_layers < 0:
            raise ValueError("cross_camera_attn_layers must be a non-negative integer")
        self.cross_camera_attn_layers = int(cross_camera_attn_layers)
        if self.cross_camera_attn_layers and (
            image_encoder != "cnn" or rendering_mode != "residual" or len(self.image_keys) < 2
            or self.use_contact_alpha or self.use_region_gate or self.phase_prediction_enabled
            or bool((high_attention_render or {}).get("enabled", False))
            or self.shared_condition_backbone or separate_gate_path or condition_backbone is not None
        ):
            raise ValueError(
                "cross_camera_attn_layers requires CNN residual rendering with at least two cameras "
                "and disabled region/contact-alpha/phase conditions and condition backbones"
            )
        if self.shared_condition_backbone and (
            separate_gate_path or not self.phase_prediction_enabled or rendering_mode != "residual"
        ):
            raise ValueError(
                "shared_condition_backbone requires phase prediction, residual rendering, "
                "and separate_gate_path=false"
            )
        self.phase_hidden_dim = int(phase_config.get("hidden_dim", 64))
        self.freeze_wrench_encoder_on_init = bool(phase_config.get("freeze_wrench_encoder", False))
        self._wrench_encoder_frozen = False
        if self.phase_prediction_enabled and self.phase_hidden_dim < 1:
            raise ValueError("phase_prediction.hidden_dim must be positive")
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
            decoder_layers=effective_decoder_layers,
        )
        if image_encoder == "dino_lora":
            self.backbone = DINOAttentionBackbone(
                base_channels, conditioning_dim, dino=dino, patch_size=patch_size,
                embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                dropout=dropout, conv_block=conv_block,
                cross_attn_layers=int(cross_attn_layers), decoder_layers=effective_decoder_layers,
            )
        elif rendering_mode == "gaussian_glow":
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
        # Keep the complete gate path separate from the RGB path.  The RGB
        # modules retain their historical names for checkpoint/visualization
        # callers; ``gate_*`` is the frozen branch used by the mask head.
        self.separate_gate_path = bool(separate_gate_path and self.high_attention_enabled)
        self.independent_condition_backbone = condition_backbone is not None
        condition_options = dict(condition_backbone or {})
        if self.independent_condition_backbone and not self.separate_gate_path:
            raise ValueError("condition_backbone requires separate_gate_path=true and "
                             "high_attention_render.enabled=true")
        unknown_condition = set(condition_options) - {
            "base_channels", "patch_size", "embed_dim", "num_heads", "mlp_ratio",
            "dropout", "conv_block", "image_self_attn_layers", "cross_attn_layers",
            "decoder_layers",
        }
        if unknown_condition:
            raise ValueError(f"Unknown condition_backbone options: {sorted(unknown_condition)}")
        if self.separate_gate_path:
            self.gate_wrench_projection = copy.deepcopy(self.wrench_projection)
            self.gate_wrench_position = nn.Parameter(self.wrench_position.detach().clone())
            self.gate_obs_position = nn.Parameter(self.obs_position.detach().clone())
            self.gate_force_modality = nn.Parameter(self.force_modality.detach().clone())
            self.gate_wrench_encoder = copy.deepcopy(self.wrench_encoder)
            if self.independent_condition_backbone:
                condition_options = {**backbone_options, **condition_options}
                condition_options["decode"] = condition_options["decoder_layers"] > 0
                self.gate_backbone = ConvAttentionBackbone(**condition_options)
                # This branch predicts conditions only and has no RGB output.
                self.gate_backbone.output = nn.Identity()
                self.gate_camera_embedding = nn.Embedding(
                    len(self.image_keys), self.gate_backbone.embed_dim,
                )
                nn.init.normal_(self.gate_camera_embedding.weight, std=0.02)
            else:
                self.gate_backbone = copy.deepcopy(self.backbone)
                self.gate_camera_embedding = copy.deepcopy(self.camera_embedding)
        else:
            self.gate_wrench_projection = None
            self.gate_wrench_position = None
            self.gate_obs_position = None
            self.gate_force_modality = None
            self.gate_wrench_encoder = None
            self.gate_backbone = None
            self.gate_camera_embedding = None
        mask_feature_dim = (self.backbone.embed_dim if self.shared_condition_backbone else
                            self.gate_backbone.feature_dim if self.gate_backbone is not None
                            else self.backbone.feature_dim)
        self.phase_temporal_encoder = (
            nn.GRU(mask_feature_dim, self.phase_hidden_dim, batch_first=True)
            if self.phase_prediction_enabled else None
        )
        # One spatial head predicts region, phase-logit and alpha maps. The
        # shared path uses fused encoder features before the RGB decoder;
        # legacy checkpoints retain their decoder feature input.
        needs_gate_head = self.high_attention_enabled or self.phase_prediction_enabled
        head_channels = 5 if self.phase_prediction_enabled else 1
        head_input_dim = mask_feature_dim + (self.phase_hidden_dim if self.phase_prediction_enabled else 0)
        self.mask_head = nn.Conv2d(head_input_dim, head_channels, 1) if needs_gate_head else None
        if self.mask_head is not None:
            if self.region_activation == "sigmoid":
                initial_strength = float(region_config.get("initial_strength", 0.1))
                if not math.isfinite(initial_strength) or not 0 < initial_strength < 1:
                    raise ValueError("Sigmoid region initial_strength must be finite and between 0 and 1")
                nn.init.zeros_(self.mask_head.weight)
                nn.init.zeros_(self.mask_head.bias)
                self.mask_head.bias.data[0] = math.log(initial_strength / (1 - initial_strength))
                if self.phase_prediction_enabled:
                    nn.init.normal_(self.mask_head.weight.data[1:], std=0.01)
                    self.mask_head.bias.data[4] = math.log(initial_strength / (1 - initial_strength))
            else:
                nn.init.zeros_(self.mask_head.weight)
                nn.init.zeros_(self.mask_head.bias)
                self.mask_head.bias.data[0] = 0.1
                if self.phase_prediction_enabled:
                    nn.init.normal_(self.mask_head.weight.data[1:], std=0.01)
        # Each batch item is one observation time. These blocks exchange only
        # camera tokens from that time, after their image/wrench encoding.
        self.cross_camera_attention = nn.ModuleList([
            ImageSelfAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(self.cross_camera_attn_layers)
        ])
        self._compiled_backbone_forward = None
        self._compiled_shared_encode = None
        self._compiled_shared_decode = None
        self.training_stage = "joint"

    def compile_backbone(self, **kwargs):
        # Compile only the image path. LSTM stays in cuDNN; state_dict keys stay
        # unchanged, and evaluation does not compile a second forward-only graph.
        if self.shared_condition_backbone or self.cross_camera_attn_layers:
            self._compiled_shared_encode = torch.compile(self.backbone.encode, **kwargs)
            # The condition path encodes every observation and decodes a
            # subset. The plain RGB path applies selection before both calls;
            # its caller supplies dynamic=True when sparse rendering is used.
            decode_options = {**kwargs, "dynamic": True}
            self._compiled_shared_decode = torch.compile(self._shared_decode, **decode_options)
        else:
            self._compiled_backbone_forward = torch.compile(self.backbone.forward, **kwargs)

    def _shared_decode(self, features, skips, height, width):
        return self.backbone.decode_features(features, skips, height, width)[0]

    def _set_requires_grad(self, module, enabled):
        if module is not None:
            module.requires_grad_(bool(enabled))

    def _gate_modules(self):
        if self.cross_camera_attn_layers:
            return ()
        if self.shared_condition_backbone:
            modules = [
                *self.backbone.encoder_modules(), self.camera_embedding,
                self.mask_head, self.phase_temporal_encoder,
            ]
            if not self._wrench_encoder_frozen:
                modules.extend([
                    self.wrench_projection, self.wrench_encoder,
                    self.wrench_position, self.obs_position, self.force_modality,
                ])
            return tuple(modules)
        if not self.separate_gate_path:
            # Compatibility mode shares the feature path between the field and
            # RGB head.  Keep its output head in the RGB-only set so the
            # historical two-stage behavior remains well-defined.
            modules = [
                *self.backbone.encoder_modules(), self.backbone.decoders, self.camera_embedding,
                self.mask_head, self.phase_temporal_encoder,
            ]
            if not self._wrench_encoder_frozen:
                modules.extend([
                    self.wrench_projection, self.wrench_encoder,
                    self.wrench_position, self.obs_position, self.force_modality,
                ])
            return tuple(modules)
        modules = [
            *self.gate_backbone.encoder_modules(), self.gate_backbone.decoders,
            self.gate_camera_embedding, self.mask_head, self.phase_temporal_encoder,
        ]
        if not self._wrench_encoder_frozen:
            modules.extend([
                self.gate_wrench_projection, self.gate_wrench_encoder,
                self.gate_wrench_position, self.gate_obs_position, self.gate_force_modality,
            ])
        return tuple(modules)

    def _rgb_modules(self):
        if self.cross_camera_attn_layers:
            modules = [self.backbone, self.camera_embedding, self.cross_camera_attention]
            if not self._wrench_encoder_frozen:
                modules.extend([
                    self.wrench_projection, self.wrench_encoder,
                    self.wrench_position, self.obs_position, self.force_modality,
                ])
            return tuple(modules)
        if self.shared_condition_backbone:
            return self.backbone.rgb_modules()
        if not self.separate_gate_path:
            # The shared encoder stays frozen in the legacy RGB-only stage;
            # renderer-specific heads (patch refinement or Gaussian blobs)
            # remain trainable alongside the final output projection.
            rgb_only = [self.backbone.output]
            for name in ("refine", "blob_queries", "blob_attention", "blob_norm"):
                module = getattr(self.backbone, name, None)
                if module is not None:
                    rgb_only.append(module)
            return tuple(rgb_only)
        modules = [self.backbone, self.camera_embedding]
        if not self._wrench_encoder_frozen:
            modules.extend([
                self.wrench_projection, self.wrench_encoder,
                self.wrench_position, self.obs_position, self.force_modality,
            ])
        return tuple(modules)

    def _wrench_conditioning_parameters(self):
        modules = [self.wrench_projection, self.wrench_encoder]
        parameters = [self.wrench_position, self.obs_position, self.force_modality]
        if self.separate_gate_path:
            modules.extend([self.gate_wrench_projection, self.gate_wrench_encoder])
            parameters.extend([
                self.gate_wrench_position, self.gate_obs_position, self.gate_force_modality,
            ])
        return modules, parameters

    def freeze_wrench_encoder(self):
        """Freeze teacher-initialized force encoders and their embeddings."""
        self._wrench_encoder_frozen = True
        modules, parameters = self._wrench_conditioning_parameters()
        for module in modules:
            module.requires_grad_(False)
            module.eval()
        for parameter in parameters:
            parameter.requires_grad_(False)

    def _enforce_wrench_freeze(self):
        if self._wrench_encoder_frozen:
            self.freeze_wrench_encoder()

    def _enforce_dino_freeze(self):
        if isinstance(self.backbone, DINOAttentionBackbone):
            self.backbone.dino_encoder.enforce_base_freeze()

    def set_training_stage(self, stage):
        """Select the condition/RGB paths that receive gradients.

        ``condition_warmup`` trains the condition branch while RGB decoders
        stay frozen. ``joint`` trains both paths, except explicitly frozen
        force encoders. Legacy ``intensity`` aliases condition warm-up and
        legacy ``rgb`` trains RGB while freezing the condition branch.
        """
        stage = str(stage)
        if stage not in {"intensity", "rgb", "condition_warmup", "joint"}:
            raise ValueError("RenderNet stage must be intensity, rgb, condition_warmup or joint")
        if self.cross_camera_attn_layers and stage not in {"rgb", "joint"}:
            raise ValueError("Cross-camera RGB rendering has no condition warm-up stage")
        self.training_stage = stage
        self.requires_grad_(False)
        if stage in {"intensity", "condition_warmup", "joint"}:
            # The conditioning phase keeps its visual/wrench-conditioned gate
            # branch trainable; RGB decoders stay frozen.
            for module in self._gate_modules():
                self._set_requires_grad(module, True)
        if stage in {"rgb", "joint"}:
            for module in self._rgb_modules():
                self._set_requires_grad(module, True)
        self._enforce_wrench_freeze()
        self._enforce_dino_freeze()

    def set_stage_mode(self, training=True):
        """Set mode while keeping inactive paths and frozen force encoders deterministic."""
        self.train(bool(training))
        if not training:
            return
        self._enforce_wrench_freeze()
        if self.training_stage == "joint":
            return
        frozen = self._gate_modules() if self.training_stage == "rgb" else self._rgb_modules()
        for module in frozen:
            self._set_requires_grad(module, False)
            if module is not None and not isinstance(module, nn.Parameter):
                module.eval()

    @staticmethod
    def _parameters_from_modules(modules):
        for module in modules:
            if module is None:
                continue
            if isinstance(module, nn.Parameter):
                yield module
            else:
                yield from module.parameters()

    def gate_parameters(self):
        """Return parameters whose updates can change the predicted gate."""
        params = list(self._parameters_from_modules(self._gate_modules()))
        frozen_ids = self._frozen_encoder_parameter_ids()
        return [parameter for parameter in params if id(parameter) not in frozen_ids]

    def rgb_parameters(self):
        """Return parameters updated by the RGB stage."""
        params = list(self._parameters_from_modules(self._rgb_modules()))
        frozen_ids = self._frozen_encoder_parameter_ids()
        return [parameter for parameter in params if id(parameter) not in frozen_ids]

    def _frozen_encoder_parameter_ids(self):
        frozen = set()
        if self._wrench_encoder_frozen:
            modules, parameters = self._wrench_conditioning_parameters()
            frozen.update(id(parameter) for parameter in parameters)
            for module in modules:
                frozen.update(id(parameter) for parameter in module.parameters())
        if isinstance(self.backbone, DINOAttentionBackbone):
            encoder = self.backbone.dino_encoder
            adapter_ids = {id(parameter) for parameter in encoder.lora_parameters()}
            frozen.update(id(parameter) for parameter in encoder.parameters()
                          if id(parameter) not in adapter_ids)
        return frozen

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
        if self.separate_gate_path:
            self.gate_wrench_projection.load_state_dict(teacher_encoder.wrench_projection.state_dict())
            self.gate_wrench_encoder.load_state_dict(teacher_encoder.force_encoder.state_dict())
            for name in ("gate_wrench_position", "gate_obs_position", "gate_force_modality"):
                target = getattr(self, name)
                source = getattr(teacher_encoder, name.removeprefix("gate_"))
                if target.shape != source.shape:
                    raise ValueError(f"Teacher {name} shape {source.shape} does not match RenderNet {target.shape}")
                target.copy_(source)

    @torch.no_grad()
    def initialize_rgb_from_gate(self):
        """Warm-start the RGB path from the completed gate snapshot."""
        if not self.separate_gate_path or self.independent_condition_backbone:
            # An independently configured condition encoder is not an RGB
            # pretraining stage. Keep the RGB initialization, including its
            # zero residual output, when the condition stage completes.
            return
        self.wrench_projection.load_state_dict(self.gate_wrench_projection.state_dict())
        self.wrench_encoder.load_state_dict(self.gate_wrench_encoder.state_dict())
        for name in ("wrench_position", "obs_position", "force_modality"):
            getattr(self, name).copy_(getattr(self, f"gate_{name}"))
        self.backbone.load_state_dict(self.gate_backbone.state_dict())
        self.camera_embedding.load_state_dict(self.gate_camera_embedding.state_dict())

    def _encode_wrench_history(self, wrench_history, *, gate=False):
        """Return [B*To,K,D] LSTM outputs for one renderer branch."""
        if wrench_history.ndim != 4:
            raise ValueError("wrench_history must have shape [B, To, K, D]")
        batch, steps, history, dims = wrench_history.shape
        if not 0 < steps <= self.max_obs_steps or history != self.wrench_history_steps:
            raise ValueError("Wrench observation/history lengths do not match RenderNet configuration")
        if gate and not self.separate_gate_path:
            gate = False
        projection = self.gate_wrench_projection if gate else self.wrench_projection
        wrench_position = self.gate_wrench_position if gate else self.wrench_position
        obs_position = self.gate_obs_position if gate else self.obs_position
        force_modality = self.gate_force_modality if gate else self.force_modality
        encoder = self.gate_wrench_encoder if gate else self.wrench_encoder
        force = (projection(wrench_history) + wrench_position
                 + obs_position[:, :steps] + force_modality)
        sequence, _ = encoder(force.reshape(batch * steps, history, -1))
        return sequence

    def encode_wrench_history(self, wrench_history):
        """Return RGB-branch [B*To,K,D] LSTM outputs."""
        return self._encode_wrench_history(wrench_history, gate=False)

    def encode_gate_wrench_history(self, wrench_history):
        """Return frozen gate-branch [B*To,K,D] LSTM outputs."""
        return self._encode_wrench_history(wrench_history, gate=True)

    def predict_contact_phase(self, images, wrench_history):
        """Predict causal phase logits/alpha through the shared spatial head.

        The condition features are pooled across image locations and
        cameras, then a GRU processes observation steps in order. Its state is
        broadcast back onto each camera's spatial feature map; the same 5-way
        convolution that predicts the region field produces phase-logit and
        alpha maps. Phase IDs are 0=free, 1=pre-contact, and 2=contact.
        """
        if not self.phase_prediction_enabled:
            raise ValueError("RenderNet phase_prediction.enabled is false")
        if self.shared_condition_backbone:
            return self.forward(
                images, wrench_history, return_aux=True, return_phase=True, condition_only=True,
            )[2]
        gate_wrench_tokens = self.encode_gate_wrench_history(wrench_history)
        _, predictions = self._gate_head_outputs(images, gate_wrench_tokens, wrench_history.shape[:2])
        return predictions

    def _gate_head_outputs(self, images, gate_wrench_tokens, batch_steps):
        """Run the shared per-camera spatial-temporal gate feature and head."""
        batch, steps = (int(batch_steps[0]), int(batch_steps[1]))
        gate_camera_embedding = (self.gate_camera_embedding
                                 if self.separate_gate_path else self.camera_embedding)
        gate_backbone = self.gate_backbone if self.separate_gate_path else self.backbone
        feature_maps = {}
        unknown_cameras = set(images) - set(self.image_keys)
        if unknown_cameras:
            raise KeyError(f"Cameras {sorted(unknown_cameras)!r} are absent from configured image_keys {self.image_keys}")
        camera_keys = tuple(key for key in self.image_keys if key in images)
        if not camera_keys:
            raise ValueError("RenderNet requires at least one configured camera observation")
        for key in camera_keys:
            image = images[key]
            if image.ndim != 5 or image.shape[:3] != (batch, steps, 3):
                raise ValueError("RenderNet requires matching [B, To, 3, H, W] images")
            flat_image = image.flatten(0, 1)
            camera = gate_camera_embedding.weight[self.image_keys.index(key)].reshape(1, 1, -1)
            if self.independent_condition_backbone and not gate_backbone.decode:
                # Keep the lightweight condition head on its patch grid. A
                # decoder-free backbone.forward would otherwise upsample every
                # feature channel to full resolution before applying the head.
                features, _, _, _ = gate_backbone.encode(flat_image, gate_wrench_tokens, camera)
            else:
                _, features = gate_backbone(flat_image, gate_wrench_tokens, camera)
            feature_maps[key] = features.reshape(batch, steps, *features.shape[1:])
        masks, predictions = self._condition_head_outputs(feature_maps, (batch, steps))
        if self.independent_condition_backbone:
            masks = {
                key: F.interpolate(mask, size=images[key].shape[-2:], mode="bilinear",
                                   align_corners=False)
                for key, mask in masks.items()
            }
        return masks, predictions

    def _condition_head_outputs(self, feature_maps, batch_steps):
        """Predict all condition outputs from per-camera spatial features."""
        batch, steps = batch_steps
        predictions = None
        masks = {}
        temporal = None
        if self.phase_prediction_enabled:
            # Camera-average summaries give one chronological state stream
            # shared by all views.  GRU is causal across the To axis.
            summary = torch.stack([
                features.mean(dim=(-2, -1)) for features in feature_maps.values()
            ], dim=0).mean(dim=0)
            temporal, _ = self.phase_temporal_encoder(summary)
            phase_maps, alpha_maps = [], []
            for key, features in feature_maps.items():
                context = temporal[..., None, None].expand(
                    -1, -1, -1, features.shape[-2], features.shape[-1]
                )
                head_input = torch.cat([features, context], dim=2).flatten(0, 1)
                output = self.mask_head(head_input).reshape(batch, steps, 5,
                                                            features.shape[-2], features.shape[-1])
                masks[key] = self._activate_region(output[:, :, :1].flatten(0, 1))
                phase_maps.append(output[:, :, 1:4].mean(dim=(-2, -1)))
                alpha_maps.append(output[:, :, 4].mean(dim=(-2, -1)))
            predictions = {
                "phase_logits": torch.stack(phase_maps, dim=0).mean(dim=0),
                "contact_alpha": torch.stack(alpha_maps, dim=0).mean(dim=0).float().sigmoid(),
            }

        for key, features in feature_maps.items():
            if self.phase_prediction_enabled:
                continue
            flat_features = features.flatten(0, 1)
            if self.mask_head is None:
                mask = torch.ones(batch * steps, 1, *features.shape[-2:],
                                  device=features.device, dtype=features.dtype)
            else:
                mask = self._activate_region(self.mask_head(flat_features))
            masks[key] = mask
        return masks, predictions

    def _condition_only_predictions(self, predictions):
        if self.training and torch.is_grad_enabled() and self.training_stage == "joint":
            # An all-free joint batch skips RGB execution. Keep zero gradients
            # for its trainable parameters so optimizer momentum/decay and DDP
            # hooks have the same semantics as a zero token loss through RGB.
            anchor = sum(parameter.reshape(-1)[0] * 0 for parameter in self.rgb_parameters()
                         if parameter.requires_grad)
            predictions = {**predictions, "phase_logits": predictions["phase_logits"] + anchor}
        return predictions

    @staticmethod
    def _render_observation_indices(render_observation_mask, batch, steps, device):
        if render_observation_mask is None:
            return None
        if render_observation_mask.shape != (batch, steps):
            raise ValueError("render_observation_mask must have shape [B, To]")
        if render_observation_mask.dtype != torch.bool:
            raise TypeError("render_observation_mask must be boolean")
        return render_observation_mask.to(device=device).flatten().nonzero(as_tuple=True)[0]

    def _forward_shared(self, images, wrench_history, *, use_mask, contact_alpha,
                        return_aux, return_phase, condition_only, render_observation_mask):
        """Encode once for conditions, then decode only the selected observations."""
        batch, steps = wrench_history.shape[:2]
        unknown_cameras = set(images) - set(self.image_keys)
        if unknown_cameras:
            raise KeyError(f"Cameras {sorted(unknown_cameras)!r} are absent from configured image_keys")
        camera_keys = tuple(key for key in self.image_keys if key in images)
        if not camera_keys:
            raise ValueError("RenderNet requires at least one configured camera observation")
        indices = self._render_observation_indices(
            render_observation_mask, batch, steps, wrench_history.device,
        )
        wrench_tokens = self.encode_wrench_history(wrench_history)
        use_compiled = self.training and torch.is_grad_enabled()
        encode = (self._compiled_shared_encode
                  if use_compiled and self._compiled_shared_encode is not None
                  else self.backbone.encode)
        encoded = {}
        feature_maps = {}
        for key in camera_keys:
            image = images[key]
            if image.ndim != 5 or image.shape[:3] != (batch, steps, 3):
                raise ValueError("RenderNet requires matching [B, To, 3, H, W] images")
            camera = self.camera_embedding.weight[self.image_keys.index(key)].reshape(1, 1, -1)
            encoded[key] = encode(image.flatten(0, 1), wrench_tokens, camera)
            features = encoded[key][0]
            feature_maps[key] = features.reshape(batch, steps, *features.shape[1:])
        region_maps, phase_predictions = self._condition_head_outputs(feature_maps, (batch, steps))
        flat_masks = {
            key: F.interpolate(region_maps[key], size=images[key].shape[-2:],
                               mode="bilinear", align_corners=False)
            for key in camera_keys
        }
        masks = {
            key: mask.reshape(batch, steps, 1, *images[key].shape[-2:])
            for key, mask in flat_masks.items()
        }
        if condition_only:
            return None, masks, self._condition_only_predictions(phase_predictions)
        if self.use_contact_alpha:
            alpha = phase_predictions["contact_alpha"] if contact_alpha is None else contact_alpha
            if alpha.shape != (batch, steps):
                raise ValueError("contact_alpha must have shape [B, To]")
        else:
            alpha = wrench_history.new_ones((batch, steps))
        decode = (self._compiled_shared_decode
                  if use_compiled and self._compiled_shared_decode is not None else self._shared_decode)
        rendered = {}
        for key in camera_keys:
            image = images[key]
            if indices is not None and indices.numel() == 0:
                rendered[key] = image
                continue
            features, skips, height, width = encoded[key]
            flat_image = image.flatten(0, 1)
            gate = (self._mask_gate(flat_masks[key], image) if use_mask
                    else torch.ones_like(flat_masks[key]))
            if self.detach_intensity_for_rgb:
                gate = gate.detach()
            image_alpha = alpha.to(image).reshape(batch * steps, 1, 1, 1)
            if indices is not None:
                features = features.index_select(0, indices)
                skips = [skip.index_select(0, indices) for skip in skips]
                flat_image = flat_image.index_select(0, indices)
                gate = gate.index_select(0, indices)
                image_alpha = image_alpha.index_select(0, indices)
            residual = decode(features, skips, height, width).tanh()
            selected_image = (
                flat_image + self.residual_scale * image_alpha * gate * residual
            ).clamp(0, 1)
            selected_image = torch.where(image_alpha > 0, selected_image, flat_image)
            if indices is not None:
                selected_image = image.flatten(0, 1).index_copy(0, indices, selected_image)
            rendered[key] = selected_image.reshape_as(image)
        if return_phase:
            return rendered, masks if use_mask else {}, phase_predictions
        return (rendered, masks if use_mask else {}) if return_aux else rendered

    def _forward_multicamera(self, images, wrench_history, *, return_aux,
                             return_phase, render_observation_mask):
        """Exchange camera features at each time, then decode their RGB separately."""
        if set(images) != set(self.image_keys):
            raise ValueError("Cross-camera rendering requires all configured image_keys")
        batch, steps = wrench_history.shape[:2]
        for image in images.values():
            if image.ndim != 5 or image.shape[:3] != (batch, steps, 3):
                raise ValueError("RenderNet requires matching [B, To, 3, H, W] images")
        indices = self._render_observation_indices(
            render_observation_mask, batch, steps, wrench_history.device,
        )
        wrench_tokens = self.encode_wrench_history(wrench_history)
        if indices is not None:
            wrench_tokens = wrench_tokens.index_select(0, indices)
        rendered = {}
        if indices is not None and indices.numel() == 0:
            rendered = dict(images)
        else:
            use_compiled = self.training and torch.is_grad_enabled()
            encode = (self._compiled_shared_encode
                      if use_compiled and self._compiled_shared_encode is not None
                      else self.backbone.encode)
            decode = (self._compiled_shared_decode
                      if use_compiled and self._compiled_shared_decode is not None
                      else self._shared_decode)
            encoded, token_groups = {}, []
            for camera_index, key in enumerate(self.image_keys):
                image = images[key].flatten(0, 1)
                if indices is not None:
                    image = image.index_select(0, indices)
                camera = self.camera_embedding.weight[camera_index].reshape(1, 1, -1)
                encoded[key] = encode(image, wrench_tokens, camera)
                token_groups.append(encoded[key][0].flatten(2).transpose(1, 2))
            tokens = torch.cat(token_groups, dim=1)
            for block in self.cross_camera_attention:
                tokens = block(tokens)
            camera_tokens = tokens.split([group.shape[1] for group in token_groups], dim=1)
            for key, tokens in zip(self.image_keys, camera_tokens):
                features, skips, height, width = encoded[key]
                features = tokens.transpose(1, 2).reshape_as(features)
                raw_residual = decode(features, skips, height, width)
                image = images[key]
                flat_image = image.flatten(0, 1)
                if indices is not None:
                    flat_image = flat_image.index_select(0, indices)
                rgb = (flat_image + self.residual_scale * raw_residual.tanh()).clamp(0, 1)
                if indices is not None:
                    rgb = image.flatten(0, 1).index_copy(0, indices, rgb)
                rendered[key] = rgb.reshape_as(image)
        if return_phase:
            return rendered, {}, None
        return (rendered, {}) if return_aux else rendered

    def _activate_region(self, logits):
        if self.region_activation == "relu":
            return F.relu(logits).clamp(max=1.0)
        return logits.float().sigmoid()

    def _mask_gate(self, mask, image):
        """Convert a predicted field into the spatial support used by RGB."""
        if not self.use_region_gate:
            return torch.ones_like(mask)
        if self.region_selection == "binary_top_percentile":
            height, width = image.shape[-2:]
            patch = self.backbone.patch_size
            grid = (math.ceil(height / patch), math.ceil(width / patch))
            scores = F.adaptive_avg_pool2d(mask, grid)
            hard = top_percentile_mask(
                scores.flatten(1), self.region_percentile,
                minimum=self.region_prediction_threshold,
            ).reshape_as(scores)
            # Hard support in the forward pass; retain a soft surrogate for
            # region/denoiser gradients when joint training is requested.
            gate = scores + (hard - scores).detach()
            padded_size = (grid[0] * patch, grid[1] * patch)
            return F.interpolate(gate, size=padded_size, mode="nearest")[:, :, :height, :width]
        # Threshold at image resolution so closed pixels stay exactly
        # unchanged and no edge values leak through interpolation.
        return torch.where(mask > self.region_threshold, mask, torch.zeros_like(mask))

    def forward(self, images, wrench_history, *, return_aux=False, use_attention_mask=None,
                future_contact=None, contact_alpha=None, return_phase=False,
                condition_only=False, render_observation_mask=None):
        """Output RGB only; downstream DINO receives no student token adapter."""
        use_mask = self.high_attention_enabled if use_attention_mask is None else bool(use_attention_mask)
        if use_mask and self.mask_head is None:
            raise ValueError("Attention rendering was disabled during training; no trained mask head exists")
        batch, steps = wrench_history.shape[:2]
        if condition_only and not (self.phase_prediction_enabled and return_aux and return_phase):
            raise ValueError(
                "condition_only requires phase prediction, return_aux=True, and return_phase=True"
            )
        if self.cross_camera_attn_layers:
            return self._forward_multicamera(
                images, wrench_history, return_aux=return_aux, return_phase=return_phase,
                render_observation_mask=render_observation_mask,
            )
        if self.shared_condition_backbone:
            return self._forward_shared(
                images, wrench_history, use_mask=use_mask, contact_alpha=contact_alpha,
                return_aux=return_aux, return_phase=return_phase, condition_only=condition_only,
                render_observation_mask=render_observation_mask,
            )
        render_indices = self._render_observation_indices(
            render_observation_mask, batch, steps, wrench_history.device,
        )
        needs_gate_features = use_mask or self.phase_prediction_enabled
        gate_wrench_tokens = (self.encode_gate_wrench_history(wrench_history)
                              if needs_gate_features else None)
        if needs_gate_features:
            gate_masks, phase_predictions = self._gate_head_outputs(
                images, gate_wrench_tokens, (batch, steps),
            )
        else:
            gate_masks, phase_predictions = {}, None
        if condition_only:
            masks = {}
            for key, flat_mask in gate_masks.items():
                height, width = images[key].shape[-2:]
                mask = flat_mask
                if self.rendering_mode in {"gaussian_glow", "patch_residual"} and use_mask:
                    patch = self.backbone.patch_size
                    padded_size = (height + (-height) % patch, width + (-width) % patch)
                    mask = F.interpolate(flat_mask, size=padded_size, mode="bilinear",
                                         align_corners=False)
                    mask = mask[:, :, :height, :width]
                masks[key] = mask.reshape(batch, steps, 1, height, width)
            return None, masks, self._condition_only_predictions(phase_predictions)
        wrench_tokens = self.encode_wrench_history(wrench_history)
        alpha = None
        if self.use_contact_alpha:
            alpha = contact_alpha
            if alpha is None and phase_predictions is not None:
                alpha = phase_predictions["contact_alpha"]
            if alpha is None and future_contact is not None:
                if future_contact.ndim == 2:
                    future_contact = future_contact.unsqueeze(-1)
                alpha = compute_contact_alpha(future_contact)
            if alpha is not None and alpha.shape != (batch, steps):
                raise ValueError("contact_alpha must have shape [B, To]")
        rendered, masks = {}, {}
        for key, image in images.items():
            if key not in self.image_keys:
                raise KeyError(f"Camera {key!r} is absent from configured image_keys {self.image_keys}")
            if image.ndim != 5 or image.shape[:2] != (batch, steps) or image.shape[2] != 3:
                raise ValueError("RenderNet requires matching [B, To, 3, H, W] images")
            flat_image = image.flatten(0, 1)
            if use_mask:
                flat_mask = gate_masks[key]
            else:
                flat_mask = torch.ones(
                    flat_image.shape[0], 1, *flat_image.shape[-2:],
                    device=flat_image.device, dtype=flat_image.dtype,
                )
            mask = flat_mask
            if self.rendering_mode in {"gaussian_glow", "patch_residual"} and use_mask:
                height, width = image.shape[-2:]
                patch = self.backbone.patch_size
                padded_size = (height + (-height) % patch, width + (-width) % patch)
                mask = F.interpolate(flat_mask, size=padded_size, mode="bilinear", align_corners=False)
                mask = mask[:, :, :height, :width]
            flat_gate = self._mask_gate(mask, image) if use_mask else mask
            rgb_image = flat_image
            rgb_wrench = wrench_tokens
            if render_indices is not None:
                rgb_image = rgb_image.index_select(0, render_indices)
                rgb_wrench = rgb_wrench.index_select(0, render_indices)
            rgb_camera = self.camera_embedding.weight[self.image_keys.index(key)].reshape(1, 1, -1)
            backbone_forward = (self._compiled_backbone_forward
                                if self._compiled_backbone_forward is not None
                                and self.training and torch.is_grad_enabled() else self.backbone)
            if render_indices is not None and render_indices.numel() == 0:
                raw_residual = torch.zeros_like(flat_image)
            else:
                raw_residual, _ = backbone_forward(
                    rgb_image, rgb_wrench, rgb_camera,
                )
                if render_indices is not None:
                    raw_residual = raw_residual.new_zeros(flat_image.shape).index_copy(
                        0, render_indices, raw_residual,
                    )
            residual = (raw_residual if self.rendering_mode == "gaussian_glow"
                        else raw_residual.tanh()).reshape_as(image)
            if use_mask:
                mask = mask.reshape(batch, steps, 1, *image.shape[-2:])
                masks[key] = mask
                render_gate = (flat_gate.detach()
                               if self.detach_intensity_for_rgb else flat_gate)
                residual = residual * render_gate.reshape(batch, steps, 1, *image.shape[-2:])
            else:
                render_gate = torch.ones_like(residual[:, :, :1])
            if alpha is None:
                image_alpha = torch.ones(batch, steps, 1, 1, 1,
                                         device=image.device, dtype=image.dtype)
            else:
                image_alpha = alpha.to(device=image.device, dtype=image.dtype).reshape(
                    batch, steps, 1, 1, 1,
                )
            if self.rendering_mode == "gaussian_glow":
                # Positive optical exposure gives smooth bounded brightening without clamp saturation.
                strength = -torch.expm1(-self.residual_scale * residual)
                rendered[key] = (image.float() + image_alpha * (1 - image.float()) * strength).to(image.dtype)
            else:
                rendered[key] = (image + self.residual_scale * image_alpha * residual).clamp(0, 1)
            if alpha is not None:
                rendered[key] = torch.where(image_alpha > 0, rendered[key], image)
        if return_phase:
            return rendered, masks, phase_predictions
        return (rendered, masks) if return_aux else rendered


def build_render_net(wrench_dim, config):
    options = dict(config)
    legacy_spatial = dict(options.pop("spatial_modulation", None) or {})
    if legacy_spatial.get("enabled", False):
        raise ValueError(
            "Decoder FiLM has been removed. Start a new no-FiLM training run "
            "without spatial_modulation in the config, or use the legacy code "
            "to reproduce a FiLM-enabled checkpoint."
        )
    # v1 serialized configs omitted the history length; that entry point used
    # seven samples.  Keep direct constructor defaults unchanged for new code.
    if ("wrench_history_steps" not in options and options.get("architecture_version", 1) == 1
            and isinstance(options.get("backbone"), str)):
        options["wrench_history_steps"] = 7
    return RenderNet(wrench_dim=wrench_dim, **options)
