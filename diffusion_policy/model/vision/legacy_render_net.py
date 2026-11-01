"""Version 1 GRU/FiLM renderer, retained for existing checkpoint inference."""

import math

import torch
from torch import nn
from torch.nn import functional as F


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


class UNetBackbone(nn.Module):
    def __init__(self, base_channels, conditioning_dim):
        super().__init__()
        c = int(base_channels)
        self.feature_dim = c
        self.encoder1 = ResidualBlock(3, c)
        self.encoder2 = ResidualBlock(c, 2 * c)
        self.encoder3 = ResidualBlock(2 * c, 4 * c)
        self.film = nn.Linear(conditioning_dim, 8 * c)
        self.bottleneck = ResidualBlock(4 * c, 4 * c)
        self.decoder2 = ResidualBlock(6 * c, 2 * c)
        self.decoder1 = ResidualBlock(3 * c, c)
        self.output = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, image, condition):
        skip1 = self.encoder1(image)
        skip2 = self.encoder2(F.avg_pool2d(skip1, 2))
        features = self.encoder3(F.avg_pool2d(skip2, 2))
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        features = features * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        features = self.bottleneck(features)
        features = self.decoder2(torch.cat([
            F.interpolate(features, size=skip2.shape[-2:], mode="bilinear", align_corners=False), skip2,
        ], dim=1))
        features = self.decoder1(torch.cat([
            F.interpolate(features, size=skip1.shape[-2:], mode="bilinear", align_corners=False), skip1,
        ], dim=1))
        return self.output(features), features


class ConditionedTransformerBlock(nn.Module):
    def __init__(self, embed_dim, conditioning_dim, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.film = nn.Linear(conditioning_dim, 2 * embed_dim)
        self.layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )

    def forward(self, tokens, condition):
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        return self.layer(tokens * (1 + gamma[:, None]) + beta[:, None])


class TransformerBackbone(nn.Module):
    def __init__(self, conditioning_dim, patch_size=16, embed_dim=128, depth=4,
                 num_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        if (patch_size < 1 or depth < 1 or num_heads < 1 or embed_dim < 4
                or embed_dim % 4 or embed_dim % num_heads):
            raise ValueError("Transformer needs positive patch_size/depth/heads and embed_dim divisible by 4 and heads")
        self.patch_size = int(patch_size)
        self.feature_dim = int(embed_dim)
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([
            ConditionedTransformerBlock(embed_dim, conditioning_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, 3 * patch_size * patch_size)

    @property
    def film(self):
        return self.blocks[0].film

    def _position(self, height, width, reference):
        # Dynamic 2D sin/cos positions; camera resolutions need not match.
        y, x = torch.meshgrid(torch.arange(height, device=reference.device),
                              torch.arange(width, device=reference.device), indexing="ij")
        frequency = 10000.0 ** (-torch.arange(self.feature_dim // 4, device=reference.device).float()
                               / (self.feature_dim // 4))
        x, y = x.flatten()[:, None] * frequency, y.flatten()[:, None] * frequency
        return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=-1).to(reference.dtype)

    def forward(self, image, condition):
        height, width = image.shape[-2:]
        p = self.patch_size
        image = F.pad(image, (0, (-width) % p, 0, (-height) % p))
        features = self.patch_embed(image)
        gh, gw = features.shape[-2:]
        tokens = features.flatten(2).transpose(1, 2)
        tokens = tokens + self._position(gh, gw, tokens)
        for block in self.blocks:
            tokens = block(tokens, condition)
        tokens = self.norm(tokens)
        features = tokens.transpose(1, 2).reshape(image.shape[0], self.feature_dim, gh, gw)
        residual = self.output(tokens).reshape(image.shape[0], gh, gw, 3, p, p)
        residual = residual.permute(0, 3, 1, 4, 2, 5).reshape(image.shape[0], 3, gh * p, gw * p)
        return residual[:, :, :height, :width], features


class LegacyRenderNet(nn.Module):
    def __init__(self, wrench_dim, base_channels=16, conditioning_dim=64, residual_scale=0.1,
                 backbone="unet", transformer=None, high_attention_render=None):
        super().__init__()
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("residual_scale must be positive and finite")
        self.residual_scale = float(residual_scale)
        self.backbone_name = str(backbone)
        self.wrench_encoder = nn.GRU(wrench_dim, conditioning_dim, batch_first=True)
        if backbone == "unet":
            self.backbone = UNetBackbone(base_channels, conditioning_dim)
        elif backbone == "transformer":
            self.backbone = TransformerBackbone(conditioning_dim, **(transformer or {}))
        else:
            raise ValueError("backbone must be 'unet' or 'transformer'")
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.high_attention_enabled = bool((high_attention_render or {}).get("enabled", False))
        self.mask_head = (nn.Conv2d(self.backbone.feature_dim, 1, 1)
                          if self.high_attention_enabled else None)

    @property
    def output(self):
        return self.backbone.output

    @property
    def film(self):
        return self.backbone.film

    def forward(self, images, wrench_history, *, return_aux=False, use_attention_mask=None):
        """RGB [B,To,3,H,W], teacher-preprocessed wrench [B,To,K,D].

        Optional masks are predicted from student features; inference never needs
        teacher attention. A trained mask can be bypassed for comparison.
        """
        use_mask = self.high_attention_enabled if use_attention_mask is None else bool(use_attention_mask)
        if use_mask and self.mask_head is None:
            raise ValueError("Attention rendering was disabled during training; no trained mask head exists")
        if wrench_history.ndim != 4:
            raise ValueError("wrench_history must have shape [B, To, K, D]")
        batch, steps, history, dims = wrench_history.shape
        _, hidden = self.wrench_encoder(wrench_history.reshape(batch * steps, history, dims))
        rendered, masks = {}, {}
        for key, image in images.items():
            if image.ndim != 5 or image.shape[:2] != (batch, steps) or image.shape[2] != 3:
                raise ValueError("RenderNet requires matching [B, To, 3, H, W] images")
            raw_residual, features = self.backbone(image.flatten(0, 1), hidden[-1])
            residual = raw_residual.tanh().reshape_as(image)
            if use_mask:
                logits = F.interpolate(self.mask_head(features), size=image.shape[-2:],
                                       mode="bilinear", align_corners=False)
                mask = logits.sigmoid().reshape(batch, steps, 1, *image.shape[-2:])
                masks[key] = mask
                residual = residual * mask
            rendered[key] = (image + self.residual_scale * residual).clamp(0, 1)
        return (rendered, masks) if return_aux else rendered
