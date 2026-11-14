"""Opt-in student ablation; the original RenderNet state layout is preserved."""

import torch
from torch import nn
from torch.nn import functional as F

from diffusion_policy.model.vision.render_net import ConvAttentionBackbone


class TorqueDecoderFiLM(nn.Module):
    """One attention pool, independent zero-initialized affine MLP per scale."""

    def __init__(self, conditioning_dim, channels):
        super().__init__()
        self.pool_score = nn.Linear(conditioning_dim, 1)
        self.affines = nn.ModuleList([
            nn.Sequential(nn.Linear(conditioning_dim, conditioning_dim), nn.GELU(),
                          nn.Linear(conditioning_dim, 2 * channel))
            for channel in channels
        ])
        for affine in self.affines:
            nn.init.zeros_(affine[-1].weight)
            nn.init.zeros_(affine[-1].bias)

    def pool(self, tokens):
        weights = self.pool_score(tokens).float().softmax(dim=1).to(tokens.dtype)
        return (weights * tokens).sum(dim=1)

    def forward(self, feature, pooled, index, gate, alpha):
        gamma, beta = self.affines[index](pooled).chunk(2, dim=-1)
        gate = F.interpolate(gate.to(feature), size=feature.shape[-2:], mode="bilinear",
                             align_corners=False)
        alpha = alpha.to(feature).reshape(-1, 1, 1, 1)
        update = gamma[..., None, None] * feature + beta[..., None, None]
        return feature + alpha * gate * update


class DualPathConvAttentionBackbone(ConvAttentionBackbone):
    """SA×2 → CA → SA×2 → CA; optional decoder /4 and /2 FiLM."""

    def __init__(self, *args, decoder_film=False, **kwargs):
        # New FiLM initialization must not consume the RNG used by shared
        # camera embeddings, condition network or spatial modulation.
        super().__init__(*args, **kwargs)
        if len(self.image_self_attention) != 4 or len(self.cross_attention) != 2:
            raise ValueError("dual_path requires four SA and two CA blocks")
        if self.patch_size != 8 or self.decoder_layers != 3 or not self.decode:
            raise ValueError("dual_path requires patch_size=8 and three decoder levels")
        channels = [block.output_channels if hasattr(block, "output_channels")
                    else block.body[-1].num_channels for block in self.decoders[:2]]
        with torch.random.fork_rng(devices=[]):
            self.decoder_film = (TorqueDecoderFiLM(kwargs["conditioning_dim"], channels)
                                 if decoder_film else None)

    def encode(self, image, wrench_tokens, camera_embedding):
        height, width = image.shape[-2:]
        p = self.patch_size
        image = F.pad(image, (0, (-width) % p, 0, (-height) % p), mode="replicate")
        features, skips = self.stem(image), []
        for encoder in self.encoders:
            skips.append(features)
            features = encoder(F.avg_pool2d(features, 2))
        gh, gw = features.shape[-2:]
        tokens = features.flatten(2).transpose(1, 2)
        tokens = tokens + self._position(gh, gw, tokens) + camera_embedding.to(tokens)
        for group, cross in enumerate(self.cross_attention):
            for block in self.image_self_attention[2 * group:2 * group + 2]:
                tokens = block(tokens)
            tokens = cross(tokens, wrench_tokens)
        features = self.norm(tokens).transpose(1, 2).reshape(image.shape[0], self.embed_dim, gh, gw)
        return features, skips, height, width

    def forward(self, image, wrench_tokens, camera_embedding, *, return_decoder_features=False,
                spatial_modulations=None, modulation_gate=None, contact_alpha=None):
        features, skips, height, width = self.encode(image, wrench_tokens, camera_embedding)
        fused_features, decoder_features = features, []
        pooled = self.decoder_film.pool(wrench_tokens) if self.decoder_film is not None else None
        if self.decoder_film is not None and (modulation_gate is None or contact_alpha is None):
            raise ValueError("decoder FiLM requires spatial gate and contact alpha")
        for index, (decoder, skip) in enumerate(zip(self.decoders, reversed(skips))):
            if self.conv_block == "bottleneck":
                features = decoder(features, skip)
            else:
                features = decoder(torch.cat([F.interpolate(features, size=skip.shape[-2:],
                    mode="bilinear", align_corners=False), skip], dim=1))
            if spatial_modulations is not None and index < len(spatial_modulations):
                features = spatial_modulations[index](features, fused_features, modulation_gate)
            if self.decoder_film is not None and index < 2:
                features = self.decoder_film(features, pooled, index, modulation_gate, contact_alpha)
            if return_decoder_features:
                decoder_features.append(features)
        if features.shape[-2:] != (height, width):
            features = F.interpolate(features, size=(height, width), mode="bilinear", align_corners=False)
        features = features[:, :, :height, :width]
        result = (self.output(features), features)
        return (*result, decoder_features, fused_features) if return_decoder_features else result


def load_common_initialization(model, source_state):
    """Explicitly audited old→new transfer; no silent partial checkpoint load."""
    destination = model.state_dict()
    removed = sorted(set(source_state) - set(destination))
    added = sorted(set(destination) - set(source_state))
    allowed_removed = ("backbone.image_self_attention.4.", "backbone.image_self_attention.5.")
    if model.context_adapter is not None and model.context_adapter.kind == "mlp":
        allowed_removed += ("context_adapter.",)
    if any(not key.startswith(allowed_removed) for key in removed):
        raise ValueError(f"Unexpected removed state keys: {removed}")
    allowed_added = ("backbone.decoder_film.", "context_adapter.")
    if model.context_adapter is None or model.context_adapter.kind != "mlp":
        allowed_added = ("backbone.decoder_film.",)
    if any(not key.startswith(allowed_added) for key in added):
        raise ValueError(f"Unexpected new state keys: {added}")
    common = {key: value for key, value in source_state.items() if key in destination}
    destination.update(common)
    model.load_state_dict(destination, strict=True)
    return {"removed": removed, "added": added, "shared_tensors": len(common)}
