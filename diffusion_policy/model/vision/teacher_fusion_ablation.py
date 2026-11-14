"""Isolated teacher fusion ablations; the production encoder is unchanged."""

import math

import torch
from torch import nn

from diffusion_policy.model.vision.force_aware_obs_encoder import ForceAwareObsEncoder


class TorqueAttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.query, std=.02)
        self.last_weights = None

    def forward(self, tokens):
        scores = (self.norm(tokens) * self.query).sum(-1) / math.sqrt(tokens.shape[-1])
        weights = scores.softmax(-1)
        self.last_weights = weights.detach()
        return (weights.unsqueeze(-1) * tokens).sum(1)


class TorqueResidualFiLM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, visual, condition, alpha):
        scale, shift = self.mlp(condition).unsqueeze(1).chunk(2, dim=-1)
        update = alpha.to(visual) * (scale * visual + shift)
        return visual + update, update


class TeacherFusionAblationEncoder(ForceAwareObsEncoder):
    """Keep the original image/torque streams, FFNs and final context norm.

    FiLM-only replaces each CA update before the existing FFN. Hybrid adds
    FiLM after each complete CA+FFN block. All torque updates use the same
    alpha once; alpha=0 preserves the ordinary visual-only FFN pathway.
    """

    def __init__(self, *args, fusion_mode="baseline", **kwargs):
        super().__init__(*args, **kwargs)
        if fusion_mode not in {"baseline", "film_only", "hybrid"}:
            raise ValueError(f"Unknown teacher fusion mode {fusion_mode!r}")
        if (self.cross_attn_query != "image" or not self.cross_attn_query_residual
                or self.pre_cross_self_attn_layers or self.camera_attention_mode != "joint"):
            raise ValueError("Ablations require residual image-query fusion without joint self-attention")
        self.fusion_mode = fusion_mode
        self.capture_fusion_diagnostics = False
        self.last_fusion_diagnostics = {}
        # Do not perturb the RNG for subsequently initialized common DiT weights.
        with torch.random.fork_rng(devices=[]):
            self.torque_pool = TorqueAttentionPool(self.n_emb) if fusion_mode != "baseline" else None
            self.film_layers = nn.ModuleList([
                TorqueResidualFiLM(self.n_emb) for _ in range(self.cross_attn_layers)
            ] if fusion_mode != "baseline" else [])
        if fusion_mode == "film_only":
            for block in (self, *self.extra_cross_attention_layers):
                block.cross_attention = nn.Identity()
                # Attention pooling owns its normalization; these CA norms
                # are unused in this variant and should not enter AdamW.
                block.force_norm = nn.Identity()
                block.image_norm = nn.Identity()

    def _record(self, name, value):
        if self.capture_fusion_diagnostics:
            self.last_fusion_diagnostics[name] = value.detach().float().square().mean().sqrt()

    def finish_cross_attention(self, block, query_tokens, attended, *, alpha=None):
        # Same operations/order as the original encoder, including dropout.
        if alpha is not None:
            attended = alpha.to(attended) * attended
        update = block.cross_dropout(attended)
        index = 0 if block is self else list(self.extra_cross_attention_layers).index(block) + 1
        self._record(f"block{index}/cross_attention_update_rms", update)
        fused = query_tokens + update
        return fused + block.cross_ffn(block.cross_output_norm(fused))

    def _apply_cross_attention(self, block, query_tokens, memory_tokens, image_keys, patch_counts, alpha=None):
        index = 0 if block is self else list(self.extra_cross_attention_layers).index(block) + 1
        if alpha is None:
            raise ValueError("Teacher ablations require the shared contact alpha")
        if self.fusion_mode == "film_only":
            condition = self.torque_pool(memory_tokens)
            fused, update = self.film_layers[index](query_tokens, condition, alpha)
            self._record(f"block{index}/film_update_rms", update)
            fused = fused + block.cross_ffn(block.cross_output_norm(fused))
            return fused, None
        fused, weights = super()._apply_cross_attention(
            block, query_tokens, memory_tokens, image_keys, patch_counts, alpha=alpha,
        )
        if self.fusion_mode == "hybrid":
            condition = self.torque_pool(memory_tokens)
            fused, update = self.film_layers[index](fused, condition, alpha)
            self._record(f"block{index}/film_update_rms", update)
        return fused, weights
