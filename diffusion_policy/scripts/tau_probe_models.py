"""Small, isolated models for teacher torque-residual capacity experiments.

These models are diagnostics, not deployment RenderNet replacements.  The
Transformer has the same parameters for direct residual regression and
conditional flow matching: only its noisy input, timestep and training target
change.  All models return a residual/velocity and never add ``base`` themselves.
"""

import math

import torch
from torch import nn


def _sinusoidal(values, channels):
    """Sin/cos embeddings with no trainable parameters."""
    half = channels // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=values.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = values.float().unsqueeze(-1) * frequencies
    result = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if channels % 2:
        result = torch.nn.functional.pad(result, (0, 1))
    return result


class TokenMLP(nn.Module):
    """Shared per-patch MLP conditioned on the complete ordered torque window."""

    def __init__(self, dim=256, tau_dim=56, width=256):
        super().__init__()
        self.dim = int(dim)
        self.tau_dim = int(tau_dim)
        self.net = nn.Sequential(
            nn.Linear(self.dim + self.tau_dim, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, self.dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, base, tau, noisy=None, t=None):
        if base.ndim != 3 or base.shape[-1] != self.dim:
            raise ValueError(f"base must have shape [B, P, {self.dim}]")
        if tau.shape[0] != base.shape[0]:
            raise ValueError("base and tau must have the same batch dimension")
        if noisy is not None or t is not None:
            raise ValueError("TokenMLP is a deterministic residual regression probe")
        flat_tau = tau.flatten(1)
        if flat_tau.shape[-1] != self.tau_dim:
            raise ValueError(f"flattened tau must have {self.tau_dim} elements")
        condition = flat_tau.to(base).unsqueeze(1).expand(-1, base.shape[1], -1)
        return self.net(torch.cat((base, condition), dim=-1))


class _ResidualBlock(nn.Module):
    """Self-attention, torque cross-attention and FFN with adaptive LayerNorm."""

    def __init__(self, hidden, heads):
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.cross_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.ffn_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(hidden, heads, dropout=0.0, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden, heads, dropout=0.0, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, tokens, memory, condition):
        s1, b1, s2, b2, s3, b3 = self.modulation(condition).unsqueeze(1).chunk(6, dim=-1)
        query = self.self_norm(tokens) * (1 + s1) + b1
        tokens = tokens + self.self_attn(query, query, query, need_weights=False)[0]
        query = self.cross_norm(tokens) * (1 + s2) + b2
        tokens = tokens + self.cross_attn(query, memory, memory, need_weights=False)[0]
        return tokens + self.ffn(self.ffn_norm(tokens) * (1 + s3) + b3)


class ResidualTransformer(nn.Module):
    """Matched deterministic / flow-matching conditional token Transformer.

    ``base``: [B, P, dim] teacher image-only tokens (already positional).
    ``tau``: [B, H, 7] normalized causal torque history.
    ``noisy``: optional [B, P, dim] normalized residual at flow time ``t``.
    ``t``: optional [B] flow time in [0, 1]; omitted inputs use zeros.

    No new image positions are added. Fixed temporal positions on the torque
    memory preserve history order, which the flattened MLP also receives.
    """

    def __init__(self, dim=256, hidden=128, depth=3, heads=4):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.base_norm = nn.LayerNorm(dim)
        self.base_projection = nn.Linear(dim, hidden)
        self.noise_projection = nn.Linear(dim, hidden, bias=False)
        self.tau_projection = nn.Linear(7, hidden)
        self.time_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList(_ResidualBlock(hidden, heads) for _ in range(depth))
        self.final_norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, base, tau, noisy=None, t=None):
        if base.ndim != 3 or base.shape[-1] != self.dim:
            raise ValueError(f"base must have shape [B, P, {self.dim}]")
        if tau.ndim != 3 or tau.shape[0] != base.shape[0] or tau.shape[-1] != 7:
            raise ValueError("tau must have shape [B, H, 7] with the base batch size")
        if noisy is not None and noisy.shape != base.shape:
            raise ValueError("noisy must match base shape")
        if t is None:
            t = base.new_zeros(base.shape[0])
        elif t.shape != (base.shape[0],):
            raise ValueError("t must have shape [B]")
        tokens = self.base_projection(self.base_norm(base))
        if noisy is not None:
            tokens = tokens + self.noise_projection(noisy.to(base))
        memory = self.tau_projection(tau.to(base))
        positions = _sinusoidal(torch.arange(tau.shape[1], device=base.device), self.hidden)
        memory = memory + positions.to(memory).unsqueeze(0)
        time_condition = self.time_mlp(_sinusoidal(t.to(base.device) * 1000.0, self.hidden).to(tokens))
        condition = memory.mean(dim=1) + time_condition
        for block in self.blocks:
            tokens = block(tokens, memory, condition)
        return self.output(self.final_norm(tokens))


@torch.no_grad()
def sample_flow(model, base, tau, steps=16, seed=42):
    """Euler sample from noise (t=0) to normalized torque residual (t=1).

    Train using x_t=(1-t)*noise+t*target and velocity target=target-noise.
    The same seed reproduces initial noise for the same batch shape/order.
    Model mode is restored after sampling; caller handles target scaling.
    """
    if int(steps) != steps or steps < 1:
        raise ValueError("steps must be a positive integer")
    generator = torch.Generator(device=base.device).manual_seed(int(seed))
    state = torch.randn(base.shape, dtype=base.dtype, device=base.device, generator=generator)
    training = model.training
    model.eval()
    try:
        for index in range(int(steps)):
            time = base.new_full((base.shape[0],), index / steps)
            state = state + model(base, tau, noisy=state, t=time) / steps
    finally:
        model.train(training)
    return state
