# Copyright (c) Sudeep Dasari, 2023
# Adapted from FACTR's action_transformer.py, which credits ACT and DETR.
# Upstream: https://github.com/RaindragonD/FACTR/blob/
# 5a20d69342d9d234fd588c10b7c3e62ea559c9b0/factr/models/action_transformer.py
# See LICENSE in this directory for upstream licensing and attribution.
# Local changes: batch-first public API, separate encode/decode methods,
# optional zero-layer encoder, and inclusion of the action queries/output head.
# Attention weights are not returned, enabling PyTorch's SDPA implementation.

"""FACTR's deterministic action-chunking transformer.

The encoder retains all visual and force tokens. Learned action queries read
that complete memory through the decoder; there is no diffusion or CVAE path.
"""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


def _activation(name):
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    raise ValueError(f"Unsupported FACTR activation: {name!r}")


class _SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, tokens):
        # Internal transformer tensors are [sequence, batch, embedding].
        return self.pe[: tokens.shape[0]].repeat(1, tokens.shape[1], 1).detach().clone()


class _EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _activation(activation)

    def forward(self, src, pos):
        query = key = src + pos
        update, _ = self.self_attn(query, key, value=src, need_weights=False)
        src = self.norm1(src + self.dropout1(update))
        update = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout3(update))


class _DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.activation = _activation(activation)

    def forward(self, tgt, memory, pos, query_pos):
        query = key = tgt + query_pos
        update, _ = self.self_attn(query, key, value=tgt, need_weights=False)
        tgt = self.norm1(tgt + self.dropout1(update))
        update, _ = self.multihead_attn(
            query=tgt + query_pos, key=memory + pos, value=memory, need_weights=False
        )
        tgt = self.norm2(tgt + self.dropout2(update))
        update = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout4(update))


class _Encoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(num_layers))
        self.num_layers = num_layers

    def forward(self, src, pos):
        for layer in self.layers:
            src = layer(src, pos)
        return src


class _Decoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(num_layers))
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos, query_pos):
        for layer in self.layers:
            tgt = layer(tgt, memory, pos, query_pos)
        return tgt


class FactrActionTransformer(nn.Module):
    """Decode complete observation-token memory into an action chunk.

    ``num_encoder_layers=0`` accepts an already-fused observation context.
    ``decode`` always consumes already-encoded memory directly. Both public
    token interfaces use [batch, sequence, embedding] tensors.
    """

    def __init__(
        self,
        n_emb=512,
        n_head=8,
        num_encoder_layers=4,
        num_decoder_layers=6,
        dim_feedforward=3200,
        dropout=0.1,
        activation="relu",
        horizon=8,
        action_dim=7,
    ):
        super().__init__()
        if n_emb % 2 or n_emb <= 0:
            raise ValueError("n_emb must be a positive even number for sinusoidal positions")
        if num_encoder_layers < 0 or num_decoder_layers < 1:
            raise ValueError("FACTR needs >=0 encoder layers and >=1 decoder layer")
        if horizon < 1 or action_dim < 1:
            raise ValueError("horizon and action_dim must be positive")

        self.n_emb = self.d_model = n_emb
        self.n_head = n_head
        self.horizon = horizon
        self.action_dim = action_dim
        self.encoder = _Encoder(
            _EncoderLayer(n_emb, n_head, dim_feedforward, dropout, activation),
            num_encoder_layers,
        )
        self.decoder = _Decoder(
            _DecoderLayer(n_emb, n_head, dim_feedforward, dropout, activation),
            num_decoder_layers,
        )
        # Match upstream: initialize the transformer before creating its
        # separately initialized learned action queries and output projection.
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
        self.pos_helper = _SinusoidalPositionEncoding(n_emb)
        self.ac_query = nn.Embedding(horizon, n_emb)
        self.ac_proj = nn.Linear(n_emb, action_dim)

    def encode(self, tokens):
        """Return all encoded input tokens, with shape [B, S, n_emb]."""
        src = tokens.transpose(0, 1)
        memory = self.encoder(src, self.pos_helper(src))
        return memory.transpose(0, 1)

    def decode(self, memory):
        """Read complete memory [B, S, n_emb], returning [B, horizon, action_dim]."""
        memory = memory.transpose(0, 1)
        query_pos = self.ac_query.weight[:, None].repeat(1, memory.shape[1], 1)
        action_tokens = self.decoder(
            torch.zeros_like(query_pos), memory, self.pos_helper(memory), query_pos
        )
        return self.ac_proj(action_tokens.transpose(0, 1))

    def forward(self, tokens):
        return self.decode(self.encode(tokens))
