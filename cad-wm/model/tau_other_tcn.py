"""Strictly causal TCN branch for fixed-length tau_other history windows."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.tau_other_sequence import TauOtherSequenceModelBase, _model_config


class _CausalResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout, activation):
        super().__init__()
        self.left_padding = (int(kernel_size) - 1) * int(dilation)
        self.convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size=int(kernel_size),
            dilation=int(dilation),
        )
        self.normalization = nn.LayerNorm(channels)
        self.activation = activation()
        self.dropout = nn.Dropout(dropout)

    def forward(self, value):
        residual = value
        value = F.pad(value, (self.left_padding, 0))
        value = self.convolution(value).transpose(1, 2)
        value = self.normalization(value)
        value = self.dropout(self.activation(value)).transpose(1, 2)
        return residual + value


class TauOtherTCNRegressor(TauOtherSequenceModelBase):
    def __init__(self, config):
        super().__init__(config, architecture="tcn")
        model_config = _model_config(config)
        data_config = config.get("dataloader") or {}
        self.history_horizon = int(data_config.get("horizon", 50))
        if self.history_horizon <= 0:
            raise ValueError("dataloader.horizon must be positive for the TCN.")

        self.tcn_kernel_size = int(model_config.get("tcn_kernel_size", 2))
        if self.tcn_kernel_size < 2:
            raise ValueError("model.tcn_kernel_size must be at least 2.")
        configured_dilations = model_config.get("tcn_dilations")
        if configured_dilations is None:
            if self.tcn_kernel_size != 2:
                raise ValueError(
                    "model.tcn_dilations is required when tcn_kernel_size is not 2."
                )
            dilations = self._exact_receptive_field_dilations(self.history_horizon)
        else:
            dilations = [int(value) for value in configured_dilations]
        if not dilations or any(value <= 0 for value in dilations):
            raise ValueError("model.tcn_dilations must contain positive integers.")

        self.tcn_dilations = tuple(dilations)
        self.temporal_receptive_field = 1 + (
            self.tcn_kernel_size - 1
        ) * sum(self.tcn_dilations)
        if self.temporal_receptive_field != self.history_horizon:
            raise ValueError(
                "TCN receptive field must equal dataloader.horizon so dense and "
                "sliding inference match; got "
                f"{self.temporal_receptive_field} != {self.history_horizon}."
            )

        activation = self._activation(model_config)
        self.tcn_input_projection = nn.Conv1d(
            self.input_dim,
            self.hidden_dim,
            kernel_size=1,
        )
        self.tcn_blocks = nn.ModuleList(
            [
                _CausalResidualBlock(
                    self.hidden_dim,
                    self.tcn_kernel_size,
                    dilation,
                    self.dropout,
                    activation,
                )
                for dilation in self.tcn_dilations
            ]
        )
        self.current_delta_skip = bool(
            model_config.get("current_delta_skip", False)
        )
        current_skip_dim = self.input_dim * (
            2 if self.current_delta_skip else 1
        )
        self.current_state_skip = nn.Linear(
            current_skip_dim,
            self.hidden_dim,
            bias=False,
        )
        self.tcn_output_norm = nn.LayerNorm(self.hidden_dim)
        self.tcn_output_activation = activation()
        self.head = self._build_head(model_config)

    @staticmethod
    def _exact_receptive_field_dilations(horizon):
        remaining = int(horizon) - 1
        dilations = []
        next_power = 1
        while remaining > 0:
            dilation = min(next_power, remaining)
            dilations.append(dilation)
            remaining -= dilation
            next_power *= 2
        return dilations or [1]

    def _encode_sequence(self, batch):
        sequence = self._prepare_inputs(batch)
        temporal = self.tcn_input_projection(sequence.transpose(1, 2))
        for block in self.tcn_blocks:
            temporal = block(temporal)
        temporal = temporal.transpose(1, 2)
        current_features = sequence
        if self.current_delta_skip:
            difference = torch.cat(
                (
                    torch.zeros_like(sequence[:, :1]),
                    sequence[:, 1:] - sequence[:, :-1],
                ),
                dim=1,
            )
            current_features = torch.cat((sequence, difference), dim=-1)
        features = self.tcn_output_activation(
            self.tcn_output_norm(
                temporal + self.current_state_skip(current_features)
            )
        )
        return features

    def forward_sequence(self, batch):
        """Return dense causal predictions for TCN-only rollout acceleration."""
        return self.head(self._encode_sequence(batch))

    def forward(self, batch):
        features = self._encode_sequence(batch)
        return self._finish_output(batch, self.head(features[:, -1]))
