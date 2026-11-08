"""Factory and shared contract for tau_other sequence models.

The temporal encoders live in separate modules because LSTM, GRU, and TCN
have different state and receptive-field semantics.  This module deliberately
contains no architecture implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


SUPPORTED_TAU_OTHER_ARCHITECTURES = ("gru", "lstm", "tcn")


def _model_config(config):
    value = config.get("model") or {}
    if not isinstance(value, dict):
        raise ValueError("model config must be a mapping")
    return value


def tau_other_architecture(config) -> str:
    architecture = str(_model_config(config).get("architecture", "lstm")).lower()
    if architecture not in SUPPORTED_TAU_OTHER_ARCHITECTURES:
        raise ValueError(
            "model.architecture must be one of "
            f"{list(SUPPORTED_TAU_OTHER_ARCHITECTURES)}, got {architecture!r}."
        )
    return architecture


class TauOtherSequenceModelBase(nn.Module):
    """Input, target, and prediction contract shared by all three branches."""

    DEFAULT_INPUT_DIMS = {
        "q": 7,
        "dq": 7,
        "ddq": 7,
        "delta_q": 7,
        "tau": 7,
    }

    def __init__(self, config, *, architecture: str):
        super().__init__()
        model_config = _model_config(config)
        configured_architecture = tau_other_architecture(config)
        if configured_architecture != architecture:
            raise ValueError(
                f"{type(self).__name__} requires model.architecture={architecture!r}, "
                f"got {configured_architecture!r}."
            )

        self.architecture = architecture
        self.active_inputs = list(
            model_config.get("inputs") or ["q", "dq", "delta_q"]
        )
        if not self.active_inputs:
            raise ValueError("model.inputs must contain at least one input key.")
        if len(set(self.active_inputs)) != len(self.active_inputs):
            raise ValueError(
                f"model.inputs contains duplicate keys: {self.active_inputs}"
            )

        self.input_dims = {
            **self.DEFAULT_INPUT_DIMS,
            **(model_config.get("input_dims") or {}),
        }
        unknown_inputs = [
            key for key in self.active_inputs if key not in self.input_dims
        ]
        if unknown_inputs:
            raise ValueError(f"Missing model.input_dims for inputs: {unknown_inputs}")

        self.history_mode = str(
            model_config.get("history_mode", "stateless_sliding_window")
        ).lower()
        if self.history_mode != "stateless_sliding_window":
            raise ValueError(
                "model.history_mode must be 'stateless_sliding_window'; "
                "tau_other windows do not carry state between samples."
            )

        self.hidden_dim = int(model_config.get("hidden_dim", 128))
        self.num_layers = int(model_config.get("num_layers", 2))
        self.output_dim = int(model_config.get("output_dim", 7))
        self.dropout = float(model_config.get("dropout", 0.1))
        self.target_key = str(model_config.get("target_key", "tau_other"))
        self.input_dim = sum(int(self.input_dims[key]) for key in self.active_inputs)
        if self.hidden_dim <= 0 or self.num_layers <= 0 or self.output_dim <= 0:
            raise ValueError("hidden_dim, num_layers, and output_dim must be positive.")

    @staticmethod
    def _activation(model_config):
        name = str(model_config.get("activation", "relu")).lower()
        activations = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
        }
        if name not in activations:
            raise ValueError(
                f"model.activation must be one of {sorted(activations)}, got {name!r}."
            )
        return activations[name]

    def _build_head(self, model_config):
        head_hidden_dim = int(model_config.get("head_hidden_dim", 256))
        head_num_layers = int(model_config.get("head_num_layers", 2))
        activation = self._activation(model_config)
        if head_hidden_dim <= 0 or head_num_layers <= 0:
            raise ValueError("head_hidden_dim and head_num_layers must be positive.")
        if head_num_layers == 1:
            return nn.Linear(self.hidden_dim, self.output_dim)

        layers = []
        in_dim = self.hidden_dim
        for _ in range(head_num_layers - 1):
            layers.extend(
                [
                    nn.Linear(in_dim, head_hidden_dim),
                    activation(),
                    nn.Dropout(self.dropout),
                ]
            )
            in_dim = head_hidden_dim
        layers.append(nn.Linear(in_dim, self.output_dim))
        return nn.Sequential(*layers)

    def _prepare_inputs(self, batch):
        sequences = []
        batch_size = time_steps = None
        for key in self.active_inputs:
            if key not in batch:
                raise KeyError(f"Missing model input {key!r} in batch.")
            value = batch[key]
            if value.ndim != 3:
                raise ValueError(
                    f"Input {key!r} must have shape [B, H, D], got {tuple(value.shape)}."
                )
            expected_dim = int(self.input_dims[key])
            if value.shape[-1] != expected_dim:
                raise ValueError(
                    f"Input {key!r} has last dimension {value.shape[-1]}, "
                    f"expected {expected_dim}."
                )
            if batch_size is None:
                batch_size, time_steps = value.shape[:2]
            elif value.shape[:2] != (batch_size, time_steps):
                raise ValueError(
                    f"All inputs must share [B, H]; {key!r} has "
                    f"{tuple(value.shape[:2])}, expected {(batch_size, time_steps)}."
                )
            sequences.append(value)
        return torch.cat(sequences, dim=-1)

    @staticmethod
    def _last_target(value):
        if value.ndim == 3:
            return value[:, -1]
        if value.ndim == 2:
            return value
        raise ValueError(
            f"tau_other target must have shape [B, H, D] or [B, D], got {tuple(value.shape)}."
        )

    def _finish_output(self, batch, prediction):
        output = {"tau_other_pred": prediction}
        if self.target_key not in batch:
            return output

        target_value = batch[self.target_key]
        target = self._last_target(target_value)
        if target.shape != prediction.shape:
            raise ValueError(
                f"Prediction shape {tuple(prediction.shape)} does not match "
                f"target shape {tuple(target.shape)}."
            )
        output["tau_other_target"] = target
        return output


def build_tau_other_sequence_model(config) -> TauOtherSequenceModelBase:
    architecture = tau_other_architecture(config)
    if architecture == "lstm":
        from model.tau_other_lstm import TauOtherLSTMRegressor

        return TauOtherLSTMRegressor(config)
    if architecture == "gru":
        from model.tau_other_gru import TauOtherGRURegressor

        return TauOtherGRURegressor(config)
    from model.tau_other_tcn import TauOtherTCNRegressor

    return TauOtherTCNRegressor(config)


class TauOtherSequenceRegressor:
    """Backward-compatible constructor that dispatches to an explicit branch."""

    def __new__(cls, config):
        return build_tau_other_sequence_model(config)
