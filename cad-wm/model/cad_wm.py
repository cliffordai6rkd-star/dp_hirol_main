"""CAD-WM: motion/torque LSTM conditions and a future-wrench flow."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

if __package__ and "." in __package__:
    from ..data_process.libero_wrench_dataset import STATE_KEYS, prediction_offsets
else:
    from data_process.libero_wrench_dataset import STATE_KEYS, prediction_offsets


def _attention(module, query, memory):
    return module(query.contiguous(), memory.contiguous(), memory.contiguous(),
                  need_weights=query.device.type == "npu")[0]


def _lstm(module, value):
    if value.device.type != "npu":
        return module(value)[0]
    with torch.autocast(device_type="npu", enabled=False):
        sequence, _ = module(value.float().contiguous())
    return sequence.to(value.dtype)


class FlowTimeEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(5, hidden_dim), nn.SiLU(),
                                        nn.Linear(hidden_dim, hidden_dim))

    def forward(self, time):
        features = torch.cat((time, torch.sin(math.pi * time), torch.cos(math.pi * time),
                              torch.sin(2 * math.pi * time), torch.cos(2 * math.pi * time)), -1)
        return self.projection(features)


class DualConditionFlowBlock(nn.Module):
    """Future self-attention followed by parallel motion and torque attention."""

    def __init__(self, dim, heads, multiplier, dropout):
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.motion_norm = nn.LayerNorm(dim)
        self.torque_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.motion_cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.torque_cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, multiplier * dim), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(multiplier * dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, trajectory, motion, torque):
        normalized = self.self_norm(trajectory)
        trajectory = trajectory + self.dropout(_attention(self.self_attention, normalized, normalized))
        # Both branches read the same future state and own separate Q/K/V projections.
        motion_update = _attention(self.motion_cross_attn, self.motion_norm(trajectory), motion)
        torque_update = _attention(self.torque_cross_attn, self.torque_norm(trajectory), torque)
        trajectory = trajectory + self.dropout(motion_update) + self.dropout(torque_update)
        return trajectory + self.dropout(self.ffn(self.ffn_norm(trajectory)))


class CADWorldModel(nn.Module):
    """Generate future wrench from two independent LSTM condition sequences.

    q/dq are concatenated per row before the motion LSTM; tau has its own
    LSTM. Only the current raw motion feature supervises the NEXT auxiliary
    head. predict() never reads targets or executes that auxiliary head.
    """

    MODEL_VERSION = "cad-wm"
    CHECKPOINT_CONTRACT_KEY = "cad_wm_contract"
    CONDITION_KEYS = STATE_KEYS
    TARGET_KEYS = ("wrench_future", "contact_future")
    is_deterministic = False

    def __init__(self, config):
        super().__init__()
        data, model = config["dataloader"], config["model"]
        if tuple(model["inputs"]) != STATE_KEYS or list(model["outputs"]) != ["wrench"]:
            raise ValueError("CAD-WM requires inputs=[q,dq,tau], outputs=[wrench]")
        if (config.get("train") or {}).get("downsample", False):
            raise ValueError("CAD-WM uses the native input/output timeline")
        self.history_horizon = int(data["state_history_horizon"])
        self.future_horizon = len(prediction_offsets(config))
        self.joint_dim = int(model.get("joint_dim", 7))
        dim = int(model.get("hidden_dim", 128))
        layers = int(model.get("lstm_layers", 2))
        heads = int(model.get("attention_heads", 4))
        blocks = int(model.get("flow_layers", 4))
        multiplier = int(model.get("ffn_multiplier", 4))
        mlp_dim = int(model.get("free_dynamics_hidden_dim", dim))
        dropout = float(model.get("dropout", 0.01))
        self.flow_inference_steps = int(model.get("flow_inference_steps", 8))
        self.flow_solver = str(model.get("flow_solver", "heun")).lower()
        if min(self.history_horizon, self.joint_dim, dim, layers, heads, blocks, multiplier,
               mlp_dim, self.flow_inference_steps) < 1:
            raise ValueError("model dimensions, layers, horizons and integration steps must be positive")
        if dim % heads or not 0 <= dropout < 1 or self.flow_solver not in {"euler", "heun"}:
            raise ValueError("invalid attention dimensions, dropout, or flow solver")
        self.motion_lstm = nn.LSTM(2 * self.joint_dim, dim, layers, batch_first=True,
                                   dropout=dropout if layers > 1 else 0.0)
        self.torque_lstm = nn.LSTM(self.joint_dim, dim, layers, batch_first=True,
                                   dropout=dropout if layers > 1 else 0.0)
        self.motion_embedding = nn.Parameter(torch.randn(dim) * 0.02)
        self.torque_embedding = nn.Parameter(torch.randn(dim) * 0.02)
        self.history_pos_embedding = nn.Embedding(self.history_horizon, dim)
        self.future_pos_embedding = nn.Embedding(self.future_horizon, dim)
        self.motion_token_norm = nn.LayerNorm(dim)
        self.torque_token_norm = nn.LayerNorm(dim)
        self.free_dynamics_head = nn.Sequential(nn.Linear(dim, mlp_dim), nn.SiLU(),
                                                nn.Linear(mlp_dim, self.joint_dim))
        self.flow_input = nn.Linear(6, dim)
        self.flow_time_embedding = FlowTimeEmbedding(dim)
        self.flow_blocks = nn.ModuleList([
            DualConditionFlowBlock(dim, heads, multiplier, dropout) for _ in range(blocks)
        ])
        self.flow_output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 6))
        # The categorical head reads conditions alone, so it has the same
        # information during training and inference, independent of CFM targets.
        self.contact_head = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim),
                                           nn.SiLU(), nn.Linear(dim, 3))
        self._contract = copy.deepcopy({
            "model_name": "cad-wm", "model": model, "history_horizon": self.history_horizon,
            "future_offsets_s": prediction_offsets(config).tolist(),
            "target_sampling": "consecutive_native_rows", "high_fps": data["high_fps"],
            "h5_fields": data["h5_fields"], "normalize_mode": data.get("normalize_mode"),
            "contact_gate": config["contact_gate"], "loss": config.get("loss", {}),
            "contact_sampling": (config.get("train") or {}).get("contact_sampling", {}),
        })

    def checkpoint_contract(self):
        return copy.deepcopy(self._contract)

    def validate_checkpoint_contract(self, actual):
        if actual != self.checkpoint_contract():
            raise ValueError("CAD-WM checkpoint architecture/data contract differs")

    def encode_conditions(self, batch, *, compute_free_dynamics=False):
        reference = batch["q"]
        for key in STATE_KEYS:
            value = batch[key]
            if (value.ndim != 3 or value.shape[1:] != (self.history_horizon, self.joint_dim)
                    or not value.is_floating_point()):
                raise ValueError(f"{key} must be floating [B,{self.history_horizon},{self.joint_dim}]")
            if value.shape != reference.shape or value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError("q/dq/tau must share shape, device and dtype")
        motion = _lstm(self.motion_lstm, torch.cat((batch["q"], batch["dq"]), -1))
        torque = _lstm(self.torque_lstm, batch["tau"])
        positions = torch.arange(self.history_horizon - 1, -1, -1, device=reference.device)
        history_pe = self.history_pos_embedding(positions)[None].to(motion.dtype)
        result = {
            "motion_tokens": self.motion_token_norm(motion + self.motion_embedding + history_pe),
            "torque_tokens": self.torque_token_norm(torque + self.torque_embedding + history_pe),
        }
        if compute_free_dynamics:
            result["free_tau_pred"] = self.free_dynamics_head(motion[:, -1])
        return result

    def _source(self, reference, source_noise):
        shape = (reference.shape[0], self.future_horizon, 6)
        if source_noise is None:
            return torch.randn(shape, device=reference.device, dtype=reference.dtype)
        if not torch.is_tensor(source_noise) or tuple(source_noise.shape) != shape:
            raise ValueError(f"source_noise must have shape {shape}")
        source = source_noise.to(reference)
        if not torch.isfinite(source).all():
            raise ValueError("source_noise must be finite")
        return source

    @staticmethod
    def _time(reference, flow_time):
        count = reference.shape[0]
        if flow_time is None:
            return torch.rand(count, 1, device=reference.device, dtype=reference.dtype)
        time = torch.as_tensor(flow_time, device=reference.device, dtype=reference.dtype)
        if time.ndim == 0:
            time = time.expand(count, 1)
        elif time.shape == (count,):
            time = time[:, None]
        if time.shape != (count, 1) or not torch.isfinite(time).all() or ((time < 0) | (time > 1)).any():
            raise ValueError("flow_time must be scalar, [B], or [B,1], finite and in [0,1]")
        return time

    def flow_velocity(self, trajectory, time, encoded):
        positions = torch.arange(self.future_horizon, device=trajectory.device)
        features = (self.flow_input(trajectory) + self.flow_time_embedding(time)[:, None]
                    + self.future_pos_embedding(positions)[None].to(trajectory.dtype))
        for block in self.flow_blocks:
            features = block(features, encoded["motion_tokens"], encoded["torque_tokens"])
        return self.flow_output(features)

    def _contact_logits(self, encoded):
        positions = torch.arange(self.future_horizon, device=encoded["motion_tokens"].device)
        pe = self.future_pos_embedding(positions)[None]
        motion = encoded["motion_tokens"].mean(1)[:, None] + pe
        torque = encoded["torque_tokens"].mean(1)[:, None] + pe
        return self.contact_head(torch.cat((motion, torque), -1))

    def forward(self, batch, *, flow_time=None, source_noise=None):
        encoded = self.encode_conditions(batch, compute_free_dynamics=True)
        target = batch["wrench_future"]
        source = self._source(batch["q"], source_noise)
        if target.shape != source.shape:
            raise ValueError("wrench_future must have shape [B,prediction_horizon,6]")
        time = self._time(source, flow_time)
        # Interpolate Gaussian source and target in flow time only. Recorded
        # HDF5 targets remain consecutive, unmodified native-rate samples.
        intermediate = (1 - time[:, None]) * source + time[:, None] * target
        velocity = self.flow_velocity(intermediate, time, encoded)
        return {**encoded, "flow_velocity_pred": velocity, "flow_velocity_target": target - source,
                "flow_source_noise": source, "flow_time": time,
                "wrench_pred": intermediate + (1 - time[:, None]) * velocity,
                "contact_logits": self._contact_logits(encoded)}

    def integrate_flow(self, source, encoded, *, steps=None, solver=None):
        steps = self.flow_inference_steps if steps is None else int(steps)
        solver = self.flow_solver if solver is None else str(solver).lower()
        if steps < 1 or solver not in {"euler", "heun"}:
            raise ValueError("flow requires positive steps and solver=euler or heun")
        trajectory = source
        dt = 1.0 / steps
        for index in range(steps):
            time = source.new_full((source.shape[0], 1), index / steps)
            first = self.flow_velocity(trajectory, time, encoded)
            proposal = trajectory + dt * first
            if solver == "euler":
                trajectory = proposal
            else:
                next_time = source.new_full((source.shape[0], 1), (index + 1) / steps)
                second = self.flow_velocity(proposal, next_time, encoded)
                trajectory = trajectory + 0.5 * dt * (first + second)
        return trajectory

    @torch.no_grad()
    def predict(self, batch, *, source_noise=None, steps=None, solver=None):
        encoded = self.encode_conditions(batch)
        source = self._source(batch["q"], source_noise)
        logits = self._contact_logits(encoded)
        return {"wrench_pred": self.integrate_flow(source, encoded, steps=steps, solver=solver),
                "contact_logits": logits, "contact_probability": logits.softmax(-1),
                "contact_state_pred": logits.argmax(-1)}
