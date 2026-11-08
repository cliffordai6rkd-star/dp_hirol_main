"""Load CAD-WM and return generated physical-unit wrench tokens for DP."""

from __future__ import annotations

import torch

if __package__ and "." in __package__:
    from ..data_process.libero_wrench_dataset import STATE_KEYS, prediction_offsets
    from .cad_wm import CADWorldModel
    from ..train.nomalizer import Normalizer
else:
    from data_process.libero_wrench_dataset import STATE_KEYS, prediction_offsets
    from model.cad_wm import CADWorldModel
    from train.nomalizer import Normalizer


class WrenchPredictor:
    """Input raw q/dq/tau [B,T,7]; output future_wrench [B,K,6] in N/N.m.

    History and predictions use the checkpoint's native rate (20 Hz in the
    LIBERO config); future tokens are t+1 through t+K at that rate.
    History ends at the current observation. Short histories repeat their
    oldest row, matching pad_history=true in training. Start a fresh history
    buffer at each episode reset. DP must apply its own wrench normalizer.
    """

    def __init__(self, checkpoint_path, device="cpu"):
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_version") != CADWorldModel.MODEL_VERSION:
            raise ValueError("checkpoint is not CAD-WM")
        self.config = checkpoint["config"]
        self.model = CADWorldModel(self.config).to(self.device).eval()
        self.model.validate_checkpoint_contract(checkpoint.get("cad_wm_contract"))
        # Shared training checkpoints store inference EMA weights in model.
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.normalizer = None
        if self.config["dataloader"].get("normalize_mode") == "gaussian":
            payload = checkpoint["normalizer"]
            if set((*STATE_KEYS, "wrench")) - set(payload["stats"]):
                raise ValueError("checkpoint is missing input/wrench normalization statistics")
            self.normalizer = Normalizer(payload["stats"], eps=float(payload["eps"]))
        self.future_time = prediction_offsets(self.config).float().to(self.device)

    @torch.no_grad()
    def __call__(self, history, *, source_noise=None, steps=None, solver=None):
        conditions = {}
        history_shape = None
        for key in STATE_KEYS:
            value = torch.as_tensor(history[key], dtype=torch.float32, device=self.device)
            if value.ndim != 3 or value.shape[-1] != self.model.joint_dim or value.shape[1] < 1:
                raise ValueError(f"{key} must have shape [B,T,{self.model.joint_dim}] with T>=1")
            if not torch.isfinite(value).all():
                raise ValueError(f"{key} must contain finite values")
            if history_shape is not None and value.shape != history_shape:
                raise ValueError("q/dq/tau histories must have the same batch size and temporal length")
            history_shape = value.shape
            value = value[:, -self.model.history_horizon:]
            if value.shape[1] < self.model.history_horizon:
                if not self.config["dataloader"].get("pad_history", True):
                    raise ValueError("checkpoint requires a complete state history")
                pad = value[:, :1].expand(-1, self.model.history_horizon - value.shape[1], -1)
                value = torch.cat((pad, value), dim=1)
            if self.normalizer is not None:
                value = self.normalizer.gaussian_normalize(key, value)
            conditions[key] = value
        result = self.model.predict(conditions, source_noise=source_noise, steps=steps, solver=solver)
        wrench = result["wrench_pred"]
        if self.normalizer is not None:
            wrench = self.normalizer.gaussian_denormalize("wrench", wrench)
        return {"future_wrench": wrench, "future_time": self.future_time,
                "contact_probability": result["contact_probability"],
                "contact_state_pred": result["contact_state_pred"]}
