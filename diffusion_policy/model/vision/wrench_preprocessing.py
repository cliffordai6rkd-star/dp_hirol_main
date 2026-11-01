"""Shared physical-unit gating and normalization, before any wrench encoder."""

import torch
from torch import nn

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.contact_curriculum import (
    ContactDetector, HysteresisContactDetector, VectorThresholdGate,
)


def fill_future_wrench_prefix(raw_wrench, future_contact):
    """Replace a free prefix by the first future contact vector; zero free windows."""
    if raw_wrench.ndim != 4 or future_contact.ndim != 3:
        raise ValueError("raw_wrench must be [B, To, K, D] and future_contact must be [B, To, K]")
    if raw_wrench.shape[:3] != future_contact.shape:
        raise ValueError("raw_wrench and future_contact must share their [B, To, K] dimensions")
    has_contact = future_contact.any(dim=-1)
    first_index = future_contact.to(torch.int64).argmax(dim=-1)
    gather_index = first_index[..., None, None].expand(-1, -1, 1, raw_wrench.shape[-1])
    first_contact_wrench = raw_wrench.gather(2, gather_index).squeeze(2)
    token_index = torch.arange(raw_wrench.shape[2], device=raw_wrench.device).view(1, 1, -1)
    filled = torch.where((token_index < first_index[..., None])[..., None],
                         first_contact_wrench[:, :, None, :], raw_wrench)
    return torch.where(has_contact[:, :, None, None], filled, torch.zeros_like(filled)), has_contact


def preprocess_wrench(raw_wrench, raw_phase_wrench, raw_future_phase_wrench, *,
                      shared_phase_signal, contact_detector, input_gate, normalizer, wrench_key):
    """The teacher's exact wrench data flow; also used by RenderNet inference."""
    if shared_phase_signal:
        gate_mask = input_gate.keep_mask(raw_wrench)
        gated_phase_wrench = torch.where(gate_mask.unsqueeze(-1), raw_wrench,
                                         torch.zeros_like(raw_wrench))
    else:
        gated_phase_wrench = raw_phase_wrench
    contact = contact_detector(gated_phase_wrench)
    future_has_contact = None
    if raw_future_phase_wrench is not None:
        if isinstance(contact_detector, HysteresisContactDetector):
            future_contact = contact_detector.rollout_future(raw_future_phase_wrench, initial_state=contact)
        else:
            future_contact = contact_detector.token_contact(raw_future_phase_wrench)
        gated_wrench, future_has_contact = fill_future_wrench_prefix(raw_wrench, future_contact)
        model_wrench_mask = future_has_contact[:, :, None, None]
    else:
        contact_for_wrench = contact[:, :, None, None]
        gated_wrench = torch.where(contact_for_wrench, raw_wrench, torch.zeros_like(raw_wrench))
        model_wrench_mask = gate_mask.unsqueeze(-1) if shared_phase_signal else contact_for_wrench
    normalized_wrench = normalizer[wrench_key].normalize(gated_wrench)
    # A suppressed physical zero must stay zero after a nonzero-mean normalizer.
    normalized_wrench = torch.where(model_wrench_mask, normalized_wrench,
                                     torch.zeros_like(normalized_wrench))
    return normalized_wrench, contact, future_has_contact


class WrenchPreprocessor(nn.Module):
    """Save just the teacher's preprocessing state, without DINO/fusion/DiT."""

    def __init__(self, n_obs_steps, wrench_key, phase_wrench_key, future_phase_wrench_key,
                 contact_detector, input_gate, normalizer):
        super().__init__()
        self.n_obs_steps = int(n_obs_steps)
        self.wrench_key = str(wrench_key)
        self.phase_wrench_key = str(phase_wrench_key)
        self.future_phase_wrench_key = future_phase_wrench_key
        self.contact_detector = contact_detector
        self.input_gate = input_gate
        self.normalizer = normalizer

    @classmethod
    def from_policy(cls, policy):
        return cls(policy.n_obs_steps, policy.wrench_key, policy.phase_wrench_key,
                   policy.future_phase_wrench_key, policy.contact_detector,
                   policy.input_gate, policy.normalizer)

    @property
    def required_keys(self):
        return tuple(dict.fromkeys(key for key in (self.wrench_key, self.phase_wrench_key,
                                                   self.future_phase_wrench_key) if key is not None))

    def forward(self, obs):
        values = {key: obs[key][:, :self.n_obs_steps] for key in self.required_keys}
        wrench = values[self.wrench_key]
        if any(value.ndim != 4 or value.shape[:2] != wrench.shape[:2]
               or value.shape[1] != self.n_obs_steps for value in values.values()):
            raise ValueError("Wrench and phase observations must have matching [B, To, K, D] windows")
        return preprocess_wrench(
            wrench, values[self.phase_wrench_key], values.get(self.future_phase_wrench_key),
            shared_phase_signal=self.phase_wrench_key == self.wrench_key,
            contact_detector=self.contact_detector, input_gate=self.input_gate,
            normalizer=self.normalizer, wrench_key=self.wrench_key,
        )

    def get_config(self):
        detector = self.contact_detector
        contact = {"force_dims": list(detector.force_dims),
                   "history_reducer": detector.history_reducer, "norm": detector.norm}
        if isinstance(detector, HysteresisContactDetector):
            contact.update(free_threshold=float(detector.free_threshold),
                           contact_threshold=float(detector.contact_threshold),
                           enter_frames=detector.enter_frames,
                           exit_frames=detector.exit_frames)
            kind = "hysteresis"
        elif isinstance(detector, ContactDetector):
            contact["threshold"] = detector.threshold
            kind = "threshold"
        else:
            raise TypeError(f"Unsupported contact detector: {type(detector)}")
        return {"version": 1, "n_obs_steps": self.n_obs_steps, "wrench_key": self.wrench_key,
                "phase_wrench_key": self.phase_wrench_key,
                "future_phase_wrench_key": self.future_phase_wrench_key,
                "contact_detector_type": kind, "contact_detector": contact,
                "input_gate": {"threshold": float(self.input_gate.threshold),
                               "enabled": bool(self.input_gate.enabled), "norm": self.input_gate.norm}}

    @classmethod
    def from_config(cls, config):
        detector_cls = {"hysteresis": HysteresisContactDetector,
                        "threshold": ContactDetector}[config["contact_detector_type"]]
        return cls(config["n_obs_steps"], config["wrench_key"], config["phase_wrench_key"],
                   config["future_phase_wrench_key"], detector_cls(**config["contact_detector"]),
                   VectorThresholdGate(**config["input_gate"]), LinearNormalizer())
