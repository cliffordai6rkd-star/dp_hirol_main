"""Shared physical-unit gating and normalization, before any wrench encoder."""

import torch
from torch import nn

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.contact_curriculum import (
    ContactDetector, HysteresisContactDetector, VectorThresholdGate,
)


WRENCH_PREPROCESSING_VERSION = 3


def validate_future_contact_mode(mode):
    if mode not in {"legacy_rollout", "token_threshold"}:
        raise ValueError(
            f"Unsupported future_contact_mode={mode!r}; "
            "expected 'legacy_rollout' or 'token_threshold'"
        )
    return mode


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


def compute_contact_alpha(future_contact):
    """Convert a token-level future-contact mask into the deterministic blend weight.

    ``future_contact`` is ``[B, To, K]``.  A completely free chunk receives
    zero wrench attention, a completely contacting chunk receives one, and a
    partially contacting chunk follows ``(m - 1) / (K - 1)`` where ``m`` is
    the number of contacting tokens.  A one-token chunk is binary by
    definition.
    """
    if not isinstance(future_contact, torch.Tensor) or future_contact.ndim != 3:
        raise ValueError("future_contact must be a bool tensor with shape [B, To, K]")
    if future_contact.dtype != torch.bool:
        raise TypeError("future_contact must be a bool tensor with shape [B, To, K]")
    token_count = int(future_contact.shape[-1])
    if token_count < 1:
        raise ValueError("future_contact must contain at least one token per chunk")
    contact_count = future_contact.sum(dim=-1, dtype=torch.float32)
    if token_count == 1:
        return contact_count
    alpha = (contact_count - 1.0) / float(token_count - 1)
    return alpha.clamp_(0.0, 1.0)


def detect_wrench_contact(raw_wrench, raw_phase_wrench, raw_future_phase_wrench, *,
                         shared_phase_signal, contact_detector, input_gate,
                         future_contact_mode="legacy_rollout"):
    """Shared contact rules for Dataset labels and model-side preprocessing."""
    validate_future_contact_mode(future_contact_mode)
    gate_mask = None
    if shared_phase_signal:
        gate_mask = input_gate.keep_mask(raw_wrench)
        gated_phase_wrench = torch.where(gate_mask.unsqueeze(-1), raw_wrench,
                                         torch.zeros_like(raw_wrench))
    else:
        gated_phase_wrench = raw_phase_wrench
    contact = contact_detector(gated_phase_wrench)
    future_contact = None
    if raw_future_phase_wrench is not None:
        if future_contact_mode == "legacy_rollout" and isinstance(contact_detector, HysteresisContactDetector):
            future_contact = contact_detector.rollout_future(raw_future_phase_wrench, initial_state=contact)
        else:
            future_contact = contact_detector.token_contact(raw_future_phase_wrench)
    return contact, future_contact, gate_mask


def preprocess_wrench(raw_wrench, raw_phase_wrench, raw_future_phase_wrench, *,
                      shared_phase_signal, contact_detector, input_gate, normalizer, wrench_key,
                      future_contact_mode="legacy_rollout", return_processed_wrench=False,
                      return_future_contact=False):
    """The teacher's exact wrench data flow; also used by RenderNet inference."""
    contact, future_contact, gate_mask = detect_wrench_contact(
        raw_wrench, raw_phase_wrench, raw_future_phase_wrench,
        shared_phase_signal=shared_phase_signal, contact_detector=contact_detector,
        input_gate=input_gate, future_contact_mode=future_contact_mode,
    )
    future_has_contact = None
    if future_contact is not None:
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
    # Keep the public preprocessing tuple stable for existing consumers while
    # allowing the teacher fusion path to request the complete token mask.
    if return_future_contact:
        future_result = future_contact if future_contact is not None else contact.unsqueeze(-1)
    else:
        future_result = future_has_contact
    result = (normalized_wrench, contact, future_result)
    if return_processed_wrench:
        result = result + (gated_wrench,)
    return result


class WrenchPreprocessor(nn.Module):
    """Save just the teacher's preprocessing state, without DINO/fusion/DiT."""

    def __init__(self, n_obs_steps, wrench_key, phase_wrench_key, future_phase_wrench_key,
                 contact_detector, input_gate, normalizer, future_contact_mode="legacy_rollout"):
        super().__init__()
        self.n_obs_steps = int(n_obs_steps)
        self.wrench_key = str(wrench_key)
        self.phase_wrench_key = str(phase_wrench_key)
        self.future_phase_wrench_key = future_phase_wrench_key
        self.future_contact_mode = validate_future_contact_mode(future_contact_mode)
        self.contact_detector = contact_detector
        self.input_gate = input_gate
        self.normalizer = normalizer

    @classmethod
    def from_policy(cls, policy):
        return cls(policy.n_obs_steps, policy.wrench_key, policy.phase_wrench_key,
                   policy.future_phase_wrench_key, policy.contact_detector,
                   policy.input_gate, policy.normalizer,
                   future_contact_mode=getattr(policy, "future_contact_mode", "legacy_rollout"))

    @property
    def required_keys(self):
        return tuple(dict.fromkeys(key for key in (self.wrench_key, self.phase_wrench_key,
                                                   self.future_phase_wrench_key) if key is not None))

    def forward(self, obs, *, return_future_contact=False):
        return self._forward(obs, return_future_contact=return_future_contact)

    def contact_alpha(self, obs):
        """Compute a label without normalizing wrench or invoking learned layers."""
        values = {key: obs[key][:, :self.n_obs_steps] for key in self.required_keys}
        contact, future_contact, _ = detect_wrench_contact(
            values[self.wrench_key], values[self.phase_wrench_key],
            values.get(self.future_phase_wrench_key),
            shared_phase_signal=self.phase_wrench_key == self.wrench_key,
            contact_detector=self.contact_detector, input_gate=self.input_gate,
            future_contact_mode=self.future_contact_mode,
        )
        return compute_contact_alpha(future_contact if future_contact is not None else contact.unsqueeze(-1))

    def forward_with_future_contact(self, obs):
        """Return the same preprocessing tuple with the full token mask."""
        return self._forward(obs, return_future_contact=True)

    def _forward(self, obs, *, return_future_contact):
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
            future_contact_mode=self.future_contact_mode,
            return_future_contact=return_future_contact,
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
        return {"version": WRENCH_PREPROCESSING_VERSION,
                "future_contact_mode": self.future_contact_mode,
                "n_obs_steps": self.n_obs_steps, "wrench_key": self.wrench_key,
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
                   VectorThresholdGate(**config["input_gate"]), LinearNormalizer(),
                   future_contact_mode=config.get("future_contact_mode", "legacy_rollout"))
