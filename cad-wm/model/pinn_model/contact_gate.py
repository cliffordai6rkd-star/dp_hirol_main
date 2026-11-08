"""Offline contact labels and contact gating shared by torque datasets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class ContactGateConfig:
    enabled: bool = False
    label_mode: str = "binary"
    metric: str = "force_xyz_l2"
    wrench_key: str = "wrench"
    contact_key: str = "contact"
    force_on_threshold_n: float = 0.5
    force_off_threshold_n: float = 0.3
    signal_on_threshold: float | None = None
    signal_off_threshold: float | None = None
    consecutive_frames: int = 3
    head_hidden_dim: int = 64
    probability_threshold: float = 0.5
    positive_class_weight: float | str = "auto"
    label_cache_path: str | None = None
    phase_label_mode: str = "transition_band"
    precontact_frames: int | None = None
    precontact_duration_s: float | None = None
    state_rate_hz: float = 100.0

    @classmethod
    def from_config(cls, config: Mapping):
        values = config.get("contact_gate") or {}
        metric = str(values.get("metric", "force_xyz_l2")).lower()
        threshold_sets = values.get("thresholds") or {}
        metric_thresholds = threshold_sets.get(metric) or {}
        # YAML 1.1 parses unquoted ``on``/``off`` as booleans.
        configured_threshold_on = metric_thresholds.get(
            "on", metric_thresholds.get(True)
        )
        configured_threshold_off = metric_thresholds.get(
            "off", metric_thresholds.get(False)
        )
        configured_on = values.get("on_threshold")
        if configured_on is None:
            configured_on = configured_threshold_on
        configured_off = values.get("off_threshold")
        if configured_off is None:
            configured_off = configured_threshold_off
        positive_weight = values.get("positive_class_weight", "auto")
        if isinstance(positive_weight, str):
            positive_weight = positive_weight.lower()
        else:
            positive_weight = float(positive_weight)
        result = cls(
            enabled=bool(values.get("enabled", False)),
            label_mode=str(values.get("label_mode", "binary")).lower(),
            metric=metric,
            wrench_key=str(values.get("wrench_key", "wrench")),
            contact_key=str(values.get("contact_key", "contact")),
            force_on_threshold_n=float(
                values.get("force_on_threshold_n", 0.5)
            ),
            force_off_threshold_n=float(
                values.get("force_off_threshold_n", 0.3)
            ),
            signal_on_threshold=(
                None
                if configured_on is None
                else float(configured_on)
            ),
            signal_off_threshold=(
                None
                if configured_off is None
                else float(configured_off)
            ),
            consecutive_frames=int(values.get("consecutive_frames", 3)),
            head_hidden_dim=int(values.get("head_hidden_dim", 64)),
            probability_threshold=float(
                values.get("probability_threshold", 0.5)
            ),
            positive_class_weight=positive_weight,
            label_cache_path=(
                None
                if values.get("label_cache_path") is None
                else str(values["label_cache_path"])
            ),
            phase_label_mode=str(
                values.get("phase_label_mode", "transition_band")
            ).lower(),
            precontact_frames=(
                None
                if values.get("precontact_frames") is None
                else int(values["precontact_frames"])
            ),
            precontact_duration_s=(
                None
                if values.get("precontact_duration_s") is None
                else float(values["precontact_duration_s"])
            ),
            state_rate_hz=float(
                (config.get("dataloader") or {}).get("high_fps", 100.0)
            ),
        )
        result.validate()
        return result

    def validate(self):
        if self.label_mode not in {"binary", "three_phase"}:
            raise ValueError(
                "contact_gate.label_mode must be 'binary' or 'three_phase'"
            )
        if self.phase_label_mode not in {"transition_band", "temporal_precontact"}:
            raise ValueError(
                "contact_gate.phase_label_mode must be 'transition_band' or "
                "'temporal_precontact'"
            )
        if self.precontact_frames is not None and self.precontact_frames < 0:
            raise ValueError("contact_gate.precontact_frames must be non-negative")
        if (
            self.precontact_duration_s is not None
            and self.precontact_duration_s < 0.0
        ):
            raise ValueError(
                "contact_gate.precontact_duration_s must be non-negative"
            )
        if (
            self.phase_label_mode == "temporal_precontact"
            and self.precontact_frames is None
            and self.precontact_duration_s is None
        ):
            raise ValueError(
                "temporal_precontact requires precontact_frames or "
                "precontact_duration_s"
            )
        if not math.isfinite(self.state_rate_hz) or self.state_rate_hz <= 0.0:
            raise ValueError("dataloader.high_fps must be positive")
        if self.metric not in {"force_xyz_l2", "wrench_l2", "tau_ext_l1", "tau_ext_l2"}:
            raise ValueError(
                "contact_gate.metric must be force_xyz_l2, wrench_l2, tau_ext_l1, or tau_ext_l2"
            )
        off_threshold = self.off_threshold
        on_threshold = self.on_threshold
        if off_threshold < 0.0:
            raise ValueError("contact_gate.off_threshold must be non-negative")
        if on_threshold <= off_threshold:
            raise ValueError(
                "contact_gate.on_threshold must be greater than off_threshold"
            )
        if not 0.0 < self.probability_threshold < 1.0:
            raise ValueError(
                "contact_gate.probability_threshold must be in (0, 1)"
            )
        if self.consecutive_frames < 1:
            raise ValueError("contact_gate.consecutive_frames must be positive")
        if self.head_hidden_dim < 1:
            raise ValueError("contact_gate.head_hidden_dim must be positive")
        if not self.wrench_key or not self.contact_key:
            raise ValueError("contact_gate wrench_key/contact_key must not be empty")
        weight = self.positive_class_weight
        if weight != "auto" and (not isinstance(weight, float) or weight <= 0.0):
            raise ValueError(
                "contact_gate.positive_class_weight must be 'auto' or positive"
            )

    @property
    def on_threshold(self) -> float:
        return (
            self.force_on_threshold_n
            if self.signal_on_threshold is None
            else self.signal_on_threshold
        )

    @property
    def off_threshold(self) -> float:
        return (
            self.force_off_threshold_n
            if self.signal_off_threshold is None
            else self.signal_off_threshold
        )


def hysteresis_binary_mask(
    signal: torch.Tensor,
    *,
    on_threshold: float,
    off_threshold: float,
    consecutive_frames: int,
    backfill: bool = True,
) -> torch.Tensor:
    """Convert one scalar episode timeline into a stable binary contact mask."""

    values = torch.as_tensor(signal)
    if values.ndim != 1:
        raise ValueError(f"hysteresis signal must have shape [T], got {values.shape}")
    if on_threshold <= off_threshold:
        raise ValueError("on_threshold must be greater than off_threshold")
    if consecutive_frames < 1:
        raise ValueError("consecutive_frames must be positive")

    mask = torch.zeros_like(values, dtype=torch.float32)
    contact = False
    candidate_count = 0
    for index in range(values.shape[0]):
        value = float(values[index])
        if not contact:
            if value >= on_threshold:
                candidate_count += 1
            else:
                candidate_count = 0
            if candidate_count >= consecutive_frames:
                contact = True
                start = index - consecutive_frames + 1 if backfill else index
                mask[start : index + 1] = 1.0
                candidate_count = 0
        else:
            mask[index] = 1.0
            if value <= off_threshold:
                candidate_count += 1
            else:
                candidate_count = 0
            if candidate_count >= consecutive_frames:
                contact = False
                if backfill:
                    start = index - consecutive_frames + 1
                    mask[start : index + 1] = 0.0
                candidate_count = 0
    return mask


def contact_labels_from_wrench(
    wrench: torch.Tensor,
    episode_bounds: Sequence[tuple[int, int]],
    config: ContactGateConfig,
) -> torch.Tensor:
    """Create [T, 1] labels while resetting hysteresis at episode boundaries."""

    values = torch.as_tensor(wrench)
    if values.ndim != 2 or values.shape[-1] != 6:
        raise ValueError(f"wrench must have shape [T, 6], got {values.shape}")
    labels = torch.zeros((values.shape[0], 1), dtype=torch.float32)
    for start, end in episode_bounds:
        start = int(start)
        end = int(end)
        if start < 0 or end <= start or end > values.shape[0]:
            raise ValueError(f"invalid episode bounds [{start}, {end})")
        force_norm = torch.linalg.vector_norm(values[start:end, :3], dim=-1)
        labels[start:end, 0] = hysteresis_binary_mask(
            force_norm,
            on_threshold=config.on_threshold,
            off_threshold=config.off_threshold,
            consecutive_frames=config.consecutive_frames,
            backfill=True,
        )
    return labels


def hysteresis_three_phase_mask(
    signal: torch.Tensor,
    *,
    on_threshold: float,
    off_threshold: float,
    consecutive_frames: int,
    backfill: bool = True,
) -> torch.Tensor:
    """Create causal free/precontact/contact labels from a scalar signal.

    State ``1`` is the rising transition band. Once ``on_threshold`` has been
    observed for ``consecutive_frames``, the state becomes ``2`` and remains
    contact until ``off_threshold`` is observed for the same confirmation
    length. The confirmed transition can be backfilled because labels are
    generated offline.
    """

    values = torch.as_tensor(signal)
    if values.ndim != 1:
        raise ValueError(f"hysteresis signal must have shape [T], got {values.shape}")
    if not torch.isfinite(values).all():
        raise ValueError("hysteresis signal must contain finite values")
    if on_threshold <= off_threshold:
        raise ValueError("on_threshold must be greater than off_threshold")
    if consecutive_frames < 1:
        raise ValueError("consecutive_frames must be positive")

    labels = torch.zeros_like(values, dtype=torch.float32)
    state = 0  # free=0, precontact=1, contact=2
    candidate_count = 0
    release_count = 0
    for index in range(values.shape[0]):
        value = float(values[index])
        if state == 2:
            labels[index] = 2.0
            if value <= off_threshold:
                release_count += 1
            else:
                release_count = 0
            if release_count >= consecutive_frames:
                state = 0
                start = index - consecutive_frames + 1 if backfill else index
                labels[start : index + 1] = 0.0
                release_count = 0
            continue

        if value <= off_threshold:
            state = 0
            candidate_count = 0
            labels[index] = 0.0
            continue

        # Values between the thresholds are explicitly exposed as the
        # precontact class, rather than being collapsed into free/contact.
        labels[index] = 1.0
        if value >= on_threshold:
            candidate_count += 1
        else:
            candidate_count = 0
        if candidate_count >= consecutive_frames:
            state = 2
            start = index - consecutive_frames + 1 if backfill else index
            labels[start : index + 1] = 2.0
            candidate_count = 0

    return labels


@torch.no_grad()
def batched_hysteresis_three_phase_mask(
    signal: torch.Tensor,
    *,
    on_threshold: float,
    off_threshold: float,
    consecutive_frames: int,
    backfill: bool = True,
) -> torch.Tensor:
    """Apply the three-phase state machine to independent ``[B, T]`` rows."""

    values = torch.as_tensor(signal).detach()
    if values.ndim != 2:
        raise ValueError(
            f"batched hysteresis signal must have shape [B, T], got {values.shape}"
        )
    if not torch.isfinite(values).all():
        raise ValueError("hysteresis signal must contain finite values")
    if on_threshold <= off_threshold:
        raise ValueError("on_threshold must be greater than off_threshold")
    if consecutive_frames < 1:
        raise ValueError("consecutive_frames must be positive")

    batch_size, horizon = values.shape
    labels = torch.zeros(
        (batch_size, horizon), device=values.device, dtype=torch.float32
    )
    contact = torch.zeros(batch_size, device=values.device, dtype=torch.bool)
    candidate_count = torch.zeros(
        batch_size, device=values.device, dtype=torch.long
    )
    release_count = torch.zeros_like(candidate_count)
    zero = labels.new_zeros(())
    one = labels.new_ones(())
    two = one + one

    # Hysteresis is sequential in time but independent across rows. Keeping
    # only the short time loop avoids the previous B*T Python scalar loop and
    # the device-to-host round trip without changing the state-machine rules.
    for index in range(horizon):
        value = values[:, index]
        was_contact = contact

        labels[:, index] = torch.where(
            was_contact,
            two,
            labels[:, index],
        )
        release_count = torch.where(
            was_contact,
            torch.where(value <= off_threshold, release_count + 1, 0),
            0,
        )
        released = was_contact & (release_count >= consecutive_frames)
        contact = was_contact & ~released
        release_count = torch.where(released, 0, release_count)
        if backfill:
            start = max(index - consecutive_frames + 1, 0)
            segment = labels[:, start : index + 1]
            labels[:, start : index + 1] = torch.where(
                released[:, None],
                torch.zeros_like(segment),
                segment,
            )
        else:
            labels[:, index] = torch.where(
                released,
                zero,
                labels[:, index],
            )

        # A row released on this frame follows the scalar implementation's
        # ``continue`` path and is not reconsidered as a new rising edge.
        active = ~was_contact
        rising = active & (value > off_threshold)
        labels[:, index] = torch.where(
            rising,
            one,
            labels[:, index],
        )
        candidate_count = torch.where(
            active,
            torch.where(
                rising & (value >= on_threshold),
                candidate_count + 1,
                0,
            ),
            0,
        )
        entered = active & (candidate_count >= consecutive_frames)
        contact = contact | entered
        candidate_count = torch.where(entered, 0, candidate_count)
        if backfill:
            start = max(index - consecutive_frames + 1, 0)
            segment = labels[:, start : index + 1]
            labels[:, start : index + 1] = torch.where(
                entered[:, None],
                torch.full_like(segment, 2.0),
                segment,
            )
        else:
            labels[:, index] = torch.where(
                entered,
                two,
                labels[:, index],
            )

    return labels


def contact_phase_labels_from_wrench(
    wrench: torch.Tensor,
    episode_bounds: Sequence[tuple[int, int]],
    config: ContactGateConfig,
) -> torch.Tensor:
    """Create [T, 1] three-phase labels from the measured wrench force."""

    values = torch.as_tensor(wrench)
    if values.ndim != 2 or values.shape[-1] != 6:
        raise ValueError(f"wrench must have shape [T, 6], got {values.shape}")
    if config.metric == "force_xyz_l2":
        signal = torch.linalg.vector_norm(values[:, :3], dim=-1)
    elif config.metric == "wrench_l2":
        signal = torch.linalg.vector_norm(values, dim=-1)
    else:
        raise ValueError(
            "tau_ext_l1 labels require a separate tau_ext tensor; use "
            "contact_phase_labels_from_signal"
        )
    return contact_phase_labels_from_signal(signal, episode_bounds, config)


def contact_phase_labels_from_signal(
    signal: torch.Tensor,
    episode_bounds: Sequence[tuple[int, int]],
    config: ContactGateConfig,
) -> torch.Tensor:
    """Create [T, 1] three-phase labels while resetting each episode."""

    values = torch.as_tensor(signal)
    if values.ndim != 1:
        raise ValueError(f"signal must have shape [T], got {values.shape}")
    labels = torch.zeros((values.shape[0], 1), dtype=torch.float32)
    for start, end in episode_bounds:
        start = int(start)
        end = int(end)
        if start < 0 or end <= start or end > values.shape[0]:
            raise ValueError(f"invalid episode bounds [{start}, {end})")
        episode_signal = values[start:end]
        if config.phase_label_mode == "transition_band":
            episode_labels = hysteresis_three_phase_mask(
                episode_signal,
                on_threshold=config.on_threshold,
                off_threshold=config.off_threshold,
                consecutive_frames=config.consecutive_frames,
                backfill=True,
            )
        else:
            contact = hysteresis_binary_mask(
                episode_signal,
                on_threshold=config.on_threshold,
                off_threshold=config.off_threshold,
                consecutive_frames=config.consecutive_frames,
                backfill=True,
            ).to(dtype=torch.bool)
            episode_labels = torch.zeros_like(episode_signal, dtype=torch.float32)
            episode_labels[contact] = 2.0
            precontact_frames = config.precontact_frames
            if precontact_frames is None:
                state_rate_hz = float(getattr(config, "state_rate_hz", 100.0))
                precontact_frames = int(
                    round(float(config.precontact_duration_s or 0.0) * state_rate_hz)
                )
            onset = contact & ~torch.cat(
                (torch.zeros(1, dtype=torch.bool, device=contact.device), contact[:-1])
            )
            for onset_index in torch.nonzero(onset, as_tuple=False).reshape(-1).tolist():
                pre_start = max(0, int(onset_index) - int(precontact_frames))
                episode_labels[pre_start:onset_index] = 1.0
        labels[start:end, 0] = episode_labels
    return labels


def probability_contact_mask(
    probability: torch.Tensor,
    config: ContactGateConfig,
) -> torch.Tensor:
    values = torch.as_tensor(probability)
    original_shape = values.shape
    if values.ndim == 2 and values.shape[-1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError(
            f"contact probability must have shape [T] or [T, 1], got {original_shape}"
        )
    return (values >= config.probability_threshold).to(torch.float32)[:, None]


def save_contact_label_cache(
    path: Path,
    labels: torch.Tensor,
    episode_bounds: Sequence[tuple[int, int]],
    config: ContactGateConfig,
):
    values = torch.as_tensor(labels).detach().cpu().numpy().astype(np.float32)
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError(f"contact labels must have shape [T, 1], got {values.shape}")
    bounds = np.asarray(episode_bounds, dtype=np.int64)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("episode_bounds must have shape [E, 2]")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        contact=values,
        episode_bounds=bounds,
        force_on_threshold_n=np.asarray(config.on_threshold),
        force_off_threshold_n=np.asarray(config.off_threshold),
        consecutive_frames=np.asarray(config.consecutive_frames, dtype=np.int64),
    )
    temporary.replace(destination)


def load_contact_label_cache(
    path: Path,
    *,
    flow_length: int,
    episode_bounds: Sequence[tuple[int, int]],
    config: ContactGateConfig,
) -> torch.Tensor:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"contact label cache does not exist: {source}")
    with np.load(source, allow_pickle=False) as payload:
        required = {
            "contact",
            "episode_bounds",
            "force_on_threshold_n",
            "force_off_threshold_n",
            "consecutive_frames",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"contact label cache is missing fields: {missing}")
        labels = np.asarray(payload["contact"], dtype=np.float32)
        cached_bounds = np.asarray(payload["episode_bounds"], dtype=np.int64)
        cached_on = float(payload["force_on_threshold_n"])
        cached_off = float(payload["force_off_threshold_n"])
        cached_frames = int(payload["consecutive_frames"])

    expected_bounds = np.asarray(episode_bounds, dtype=np.int64)
    if labels.shape != (flow_length, 1):
        raise ValueError(
            "contact label cache length mismatch: "
            f"{labels.shape} vs {(flow_length, 1)}"
        )
    if not np.array_equal(cached_bounds, expected_bounds):
        raise ValueError("contact label cache episode bounds do not match dataset")
    if (
        cached_on != config.on_threshold
        or cached_off != config.off_threshold
        or cached_frames != config.consecutive_frames
    ):
        raise ValueError(
            "contact label cache thresholds do not match config: "
            f"cache=({cached_on}, {cached_off}, {cached_frames}) "
            f"config=({config.force_on_threshold_n}, "
            f"{config.force_off_threshold_n}, {config.consecutive_frames})"
        )
    return torch.from_numpy(labels.copy())
