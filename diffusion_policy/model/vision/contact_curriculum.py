import bisect
import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContactDetector(nn.Module):
    """Detect contact from raw, physical-unit wrench history."""

    def __init__(
        self,
        threshold: float,
        force_dims: Sequence[int] = (0, 1, 2),
        history_reducer: str = "max",
        norm: str = "l1",
    ) -> None:
        super().__init__()
        if threshold < 0:
            raise ValueError("contact threshold must be non-negative")
        if not force_dims:
            raise ValueError("force_dims cannot be empty")
        if history_reducer not in {"last", "max", "mean"}:
            raise ValueError(
                f"Unsupported history_reducer={history_reducer!r}; "
                "expected 'last', 'max', or 'mean'."
            )
        if norm not in {"l1", "l2", "sum"}:
            raise ValueError("norm must be one of 'l1', 'l2', or 'sum'")
        self.threshold = float(threshold)
        self.force_dims = tuple(int(dim) for dim in force_dims)
        self.history_reducer = history_reducer
        self.norm = "l1" if norm == "sum" else norm

    def _magnitude(self, raw_wrench: torch.Tensor) -> torch.Tensor:
        if raw_wrench.ndim < 1:
            raise ValueError("raw_wrench must have at least one dimension")
        if max(self.force_dims) >= raw_wrench.shape[-1]:
            raise ValueError(
                f"force_dims={self.force_dims} exceed wrench dim {raw_wrench.shape[-1]}"
            )
        force = raw_wrench[..., list(self.force_dims)]
        if self.norm == "l1":
            return force.abs().sum(dim=-1)
        return torch.linalg.vector_norm(force, ord=2, dim=-1)

    def forward(self, raw_wrench: torch.Tensor) -> torch.Tensor:
        if raw_wrench.ndim != 4:
            raise ValueError(
                "raw_wrench must have shape [B, To, K, D], "
                f"got {tuple(raw_wrench.shape)}"
            )
        if max(self.force_dims) >= raw_wrench.shape[-1]:
            raise ValueError(
                f"force_dims={self.force_dims} exceed wrench dim {raw_wrench.shape[-1]}"
            )

        magnitude = self._magnitude(raw_wrench)
        if self.history_reducer == "last":
            reduced = magnitude[..., -1]
        elif self.history_reducer == "max":
            reduced = magnitude.amax(dim=-1)
        else:
            reduced = magnitude.mean(dim=-1)
        return reduced > self.threshold

    def token_contact(self, raw_wrench: torch.Tensor) -> torch.Tensor:
        """Return a contact mask for every wrench token.

        This is used for a future window where each token needs its own phase
        state instead of reducing the complete history to one boolean.
        """
        return self._magnitude(raw_wrench) > self.threshold


class HysteresisContactDetector(nn.Module):
    """Causal contact state detector with optional consecutive-frame debounce.

    The detector enters contact above ``contact_threshold`` and leaves it only
    below ``free_threshold``.  ``enter_frames`` and ``exit_frames`` require
    that many consecutive samples before changing state.  Input is
    ``[B, To, K, D]`` and the returned state is ``[B, To]``.  With debounce
    enabled, the K wrench samples are scanned in temporal order and the state
    is carried from one observation window to the next.
    """

    def __init__(
        self,
        free_threshold: float,
        contact_threshold: float,
        force_dims: Sequence[int] = (0, 1, 2),
        history_reducer: str = "max",
        norm: str = "l1",
        enter_frames: int = 1,
        exit_frames: int = 1,
    ) -> None:
        super().__init__()
        if free_threshold < 0 or contact_threshold < 0:
            raise ValueError("contact thresholds must be non-negative")
        if free_threshold > contact_threshold:
            raise ValueError("free_threshold must not exceed contact_threshold")
        if not force_dims:
            raise ValueError("force_dims cannot be empty")
        if history_reducer not in {"last", "max", "mean"}:
            raise ValueError(
                f"Unsupported history_reducer={history_reducer!r}; "
                "expected 'last', 'max', or 'mean'."
            )
        if norm not in {"l1", "l2", "sum"}:
            raise ValueError("norm must be one of 'l1', 'l2', or 'sum'")
        if int(enter_frames) < 1 or int(exit_frames) < 1:
            raise ValueError("enter_frames and exit_frames must be positive")
        self.register_buffer("free_threshold", torch.tensor(float(free_threshold)))
        self.register_buffer("contact_threshold", torch.tensor(float(contact_threshold)))
        self.force_dims = tuple(int(dim) for dim in force_dims)
        self.history_reducer = history_reducer
        self.norm = "l1" if norm == "sum" else norm
        self.enter_frames = int(enter_frames)
        self.exit_frames = int(exit_frames)

    def _reduced_magnitude(self, raw_wrench: torch.Tensor) -> torch.Tensor:
        if raw_wrench.ndim != 4:
            raise ValueError(
                "raw_wrench must have shape [B, To, K, D], "
                f"got {tuple(raw_wrench.shape)}"
            )
        if max(self.force_dims) >= raw_wrench.shape[-1]:
            raise ValueError(
                f"force_dims={self.force_dims} exceed wrench dim {raw_wrench.shape[-1]}"
            )
        if self.norm == "l1":
            force = raw_wrench[..., list(self.force_dims)]
            magnitude = force.abs().sum(dim=-1)
        else:
            force = raw_wrench[..., list(self.force_dims)]
            magnitude = torch.linalg.vector_norm(force, ord=2, dim=-1)
        if self.history_reducer == "last":
            return magnitude[..., -1]
        if self.history_reducer == "max":
            return magnitude.amax(dim=-1)
        return magnitude.mean(dim=-1)

    def forward(self, raw_wrench: torch.Tensor) -> torch.Tensor:
        if self.enter_frames == 1 and self.exit_frames == 1:
            magnitude = self._reduced_magnitude(raw_wrench)
            batch_size, obs_steps = magnitude.shape
            result = torch.zeros_like(magnitude, dtype=torch.bool)
            contact_threshold = self.contact_threshold.to(magnitude)
            free_threshold = self.free_threshold.to(magnitude)
            if obs_steps == 0:
                return result
            result[:, 0] = magnitude[:, 0] > contact_threshold
            for index in range(1, obs_steps):
                result[:, index] = torch.where(
                    result[:, index - 1],
                    magnitude[:, index] >= free_threshold,
                    magnitude[:, index] > contact_threshold,
                )
            return result

        # Consecutive-frame debounce must inspect every wrench token.  Flatten
        # [To, K] while retaining the state at the end of each observation.
        magnitude = self._per_token_magnitude(raw_wrench)
        states = self._scan_tokens(magnitude.reshape(magnitude.shape[0], -1))
        return states.reshape_as(magnitude)[..., -1]

    def _scan_tokens(
        self, magnitude: torch.Tensor, initial_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Scan ``[B, N]`` magnitudes and return the latched state per token."""
        if magnitude.ndim != 2:
            raise ValueError(f"magnitude must have shape [B, N], got {tuple(magnitude.shape)}")
        batch_size, token_count = magnitude.shape
        state = (
            torch.zeros((batch_size,), dtype=torch.bool, device=magnitude.device)
            if initial_state is None
            else initial_state.to(device=magnitude.device, dtype=torch.bool).reshape(batch_size)
        )
        result = torch.zeros_like(magnitude, dtype=torch.bool)
        above_count = torch.zeros((batch_size,), dtype=torch.int64, device=magnitude.device)
        below_count = torch.zeros((batch_size,), dtype=torch.int64, device=magnitude.device)
        contact_threshold = self.contact_threshold.to(magnitude)
        free_threshold = self.free_threshold.to(magnitude)
        for index in range(token_count):
            value = magnitude[:, index]
            above = value > contact_threshold
            below = value < free_threshold
            above_count = torch.where(above, above_count + 1, torch.zeros_like(above_count))
            below_count = torch.where(below, below_count + 1, torch.zeros_like(below_count))
            enter = (~state) & (above_count >= self.enter_frames)
            leave = state & (below_count >= self.exit_frames)
            state = torch.where(enter, torch.ones_like(state), state)
            state = torch.where(leave, torch.zeros_like(state), state)
            # A transition consumes the streak that caused it.  This prevents
            # a stale counter from immediately triggering the opposite edge.
            above_count = torch.where(enter, torch.zeros_like(above_count), above_count)
            below_count = torch.where(leave, torch.zeros_like(below_count), below_count)
            result[:, index] = state
        return result

    def rollout_future(
        self,
        raw_wrench: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> torch.Tensor:
        """Scan a future window using the current latched state.

        ``raw_wrench`` has shape ``[B, To, K, D]`` and ``initial_state`` has
        shape ``[B, To]``.  The returned mask has shape ``[B, To, K]``.
        Strict ``> contact_threshold`` enters contact and strict
        ``< free_threshold`` exits contact; boundary values retain state.
        """
        magnitude = self._per_token_magnitude(raw_wrench)
        if initial_state.shape != magnitude.shape[:2]:
            raise ValueError(
                "initial_state must have shape [B, To] matching the future wrench"
            )
        if self.enter_frames == 1 and self.exit_frames == 1:
            contact_threshold = self.contact_threshold.to(magnitude)
            free_threshold = self.free_threshold.to(magnitude)
            result = torch.zeros_like(magnitude, dtype=torch.bool)
            state = initial_state.to(device=magnitude.device, dtype=torch.bool)
            for index in range(magnitude.shape[-1]):
                state = torch.where(
                    state,
                    magnitude[..., index] >= free_threshold,
                    magnitude[..., index] > contact_threshold,
                )
                result[..., index] = state
            return result

        # A future chunk is only K samples long (often fewer than
        # ``enter_frames``).  Require the configured streak when it fits, but
        # shorten it to the remaining suffix when contact starts late in the
        # chunk.  This preserves debounce for early noise while preventing a
        # late contact from being discarded merely because the horizon ends
        # first; ``fill_future_wrench_prefix`` then fills the physical prefix
        # from the first confirmed token.
        states = []
        for obs_index in range(magnitude.shape[1]):
            values = magnitude[:, obs_index]
            state = initial_state[:, obs_index].to(device=values.device, dtype=torch.bool)
            chunk = torch.zeros_like(values, dtype=torch.bool)
            above_count = torch.zeros_like(state, dtype=torch.long)
            required_count = torch.full_like(above_count, self.enter_frames)
            below_count = torch.zeros_like(state, dtype=torch.long)
            for index in range(values.shape[-1]):
                above = values[:, index] > self.contact_threshold.to(values)
                below = values[:, index] < self.free_threshold.to(values)
                above_count = torch.where(above, above_count + 1, torch.zeros_like(above_count))
                late_onset = values.shape[-1] - index < self.enter_frames
                required_count = torch.where(
                    above & (above_count == 1),
                    torch.full_like(required_count, 1 if late_onset else self.enter_frames),
                    required_count,
                )
                below_count = torch.where(below, below_count + 1, torch.zeros_like(below_count))
                enter = (~state) & (above_count >= required_count)
                leave = state & (below_count >= self.exit_frames)
                state = torch.where(enter, torch.ones_like(state), state)
                state = torch.where(leave, torch.zeros_like(state), state)
                chunk[:, index] |= state
                above_count = torch.where(enter, torch.zeros_like(above_count), above_count)
                required_count = torch.where(
                    enter, torch.full_like(required_count, self.enter_frames), required_count
                )
                below_count = torch.where(leave, torch.zeros_like(below_count), below_count)
            states.append(chunk)
        return torch.stack(states, dim=1)

    def token_contact(self, raw_wrench: torch.Tensor) -> torch.Tensor:
        """Threshold each physical-unit token independently at the entry threshold.

        This deliberately has no hysteresis, debounce or initial state. It does
        not change the historical state detection in ``forward`` or the legacy
        future state scan in ``rollout_future``.
        """
        magnitude = self._per_token_magnitude(raw_wrench)
        return magnitude > self.contact_threshold.to(magnitude)

    def _per_token_magnitude(self, raw_wrench: torch.Tensor) -> torch.Tensor:
        if raw_wrench.ndim != 4:
            raise ValueError(
                "raw_wrench must have shape [B, To, K, D], "
                f"got {tuple(raw_wrench.shape)}"
            )
        if max(self.force_dims) >= raw_wrench.shape[-1]:
            raise ValueError(
                f"force_dims={self.force_dims} exceed wrench dim {raw_wrench.shape[-1]}"
            )
        force = raw_wrench[..., list(self.force_dims)]
        if self.norm == "l1":
            return force.abs().sum(dim=-1)
        return torch.linalg.vector_norm(force, ord=2, dim=-1)


class VectorThresholdGate(nn.Module):
    """Zero complete vectors whose magnitude is below a physical threshold.

    The gate operates on the final vector dimension of a tensor shaped like
    ``[B, To, K, D]``.  Its threshold and enabled flag are buffers so the
    preprocessing contract is present in a model state dict as well as in the
    training config.
    """

    def __init__(self, threshold: float, norm: str = "l1", enabled: bool = True) -> None:
        super().__init__()
        if threshold < 0:
            raise ValueError("gate threshold must be non-negative")
        if norm not in {"l1", "l2", "sum"}:
            raise ValueError("gate norm must be one of 'l1', 'l2', or 'sum'")
        self.register_buffer("threshold", torch.tensor(float(threshold)))
        self.register_buffer("enabled", torch.tensor(bool(enabled)))
        self.norm = "l1" if norm == "sum" else norm
        self.register_buffer("norm_code", torch.tensor(1 if self.norm == "l1" else 2))

    def magnitude(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim < 1:
            raise ValueError("gated values must have at least one dimension")
        if self.norm == "l1":
            return values.abs().sum(dim=-1)
        return torch.linalg.vector_norm(values, ord=2, dim=-1)

    def keep_mask(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim < 1:
            raise ValueError("gated values must have at least one dimension")
        if not bool(self.enabled.item()):
            return torch.ones(values.shape[:-1], dtype=torch.bool, device=values.device)
        threshold = self.threshold.to(device=values.device, dtype=values.dtype)
        return self.magnitude(values) >= threshold

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim < 1:
            raise ValueError("gated values must have at least one dimension")
        keep = self.keep_mask(values)
        return torch.where(keep.unsqueeze(-1), values, torch.zeros_like(values))


class MaskProbabilityScheduler(nn.Module):
    """Map optimizer update steps to contact-image masking probability."""

    def __init__(
        self,
        schedule_type: str = "cosine",
        start_step: int = 0,
        end_step: int = 50_000,
        start_probability: float = 1.0,
        end_probability: float = 0.0,
        exponential_rate: float = 5.0,
        piecewise_steps: Optional[Sequence[int]] = None,
        piecewise_probabilities: Optional[Sequence[float]] = None,
        piecewise_interpolation: str = "linear",
        total_steps: Optional[int] = None,
        piecewise_fractions: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        supported = {"constant", "linear", "cosine", "exponential", "piecewise"}
        if schedule_type not in supported:
            raise ValueError(
                f"Unsupported schedule_type={schedule_type!r}; expected one of {sorted(supported)}"
            )
        if end_step < start_step:
            raise ValueError("end_step must be greater than or equal to start_step")
        for value in (start_probability, end_probability):
            if not 0.0 <= value <= 1.0:
                raise ValueError("mask probabilities must be in [0, 1]")
        if exponential_rate <= 0:
            raise ValueError("exponential_rate must be positive")
        if piecewise_interpolation not in {"linear", "hold"}:
            raise ValueError("piecewise_interpolation must be 'linear' or 'hold'")

        self.schedule_type = schedule_type
        self.start_step = int(start_step)
        self.end_step = int(end_step)
        self.start_probability = float(start_probability)
        self.end_probability = float(end_probability)
        self.exponential_rate = float(exponential_rate)
        self.piecewise_interpolation = piecewise_interpolation
        self.piecewise_fractions = None
        self.auto_total_steps = piecewise_fractions is not None and total_steps is None
        self.total_steps = None

        if piecewise_fractions is not None:
            if piecewise_steps:
                raise ValueError(
                    "provide either piecewise_steps or piecewise_fractions, not both"
                )
            fractions = tuple(float(value) for value in piecewise_fractions)
            if (
                len(fractions) < 1
                or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions)
                or tuple(sorted(set(fractions))) != fractions
            ):
                raise ValueError(
                    "piecewise_fractions must be sorted finite values in [0, 1]"
                )
            self.piecewise_fractions = fractions
        self.piecewise_steps = tuple(int(step) for step in (piecewise_steps or ()))
        if self.piecewise_fractions is not None and total_steps is not None:
            self.set_total_steps(total_steps)
        self.piecewise_probabilities = tuple(
            float(value) for value in (piecewise_probabilities or ())
        )
        if self.schedule_type == "piecewise":
            boundaries = self.piecewise_fractions or self.piecewise_steps
            if len(boundaries) < 1:
                raise ValueError("piecewise schedule requires at least one step")
            if len(boundaries) != len(self.piecewise_probabilities):
                raise ValueError("piecewise steps and probabilities must have equal length")
            if tuple(sorted(self.piecewise_steps)) != self.piecewise_steps:
                raise ValueError("piecewise steps must be sorted")
            if any(not 0.0 <= value <= 1.0 for value in self.piecewise_probabilities):
                raise ValueError("piecewise probabilities must be in [0, 1]")

    def set_total_steps(self, total_steps: int) -> None:
        total_steps = int(total_steps)
        if total_steps < 1:
            raise ValueError("piecewise_fractions requires a positive total_steps")
        if self.piecewise_fractions is not None:
            steps = tuple(int(round(value * total_steps)) for value in self.piecewise_fractions)
            if tuple(sorted(set(steps))) != steps:
                raise ValueError("piecewise_fractions must map to distinct optimizer steps")
            self.piecewise_steps = steps
        self.total_steps = total_steps

    def probability(self, optimizer_step: int) -> float:
        step = int(optimizer_step)
        if self.schedule_type == "constant":
            return self.start_probability
        if self.schedule_type == "piecewise":
            if self.piecewise_fractions is not None and self.total_steps is None:
                raise RuntimeError("Set the optimizer-step budget before using the mask curriculum")
            return self._piecewise_probability(step)
        if step <= self.start_step:
            return self.start_probability
        if step >= self.end_step:
            return self.end_probability

        duration = max(1, self.end_step - self.start_step)
        progress = (step - self.start_step) / duration
        if self.schedule_type == "linear":
            weight = 1.0 - progress
        elif self.schedule_type == "cosine":
            weight = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            end_value = math.exp(-self.exponential_rate)
            weight = (math.exp(-self.exponential_rate * progress) - end_value) / (
                1.0 - end_value
            )
        return self.end_probability + (
            self.start_probability - self.end_probability
        ) * weight

    def _piecewise_probability(self, step: int) -> float:
        insertion = bisect.bisect_right(self.piecewise_steps, step)
        if insertion == 0:
            return self.piecewise_probabilities[0]
        if insertion >= len(self.piecewise_steps):
            return self.piecewise_probabilities[-1]
        left = insertion - 1
        if self.piecewise_interpolation == "hold":
            return self.piecewise_probabilities[left]
        left_step = self.piecewise_steps[left]
        right_step = self.piecewise_steps[insertion]
        progress = (step - left_step) / max(1, right_step - left_step)
        left_value = self.piecewise_probabilities[left]
        right_value = self.piecewise_probabilities[insertion]
        return left_value + (right_value - left_value) * progress

    def forward(self, optimizer_step: int) -> float:
        return self.probability(optimizer_step)

class ContactAwareImageMasker(nn.Module):
    """Sample whole-image token masks for selected contact-target windows.

    The caller supplies the phase condition that should trigger masking.  For
    future-phase training this is the future-window contact flag, rather than
    the current observation's contact state.
    """

    def __init__(self, scope: str = "full_context") -> None:
        super().__init__()
        if scope not in {"current_observation", "full_context", "precontact_contact", "current_contact"}:
            raise ValueError(
                f"Unsupported mask scope={scope!r}; "
                "expected 'current_observation', 'full_context', 'precontact_contact', or 'current_contact'."
            )
        self.scope = scope

    def forward(
        self,
        contact_mask: torch.Tensor,
        probability: float,
        enabled: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if contact_mask.ndim != 2:
            raise ValueError(
                f"contact_mask must have shape [B, To], got {tuple(contact_mask.shape)}"
            )
        result = torch.zeros_like(contact_mask, dtype=torch.bool)
        probability = float(probability)
        if not enabled or probability <= 0.0:
            return result

        if self.scope in {"precontact_contact", "current_contact"}:
            if probability >= 1.0:
                return contact_mask.clone()
            return contact_mask & (torch.rand(contact_mask.shape, device=contact_mask.device,
                                              generator=generator) < probability)

        latest_contact = contact_mask[:, -1]
        if probability >= 1.0:
            selected = latest_contact
        else:
            random_values = torch.rand(
                latest_contact.shape,
                device=contact_mask.device,
                generator=generator,
            )
            selected = latest_contact & (random_values < probability)

        if self.scope == "full_context":
            return selected[:, None].expand_as(contact_mask).clone()
        result[:, -1] = selected
        return result


class FactrBlurScheduler(nn.Module):
    """Hold Gaussian blur, then linearly open vision between two step fractions.

    Fractions are relative to the planned optimizer-update budget. A null
    budget is supplied by the training workspace after building dataloaders;
    explicit budgets and phase boundaries retain historical run behavior.
    """

    def __init__(
        self,
        total_steps: Optional[int] = None,
        schedule_type: str = "linear",
        start_sigma: float = 5.0,
        end_sigma: float = 0.0,
        start_fraction: float = 0.25,
        end_fraction: float = 0.75,
    ) -> None:
        super().__init__()
        if schedule_type != "linear":
            raise ValueError("FACTR blur schedule_type must be 'linear'")
        start_sigma = float(start_sigma)
        end_sigma = float(end_sigma)
        if not math.isfinite(start_sigma) or not math.isfinite(end_sigma):
            raise ValueError("blur sigmas must be finite")
        if start_sigma < 0 or end_sigma < 0 or start_sigma < end_sigma:
            raise ValueError("blur sigmas must be non-negative with start_sigma >= end_sigma")
        start_fraction = float(start_fraction)
        end_fraction = float(end_fraction)
        if (
            not math.isfinite(start_fraction)
            or not math.isfinite(end_fraction)
            or not 0.0 <= start_fraction < end_fraction <= 1.0
        ):
            raise ValueError("blur phase fractions must satisfy 0 <= start < end <= 1")
        self.auto_total_steps = total_steps is None
        self.total_steps = None
        if total_steps is not None:
            self.set_total_steps(total_steps)
        self.schedule_type = schedule_type
        self.start_sigma = start_sigma
        self.end_sigma = end_sigma
        self.start_fraction = start_fraction
        self.end_fraction = end_fraction

    def set_total_steps(self, total_steps: int) -> None:
        total_steps = int(total_steps)
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        self.total_steps = total_steps

    def sigma(self, optimizer_step: int) -> float:
        if self.total_steps is None:
            raise RuntimeError("Set the optimizer-step budget before using the blur curriculum")
        progress = min(max(int(optimizer_step), 0), self.total_steps) / self.total_steps
        if progress <= self.start_fraction:
            return self.start_sigma
        if progress >= self.end_fraction:
            return self.end_sigma
        middle_progress = (progress - self.start_fraction) / (
            self.end_fraction - self.start_fraction
        )
        return self.start_sigma + middle_progress * (self.end_sigma - self.start_sigma)

    def forward(self, optimizer_step: int) -> float:
        return self.sigma(optimizer_step)


# Copied from FACTR's factr/utils.py (Apache-2.0):
# https://github.com/RaindragonD/FACTR/blob/5a20d69342d9d234fd588c10b7c3e62ea559c9b0/factr/utils.py
def gaussian_2d_kernel(
    kernel_size: int, sigma: float, device=None, dtype=None
) -> torch.Tensor:
    """Build FACTR's normalized square Gaussian kernel."""
    if int(kernel_size) < 1:
        raise ValueError("kernel_size must be positive")
    if not math.isfinite(float(sigma)) or float(sigma) <= 0:
        raise ValueError("sigma must be finite and positive")
    coords = torch.arange(kernel_size, device=device, dtype=dtype)
    coords = coords - (kernel_size - 1) / 2.0
    x, y = torch.meshgrid(coords, coords, indexing="xy")
    kernel = torch.exp(-0.5 * (x.square() + y.square()) / sigma**2)
    return kernel / kernel.sum()


def gaussian_2d_smoothing(img: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Apply FACTR's depthwise 2-D Gaussian smoothing to ``[..., C, H, W]``."""
    scale = float(scale)
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("blur scale must be finite and non-negative")
    if scale == 0:
        return img
    if img.ndim < 3:
        raise ValueError("img must have shape [..., C, H, W]")
    channels, height, width = (int(value) for value in img.shape[-3:])
    kernel_size = max(3, 2 * math.ceil(3 * scale) + 1)
    # Float16 convolution is not available on every supported CPU.  FACTR
    # performs the same operation in the input dtype; use float32 only for
    # those low-precision inputs and cast the result back afterward.
    work_dtype = torch.float32 if img.dtype in {torch.float16, torch.bfloat16} else img.dtype
    input_image = img.to(work_dtype)
    kernel = gaussian_2d_kernel(
        kernel_size, scale, device=img.device, dtype=work_dtype
    ).view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    original_shape = tuple(img.shape)
    flat = input_image.reshape(-1, channels, height, width)
    blurred = F.conv2d(flat, kernel, groups=channels, padding=kernel_size // 2)
    return blurred.reshape(original_shape).to(img.dtype)


def select_contact_blur_mask(
    contact_mask: torch.Tensor, scope: str = "full_context"
) -> torch.Tensor:
    """Select contact-target observations for deterministic pixel blurring."""
    if contact_mask.ndim != 2 or contact_mask.dtype != torch.bool:
        raise ValueError("contact_mask must be a bool tensor with shape [B, To]")
    if scope not in {"current_observation", "full_context"}:
        raise ValueError(
            f"Unsupported blur scope={scope!r}; expected 'current_observation' or 'full_context'."
        )
    selected = contact_mask[:, -1]
    if scope == "full_context":
        return selected[:, None].expand_as(contact_mask)
    result = torch.zeros_like(contact_mask)
    result[:, -1] = selected
    return result
