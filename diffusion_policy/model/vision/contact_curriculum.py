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

        # Each future observation has its own initial state, then scans its K
        # future wrench samples in order.
        states = []
        for obs_index in range(magnitude.shape[1]):
            states.append(self._scan_tokens(
                magnitude[:, obs_index], initial_state=initial_state[:, obs_index]
            ))
        return torch.stack(states, dim=1)

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

        if piecewise_fractions is not None:
            if piecewise_steps:
                raise ValueError(
                    "provide either piecewise_steps or piecewise_fractions, not both"
                )
            if total_steps is None or int(total_steps) < 1:
                raise ValueError(
                    "piecewise_fractions requires a positive total_steps"
                )
            fractions = tuple(float(value) for value in piecewise_fractions)
            if (
                len(fractions) < 1
                or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions)
                or tuple(sorted(fractions)) != fractions
            ):
                raise ValueError(
                    "piecewise_fractions must be sorted finite values in [0, 1]"
                )
            piecewise_steps = tuple(
                int(round(value * int(total_steps))) for value in fractions
            )
            if tuple(sorted(set(piecewise_steps))) != piecewise_steps:
                raise ValueError(
                    "piecewise_fractions must map to distinct optimizer steps"
                )
        self.piecewise_steps = tuple(int(step) for step in (piecewise_steps or ()))
        self.piecewise_probabilities = tuple(
            float(value) for value in (piecewise_probabilities or ())
        )
        if self.schedule_type == "piecewise":
            if len(self.piecewise_steps) < 1:
                raise ValueError("piecewise schedule requires at least one step")
            if len(self.piecewise_steps) != len(self.piecewise_probabilities):
                raise ValueError("piecewise steps and probabilities must have equal length")
            if tuple(sorted(self.piecewise_steps)) != self.piecewise_steps:
                raise ValueError("piecewise steps must be sorted")
            if any(not 0.0 <= value <= 1.0 for value in self.piecewise_probabilities):
                raise ValueError("piecewise probabilities must be in [0, 1]")

    def probability(self, optimizer_step: int) -> float:
        step = int(optimizer_step)
        if self.schedule_type == "constant":
            return self.start_probability
        if self.schedule_type == "piecewise":
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


class BlurSigmaScheduler(MaskProbabilityScheduler):
    """Schedule Gaussian standard deviation in input-image pixels."""

    def __init__(
        self,
        start_sigma: float = 5.0,
        end_sigma: float = 0.0,
        piecewise_sigmas: Optional[Sequence[float]] = None,
        **kwargs,
    ) -> None:
        sigmas = [float(start_sigma), float(end_sigma)]
        points = [float(value) for value in (piecewise_sigmas or ())]
        if any(not math.isfinite(value) or value < 0 for value in (*sigmas, *points)):
            raise ValueError("blur sigmas must be finite and non-negative")
        self.sigma_scale = max(*sigmas, *points, 1.0)
        super().__init__(
            start_probability=sigmas[0] / self.sigma_scale,
            end_probability=sigmas[1] / self.sigma_scale,
            piecewise_probabilities=[value / self.sigma_scale for value in points],
            **kwargs,
        )

    def sigma(self, optimizer_step: int) -> float:
        return self.probability(optimizer_step) * self.sigma_scale

    def forward(self, optimizer_step: int) -> float:
        return self.sigma(optimizer_step)


def gaussian_blur_images(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on [..., C, H, W], with an exact identity at zero.

    Kernels span three standard deviations, limited by reflect-padding bounds.
    Filtering runs in float32 even inside the policy's AMP context.
    """
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("blur sigma must be finite and non-negative")
    if sigma == 0:
        return images
    channels, height, width = images.shape[-3:]
    with torch.autocast(device_type=images.device.type, enabled=False):
        result = images.reshape(-1, channels, height, width).float()
        for axis, size in ((-1, width), (-2, height)):
            radius = min(math.ceil(3 * sigma), size - 1)
            if radius == 0:
                continue
            positions = torch.arange(-radius, radius + 1, device=images.device, dtype=torch.float32)
            kernel = torch.exp(-0.5 * (positions / sigma).square())
            kernel = kernel / kernel.sum()
            if axis == -1:
                padding, shape = (radius, radius, 0, 0), (1, 1, 1, -1)
            else:
                padding, shape = (0, 0, radius, radius), (1, 1, -1, 1)
            weight = kernel.reshape(shape).expand(channels, -1, -1, -1)
            result = F.conv2d(F.pad(result, padding, mode="reflect"), weight, groups=channels)
    return result.reshape_as(images).to(images.dtype)


class ContactAwareImageMasker(nn.Module):
    """Sample whole-image token masks for selected contact-target windows.

    The caller supplies the phase condition that should trigger masking.  For
    future-phase training this is the future-window contact flag, rather than
    the current observation's contact state.
    """

    def __init__(self, scope: str = "full_context") -> None:
        super().__init__()
        if scope not in {"current_observation", "full_context"}:
            raise ValueError(
                f"Unsupported mask scope={scope!r}; "
                "expected 'current_observation' or 'full_context'."
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
