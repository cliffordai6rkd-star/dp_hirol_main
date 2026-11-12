"""Episode-local, physical-unit first-order force/torque filtering."""

import math

import numpy as np


def relative_time_seconds(timestamps, unit="s"):
    """Subtract the episode origin before converting integer clocks to seconds."""
    values = np.asarray(timestamps).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Wrench filter timestamps must be non-empty and finite")
    scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[unit]
    return (values - values[0]).astype(np.float64) * scale


def first_order_lowpass(times, values, cutoff_hz):
    """Causal one-pole: y[0]=x[0], alpha=1-exp(-2*pi*fc*dt).

    Call once per complete episode, before constructing observation windows or
    fitting normalization statistics. Each force/torque channel is independent.
    """
    if not math.isfinite(cutoff_hz) or cutoff_hz <= 0:
        raise ValueError("Wrench filter cutoff_hz (--cutoff-hz) must be positive and finite")
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if (times.ndim != 1 or not len(times) or values.ndim != 2
            or values.shape[0] != len(times) or values.shape[1] < 1):
        raise ValueError("Expected timestamps [N] and force/torque values [N, D], with N, D >= 1")
    if not np.isfinite(times).all() or not np.isfinite(values).all():
        raise ValueError("Timestamps and force/torque values must contain only finite values")
    intervals = np.diff(times)
    if np.any(intervals <= 0):
        raise ValueError("Timestamps must strictly increase within the selected episode")
    alpha = -np.expm1(-2.0 * np.pi * cutoff_hz * intervals)
    filtered = np.empty_like(values)
    filtered[0] = values[0]
    for index, weight in enumerate(alpha, start=1):
        filtered[index] = filtered[index - 1] + weight * (values[index] - filtered[index - 1])
    return filtered
