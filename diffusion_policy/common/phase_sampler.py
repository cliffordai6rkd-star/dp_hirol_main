"""Distributed phase-aware resampling for contact-rich behavior cloning."""

from __future__ import annotations

import math
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


class PhaseWeightedDistributedSampler(Sampler[int]):
    """Sample free/pre-contact/contact examples with fixed phase ratios.

    The phase weights are normalized by the number of examples in each pool,
    so the configured phase ratio is the expected sampling ratio rather than a
    ratio silently distorted by the dataset's natural class imbalance.
    Sampling is with replacement, matching FIRST-style re-sampling while
    keeping the epoch length unchanged.
    """

    def __init__(
        self,
        dataset,
        phase_labels: Sequence[int],
        phase_weights: dict[int, float],
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ) -> None:
        if num_replicas < 1:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        labels = np.asarray(phase_labels, dtype=np.int8).reshape(-1)
        if np.any(~np.isin(labels, (0, 1, 2))):
            raise ValueError(
                "phase_labels must contain only 0=free, 1=pre-contact, 2=contact"
            )
        if len(dataset) != labels.shape[0]:
            raise ValueError(
                "phase_labels must have one entry per dataset sample; "
                f"got {labels.shape[0]} labels for dataset length {len(dataset)}"
            )
        normalized_weights = {
            int(phase): float(weight) for phase, weight in phase_weights.items()
        }
        if set(normalized_weights) != {0, 1, 2}:
            raise ValueError("phase_weights must define phases 0, 1, and 2")
        if any(weight <= 0 for weight in normalized_weights.values()):
            raise ValueError("phase sampling weights must be positive")
        phase_counts = {
            phase: int(np.count_nonzero(labels == phase)) for phase in (0, 1, 2)
        }
        missing = [phase for phase, count in phase_counts.items() if count == 0]
        if missing:
            raise ValueError(
                "phase resampling requires all phase pools to be non-empty; "
                f"counts={phase_counts}, missing={missing}"
            )

        self.dataset = dataset
        self.phase_labels = labels
        self.phase_counts = phase_counts
        self.phase_weights = normalized_weights
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0

        # Equalize the total mass of each pool before applying the requested
        # phase ratio.  This makes the expected three-phase ratio exact.
        weights = np.zeros(labels.shape[0], dtype=np.float64)
        for phase, count in phase_counts.items():
            weights[labels == phase] = normalized_weights[phase] / count
        self.weights = torch.as_tensor(weights, dtype=torch.double)

        self.num_samples = int(math.ceil(len(dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    @property
    def expected_phase_fractions(self) -> dict[int, float]:
        total = sum(self.phase_weights.values())
        return {phase: weight / total for phase, weight in self.phase_weights.items()}

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        sampled = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        ).tolist()
        indices = sampled[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError(
                f"phase sampler produced {len(indices)} samples, "
                f"expected {self.num_samples}"
            )
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
