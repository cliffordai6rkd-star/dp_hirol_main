import numpy as np

from diffusion_policy.common.phase_sampler import PhaseWeightedDistributedSampler


def test_phase_sampler_matches_configured_three_phase_ratio():
    labels = np.array([0] * 400 + [1] * 800 + [2] * 1200, dtype=np.int8)
    sampler = PhaseWeightedDistributedSampler(
        list(range(len(labels))),
        labels,
        phase_weights={0: 1.0, 1: 5.0, 2: 2.0},
        seed=7,
    )
    sampled = np.asarray(list(iter(sampler)), dtype=np.int64)
    sampled_labels = labels[sampled]
    counts = np.bincount(sampled_labels, minlength=3)
    fractions = counts / counts.sum()
    np.testing.assert_allclose(fractions, [1 / 8, 5 / 8, 2 / 8], atol=0.02)


def test_phase_sampler_is_deterministic_per_epoch_and_ddp_length_is_even():
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int8)
    sampler = PhaseWeightedDistributedSampler(
        list(range(len(labels))),
        labels,
        phase_weights={0: 1.0, 1: 5.0, 2: 2.0},
        num_replicas=2,
        rank=1,
        seed=11,
    )
    first = list(iter(sampler))
    second = list(iter(sampler))
    assert first == second
    assert len(first) == len(sampler)
    sampler.set_epoch(1)
    assert first != list(iter(sampler))
