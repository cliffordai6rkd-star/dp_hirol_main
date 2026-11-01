import pytest
import torch

from diffusion_policy.model.vision.render_net_attention import percentile_region


@pytest.mark.parametrize("percentiles", [(2, 98), (10, 90), (82.5, 97.3), (0, 100)])
def test_teacher_percentile_region_accepts_arbitrary_bounds_per_observation(percentiles):
    values = torch.stack([torch.arange(101).float(), torch.arange(101).float() * 0.01 + 1])
    low, high = percentiles
    expected = ((torch.arange(101).float() - low) / (high - low)).clamp(0, 1)
    actual = percentile_region(values, percentiles)
    torch.testing.assert_close(actual, expected.expand_as(actual), atol=2e-6, rtol=2e-5)
    assert not actual.requires_grad


@pytest.mark.parametrize("percentiles", [(-1, 99), (10, 101), (90, 10), (50, 50), (float("nan"), 99)])
def test_teacher_percentile_region_rejects_invalid_bounds(percentiles):
    with pytest.raises(ValueError):
        percentile_region(torch.ones(1, 100), percentiles)


def test_teacher_region_handles_flat_and_sparse_attention():
    torch.testing.assert_close(percentile_region(torch.ones(2, 100), [10, 90]), torch.ones(2, 100))
    sparse = torch.zeros(1, 100)
    sparse[0, -1] = 1
    torch.testing.assert_close(percentile_region(sparse, [10, 90]), sparse)
