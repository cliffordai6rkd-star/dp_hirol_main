import pytest
import torch

from train.nomalizer import Normalizer


@pytest.mark.parametrize(
    "method",
    ["gaussian_normalize", "limit_normalize", "quantile_normalize"],
)
def test_normalize_moves_statistics_to_input_device_and_dtype(method):
    normalizer = Normalizer(
        {
            "value": {
                "mean": torch.tensor([1.0, 2.0]),
                "std": torch.tensor([2.0, 4.0]),
                "min": torch.tensor([0.0, 0.0]),
                "max": torch.tensor([2.0, 4.0]),
                "q01": torch.tensor([0.0, 0.0]),
                "q99": torch.tensor([2.0, 4.0]),
            }
        }
    )
    value = torch.ones(2, device="meta", dtype=torch.float64)

    normalized = getattr(normalizer, method)("value", value)

    assert normalized.device == value.device
    assert normalized.dtype == value.dtype
