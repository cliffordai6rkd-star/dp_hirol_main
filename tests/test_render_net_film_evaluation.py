from types import SimpleNamespace

import pytest
import torch
from diffusers.utils.torch_utils import randn_tensor

from diffusion_policy.scripts import evaluate_render_net_film as evaluation


class _ScaleNormalizer:
    def normalize(self, value):
        return value * 2


def test_action_metrics_use_executed_chunk_and_sign_invariant_rotation():
    policy = SimpleNamespace(action_start_index=1, n_action_steps=2,
                             normalizer={"action": _ScaleNormalizer()})
    target = torch.zeros(1, 4, 8)
    target[..., 6] = 1
    prediction = target.clone()
    prediction[:, (0, 3)] = 999  # Past/unexecuted actions must be excluded.
    prediction[:, 1:3, :2] = torch.tensor([0.003, 0.004])
    prediction[:, 1:3, 6] = -1  # Same physical rotation with the opposite quaternion sign.
    prediction[:, 1:3, 7] = 0.2
    metrics = evaluation.action_metrics(policy, prediction, target)
    assert metrics["position_mse_m2"].item() == pytest.approx(0.005**2)
    assert metrics["rotation_mean_deg"].item() == pytest.approx(0)
    assert metrics["gripper_mse"].item() == pytest.approx(0.04)
    assert metrics["action_mse"].item() == pytest.approx((0.005**2 + 4 + 0.04) / 8)
    assert metrics["normalized_action_mse"].item() == pytest.approx(4 * metrics["action_mse"].item())


def test_paired_diffusion_noise_is_equal_across_branches_and_batch_sizes():
    device = torch.device("cpu")
    indices = [7, 21]
    first = randn_tensor((2, 9, 8), generator=evaluation.noise_generators(device, 42, indices))
    paired = randn_tensor((2, 9, 8), generator=evaluation.noise_generators(device, 42, indices))
    single = torch.cat([randn_tensor((1, 9, 8), generator=evaluation.noise_generators(device, 42, [index]))
                        for index in indices])
    other_seed = randn_tensor((2, 9, 8), generator=evaluation.noise_generators(device, 43, indices))
    torch.testing.assert_close(first, paired, rtol=0, atol=0)
    torch.testing.assert_close(first, single, rtol=0, atol=0)
    assert not torch.equal(first, other_seed)


def test_summary_pools_squared_position_error_and_counts_distinct_windows():
    records = []
    for seed in (42, 43):
        for index, position_error in enumerate((0.003, 0.004)):
            records.append({"sample_index": index, "seed": seed,
                            "film_on_token_mse": 0.2, "film_off_token_mse": 0.22,
                            "film_on_position_mse_m2": position_error**2,
                            "film_off_position_mse_m2": 4 * position_error**2})
    summary = evaluation.summarize(records)
    assert summary["num_windows"] == 2 and summary["num_action_predictions"] == 4
    assert summary["relative_change_percent"]["token_mse"] == pytest.approx(10)
    assert summary["film_on"]["position_rmse_mm"] == pytest.approx(1000 * ((0.003**2 + 0.004**2) / 2)**0.5)
    assert summary["film_off"]["position_rmse_mm"] == pytest.approx(2 * summary["film_on"]["position_rmse_mm"])
