import pytest
import torch

from tools.evaluate_torque_response import aggregate_response, contact_donors, response_values


@pytest.mark.parametrize("scale,cosine,amplitude,nmse", [(1., 1., 1., 0.), (0., 0., 0., 1.), (-1., -1., 1., 4.)])
def test_response_metrics_have_known_perfect_zero_and_reversed_baselines(scale, cosine, amplitude, nmse):
    teacher = torch.tensor([[1., 2., 3.], [2., 4., 6.]])
    values = response_values(teacher, scale * teacher)
    result = aggregate_response(values, torch.ones(2, dtype=torch.bool))
    assert result["cosine_mean"] == pytest.approx(cosine)
    assert result["amplitude_rms_ratio"] == pytest.approx(amplitude)
    assert result["relative_difference_mse"] == pytest.approx(nmse)


def test_unresponsive_teacher_is_not_reported_as_a_valid_alignment_score():
    result = aggregate_response(response_values(torch.zeros(2, 4), torch.zeros(2, 4)),
                                torch.ones(2, dtype=torch.bool))
    assert result["responsive_observations"] == 0
    assert result["cosine_mean"] is None
    assert result["relative_difference_mse"] is None


def test_donors_are_contact_histories_from_another_episode():
    phase = torch.tensor([[0, 1], [1, 2], [2, 2], [0, 2]])
    episodes = torch.tensor([3, 3, 26, 30])
    donor = contact_donors(phase, episodes).flatten()
    episode = episodes[:, None].expand_as(phase).flatten()
    assert (phase.flatten()[donor] == 2).all()
    assert (episode[donor] != episode).all()
