"""Loss, coverage and continuation contracts; real CUDA preflight covers model paths."""
import numpy as np
import torch

from paired_response_full_common import (spread_order, active_response_weights, weighted_difference,
                                         explicit_zero_alpha, extension_allowed)


def test_all_rows_and_episodes_are_covered_without_adjacent_window_concentration():
    episodes = np.repeat(np.arange(7), 60)
    phases = np.tile(np.repeat(np.arange(3), 20), 7)
    order = spread_order(phases, episodes, 42, 0)
    assert sorted(order.tolist()) == list(range(len(phases)))
    assert np.array_equal(order, spread_order(phases, episodes, 42, 0))
    assert not np.array_equal(order, spread_order(phases, episodes, 42, 1))
    for phase in range(3):
        # One row per episode before another row from the same episode is drawn.
        selected = order[phases[order] == phase]
        assert len(set(episodes[selected[:7]])) == 7


def test_pair_loss_keeps_all_dimensions_and_normalizes_by_effective_weight():
    teacher = torch.tensor([[[0., 0.], [1., 2.], [2., 1.]]], requires_grad=True)
    student = torch.zeros_like(teacher, requires_grad=True)
    weights = active_response_weights(teacher, .1)
    loss = weighted_difference(student, teacher, weights)
    expected = ((student-teacher.detach()).square().mean(-1)*weights).sum()/weights.sum()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert teacher.grad is None and not weights.requires_grad
    assert torch.count_nonzero(student.grad[0, 0]) == 0
    assert torch.count_nonzero(student.grad[0, 1:]) == 4
    quiet = active_response_weights(torch.zeros_like(teacher), .1)
    assert weighted_difference(student, teacher, quiet) == 0


def test_microbatch_preserves_the_whole_batch_gradient():
    torch.manual_seed(1)
    teacher = torch.randn(7, 5, 4)
    student = torch.randn_like(teacher, requires_grad=True)
    weights = active_response_weights(teacher, .3)
    full = weighted_difference(student, teacher, weights)
    parts = sum(weighted_difference(student[i:i+2], teacher[i:i+2], weights[i:i+2], weights.sum())
                for i in range(0, 7, 2))
    assert torch.allclose(full, parts)
    assert torch.allclose(torch.autograd.grad(full, student)[0], torch.autograd.grad(parts, student)[0])


def test_zero_alpha_override_is_only_an_identity_diagnostic():
    class Condition:
        def forward(self):
            return {}, {'contact_alpha': torch.ones(2, 3)}
    class Model:
        condition_network = Condition()
    model = Model()
    with explicit_zero_alpha(model):
        assert torch.count_nonzero(model.condition_network.forward()[1]['contact_alpha']) == 0
    assert torch.all(model.condition_network.forward()[1]['contact_alpha'] == 1)


def test_extension_requires_contact_and_free_validation_to_hold():
    initial = dict(contact=dict(token_mse=.2, response_difference_mse=.1), free=dict(zero_token_mse=.1))
    assert extension_allowed(initial, initial)
    worse = dict(contact=dict(token_mse=.19, response_difference_mse=.11), free=dict(zero_token_mse=.09))
    assert not extension_allowed(initial, worse)
