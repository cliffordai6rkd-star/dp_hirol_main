"""Contracts for matched initialization, sampling and validation-only duration decisions."""
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn

from new_teacher_response_common import SharedCNNLSTM, balanced_batches, continuation_decision


def test_sampling_has_identical_batches_and_uniform_episode_visits_in_each_phase():
    phases = np.tile(np.repeat([0, 1, 2], 9), 5)
    episodes = np.repeat(np.arange(5), 27)
    plan = balanced_batches(phases, episodes, 20, 8, 42)
    assert np.array_equal(plan, balanced_batches(phases, episodes, 20, 8, 42))
    assert np.all(phases[plan[:, :4]] == 2)
    assert np.all(phases[plan[:, 4:]] == 0)
    for selected in [plan[:, :4].flatten(), plan[:, 4:].flatten()]:
        assert np.bincount(episodes[selected]).tolist() == [16] * 5
        # First full window cycle visits all9 windows of each episode once.
        assert len(np.unique(selected[:45])) == 45


def test_shared_lstm_starts_at_identity_and_all_paths_receive_gradients_after_head_opens():
    torch.manual_seed(42)
    reference = SimpleNamespace(wrench_projection=nn.Linear(7, 256),
        wrench_encoder=nn.LSTM(256, 256, num_layers=2, batch_first=True),
        wrench_position=nn.Parameter(torch.randn(1, 1, 8, 256)),
        obs_position=nn.Parameter(torch.randn(1, 2, 1, 256)),
        force_modality=nn.Parameter(torch.randn(1, 1, 1, 256)), image_keys=('side', 'wrist'))
    generator = SharedCNNLSTM(reference)
    for key, value in reference.wrench_encoder.state_dict().items():
        assert torch.equal(generator.wrench_encoder.state_dict()[key], value)
    images = {key: torch.rand(2, 2, 3, 32, 32) for key in reference.image_keys}
    wrench = torch.randn(2, 2, 8, 7)
    assert all(torch.count_nonzero(v) == 0 for v in generator(images, wrench).values())
    nn.init.normal_(generator.decoder[-1].weight, std=.01)
    sum(v.square().mean() for v in generator(images, wrench).values()).backward()
    assert generator.encoder[0].weight.grad.norm() > 0
    assert generator.wrench_encoder.weight_ih_l0.grad.norm() > 0
    assert generator.wrench_projection.weight.grad.norm() > 0


def test_extension_uses_internal_improvement_and_preservation():
    config = SimpleNamespace(extension_min_relative_improvement=.005, preservation_relative_tolerance=.01)
    def row(step, response, free=.1):
        return dict(step=step, selection=dict(contact=dict(response_difference_mse=response, token_mse=.15),
                                             free=dict(token_mse=free)), external={'arbitrary': step})
    improving = [row(0, .05), row(600, .04), row(800, .039), row(1000, .038)]
    assert continuation_decision(improving, config)['extend']
    assert not continuation_decision(improving[:-1]+[row(1000, .04)], config)['extend']
    assert not continuation_decision(improving[:-1]+[row(1000, .038, free=.2)], config)['extend']
    # The user's continuation rule accepts sustained small improvements;
    # magnitude thresholds are explicit policy choices, not implicit limits.
    small = [row(0, .05), row(600, .04), row(800, .03999), row(1000, .03998)]
    config.extension_min_relative_improvement = 0.0
    assert continuation_decision(small, config)['extend']
