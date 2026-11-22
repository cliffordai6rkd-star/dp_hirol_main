import torch
import torch.nn.functional as F

from diffusion_policy.model.vision.image_tau_phase_gate import ImageTauPhaseGate


def test_image_tau_gate_fuses_current_images_and_full_wrench_history_for_three_class_ce():
    torch.set_num_threads(1)
    model = ImageTauPhaseGate(("side", "wrist"), wrench_dim=7, hidden_dim=16)
    images = {key: torch.rand(2, 2, 3, 240, 320) for key in ("side", "wrist")}
    tau_ext = torch.randn(2, 2, 50, 7)
    labels = torch.tensor([[0, 1], [2, 2]])

    assert sum(parameter.numel() for parameter in model.parameters()) < 10_000
    logits = model(images, tau_ext)
    assert logits.shape == (2, 2, 3)
    loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    loss.backward()
    assert model.image_cnn[0].weight.grad is not None
    assert model.tau_gru.weight_ih_l0.grad is not None
    assert model.head[-1].weight.grad is not None


def test_image_tau_gate_rejects_missing_modalities():
    model = ImageTauPhaseGate(("side", "wrist"), wrench_dim=7, hidden_dim=8)
    images = {"side": torch.rand(1, 1, 3, 32, 32)}
    tau = torch.rand(1, 1, 50, 7)
    try:
        model(images, tau)
    except ValueError as exc:
        assert "Image keys" in str(exc)
    else:
        raise AssertionError("Missing camera input must be rejected")


def test_temporal_gate_uses_past_images_and_preserves_three_class_gradients():
    model = ImageTauPhaseGate(hidden_dim=16, pool_size=2, image_offsets=(-8, -4, 0),
                              cnn_norm=True, center_images=True)
    images = {key: torch.rand(2, 2, 3, 3, 64, 64, requires_grad=True)
              for key in model.image_keys}
    logits = model(images, torch.randn(2, 2, 50, 7))
    F.cross_entropy(logits.flatten(0, 1), torch.tensor([0, 1, 2, 1])).backward()
    assert logits.shape == (2, 2, 3)
    assert images['side'].grad[:, :, 0].abs().sum() > 0
    assert model.image_temporal.weight_hh_l0.grad.abs().sum() > 0


def test_cached_image_histories_stay_inside_episode_and_never_read_future():
    from tools.search_image_tau_phase_gate import inputs

    frame_numbers = torch.arange(12).reshape(12, 1, 1, 1).expand(-1, 3, 2, 2)
    cache = {'side_64': frame_numbers, 'episode_start': torch.tensor([0] * 6 + [6] * 6),
             'tau': torch.zeros(12, 50, 7), 'labels': torch.zeros(12, dtype=torch.long)}
    images, _, _ = inputs(cache, torch.tensor([[1, 6], [7, 11]]),
                           {'image_keys': ['side'], 'image_size': 64, 'image_offsets': [-8, -4, 0]})
    expected = torch.tensor([[[0, 0, 1], [6, 6, 6]], [[6, 6, 7], [6, 7, 11]]])
    torch.testing.assert_close(images['side'][..., 0, 0, 0], expected)
