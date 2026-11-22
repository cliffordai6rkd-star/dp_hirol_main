import torch

from diffusion_policy.model.vision.render_net import RenderNet
from diffusion_policy.model.vision.target_conditioned_renderer import TargetConditionedRenderer


def build_case():
    torch.manual_seed(17)
    reference = RenderNet(
        7, base_channels=4, conditioning_dim=16, embed_dim=16, num_heads=4,
        wrench_history_steps=3, patch_size=8, image_self_attn_layers=1,
        cross_attn_layers=1, decoder_layers=3, image_keys=("side", "wrist"),
        rendering_mode="residual", residual_scale=1.0,
    )
    with torch.no_grad():
        reference.output.weight.normal_(std=0.03)
    model = TargetConditionedRenderer(reference, 16, {"side": 4, "wrist": 4})
    images = {key: torch.rand(2, 2, 3, 16, 16) * 0.6 + 0.2 for key in model.image_keys}
    gates = {key: torch.ones(2, 2, 1, 16, 16) for key in images}
    z0 = torch.randn(2, 16, 16)
    return reference, model, images, gates, z0


def test_free_and_outside_region_remain_identical_to_input():
    _, model, images, gates, z0 = build_case()
    for gate in gates.values():
        gate[..., :8] = 0
    phase = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    output = model(images, z0, z0 + torch.randn_like(z0), gates, phase)
    for key in images:
        torch.testing.assert_close(output[key][phase == 0], images[key][phase == 0], rtol=0, atol=0)
        torch.testing.assert_close(output[key][..., :8], images[key][..., :8], rtol=0, atol=0)
        assert output[key].min() >= 0 and output[key].max() <= 1


def test_target_condition_changes_rgb_and_receives_gradient_without_updating_reference():
    reference, model, images, gates, z0 = build_case()
    before = {key: value.clone() for key, value in reference.state_dict().items()}
    target = (z0 + torch.randn_like(z0)).requires_grad_()
    phase = torch.ones(2, 2)
    output = model(images, z0, target, gates, phase)
    zero = model(images, z0, z0, gates, phase)
    assert sum((output[key] - zero[key]).abs().sum() for key in images) > 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sum(value.square().mean() for value in output.values()).backward()
    assert target.grad.abs().sum() > 0
    assert model.context_projection[-1].weight.grad.abs().sum() > 0
    optimizer.step()
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
