import torch

from diffusion_policy.model.vision.force_aware_obs_encoder import ForceAwareObsEncoder
from test_render_net_distillation import PatchBackbone
from tools.quick_validate_student_ffn import PhaseGatedFFN


def encoder():
    return ForceAwareObsEncoder(
        "unused", backbone=PatchBackbone(), n_emb=16, n_head=4, max_obs_steps=2,
        wrench_history_steps=3, cross_attn_query="image", cross_attn_layers=2,
        cross_attn_query_residual=True, dropout=0.1,
        image_shapes={"side": (3, 16, 16), "wrist": (3, 16, 16)},
    ).eval().requires_grad_(False)


def test_frozen_student_ffn_matches_teacher_replay_and_bypasses_closed_phase():
    torch.manual_seed(21)
    teacher = encoder()
    head = PhaseGatedFFN(teacher).train()
    assert not head.training and not any(p.requires_grad for p in head.parameters())
    images = {key: torch.rand(2, 2, 3, 16, 16) for key in teacher.image_keys}
    opened = torch.ones(2, 2)
    actual = teacher.encode_image_context(images, context_adapter=lambda x: head(x, opened))
    expected = teacher.encode_image_context(images, apply_fusion_ffn=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    closed = torch.zeros(2, 2)
    actual = teacher.encode_image_context(images, context_adapter=lambda x: head(x, closed))
    torch.testing.assert_close(actual, teacher.encode_image_context(images), rtol=0, atol=0)


def test_student_ffn_gets_image_and_parameter_gradients_without_changing_teacher():
    torch.manual_seed(22)
    teacher = encoder()
    before = {key: value.clone() for key, value in teacher.state_dict().items()}
    head = PhaseGatedFFN(teacher, trainable=True).train()
    images = {key: torch.rand(1, 2, 3, 16, 16, requires_grad=True) for key in teacher.image_keys}
    gate = torch.tensor([[0.0, 1.0]])
    z = teacher.encode_image_context(images, input_grad=True, context_adapter=lambda x: head(x, gate))
    torch.optim.Adam(head.parameters(), lr=1e-3).zero_grad()
    (z - torch.randn_like(z)).square().mean().backward()
    assert all(image.grad.abs().sum() > 0 for image in images.values())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    with torch.no_grad():
        for p in head.parameters():
            if p.grad is not None:
                p.add_(p.grad, alpha=-1e-3)
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(p.grad is None for p in teacher.parameters())
