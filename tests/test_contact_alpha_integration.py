"""Minimal teacher and student integration checks for contact alpha."""

import torch
from hydra import compose, initialize_config_dir

from test_contact_alpha import make_teacher, observations
from diffusion_policy.model.vision.render_net import RenderNet
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor
from diffusion_policy.workspace.train_distill_render_net_workspace import student_context_tokens


def test_teacher_action_diffusion_loss_backpropagates_through_alpha_path():
    teacher = make_teacher().train()
    obs = observations()
    obs["future_tau_ext"][0, 1, -2:, 5] = 3.0
    loss = teacher.compute_loss({"obs": obs, "action": torch.randn(2, 9, 7)}, optimizer_step=0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert teacher.obs_encoder.cross_attention.out_proj.weight.grad.abs().sum() > 0
    assert teacher.model.output_head.weight.grad.abs().sum() > 0


def test_student_context_uses_the_same_complete_future_contact_mask():
    teacher = make_teacher().eval().requires_grad_(False)
    obs = observations()
    obs["future_tau_ext"][0, 0, -2:, 5] = 3.0
    obs["contact_alpha"] = WrenchPreprocessor.from_policy(teacher).contact_alpha(obs)
    renderer = RenderNet(
        6, base_channels=4, conditioning_dim=16, wrench_history_steps=8,
        embed_dim=16, image_self_attn_layers=1, cross_attn_layers=1,
        decoder_layers=3, patch_size=8, num_heads=4,
    ).eval()
    seen = []
    handle = renderer.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs["contact_alpha"].clone()), with_kwargs=True,
    )
    try:
        tokens, _, future_contact = student_context_tokens(
            teacher, renderer, obs, return_aux=True, return_contact=True,
        )
    finally:
        handle.remove()
    torch.testing.assert_close(seen[0], obs["contact_alpha"], rtol=0, atol=0)
    assert tokens.shape[:2] == (2, 20)
    assert future_contact.shape == (2, 2, 8)
    assert future_contact[0, 0, -2:].all()
