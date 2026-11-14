"""Check the restored teacher context_norm -> student image-token contract."""

from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

from diffusion_policy.model.vision.render_net import RenderNet
from diffusion_policy.workspace.train_distill_render_net_workspace import TrainDistillRenderNetWorkspace
from test_render_net_distillation import make_attention_teacher


@pytest.mark.parametrize("cached", [False, True])
def test_final_teacher_context_norm_supervises_image_tokens_without_student_fusion(cached):
    teacher, obs, _ = make_attention_teacher()
    renderer = RenderNet(
        6, base_channels=4, conditioning_dim=16, wrench_history_steps=7,
        embed_dim=16, image_self_attn_layers=1, cross_attn_layers=1,
        patch_size=8, decoder_layers=3, num_heads=4, separate_gate_path=True,
        spatial_modulation={"enabled": True},
        high_attention_render={"enabled": True, "threshold": .1},
    )
    with torch.no_grad():
        renderer.mask_head.bias.zero_()  # Open the region gate for RGB gradients.
    renderer.set_training_stage("rgb")
    workspace = object.__new__(TrainDistillRenderNetWorkspace)
    workspace.cfg = OmegaConf.load(Path(__file__).resolve().parents[1] /
                                  "diffusion_policy/config/train_distill_render_net_student.yaml")
    workspace.teacher_policy = teacher
    workspace.model = renderer
    workspace.device = torch.device("cpu")
    workspace.amp_enabled = False
    workspace.attention_enabled = False
    workspace.shape_info = {}
    workspace.render_stages_enabled = True
    workspace.training_stage = "rgb"
    events = {"norm": [], "ffn": 0, "force": 0}

    def count(name):
        def hook(*args):
            events[name] += 1
        return hook

    handles = [teacher.obs_encoder.context_norm.register_forward_hook(
        lambda module, args, output: events["norm"].append(output)
    )]
    blocks = (teacher.obs_encoder, *teacher.obs_encoder.extra_cross_attention_layers)
    handles += [block.cross_ffn.register_forward_hook(count("ffn")) for block in blocks]
    handles.append(teacher.obs_encoder.force_encoder.register_forward_hook(count("force")))
    try:
        with torch.no_grad():
            target = teacher._encode_context(obs, apply_curriculum_mask=False).detach()
        torch.testing.assert_close(target, events["norm"][-1], rtol=0, atol=0)
        events.update(norm=[], ffn=0, force=0)
        cached_targets = {"tokens": target} if cached else None
        loss = workspace._loss(obs, cached_targets=cached_targets)

        # The final norm runs on both paths; force/fusion runs only to form
        # an uncached teacher target. Student image tokens skip those modules.
        assert events["ffn"] == (0 if cached else len(blocks))
        assert events["force"] == (0 if cached else 1)
        assert len(events["norm"]) == (1 if cached else 2)
        student = events["norm"][-1]
        assert not target.requires_grad and student.requires_grad
        torch.testing.assert_close(loss, (student.float() - target.float()).square().mean())
        loss.backward()
        assert renderer.output.weight.grad.abs().sum() > 0
        assert all(parameter.grad is None for parameter in teacher.parameters())
    finally:
        for handle in handles:
            handle.remove()
