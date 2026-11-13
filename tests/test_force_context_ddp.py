"""Paired force-context loss synchronizes two real CPU RenderNet replicas."""

from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from diffusion_policy.model.vision.force_context_distillation import teacher_context_pair
from diffusion_policy.model.vision.render_net import RenderNet
from diffusion_policy.workspace.train_distill_render_net_workspace import (
    TrainDistillRenderNetWorkspace,
)
from diffusion_policy.workspace.train_force_aware_diffusion_workspace import DistributedContext
from test_civi_action_heads import _make_policy


def _assert_replicas_match(module):
    values = torch.cat([parameter.detach().flatten() for parameter in module.parameters()])
    replicas = [torch.empty_like(values) for _ in range(dist.get_world_size())]
    dist.all_gather(replicas, values)
    for replica in replicas[1:]:
        torch.testing.assert_close(replica, replicas[0], rtol=0, atol=0)


def _forbidden(*args, **kwargs):
    raise AssertionError("Cached force-context loss must not run teacher fusion or ACT")


def _exercise_layout(rank, upsample):
    teacher = _make_policy("act", fusion_downsample_factor=2).eval().requires_grad_(False)
    teacher.obs_encoder.fusion_output_mode = "visual"
    teacher.obs_encoder.fusion_upsample = upsample
    teacher.obs_encoder.fusion_highres_skip = upsample
    torch.manual_seed(31)
    renderer = RenderNet(
        wrench_dim=7, conditioning_dim=16, base_channels=4,
        wrench_history_steps=8, max_obs_steps=2, wrench_layers=2,
        image_keys=("side", "wrist"), patch_size=4, embed_dim=16,
        num_heads=4, image_self_attn_layers=1, cross_attn_layers=1,
        cross_camera_attn_layers=1, decoder_layers=2, mlp_ratio=2.0,
        use_contact_alpha=False, use_region_gate=False,
        high_attention_render={"enabled": False},
        phase_prediction={"enabled": False},
    )
    renderer.initialize_wrench_encoder(teacher.obs_encoder)
    renderer.freeze_wrench_encoder()
    renderer.set_training_stage("joint")
    renderer.set_stage_mode(True)
    assert not renderer.wrench_encoder.training

    workspace = object.__new__(TrainDistillRenderNetWorkspace)
    workspace.cfg = OmegaConf.create({
        "distillation": {
            "objective": "token_mse",
            "force_context": {"enabled": True, "difference_weight": 1.0, "region_weight": 1.0},
        },
        "training": {"distributed": {"find_unused_parameters": False}},
    })
    workspace.distributed = DistributedContext(rank=rank, local_rank=rank, world_size=2)
    workspace.device = torch.device("cpu")
    workspace.teacher_policy = teacher
    workspace.model = renderer
    workspace.training_stage = "joint"
    workspace.amp_enabled = False
    workspace.shape_info = None
    workspace.force_difference_scale = 1.0
    workspace.force_region_scale = 1.0
    workspace._setup_distributed_model()
    optimizer = torch.optim.AdamW(renderer.parameters(), lr=1e-3)

    generator = torch.Generator().manual_seed(101 + rank)
    obs = {
        key: torch.rand(2, 2, 3, 16, 16, generator=generator)
        for key in teacher.obs_encoder.image_keys
    }
    obs["tau_ext"] = torch.randn(2, 2, 8, 7, generator=generator) + rank
    fingerprints = torch.stack([obs["side"].sum(), obs["wrist"].sum(), obs["tau_ext"].sum()])
    replica_inputs = [torch.empty_like(fingerprints) for _ in range(2)]
    dist.all_gather(replica_inputs, fingerprints)
    assert torch.all(replica_inputs[0] != replica_inputs[1])
    targets = teacher_context_pair(teacher, obs)
    expected_tokens = 16 if upsample else 4
    assert targets["tokens"].shape == (2, expected_tokens, 16)
    assert (targets["tokens"] - targets["zero_tokens"]).abs().sum() > 0
    teacher._encode_context = _forbidden
    teacher.model.forward = _forbidden
    frozen_teacher = {name: value.clone() for name, value in teacher.state_dict().items()}
    frozen_renderer = {name: parameter.detach().clone()
                       for name, parameter in renderer.named_parameters() if not parameter.requires_grad}
    initial_trainable = {name: parameter.detach().clone()
                         for name, parameter in renderer.named_parameters() if parameter.requires_grad}
    assert frozen_renderer and initial_trainable
    rendered_calls = []

    def capture_rendered(module, args, output):
        images = output[0]
        assert all(image.shape[0] == 4 for image in images.values())
        for image in images.values():
            image.retain_grad()
        rendered_calls.append(images)

    handle = renderer.register_forward_hook(capture_rendered)
    try:
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            rendered_calls.clear()
            loss = workspace._loss(obs, cached_targets=targets)
            assert loss.requires_grad and torch.isfinite(loss)
            assert len(rendered_calls) == 1
            assert workspace.loss_metrics["loss"] == loss.item()
            assert workspace.loss_metrics["token_region_observation_count"] == 4
            loss.backward()
            for image in rendered_calls[0].values():
                real_grad, zero_grad = image.grad.chunk(2, dim=0)
                assert torch.isfinite(real_grad).all() and real_grad.abs().sum() > 0
                assert torch.isfinite(zero_grad).all() and zero_grad.abs().sum() > 0
            for name, parameter in renderer.named_parameters():
                if parameter.requires_grad:
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                else:
                    assert parameter.grad is None, name
            assert all(parameter.grad is None for parameter in teacher.parameters())
            optimizer.step()
            _assert_replicas_match(renderer)
            for name, parameter in renderer.named_parameters():
                if name in frozen_renderer:
                    torch.testing.assert_close(parameter, frozen_renderer[name], rtol=0, atol=0)
            for name, value in teacher.state_dict().items():
                torch.testing.assert_close(value, frozen_teacher[name], rtol=0, atol=0)
        assert any(not torch.equal(parameter, initial_trainable[name])
                   for name, parameter in renderer.named_parameters() if parameter.requires_grad)
        assert workspace.shape_info["teacher_token_shape"] == [2, expected_tokens, 16]
        assert workspace.shape_info["student_token_shape"] == [2, expected_tokens, 16]
        assert not renderer.wrench_encoder.training and not teacher.training
    finally:
        handle.remove()


def _worker(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{Path(root) / 'rendezvous'}",
        rank=rank, world_size=2, timeout=timedelta(seconds=60),
    )
    try:
        for upsample in (True, False):
            _exercise_layout(rank, upsample)
    finally:
        dist.destroy_process_group()


def test_paired_force_context_ddp_freezes_teacher_force_path_and_syncs_renderer(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path),), nprocs=2, join=True)
