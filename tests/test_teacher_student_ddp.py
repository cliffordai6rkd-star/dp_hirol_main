"""Two real CPU DDP replicas of the alpha teacher and active student architecture."""

from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel

from test_contact_alpha import make_teacher, observations
from diffusion_policy.model.vision.render_net import RenderNet
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor
from diffusion_policy.workspace.train_distill_render_net_workspace import TrainDistillRenderNetWorkspace
from diffusion_policy.workspace.train_force_aware_diffusion_workspace import DistributedContext


def assert_replicas_match(module):
    state = torch.cat([p.detach().flatten() for p in module.parameters()])
    replicas = [torch.empty_like(state) for _ in range(dist.get_world_size())]
    dist.all_gather(replicas, state)
    for replica in replicas[1:]:
        torch.testing.assert_close(replica, replicas[0], rtol=0, atol=0)


def _worker(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{Path(root) / 'rendezvous'}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        teacher = make_teacher().train()
        wrapped = DistributedDataParallel(teacher, broadcast_buffers=False)
        optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-3)
        torch.manual_seed(101 + rank)
        obs = observations(1)
        # One rank is free; the other has a partially contacting future chunk.
        if rank == 1:
            obs["future_tau_ext"][..., -4:, 5] = 3.
        obs["contact_alpha"] = WrenchPreprocessor.from_policy(teacher).contact_alpha(obs)
        action = torch.randn(1, 9, 7)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = wrapped({"obs": obs, "action": action}, optimizer_step=step)
            loss.backward()
            optimizer.step()
        assert_replicas_match(teacher)
        del wrapped
        teacher.eval().requires_grad_(False)
        teacher.zero_grad(set_to_none=True)

        cfg = {"enabled": True, "source": "context", "selection": "soft",
               "baseline": "raw_zero", "normalization": "image_minmax"}
        workspace = object.__new__(TrainDistillRenderNetWorkspace)
        workspace.cfg = OmegaConf.create({"distillation": {"objective": "denoiser_mse"},
            "training": {"distributed": {"find_unused_parameters": False}}})
        workspace.distributed = DistributedContext(rank=rank, local_rank=rank, world_size=2)
        workspace.device, workspace.teacher_policy = torch.device("cpu"), teacher
        torch.manual_seed(7)
        workspace.model = RenderNet(
            6, conditioning_dim=16, base_channels=4, wrench_history_steps=8,
            patch_size=8, embed_dim=16, num_heads=4, image_self_attn_layers=1,
            cross_attn_layers=1, decoder_layers=3, high_attention_render=cfg,
        ).train()
        workspace.attention_enabled, workspace.attention_config, workspace.region_source = True, cfg, "context"
        workspace.intensity_loss_weight, workspace.context_patch_loss_weight, workspace.smoothness_loss_weight = 1., 1., 0.
        workspace.amp_enabled, workspace.shape_info, workspace.intensity_warmup_epochs = False, {}, 2
        workspace._setup_distributed_model()
        optimizer = torch.optim.AdamW(workspace.model.parameters(), lr=1e-3)
        torch.manual_seed(201 + rank)
        for epoch in (0, 0, 2, 2):
            workspace.epoch = epoch
            optimizer.zero_grad(set_to_none=True)
            loss = workspace._loss(obs, action=action)
            assert torch.isfinite(loss)
            loss.backward()
            if workspace.model.output.weight.grad is not None:
                assert torch.isfinite(workspace.model.output.weight.grad).all()
            if epoch > 0:
                assert workspace.model.output.weight.grad is not None
                assert workspace.model.output.weight.grad.abs().sum() > 0
            optimizer.step()
            assert_replicas_match(workspace.model)
        assert all(p.grad is None for p in teacher.parameters())
    finally:
        dist.destroy_process_group()


def test_alpha_teacher_and_student_warmup_then_joint_ddp(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path),), nprocs=2, join=True)
