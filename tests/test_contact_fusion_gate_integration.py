"""Checkpoint and training-path integration for the opt-in contact fusion gate."""

import copy
from pathlib import Path

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from test_contact_fusion_gate import make_teacher, observations
from test_configurable_force_fusion import IMAGE_SHAPES, PatchBackbone
from diffusion_policy.dataset.render_net_teacher_cache import TeacherTargetCache
from diffusion_policy.model.vision.render_net import RenderNet
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor
from diffusion_policy.scripts.visualize_cross_attention_episode import instantiate_policy
from diffusion_policy.workspace.train_distill_render_net_workspace import (
    TrainDistillRenderNetWorkspace, student_context_tokens,
)


def test_gated_teacher_action_diffusion_loss_backpropagates():
    teacher = make_teacher().train()
    obs = observations()
    obs["future_tau_ext"][0, 1, -1, 5] = 3.0
    loss = teacher.compute_loss({"obs": obs, "action": torch.randn(2, 9, 7)}, optimizer_step=0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert teacher.obs_encoder.cross_attention.out_proj.weight.grad.abs().sum() > 0
    assert teacher.model.output_head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("objective", ["token_mse", "denoiser_mse"])
def test_gated_online_and_cached_distillation_losses_and_gradients_agree(objective, tmp_path):
    teacher, obs = make_teacher().requires_grad_(False), observations()
    obs["future_tau_ext"][0, 1, -1, 5] = 3.0
    config = {"enabled": True, "source": "context", "selection": "soft", "baseline": "raw_zero",
              "normalization": "image_minmax"}
    workspace = object.__new__(TrainDistillRenderNetWorkspace)
    workspace.cfg = OmegaConf.create({"distillation": {"objective": objective}})
    workspace.device, workspace.teacher_policy = torch.device("cpu"), teacher
    workspace.model = RenderNet(
        6, base_channels=4, conditioning_dim=8, wrench_history_steps=8, embed_dim=16,
        image_self_attn_layers=1, cross_attn_layers=1, decoder_layers=3, patch_size=8,
        num_heads=4, high_attention_render=config,
    ).train()
    workspace.attention_config, workspace.region_source, workspace.attention_enabled = config, "context", True
    workspace.intensity_loss_weight, workspace.context_patch_loss_weight, workspace.smoothness_loss_weight = 1., 1., 0.
    workspace.epoch, workspace.intensity_warmup_epochs, workspace.amp_enabled, workspace.shape_info = 2, 0, False, {}
    tokens, patches = workspace._region_targets(obs, return_patch_attention=True)
    # Exercise disk storage as well as the actual cached-target loss path.
    cache = TeacherTargetCache(tmp_path, {"mode": teacher.future_contact_mode, "gate": True}, 2)
    cache.append(tokens, patches)
    rows = [cache.read(index) for index in range(2)]
    cached_targets = {"tokens": torch.stack([row["tokens"] for row in rows]),
                      "patch_attention": {key: torch.stack([row["patch_attention"][key] for row in rows])
                                          for key in patches}}
    action = torch.randn(2, 9, 7)
    torch.manual_seed(91)
    online = workspace._loss(obs, action=action)
    online.backward()
    gradient = workspace.model.output.weight.grad.clone()
    workspace.model.zero_grad(set_to_none=True)
    torch.manual_seed(91)
    cached = workspace._loss(obs, cached_targets=cached_targets, action=action)
    cached.backward()
    torch.testing.assert_close(cached, online, rtol=0, atol=0)
    torch.testing.assert_close(workspace.model.output.weight.grad, gradient, rtol=0, atol=0)
    assert gradient.abs().sum() > 0
    assert all(parameter.grad is None for parameter in teacher.parameters())
    received = []
    handle = workspace.model.register_forward_pre_hook(lambda module, args: received.append(args[1].detach()))
    try:
        student_context_tokens(teacher, workspace.model, obs)
    finally:
        handle.remove()
    torch.testing.assert_close(received[0], WrenchPreprocessor.from_policy(teacher)(obs)[0], rtol=0, atol=0)


def test_checkpoint_restores_mode_gate_and_old_policy_defaults(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1] / "diffusion_policy/config"
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        cfg = compose(config_name="sim_force_aware_dp_train_20hz_lstm2_gripper_contact_gate")
    from transformers import AutoModel
    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *args, **kwargs: PatchBackbone())
    cfg.policy.n_emb, cfg.policy.n_layer, cfg.policy.p_drop_attn = 16, 1, 0.0
    for key, shape in IMAGE_SHAPES.items():
        cfg.task.shape_meta.obs[key].shape = list(shape)
    policy = hydra.utils.instantiate(cfg.policy).eval()
    policy.normalizer.fit({"future_wrench": torch.randn(32, 6), "action": torch.randn(32, 8)})
    checkpoint = tmp_path / "teacher.ckpt"
    torch.save({"cfg": cfg, "state_dicts": {"model": policy.state_dict()}}, checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    restored = instantiate_policy(payload, payload["cfg"], torch.device("cpu"), capture_attention=False)
    assert restored.future_contact_mode == "token_threshold" and restored.contact_fusion_gate
    obs = {key: torch.rand(1, 2, *shape) for key, shape in IMAGE_SHAPES.items()}
    obs.update(wrench=torch.zeros(1, 2, 8, 6), future_wrench=torch.zeros(1, 2, 8, 6))
    obs["future_wrench"][0, 1, -1, 0] = 9.0
    with torch.no_grad():
        torch.testing.assert_close(restored._encode_context(obs, False), policy._encode_context(obs, False),
                                   rtol=0, atol=0)
    old_cfg = copy.deepcopy(cfg)
    del old_cfg.policy.future_contact_mode
    del old_cfg.policy.contact_fusion_gate
    old = instantiate_policy(payload, old_cfg, torch.device("cpu"), capture_attention=False)
    assert old.future_contact_mode == "legacy_rollout" and not old.contact_fusion_gate


def test_contact_gate_is_applied_without_duplicate_dino_calls():
    teacher, obs = make_teacher(), observations()
    images = {key: obs[key] for key in IMAGE_SHAPES}
    gate = torch.tensor([[False, True], [True, False]])
    calls = []
    handle = teacher.obs_encoder.backbone.register_forward_hook(lambda *args: calls.append(1))
    try:
        with torch.no_grad():
            output = teacher.obs_encoder(
                images=images, wrench_history=obs["future_wrench"], contact_gate=gate
            )
        assert len(calls) == 2  # Each camera size is encoded once, with all observations batched.
    finally:
        handle.remove()
    with torch.no_grad():
        visual = teacher.obs_encoder.encode_image_context(images)
    torch.testing.assert_close(output.reshape(2, 2, 10, 16)[~gate], visual.reshape(2, 2, 10, 16)[~gate],
                               rtol=0, atol=0)
