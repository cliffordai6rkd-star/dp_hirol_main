from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.dino_token_baseline_obs_encoder import DinoTokenBaselineObsEncoder
from diffusion_policy.policy.dino_token_baseline_policy import DinoTokenBaselinePolicy


class PatchBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=12, patch_size=8, num_register_tokens=2)
        self.patches = nn.Conv2d(3, 12, kernel_size=8, stride=8)

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        patches = self.patches(pixel_values).flatten(2).transpose(1, 2)
        prefix = torch.zeros(patches.shape[0], 3, 12, device=patches.device)
        return SimpleNamespace(last_hidden_state=torch.cat([prefix, patches], dim=1))


def make_policy(mode, *, force_temporal_encoder="none", force_encoder_layers=0, inject_encoder=True):
    shape_meta = {"obs": {
        "wrist": {"type": "rgb", "shape": [3, 16, 24]},
        "side": {"type": "rgb", "shape": [3, 16, 24]},
        "tau_ext": {"type": "low_dim", "shape": [8, 7]},
        "future_wrench": {"type": "low_dim", "shape": [8, 6]},
        "future_tau_ext": {"type": "low_dim", "shape": [8, 7]},
    }, "action": {"shape": [7]}}
    encoder = DinoTokenBaselineObsEncoder(
        "unused", {key: [3, 16, 24] for key in ("wrist", "side")},
        mode=mode, n_emb=16, n_head=4, dropout=0.0, backbone=PatchBackbone(),
        force_temporal_encoder=force_temporal_encoder,
        force_encoder_layers=force_encoder_layers,
    ) if inject_encoder else None
    policy = DinoTokenBaselinePolicy(
        baseline_mode=mode, shape_meta=shape_meta,
        noise_scheduler=DDIMScheduler(num_train_timesteps=10, prediction_type="epsilon"),
        horizon=9, n_action_steps=8, n_obs_steps=2, num_inference_steps=2,
        wrench_key="future_wrench", phase_wrench_key="tau_ext",
        future_phase_wrench_key="future_tau_ext", wrench_history_steps=8,
        obs_encoder=encoder, n_emb=16, n_head=4, n_layer=1, n_cond_layers=1,
        p_drop_attn=0.0, dino_model_name_or_path="unused",
        force_temporal_encoder=force_temporal_encoder,
        force_encoder_layers=force_encoder_layers,
        # Even a nonzero inherited curriculum must never mask baseline images.
        mask_schedule={"schedule_type": "constant", "start_probability": 1.0},
    )
    normalizer = LinearNormalizer()
    normalizer.fit({"action": torch.randn(64, 7), "future_wrench": torch.randn(64, 6) + 2})
    policy.set_normalizer(normalizer)
    return policy


def observations():
    return {
        "wrist": torch.rand(2, 2, 3, 16, 24),
        "side": torch.rand(2, 2, 3, 16, 24),
        "tau_ext": torch.zeros(2, 2, 8, 7),
        "future_wrench": torch.randn(2, 2, 8, 6),
        "future_tau_ext": torch.zeros(2, 2, 8, 7),
    }


@pytest.mark.parametrize("mode", ["dino", "image_self_attn"])
def test_visual_baseline_preserves_patches_and_never_reads_force(mode):
    policy = make_policy(mode).eval()
    obs = observations()
    first = policy._encode_context(obs, apply_curriculum_mask=True)
    only_images = {key: obs[key] for key in ("side", "wrist")}
    second = policy._encode_context(only_images, apply_curriculum_mask=False)
    assert first.shape == (2, 24, 16)  # 2 observations x 2 cameras x 6 patches
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not hasattr(policy.obs_encoder, "wrench_projection")
    assert policy.last_curriculum_metrics["curriculum/masked_image_fraction"] == 0
    prediction = policy.predict_action(only_images)
    assert prediction["action"].shape == (2, 8, 7)


def test_self_attention_uses_direct_normalized_force_and_changes_image_tokens():
    policy = make_policy("self_attn").eval()
    obs = observations()
    captured = []
    hook = policy.obs_encoder.wrench_projection.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    first = policy._encode_context(obs, apply_curriculum_mask=True)
    torch.testing.assert_close(captured[0], policy.normalizer["future_wrench"].normalize(obs["future_wrench"]))
    assert first.shape == (2, 40, 16)  # image tokens plus 2 x 8 wrench tokens
    changed = dict(obs, future_wrench=obs["future_wrench"] + 10)
    second = policy._encode_context(changed, apply_curriculum_mask=False)
    hook.remove()
    assert not torch.allclose(first[:, :12], second[:, :12])
    assert not hasattr(policy.obs_encoder, "cross_attention")


@pytest.mark.parametrize("mode,temporal_encoder,layers", [
    ("dino", "none", 0), ("self_attn", "none", 0),
    ("image_self_attn", "none", 0),
    ("self_attn", "gru", 1), ("self_attn", "gru", 3),
    ("self_attn", "lstm", 2), ("self_attn", "transformer", 2),
])
def test_baseline_training_has_no_unused_parameters_and_checkpoint_restores(mode, temporal_encoder, layers):
    kwargs = dict(force_temporal_encoder=temporal_encoder, force_encoder_layers=layers)
    policy = make_policy(mode, **kwargs).train()
    obs = observations()
    loss = policy.compute_loss({"obs": obs, "action": torch.randn(2, 9, 7)})
    assert torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is not None for parameter in policy.parameters() if parameter.requires_grad)
    assert all(parameter.grad is None for parameter in policy.obs_encoder.backbone.parameters())
    assert not policy.obs_encoder.backbone.training
    optimizer = policy.get_optimizer(learning_rate=1e-4)
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in policy.parameters() if p.requires_grad}
    restored = make_policy(mode, **kwargs).eval()
    restored.load_state_dict(policy.state_dict(), strict=True)
    policy.eval()
    a = policy.predict_action(obs, generator=torch.Generator().manual_seed(2))
    b = restored.predict_action(obs, generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(a["action"], b["action"])


def test_image_only_self_attention_runs_once_per_observation_across_cameras():
    policy = make_policy("image_self_attn").eval()
    encoder = policy.obs_encoder
    assert len(encoder.fusion.layers) == 1
    assert not hasattr(encoder, "force_encoder")
    seen = []
    hook = encoder.fusion.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach().clone())
    )
    obs = observations()
    try:
        first = policy._encode_context(obs, apply_curriculum_mask=False).reshape(2, 2, 12, 16)
        changed = dict(obs, side=obs["side"].clone())
        changed["side"][0, 0] += 0.5
        second = policy._encode_context(changed, apply_curriculum_mask=False).reshape(2, 2, 12, 16)
    finally:
        hook.remove()
    assert seen[0].shape == (4, 12, 16)  # B * To, side patches + wrist patches, D
    # The wrist representation reads the changed side camera at the same time.
    assert not torch.allclose(first[0, 0, 6:], second[0, 0, 6:])
    torch.testing.assert_close(first[0, 1], second[0, 1])
    torch.testing.assert_close(first[1], second[1])


def test_gru_keeps_all_tokens_and_resets_each_observation_sequence():
    policy = make_policy("self_attn", force_temporal_encoder="gru", force_encoder_layers=3).eval()
    encoder = policy.obs_encoder
    obs = observations()
    fusion_inputs = []
    projection_inputs = []
    gru_outputs = []
    handles = [
        encoder.fusion.register_forward_pre_hook(
            lambda module, args: fusion_inputs.append(args[0].detach().clone())
        ),
        encoder.wrench_projection.register_forward_pre_hook(
            lambda module, args: projection_inputs.append(args[0].detach().clone())
        ),
        encoder.force_encoder.register_forward_hook(
            lambda module, args, output: gru_outputs.append(output[0].detach().clone())
        ),
    ]
    try:
        with torch.inference_mode():
            policy._encode_context(obs, apply_curriculum_mask=True)
            changed = dict(obs, future_wrench=obs["future_wrench"].clone())
            changed["future_wrench"][0, 0, -1] += 10
            policy._encode_context(changed, apply_curriculum_mask=False)
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(
        projection_inputs[0], policy.normalizer["future_wrench"].normalize(obs["future_wrench"])
    )
    assert encoder.force_encoder.num_layers == 3
    assert gru_outputs[0].shape == (4, 8, 16)
    first, changed = (value.reshape(2, 2, 20, 16) for value in fusion_inputs)
    torch.testing.assert_close(first[:, :, 12:], gru_outputs[0].reshape(2, 2, 8, 16))
    torch.testing.assert_close(first[:, :, :12], changed[:, :, :12])
    torch.testing.assert_close(first[0, 0, 12:-1], changed[0, 0, 12:-1])
    assert not torch.allclose(first[0, 0, -1], changed[0, 0, -1])
    torch.testing.assert_close(first[0, 1], changed[0, 1])
    torch.testing.assert_close(first[1], changed[1])


@pytest.mark.parametrize("layers", [1, 3])
@pytest.mark.parametrize("backbone,encoder_type", [
    ("gru", nn.GRU), ("lstm", nn.LSTM), ("transformer", nn.TransformerEncoder),
])
def test_policy_passes_temporal_configuration_into_its_encoder(monkeypatch, layers, backbone, encoder_type):
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", lambda *args, **kwargs: PatchBackbone())
    policy = make_policy(
        "self_attn", force_temporal_encoder=backbone, force_encoder_layers=layers, inject_encoder=False
    )
    assert isinstance(policy.obs_encoder.force_encoder, encoder_type)
    assert policy.obs_encoder.force_encoder.num_layers == layers
    assert policy._encode_context(observations(), apply_curriculum_mask=False).shape == (2, 40, 16)


@pytest.mark.parametrize("layers", [0, -1])
def test_gru_requires_positive_layer_count(layers):
    with pytest.raises(ValueError, match="force_encoder_layers >= 1"):
        make_policy("self_attn", force_temporal_encoder="gru", force_encoder_layers=layers)


def test_baseline_configs_inherit_corrected_data_and_separate_run_names():
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    root = Path(__file__).resolve().parents[1] / "diffusion_policy/config"
    with initialize_config_dir(config_dir=str(root), version_base=None):
        original = compose(config_name="train_dp_insert_usb_force_aware")
        for name, mode in [("dinobase", "dino"), ("selfattn", "self_attn")]:
            config = compose(config_name=f"baseline_train/train_dp_{name}_insert_usb")
            assert config.name == f"train_dp_{name}_insert_usb"
            assert config.policy.baseline_mode == mode
            assert config.policy.mask_schedule.start_probability == 0
            assert config.policy.mask_schedule.end_probability == 0
            assert OmegaConf.to_container(config.task.dataset) == OmegaConf.to_container(original.task.dataset)
            assert config.training.num_optimizer_steps > 0
            assert config.training.num_epochs is None
            assert config.horizon == original.horizon
