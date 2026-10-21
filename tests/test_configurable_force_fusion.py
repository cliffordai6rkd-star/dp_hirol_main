from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from hydra import compose, initialize_config_dir
from torch import nn

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.force_aware_obs_encoder import ForceAwareObsEncoder
from diffusion_policy.policy.force_aware_diffusion_transformer_policy import ForceAwareDiffusionTransformerPolicy


class PatchBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=12, patch_size=8, num_register_tokens=2)
        self.patches = nn.Conv2d(3, 12, kernel_size=8, stride=8)

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        patches = self.patches(pixel_values).flatten(2).transpose(1, 2)
        prefix = patches.new_zeros(patches.shape[0], 3, 12)
        return SimpleNamespace(last_hidden_state=torch.cat([prefix, patches], dim=1))


IMAGE_SHAPES = {"side": (3, 16, 16), "wrist": (3, 16, 24)}


def make_encoder(**kwargs):
    return ForceAwareObsEncoder(
        "unused", image_shapes=IMAGE_SHAPES, backbone=PatchBackbone(),
        n_emb=16, n_head=4, max_obs_steps=2, wrench_history_steps=8,
        force_encoder_layers=2, dropout=0.0, **kwargs,
    )


def observations():
    return {
        **{key: torch.rand(2, 2, *shape) for key, shape in IMAGE_SHAPES.items()},
        "wrench_ext": torch.randn(2, 2, 8, 6),
    }


def make_policy(monkeypatch, **kwargs):
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", lambda *a, **kw: PatchBackbone())
    policy = ForceAwareDiffusionTransformerPolicy(
        shape_meta={"obs": {
            **{key: {"shape": shape, "type": "rgb"} for key, shape in IMAGE_SHAPES.items()},
            "wrench_ext": {"shape": (8, 6), "type": "low_dim"},
        }, "action": {"shape": (7,)}},
        noise_scheduler=DDIMScheduler(num_train_timesteps=10, prediction_type="epsilon"),
        horizon=9, n_obs_steps=2, n_action_steps=8, num_inference_steps=2,
        dino_model_name_or_path="unused", n_emb=16, n_head=4, n_layer=1,
        p_drop_attn=0.0, mask_schedule={"schedule_type": "constant", "start_probability": 0.0},
        **kwargs,
    )
    normalizer = LinearNormalizer()
    normalizer.fit({"action": torch.randn(64, 7), "wrench_ext": torch.randn(64, 6)})
    policy.set_normalizer(normalizer)
    return policy


@pytest.mark.parametrize("temporal", ["gru", "lstm", "transformer"])
@pytest.mark.parametrize("sa_layers", [0, 2])
@pytest.mark.parametrize("query", ["wrench", "image"])
@pytest.mark.parametrize("ca_layers", [0, 1, 2])
@pytest.mark.parametrize("camera_mode", ["joint", "per_camera"])
def test_fusion_combinations_preserve_tokens_and_backpropagate(temporal, sa_layers, query, ca_layers, camera_mode):
    encoder = make_encoder(
        force_temporal_encoder=temporal, pre_cross_self_attn_layers=sa_layers,
        cross_attn_query=query, cross_attn_layers=ca_layers, camera_attention_mode=camera_mode,
    ).train()
    encoder.set_attention_capture(True)
    obs = observations()
    context = encoder(
        images={key: obs[key] for key in IMAGE_SHAPES}, wrench_history=obs["wrench_ext"],
        image_token_mask=torch.tensor([[False, True], [False, False]]),
    )
    per_obs = 18 if ca_layers == 0 else (8 if query == "wrench" else 10)
    assert encoder.context_tokens_per_observation == per_obs
    assert context.shape == (2, 2 * per_obs, 16)
    assert torch.isfinite(context).all()
    assert len(encoder.last_cross_attention_weights_by_layer) == ca_layers
    for layer_weights in encoder.last_cross_attention_weights_by_layer:
        maps = layer_weights.values() if isinstance(layer_weights, dict) else [layer_weights]
        for weights in maps:
            assert weights.shape[:2] == (4, 4)
            if query == "wrench":
                assert weights.shape[2] == 8
            else:
                assert weights.shape[3] == 8
            torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
    (context * torch.randn_like(context)).mean().backward()
    assert encoder.wrench_projection.weight.grad.abs().sum() > 0
    assert encoder.image_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is not None for p in encoder.force_encoder.parameters())
    # The historical per-camera/no-SA branch omits image position embeddings.
    if not (ca_layers and query == "wrench" and camera_mode == "per_camera" and not sa_layers):
        assert all(p.grad is not None for p in encoder.parameters() if p.requires_grad)
    assert all(p.grad is None for p in encoder.backbone.parameters())
    if ca_layers == 0:
        assert encoder.last_cross_attention_weights is None
        assert not list(encoder.cross_attention.parameters())
    encoder.set_attention_capture(False)
    assert encoder.last_cross_attention_weights_by_layer == []


@pytest.mark.parametrize("query", ["wrench", "image"])
def test_cross_layers_use_correct_query_and_fixed_post_sa_memory(query):
    encoder = make_encoder(pre_cross_self_attn_layers=1, cross_attn_layers=2, cross_attn_query=query).eval()
    seen, inputs = [], []
    handles = [encoder.pre_cross_self_attention.register_forward_hook(
        lambda module, args, output: seen.append(output.detach().clone())
    )]
    for block in (encoder, *encoder.extra_cross_attention_layers):
        handles.append(block.cross_attention.register_forward_pre_hook(
            lambda module, args, kwargs: inputs.append({
                key: kwargs[key].detach().clone() for key in ("query", "key", "value")
            }), with_kwargs=True,
        ))
    obs = observations()
    try:
        encoder(images={key: obs[key] for key in IMAGE_SHAPES}, wrench_history=obs["wrench_ext"])
    finally:
        for handle in handles:
            handle.remove()
    images, force = seen[0][:, :10], seen[0][:, 10:]
    expected_query = encoder.force_norm(force) if query == "wrench" else encoder.image_norm(images)
    expected_memory = images if query == "wrench" else force
    torch.testing.assert_close(inputs[0]["query"], expected_query)
    for values in inputs:
        torch.testing.assert_close(values["value"], expected_memory)
    assert not torch.allclose(inputs[0]["query"], inputs[1]["query"])


@pytest.mark.parametrize("sa_layers", [0, 2])
def test_zero_cross_returns_both_streams_and_ignores_query_selection(sa_layers):
    encoder = make_encoder(pre_cross_self_attn_layers=sa_layers, cross_attn_layers=0).eval()
    other = make_encoder(pre_cross_self_attn_layers=sa_layers, cross_attn_layers=0, cross_attn_query="image").eval()
    other.load_state_dict(encoder.state_dict(), strict=True)
    seen = []
    handle = encoder.context_norm.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    obs = observations()
    try:
        first = encoder(images={key: obs[key] for key in IMAGE_SHAPES}, wrench_history=obs["wrench_ext"])
        second = other(images={key: obs[key] for key in IMAGE_SHAPES}, wrench_history=obs["wrench_ext"])
    finally:
        handle.remove()
    torch.testing.assert_close(first, second)
    assert seen[0].shape == (2, 36, 16)
    # Disabling cross-attention must not leave an accidental force-only path.
    changed = dict(obs, side=obs["side"] + 0.2)
    changed_context = encoder(images={key: changed[key] for key in IMAGE_SHAPES}, wrench_history=obs["wrench_ext"])
    assert not torch.allclose(first, changed_context)


def test_image_query_keeps_visual_residual_and_reads_wrench():
    encoder = make_encoder(cross_attn_query="image").eval()
    obs = observations()
    kwargs = dict(images={key: obs[key] for key in IMAGE_SHAPES}, wrench_history=obs["wrench_ext"])
    first = encoder(**kwargs)
    changed = encoder(**dict(kwargs, wrench_history=obs["wrench_ext"] + 3))
    assert not torch.allclose(first, changed)
    for module in (encoder.cross_attention, encoder.cross_ffn):
        for parameter in module.parameters():
            nn.init.zeros_(parameter)
    seen = []
    handle = encoder.image_norm.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    try:
        actual = encoder(**kwargs)
    finally:
        handle.remove()
    expected = encoder.context_norm(seen[0].reshape(2, 20, 16))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("query,ca_layers,sa_layers,temporal", [
    ("wrench", 1, 0, "gru"), ("wrench", 2, 1, "lstm"),
    ("image", 1, 0, "transformer"), ("image", 2, 1, "gru"),
    ("wrench", 0, 0, "lstm"), ("image", 0, 2, "transformer"),
])
def test_policy_sizes_dit_trains_predicts_and_restores(monkeypatch, query, ca_layers, sa_layers, temporal):
    kwargs = dict(cross_attn_query=query, cross_attn_layers=ca_layers,
                  pre_cross_self_attn_layers=sa_layers, force_temporal_encoder=temporal, force_encoder_layers=2)
    policy = make_policy(monkeypatch, **kwargs).train()
    expected_tokens = 36 if ca_layers == 0 else (16 if query == "wrench" else 20)
    assert policy.model.max_context_tokens == expected_tokens
    obs = observations()
    loss = policy.compute_loss({"obs": obs, "action": torch.randn(2, 9, 7)})
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in policy.parameters() if p.requires_grad)
    restored = make_policy(monkeypatch, **kwargs).eval()
    restored.load_state_dict(policy.state_dict(), strict=True)
    policy.eval()
    with torch.inference_mode():
        first = policy.predict_action(obs, generator=torch.Generator().manual_seed(42))
        second = restored.predict_action(obs, generator=torch.Generator().manual_seed(42))
    assert first["action"].shape == (2, 8, 7)
    torch.testing.assert_close(first["action"], second["action"])


def test_freeze_cross_attention_includes_every_layer(monkeypatch):
    policy = make_policy(monkeypatch, cross_attn_layers=3, freeze_modules=["cross_attn"])
    for block in (policy.obs_encoder, *policy.obs_encoder.extra_cross_attention_layers):
        assert all(not p.requires_grad for p in block.cross_attention.parameters())
        assert all(p.requires_grad for p in block.cross_ffn.parameters())


@pytest.mark.parametrize("kwargs,message", [
    ({"cross_attn_layers": -1}, "cross_attn_layers must be non-negative"),
    ({"cross_attn_query": "invalid"}, "cross_attn_query must be"),
    ({"pre_cross_self_attn_layers": -1}, "pre_cross_self_attn_layers must be non-negative"),
    ({"force_temporal_encoder": "invalid"}, "force_temporal_encoder must be"),
])
def test_invalid_fusion_options_fail_explicitly(kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_encoder(**kwargs)


def test_training_config_accepts_fusion_overrides():
    root = Path(__file__).resolve().parents[1] / "diffusion_policy/config"
    with initialize_config_dir(config_dir=str(root), version_base=None):
        cfg = compose(config_name="baseline_train/train_dp_self_crossattn_insert_usb", overrides=[
            "policy.pre_cross_self_attn_layers=0", "policy.cross_attn_layers=0",
            "policy.cross_attn_query=image", "policy.force_temporal_encoder=lstm",
            "policy.force_encoder_layers=3",
        ])
    assert cfg.policy.pre_cross_self_attn_layers == cfg.policy.cross_attn_layers == 0
    assert cfg.policy.cross_attn_query == "image"
    assert cfg.policy.force_temporal_encoder == "lstm"
    assert cfg.policy.force_encoder_layers == 3


@pytest.mark.parametrize("query,layers,message", [
    ("wrench", 0, "no cross-attention weights"),
    ("image", 1, "requires wrench queries"),
])
def test_spatial_heatmap_rejects_incompatible_cross_attention(monkeypatch, query, layers, message):
    from diffusion_policy.scripts import visualize_cross_attention_episode as visualization

    policy = make_policy(monkeypatch, cross_attn_query=query, cross_attn_layers=layers)
    monkeypatch.setattr(visualization.hydra.utils, "instantiate", lambda cfg: policy)
    with pytest.raises(ValueError, match=message):
        visualization.instantiate_policy({}, SimpleNamespace(policy={}), torch.device("cpu"))
