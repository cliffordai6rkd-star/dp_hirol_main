from types import SimpleNamespace

import pytest
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from diffusion_policy.model.vision.dino_token_baseline_obs_encoder import DinoTokenBaselineObsEncoder
from diffusion_policy.model.vision.force_aware_obs_encoder import ForceAwareObsEncoder
from diffusion_policy.scripts.checkpoint_attention_capture import AttentionCapture


def make_policy(mode="force", *, query="wrench", camera_mode="joint", cross_layers=2,
                self_layers=2, mixed_shapes=True):
    torch.manual_seed(29)
    config = DINOv3ViTConfig(
        image_size=24, patch_size=8, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_register_tokens=2,
    )
    backbone = DINOv3ViTModel(config).eval()
    shapes = {"side": (3, 16, 24), "wrist": (3, 16, 16 if mixed_shapes else 24)}
    options = dict(
        pretrained_model_name_or_path="unused", image_shapes=shapes,
        n_emb=16, n_head=4, max_obs_steps=2, wrench_history_steps=3,
        dropout=0.0, backbone=backbone,
    )
    if mode == "force":
        encoder = ForceAwareObsEncoder(
            **options, pre_cross_self_attn_layers=self_layers,
            cross_attn_layers=cross_layers, cross_attn_query=query,
            camera_attention_mode=camera_mode, force_temporal_encoder="lstm",
        ).eval()
    else:
        encoder = DinoTokenBaselineObsEncoder(
            **options, mode=mode, self_attn_layers=self_layers,
            force_temporal_encoder="gru", force_encoder_layers=1,
        ).eval()
    policy = SimpleNamespace(obs_encoder=encoder, n_obs_steps=2,
                             shape_meta={"obs": {key: {"type": "rgb", "shape": shape}
                                                 for key, shape in shapes.items()}})
    sample = {"images": {key: torch.rand(1, 2, *shape) for key, shape in shapes.items()},
              "wrench_history": torch.randn(1, 2, 3, 6)}
    return policy, sample


@pytest.mark.parametrize("query", ["wrench", "image"])
@pytest.mark.parametrize("camera_mode", ["joint", "per_camera"])
@pytest.mark.parametrize("layer_index", [0, -1])
@pytest.mark.parametrize("query_mode", ["all", "last"])
def test_cross_both_query_directions_exact_camera_slices_and_update_norms(query, camera_mode, layer_index, query_mode):
    policy, sample = make_policy(query=query, camera_mode=camera_mode)
    encoder = policy.obs_encoder
    capture = AttentionCapture(policy, mode="cross-attn", layer_index=layer_index, query_mode=query_mode)
    assert capture.mode == "cross-attn"
    assert capture.query_source == query
    assert capture.has_updates == (query == "image")
    assert capture.layer_count == 2
    records = []
    with torch.inference_mode():
        original = encoder(**sample)
        with capture:
            handle = capture.attention.register_forward_hook(
                lambda module, args, out: records.append(tuple(value.detach().clone() for value in out))
            )
            try:
                actual = encoder(**sample)
            finally:
                handle.remove()
            maps = capture.attention_by_camera()
            if capture.has_updates:
                updates = capture.update_by_camera()
            else:
                with pytest.raises(ValueError, match="image-query"):
                    capture.update_by_camera()
    torch.testing.assert_close(actual, original, rtol=1e-5, atol=1e-6)
    offset = 0
    for index, key in enumerate(encoder.image_keys):
        attended, weights = records[0 if camera_mode == "joint" else index]
        patches = encoder.patch_counts[key]
        start = offset if camera_mode == "joint" else 0
        if query == "image":
            expected = weights[-1, :, start:start + patches].mean(0).T
            torch.testing.assert_close(updates[key], attended[-1, start:start + patches].norm(dim=-1))
        else:
            expected = weights[-1, :, :, start:start + patches].mean(0)
        if query_mode == "last":
            expected = expected[-1:]
        torch.testing.assert_close(maps[key], expected)
        assert maps[key].shape == (3 if query_mode == "all" else 1, patches)
        assert maps[key].device.type == "cpu" and maps[key].dtype == torch.float32
        offset += patches
    if query == "wrench":
        expected_sums = torch.ones(3 if query_mode == "all" else 1)
        if camera_mode == "joint":
            torch.testing.assert_close(sum(value.sum(-1) for value in maps.values()), expected_sums)
        else:
            for value in maps.values():
                torch.testing.assert_close(value.sum(-1), expected_sums)
    elif query_mode == "all":
        for value in maps.values():
            torch.testing.assert_close(value.sum(0), torch.ones(value.shape[-1]))
    assert not capture.attention._forward_hooks
    assert not capture.attention._forward_pre_hooks
    assert not encoder._forward_pre_hooks


@pytest.mark.parametrize("mode", ["force", "self_attn", "image_self_attn"])
@pytest.mark.parametrize("query_mode", ["all", "last"])
@pytest.mark.parametrize("layer_index", [0, -1])
def test_fusion_sa_preserves_wrench_rows_and_latest_observation_columns(mode, query_mode, layer_index):
    policy, sample = make_policy(mode, camera_mode="per_camera")
    capture = AttentionCapture(policy, mode="self-attn", query_mode=query_mode, layer_index=layer_index)
    encoder = policy.obs_encoder
    records = []
    with torch.inference_mode():
        original = encoder(**sample)
        with capture:
            handle = capture.attention.register_forward_hook(
                lambda module, args, out: records.append(out[1].detach().clone())
            )
            try:
                actual = encoder(**sample)
            finally:
                handle.remove()
            maps = capture.attention_by_camera()
    torch.testing.assert_close(actual, original, rtol=1e-5, atol=1e-6)
    weights = records[0]
    patches = sum(encoder.patch_counts.values())
    wrench = 0 if mode == "image_self_attn" else 3
    per_step = patches + wrench
    start = per_step if mode == "self_attn" else 0
    assert capture.temporal_layout == ("flattened_observations" if mode == "self_attn" else "per_observation")
    queries = (slice(start + patches, start + per_step) if wrench else slice(start, start + patches))
    offset = start
    for key in encoder.image_keys:
        count = encoder.patch_counts[key]
        expected = weights[-1, :, queries, offset:offset + count]
        if query_mode == "last":
            expected = expected[:, -1:]
        expected = expected.mean(0) if wrench else expected.mean((0, 1)).unsqueeze(0)
        torch.testing.assert_close(maps[key], expected)
        assert maps[key].shape == (3 if wrench and query_mode == "all" else 1, count)
        offset += count
    # Kept key slices never renormalize: wrench/earlier-time keys retain mass.
    if wrench:
        assert all(torch.all(value.sum(-1) < 1) for value in maps.values())


@pytest.mark.parametrize("mode", ["force", "dino", "image_self_attn", "self_attn"])
@pytest.mark.parametrize("mixed_shapes", [False, True])
@pytest.mark.parametrize("layer_index", [0, -1])
def test_dino_layer_selection_group_order_and_backend_restoration(mode, mixed_shapes, layer_index):
    policy, sample = make_policy(mode, mixed_shapes=mixed_shapes)
    encoder = policy.obs_encoder
    capture = AttentionCapture(policy, mode="dino", layer_index=layer_index)
    backend = capture.attention.config._attn_implementation
    records = []
    before = {key: value.detach().clone() for key, value in encoder.state_dict().items()}
    with torch.inference_mode():
        original = encoder(**sample)
        with capture:
            handle = capture.attention.register_forward_hook(
                lambda module, args, out: records.append(out[1].detach().clone())
            )
            try:
                actual = encoder(**sample)
            finally:
                handle.remove()
            maps = capture.attention_by_camera()
            assert capture.attention.config._attn_implementation == backend
    torch.testing.assert_close(actual, original, rtol=1e-5, atol=1e-6)
    prefix = 1 + encoder.num_register_tokens
    side = records[0][1, :, prefix:, prefix:].mean((0, 1)).unsqueeze(0)
    wrist = (records[1][1] if mixed_shapes else records[0][3])[:, prefix:, prefix:].mean((0, 1)).unsqueeze(0)
    torch.testing.assert_close(maps["side"], side)
    torch.testing.assert_close(maps["wrist"], wrist)
    assert len(records) == (2 if mixed_shapes else 1)
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert not capture.attention._forward_hooks
    assert not capture.attention._forward_pre_hooks
    assert not encoder._forward_pre_hooks


@pytest.mark.parametrize("mode,cross,self_layers,expected", [
    ("force", 2, 2, "cross-attn"),
    ("force", 0, 2, "self-attn"),
    ("force", 0, 0, "dino"),
    ("self_attn", 0, 2, "self-attn"),
    ("image_self_attn", 0, 2, "self-attn"),
    ("dino", 0, 0, "dino"),
])
def test_auto_selects_actual_fusion_then_backbone(mode, cross, self_layers, expected):
    policy, _ = make_policy(mode, cross_layers=cross, self_layers=self_layers)
    capture = AttentionCapture(policy)
    assert capture.mode == expected
    assert capture.architecture["mode"] == expected
    assert capture.architecture["n_obs_steps"] == 2
    assert capture.architecture["image_patch_counts"] == {"side": 6, "wrist": 4}
    assert capture.architecture["selected_heads"] == 4
    assert capture.architecture["force_encoder_layers"] == (1 if mode in {"force", "self_attn"} else 0)
    assert capture.architecture["fusion_self_attn_layers"] == (self_layers if mode != "dino" else 0)
    assert capture.architecture["pre_cross_self_attn_layers"] == (self_layers if mode == "force" else 0)
    assert capture.architecture["cross_attn_layers"] == (cross if mode == "force" else 0)
    assert capture.architecture["wrench_history_steps"] == (3 if mode in {"force", "self_attn"} else 0)
    assert capture.layer_index == capture.layer_count - 1
    if mode == "dino":
        assert AttentionCapture(policy, mode="self-attn").mode == "dino"


@pytest.mark.parametrize("layer", [-3, 2, 1.5])
def test_invalid_layer_lists_available_indices(layer):
    policy, _ = make_policy()
    with pytest.raises(ValueError, match="layer_index"):
        AttentionCapture(policy, layer_index=layer)


def test_missing_mode_lists_available_layers_and_does_not_silently_change_module():
    policy, _ = make_policy(cross_layers=0, self_layers=0)
    with pytest.raises(ValueError, match=r"available: dino \(2 layers\)"):
        AttentionCapture(policy, mode="cross-attn")
    with pytest.raises(ValueError, match="unavailable"):
        AttentionCapture(policy, mode="self-attn")
    with pytest.raises(ValueError, match="query_mode"):
        AttentionCapture(policy, query_mode="unknown")


def test_legacy_force_patch_counts_derive_from_policy_metadata_without_mutation():
    policy, sample = make_policy(query="wrench")
    policy.obs_encoder.patch_counts = {}
    policy.obs_encoder.image_shapes = {}
    with AttentionCapture(policy) as capture:
        policy.obs_encoder(**sample)
        assert capture.patch_counts == {"side": 6, "wrist": 4}
        assert set(capture.attention_by_camera()) == {"side", "wrist"}
    assert policy.obs_encoder.patch_counts == {}
    assert policy.obs_encoder.image_shapes == {}


def test_incomplete_and_failed_passes_restore_backend_and_hooks(monkeypatch):
    policy, sample = make_policy("dino")
    capture = AttentionCapture(policy)
    backend = capture.attention.config._attn_implementation

    def fail(*args, **kwargs):
        assert capture.attention.config._attn_implementation == "eager"
        raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        with capture:
            with pytest.raises(RuntimeError, match="No complete"):
                capture.attention_by_camera()
            monkeypatch.setattr(capture.attention, "forward", fail)
            policy.obs_encoder(**sample)
    assert capture.attention.config._attn_implementation == backend
    assert not capture.active
    assert not capture.attention._forward_hooks
    assert not capture.attention._forward_pre_hooks
    assert not policy.obs_encoder._forward_pre_hooks


def test_new_forward_resets_previous_maps_and_bad_batch_cannot_return_stale_capture():
    policy, sample = make_policy()
    capture = AttentionCapture(policy)
    with capture:
        policy.obs_encoder(**sample)
        capture.attention_by_camera()
        sample["images"]["side"] = sample["images"]["side"].expand(2, -1, -1, -1, -1)
        with pytest.raises(ValueError, match="one sample"):
            policy.obs_encoder(**sample)
        with pytest.raises(RuntimeError, match="No complete"):
            capture.attention_by_camera()


def test_dino_backend_none_is_restored_even_without_forward():
    policy, _ = make_policy("dino")
    capture = AttentionCapture(policy)
    capture.attention.config._attn_implementation = None
    capture._request_dino(capture.attention, (), {})
    assert capture.attention.config._attn_implementation == "eager"
    capture.__exit__()
    assert capture.attention.config._attn_implementation is None
