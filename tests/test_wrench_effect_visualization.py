"""Focused checks for wrench ablation visualisation helpers.

These tests use a tiny image-query attention module so the visualization can be
checked without loading DINO weights or running a diffusion sampler.
"""

from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf

from diffusion_policy.scripts import visualize_wrench_effect_episode as visual

from diffusion_policy.scripts.visualize_wrench_effect_episode import (
    WrenchEffectCapture,
    capture_pair,
    effect_color_bounds,
    make_zero_wrench_obs,
)


class TinyCrossAttention(nn.Module):
    def __init__(self, dim=3):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(7.0))
        self.dim = dim

    def forward(self, query, key, value, **kwargs):
        # The constant bias is present in both runs and must disappear after
        # subtraction. The value stream is deliberately wrench dependent.
        update = value[..., : self.dim].mean(dim=-2, keepdim=True).expand_as(query) + self.bias
        weights = torch.full((*query.shape[:-1], key.shape[-2]), 1 / key.shape[-2])
        return update, weights.unsqueeze(1)


class TinyEncoder(nn.Module):
    def __init__(self, query="image", extra_layers=0):
        super().__init__()
        self.cross_attention = TinyCrossAttention()
        self.extra_cross_attention_layers = nn.ModuleList(
            [TinyCrossAttention() for _ in range(extra_layers)]
        )
        self.cross_attn_query = query
        self.cross_attn_layers = 1 + extra_layers
        self.pre_cross_self_attn_layers = 0
        self.image_keys = ("side", "wrist")
        self.camera_attention_mode = "joint"
        self.patch_counts = {"side": 1, "wrist": 2}
        self.backbone = SimpleNamespace(config=SimpleNamespace(patch_size=2))
        self.received_wrench = []

    def forward(self, images, wrench_history):
        # [B, To, P, D], with image tokens in the query.  The image values are
        # held fixed by make_zero_wrench_obs; wrench changes only the value.
        self.received_wrench.append(wrench_history.detach().clone())
        batch, steps = wrench_history.shape[:2]
        queries = [F.avg_pool2d(images[key].flatten(0, 1), 2).flatten(2).transpose(1, 2)
                   for key in self.image_keys]
        query = torch.cat(queries, dim=1)
        value = wrench_history[..., :3].reshape(batch * steps, -1, 3)
        if self.camera_attention_mode == "joint":
            update, _ = self.cross_attention(query, value, value)
        else:
            update = torch.cat([self.cross_attention(q, value, value)[0] for q in queries], dim=1)
        context = query + update
        for layer in self.extra_cross_attention_layers:
            context = context + layer(context, value, value)[0]
        return context.reshape(batch, steps * query.shape[1], 3)


class TinyPolicy(nn.Module):
    image_keys = ("side", "wrist")
    wrench_key = "wrench"
    phase_wrench_key = "wrench"
    future_phase_wrench_key = "future_wrench"
    n_obs_steps = 2

    def __init__(self, query="image", extra_layers=0):
        super().__init__()
        self.image_keys = self.__class__.image_keys
        self.obs_encoder = TinyEncoder(query, extra_layers)
        self.prepare_calls = []
        self.encode_calls = 0
        self.eval()

    def _encode_context(self, obs, apply_curriculum_mask=False):
        assert apply_curriculum_mask is False
        self.encode_calls += 1
        images, normalized, _, _ = self.prepare_context_inputs(obs)
        return self.obs_encoder(images=images, wrench_history=normalized)

    def prepare_context_inputs(self, obs):
        self.prepare_calls.append(obs)
        contact = obs[self.phase_wrench_key][..., :3].abs().sum(dim=(-1, -2)) > 0
        future = obs[self.future_phase_wrench_key][..., :3].abs().sum(dim=(-1, -2)) > 0
        return {key: obs[key] for key in self.image_keys}, obs[self.wrench_key] * 2, contact, future

    def predict_action(self, *args, **kwargs):
        raise AssertionError("Wrench visualization must not run diffusion sampling")


def tiny_obs(batch=1, obs_steps=2):
    images = torch.arange(batch * obs_steps * 3 * 2 * 2, dtype=torch.float32).reshape(batch, obs_steps, 3, 2, 2)
    wrench = torch.zeros(batch, obs_steps, 2, 6)
    wrench[..., 0, 0] = 2.
    wrench[..., 1, 0] = 4.
    return {
        "side": images,
        "wrist": torch.cat((images + 10, images + 20), dim=-1),
        "wrench": wrench,
        "future_wrench": wrench.clone(),
        "unrelated": torch.tensor([3.]),
    }


def test_make_zero_wrench_obs_only_copies_and_zeros_raw_sensor_keys():
    policy = TinyPolicy()
    obs = tiny_obs()
    zero = make_zero_wrench_obs(policy, obs)
    assert zero is not obs
    assert zero["side"] is obs["side"] and zero["wrist"] is obs["wrist"]
    assert zero["unrelated"] is obs["unrelated"]
    assert torch.count_nonzero(obs["wrench"]) > 0
    assert torch.count_nonzero(obs["future_wrench"]) > 0
    assert torch.count_nonzero(zero["wrench"]) == 0
    assert torch.count_nonzero(zero["future_wrench"]) == 0


def test_make_zero_wrench_obs_deduplicates_shared_phase_key():
    policy = TinyPolicy()
    policy.phase_wrench_key = policy.wrench_key
    zero = make_zero_wrench_obs(policy, tiny_obs())
    assert set(zero).issuperset({"wrench", "future_wrench"})
    assert torch.count_nonzero(zero["wrench"]) == 0


def test_capture_pair_subtracts_before_norm_and_cancels_encoder_bias():
    policy = TinyPolicy()
    obs = tiny_obs()
    with WrenchEffectCapture(policy) as capture:
        full = capture.encode(obs)
        zero = capture.encode(make_zero_wrench_obs(policy, obs))
    assert full["context"].shape == zero["context"].shape == (2, 3, 3)
    pair = capture_pair(policy, obs, capture=None)
    assert pair["context_effect"].shape == (2, 3)
    expected = torch.linalg.vector_norm(full["context"] - zero["context"], dim=-1)
    torch.testing.assert_close(torch.as_tensor(pair["context_effect"]), expected)
    # A norm of the two runs would include the constant bias and be much larger.
    assert float(pair["context_effect"].max()) < float(torch.linalg.vector_norm(full["context"], dim=-1).max())
    assert pair["ca_update"].shape == pair["injection_effect"].shape == (2, 3)
    assert pair["normalized_wrench"].shape[-1] == 6
    assert pair["contact"].shape == pair["future_has_contact"].shape == (2,)


def test_effect_is_vector_difference_not_difference_of_norms():
    # Equal-norm vectors pointing in opposite directions have a nonzero effect;
    # subtracting their norms would incorrectly report zero.
    full = torch.tensor([[[1., 0., 0.], [0., 2., 0.]]]).expand(2, -1, -1)
    zero = -full

    class Capture:
        calls = 0

        def encode(self, obs, **kwargs):
            result = full if self.calls == 0 else zero
            self.calls += 1
            return {"context": result, "ca_update": result}

    pair = capture_pair(TinyPolicy(), tiny_obs(), Capture())
    np.testing.assert_allclose(pair["injection_effect"], [[2., 4.], [2., 4.]])
    np.testing.assert_allclose(pair["context_effect"], [[2., 4.], [2., 4.]])
    np.testing.assert_allclose(pair["ca_update"], pair["baseline_ca_update"])


def test_effect_color_bounds_share_all_cameras_and_observation_slots():
    records = [
        {"cameras": {"side": {"context_effect": np.array([[1., 2.], [3., 4.]])},
                      "wrist": {"context_effect": np.array([[5., 6.], [7., 8.]])}}},
        {"cameras": {"side": {"context_effect": np.array([[9., 10.], [11., 12.]])},
                      "wrist": {"context_effect": np.array([[13., 14.], [15., 16.]])}}},
    ]
    assert effect_color_bounds(records, "context_effect", (0, 100)) == (1., 16.)
    np.testing.assert_allclose(effect_color_bounds(records, "context_effect", (25, 75)), [4.75, 12.25])


@pytest.mark.parametrize("bad", [np.array([[np.nan]]), np.array([[-1.]])])
def test_effect_color_bounds_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        effect_color_bounds([{"cameras": {"side": {"context_effect": bad}}}], "context_effect")


def test_effect_color_bounds_degenerate_zero_and_constant_ranges():
    records = [{"cameras": {"side": {"context_effect": np.zeros((2, 3))}}}]
    assert effect_color_bounds(records, "context_effect") == (0., 1.)
    records[0]["cameras"]["side"]["context_effect"][:] = 4.
    assert effect_color_bounds(records, "context_effect") == (0., 4.)


def test_each_image_uses_its_own_p2_p98_without_pooling_cameras_or_observations():
    shape = np.arange(100, dtype=np.float32)
    side = np.stack((shape, shape * 3 + 100))
    wrist = np.stack((shape * .01 + 5, shape * 10 + 1000))
    side_bounds = visual.image_color_bounds(side)
    wrist_bounds = visual.image_color_bounds(wrist)
    np.testing.assert_allclose(side_bounds, [[1.98, 97.02], [105.94, 391.06]], rtol=1e-6)
    np.testing.assert_allclose(wrist_bounds, [[5.0198, 5.9702], [1019.8, 1970.2]], rtol=1e-6)
    assert side_bounds.shape == wrist_bounds.shape == (2, 2)
    normalized = lambda image, bounds: np.clip((image - bounds[0]) / (bounds[1] - bounds[0]), 0, 1)
    expected = normalized(side[0], side_bounds[0])
    for images, bounds in ((side, side_bounds), (wrist, wrist_bounds)):
        for image, limits in zip(images, bounds):
            np.testing.assert_allclose(normalized(image, limits), expected, atol=1e-6)
    # A later frame also gets a fresh range, even when its scale is much larger.
    next_frame = visual.image_color_bounds(side * 50 + 700)
    np.testing.assert_allclose(next_frame, side_bounds * 50 + 700, rtol=1e-6)


def test_per_image_color_bounds_handle_zero_constant_and_invalid_maps():
    np.testing.assert_array_equal(visual.image_color_bounds(np.zeros((2, 4))), [[0., 1.], [0., 1.]])
    np.testing.assert_array_equal(visual.image_color_bounds(np.full((2, 4), 7.)), [[0., 7.], [0., 7.]])
    for bad in (np.array([[np.nan, 1.]]), np.array([[-1., 1.]])):
        with pytest.raises(ValueError):
            visual.image_color_bounds(bad)


def test_per_image_affine_maps_have_matching_heat_colors():
    pattern = np.arange(100, dtype=np.float32)[None]
    transformed = pattern * 10 + 20
    bounds = visual.image_color_bounds(pattern)[0]
    transformed_bounds = visual.image_color_bounds(transformed)[0]
    image = np.zeros((10, 100, 3), dtype=np.uint8)
    first = visual.heat_overlay(image, pattern[0], (1, 100), bounds, 1.)
    second = visual.heat_overlay(image, transformed[0], (1, 100), transformed_bounds, 1.)
    # Small floating-point boundary changes can move the JET lookup by 1 slot.
    assert np.abs(first.astype(int) - second.astype(int)).max() <= 4


def test_color_scope_defaults_to_image_and_retains_explicit_episode_mode():
    base = ["--checkpoint", "model.ckpt", "--episode", "0"]
    assert visual.parse_args(base).color_scope == "image"
    assert visual.parse_args(base + ["--color-scope", "episode"]).color_scope == "episode"


@pytest.mark.parametrize("flat", [False, True])
def test_render_uses_independent_panel_bounds_and_labels_flat_images(monkeypatch, flat):
    values = {
        "side": np.array([[1., 2., 3., 4.], [10., 20., 30., 40.]], dtype=np.float32),
        "wrist": np.array([[101., 102., 103., 104.], [1000., 2000., 3000., 4000.]], dtype=np.float32),
    }
    if flat:
        values = {"side": np.zeros((2, 4), dtype=np.float32),
                  "wrist": np.full((2, 4), 7., dtype=np.float32)}
    record = {"anchor": 3, "cameras": {
        key: {"images": np.zeros((2, 16, 256, 3), dtype=np.uint8), "grid": (1, 4),
              "ca_update": maps, "ca_update_color_bounds": visual.image_color_bounds(maps)}
        for key, maps in values.items()
    }}
    args = visual.parse_args(["--checkpoint", "model.ckpt", "--episode", "0", "--observation", "all",
                              "--alpha", "1"])
    overlays, labels = [], []
    original_overlay = visual.heat_overlay

    def capture_overlay(image, patches, grid, bounds, alpha):
        rendered = original_overlay(image, patches, grid, bounds, alpha)
        overlays.append((np.asarray(patches).copy(), tuple(bounds), rendered))
        return rendered

    monkeypatch.setattr(visual, "heat_overlay", capture_overlay)
    monkeypatch.setattr(visual, "_write", lambda image, text, *a, **kw: labels.append(text))
    chart = SimpleNamespace(render=lambda time: np.zeros((20, 512, 3), dtype=np.uint8))
    frame = visual.render_frame(record, 0, ["side", "wrist"], (0., 10000.), args, "ca_update",
                                {"anchor_times": np.array([.15])}, chart)
    assert frame.shape[1] == 512 and frame.flags.c_contiguous
    assert len(overlays) == 4
    for index, (key, obs) in enumerate((("side", 0), ("wrist", 0), ("side", 1), ("wrist", 1))):
        bounds = record["cameras"][key]["ca_update_color_bounds"][obs]
        np.testing.assert_allclose(overlays[index][1], bounds)
        assert f"{bounds[0]:.5g}" in labels and f"{bounds[1]:.5g}" in labels
    assert any("Each image: own p2-p98" in label for label in labels)
    assert any("within-image" in label for label in labels)
    if flat:
        assert labels.count("flat") == 4
        for patches, bounds, rendered in overlays:
            # Positive constant maps, as well as zeros, use a spatially
            # uniform cold color rather than a fabricated hot region.
            np.testing.assert_array_equal(patches, bounds[0])
            assert np.unique(rendered.reshape(-1, 3), axis=0).shape[0] == 1
        np.testing.assert_array_equal(overlays[0][2], overlays[1][2])
    else:
        assert "flat" not in labels
        reference = overlays[0][2].astype(int)
        assert max(np.abs(entry[2].astype(int) - reference).max() for entry in overlays) <= 4


def test_capture_cleans_hooks_and_rejects_non_image_query_or_extra_ca():
    policy = TinyPolicy(query="wrench")
    capture = WrenchEffectCapture(policy)
    with pytest.raises(ValueError, match="image-query|image query|supported"):
        capture.__enter__()
    assert not getattr(capture, "handles", [1])
    policy = TinyPolicy(extra_layers=1)
    with pytest.raises(ValueError, match="one|single|supported"):
        WrenchEffectCapture(policy).__enter__()


def test_context_only_two_layer_tau_ext_matches_training_supervision():
    from diffusion_policy.model.vision.render_net_attention import teacher_context_effect_targets

    class TauPolicy(TinyPolicy):
        wrench_key = phase_wrench_key = "tau_ext"
        future_phase_wrench_key = None

        def prepare_context_inputs(self, obs):
            self.prepare_calls.append(obs)
            contact = obs["tau_ext"].abs().sum(dim=(-1, -2)) > 0
            return {key: obs[key] for key in self.image_keys}, obs["tau_ext"] * 2, contact, None

    policy = TauPolicy(extra_layers=1).requires_grad_(False)
    obs = tiny_obs()
    obs.pop("future_wrench")
    obs["tau_ext"] = F.pad(obs.pop("wrench"), (0, 1))
    obs["tau_ext"][..., 6] = 3.
    obs["contact_alpha"] = torch.ones(1, policy.n_obs_steps)
    with WrenchEffectCapture(policy, capture_residuals=False) as capture:
        pair = capture_pair(policy, obs, capture, save_vectors=True)
    _, expected = teacher_context_effect_targets(policy, obs)
    expected = torch.cat([expected[key] for key in policy.image_keys], dim=-1)[0]
    torch.testing.assert_close(torch.from_numpy(pair["context_effect"]), expected)
    assert pair["normalized_wrench"].shape == (2, 2, 7)
    assert np.count_nonzero(pair["baseline_normalized_wrench"]) == 0
    assert np.any(pair["context_effect"] > 0)
    assert pair["delta_z_vector"].shape == (2, 3, 3)
    assert "ca_update" not in pair and "delta_u_vector" not in pair
    baseline_obs = policy.prepare_calls[1]
    assert "contact_alpha" not in baseline_obs
    assert baseline_obs["side"] is obs["side"]
    assert not policy.obs_encoder._forward_pre_hooks
    assert not policy.obs_encoder.cross_attention._forward_hooks


@pytest.mark.parametrize("mode", ["joint", "per_camera"])
def test_capture_preserves_observation_then_camera_patch_order_and_detaches(mode):
    policy = TinyPolicy()
    policy.obs_encoder.camera_attention_mode = mode
    obs = tiny_obs()
    with WrenchEffectCapture(policy) as capture:
        result = capture.encode(obs)
    assert result["ca_update"].shape == result["context"].shape == (2, 3, 3)
    for tensor in result.values():
        assert tensor.device.type == "cpu" and tensor.dtype == torch.float32
        assert not tensor.requires_grad
    # Identical U is broadcast to all patches here. Subtracting U from context
    # recovers the fixed image queries and exposes camera/observation order.
    image_queries = result["context"] - result["ca_update"]
    expected = torch.cat([
        F.avg_pool2d(obs[key].flatten(0, 1), 2).flatten(2).transpose(1, 2)
        for key in policy.image_keys
    ], dim=1)
    torch.testing.assert_close(image_queries, expected)
    assert not policy.obs_encoder.cross_attention._forward_hooks


def test_normalized_zero_preserves_phase_preprocessing_and_only_replaces_encoder_input():
    policy = TinyPolicy()
    obs = tiny_obs()
    pair = capture_pair(policy, obs, baseline="normalized_zero")
    assert policy.encode_calls == 2
    assert all(value is obs for value in policy.prepare_calls)
    assert torch.count_nonzero(policy.obs_encoder.received_wrench[0]) > 0
    assert torch.count_nonzero(policy.obs_encoder.received_wrench[1]) == 0
    assert np.all(pair["contact"]) and np.all(pair["future_has_contact"])
    np.testing.assert_array_equal(pair["baseline_normalized_wrench"], 0)
    assert np.any(pair["normalized_wrench"] != 0)
    assert not policy.obs_encoder._forward_pre_hooks


def test_raw_zero_runs_complete_bias_path_but_cancels_it_in_delta():
    policy = TinyPolicy()
    obs = tiny_obs()
    pair = capture_pair(policy, obs, baseline="raw_zero", save_vectors=True)
    assert policy.encode_calls == 2
    assert policy.prepare_calls[0] is obs
    baseline_obs = policy.prepare_calls[1]
    assert baseline_obs["side"] is obs["side"] and baseline_obs["wrist"] is obs["wrist"]
    assert torch.count_nonzero(baseline_obs["wrench"]) == 0
    assert torch.count_nonzero(baseline_obs["future_wrench"]) == 0
    np.testing.assert_allclose(pair["baseline_ca_update_vector"], 7.)
    assert np.all(pair["baseline_ca_update"] > 0)
    np.testing.assert_allclose(pair["delta_u_vector"], pair["ca_update_vector"] - 7.)


def test_normalized_zero_pre_hook_is_removed_when_encoder_raises():
    policy = TinyPolicy()
    original_forward = policy.obs_encoder.forward

    def fail(images, wrench_history):
        assert torch.count_nonzero(wrench_history) == 0
        raise RuntimeError("intentional encoder failure")

    policy.obs_encoder.forward = fail
    with pytest.raises(RuntimeError, match="intentional"):
        with WrenchEffectCapture(policy) as capture:
            capture.encode(tiny_obs(), normalized_zero=True)
    assert not policy.obs_encoder._forward_pre_hooks
    assert not policy.obs_encoder.cross_attention._forward_hooks
    policy.obs_encoder.forward = original_forward


def test_capture_requires_eval_policy_and_encoder():
    policy = TinyPolicy().train()
    with pytest.raises(ValueError, match="eval"):
        WrenchEffectCapture(policy).__enter__()


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("scope", ["image", "episode"])
@pytest.mark.parametrize("context_only", [False, True])
def test_run_exports_effect_videos_without_diffusion_sampling(monkeypatch, tmp_path, stride, scope, context_only):
    checkpoint = tmp_path / "teacher" / "checkpoints" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    policy = TinyPolicy(extra_layers=1 if context_only else 0)

    class Dataset:
        fps = 20
        phase_source_key = "observation.wrench"
        timestamp_key = "timestamp"
        episode_ranges = [range(4)]
        timestamps = np.arange(4, dtype=np.float64) * .05
        contact_threshold = 8.
        contact_force_dims = (0, 1, 2)
        contact_norm = "l2"
        contact_enter_frames = 2
        contact_exit_frames = 2
        raw = np.array([[0, 0, 0, 0, 0, 0], [9, 0, 0, 1, 2, 3],
                        [10, 0, 0, 4, 5, 6], [0, 0, 0, 7, 8, 9]], dtype=np.float32)

        def __init__(self):
            self.closed = False
            self.lerobot_dataset = SimpleNamespace(close=self.close)

        def _load_column(self, key, dtype):
            assert key == self.phase_source_key
            return self.raw.astype(dtype)

        def __getitem__(self, index):
            obs = tiny_obs()
            return {"obs": {key: value[0] for key, value in obs.items() if key != "unrelated"}}

        def close(self):
            self.closed = True

    class Writer:
        instances = []

        def __init__(self, path, width, height, fps, codec):
            self.path, self.fps = path, fps
            self.frames = []
            self.closed = False
            self.instances.append(self)
            path.write_bytes(b"video")

        def write(self, frame):
            self.frames.append(frame)

        def close(self):
            self.closed = True

    dataset = Dataset()
    monkeypatch.setattr(visual.common, "choose_checkpoint", lambda *a: checkpoint)
    monkeypatch.setattr(visual.common, "load_checkpoint_payload", lambda *a: ({}, OmegaConf.create({})))
    monkeypatch.setattr(visual.common, "resolve_device", lambda *a: torch.device("cpu"))
    monkeypatch.setattr(visual.common, "instantiate_policy", lambda *a, **kw: policy)
    monkeypatch.setattr(visual.common, "instantiate_episode_dataset", lambda *a, **kw: dataset)
    monkeypatch.setattr(visual.common, "episode_sample_indices", lambda *a: [(0, 0), (1, 1), (2, 2), (3, 3)])
    monkeypatch.setattr(visual.common, "FFmpegVideoWriter", Writer)
    # Actual rendering is checked elsewhere. Keep this test focused on the two
    # independent video streams, capture calls, file layout and raw alignment.
    monkeypatch.setattr(visual, "render_frame", lambda *a, **kw: np.zeros((64, 512, 3), dtype=np.uint8))
    args = visual.parse_args(["--checkpoint", str(checkpoint), "--episode", "0", "--stride", str(stride),
                              "--color-scope", scope] + (["--metrics", "context_effect"] if context_only else []))
    assert args.metrics == (("context_effect",) if context_only else ("ca_update", "injection_effect"))
    paths = visual.run(args)
    count = len(range(0, 4, stride))
    assert dataset.closed and policy.encode_calls == 2 * count
    assert len(Writer.instances) == len(args.metrics)
    assert all(writer.fps == 20 / stride and len(writer.frames) == count and writer.closed
               for writer in Writer.instances)
    assert all(paths[metric].is_file() for metric in args.metrics)
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["diffusion_sampling"] is False and metadata["precision"] == "float32"
    assert metadata["raw_camera_keys"] == ["side", "wrist"]
    assert metadata["color_percentiles"] == [2., 98.]
    assert metadata["color_scope"] == scope
    assert set(metadata["color_bounds"]) == set(args.metrics)
    assert metadata["cross_attn_layers"] == (2 if context_only else 1)
    if not context_only:
        assert metadata["color_bounds"]["ca_update"] != metadata["color_bounds"]["injection_effect"]
    with np.load(paths["raw"]) as data:
        metric_fields = (("context_effect",) if context_only else
                         ("ca_update", "baseline_ca_update", "injection_effect", "context_effect"))
        for metric in metric_fields:
            assert data[f"camera_0_{metric}"].shape == (count, 2, 1)
            assert data[f"camera_1_{metric}"].shape == (count, 2, 2)
        for metric in args.metrics:
            for camera in range(2):
                limits = data[f"camera_{camera}_{metric}_color_bounds"]
                assert limits.shape == (count, 2, 2)
                expected = np.stack([visual.image_color_bounds(frame, args.attention_percentiles)
                                     for frame in data[f"camera_{camera}_{metric}"]])
                np.testing.assert_allclose(limits, expected)
                if scope == "image":
                    key = metadata["raw_camera_keys"][camera]
                    np.testing.assert_allclose(metadata["color_bounds"][metric][key], limits)
        np.testing.assert_allclose(data["episode_time_sec"], np.arange(0, 4, stride) * .05)
        np.testing.assert_array_equal(data["recorded_wrench"], Dataset.raw)
        np.testing.assert_array_equal(data["wrench_contact"], [False, True, True, True])
        np.testing.assert_array_equal(data["wrench_causal_contact"], [False, False, True, True])
        np.testing.assert_array_equal(data["frame_contact"], np.array([False, True, True, True])[::stride])
        assert data["model_normalized_wrench"].shape == (count, 2, 2, 6)
        np.testing.assert_array_equal(data["model_baseline_normalized_wrench"], 0)

