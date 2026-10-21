"""Final-action visualizations must follow the real serial sampler graph."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from omegaconf import OmegaConf
from torch import nn

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.dino_token_baseline_obs_encoder import DinoTokenBaselineObsEncoder
from diffusion_policy.model.vision.force_aware_obs_encoder import ForceAwareObsEncoder
from diffusion_policy.policy.dino_token_baseline_policy import DinoTokenBaselinePolicy
from diffusion_policy.policy.force_aware_diffusion_transformer_policy import ForceAwareDiffusionTransformerPolicy
from diffusion_policy.scripts import visualize_action_image_attribution_episode as visual


def test_default_is_exact_final_action_sensitivity_and_familiar_output_paths(tmp_path):
    args = visual.parse_args(["--checkpoint", "model.ckpt", "--episode", "3"])
    assert args.gradient_probes is None
    assert args.percentiles == (1, 99)
    assert args.observation == "all"
    assert args.camera == "both"
    assert args.video_backend == "torchcodec"
    assert args.video_codec == "av1"
    assert args.autocast == "off"
    checkpoint = tmp_path / "run/checkpoints/best.ckpt"
    paths = visual.output_paths(checkpoint, args)
    assert paths["video"] == tmp_path / "run/action_image_attribution_episode_3.mp4"
    args.output = tmp_path / "custom.mp4"
    paths = visual.output_paths(checkpoint, args)
    assert paths == {"video": tmp_path / "custom.mp4", "raw": tmp_path / "custom_raw.npz",
                     "metadata": tmp_path / "custom_metadata.json"}
    approximate = visual.parse_args(["--checkpoint-dir", "checkpoints", "--episode", "0",
                                     "--gradient-probes", "2", "--attention-percentiles", "2", "98"])
    assert approximate.gradient_probes == 2
    assert approximate.percentiles == [2, 98]
    assert visual.parse_args(["--checkpoint", "a", "--episode", "0", "--autocast", "bfloat16"]).autocast == "bfloat16"


@pytest.mark.parametrize("extra", [
    ["--gradient-probes", "0"], ["--gradient-probes", "-1"],
    ["--stride", "0"], ["--max-frames", "0"], ["--alpha", "nan"],
    ["--alpha", "1.01"], ["--alpha", "-0.1"], ["--fps", "nan"],
    ["--fps", "0"], ["--percentiles", "99", "1"], ["--percentiles", "1", "inf"],
    ["--percentiles", "-1", "99"], ["--percentiles", "1", "101"],
])
def test_rejects_invalid_numeric_options(extra):
    with pytest.raises(SystemExit):
        visual.parse_args(["--checkpoint", "model.ckpt", "--episode", "0", *extra])


@pytest.mark.parametrize("argv", [
    ["--checkpoint", "model.ckpt"],
    ["--checkpoint", "model.ckpt", "--episode", "-1"],
    ["--checkpoint", "a", "--checkpoint-dir", "b", "--episode", "0"],
])
def test_requires_episode_and_one_checkpoint_source(argv):
    with pytest.raises(SystemExit):
        visual.parse_args(argv)


def test_target_is_actual_executed_postprocessed_actions_in_checkpoint_scale():
    normalizer = LinearNormalizer()
    normalizer.fit({"action": torch.stack([torch.zeros(7), torch.arange(1, 8).float()])})
    final = torch.arange(14, dtype=torch.float32).reshape(1, 2, 7).requires_grad_()
    # Internal relative/model-space actions deliberately differ in values and
    # horizon length: they must not silently become the attribution target.
    prediction = {"action": final, "action_pred": torch.full((1, 9, 7), -999.0),
                  "model_action_pred": torch.full((1, 9, 7), 999.0)}
    policy = SimpleNamespace(normalizer=normalizer, relative_pose_actions=True,
                             action_start_index=3, n_action_steps=2)
    target = visual.final_action_target(policy, prediction, {"action_reference": torch.ones(1, 7)})
    torch.testing.assert_close(target, normalizer["action"].normalize(final))
    gradient, = torch.autograd.grad(target.sum(), final)
    torch.testing.assert_close(gradient, normalizer["action"].params_dict["scale"].expand_as(final))


def test_patch_pooling_is_rms_energy_with_camera_native_rectangular_patches():
    values = np.arange(1, 49, dtype=np.float32).reshape(2, 4, 6)
    backbone = SimpleNamespace(config=SimpleNamespace(patch_size=(2, 3)))
    result = visual.pool_pixel_sensitivity(values, backbone)
    expected = np.empty((2, 2, 2), dtype=np.float32)
    for time in range(2):
        for row in range(2):
            for column in range(2):
                region = values[time, row * 2:(row + 1) * 2, column * 3:(column + 1) * 3]
                expected[time, row, column] = np.sqrt(np.mean(region.astype(np.float64) ** 2))
    np.testing.assert_allclose(result, expected)
    # Energy must be pooled before square root, not averaged after it.
    assert result[0, 0, 0] != pytest.approx(values[0, :2, :3].mean())


@pytest.mark.parametrize("values,patch", [
    (np.zeros((2, 3, 4)), 2), (np.zeros((3, 4)), 2),
    (np.full((1, 4, 4), -1), 2), (np.full((1, 4, 4), np.nan), 2),
    (np.zeros((1, 4, 4)), 0),
])
def test_patch_pooling_rejects_invalid_values_or_grids(values, patch):
    with pytest.raises(ValueError):
        visual.pool_pixel_sensitivity(values, SimpleNamespace(config=SimpleNamespace(patch_size=patch)))


def test_tiny_sensitivity_keeps_shared_colors_instead_of_per_panel_stretch():
    black = np.zeros((8, 8, 3), dtype=np.uint8)
    bounds = (0, 1e-14)
    a = visual.overlay(black, np.full((4, 4), 0.25e-14), bounds, 1)
    other = np.full((4, 4), 0.25e-14)
    other[-1] = 0.9e-14
    b = visual.overlay(black, other, bounds, 1)
    np.testing.assert_array_equal(a[0, 0], b[0, 0])
    expected = cv2.cvtColor(cv2.applyColorMap(np.array([[63]], np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)[0, 0]
    np.testing.assert_array_equal(a[0, 0], expected)
    assert not np.array_equal(a[-1, 0], b[-1, 0])
    records = [{"cameras": {"side": {"saliency": np.array([0, 1, 2]) * 1e-15},
                             "wrist": {"saliency": np.array([3, 4, 100]) * 1e-15}}},
               {"cameras": {"side": {"saliency": np.array([6, 7]) * 1e-15}}}]
    np.testing.assert_allclose(visual.shared_bounds(records, (1, 99)),
                               np.percentile(np.array([0, 1, 2, 3, 4, 100, 6, 7]) * 1e-15, [1, 99]),
                               rtol=1e-7, atol=0)


def test_output_collision_is_atomic_and_overwrite_clears_old_frames(tmp_path):
    args = visual.parse_args(["--checkpoint", "model.ckpt", "--episode", "0", "--output", str(tmp_path / "custom")])
    paths = visual.output_paths(Path("model.ckpt"), args)
    paths["video"].write_bytes(b"old video")
    paths["metadata"].write_text("old metadata")
    frame_dir = visual.rendering.frame_directory(paths["video"])
    frame_dir.mkdir()
    (frame_dir / "stale.png").write_bytes(b"stale frame")
    with pytest.raises(FileExistsError):
        visual.rendering.prepare_outputs(paths, overwrite=False, save_frames=True)
    assert paths["video"].read_bytes() == b"old video"
    assert paths["metadata"].read_text() == "old metadata"
    assert (frame_dir / "stale.png").exists()
    visual.rendering.prepare_outputs(paths, overwrite=True, save_frames=True)
    assert not paths["video"].exists() and not paths["metadata"].exists()
    assert frame_dir.is_dir() and not list(frame_dir.iterdir())


class TinyDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, patch_size=8, num_register_tokens=0)
        self.projection = nn.Conv2d(3, 8, kernel_size=8, stride=8)

    def forward(self, pixel_values):
        patches = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return SimpleNamespace(last_hidden_state=torch.cat([patches.mean(1, keepdim=True), patches], dim=1))


@pytest.fixture
def pipeline(monkeypatch, tmp_path, request):
    options = getattr(request, "param", {})
    torch.manual_seed(123)
    shapes = {"side": (3, 16, 16), "wrist": (3, 16, 24)}
    encoder_kwargs = dict(pretrained_model_name_or_path="unused", image_shapes=shapes,
                          backbone=TinyDino(), n_emb=8, n_head=2, max_obs_steps=2, dropout=0)
    baseline = options.get("baseline")
    if baseline:
        encoder = DinoTokenBaselineObsEncoder(**encoder_kwargs, mode=baseline, self_attn_layers=1)
        policy_type = DinoTokenBaselinePolicy
        policy_options = {"baseline_mode": baseline}
    else:
        encoder = ForceAwareObsEncoder(**encoder_kwargs, force_temporal_encoder="lstm",
                                      cross_attn_query="image", cross_attn_layers=1)
        policy_type, policy_options = ForceAwareDiffusionTransformerPolicy, {}
    policy = policy_type(
        shape_meta={"obs": {**{key: {"shape": shape, "type": "rgb"} for key, shape in shapes.items()},
                             "wrench_ext": {"shape": (8, 6), "type": "low_dim"}},
                    "action": {"shape": (7,)}},
        noise_scheduler=DDIMScheduler(num_train_timesteps=12, clip_sample=False),
        horizon=4, n_obs_steps=2, n_action_steps=2, action_start_index=1,
        num_inference_steps=3, obs_encoder=encoder,
        n_emb=8, n_head=2, n_layer=2, n_cond_layers=1, p_drop_attn=0,
        **policy_options,
    ).eval()
    normalizer = LinearNormalizer()
    normalizer.fit({"action": torch.randn(32, 7), "wrench_ext": torch.randn(32, 6)})
    policy.set_normalizer(normalizer)

    class Dataset:
        fps = 50
        closed = False

        def __init__(self):
            self.lerobot_dataset = SimpleNamespace(close=self.close)
            self.sample = {"obs": {**{key: torch.rand(2, *shape) for key, shape in shapes.items()},
                                   "wrench_ext": torch.randn(2, 8, 6)}}

        def __getitem__(self, index):
            torch.rand(17)  # Global RNG changes must not change the sampler/probes.
            return self.sample

        def close(self):
            self.closed = True

    dataset = Dataset()
    checkpoint = tmp_path / "run/checkpoints/model.ckpt"
    monkeypatch.setattr(visual.common, "choose_checkpoint", lambda *args: checkpoint)
    cfg = OmegaConf.create({"policy": {"_target_": f"{policy_type.__module__}.{policy_type.__name__}"}})
    monkeypatch.setattr(visual.common, "load_checkpoint_payload", lambda path: ({}, cfg))
    monkeypatch.setattr(visual.common, "instantiate_policy", lambda *args, **kwargs: policy)
    monkeypatch.setattr(visual.common, "resolve_device", lambda *args: torch.device("cpu"))
    monkeypatch.setattr(visual.common, "instantiate_episode_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(visual.common, "episode_sample_indices", lambda *args: [(100, 0), (101, 1)])

    class Writer:
        instances = []

        def __init__(self, path, width, height, fps, codec):
            self.path, self.width, self.height, self.fps = path, width, height, fps
            self.frames, self.closed = [], False
            self.instances.append(self)
            self.path.write_bytes(b"fake encoded video")

        def write(self, frame):
            assert frame.shape == (self.height, self.width, 3)
            assert frame.dtype == np.uint8 and frame.flags.c_contiguous
            self.frames.append(frame.copy())

        def close(self):
            self.closed = True

    monkeypatch.setattr(visual.common, "FFmpegVideoWriter", Writer)
    args = visual.parse_args(["--checkpoint", str(checkpoint), "--episode", "0",
                              "--gradient-probes", "2", "--save-frames"])
    return args, policy, dataset, Writer


def test_real_serial_sampler_gradients_export_all_camera_times_and_reset_noise(pipeline):
    args, policy, dataset, Writer = pipeline
    calls, backwards = [], []

    def observe(module, inputs, output):
        index = len(calls)
        calls.append(inputs[0].detach().clone())
        output.register_hook(lambda grad: backwards.append((index, float(grad.abs().sum()))))

    handle = policy.model.register_forward_hook(observe)
    flags = [parameter.requires_grad for parameter in policy.parameters()]
    try:
        paths = visual.run(args)
    finally:
        handle.remove()
    assert dataset.closed and len(Writer.instances) == 1
    assert Writer.instances[0].closed and len(Writer.instances[0].frames) == 2
    assert Writer.instances[0].fps == 50
    assert len(calls) == 6  # Two images, each using the real three-step DDIM chain.
    assert {index for index, energy in backwards if energy > 0} == set(range(6))
    assert all(sum(index == call for index, _ in backwards) == 2 for call in range(6))
    torch.testing.assert_close(calls[0], calls[3])
    assert [parameter.requires_grad for parameter in policy.parameters()] == flags
    assert policy.obs_encoder.freeze_backbone
    assert not policy.model._forward_pre_hooks
    assert not any(parameter.grad is not None for parameter in policy.parameters())
    assert len(list(visual.rendering.frame_directory(paths["video"]).glob("*.png"))) == 2
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["target_shape"] == [2, 7]
    assert metadata["gradient_probes"] == 2 and metadata["estimator"] == "hutchinson"
    assert metadata["all_camera_keys"] == ["side", "wrist"]
    assert metadata["camera_patch_grids"] == {"side": [2, 2], "wrist": [2, 3]}
    assert metadata["observation_steps"] == 2
    assert [row["num_model_calls"] for row in metadata["sampling"]] == [3, 3]
    assert all(row["target_count"] == 14 and row["num_vjps"] == 2 for row in metadata["sampling"])
    with np.load(paths["raw"]) as raw:
        np.testing.assert_array_equal(raw["anchors"], [100, 101])
        assert raw["action"].shape == raw["scaled_action"].shape == (2, 2, 7)
        values = []
        for index, grid in enumerate(((2, 2), (2, 3))):
            maps = raw[f"camera_{index}_sensitivity"]
            assert maps.shape == (2, 2, *grid)
            assert np.isfinite(maps).all() and (maps >= 0).all() and (maps > 0).any()
            np.testing.assert_array_equal(maps[0], maps[1])
            values.append(maps.reshape(-1))
        np.testing.assert_array_equal(raw["action"][0], raw["action"][1])
        np.testing.assert_allclose(metadata["bounds"], np.percentile(np.concatenate(values), [1, 99]))
        scaled = policy.normalizer["action"].normalize(torch.from_numpy(raw["action"]))
        np.testing.assert_allclose(raw["scaled_action"], scaled.numpy())


@pytest.mark.parametrize("pipeline", [{"baseline": "dino"}, {"baseline": "image_self_attn"}], indirect=True)
def test_image_only_checkpoints_use_same_full_chain_target_and_camera_outputs(pipeline):
    args, policy, dataset, Writer = pipeline
    args.max_frames = 1
    args.observation = "last"
    args.camera = "wrist"
    paths = visual.run(args)
    assert dataset.closed and Writer.instances[0].closed
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["baseline_mode"] == policy.baseline_mode
    assert metadata["sampling"][0]["connected_cameras"] == {"side": True, "wrist": True}
    # Display restriction must retain all camera and observation-time raw maps.
    assert metadata["display_camera_keys"] == ["wrist"]
    with np.load(paths["raw"]) as raw:
        assert raw["camera_0_sensitivity"].shape == (1, 2, 2, 2)
        assert raw["camera_1_sensitivity"].shape == (1, 2, 2, 3)


def test_prediction_failure_closes_dataset_and_restores_gradient_hooks_and_flags(pipeline, monkeypatch):
    args, policy, dataset, Writer = pipeline
    flags = [parameter.requires_grad for parameter in policy.parameters()]

    def fail(*args, **kwargs):
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(policy, "predict_action", fail)
    with pytest.raises(RuntimeError, match="sampling failed"):
        visual.run(args)
    assert dataset.closed and not Writer.instances
    assert not policy.model._forward_pre_hooks
    assert policy.obs_encoder.freeze_backbone
    assert [parameter.requires_grad for parameter in policy.parameters()] == flags
    paths = visual.output_paths(args.checkpoint, args)
    assert not paths["metadata"].exists() and not paths["raw"].exists()


def test_encoder_write_failure_closes_writer_and_dataset(pipeline, monkeypatch):
    args, policy, dataset, Writer = pipeline

    def fail(*args):
        raise RuntimeError("encoding failed")

    monkeypatch.setattr(Writer, "write", fail)
    with pytest.raises(RuntimeError, match="encoding failed"):
        visual.run(args)
    assert dataset.closed and all(writer.closed for writer in Writer.instances)
    assert not visual.output_paths(args.checkpoint, args)["metadata"].exists()
