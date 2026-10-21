import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.force_aware_obs_encoder import ForceAwareObsEncoder
from diffusion_policy.model.vision.dino_token_baseline_obs_encoder import DinoTokenBaselineObsEncoder
from diffusion_policy.policy.dino_token_baseline_policy import DinoTokenBaselinePolicy
from diffusion_policy.policy.force_aware_diffusion_transformer_policy import ForceAwareDiffusionTransformerPolicy
from diffusion_policy.scripts import visualize_image_query_cross_attention_episode as visual


def record(anchor=0, value=0.125):
    return {"anchor": anchor, "cameras": {
        key: {"image": np.zeros((16, 24, 3), dtype=np.uint8), "grid": (2, 3),
              "attention": np.full((8, 6), value, dtype=np.float32),
              "update": np.arange(6, dtype=np.float32) + 1}
        for key in ("side", "wrist")
    }}


def test_cli_defaults_and_two_output_prefixes(tmp_path):
    args = visual.parse_args(["--checkpoint", "model.ckpt", "--episode", "3"])
    assert args.attention_normalization == "percentile"
    assert args.attention_percentiles == (1, 99)
    assert args.mode == "auto"
    assert args.layer_index == -1
    assert args.video_backend == "torchcodec"
    assert args.video_codec == "av1"
    assert args.camera == "both"
    args.output = tmp_path / "custom.mp4"
    paths = visual.output_paths(Path("runs/checkpoints/best.ckpt"), args)
    assert paths["attention"] == tmp_path / "custom_attention.mp4"
    assert paths["update"] == tmp_path / "custom_ca_update.mp4"


@pytest.mark.parametrize("extra", [
    ["--attention-normalization", "frame"], ["--attention-percentiles", "99", "1"],
    ["--attention-percentiles", "-1", "99"], ["--attention-percentiles", "1", "nan"],
    ["--stride", "0"], ["--max-frames", "0"], ["--alpha", "nan"], ["--fps", "-1"],
])
def test_cli_rejects_ambiguous_normalization_and_invalid_ranges(extra):
    with pytest.raises(SystemExit):
        visual.parse_args(["--checkpoint", "model.ckpt", "--episode", "3", *extra])


def test_attention_color_is_absolute_across_frames_cameras_and_keys():
    first, second = record(), record(anchor=1, value=0.8)
    # Identical weights at a chosen key must have identical colors even when
    # the rest of its frame is very different. No per-panel min/max is allowed.
    second["cameras"]["wrist"]["attention"][3] = 0.125
    a = visual._camera_row(first, ["wrist"], alpha=1.0, token_index=3, bounds=(0, 1))
    b = visual._camera_row(second, ["wrist"], alpha=1.0, token_index=3, bounds=(0, 1))
    np.testing.assert_array_equal(a, b)
    expected = cv2.cvtColor(cv2.applyColorMap(np.array([[31]], dtype=np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)[0, 0]
    np.testing.assert_array_equal(a[24 + 8, 12], expected)


def test_attention_grid_has_all_eight_panels_and_fixed_colorbar(monkeypatch):
    seen = []
    original = visual.common.overlay_attention

    def remember(image, attention, grid, alpha, normalization_bounds=None):
        seen.append((np.asarray(attention).copy(), normalization_bounds))
        return original(image, attention, grid, alpha, normalization_bounds)

    monkeypatch.setattr(visual.common, "overlay_attention", remember)
    frame = record()
    for key in frame["cameras"]:
        for token in range(8):
            frame["cameras"][key]["attention"][token] = token / 8
    result = visual.render_attention(frame, ["wrist", "side"], [f"k={i}" for i in range(8)],
                                     episode=3, alpha=1, layer_index=0)
    assert len(seen) == 16
    assert all(bounds == (0, 1) for _, bounds in seen)
    assert [float(value[0]) for value, _ in seen] == [i / 8 for i in range(8) for _ in range(2)]
    assert result.dtype == np.uint8 and result.flags.c_contiguous
    assert result.shape[0] % 2 == result.shape[1] % 2 == 0
    assert result.shape[1] > 4 * 2 * 24  # Four tiles, each retaining both cameras.


def test_update_percentiles_share_all_cameras_and_frames():
    records = [record(), record(1)]
    records[1]["cameras"]["side"]["update"] *= 50
    all_values = np.concatenate([camera["update"] for row in records for camera in row["cameras"].values()])
    np.testing.assert_allclose(visual.update_bounds(records), np.percentile(all_values, [1, 99]))
    constant = record()
    for camera in constant["cameras"].values():
        camera["update"].fill(4)
    assert visual.update_bounds([constant]) == (0, 4)


def test_attention_percentiles_pool_every_frame_camera_and_wrench_position():
    records = [record(value=0.0001), record(1, value=0.001)]
    records[1]["cameras"]["side"]["attention"][0] = 0.8
    all_values = np.concatenate([camera["attention"].reshape(-1)
                                for row in records for camera in row["cameras"].values()])
    np.testing.assert_allclose(visual.percentile_bounds(records, "attention"),
                               np.percentile(all_values, [1, 99]))
    # Stretch affects display only, with one bound pair for every panel.
    original = records[0]["cameras"]["side"]["attention"].copy()
    visual.render_attention(records[0], ["side", "wrist"], [str(i) for i in range(8)],
                            episode=0, alpha=1, layer_index=0,
                            bounds=(0.0001, 0.001), percentile_range=(1, 99))
    np.testing.assert_array_equal(records[0]["cameras"]["side"]["attention"], original)


def test_mode_aliases_layer_alias_and_inspection_without_episode():
    base = ["--checkpoint", "model.ckpt"]
    args = visual.parse_args([*base, "--list-layers"])
    assert args.list_layers and args.episode is None
    assert visual.parse_args([*base, "--episode", "0", "--sa", "--layer", "0"]).mode == "self-attn"
    assert visual.parse_args([*base, "--episode", "0", "--ca"]).mode == "cross-attn"
    assert visual.parse_args([*base, "--episode", "0", "--attention-mode", "dino"]).mode == "dino"
    with pytest.raises(SystemExit):
        visual.parse_args([*base, "--episode", "0", "--sa", "--ca"])
    with pytest.raises(SystemExit):
        visual.parse_args(base)


def test_output_collision_is_checked_before_deleting_any_file(tmp_path):
    args = visual.parse_args(["--checkpoint", "model.ckpt", "--episode", "3", "--output", str(tmp_path / "test")])
    paths = visual.output_paths(Path("model.ckpt"), args)
    paths["attention"].write_bytes(b"old video")
    paths["metadata"].write_text("old metadata")
    with pytest.raises(FileExistsError):
        visual.prepare_outputs(paths, overwrite=False, save_frames=True)
    assert paths["attention"].read_bytes() == b"old video"
    assert paths["metadata"].read_text() == "old metadata"
    visual.prepare_outputs(paths, overwrite=True, save_frames=True)
    assert not paths["attention"].exists()
    assert visual.frame_directory(paths["attention"]).is_dir()
    assert visual.frame_directory(paths["update"]).is_dir()


def test_wrench_state_labels_use_checkpoint_offsets_without_guessing():
    cfg = OmegaConf.create({"task": {"dataset": {"windowed_lowdim_feature_groups": {
        "future_wrench": {"offsets_sec": [0.04 * (i + 1) for i in range(8)]}
    }}}})
    policy = SimpleNamespace(wrench_key="future_wrench", obs_encoder=SimpleNamespace(wrench_history_steps=8))
    labels, offsets = visual.wrench_token_labels(cfg, policy)
    assert labels[0] == "Wrench state 1 (+40 ms)"
    assert labels[-1] == "Wrench state 8 (+320 ms)"
    assert len(offsets) == 8
    labels, offsets = visual.wrench_token_labels(OmegaConf.create({}), policy)
    assert labels[0] == "Wrench state 1" and offsets is None


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
    shapes = {"side": (3, 16, 16), "wrist": (3, 16, 24)}
    baseline_mode = options.get("baseline_mode")
    encoder_kwargs = dict(
        pretrained_model_name_or_path="unused", image_shapes=shapes, backbone=TinyDino(),
        n_emb=8, n_head=2, max_obs_steps=2, force_temporal_encoder="lstm", dropout=0,
    )
    if baseline_mode:
        if baseline_mode == "dino":
            from transformers import DINOv3ViTConfig, DINOv3ViTModel
            encoder_kwargs["backbone"] = DINOv3ViTModel(DINOv3ViTConfig(
                image_size=24, patch_size=8, hidden_size=16, intermediate_size=32,
                num_hidden_layers=2, num_attention_heads=4, num_register_tokens=2,
            ))
        encoder = DinoTokenBaselineObsEncoder(**encoder_kwargs, mode=baseline_mode,
                                              self_attn_layers=2, force_encoder_layers=1)
        policy_type = DinoTokenBaselinePolicy
        policy_options = {"baseline_mode": baseline_mode}
    else:
        encoder = ForceAwareObsEncoder(
            **encoder_kwargs, cross_attn_query=options.get("query", "image"),
            pre_cross_self_attn_layers=options.get("self_layers", 0),
            cross_attn_layers=options.get("cross_layers", 1),
        )
        policy_type = ForceAwareDiffusionTransformerPolicy
        policy_options = {}
    policy = policy_type(
        shape_meta={"obs": {
            **{key: {"shape": shape, "type": "rgb"} for key, shape in shapes.items()},
            "wrench_ext": {"shape": (8, 6), "type": "low_dim"},
        }, "action": {"shape": (7,)}},
        noise_scheduler=DDIMScheduler(num_train_timesteps=10), horizon=9, n_obs_steps=2,
        n_action_steps=8, num_inference_steps=2, obs_encoder=encoder,
        n_emb=8, n_head=2, n_layer=1, p_drop_attn=0,
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
            return self.sample

        def close(self):
            self.closed = True

    dataset = Dataset()
    checkpoint = tmp_path / "run/checkpoints/model.ckpt"
    monkeypatch.setattr(visual.common, "choose_checkpoint", lambda *a: checkpoint)
    cfg = OmegaConf.create({
        "policy": {"_target_": f"{policy_type.__module__}.{policy_type.__name__}"},
        "task": {"dataset": {"windowed_lowdim_feature_groups": {
            "wrench_ext": {"offsets_sec": [0.04 * (index + 1) for index in range(8)]}
        }}},
    })
    monkeypatch.setattr(visual.common, "load_checkpoint_payload", lambda path: ({}, cfg))
    monkeypatch.setattr(visual.common, "instantiate_policy", lambda *a, **kw: policy)
    monkeypatch.setattr(visual.common, "resolve_device", lambda *a: torch.device("cpu"))
    monkeypatch.setattr(visual.common, "instantiate_episode_dataset", lambda *a, **kw: dataset)
    monkeypatch.setattr(visual.common, "episode_sample_indices", lambda *a: [(100, 0), (101, 1)])

    class Writer:
        instances = []

        def __init__(self, output_path, width, height, fps, codec):
            self.path, self.width, self.height, self.fps, self.codec = output_path, width, height, fps, codec
            self.frames = []
            self.closed = False
            self.instances.append(self)
            self.path.write_bytes(b"test encoder")

        def write(self, frame):
            assert frame.shape == (self.height, self.width, 3)
            self.frames.append(frame.copy())

        def close(self):
            self.closed = True

    monkeypatch.setattr(visual.common, "FFmpegVideoWriter", Writer)
    args = visual.parse_args(["--checkpoint", str(checkpoint), "--episode", "0", "--save-frames"])
    args.mode = options.get("mode", "auto")
    args.layer_index = options.get("layer", -1)
    args.attention_query = options.get("attention_query", "all")
    return args, policy, dataset, Writer


def test_complete_replay_exports_both_videos_frames_and_raw_values(pipeline):
    args, policy, dataset, Writer = pipeline
    paths = visual.run(args)
    assert dataset.closed
    assert len(Writer.instances) == 2
    assert all(len(writer.frames) == 2 and writer.closed for writer in Writer.instances)
    assert all(writer.fps == 50 for writer in Writer.instances)
    assert Writer.instances[0].width > Writer.instances[1].width * 3
    assert all(len(list(visual.frame_directory(paths[key]).glob("*.png"))) == 2 for key in ("attention", "update"))
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["attention_normalization"] == "percentile"
    assert metadata["attention_percentiles"] == [1, 99]
    assert metadata["selected_mode"] == "cross-attn" and metadata["query_source"] == "image"
    assert metadata["camera_patch_grids"] == {"side": [2, 2], "wrist": [2, 3]}
    with np.load(paths["raw"]) as data:
        np.testing.assert_array_equal(data["anchors"], [100, 101])
        assert data["camera_0_attention"].shape == (2, 8, 4)
        assert data["camera_1_attention"].shape == (2, 8, 6)
        np.testing.assert_allclose(data["camera_1_attention"].sum(axis=1), 1, atol=1e-6)
        assert data["camera_1_update_l2"].shape == (2, 6)
        assert np.isfinite(data["camera_1_update_l2"]).all()
        probabilities = np.concatenate([data[f"camera_{index}_attention"].reshape(-1) for index in (0, 1)])
        np.testing.assert_allclose(metadata["attention_bounds"], np.percentile(probabilities, [1, 99]))
    assert not policy.obs_encoder._forward_pre_hooks
    assert not policy.obs_encoder.cross_attention._forward_hooks


def test_inference_failure_closes_dataset_and_removes_capture_hooks(pipeline, monkeypatch):
    args, policy, dataset, Writer = pipeline

    def fail(*args, **kwargs):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(policy, "predict_action", fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        visual.run(args)
    assert dataset.closed and not Writer.instances
    assert not policy.obs_encoder._forward_pre_hooks
    assert not policy.obs_encoder.cross_attention._forward_hooks


def test_both_video_writers_close_even_when_an_encoder_fails(pipeline, monkeypatch):
    args, policy, dataset, Writer = pipeline
    original = Writer.close

    def close(writer):
        original(writer)
        if writer.path.name.endswith("_attention.mp4"):
            raise RuntimeError("encoder failed")

    monkeypatch.setattr(Writer, "close", close)
    with pytest.raises(RuntimeError, match="encoder failed"):
        visual.run(args)
    assert dataset.closed
    assert all(writer.closed for writer in Writer.instances)
    paths = visual.output_paths(args.checkpoint, args, visual.AttentionCapture(policy))
    assert not paths["metadata"].exists()


@pytest.mark.parametrize("pipeline,mode,query,layers", [
    ({"query": "wrench", "cross_layers": 2, "self_layers": 1, "layer": 0}, "cross-attn", "wrench", 2),
    ({"query": "image", "self_layers": 2, "mode": "self-attn", "layer": 0}, "self-attn", "wrench", 2),
    ({"cross_layers": 0, "self_layers": 1}, "self-attn", "wrench", 1),
    ({"baseline_mode": "self_attn", "attention_query": "last"}, "self-attn", "wrench", 2),
    ({"baseline_mode": "image_self_attn"}, "self-attn", "image", 2),
    ({"baseline_mode": "dino", "mode": "dino"}, "dino", "image", 2),
], indirect=["pipeline"])
def test_generic_checkpoint_replay_selects_structure_and_exports_only_spatial_metrics(pipeline, mode, query, layers):
    args, policy, dataset, Writer = pipeline
    paths = visual.run(args)
    assert dataset.closed
    assert len(Writer.instances) == 1 and Writer.instances[0].closed
    assert "update" not in paths
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["selected_mode"] == mode
    assert metadata["query_source"] == query
    assert metadata["architecture"]["layer_count"] == layers
    assert metadata["layer_index"] == args.layer_index % layers
    assert metadata["update_bounds"] is None
    assert metadata["checkpoint_policy_target"].endswith(type(policy).__name__)
    if query == "wrench":
        if args.attention_query == "last":
            assert metadata["displayed_wrench_indices"] == [7]
            assert metadata["wrench_offsets_sec"] == [0.32]
            assert len(metadata["all_wrench_offsets_sec"]) == 8
        else:
            assert metadata["displayed_wrench_indices"] == list(range(8))
        if metadata["temporal_layout"] == "per_observation":
            assert "earlier-time" not in metadata["attention_definition"].lower()
    else:
        assert metadata["wrench_state_labels"] is None
        assert metadata["displayed_wrench_indices"] is None
    with np.load(paths["raw"]) as data:
        panel_count = 8 if query == "wrench" and args.attention_query == "all" else 1
        assert data["camera_0_attention"].shape == (2, panel_count, 4)
        assert data["camera_1_attention"].shape == (2, panel_count, 6)
        assert "camera_0_update_l2" not in data


def test_fixed_scale_remains_available_explicitly(pipeline):
    args, _, _, _ = pipeline
    args.attention_normalization = "fixed"
    paths = visual.run(args)
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["attention_bounds"] == [0, 1]
    assert metadata["attention_percentiles"] is None


def test_list_layers_uses_cpu_and_does_not_open_dataset_or_write_files(pipeline, monkeypatch):
    args, _, dataset, Writer = pipeline
    args.list_layers, args.episode = True, None

    def fail(*args, **kwargs):
        raise AssertionError("Inspection must not open a dataset or run inference")

    monkeypatch.setattr(visual.common, "instantiate_episode_dataset", fail)
    monkeypatch.setattr(visual.common, "resolve_device", fail)
    result = visual.run(args)
    assert result["selected_mode"] == "cross-attn"
    assert result["available_modes"]["cross-attn"]["indices"] == [0]
    assert not dataset.closed and not Writer.instances
