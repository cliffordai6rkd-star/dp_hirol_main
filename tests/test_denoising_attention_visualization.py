from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from diffusion_policy.scripts import visualize_denoising_attention_episode as visual
from diffusion_policy.scripts.episode_wrench_plot import WrenchEpisodePlot, load_episode_wrench


@pytest.mark.parametrize("camera_keys, expected_visual", [
    (None, [50, 75]), (["side"], [25, 50]), (["wrist"], [25, 25]),
])
def test_percentages_use_one_step_condition_mass_and_conserve_routes(camera_keys, expected_visual):
    class Trace:
        def project(self, context):
            assert context.shape == (2, 3)
            return {"side": torch.tensor([[[1.0, 1.0]], [[2.0, 0.0]]]),
                    "wrist": torch.tensor([[[2.0]], [[1.0]]])}, {
                "input_mass": torch.tensor([8.0, 4.0]),
                "visual_mass": torch.tensor([4.0, 3.0]),
                "wrench_mass": torch.tensor([4.0, 1.0]),
            }

    maps, image, wrench, mass = visual.routing_percentages(torch.ones(2, 3), Trace(), camera_keys)
    np.testing.assert_allclose(image, expected_visual)
    np.testing.assert_allclose(wrench, [50, 25])
    np.testing.assert_allclose(mass, [8, 4])
    assert set(maps) == set(camera_keys or ("side", "wrist"))
    if "side" in maps:
        np.testing.assert_allclose(maps["side"][0, 0], [12.5, 12.5])
    if "wrist" in maps:
        np.testing.assert_allclose(maps["wrist"][1, 0], [25])


@pytest.mark.parametrize("camera", ["both", "wrist", "side"])
def test_video_layout_marks_cross_or_self_and_moves_step_marker(tmp_path, camera):
    checkpoint = tmp_path / "run" / "checkpoints" / "best.ckpt"
    paths = visual.output_paths(checkpoint, 3, "cross+self")
    assert paths["video"] == tmp_path / "run" / "attention_visualize" / "denoising_cross_self_episode_3.mp4"
    assert visual.output_paths(checkpoint, 3, "cross", normalization="image_relative")["video"].stem.endswith("_image_relative")
    if camera != "both":
        assert visual.output_paths(checkpoint, 3, "cross+self", camera)["video"].stem.endswith(f"_{camera}")
    assert visual.fusion_label(SimpleNamespace(cross_attention=[1], self_attention=[])) == "cross"
    assert visual.fusion_label(SimpleNamespace(cross_attention=[], self_attention=[1])) == "self"
    record = {"anchor": 14, "timesteps": np.array([9, 4, 0]),
              "visual": np.array([25., 50., 75.]), "wrench": np.array([75., 50., 25.]),
              "cameras": {key: {"images": np.zeros((1, 16, 128, 3), dtype=np.uint8),
                                "percent": np.array([[[0., 2.]], [[2., 0.]], [[1., 1.]]]),
                                "grid": (1, 2)} for key in ("side", "wrist")}}
    args = SimpleNamespace(episode=3, observation="last", alpha=0.55, camera=camera,
                           attention_percentiles=(1.0, 99.0))
    times = np.array([0.0, 0.5, 1.0])
    values = np.array([[0., 1., 2., 0.1, 0.2, 0.3],
                       [9., 2., 3., 0.2, 0.3, 0.4],
                       [0., 3., 4., 0.3, 0.4, 0.5]], dtype=np.float32)
    dataset = SimpleNamespace(
        phase_source_key="observation.wrench", episode_ranges=[range(3)],
        timestamps=times, timestamp_key="timestamp",
        _timestamp_seconds=lambda values, **kw: np.asarray(values),
        _load_column=lambda key, dtype: values.astype(dtype),
        contact_threshold=8., contact_force_dims=(0, 1, 2), contact_norm="l2",
        contact_enter_frames=1, contact_exit_frames=1,
    )
    timeline = load_episode_wrench(dataset, SimpleNamespace(), 0, [0, 1, 2], 2)
    keys = ["wrist", "side"] if camera == "both" else [camera]
    chart = WrenchEpisodePlot(timeline, 256 if camera == "both" else 512)
    first = visual.render_frame(record, 0, keys, (0.0, 2.0), args, "cross+self",
                                timeline, chart)
    last = visual.render_frame(record, 2, keys, (0.0, 2.0), args, "cross+self",
                               timeline, chart)
    assert first.shape == last.shape and first.shape[1] == (256 if camera == "both" else 512)
    assert first.shape[0] % 2 == 0 and first.flags.c_contiguous
    assert not np.array_equal(first, last)
    bright = visual.heat_overlay(np.zeros((16, 32, 3), dtype=np.uint8), [0, 2], (1, 2), (0.0, 2.0), 1)
    assert bright[8, 24, 0] > bright[8, 8, 0]
    # The lower chart is genuinely part of the same RGB frame.
    assert np.any(first[-420:] > 200)


def test_color_scale_uses_episode_percentiles_after_denoising_mean():
    records = []
    for frame in range(2):
        cameras = {}
        for index, key in enumerate(("side", "wrist")):
            mean = np.arange(1000 + frame * 40 + index * 20,
                             1020 + frame * 40 + index * 20).reshape(2, 10)
            cameras[key] = {"percent": np.stack((mean - 100, mean + 100))}
        records.append({"cameras": cameras})
    np.testing.assert_allclose(visual.heatmap_color_bounds(records), [1000.79, 1078.21])
    np.testing.assert_allclose(visual.heatmap_color_bounds(records, (10, 90)), [1007.9, 1071.1])
    # Values outside the shared range saturate to the same endpoint colors.
    image = np.zeros((1, 4, 3), dtype=np.uint8)
    overlay = visual.heat_overlay(image, [0, 1, 9, 10], (1, 4), (1, 9), 1)
    np.testing.assert_array_equal(overlay[0, 0], overlay[0, 1])
    np.testing.assert_array_equal(overlay[0, 2], overlay[0, 3])


def test_episode_max_uses_denoising_mean_and_shares_all_frames_cameras_and_observations():
    records = [
        {"cameras": {
            "side": {"percent": np.array([[[2., 20.], [6., 4.]], [[2., 0.], [6., 4.]]])},
            "wrist": {"percent": np.full((2, 2, 2), 3.)},
        }},
        {"cameras": {
            "side": {"percent": np.full((2, 2, 2), 4.)},
            "wrist": {"percent": np.array([[[8., 8.], [12., 3.]], [[8., 8.], [12., 3.]]])},
        }},
    ]
    # The instantaneous 20 in side becomes 10 after denoising averaging;
    # the episode maximum is 12 in another frame/camera/observation slot.
    bounds = visual.episode_max_color_bounds(records)
    assert bounds == (0., 12.)
    first_frame_bounds = visual.episode_max_color_bounds(records[:1])
    assert first_frame_bounds == (0., 10.)
    side_only = [{"cameras": {"side": record["cameras"]["side"]}} for record in records]
    assert visual.episode_max_color_bounds(side_only) == (0., 10.)
    # A positive minimum is not subtracted: 3 remains 25% of the max.
    assert records[0]["cameras"]["wrist"]["percent"].mean(axis=0)[0, 0] / bounds[1] == .25


def test_episode_max_rejects_an_empty_attention_signal():
    with pytest.raises(ValueError, match="positive image attention"):
        visual.episode_max_color_bounds([{"cameras": {"side": {"percent": np.zeros((2, 1, 2))}}}])


def test_image_relative_normalization_removes_absolute_visual_route_scale():
    records = [{
        "visual": np.array([50.0]), "wrench": np.array([50.0]),
        "cameras": {
            "side": {"percent": np.array([[[2.0, 2.0]]]), "images": np.zeros((1, 2, 2, 3), dtype=np.uint8), "grid": (1, 2)},
            "wrist": {"percent": np.array([[[1.0, 1.0]]]), "images": np.zeros((1, 2, 2, 3), dtype=np.uint8), "grid": (1, 2)},
        },
    }]
    normalized = visual.image_relative_records(records)
    np.testing.assert_allclose(normalized[0]["cameras"]["side"]["percent"], [[[100 / 3, 100 / 3]]])
    np.testing.assert_allclose(normalized[0]["cameras"]["wrist"]["percent"], [[[100 / 6, 100 / 6]]])
    np.testing.assert_allclose(sum(camera["percent"].sum() for camera in normalized[0]["cameras"].values()), 100.0)
    # The visual route curve is display-independent and is explicitly reset.
    np.testing.assert_allclose(normalized[0]["visual"], [100.0])


def test_query_slot_routing_supports_residual_free_image_query_cross_attention():
    trace = SimpleNamespace(
        cross_attention=[object()], cross_query="image", cross_query_residual=False,
        self_attention=[], context_tokens=6, observation_steps=1, image_token_count=6,
        image_keys=("side", "wrist"), patch_counts={"side": 2, "wrist": 4},
    )
    context = torch.tensor([[1.0, 2.0, 1.0, 0.0, 2.0, 4.0]])
    maps, visual_percent, wrench, total = visual.query_slot_routing_percentages(
        context, trace, ("side", "wrist")
    )
    np.testing.assert_allclose(total, [10.0])
    np.testing.assert_allclose(maps["side"], [[[10.0, 20.0]]])
    np.testing.assert_allclose(maps["wrist"], [[[10.0, 0.0, 20.0, 40.0]]])
    np.testing.assert_allclose(visual_percent, [100.0])
    np.testing.assert_allclose(wrench, [0.0])


def test_concentration_metrics_report_entropy_and_top_k_mass():
    records = [{
        "cameras": {
            "side": {"percent": np.array([[[9.0, 1.0, 0.0, 0.0]]])},
            "wrist": {"percent": np.array([[[1.0, 1.0, 1.0, 1.0]]])},
        },
    }]
    metrics = visual.concentration_metrics(records, ("side", "wrist"), top_k_fraction=0.25)
    assert metrics["global"]["entropy"].shape == (1,)
    assert metrics["global"]["top_k_mass_percent"].shape == (1,)
    assert metrics["cameras"]["side"]["entropy"][0] < metrics["cameras"]["wrist"]["entropy"][0]
    assert metrics["cameras"]["side"]["top_k_mass_percent"][0] > metrics["cameras"]["wrist"]["top_k_mass_percent"][0]


@pytest.mark.parametrize("percentiles", [(2, 98), (10, 90), (0, 100)])
def test_display_percentiles_are_configurable(percentiles):
    args = visual.parse_args(["--checkpoint", "model.ckpt", "--episode", "0",
                              "--attention-percentiles", *map(str, percentiles)])
    assert args.attention_percentiles == list(percentiles)
    assert args.color_scale == "percentile"


def test_default_color_scale_is_episode_max_and_explicit_percentile_retains_old_default():
    base = ["--checkpoint", "model.ckpt", "--episode", "0"]
    assert visual.parse_args(base).color_scale == "episode_max"
    percentile = visual.parse_args(base + ["--color-scale", "percentile"])
    assert percentile.color_scale == "percentile"
    assert percentile.attention_percentiles == (1., 99.)
    assert visual.parse_args(base + ["--heatmap-normalization", "image_relative"]).color_scale == "episode_max"


def test_relative_heatmap_and_fixed_bounds_are_configurable():
    args = visual.parse_args(["--checkpoint", "model.ckpt", "--episode", "0",
                              "--heatmap-normalization", "image_relative",
                              "--color-bounds", "0.01", "1.0",
                              "--concentration-top-k", "0.2"])
    assert args.heatmap_normalization == "image_relative"
    assert args.color_bounds == [0.01, 1.0]
    assert args.concentration_top_k == 0.2


@pytest.mark.parametrize("argv", [
    ["--color-bounds", "1", "1"], ["--color-bounds", "-1", "1"],
    ["--concentration-top-k", "0"], ["--concentration-top-k", "1.1"],
    ["--color-scale", "episode_max", "--attention-percentiles", "1", "99"],
])
def test_invalid_relative_display_options_are_rejected(argv):
    with pytest.raises(SystemExit):
        visual.parse_args(["--checkpoint", "model.ckpt", "--episode", "0", *argv])


@pytest.mark.parametrize("percentiles", [(-1, 99), (2, 101), (90, 10), (50, 50), ("nan", 99)])
def test_invalid_display_percentiles_are_rejected(percentiles):
    with pytest.raises(SystemExit):
        visual.parse_args(["--checkpoint", "model.ckpt", "--episode", "0",
                           "--attention-percentiles", *map(str, percentiles)])


@pytest.mark.parametrize("bad", [0, -1])
def test_zero_or_negative_route_mass_is_rejected(bad):
    trace = SimpleNamespace(project=lambda _: ({}, {"input_mass": torch.tensor([float(bad)])}))
    with pytest.raises(ValueError, match="positive finite mass"):
        visual.routing_percentages(torch.ones(1, 1), trace)


@pytest.mark.parametrize("scale_mode, extra_args", [
    ("episode_max", []),
    ("percentile", ["--color-scale", "percentile"]),
    ("percentile", ["--attention-percentiles", "2", "98"]),
    ("percentile", ["--attention-percentiles", "10", "90"]),
    ("fixed", ["--color-bounds", "1", "20"]),
    ("image_relative_episode_max", ["--heatmap-normalization", "image_relative"]),
])
@pytest.mark.parametrize("camera", ["both", "wrist", "side"])
def test_episode_video_uses_one_frame_per_observation_at_dataset_fps(monkeypatch, tmp_path, scale_mode, extra_args, camera):
    checkpoint = tmp_path / "run" / "checkpoints" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    class Capture:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Fusion(Capture):
        image_keys = ("side", "wrist")
        patch_counts = {"side": 2, "wrist": 2}
        cross_attention = [1]
        self_attention = []
        architecture = {"cross_attn_layers": 1}

        def project(self, context):
            assert context.shape == (3, 4)
            maps = {"side": torch.ones(3, 1, 2),
                    "wrist": torch.tensor([[[0.5, 1.5]]]).expand(3, 1, 2)}
            return maps, {"input_mass": torch.full((3,), 8.0),
                          "visual_mass": torch.full((3,), 4.0),
                          "wrench_mass": torch.full((3,), 4.0)}

    class Dit(Capture):
        def step_context_relevance(self):
            return torch.ones(3, 4)

        def diagnostics(self):
            return {"timesteps": [9, 4, 0]}

    class Dataset:
        fps = 50
        closed = False
        phase_source_key = "observation.wrench"
        timestamp_key = "timestamp"
        contact_threshold = 8.
        contact_force_dims = (0, 1, 2)
        contact_norm = "l2"
        contact_enter_frames = 2
        contact_exit_frames = 2

        def __init__(self):
            self.lerobot_dataset = SimpleNamespace(close=self.close)
            self.episode_ranges = [range(100, 104)]
            self.timestamps = np.arange(104, dtype=np.float64) / self.fps
            self.raw_wrench = np.zeros((104, 6), dtype=np.float32)
            self.raw_wrench[100:104] = [[1, 2, 3, .1, .2, .3],
                                       [10, 0, 0, .4, .5, .6],
                                       [11, 0, 0, .7, .8, .9],
                                       [0, 0, 0, 1., 1.1, 1.2]]

        def _load_column(self, key, dtype):
            assert key == "observation.wrench"
            return self.raw_wrench.astype(dtype)

        @staticmethod
        def _timestamp_seconds(values, **kwargs):
            return np.asarray(values, dtype=np.float64)

        def __getitem__(self, index):
            image = torch.full((1, 3, 8, 128), index / 2)
            return {"obs": {"side": image, "wrist": image}}

        def close(self):
            self.closed = True

    class Writer:
        instances = []

        def __init__(self, path, width, height, fps, codec):
            self.frames = []
            self.fps = fps
            self.path = path
            self.instances.append(self)
            path.write_bytes(b"video")

        def write(self, frame):
            self.frames.append(frame.copy())

        def close(self):
            pass

    encoder = SimpleNamespace(image_keys=("side", "wrist"),
                              backbone=SimpleNamespace(config=SimpleNamespace(patch_size=8)))
    inference_inputs = []
    policy = SimpleNamespace(obs_encoder=encoder, n_obs_steps=1,
                             predict_action=lambda obs, **kw: inference_inputs.append(set(obs)))
    dataset = Dataset()
    monkeypatch.setattr(visual.common, "choose_checkpoint", lambda *a: checkpoint)
    monkeypatch.setattr(visual.common, "load_checkpoint_payload", lambda *a: ({}, {}))
    monkeypatch.setattr(visual.common, "resolve_device", lambda *a: torch.device("cpu"))
    monkeypatch.setattr(visual.common, "instantiate_policy", lambda *a, **kw: policy)
    monkeypatch.setattr(visual.common, "instantiate_episode_dataset", lambda *a, **kw: dataset)
    monkeypatch.setattr(visual.common, "episode_sample_indices", lambda *a: [(100, 0), (101, 1)])
    monkeypatch.setattr(visual.common, "FFmpegVideoWriter", Writer)
    monkeypatch.setattr(visual, "FusionVisualAttentionTrace", lambda *a: Fusion())
    monkeypatch.setattr(visual, "DitActionAttentionCapture", lambda *a, **kw: Dit())

    argv = ["--checkpoint", str(checkpoint), "--episode", "0", *extra_args]
    if camera != "both":
        argv.extend(["--camera", camera])
    args = visual.parse_args(argv)
    paths = visual.run(args)
    assert dataset.closed
    assert paths["video"].parent.name == "attention_visualize"
    assert len(Writer.instances) == 1
    assert len(Writer.instances[0].frames) == 2  # 2 observations, each with 3 denoising calls.
    assert Writer.instances[0].fps == 50
    assert inference_inputs == [{"side", "wrist"}, {"side", "wrist"}]
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["video_frames"] == 2
    assert metadata["color_percentiles"] == list(args.attention_percentiles)
    expected_source = {"fixed": "fixed_cli", "percentile": "episode_percentiles"}.get(scale_mode, "episode_max")
    assert metadata["color_bounds_source"] == expected_source
    if scale_mode == "percentile":
        assert f"p{args.attention_percentiles[0]:g} to p{args.attention_percentiles[1]:g}" in metadata["color_scale"]
    elif scale_mode != "fixed":
        assert "0 to 100% of episode maximum" in metadata["color_scale"]
    expected_keys = ["side", "wrist"] if camera == "both" else [camera]
    assert metadata["all_camera_keys"] == ["side", "wrist"]
    assert metadata["camera_selection"] == camera
    assert metadata["attention_camera_keys"] == expected_keys
    assert metadata["raw_camera_keys"] == expected_keys
    assert set(metadata["camera_patch_grids"]) == set(expected_keys)
    raw_max = 12.5 if camera == "side" else 18.75
    if scale_mode == "fixed":
        expected_bounds = [1., 20.]
    elif scale_mode == "percentile":
        expected_bounds = [0., 12.5] if camera == "side" else [6.25, 18.75]
    elif scale_mode == "image_relative_episode_max":
        expected_bounds = [0., {"side": 50., "wrist": 75., "both": 37.5}[camera]]
    else:
        expected_bounds = [0., raw_max]
    assert metadata["color_bounds_percent"] == expected_bounds
    assert metadata["episode_max_scale"]["max_raw_percent"] == raw_max
    assert metadata["episode_max_scale"]["camera_keys"] == expected_keys
    assert "not all dataset episodes" in metadata["episode_max_scale"]["scope"]
    assert "wrench_plot" in metadata
    assert metadata["wrench_plot"]["contact"]["annotation_mode"] == "offline_backdated_confirmation_windows"
    with np.load(paths["raw"]) as data:
        assert data["visual_percent"].shape == (2, 3)
        assert data["visual_episode_percent"].shape == (2,)
        np.testing.assert_allclose(data["visual_percent"], 50 if camera == "both" else 25)
        np.testing.assert_allclose(data["wrench_percent"], 50)
        assert sum(key.endswith("_percent") and key.startswith("camera_") and
                   key[len("camera_"):].split("_", 1)[0].isdigit() and
                   key.count("_") == 2 for key in data.files) == len(expected_keys)
        assert all(f"camera_{index}_image_relative_percent" in data.files
                   for index in range(len(expected_keys)))
        assert data["image_entropy"].shape == (2,)
        for index in range(len(expected_keys)):
            averaged = data[f"camera_{index}_percent"].mean(axis=1)
            np.testing.assert_allclose(data[f"camera_{index}_episode_max_relative"], averaged / raw_max)
            assert data[f"camera_{index}_episode_max_relative"].shape == (2, 1, 2)
        assert max(float(data[f"camera_{index}_episode_max_relative"].max())
                   for index in range(len(expected_keys))) == 1.
        # The displayed signals are current raw measurements, independently
        # of teacher routing percentages and future conditioning windows.
        np.testing.assert_array_equal(data["recorded_wrench"], dataset.raw_wrench[100:104])
        np.testing.assert_array_equal(data["frame_wrench"], dataset.raw_wrench[[100, 101]])
        np.testing.assert_allclose(data["wrench_time_sec"], [0., .02, .04, .06])
        np.testing.assert_allclose(data["episode_time_sec"], [0., .02])
        np.testing.assert_array_equal(data["wrench_contact"], [False, True, True, True])
        np.testing.assert_array_equal(data["frame_contact"], [False, True])
        np.testing.assert_array_equal(data["wrench_causal_contact"], [False, False, True, True])
        np.testing.assert_array_equal(data["frame_causal_contact"], [False, False])
