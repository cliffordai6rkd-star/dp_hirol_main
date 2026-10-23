from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from diffusion_policy.scripts import visualize_denoising_attention_episode as visual


def test_percentages_use_one_step_condition_mass_and_conserve_routes():
    class Trace:
        def project(self, context):
            assert context.shape == (2, 3)
            return {"side": torch.tensor([[[1.0, 1.0]], [[2.0, 0.0]]]),
                    "wrist": torch.tensor([[[2.0]], [[1.0]]])}, {
                "input_mass": torch.tensor([8.0, 4.0]),
                "visual_mass": torch.tensor([4.0, 3.0]),
                "wrench_mass": torch.tensor([4.0, 1.0]),
            }

    maps, image, wrench, mass = visual.routing_percentages(torch.ones(2, 3), Trace())
    np.testing.assert_allclose(image, [50, 75])
    np.testing.assert_allclose(wrench, [50, 25])
    np.testing.assert_allclose(mass, [8, 4])
    np.testing.assert_allclose(maps["side"][0, 0], [12.5, 12.5])
    np.testing.assert_allclose(maps["wrist"][1, 0], [25])


def test_video_layout_marks_cross_or_self_and_moves_step_marker(tmp_path):
    checkpoint = tmp_path / "run" / "checkpoints" / "best.ckpt"
    paths = visual.output_paths(checkpoint, 3, "cross+self")
    assert paths["video"] == tmp_path / "run" / "attention_visualize" / "denoising_cross_self_episode_3.mp4"
    assert visual.fusion_label(SimpleNamespace(cross_attention=[1], self_attention=[])) == "cross"
    assert visual.fusion_label(SimpleNamespace(cross_attention=[], self_attention=[1])) == "self"
    record = {"anchor": 14, "timesteps": np.array([9, 4, 0]),
              "visual": np.array([25., 50., 75.]), "wrench": np.array([75., 50., 25.]),
              "cameras": {key: {"images": np.zeros((1, 16, 128, 3), dtype=np.uint8),
                                "percent": np.array([[[0., 2.]], [[2., 0.]], [[1., 1.]]]),
                                "grid": (1, 2)} for key in ("side", "wrist")}}
    args = SimpleNamespace(episode=3, observation="last", alpha=0.55,
                           attention_percentiles=(1.0, 99.0))
    image_route = np.array([25., 50., 75.])
    wrench_route = 100 - image_route
    times = np.array([0.0, 0.5, 1.0])
    first = visual.render_frame(record, 0, ["wrist", "side"], (0.0, 2.0), args, "cross+self",
                                image_route, wrench_route, times)
    last = visual.render_frame(record, 2, ["wrist", "side"], (0.0, 2.0), args, "cross+self",
                               image_route, wrench_route, times)
    assert first.shape == last.shape and first.shape[1] == 256
    assert first.shape[0] % 2 == 0 and first.flags.c_contiguous
    assert not np.array_equal(first, last)
    bright = visual.heat_overlay(np.zeros((16, 32, 3), dtype=np.uint8), [0, 2], (1, 2), (0.0, 2.0), 1)
    assert bright[8, 24, 0] > bright[8, 8, 0]
    # The lower chart is genuinely part of the same RGB frame.
    assert np.any(first[-245:] > 30)


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


@pytest.mark.parametrize("percentiles", [(2, 98), (10, 90), (0, 100)])
def test_display_percentiles_are_configurable(percentiles):
    args = visual.parse_args(["--checkpoint", "model.ckpt", "--episode", "0",
                              "--attention-percentiles", *map(str, percentiles)])
    assert args.attention_percentiles == list(percentiles)


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


def test_episode_video_uses_one_frame_per_observation_at_dataset_fps(monkeypatch, tmp_path):
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
            maps = {key: torch.ones(3, 1, 2) for key in self.image_keys}
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

        def __init__(self):
            self.lerobot_dataset = SimpleNamespace(close=self.close)

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
    policy = SimpleNamespace(obs_encoder=encoder, n_obs_steps=1, predict_action=lambda *a, **kw: None)
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

    args = visual.parse_args(["--checkpoint", str(checkpoint), "--episode", "0"])
    paths = visual.run(args)
    assert dataset.closed
    assert paths["video"].parent.name == "attention_visualize"
    assert len(Writer.instances) == 1
    assert len(Writer.instances[0].frames) == 2  # 2 observations, each with 3 denoising calls.
    assert Writer.instances[0].fps == 50
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["video_frames"] == 2
    assert metadata["color_percentiles"] == [1, 99]
    assert metadata["color_bounds_percent"] == [0.0, 12.5]
    with np.load(paths["raw"]) as data:
        assert data["visual_percent"].shape == (2, 3)
        assert data["visual_episode_percent"].shape == (2,)
