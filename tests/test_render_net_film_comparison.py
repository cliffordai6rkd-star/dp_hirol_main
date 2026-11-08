import csv
import json
import math
from types import SimpleNamespace

import dill
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from diffusion_policy.scripts import compare_render_net_film_episode as comparison
from test_render_net_visualization import make_checkpoint


def film_checkpoint(tmp_path):
    torch.manual_seed(123)
    checkpoint, _, preprocessor = make_checkpoint(tmp_path, region_enabled=True)
    payload = torch.load(checkpoint, pickle_module=dill)
    payload["cfg"].render_net.update({"spatial_modulation": {"enabled": True}, "separate_gate_path": True})
    payload["cfg"].render_net.high_attention_render.threshold = 0.1
    model = comparison.visual.build_render_net(
        6, OmegaConf.to_container(payload["cfg"].render_net, resolve=True),
    ).eval().requires_grad_(False)
    with torch.no_grad():
        model.output.weight.normal_(std=0.05)
        model.mask_head.bias.fill_(1.0)
        for modulation in model.spatial_modulations:
            modulation.affine[-1].bias.fill_(0.3)
    payload["state_dicts"]["model"] = model.state_dict()
    torch.save(payload, checkpoint, pickle_module=dill)
    return checkpoint, model, preprocessor


def observations():
    return {
        "side": torch.rand(1, 2, 3, 8, 12) * 0.5 + 0.25,
        "wrist": torch.rand(1, 2, 3, 12, 8) * 0.5 + 0.25,
        "future_wrench": torch.randn(1, 2, 7, 6),
        "tau_ext": torch.zeros(1, 2, 7, 7),
        "future_tau_ext": torch.zeros(1, 2, 7, 7),
    }


def test_ablation_bypasses_only_film_and_preserves_gates_weights_and_preprocessing(tmp_path):
    _, model, preprocessor = film_checkpoint(tmp_path)
    obs = observations()
    obs["future_tau_ext"][:, 1, :, 0] = 2.0
    before = {key: value.clone() for key, value in model.state_dict().items()}
    calls = {"film": 0, "preprocessor": 0, "rgb": 0}

    def count(name):
        def hook(*args):
            calls[name] += 1
        return hook

    handles = [module.register_forward_hook(count("film")) for module in model.spatial_modulations]
    handles += [preprocessor.register_forward_hook(count("preprocessor")),
                model.backbone.register_forward_hook(count("rgb"))]
    try:
        images, branches = comparison.compare_observations(model, preprocessor, obs, ["side", "wrist"])
    finally:
        for handle in handles:
            handle.remove()
    assert calls == {"film": 2 * len(model.spatial_modulations), "preprocessor": 1, "rgb": 4}
    assert model.spatial_modulation_enabled
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    for camera in images:
        for field in ("gate", "intensity"):
            torch.testing.assert_close(branches["film_on"][field][camera],
                                       branches["film_off"][field][camera], atol=0, rtol=0)
        for branch in branches.values():
            torch.testing.assert_close(branch["rendered"][camera][:, 0], images[camera][:, 0], atol=0, rtol=0)
        delta = branches["film_on"]["rendered"][camera] - branches["film_off"]["rendered"][camera]
        assert delta[:, 1].abs().max() > 0


def test_film_setting_is_restored_if_disabled_pass_fails(tmp_path):
    _, model, preprocessor = film_checkpoint(tmp_path)

    def fail_disabled(*args):
        if not model.spatial_modulation_enabled:
            raise RuntimeError("disabled pass failed")

    handle = model.backbone.register_forward_pre_hook(fail_disabled)
    try:
        with pytest.raises(RuntimeError, match="disabled pass failed"):
            comparison.compare_observations(model, preprocessor, observations(), ["side", "wrist"])
    finally:
        handle.remove()
    assert model.spatial_modulation_enabled


def test_difference_uses_latest_float_rgb_and_weights_cameras_by_pixel_count():
    first = {"small": torch.zeros(1, 2, 3, 2, 2), "large": torch.zeros(1, 2, 3, 2, 6)}
    second = {key: value.clone() for key, value in first.items()}
    second["small"][:, 0] = 0.9  # An undisplayed observation must not affect the metric.
    second["small"][:, -1] = 0.1
    second["large"][:, -1] = 0.3
    metrics = comparison.difference_metrics({"film_on": {"rendered": first},
                                              "film_off": {"rendered": second}}, ["small", "large"])
    assert metrics["small"]["mae"] == pytest.approx(0.1)
    assert metrics["overall"]["mae"] == pytest.approx(0.25)
    assert metrics["overall"]["mse"] == pytest.approx(0.07)
    assert metrics["overall"]["rmse"] == pytest.approx(math.sqrt(0.07))
    assert metrics["overall"]["max_abs_delta"] == pytest.approx(0.3)
    summary = comparison.summarize([metrics, metrics], ["small", "large", "overall"])
    assert summary["overall"]["rmse"] == pytest.approx(metrics["overall"]["rmse"])


def test_export_has_three_rows_synchronized_curves_and_csv(monkeypatch, tmp_path):
    checkpoint, _, _ = film_checkpoint(tmp_path)

    class Dataset:
        fps = 20
        closed = False

        def __init__(self):
            self.lerobot_dataset = SimpleNamespace(close=self.close)

        def __getitem__(self, index):
            obs = observations()
            obs["future_tau_ext"][..., 0] = 2.0
            return {"obs": {key: value.squeeze(0) for key, value in obs.items()}}

        def close(self):
            self.closed = True

    class Writer:
        instances = []

        def __init__(self, path, width, height, fps, codec):
            self.frames, self.fps, self.closed = [], fps, False
            self.instances.append(self)
            path.write_bytes(b"video")

        def write(self, frame):
            self.frames.append(frame.copy())

        def close(self):
            self.closed = True

    dataset = Dataset()
    monkeypatch.setattr(comparison.visual.common, "instantiate_episode_dataset", lambda *args: dataset)
    monkeypatch.setattr(comparison.visual.common, "episode_sample_indices", lambda *args: [(100 + i, i) for i in range(6)])
    monkeypatch.setattr(comparison.visual.common, "FFmpegVideoWriter", Writer)
    args = comparison.parse_args(["--checkpoint", str(checkpoint), "--stride", "2", "--panel-width", "128"])
    paths = comparison.run(args)
    assert dataset.closed and len(Writer.instances) == 1
    writer = Writer.instances[0]
    assert writer.closed and len(writer.frames) == 3 and writer.fps == 10
    assert paths["preview"].is_file()
    metadata = json.loads(paths["metadata"].read_text())
    assert len(metadata["rows"]) == len(metadata["row_heights"]) == 3
    assert sum(metadata["row_heights"]) <= writer.frames[0].shape[0] <= sum(metadata["row_heights"]) + 1
    assert metadata["display_camera_keys"] == ["wrist", "side"]
    assert metadata["summary"]["overall"]["mae"] > 0
    assert metadata["region_threshold"] == 0.1
    assert all(frame.shape == tuple(metadata["video_shape_hwc"]) for frame in writer.frames)
    with paths["metrics"].open() as stream:
        records = list(csv.DictReader(stream))
    np.testing.assert_allclose([float(record["episode_time_sec"]) for record in records], [0, 0.1, 0.2])
    assert all(float(record["overall_rmse"]) >= float(record["overall_mae"]) > 0 for record in records)


def test_checkpoint_without_film_is_rejected_before_loading_dataset(tmp_path):
    checkpoint, _, _ = make_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="spatial_modulation.enabled=true"):
        comparison.run(comparison.parse_args(["--checkpoint", str(checkpoint)]))
