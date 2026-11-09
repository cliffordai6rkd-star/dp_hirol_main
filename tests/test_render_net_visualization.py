import copy
import json
from types import SimpleNamespace

import dill
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.render_net import build_render_net
from diffusion_policy.model.vision.contact_curriculum import HysteresisContactDetector, VectorThresholdGate
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor
from diffusion_policy.scripts import visualize_render_net_episode as visual


def make_checkpoint(tmp_path, backbone="unet", region_enabled=False, version=2, conv_block="standard",
                    rendering_mode="residual", use_ema=False):
    root = tmp_path / "distill"
    checkpoint = root / "checkpoints" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    cfg = OmegaConf.create({
        "render_net": {"base_channels": 4, "conditioning_dim": 8, "residual_scale": 0.1,
                       "backbone": backbone,
                       "transformer": {"embed_dim": 16, "depth": 1, "patch_size": 8, "num_heads": 4},
                       "high_attention_render": {"enabled": region_enabled}},
        "dataset": {}, "training": {"device": "cpu", "amp": {"enabled": False, "dtype": "bfloat16"}},
    })
    teacher_cfg = OmegaConf.create({
        "policy": {"wrench_key": "future_wrench", "n_obs_steps": 2, "shape_meta": {"obs": {
            "side": {"type": "rgb", "shape": [3, 8, 12]},
            "wrist": {"type": "rgb", "shape": [3, 8, 12]},
            "future_wrench": {"type": "low_dim", "shape": [7, 6]},
        }}}, "task": {"dataset": {}},
    })
    if version == 2:
        cfg.render_net.update({"architecture_version": 2, "wrench_history_steps": 7,
                               "image_keys": ["side", "wrist"], "conv_block": conv_block})
        if rendering_mode == "gaussian_glow":
            cfg.render_net.update({"rendering_mode": rendering_mode, "glow": {"num_blobs": 3}})
        elif rendering_mode == "patch_residual":
            cfg.render_net.update({"rendering_mode": rendering_mode, "patch_residual_refine": True})
            cfg.render_net.high_attention_render.update({"source": "context", "selection": "soft",
                                                         "activation": "relu", "strength_scale": 2.0})
    OmegaConf.save(teacher_cfg, root / "teacher_config.yaml")
    model = build_render_net(6, OmegaConf.to_container(cfg.render_net, resolve=True))
    with torch.no_grad():
        model.output.bias.fill_(0.3)
    model.eval().requires_grad_(False)
    normalizer = LinearNormalizer()
    normalizer.fit({"future_wrench": torch.arange(120).reshape(20, 6).float()})
    preprocessor = WrenchPreprocessor(
        2, "future_wrench", "tau_ext", "future_tau_ext",
        HysteresisContactDetector(0.5, 1.5), VectorThresholdGate(0.0, enabled=False), normalizer,
    )
    cfg.wrench_preprocessing = preprocessor.get_config()
    state_dicts = {"model": model.state_dict(), "normalizer": normalizer.state_dict(),
                  "wrench_preprocessor": preprocessor.state_dict()}
    if use_ema:
        model = copy.deepcopy(model)
        with torch.no_grad():
            model.output.bias.fill_(0.6)
        state_dicts["ema_model"] = model.state_dict()
    torch.save({"cfg": cfg, "state_dicts": state_dicts,
                "pickles": {"global_step": dill.dumps(400), "best_val_loss": dill.dumps(0.4)}},
               checkpoint, pickle_module=dill)
    return checkpoint, model, preprocessor


@pytest.mark.parametrize("backbone", ["unet", "transformer"])
@pytest.mark.parametrize("region_enabled", [False, True])
@pytest.mark.parametrize("version,conv_block,rendering_mode", [
    (1, "standard", "residual"), (2, "standard", "residual"), (2, "bottleneck", "residual"),
    (2, "bottleneck", "gaussian_glow"), (2, "bottleneck", "patch_residual"),
])
def test_checkpoint_restores_rendering_and_training_normalization(
    tmp_path, backbone, region_enabled, version, conv_block, rendering_mode,
):
    checkpoint, expected_model, preprocessor = make_checkpoint(
        tmp_path, backbone, region_enabled, version, conv_block, rendering_mode,
    )
    model, restored_preprocessor, _, _, device, info = visual.load_render_net(checkpoint)
    assert device.type == "cpu" and info == {
        "global_step": 400, "best_val_loss": 0.4, "student_state_key": "model",
    }
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    if rendering_mode == "patch_residual":
        assert model.rendering_mode == "patch_residual" and model.region_activation == "relu"
    obs = {"side": torch.rand(1, 3, 3, 8, 12), "wrist": torch.rand(1, 3, 3, 8, 12),
           "future_wrench": torch.randn(1, 3, 7, 6), "tau_ext": torch.zeros(1, 3, 7, 7),
           "future_tau_ext": torch.zeros(1, 3, 7, 7)}
    obs["future_tau_ext"][:, :, 3:, 0] = 2.0
    with torch.no_grad():
        images, rendered = visual.render_observations(model, restored_preprocessor, obs,
                                                      ["side", "wrist"], 2)
        expected = expected_model({key: obs[key][:, :2] for key in ("side", "wrist")},
                                  preprocessor(obs)[0],
                                  future_contact=preprocessor(obs, return_future_contact=True)[2])
    torch.testing.assert_close(restored_preprocessor(obs)[0], preprocessor(obs)[0], rtol=0, atol=0)
    for key in images:
        torch.testing.assert_close(rendered[key], expected[key], rtol=0, atol=0)
        assert rendered[key].shape == (1, 2, 3, 8, 12)


def test_checkpoint_prefers_ema_for_rendered_rgb(tmp_path):
    checkpoint, expected_model, preprocessor = make_checkpoint(tmp_path, use_ema=True)
    model, restored_preprocessor, _, _, _, info = visual.load_render_net(checkpoint)
    assert info["student_state_key"] == "ema_model"
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    obs = {"side": torch.full((1, 2, 3, 8, 12), 0.25),
           "wrist": torch.full((1, 2, 3, 8, 12), 0.5),
           "future_wrench": torch.zeros(1, 2, 7, 6),
           "tau_ext": torch.zeros(1, 2, 7, 7), "future_tau_ext": torch.zeros(1, 2, 7, 7)}
    obs["future_tau_ext"][..., 0] = 2.0
    with torch.no_grad():
        images, rendered = visual.render_observations(model, restored_preprocessor, obs,
                                                      ["side", "wrist"], 2)
        expected = expected_model(images, preprocessor(obs)[0],
                                  future_contact=preprocessor(obs, return_future_contact=True)[2])
        payload = torch.load(checkpoint, pickle_module=dill)
        expected_model.load_state_dict(payload["state_dicts"]["model"])
        online = expected_model(images, preprocessor(obs)[0],
                                future_contact=preprocessor(obs, return_future_contact=True)[2])
    for key in images:
        torch.testing.assert_close(rendered[key], expected[key], rtol=0, atol=0)
        assert not torch.allclose(rendered[key], online[key])


def test_video_shows_actual_rendered_rgb_and_gain_affects_only_difference():
    images = {key: torch.zeros(1, 2, 3, 4, 6) for key in ("wrist", "side")}
    rendered = {key: value.clone() for key, value in images.items()}
    rendered["wrist"][:, -1, 0] = 0.1
    rendered["side"][:, -1, 2] = 0.2
    first = visual.video_frames(images, rendered, ["wrist", "side"], title="test", difference_gain=1)
    amplified = visual.video_frames(images, rendered, ["wrist", "side"], title="test", difference_gain=5)
    np.testing.assert_array_equal(first["rendered"], amplified["rendered"])
    np.testing.assert_array_equal(first["rendered"][0, 0], [25, 0, 0])
    np.testing.assert_array_equal(first["rendered"][0, 6], [0, 0, 51])
    np.testing.assert_array_equal(first["comparison"][48:52, :6], np.zeros((4, 6, 3), np.uint8))
    np.testing.assert_array_equal(first["comparison"][48:52, 6:12], first["rendered"][:, :6])
    assert not np.array_equal(first["comparison"][48:52, 12:18], amplified["comparison"][48:52, 12:18])


@pytest.mark.parametrize("camera", ["both", "wrist"])
@pytest.mark.parametrize("rendering_mode", ["residual", "gaussian_glow", "patch_residual"])
def test_episode_export_uses_real_time_and_selected_cameras(monkeypatch, tmp_path, capsys, camera, rendering_mode):
    use_ema = rendering_mode == "residual"
    checkpoint, _, _ = make_checkpoint(tmp_path, rendering_mode=rendering_mode, use_ema=use_ema)

    class Dataset:
        fps = 50
        closed = False

        def __init__(self):
            self.lerobot_dataset = SimpleNamespace(close=self.close)

        def __getitem__(self, index):
            return {"obs": {"side": torch.full((2, 3, 8, 12), 0.25),
                            "wrist": torch.full((2, 3, 8, 12), 0.5),
                            "future_wrench": torch.zeros(2, 7, 6),
                            "tau_ext": torch.zeros(2, 7, 7), "future_tau_ext": torch.full((2, 7, 7), 2.0)}}

        def close(self):
            self.closed = True

    class Writer:
        instances = []

        def __init__(self, path, width, height, fps, codec):
            self.frames = []
            self.fps = fps
            self.closed = False
            self.instances.append(self)
            path.write_bytes(b"video")

        def write(self, frame):
            self.frames.append(frame.copy())

        def close(self):
            self.closed = True

    dataset = Dataset()
    monkeypatch.setattr(visual.common, "instantiate_episode_dataset", lambda *a: dataset)
    monkeypatch.setattr(visual.common, "episode_sample_indices", lambda *a: [(100 + i, i) for i in range(6)])
    monkeypatch.setattr(visual.common, "FFmpegVideoWriter", Writer)
    args = visual.parse_args(["--checkpoint", str(checkpoint), "--episode", "0",
                              "--camera", camera, "--stride", "2", "--save-frames"])
    paths = visual.run(args)
    assert dataset.closed and len(Writer.instances) == 2
    assert all(writer.closed and len(writer.frames) == 3 and writer.fps == 25 for writer in Writer.instances)
    assert paths["comparison"].parent.name == "attention_visualize"
    assert paths["preview"].is_file() and paths["rendered_preview"].is_file()
    metadata = json.loads(paths["metadata"].read_text())
    state_key = "ema_model" if use_ema else "model"
    assert metadata["student_state_key"] == state_key
    assert f"weights={state_key}" in capsys.readouterr().out
    assert metadata["rendering_mode"] == rendering_mode
    assert ("Gaussian glow" in metadata["rendered_rgb"]) == (rendering_mode == "gaussian_glow")
    assert metadata["video_frames"] == 3 and metadata["fps"] == 25
    assert metadata["input_camera_keys"] == ["side", "wrist"]
    assert metadata["display_camera_keys"] == (["wrist", "side"] if camera == "both" else ["wrist"])
    with np.load(paths["raw"]) as values:
        np.testing.assert_array_equal(values["anchors"], [100, 102, 104])
        np.testing.assert_allclose(values["episode_time_sec"], [0, 0.04, 0.08])
        assert (values["camera_0_mae"] > 0).all()
    for key in ("comparison", "rendered"):
        assert len(list(visual.rendering.frame_directory(paths[key]).glob("*.png"))) == 3
