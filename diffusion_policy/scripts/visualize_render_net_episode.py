"""Render an episode with a distilled RenderNet and compare its RGB residual.

Loads RenderNet and the teacher's saved wrench preprocessing state. Inference
shares contact gating, prefix filling and normalization with the teacher;
it never instantiates the teacher policy, DINO, teacher fusion or DiT.
Outputs run at the dataset FPS, with one frame per selected episode anchor.
"""

from __future__ import annotations

if __name__ == "__main__":
    import os
    import pathlib
    import sys

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import argparse
import contextlib
import json
import math
import pathlib

import cv2
import dill
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from diffusion_policy.model.vision.render_net import build_render_net
from diffusion_policy.model.vision.wrench_preprocessing import WrenchPreprocessor
from diffusion_policy.scripts import visualize_cross_attention_episode as common
from diffusion_policy.scripts import visualize_image_query_cross_attention_episode as rendering
from diffusion_policy.workspace.train_force_aware_diffusion_workspace import resolve_amp_dtype


def parse_args(argv=None, *, checkpoint_required=True, episode_required=True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=checkpoint_required,
                        help="RenderNet distillation checkpoint, e.g. checkpoints/best.ckpt.")
    parser.add_argument("--episode", type=int, required=episode_required)
    parser.add_argument("--camera", choices=("wrist", "side", "both"), default="both")
    parser.add_argument("--dataset-path", type=pathlib.Path)
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav")
    parser.add_argument("--device", help="Default: distillation training device, with CPU fallback.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fps", type=float, help="Default: dataset FPS divided by stride.")
    parser.add_argument("--video-codec", choices=("h264", "av1", "mpeg4"), default="h264")
    parser.add_argument("--difference-gain", type=float, default=1.0,
                        help="Magnification for the signed RGB difference panel only (default: 1).")
    parser.add_argument("--save-frames", action="store_true", help="Also save PNGs for every video frame.")
    parser.add_argument("--disable-high-attention-render", action="store_true",
                        help="Bypass a trained region head and apply the residual across the full image.")
    parser.add_argument("--threshold", type=float,
                        help="Override the soft-mask threshold: retain M only where M > threshold.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.episode < 0 or args.stride < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("--episode must be nonnegative; --stride and --max-frames must be positive")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps must be positive and finite")
    if not math.isfinite(args.difference_gain) or args.difference_gain <= 0:
        parser.error("--difference-gain must be positive and finite")
    if args.threshold is not None and (not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1):
        parser.error("--threshold must be finite and between 0 and 1")
    return args


def output_paths(checkpoint, args):
    directory = checkpoint.parent.parent / "attention_visualize"
    stem = f"render_net_episode_{args.episode}_{args.camera}"
    if args.threshold is not None:
        stem += f"_threshold_{args.threshold:g}"
    return {
        "comparison": directory / f"{stem}_comparison.mp4",
        "rendered": directory / f"{stem}_rendered.mp4",
        "preview": directory / f"{stem}_comparison_preview.png",
        "rendered_preview": directory / f"{stem}_rendered_preview.png",
        "raw": directory / f"{stem}_stats.npz",
        "metadata": directory / f"{stem}_metadata.json",
    }


def load_render_net(checkpoint, device=None):
    payload, cfg = common.load_checkpoint_payload(checkpoint)
    if "render_net" not in cfg:
        raise ValueError("Expected a RenderNet distillation checkpoint")
    if not cfg.get("wrench_preprocessing") or "wrench_preprocessor" not in payload["state_dicts"]:
        raise ValueError("This checkpoint predates teacher-aligned wrench preprocessing. "
                         "Retrain RenderNet with the corrected training entry point.")
    # The training run saves the resolved teacher config beside checkpoints.
    teacher_config = checkpoint.parent.parent / "teacher_config.yaml"
    if teacher_config.is_file():
        teacher_cfg = OmegaConf.load(teacher_config)
    else:
        _, teacher_cfg = common.load_checkpoint_payload(pathlib.Path(cfg.teacher_checkpoint))
    device = common.resolve_device(device, cfg)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    wrench_key = str(teacher_cfg.policy.wrench_key)
    wrench_dim = int(teacher_cfg.policy.shape_meta.obs[wrench_key].shape[-1])
    model = build_render_net(wrench_dim, OmegaConf.to_container(cfg.render_net, resolve=True))
    state_key = "ema_model" if "ema_model" in payload["state_dicts"] else "model"
    model.load_state_dict(payload["state_dicts"][state_key], strict=True)
    model.to(device).eval().requires_grad_(False)
    preprocessor = WrenchPreprocessor.from_config(cfg.wrench_preprocessing)
    preprocessor.load_state_dict(payload["state_dicts"]["wrench_preprocessor"], strict=True)
    preprocessor.to(device).eval().requires_grad_(False)
    pickles = payload.get("pickles", {})
    info = {key: dill.loads(pickles[key]) for key in ("global_step", "best_val_loss") if key in pickles}
    info["student_state_key"] = state_key
    return model, preprocessor, cfg, teacher_cfg, device, info


def render_observations(model, preprocessor, obs, image_keys, n_obs_steps, *, use_attention_mask=None):
    """Use the same physical gating and normalization as teacher and student training."""
    images = {key: obs[key][:, :n_obs_steps] for key in image_keys}
    predict_phase = bool(getattr(model, "phase_prediction_enabled", False))
    # Dataset phase/alpha targets use future contact onset.  Deployment must
    # infer its gate from the causal RGB and measured wrench observations.
    preprocessing_obs = ({key: value for key, value in obs.items()
                          if key not in {"contact_alpha", "contact_phase"}}
                         if predict_phase else obs)
    wrench, contact, future_contact = preprocessor(preprocessing_obs, return_future_contact=True)
    contact_gate = future_contact if future_contact is not None else contact
    return images, model(
        images, wrench, future_contact=contact_gate,
        contact_alpha=(obs["contact_alpha"][:, :n_obs_steps]
                       if not predict_phase and "contact_alpha" in obs else None),
        use_attention_mask=use_attention_mask,
    )


def _to_rgb(values):
    values = values.detach().float().cpu().numpy().transpose(1, 2, 0)
    return np.ascontiguousarray((values.clip(0, 1) * 255).astype(np.uint8))


def _label(image, text):
    title = np.full((24, image.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(title, text, (7, 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.43, (245, 245, 245), 1, cv2.LINE_AA)
    return np.concatenate((title, image), axis=0)


def _tile(images, *, horizontal):
    """Pad to accommodate checkpoints with different camera resolutions."""
    if horizontal:
        canvas = np.zeros((max(x.shape[0] for x in images), sum(x.shape[1] for x in images), 3), np.uint8)
    else:
        canvas = np.zeros((sum(x.shape[0] for x in images), max(x.shape[1] for x in images), 3), np.uint8)
    offset = 0
    for image in images:
        if horizontal:
            canvas[:image.shape[0], offset:offset + image.shape[1]] = image
            offset += image.shape[1]
        else:
            canvas[offset:offset + image.shape[0], :image.shape[1]] = image
            offset += image.shape[0]
    return canvas


def video_frames(images, rendered, camera_keys, *, title, difference_gain):
    """Show latest observation; signed differences are centered at neutral gray."""
    rows, rendered_panels = [], []
    for key in camera_keys:
        original = images[key][0, -1].float()
        result = rendered[key][0, -1].float()
        difference = _to_rgb(0.5 + difference_gain * (result - original))
        original_rgb, rendered_rgb = _to_rgb(original), _to_rgb(result)
        rows.append(_tile([
            _label(original_rgb, f"{key} | Original RGB"),
            _label(rendered_rgb, f"{key} | Rendered RGB"),
            _label(difference, f"{key} | RGB delta x{difference_gain:g}"),
        ], horizontal=True))
        rendered_panels.append(rendered_rgb)
    comparison = _tile(rows, horizontal=False)
    comparison = _label(comparison, title)
    comparison = _tile([
        comparison,
        _label(np.zeros((0, comparison.shape[1], 3), np.uint8),
               "Delta: gray = zero; displayed RGB = clip(0.5 + gain * (rendered - original), 0, 1)"),
    ], horizontal=False)
    outputs = {"comparison": comparison, "rendered": _tile(rendered_panels, horizontal=True)}
    return {key: np.ascontiguousarray(np.pad(value, ((0, value.shape[0] % 2),
                                                    (0, value.shape[1] % 2), (0, 0))))
            for key, value in outputs.items()}


def _save_rgb(path, frame):
    if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not save image: {path}")


def run(args):
    checkpoint = common.choose_checkpoint(args.checkpoint, None)
    model, preprocessor, cfg, teacher_cfg, device, checkpoint_info = load_render_net(checkpoint, args.device)
    if args.threshold is not None:
        if (not hasattr(model, "region_threshold") or not model.high_attention_enabled
                or model.region_selection != "soft" or args.disable_high_attention_render):
            raise ValueError("--threshold requires a RenderNet v2 checkpoint with a soft region mask")
        model.region_threshold = args.threshold
        with open_dict(cfg):
            cfg.render_net.high_attention_render.threshold = args.threshold
    attention_enabled = model.high_attention_enabled and not args.disable_high_attention_render
    image_keys = sorted(key for key, spec in teacher_cfg.policy.shape_meta.obs.items() if spec.get("type") == "rgb")
    camera_keys = [key for key in ("wrist", "side") if key in image_keys]
    camera_keys.extend(key for key in image_keys if key not in camera_keys)
    if args.camera != "both":
        if args.camera not in image_keys:
            raise KeyError(f"Camera {args.camera!r} unavailable: {image_keys}")
        camera_keys = [args.camera]
    steps = preprocessor.n_obs_steps
    wrench_key = preprocessor.wrench_key
    # Preserve the dataset overrides used for distillation, without instantiating a teacher policy.
    teacher_cfg = OmegaConf.create(OmegaConf.to_container(teacher_cfg, resolve=True))
    with open_dict(teacher_cfg):
        teacher_cfg.task.dataset.update(OmegaConf.to_container(cfg.dataset, resolve=True))
    dataset = common.instantiate_episode_dataset(teacher_cfg, args.dataset_path, args.video_backend)
    paths = output_paths(checkpoint, args)
    writers = {}
    amp_enabled = device.type == "cuda" and bool(cfg.training.amp.enabled)
    amp_dtype = resolve_amp_dtype(cfg.training.amp.dtype)
    print(f"RenderNet: {checkpoint}; step={checkpoint_info.get('global_step')}; "
          f"weights={checkpoint_info['student_state_key']}\n"
          f"Device: {device}; cameras={image_keys}; displayed={camera_keys}; wrench={wrench_key}", flush=True)
    try:
        selected = common.episode_sample_indices(dataset, args.episode)[::args.stride]
        if args.max_frames is not None:
            selected = selected[:args.max_frames]
        dataset_fps = common.infer_dataset_fps(dataset)
        fps = args.fps if args.fps is not None else dataset_fps / args.stride
        rendering.prepare_outputs(paths, overwrite=args.overwrite, save_frames=args.save_frames)
        stats = {key: {"mae": [], "max_abs_delta": []} for key in camera_keys}
        for index, (anchor, sample_index) in enumerate(selected):
            sample = dataset[sample_index]
            obs = {key: value.unsqueeze(0).to(device) for key, value in sample["obs"].items()
                   if key in image_keys or key in preprocessor.required_keys}
            autocast = torch.autocast("cuda", dtype=amp_dtype) if amp_enabled else contextlib.nullcontext()
            with torch.inference_mode(), autocast:
                images, rendered = render_observations(model, preprocessor, obs, image_keys, steps,
                                                       use_attention_mask=attention_enabled)
            for key in camera_keys:
                delta = (rendered[key][0, -1].float() - images[key][0, -1].float()).abs()
                stats[key]["mae"].append(float(delta.mean()))
                stats[key]["max_abs_delta"].append(float(delta.max()))
            seconds = (anchor - selected[0][0]) / dataset_fps
            title = f"RenderNet RGB | ep {args.episode} | {index + 1}/{len(selected)} | {seconds:.2f}s | step {checkpoint_info.get('global_step', '?')}"
            frames = video_frames(images, rendered, camera_keys, title=title, difference_gain=args.difference_gain)
            for key, frame in frames.items():
                if key not in writers:
                    writers[key] = common.FFmpegVideoWriter(paths[key], frame.shape[1], frame.shape[0], fps, args.video_codec)
                writers[key].write(frame)
                if args.save_frames:
                    _save_rgb(rendering.frame_directory(paths[key]) / f"frame_{index:06d}.png", frame)
            if index == len(selected) // 2:
                _save_rgb(paths["preview"], frames["comparison"])
                _save_rgb(paths["rendered_preview"], frames["rendered"])
            if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(selected):
                print(f"Rendered {index + 1}/{len(selected)} episode frames", flush=True)
        while writers:
            _, writer = writers.popitem()
            writer.close()
        arrays = {"anchors": np.asarray([a for a, _ in selected], np.int64),
                  "episode_time_sec": (np.asarray([a for a, _ in selected]) - selected[0][0]) / dataset_fps}
        for i, key in enumerate(camera_keys):
            for metric, values in stats[key].items():
                arrays[f"camera_{i}_{metric}"] = np.asarray(values, dtype=np.float32)
        np.savez_compressed(paths["raw"], **arrays)
        metadata = {
            "checkpoint": str(checkpoint), **checkpoint_info, "episode": args.episode,
            "video_frames": len(selected), "fps": fps, "dataset_fps": dataset_fps,
            "video_codec": args.video_codec, "stride": args.stride,
            "input_camera_keys": image_keys, "display_camera_keys": camera_keys,
            "observation_steps": steps, "display_observation": "last",
            "wrench_key": wrench_key, "wrench_normalizer": "restored from RenderNet checkpoint, identical to teacher",
            "wrench_preprocessing": OmegaConf.to_container(cfg.wrench_preprocessing, resolve=True),
            "wrench_preprocessing_observation_keys": list(preprocessor.required_keys),
            "residual_scale": model.residual_scale, "difference_gain": args.difference_gain,
            "image_encoder": model.image_encoder_name,
            "wrench_encoder": model.wrench_encoder_name,
            "image_self_attn_layers": len(model.backbone.image_self_attention),
            "cross_attn_layers": len(model.backbone.cross_attention),
            "decoder_layers": model.backbone.decoder_layers,
            "high_attention_render_enabled": attention_enabled,
            "region_source": "student-predicted teacher region" if attention_enabled else "full image",
            "high_attention_render_config": OmegaConf.to_container(cfg.render_net.high_attention_render, resolve=True)
            if "high_attention_render" in cfg.render_net else None,
            "rendering_mode": getattr(model, "rendering_mode", "residual"),
            "rendered_rgb": (
                "original + contact_alpha * (1 - original) * (1 - exp(-residual_scale * region_weight * Gaussian glow)); no display amplification"
                if getattr(model, "rendering_mode", "residual") == "gaussian_glow" else
                "clamp(original + contact_alpha * residual_scale * region_weight * tanh(RenderNet RGB residual), 0, 1); region_weight=1 when disabled; no display amplification"
            ),
            "difference_rgb": "clip(0.5 + difference_gain * (rendered - original), 0, 1); signed RGB delta centered at gray, not attention",
            "stats_scope": "displayed cameras and latest observation; RGB units in [0,1], before video encoding",
            "camera_stats": {key: {"mean_abs_rgb_delta": float(np.mean(stats[key]["mae"])),
                                   "max_abs_rgb_delta": float(np.max(stats[key]["max_abs_delta"]))}
                             for key in camera_keys},
            "preview_frame_index": len(selected) // 2,
            "inference_modules": ["RenderNet", "teacher wrench preprocessing (contact/gate/normalizer)"],
        }
        paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Comparison: {paths['comparison']}\nRendered RGB: {paths['rendered']}\nPreview: {paths['preview']}", flush=True)
        return paths
    finally:
        for writer in writers.values():
            writer.close()
        dataset.lerobot_dataset.close()


if __name__ == "__main__":
    run(parse_args())
