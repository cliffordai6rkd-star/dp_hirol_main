"""Visualize cross-attention residual U and wrench-induced vector difference ΔU.

Same RGB, same frozen encoder, two wrench conditions. U is captured after MHA
output projection, before the image residual, FFN and final normalization.
The default videos show ||U(W)||2 and ||U(W)-U(W0)||2, using each individual
camera/observation image's own patch p2/p98 color scale. Use --color-scope
episode for a shared episode scale. These are feature magnitudes, not attention
probabilities, physical force units, or final-action attribution. No diffusion
sampling is needed. Original values and preprocessing diagnostics are saved.
The captured update includes the same deterministic contact alpha used by the
teacher encoder, so zero-alpha observations contribute no wrench residual.
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
import inspect
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.scripts import visualize_cross_attention_episode as common
from diffusion_policy.scripts import visualize_image_query_cross_attention_episode as rendering
from diffusion_policy.scripts.episode_wrench_plot import WrenchEpisodePlot, load_episode_wrench
from diffusion_policy.scripts.visualize_denoising_attention_episode import heat_overlay
from diffusion_policy.model.vision.wrench_preprocessing import compute_contact_alpha


METRICS = {
    "ca_update": ("Residual U", "||U(W)||2"),
    "injection_effect": ("Wrench-induced residual change", "||U(W)-U(W0)||2"),
    "context_effect": ("Final image-token change", "||Z(W)-Z(W0)||2"),
}
METRIC_LABELS = {
    **METRICS,
    "action_sensitivity": ("Action sensitivity | single-patch fusion replacement",
                           "mean_DDIM ||epsilon(W)-epsilon(Z_p<-Z_p(W0))||2"),
}


def make_zero_wrench_obs(policy, obs):
    """Keep image objects unchanged; zero every required raw wrench/phase key."""
    zero = dict(obs)
    zero.pop("contact_alpha", None)
    keys = dict.fromkeys(getattr(policy, name, None) for name in
                         ("wrench_key", "phase_wrench_key", "future_phase_wrench_key"))
    for key in keys:
        if key is not None:
            if key not in obs:
                raise KeyError(f"Required wrench observation {key!r} is missing")
            zero[key] = torch.zeros_like(obs[key])
    return zero


class WrenchEffectCapture:
    """Capture ordinary encoder outputs without changing attention kernels.

    Supports the current image-query, one-CA-layer, no-preceding-SA structure.
    Both outputs have [observation, all_camera_patches, channels], batch one.
    """

    def __init__(self, policy):
        self.policy = policy
        self.encoder = policy.obs_encoder
        self.steps = int(policy.n_obs_steps)
        self.handles = []
        self.active = False
        self._updates = []
        self._alpha = None

    def __enter__(self):
        if self.active:
            raise RuntimeError("Wrench effect capture is already active")
        if getattr(self.encoder, "cross_attn_query", None) != "image":
            raise ValueError("Wrench residual maps require image-query cross-attention")
        if int(getattr(self.encoder, "cross_attn_layers", 0)) != 1:
            raise ValueError("Wrench residual maps currently support a single cross-attention layer")
        if int(getattr(self.encoder, "pre_cross_self_attn_layers", 0)):
            raise ValueError("Preceding joint self-attention is not supported by this residual-only diagnostic")
        if getattr(self.policy, "training", False) or getattr(self.encoder, "training", False):
            raise ValueError("Wrench effect capture requires an eval-mode policy and encoder")
        mode = getattr(self.encoder, "camera_attention_mode", "joint")
        if mode not in {"joint", "per_camera"}:
            raise ValueError(f"Unsupported camera attention mode {mode!r}")
        self.mode = mode
        self.handles.append(self.encoder.cross_attention.register_forward_hook(self._capture))
        self.active = True
        return self

    def __exit__(self, *exc_info):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self._updates = []
        self._alpha = None
        self.active = False

    def _capture(self, module, inputs, output):
        update = output[0]
        if not torch.is_tensor(update) or update.ndim != 3 or update.shape[0] != self.steps:
            raise ValueError("Expected CA update [n_obs_steps, image_patches, channels] with batch one")
        if self._alpha is not None:
            update = update * self._alpha.to(device=update.device, dtype=update.dtype)
        self._updates.append(update.detach().float().cpu().clone())

    @torch.inference_mode()
    def encode(self, obs, *, normalized_zero=False):
        if not self.active:
            raise RuntimeError("Use WrenchEffectCapture as a context manager before encoding")
        first = obs[self.encoder.image_keys[0]]
        if first.shape[0] != 1:
            raise ValueError("Episode residual visualization requires batch size one")
        self._updates = []
        self._alpha = None
        signature = inspect.signature(self.encoder.forward)

        def capture_inputs(module, args, kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            future_contact = bound.arguments.get("future_contact")
            if future_contact is not None:
                self._alpha = compute_contact_alpha(future_contact)[0].detach().cpu().reshape(
                    self.steps, 1, 1
                )
            if normalized_zero:
                if "wrench_history" not in bound.arguments:
                    raise ValueError("Encoder has no explicit wrench_history input")
                bound.arguments["wrench_history"] = torch.zeros_like(bound.arguments["wrench_history"])
                return bound.args, bound.kwargs

        replacement = self.encoder.register_forward_pre_hook(capture_inputs, with_kwargs=True)
        try:
            context = self.policy._encode_context(obs, apply_curriculum_mask=False)
        finally:
            replacement.remove()
        expected = 1 if self.mode == "joint" else len(self.encoder.image_keys)
        if len(self._updates) != expected:
            raise RuntimeError(f"Expected {expected} CA calls, captured {len(self._updates)}")
        update = torch.cat(self._updates, dim=1)
        patches, channels = update.shape[1:]
        if tuple(context.shape) != (1, self.steps * patches, channels):
            raise ValueError("Final context does not preserve the expected observation/image-patch layout")
        context = context.detach().float().cpu().reshape(self.steps, patches, channels)
        if not torch.isfinite(update).all() or not torch.isfinite(context).all():
            raise ValueError("Encoder produced non-finite residual updates or context")
        result = {"context": context, "ca_update": update,
                  "alpha": (self._alpha if self._alpha is not None
                             else torch.ones(self.steps, 1, 1))}
        return result


def applied_ca_update(branch):
    """Return the alpha-scaled CA update captured from the encoder."""
    return branch["ca_update"]


@torch.inference_mode()
def capture_pair(policy, obs, capture=None, *, baseline="raw_zero", save_vectors=False):
    """Compute vector differences before L2, keeping the complete zero-force path."""
    if baseline not in {"raw_zero", "normalized_zero"}:
        raise ValueError(f"Unsupported wrench baseline {baseline!r}")
    if capture is None:
        with WrenchEffectCapture(policy) as active:
            return capture_pair(policy, obs, active, baseline=baseline, save_vectors=save_vectors)
    full = capture.encode(obs)
    zero_obs = make_zero_wrench_obs(policy, obs) if baseline == "raw_zero" else obs
    zero = capture.encode(zero_obs, normalized_zero=baseline == "normalized_zero")
    _, normalized, contact, future = policy.prepare_context_inputs(obs)
    if baseline == "raw_zero":
        _, normalized_baseline, _, _ = policy.prepare_context_inputs(zero_obs)
    else:
        normalized_baseline = torch.zeros_like(normalized)
    full_update, zero_update = applied_ca_update(full), applied_ca_update(zero)
    delta_u = full_update - zero_update
    delta_z = full["context"] - zero["context"]
    pair = {
        "ca_update": torch.linalg.vector_norm(full_update, dim=-1).numpy(),
        "baseline_ca_update": torch.linalg.vector_norm(zero_update, dim=-1).numpy(),
        "injection_effect": torch.linalg.vector_norm(delta_u, dim=-1).numpy(),
        "context_effect": torch.linalg.vector_norm(delta_z, dim=-1).numpy(),
        "normalized_wrench": normalized[0].detach().float().cpu().numpy(),
        "baseline_normalized_wrench": normalized_baseline[0].detach().float().cpu().numpy(),
        "contact": None if contact is None else contact[0].detach().cpu().numpy(),
        "future_has_contact": None if future is None else future[0].detach().cpu().numpy(),
    }
    if save_vectors:
        pair.update(ca_update_vector=full_update.numpy(),
                    baseline_ca_update_vector=zero_update.numpy(),
                    delta_u_vector=delta_u.numpy(), delta_z_vector=delta_z.numpy())
    return pair


def effect_color_bounds(records, metric, percentiles=(2, 98)):
    """One absolute feature scale per metric across frames, cameras and obs slots."""
    low_p, high_p = map(float, percentiles)
    if not 0 <= low_p < high_p <= 100:
        raise ValueError("Percentiles must satisfy 0 <= LOW < HIGH <= 100")
    arrays = [np.asarray(camera[metric], dtype=np.float32).reshape(-1)
              for record in records for camera in record["cameras"].values()]
    if not arrays or any(values.size == 0 for values in arrays):
        raise ValueError("Cannot define a scale without patch effect values")
    values = np.concatenate(arrays)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Patch effect magnitudes must be finite and nonnegative")
    low, high = map(float, np.percentile(values, (low_p, high_p)))
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        low, high = 0.0, high if high > 0 else 1.0
    return low, high


def image_color_bounds(values, percentiles=(2, 98)):
    """Independent patch percentiles for each observation image, never cameras/time."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not values.size:
        raise ValueError("Per-image scaling requires nonempty [observations, patches] values")
    return np.asarray([
        effect_color_bounds([{"cameras": {"image": {"value": row}}}], "value", percentiles)
        for row in values
    ], dtype=np.float64)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoints = parser.add_mutually_exclusive_group(required=True)
    checkpoints.add_argument("--checkpoint", type=Path)
    checkpoints.add_argument("--checkpoint-dir", type=Path)
    checkpoints.add_argument("--from-cache", type=Path,
                             help="Render cached metrics from *_raw.npz without loading a teacher.")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--baseline", choices=("raw_zero", "normalized_zero"), default="raw_zero",
                        help="Zero raw wrench+phase inputs, or only zero encoder normalized wrench (preserving preprocessing).")
    parser.add_argument("--metrics", nargs="+", choices=tuple(METRIC_LABELS),
                        default=("ca_update", "injection_effect"))
    parser.add_argument("--camera", choices=("both", "side", "wrist"), default="both")
    parser.add_argument("--observation", choices=("last", "all"), default="last")
    parser.add_argument("--color-scope", choices=("image", "episode"), default="image",
                        help="Per-camera/per-observation patch percentiles, or a shared episode scale.")
    parser.add_argument("--attention-percentiles", "--percentiles", nargs=2, type=float,
                        default=(2.0, 98.0), metavar=("LOW", "HIGH"))
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32",
                        help="FP32 avoids cancellation errors in paired feature differences.")
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav")
    parser.add_argument("--video-codec", choices=("h264", "av1", "mpeg4"), default="h264")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--save-vectors", action="store_true", help="Also cache per-patch U, U0, delta-U, delta-Z vectors (large).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.metrics = tuple(dict.fromkeys(args.metrics))
    if args.from_cache is None and "action_sensitivity" in args.metrics:
        parser.error("Use visualize_wrench_action_sensitivity_episode.py to acquire action sensitivity, or --from-cache to render it")
    if args.episode < 0 or args.stride < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("Episode must be nonnegative; stride and max-frames must be positive")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("FPS must be positive and finite")
    if not math.isfinite(args.alpha) or not 0 <= args.alpha <= 1:
        parser.error("Alpha must be between zero and one")
    low, high = args.attention_percentiles
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high <= 100):
        parser.error("Percentiles must satisfy 0 <= LOW < HIGH <= 100")
    return args


def output_paths(checkpoint, args):
    directory = (args.output_dir.expanduser().resolve() if args.output_dir is not None else
                 checkpoint.parent.parent / "attention_visualize")
    stem = f"wrench_effect_episode_{args.episode}"
    if args.camera != "both":
        stem += f"_{args.camera}"
    if args.baseline != "raw_zero":
        stem += f"_{args.baseline}"
    return {**{metric: directory / f"{stem}_{metric}.mp4" for metric in args.metrics},
            "raw": directory / f"{stem}_raw.npz",
            "metadata": directory / f"{stem}_metadata.json"}


def _write(image, text, x, y, scale=0.43):
    available = max(1, image.shape[1] - x - 8)
    text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
    scale *= min(1.0, available / max(1, text_width))
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (245, 245, 245), 1, cv2.LINE_AA)


def render_frame(record, frame_index, camera_keys, bounds, args, metric, timeline, chart):
    rows = []
    count = record["cameras"][camera_keys[0]][metric].shape[0]
    for obs_index in (range(count) if args.observation == "all" else [count - 1]):
        panels = []
        for key in camera_keys:
            camera = record["cameras"][key]
            local = args.color_scope == "image"
            panel_bounds = tuple(camera[metric + "_color_bounds"][obs_index]) if local else bounds
            values = camera[metric][obs_index]
            flat = bool(values.max() == values.min())
            display_values = np.full_like(values, panel_bounds[0]) if local and flat else values
            image = heat_overlay(camera["images"][obs_index], display_values,
                                 camera["grid"], panel_bounds, args.alpha)
            title = np.full((24, image.shape[1], 3), 20, dtype=np.uint8)
            _write(title, f"{key} | observation {obs_index + 1}", 7, 17)
            panel = np.concatenate((title, image), axis=0)
            if local:
                bar = np.full((45, panel.shape[1], 3), 18, dtype=np.uint8)
                gradient = np.linspace(0, 255, panel.shape[1] - 24, dtype=np.uint8)[None]
                bar[5:17, 12:-12] = cv2.cvtColor(cv2.applyColorMap(gradient, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                low, high = panel_bounds
                _write(bar, f"{low:.5g}", 9, 36, 0.35)
                if flat:
                    _write(bar, "flat", max(60, panel.shape[1] // 2 - 12), 36, 0.35)
                _write(bar, f"{high:.5g}", max(100, panel.shape[1] - 75), 36, 0.35)
                panel = np.concatenate((panel, bar), axis=0)
            panels.append(panel)
        rows.append(np.concatenate(panels, axis=1))
    images = np.concatenate(rows, axis=0)
    if len(camera_keys) == 1 and images.shape[1] < 512:
        images = cv2.resize(images, (512, round(images.shape[0] * 512 / images.shape[1])),
                            interpolation=cv2.INTER_CUBIC)
    width = images.shape[1]
    banner = np.full((56, width, 3), 17, dtype=np.uint8)
    _write(banner, f"{METRIC_LABELS[metric][0]} | ep {args.episode} | {frame_index + 1}/{len(timeline['anchor_times'])}", 9, 20)
    _write(banner, f"{METRIC_LABELS[metric][1]} | anchor {record['anchor']} | baseline {args.baseline}", 9, 44)
    p_low, p_high = args.attention_percentiles
    units = "epsilon L2 units" if metric == "action_sensitivity" else "feature L2 units"
    if args.color_scope == "image":
        scale = np.full((44, width, 3), 18, dtype=np.uint8)
        _write(scale, f"Each image: own p{p_low:g}-p{p_high:g} | {units}", 10, 17)
        _write(scale, "Colors show within-image contrast; compare values using panel scales", 10, 36, 0.38)
    else:
        scale = np.full((62, width, 3), 18, dtype=np.uint8)
        _write(scale, f"{units} | shared episode p{p_low:g}-p{p_high:g}", 10, 17)
        gradient = np.linspace(0, 255, width - 32, dtype=np.uint8)[None]
        scale[25:39, 16:width - 16] = cv2.cvtColor(cv2.applyColorMap(gradient, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        _write(scale, f"{bounds[0]:.4g}", 16, 56)
        _write(scale, f"{bounds[1]:.4g}", max(100, width - 80), 56)
    frame = np.concatenate((banner, images, scale, chart.render(timeline["anchor_times"][frame_index])), axis=0)
    return np.ascontiguousarray(np.pad(frame, ((0, frame.shape[0] % 2), (0, frame.shape[1] % 2), (0, 0))))


def render_cached_effects(cache_path, args):
    """Render saved patch norms with the original RGB windows and wrench plot.

    Write separate rendering metadata so existing metric videos, raw data and
    acquisition metadata are preserved. No policy or DINO model is loaded.
    """
    cache_path = Path(cache_path).expanduser().resolve()
    stem = cache_path.name.removesuffix("_raw.npz")
    source_metadata = cache_path.with_name(stem + "_metadata.json")
    metadata = json.loads(source_metadata.read_text())
    if args.episode != metadata["episode"]:
        raise ValueError("Requested episode does not match the cache")
    args.baseline = metadata["baseline"]
    checkpoint = Path(metadata["checkpoint"])
    cfg = OmegaConf.load(checkpoint.parent.parent / ".hydra/config.yaml")
    dataset = common.instantiate_episode_dataset(cfg, args.dataset_path, video_backend=args.video_backend)
    writers = {}
    directory = args.output_dir.expanduser().resolve() if args.output_dir else cache_path.parent
    paths = {metric: directory / f"{stem}_{metric}.mp4" for metric in args.metrics}
    paths["metadata"] = directory / f"{stem}_{'_'.join(args.metrics)}_render_metadata.json"
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            anchors = cached["anchors"]
            indices = common.episode_sample_indices(dataset, args.episode)
            lookup = dict(indices)
            selected = list(range(len(anchors)))[::args.stride]
            if args.max_frames is not None:
                selected = selected[:args.max_frames]
            if not selected:
                raise ValueError("No cached frames selected")
            raw_keys = metadata["raw_camera_keys"]
            camera_keys = [key for key in metadata["display_camera_keys"]
                           if args.camera == "both" or key == args.camera]
            if not camera_keys:
                raise ValueError("Requested camera is absent from the cache")
            timeline = {
                "values": cached["recorded_wrench"], "times": cached["wrench_time_sec"],
                "contact": cached["wrench_contact"],
                "contact_intervals_sec": cached["contact_intervals_sec"],
                "anchor_times": cached["episode_time_sec"][selected],
                "metadata": metadata["wrench_plot"],
            }
            contact_rule = metadata["wrench_plot"]["contact"]
            force = timeline["values"][:, contact_rule["force_dims"]]
            timeline["contact_magnitude"] = (
                np.abs(force).sum(axis=-1) if contact_rule["norm"] in {"l1", "sum"}
                else np.linalg.norm(force, axis=-1)
            )
            records = []
            for index in selected:
                anchor = int(anchors[index])
                sample = dataset[lookup[anchor]]
                cameras = {}
                for key in camera_keys:
                    camera_index = raw_keys.index(key)
                    camera = {
                        "images": np.stack([common.image_to_uint8(image) for image in sample["obs"][key]]),
                        "grid": tuple(metadata["camera_patch_grids"][key]),
                    }
                    for metric in args.metrics:
                        camera[metric] = cached[f"camera_{camera_index}_{metric}"][index]
                        camera[metric + "_color_bounds"] = image_color_bounds(camera[metric], args.attention_percentiles)
                    cameras[key] = camera
                records.append({"anchor": anchor, "cameras": cameras})
        bounds = {metric: effect_color_bounds(records, metric, args.attention_percentiles) for metric in args.metrics}
        fps = args.fps if args.fps is not None else metadata["fps"] / args.stride
        width = sum(records[0]["cameras"][key]["images"].shape[2] for key in camera_keys)
        chart = WrenchEpisodePlot(timeline, max(512, width) if len(camera_keys) == 1 else width)
        rendering.prepare_outputs(paths, overwrite=args.overwrite, save_frames=args.save_frames)
        for index, record in enumerate(records):
            for metric in args.metrics:
                frame = render_frame(record, index, camera_keys, bounds[metric], args, metric, timeline, chart)
                if metric not in writers:
                    writers[metric] = common.FFmpegVideoWriter(paths[metric], frame.shape[1], frame.shape[0], fps, args.video_codec)
                writers[metric].write(frame)
                if args.save_frames:
                    target = rendering.frame_directory(paths[metric]) / f"frame_{index:06d}.png"
                    if not cv2.imwrite(str(target), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                        raise RuntimeError(f"Could not write {target}")
        for metric in tuple(writers):
            writers.pop(metric).close()
        result = {
            "source_cache": str(cache_path), "source_metadata": str(source_metadata),
            "teacher_loaded": False, "video_frames": len(records), "fps": fps,
            "anchors": [record["anchor"] for record in records],
            "metrics": {metric: METRIC_LABELS[metric][1] for metric in args.metrics},
            "display_camera_keys": camera_keys, "observation_display": args.observation,
            "color_scope": args.color_scope, "color_percentiles": list(args.attention_percentiles),
            "episode_color_bounds": bounds,
            "image_color_bounds": {
                metric: {key: [record["cameras"][key][metric + "_color_bounds"].tolist() for record in records]
                         for key in camera_keys} for metric in args.metrics},
            "videos": {metric: str(paths[metric]) for metric in args.metrics},
        }
        paths["metadata"].write_text(json.dumps(result, indent=2) + "\n")
        for name, path in paths.items():
            print(f"{name}: {path}", flush=True)
        return paths
    finally:
        for writer in writers.values():
            writer.close()
        dataset.lerobot_dataset.close()


def run(args):
    if args.from_cache is not None:
        return render_cached_effects(args.from_cache, args)
    checkpoint = common.choose_checkpoint(args.checkpoint, args.checkpoint_dir)
    payload, cfg = common.load_checkpoint_payload(checkpoint)
    device = common.resolve_device(args.device, cfg)
    policy = common.instantiate_policy(payload, cfg, device, capture_attention=False).float()
    del payload
    camera_keys = common.display_camera_keys(policy, args.camera)
    all_keys = tuple(policy.obs_encoder.image_keys)
    dataset = common.instantiate_episode_dataset(cfg, args.dataset_path, video_backend=args.video_backend)
    paths = output_paths(checkpoint, args)
    writers = {}
    try:
        selected = common.episode_sample_indices(dataset, args.episode)[::args.stride]
        if args.max_frames is not None:
            selected = selected[:args.max_frames]
        if not selected:
            raise RuntimeError("No episode frames remain after selection")
        dataset_fps = common.infer_dataset_fps(dataset)
        output_fps = args.fps if args.fps is not None else dataset_fps / args.stride
        timeline = load_episode_wrench(dataset, policy, args.episode,
                                       [anchor for anchor, _ in selected], dataset_fps)
        records = []
        with WrenchEffectCapture(policy) as capture:
            # Validate architecture before touching any old output files.
            rendering.prepare_outputs(paths, overwrite=args.overwrite, save_frames=args.save_frames)
            for number, (anchor, sample_index) in enumerate(selected, 1):
                sample = dataset[sample_index]
                obs = {key: value.unsqueeze(0).to(device) for key, value in sample["obs"].items()}
                amp = (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                       if args.precision == "bfloat16" else contextlib.nullcontext())
                with torch.inference_mode(), amp:
                    pair = capture_pair(policy, obs, capture, baseline=args.baseline,
                                        save_vectors=args.save_vectors)
                cameras, offset = {}, 0
                for key in all_keys:
                    patches = int(policy.obs_encoder.patch_counts[key])
                    end = offset + patches
                    if key in camera_keys:
                        images = np.stack([common.image_to_uint8(image)
                                           for image in sample["obs"][key][:policy.n_obs_steps]])
                        camera = {"images": images,
                                  "grid": common.patch_grid(patches, images.shape[1], images.shape[2],
                                                            policy.obs_encoder.backbone)}
                        for metric in (*METRICS, "baseline_ca_update"):
                            camera[metric] = pair[metric][:, offset:end]
                        if args.save_vectors:
                            for metric in ("ca_update_vector", "baseline_ca_update_vector", "delta_u_vector", "delta_z_vector"):
                                camera[metric] = pair[metric][:, offset:end]
                        cameras[key] = camera
                    offset = end
                if offset != pair["ca_update"].shape[1]:
                    raise ValueError("Camera patch counts do not match captured residual token layout")
                records.append({"anchor": anchor, "cameras": cameras,
                                **{key: pair[key] for key in ("normalized_wrench", "baseline_normalized_wrench", "contact", "future_has_contact")}})
                if number == 1 or number % 25 == 0 or number == len(selected):
                    print(f"Captured {number}/{len(selected)} wrench pairs", flush=True)
        episode_bounds = {metric: effect_color_bounds(records, metric, args.attention_percentiles) for metric in args.metrics}
        for record in records:
            for camera in record["cameras"].values():
                for metric in args.metrics:
                    camera[metric + "_color_bounds"] = image_color_bounds(camera[metric], args.attention_percentiles)
        bounds = (episode_bounds if args.color_scope == "episode" else {
            metric: {key: [record["cameras"][key][metric + "_color_bounds"].tolist()
                           for record in records] for key in all_keys if key in camera_keys}
            for metric in args.metrics
        })
        width = sum(records[0]["cameras"][key]["images"].shape[2] for key in camera_keys)
        chart = WrenchEpisodePlot(timeline, max(512, width) if len(camera_keys) == 1 else width)
        for index, record in enumerate(records):
            for metric in args.metrics:
                frame = render_frame(record, index, camera_keys, episode_bounds[metric], args, metric, timeline, chart)
                if metric not in writers:
                    writers[metric] = common.FFmpegVideoWriter(paths[metric], frame.shape[1], frame.shape[0],
                                                              output_fps, args.video_codec)
                writers[metric].write(frame)
                if args.save_frames:
                    target = rendering.frame_directory(paths[metric]) / f"frame_{index:06d}.png"
                    if not cv2.imwrite(str(target), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                        raise RuntimeError(f"Could not write frame {target}")
        for metric in tuple(writers):
            writers.pop(metric).close()
        arrays = {
            "anchors": np.asarray([record["anchor"] for record in records], dtype=np.int64),
            "episode_time_sec": timeline["anchor_times"],
            "recorded_wrench": timeline["values"], "wrench_time_sec": timeline["times"],
            "wrench_contact": timeline["contact"], "wrench_causal_contact": timeline["causal_contact"],
            "frame_wrench": timeline["values"][timeline["anchor_offsets"]],
            "frame_contact": timeline["contact"][timeline["anchor_offsets"]],
            "contact_intervals_sec": timeline["contact_intervals_sec"],
        }
        for field in ("normalized_wrench", "baseline_normalized_wrench", "contact", "future_has_contact"):
            if records[0][field] is not None:
                arrays["model_" + field] = np.stack([record[field] for record in records])
        raw_camera_keys = [key for key in all_keys if key in camera_keys]
        for camera_index, key in enumerate(raw_camera_keys):
            fields = (*METRICS, "baseline_ca_update")
            if args.save_vectors:
                fields += ("ca_update_vector", "baseline_ca_update_vector", "delta_u_vector", "delta_z_vector")
            for field in fields:
                arrays[f"camera_{camera_index}_{field}"] = np.stack([record["cameras"][key][field] for record in records])
            for metric in args.metrics:
                arrays[f"camera_{camera_index}_{metric}_color_bounds"] = np.stack([
                    record["cameras"][key][metric + "_color_bounds"] for record in records
                ])
        np.savez_compressed(paths["raw"], **arrays)
        offsets = OmegaConf.select(cfg, f"task.dataset.windowed_lowdim_feature_groups.{policy.wrench_key}.offsets_sec", default=None)
        metadata = {
            "checkpoint": str(checkpoint), "episode": args.episode,
            "video_frames": len(records), "fps": output_fps, "dataset_fps": dataset_fps,
            "stride": args.stride, "precision": args.precision, "diffusion_sampling": False,
            "metrics": {
                "ca_update": "L2(U(W)) after MHA output projection, before query residual/FFN; includes the zero-wrench encoder bias/position response",
                "injection_effect": "L2(U(W)-U(W0)); subtract vectors before channel L2",
                "context_effect": "L2(Z(W)-Z(W0)); final obs_encoder image tokens after FFN and context normalization",
                "baseline_ca_update": "L2(U(W0)), retained to identify the zero-wrench branch response",
            },
            "baseline": args.baseline,
            "baseline_definition": (
                "All required raw wrench/phase keys are set to zero; both branches use the same complete preprocessing, force encoder and fusion. Contact and future-prefix fill may change."
                if args.baseline == "raw_zero" else
                "Original raw inputs and phase preprocessing are preserved; only normalized wrench_history at the learned encoder entrance is replaced by zero."
            ),
            "wrench_keys_zeroed": list(dict.fromkeys(key for key in
                (policy.wrench_key, policy.phase_wrench_key, policy.future_phase_wrench_key) if key is not None)) if args.baseline == "raw_zero" else [],
            "wrench_key": policy.wrench_key, "phase_wrench_key": policy.phase_wrench_key,
            "future_phase_wrench_key": policy.future_phase_wrench_key,
            "model_wrench_offsets_sec": None if offsets is None else [float(value) for value in offsets],
            "same_rgb": True, "curriculum_blur": False,
            "all_camera_keys": list(all_keys), "display_camera_keys": camera_keys, "raw_camera_keys": raw_camera_keys,
            "observation_display": args.observation,
            "camera_patch_grids": {key: list(records[0]["cameras"][key]["grid"]) for key in raw_camera_keys},
            "color_percentiles": list(args.attention_percentiles), "color_bounds": bounds,
            "color_scope": args.color_scope,
            "episode_color_bounds": episode_bounds,
            "color_bounds_scope": (
                "Independent patch percentiles for every camera, observation slot and frame, per metric; colors are not comparable across images. Constant maps use uniform cold color and a flat label."
                if args.color_scope == "image" else
                "One independent scale per metric, shared across all captured episode anchors, selected cameras, observation slots and patches"
            ),
            "units": "Learned feature L2 units; not percentages or physical force units",
            "wrench_plot": timeline["metadata"],
            "interpretation": "Model-internal residual magnitude and paired wrench-conditioned feature change; not physical contact location or final-action importance. Current physical wrench curves may precede/follow maps because the model uses the recorded future wrench window.",
            "save_vectors": args.save_vectors,
            "videos": {metric: str(paths[metric]) for metric in args.metrics},
        }
        paths["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        for metric in args.metrics:
            print(f"{metric}: {paths[metric]}", flush=True)
        print(f"Raw: {paths['raw']}\nMetadata: {paths['metadata']}", flush=True)
        return paths
    finally:
        for writer in writers.values():
            writer.close()
        dataset.lerobot_dataset.close()


if __name__ == "__main__":
    run(parse_args())
