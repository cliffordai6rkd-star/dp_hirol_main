"""Render episode video with denoising-averaged image heatmaps and route curves.

Each video frame represents one observation anchor at the dataset frame rate.
The upper panels average the eight (or configured number of) denoiser calls
into a percentage heatmap on each input image, with a shared episode-wide
percentile color scale across selected cameras and observation times (default
p1/p99). All configured cameras remain model inputs. The lower curves track
the selected image/wrench routing percentages across
physical episode time; a vertical
cursor marks the current video frame. Attention rollout is a routing
diagnostic, not final-action attribution. In image-query CA without preceding
joint SA, equal-residual rollout fixes the total image/wrench route split at
50/50 by construction; selecting one camera excludes the other image route.
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
import numpy as np
import torch

from diffusion_policy.scripts import visualize_cross_attention_episode as common
from diffusion_policy.scripts import visualize_image_query_cross_attention_episode as rendering
from diffusion_policy.scripts.dit_action_attention_capture import DitActionAttentionCapture
from diffusion_policy.scripts.fusion_visual_attention_trace import FusionVisualAttentionTrace


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--checkpoint", type=pathlib.Path)
    checkpoint.add_argument("--checkpoint-dir", type=pathlib.Path)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--dataset-path", type=pathlib.Path)
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fps", type=float, help="Output FPS; default follows the dataset FPS divided by --stride.")
    parser.add_argument("--video-codec", choices=("av1", "h264", "mpeg4"), default="h264")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--camera", choices=("wrist", "side", "both"), default="both",
                        help="Cameras included in attention statistics and display; inference still uses all cameras.")
    parser.add_argument("--attention-percentiles", nargs=2, type=float,
                        metavar=("LOW", "HIGH"), default=(1.0, 99.0),
                        help="Shared episode heatmap display percentiles (default: 1 99).")
    parser.add_argument("--heatmap-normalization", choices=("absolute", "image_relative"),
                        default="absolute",
                        help="Display absolute route percentages or renormalize selected image patches to 100%%.")
    parser.add_argument("--color-bounds", nargs=2, type=float, metavar=("LOW", "HIGH"),
                        help="Fixed heatmap color bounds in percent; otherwise compute episode percentiles.")
    parser.add_argument("--concentration-top-k", type=float, default=0.10,
                        help="Fraction of patches used for top-k concentration metrics (default: 0.10).")
    parser.add_argument("--observation", choices=("last", "all"), default="last")
    parser.add_argument("--action-index", type=int, help="Absolute action position; default averages the executed chunk.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--save-frames", action="store_true")
    args = parser.parse_args(argv)
    if args.episode < 0 or args.stride < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("--episode must be nonnegative; --stride and --max-frames must be positive")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps must be positive and finite")
    if not math.isfinite(args.alpha) or not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")
    low, high = args.attention_percentiles
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high <= 100):
        parser.error("--attention-percentiles must satisfy 0 <= LOW < HIGH <= 100")
    if args.color_bounds is not None:
        low, high = args.color_bounds
        if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high):
            parser.error("--color-bounds must satisfy 0 <= LOW < HIGH")
    if not math.isfinite(args.concentration_top_k) or not 0 < args.concentration_top_k <= 1:
        parser.error("--concentration-top-k must satisfy 0 < value <= 1")
    return args


def fusion_label(trace):
    modes = []
    if trace.cross_attention:
        modes.append("cross")
    if trace.self_attention:
        modes.append("self")
    if not modes:
        raise ValueError("This checkpoint has no fusion cross-attention or self-attention to visualize")
    return "+".join(modes)


def output_paths(checkpoint, episode, label, camera="both", normalization="absolute"):
    directory = checkpoint.parent.parent / "attention_visualize"
    stem = f"denoising_{label.replace('+', '_')}_episode_{episode}"
    if camera != "both":
        stem += f"_{camera}"
    if normalization != "absolute":
        stem += f"_{normalization}"
    return {
        "video": directory / f"{stem}.mp4",
        "raw": directory / f"{stem}_raw.npz",
        "metadata": directory / f"{stem}_metadata.json",
    }


def routing_percentages(context, trace, camera_keys=None):
    """Route every DiT call back to encoder inputs, keeping its own denominator."""
    maps, diagnostics = trace.project(context)
    total = diagnostics["input_mass"].numpy()
    if not np.isfinite(total).all() or (total <= 0).any():
        raise ValueError("DiT context routing must have positive finite mass at every denoising step")
    keys = tuple(maps) if camera_keys is None else camera_keys
    camera_maps = {key: maps[key].numpy() * (100.0 / total)[:, None, None]
                   for key in keys}
    all_visual = diagnostics["visual_mass"].numpy() * 100.0 / total
    wrench = diagnostics["wrench_mass"].numpy() * 100.0 / total
    if not np.allclose(all_visual + wrench, 100.0, atol=0.01):
        raise RuntimeError("Image and wrench routing percentages do not sum to 100")
    visual = (all_visual if len(camera_maps) == len(maps) else
              sum(values.sum(axis=(-2, -1)) for values in camera_maps.values()))
    return camera_maps, visual, wrench, total


def query_slot_routing_percentages(context, trace, camera_keys=None):
    """Map DiT relevance to image-query output slots for residual-free CA.

    Image-query cross-attention without a query residual has no linear value
    path from the output tokens back to the image inputs: image information
    only changes the attention weights.  The ordinary residual-aware rollout
    therefore reports zero image mass.  This diagnostic keeps the DiT
    action-to-context relevance and assigns each output image-query slot to
    its corresponding image patch position.  It is deliberately reported as
    query-slot attribution rather than input-token value-path routing.
    """
    if not (trace.cross_attention and trace.cross_query == "image"
            and not trace.cross_query_residual and not trace.self_attention):
        raise ValueError("Query-slot attribution requires residual-free image-query cross-attention")
    relevance = torch.as_tensor(context).detach().float().cpu()
    if relevance.ndim != 2 or relevance.shape[-1] != trace.context_tokens:
        raise ValueError(
            f"context must have shape [denoising_calls, {trace.context_tokens}]"
        )
    total = relevance.sum(-1).numpy()
    if not np.isfinite(total).all() or (total <= 0).any():
        raise ValueError("DiT context routing must have positive finite mass at every denoising step")
    query = relevance.reshape(relevance.shape[0], trace.observation_steps, trace.image_token_count)
    keys = tuple(trace.image_keys) if camera_keys is None else tuple(camera_keys)
    if not keys or any(key not in trace.image_keys for key in keys):
        raise ValueError("Query-slot attribution requires configured image cameras")
    maps = {}
    offset = 0
    scale = 100.0 / total
    for key in trace.image_keys:
        end = offset + trace.patch_counts[key]
        if key in keys:
            maps[key] = query[..., offset:end].numpy() * scale[:, None, None]
        offset = end
    visual = sum(values.sum(axis=(-2, -1)) for values in maps.values())
    wrench = np.zeros_like(visual)
    return maps, visual, wrench, total


def heatmap_color_bounds(records, percentiles=(1.0, 99.0)):
    """Set one episode scale after averaging each patch over denoising calls."""
    averaged = [{"cameras": {key: {"percent": camera["percent"].mean(axis=0)}
                             for key, camera in record["cameras"].items()}}
                for record in records]
    return rendering.percentile_bounds(averaged, "percent", percentiles)


def image_relative_records(records):
    """Renormalize selected image routes to 100% per denoising call.

    The absolute maps retain the all-condition denominator. This display-only
    transform removes the fixed 0.5 image residual factor in image-query CA so
    concentration of visual routing can be compared with image-only models.
    """
    normalized = []
    for record in records:
        cameras = record["cameras"]
        if not cameras:
            raise ValueError("Image-relative normalization requires at least one camera")
        denominators = None
        for camera in cameras.values():
            values = np.asarray(camera["percent"], dtype=np.float64)
            if values.ndim != 3 or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Camera route percentages must be finite, nonnegative [steps, obs, patches]")
            contribution = values.sum(axis=(-2, -1))
            denominators = contribution if denominators is None else denominators + contribution
        if (denominators <= 0).any() or not np.isfinite(denominators).all():
            raise ValueError("Image route mass must be positive and finite for image-relative normalization")
        copied = dict(record)
        copied["cameras"] = {}
        for key, camera in cameras.items():
            values = np.asarray(camera["percent"], dtype=np.float32)
            copied_camera = dict(camera)
            copied_camera["percent"] = values * (100.0 / denominators).astype(np.float32)[:, None, None]
            copied["cameras"][key] = copied_camera
        copied["visual"] = np.full_like(np.asarray(record["visual"], dtype=np.float32), 100.0)
        normalized.append(copied)
    return normalized


def _distribution_metrics(values, top_k_fraction):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("Concentration values must be a nonempty vector or matrix")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Concentration values must be finite and nonnegative")
    totals = values.sum(axis=-1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("Concentration values must have positive mass")
    probabilities = values / totals
    entropy = -(probabilities * np.log(np.maximum(probabilities, np.finfo(float).tiny))).sum(axis=-1)
    entropy /= math.log(values.shape[1]) if values.shape[1] > 1 else 1.0
    count = max(1, int(math.ceil(float(top_k_fraction) * values.shape[1])))
    top_k = np.partition(probabilities, -count, axis=-1)[:, -count:].sum(axis=-1) * 100.0
    return entropy.astype(np.float32), top_k.astype(np.float32)


def concentration_metrics(records, camera_keys, top_k_fraction=0.10):
    """Compute visual concentration after averaging denoising calls.

    Global metrics normalize all selected camera/observation patches together;
    per-camera metrics normalize each camera separately, so camera allocation
    and within-camera spatial concentration remain distinguishable.
    """
    if not camera_keys:
        raise ValueError("Concentration metrics require at least one camera")
    global_entropy, global_top = [], []
    per_camera = {key: {"entropy": [], "top_k_mass_percent": []} for key in camera_keys}
    for record in records:
        averaged = {key: np.asarray(record["cameras"][key]["percent"], dtype=np.float64).mean(axis=0)
                    for key in camera_keys}
        global_values = np.concatenate([averaged[key].reshape(-1) for key in camera_keys])
        entropy, top = _distribution_metrics(global_values, top_k_fraction)
        global_entropy.append(float(entropy[0]))
        global_top.append(float(top[0]))
        for key in camera_keys:
            entropy, top = _distribution_metrics(averaged[key].reshape(-1), top_k_fraction)
            per_camera[key]["entropy"].append(float(entropy[0]))
            per_camera[key]["top_k_mass_percent"].append(float(top[0]))
    result = {
        "global": {"entropy": np.asarray(global_entropy, dtype=np.float32),
                   "top_k_mass_percent": np.asarray(global_top, dtype=np.float32)},
        "cameras": {key: {name: np.asarray(values, dtype=np.float32)
                           for name, values in metrics.items()}
                    for key, metrics in per_camera.items()},
    }
    return result


def concentration_metadata(metrics, top_k_fraction):
    def describe(values):
        values = np.asarray(values, dtype=np.float32)
        return {"per_frame": values.tolist(), "mean": float(values.mean()),
                "median": float(np.median(values))}

    return {
        "definition": "Denoising-averaged nonnegative image route distribution; lower entropy and higher top-k mass mean more concentration.",
        "top_k_fraction": float(top_k_fraction),
        "global": {key: describe(value) for key, value in metrics["global"].items()},
        "cameras": {camera: {key: describe(value) for key, value in values.items()}
                    for camera, values in metrics["cameras"].items()},
    }


def heat_overlay(image, patch_percent, grid, color_bounds, alpha):
    values = np.asarray(patch_percent, dtype=np.float32).reshape(grid)
    low, high = color_bounds
    scaled = np.clip((values - low) / (high - low), 0, 1)
    scaled = cv2.resize(scaled, (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_CUBIC).clip(0, 1)
    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(scaled * 255), cv2.COLORMAP_JET),
                           cv2.COLOR_BGR2RGB)
    # Match wrench_image_self_attn_episode_50.mp4: JET overlay at fixed alpha.
    return np.ascontiguousarray(np.clip(image * (1 - alpha) + heatmap * alpha, 0, 255).astype(np.uint8))


def _write(frame, message, x, y, scale=0.45, color=(245, 245, 245)):
    cv2.putText(frame, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def render_chart(visual, wrench, episode_times, selected, width, image_label="image"):
    height = 245
    chart = np.full((height, width, 3), 18, dtype=np.uint8)
    _write(chart, "Episode route (%)", 16, 23, 0.46)
    left, right, top, bottom = 52, width - 24, 46, height - 37
    if right <= left:
        raise ValueError("Camera panels are too narrow for the routing chart")
    count = len(visual)
    duration = float(episode_times[-1] - episode_times[0])
    xs = (left + (np.asarray(episode_times) - episode_times[0]) *
          (right - left) / duration).round().astype(int) if duration > 0 else np.full(count, left, dtype=int)
    ys = lambda values: np.rint(bottom - np.asarray(values) * (bottom - top) / 100).astype(int)
    for value in (0, 25, 50, 75, 100):
        y = int(ys([value])[0])
        cv2.line(chart, (left, y), (right, y), (53, 53, 53), 1)
        _write(chart, str(value), 10, y + 4, 0.38, (175, 175, 175))
    cv2.line(chart, (left, top), (left, bottom), (170, 170, 170), 1)
    image_points = np.stack((xs, ys(visual)), axis=1).astype(np.int32)
    wrench_points = np.stack((xs, ys(wrench)), axis=1).astype(np.int32)
    if count > 1:
        cv2.polylines(chart, [image_points], False, (255, 220, 65), 2, cv2.LINE_AA)
        # The dashed line leaves both routes visible when both equal 50%.
        for start, end in zip(wrench_points[:-1], wrench_points[1:]):
            for fraction in np.arange(0.0, 1.0, 0.16):
                first = start + (end - start) * fraction
                last = start + (end - start) * min(fraction + 0.08, 1.0)
                cv2.line(chart, tuple(np.rint(first).astype(int)),
                         tuple(np.rint(last).astype(int)), (105, 225, 255), 2, cv2.LINE_AA)
    for x, y in image_points:
        cv2.circle(chart, (int(x), int(y)), 3, (255, 220, 65), -1, cv2.LINE_AA)
    for x, y in wrench_points:
        cv2.circle(chart, (int(x), int(y)), 5, (105, 225, 255), 1, cv2.LINE_AA)
    cv2.line(chart, (int(xs[selected]), top), (int(xs[selected]), bottom),
             (255, 255, 255), 1, cv2.LINE_AA)
    _write(chart, image_label, max(left, width - 230), 22, 0.42, (255, 220, 65))
    _write(chart, "wrench", max(left, width - 102), 22, 0.42, (105, 225, 255))
    for fraction in np.linspace(0, 1, 5):
        x = int(left + fraction * (right - left))
        precision = 2 if duration < 1 else 1
        _write(chart, f"{fraction * duration:.{precision}f}s", min(x - 8, width - 35),
               bottom + 17, 0.35, (180, 180, 180))
    _write(chart, f"episode {episode_times[selected] - episode_times[0]:.2f}s | "
                  f"{image_label} {visual[selected]:.2f}% | wrench {wrench[selected]:.2f}%",
           left, height - 5, 0.42)
    return chart


def render_frame(record, frame_index, camera_keys, color_bounds, args, label,
                 visual_curve, wrench_curve, episode_times):
    rows = []
    observation_indices = (range(len(record["cameras"][camera_keys[0]]["images"]))
                           if args.observation == "all" else [len(record["cameras"][camera_keys[0]]["images"]) - 1])
    for obs_index in observation_indices:
        panels = []
        for key in camera_keys:
            camera = record["cameras"][key]
            panel = heat_overlay(camera["images"][obs_index],
                                 camera["percent"].mean(axis=0)[obs_index], camera["grid"],
                                 color_bounds, args.alpha)
            title = np.full((24, panel.shape[1], 3), 23, dtype=np.uint8)
            _write(title, f"{key} | observation {obs_index + 1}", 7, 17, 0.42)
            panels.append(np.concatenate((title, panel), axis=0))
        rows.append(np.concatenate(panels, axis=1))
    images = np.concatenate(rows, axis=0)
    if len(camera_keys) == 1 and images.shape[1] < 512:
        # Keep the chart and annotations readable for a single camera panel.
        height = round(images.shape[0] * 512 / images.shape[1])
        images = cv2.resize(images, (512, height), interpolation=cv2.INTER_CUBIC)
    width = images.shape[1]
    banner = np.full((34, width, 3), 17, dtype=np.uint8)
    _write(banner, f"{label.upper()} attention | ep {args.episode} | {frame_index + 1}/{len(visual_curve)} "
                   f"| anchor {record['anchor']}", 9, 23, 0.46)
    scale = np.full((56, width, 3), 18, dtype=np.uint8)
    low, high = color_bounds
    p_low, p_high = args.attention_percentiles
    normalization = getattr(args, "heatmap_normalization", "absolute")
    bounds_label = (f"fixed {low:g}-{high:g}" if getattr(args, "color_bounds", None) is not None
                    else f"p{p_low:g}-p{p_high:g}")
    _write(scale, f"Action to image patches | {normalization} | denoising-step mean | {bounds_label}",
           10, 17, 0.40)
    gradient = np.linspace(0, 255, width - 32, dtype=np.uint8)[None, :]
    scale[25:39, 16:width - 16] = cv2.cvtColor(
        cv2.applyColorMap(gradient, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    _write(scale, f"{low:.3g}%", 16, 54, 0.37)
    _write(scale, f"{high:.3g}%", width - 78, 54, 0.37)
    image_label = "image" if args.camera == "both" else f"image:{camera_keys[0]}"
    chart = render_chart(visual_curve, wrench_curve, episode_times, frame_index, width, image_label)
    frame = np.concatenate((banner, images, scale, chart), axis=0)
    return np.ascontiguousarray(np.pad(
        frame, ((0, frame.shape[0] % 2), (0, frame.shape[1] % 2), (0, 0))))


def run(args):
    checkpoint = common.choose_checkpoint(args.checkpoint, args.checkpoint_dir)
    payload, cfg = common.load_checkpoint_payload(checkpoint)
    device = common.resolve_device(args.device, cfg)
    policy = common.instantiate_policy(payload, cfg, device, capture_attention=False)
    del payload
    fusion = FusionVisualAttentionTrace(policy)
    query_slot_attribution = bool(
        fusion.cross_attention and getattr(fusion, "cross_query", None) == "image"
        and not getattr(fusion, "cross_query_residual", True) and not fusion.self_attention
    )
    label = fusion_label(fusion)
    dit = DitActionAttentionCapture(policy, action_index=args.action_index)
    camera_keys = common.display_camera_keys(policy, args.camera)
    attention_camera_keys = [key for key in fusion.image_keys if key in camera_keys]
    paths = output_paths(checkpoint, args.episode, label, args.camera, args.heatmap_normalization)
    if query_slot_attribution:
        print(
            "Using DiT action-to-image-query slot attribution for residual-free "
            "image-query cross-attention; this is not value-path input routing.",
            flush=True,
        )
    dataset = common.instantiate_episode_dataset(cfg, args.dataset_path, video_backend=args.video_backend)
    writer = None
    try:
        selected = common.episode_sample_indices(dataset, args.episode)[::args.stride]
        if args.max_frames is not None:
            selected = selected[:args.max_frames]
        if not selected:
            raise RuntimeError("No episode frames remain after filtering")
        dataset_fps = common.infer_dataset_fps(dataset)
        output_fps = args.fps if args.fps is not None else dataset_fps / args.stride
        rendering.prepare_outputs(paths, overwrite=args.overwrite, save_frames=args.save_frames)
        records = []
        with fusion, dit:
            for frame_number, (anchor, sample_index) in enumerate(selected, 1):
                sample = dataset[sample_index]
                obs = {key: value.unsqueeze(0).to(device) for key, value in sample["obs"].items()}
                generator = torch.Generator(device=device).manual_seed(args.seed)
                autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                            if device.type == "cuda" else contextlib.nullcontext())
                with torch.inference_mode(), autocast:
                    policy.predict_action(obs, generator=generator)
                context = dit.step_context_relevance()
                if query_slot_attribution:
                    maps, visual, wrench, context_mass = query_slot_routing_percentages(
                        context, fusion, attention_camera_keys
                    )
                else:
                    maps, visual, wrench, context_mass = routing_percentages(
                        context, fusion, attention_camera_keys
                    )
                cameras = {}
                for key in attention_camera_keys:
                    images = np.stack([common.image_to_uint8(image)
                                       for image in sample["obs"][key][:policy.n_obs_steps]])
                    grid = common.patch_grid(fusion.patch_counts[key], images.shape[1], images.shape[2],
                                             policy.obs_encoder.backbone)
                    cameras[key] = {"images": images, "percent": maps[key], "grid": grid}
                records.append({"anchor": anchor, "cameras": cameras, "visual": visual,
                                "wrench": wrench, "context_mass": context_mass,
                                "timesteps": np.asarray(dit.diagnostics()["timesteps"], dtype=np.float64)})
                if frame_number == 1 or frame_number % 25 == 0 or frame_number == len(selected):
                    print(f"Captured {frame_number}/{len(selected)} observation windows", flush=True)
        relative_records = image_relative_records(records)
        display_records = (relative_records if args.heatmap_normalization == "image_relative" else records)
        color_bounds = (tuple(args.color_bounds) if args.color_bounds is not None else
                        heatmap_color_bounds(display_records, args.attention_percentiles))
        concentration = concentration_metrics(records, attention_camera_keys,
                                              args.concentration_top_k)
        visual_curve = np.asarray([record["visual"].mean() for record in records])
        wrench_curve = np.asarray([record["wrench"].mean() for record in records])
        episode_times = np.asarray([record["anchor"] for record in records], dtype=np.float64) / dataset_fps
        for frame_index, record in enumerate(records):
            frame = render_frame(display_records[frame_index], frame_index, camera_keys, color_bounds, args, label,
                                 visual_curve, wrench_curve, episode_times)
            if writer is None:
                writer = common.FFmpegVideoWriter(paths["video"], frame.shape[1], frame.shape[0],
                                                   output_fps, args.video_codec)
            writer.write(frame)
            if args.save_frames:
                path = rendering.frame_directory(paths["video"]) / f"frame_{frame_index:06d}.png"
                if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError(f"Could not save frame: {path}")
        closing_writer, writer = writer, None
        closing_writer.close()
        arrays = {"anchors": np.asarray([record["anchor"] for record in records], dtype=np.int64),
                  "timesteps": np.stack([record["timesteps"] for record in records]),
                  "visual_percent": np.stack([record["visual"] for record in records]),
                  "wrench_percent": np.stack([record["wrench"] for record in records]),
                  "visual_episode_percent": visual_curve, "wrench_episode_percent": wrench_curve,
                  "episode_time_sec": episode_times - episode_times[0],
                  "context_mass": np.stack([record["context_mass"] for record in records])}
        for index, key in enumerate(attention_camera_keys):
            absolute = np.stack([record["cameras"][key]["percent"]
                                                           for record in records])
            relative = np.stack([record["cameras"][key]["percent"]
                                 for record in relative_records])
            arrays[f"camera_{index}_percent"] = absolute
            arrays[f"camera_{index}_episode_percent"] = absolute.mean(axis=1)
            arrays[f"camera_{index}_image_relative_percent"] = relative
            arrays[f"camera_{index}_image_relative_episode_percent"] = relative.mean(axis=1)
            arrays[f"camera_{index}_entropy"] = concentration["cameras"][key]["entropy"]
            arrays[f"camera_{index}_top_k_mass_percent"] = concentration["cameras"][key]["top_k_mass_percent"]
        arrays["image_entropy"] = concentration["global"]["entropy"]
        arrays["image_top_k_mass_percent"] = concentration["global"]["top_k_mass_percent"]
        np.savez_compressed(paths["raw"], **arrays)
        color_scale = (f"fixed {color_bounds[0]:g} to {color_bounds[1]:g}"
                       if args.color_bounds is not None else
                       f"shared p{args.attention_percentiles[0]:g} to p{args.attention_percentiles[1]:g}")
        metadata = {
            "checkpoint": str(checkpoint), "episode": args.episode, "fusion_label": label,
            "inference_steps": int(len(records[0]["timesteps"])), "observation_windows": len(records),
            "video_frames": len(records), "fps": output_fps, "dataset_fps": dataset_fps,
            "video_codec": args.video_codec,
            "all_camera_keys": list(fusion.image_keys), "display_camera_keys": camera_keys,
            "camera_selection": args.camera, "attention_camera_keys": attention_camera_keys,
            "raw_camera_keys": attention_camera_keys,
            "camera_patch_grids": {key: list(records[0]["cameras"][key]["grid"]) for key in attention_camera_keys},
            "observation_display": args.observation,
            "route_scope": "all observation frames of selected cameras; wrench includes all wrench inputs",
            "percentage_denominator": (
                "all DiT context relevance mass; image-query output slots are assigned to their corresponding image patches"
                if query_slot_attribution else
                "all routed condition mass, including every model camera and wrench; excluded cameras are not renormalized away"
            ),
            "action_query": "executed chunk" if args.action_index is None else f"action index {args.action_index}",
            "image_heatmap": (
                "Arithmetic mean over denoising calls of DiT action-to-image-query slot relevance"
                if query_slot_attribution else
                ("Arithmetic mean over denoising calls of (100 * image patch route / all routed condition mass at that call)"
                 if args.heatmap_normalization == "absolute" else
                 "Arithmetic mean over denoising calls after normalizing selected image patch routes to 100% per call")
            ),
            "route_curves": (
                "One point per episode observation: mean over denoising calls of selected image-query slot relevance; "
                "curves need not sum to 100 when cameras are excluded"
                if query_slot_attribution else
                "One point per episode observation: mean over denoising calls of (100 * selected-camera image or wrench route / all routed condition mass at that call); curves need not sum to 100 when cameras are excluded"
            ),
            "time_axis": "Episode anchor time in seconds; one video frame per selected observation anchor",
            "heatmap_normalization": args.heatmap_normalization,
            "color_scale": f"{color_scale} of denoising-averaged {args.heatmap_normalization} patch percentages over selected cameras and observation windows/times; display clipped outside bounds",
            "color_percentiles": list(args.attention_percentiles),
            "color_bounds_percent": list(color_bounds),
            "color_bounds_source": "fixed_cli" if args.color_bounds is not None else "episode_percentiles",
            "color_peak_percent": color_bounds[1], "color_map": "JET, matching wrench_image_self_attn episode videos",
            "concentration": concentration_metadata(concentration, args.concentration_top_k),
            "fusion_architecture": fusion.architecture, "dit_diagnostics": dit.diagnostics(),
            "routing_method": (
                "dit_to_image_query_slots"
                if query_slot_attribution else "residual_aware_value_path_rollout"
            ),
            "interpretation": (
                "DiT action-to-image-query slot attribution for residual-free image-query CA; "
                "it visualizes which image-query output slots the denoiser reads and does not "
                "claim a value-path attribution to raw image inputs."
                if query_slot_attribution else
                "Equal-residual attention rollout, not causal final-action attribution. Image-query CA without preceding self-attention fixes the total image/wrench route split at 50/50 by construction; selecting one camera keeps only its share of the total image route."
            ),
            "seed": args.seed, "stride": args.stride,
        }
        paths["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        print(f"Video: {paths['video']}\nRaw percentages: {paths['raw']}\nMetadata: {paths['metadata']}")
        return paths
    finally:
        if writer is not None:
            writer.close()
        dataset.lerobot_dataset.close()


if __name__ == "__main__":
    run(parse_args())
