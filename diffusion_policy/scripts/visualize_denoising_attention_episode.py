"""Render episode video with denoising-averaged image heatmaps and route curves.

Each video frame represents one observation anchor at the dataset frame rate.
The upper panels average the eight (or configured number of) denoiser calls
into a percentage heatmap on each input image, with a shared episode-wide
percentile color scale across cameras and observation times (default p1/p99).
The lower curves track the mean image/wrench routing percentages across
physical episode time; a vertical
cursor marks the current video frame. Attention rollout is a routing
diagnostic, not final-action attribution. In image-query CA without preceding
joint SA, equal-residual rollout fixes the route split at 50/50 by construction.
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
    parser.add_argument("--attention-percentiles", nargs=2, type=float,
                        metavar=("LOW", "HIGH"), default=(1.0, 99.0),
                        help="Shared episode heatmap display percentiles (default: 1 99).")
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


def output_paths(checkpoint, episode, label):
    directory = checkpoint.parent.parent / "attention_visualize"
    stem = f"denoising_{label.replace('+', '_')}_episode_{episode}"
    return {
        "video": directory / f"{stem}.mp4",
        "raw": directory / f"{stem}_raw.npz",
        "metadata": directory / f"{stem}_metadata.json",
    }


def routing_percentages(context, trace):
    """Route every DiT call back to encoder inputs, keeping its own denominator."""
    maps, diagnostics = trace.project(context)
    total = diagnostics["input_mass"].numpy()
    if not np.isfinite(total).all() or (total <= 0).any():
        raise ValueError("DiT context routing must have positive finite mass at every denoising step")
    camera_maps = {key: value.numpy() * (100.0 / total)[:, None, None]
                   for key, value in maps.items()}
    visual = diagnostics["visual_mass"].numpy() * 100.0 / total
    wrench = diagnostics["wrench_mass"].numpy() * 100.0 / total
    if not np.allclose(visual + wrench, 100.0, atol=0.01):
        raise RuntimeError("Image and wrench routing percentages do not sum to 100")
    return camera_maps, visual, wrench, total


def heatmap_color_bounds(records, percentiles=(1.0, 99.0)):
    """Set one episode scale after averaging each patch over denoising calls."""
    averaged = [{"cameras": {key: {"percent": camera["percent"].mean(axis=0)}
                             for key, camera in record["cameras"].items()}}
                for record in records]
    return rendering.percentile_bounds(averaged, "percent", percentiles)


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


def render_chart(visual, wrench, episode_times, selected, width):
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
    _write(chart, "image", max(left, width - 198), 22, 0.42, (255, 220, 65))
    _write(chart, "wrench", max(left, width - 102), 22, 0.42, (105, 225, 255))
    for fraction in np.linspace(0, 1, 5):
        x = int(left + fraction * (right - left))
        precision = 2 if duration < 1 else 1
        _write(chart, f"{fraction * duration:.{precision}f}s", min(x - 8, width - 35),
               bottom + 17, 0.35, (180, 180, 180))
    _write(chart, f"episode {episode_times[selected] - episode_times[0]:.2f}s | "
                  f"image {visual[selected]:.2f}% | wrench {wrench[selected]:.2f}%",
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
    width = images.shape[1]
    banner = np.full((34, width, 3), 17, dtype=np.uint8)
    _write(banner, f"{label.upper()} attention | ep {args.episode} | {frame_index + 1}/{len(visual_curve)} "
                   f"| anchor {record['anchor']}", 9, 23, 0.46)
    scale = np.full((56, width, 3), 18, dtype=np.uint8)
    low, high = color_bounds
    p_low, p_high = args.attention_percentiles
    _write(scale, f"Action to image patches | denoising-step mean | p{p_low:g}-p{p_high:g}", 10, 17, 0.43)
    gradient = np.linspace(0, 255, width - 32, dtype=np.uint8)[None, :]
    scale[25:39, 16:width - 16] = cv2.cvtColor(
        cv2.applyColorMap(gradient, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    _write(scale, f"{low:.3g}%", 16, 54, 0.37)
    _write(scale, f"{high:.3g}%", width - 78, 54, 0.37)
    chart = render_chart(visual_curve, wrench_curve, episode_times, frame_index, width)
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
    label = fusion_label(fusion)
    dit = DitActionAttentionCapture(policy, action_index=args.action_index)
    camera_keys = common.display_camera_keys(policy, "both")
    paths = output_paths(checkpoint, args.episode, label)
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
                maps, visual, wrench, context_mass = routing_percentages(context, fusion)
                cameras = {}
                for key in fusion.image_keys:
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
        color_bounds = heatmap_color_bounds(records, args.attention_percentiles)
        visual_curve = np.asarray([record["visual"].mean() for record in records])
        wrench_curve = np.asarray([record["wrench"].mean() for record in records])
        episode_times = np.asarray([record["anchor"] for record in records], dtype=np.float64) / dataset_fps
        for frame_index, record in enumerate(records):
            frame = render_frame(record, frame_index, camera_keys, color_bounds, args, label,
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
        for index, key in enumerate(fusion.image_keys):
            arrays[f"camera_{index}_percent"] = np.stack([record["cameras"][key]["percent"]
                                                           for record in records])
            arrays[f"camera_{index}_episode_percent"] = arrays[f"camera_{index}_percent"].mean(axis=1)
        np.savez_compressed(paths["raw"], **arrays)
        metadata = {
            "checkpoint": str(checkpoint), "episode": args.episode, "fusion_label": label,
            "inference_steps": int(len(records[0]["timesteps"])), "observation_windows": len(records),
            "video_frames": len(records), "fps": output_fps, "dataset_fps": dataset_fps,
            "video_codec": args.video_codec,
            "all_camera_keys": list(fusion.image_keys), "display_camera_keys": camera_keys,
            "camera_patch_grids": {key: list(records[0]["cameras"][key]["grid"]) for key in fusion.image_keys},
            "observation_display": args.observation, "route_scope": "all observation frames and cameras",
            "action_query": "executed chunk" if args.action_index is None else f"action index {args.action_index}",
            "image_heatmap": "Arithmetic mean over denoising calls of (100 * image patch route / all routed condition mass at that call)",
            "route_curves": "One point per episode observation: mean over denoising calls of (100 * image or wrench route / all routed condition mass at that call)",
            "time_axis": "Episode anchor time in seconds; one video frame per selected observation anchor",
            "color_scale": f"shared p{args.attention_percentiles[0]:g} to p{args.attention_percentiles[1]:g} of denoising-averaged patch percentages over selected observation windows, cameras and observation times; display clipped outside bounds",
            "color_percentiles": list(args.attention_percentiles), "color_bounds_percent": list(color_bounds),
            "color_peak_percent": color_bounds[1], "color_map": "JET, matching wrench_image_self_attn episode videos",
            "fusion_architecture": fusion.architecture, "dit_diagnostics": dit.diagnostics(),
            "interpretation": "Equal-residual attention rollout, not causal final-action attribution. Image-query CA without preceding self-attention fixes the image/wrench route split at 50/50 by construction.",
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
