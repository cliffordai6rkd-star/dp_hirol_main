"""Render checkpoint-selected self-attention or cross-attention for an episode.

The checkpoint config determines the architecture, query direction and token
layout. Select a fusion SA, CA or DINO layer with --mode and --layer-index.
Display ranges default to episode-wide 1st/99th percentiles, shared across
all cameras and wrench positions. Raw probabilities are saved unchanged.

Image-query CA also produces a projected CA-update L2 video. This filename
remains an entry point for existing commands; visualize_attention_episode.py
provides the same checkpoint-general CLI.

Only observation encoding and fusion run; action diffusion is unnecessary for
these attention maps. --attention-only suppresses the separate CA-update video.
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
import shutil

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.scripts import visualize_cross_attention_episode as common
from diffusion_policy.scripts.checkpoint_attention_capture import AttentionCapture


ATTENTION_BOUNDS = (0.0, 1.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoints = parser.add_mutually_exclusive_group(required=True)
    checkpoints.add_argument("--checkpoint", type=pathlib.Path)
    checkpoints.add_argument("--checkpoint-dir", type=pathlib.Path)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--dataset-path", type=pathlib.Path, default=None)
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="torchcodec")
    parser.add_argument(
        "--output", type=pathlib.Path, default=None,
        help="Output prefix (optional .mp4 suffix). Creates <prefix>_attention.mp4, plus CA-update video when image queries exist.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--camera", choices=("wrist", "side", "both"), default="both")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--video-codec", choices=("av1", "h264", "mpeg4"), default="av1")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.45)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--mode", "--attention-mode", choices=("auto", "self-attn", "cross-attn", "dino"),
        default="auto", help="auto selects CA, otherwise fusion SA, otherwise DINO backbone attention.",
    )
    modes.add_argument("--sa", dest="mode", action="store_const", const="self-attn", help="Alias for --mode self-attn.")
    modes.add_argument("--ca", dest="mode", action="store_const", const="cross-attn", help="Alias for --mode cross-attn.")
    parser.add_argument(
        "--attention-query", choices=("all", "last"), default="all",
        help="Keep all wrench positions, or only the last. Image-only SA averages patch queries (last selects the final patch).",
    )
    parser.add_argument(
        "--attention-normalization", choices=("percentile", "global", "fixed"), default="percentile",
        help="percentile/global share episode percentile bounds; fixed uses 0..1. Neither changes saved probabilities.",
    )
    parser.add_argument(
        "--attention-percentiles", nargs=2, type=float, metavar=("LOW", "HIGH"), default=(1.0, 99.0),
        help="Shared attention display percentiles (default: 1 99).",
    )
    parser.add_argument("--layer-index", "--layer", type=int, default=-1, help="Zero-based layer in the selected mode; -1 selects the final layer.")
    parser.add_argument("--list-layers", action="store_true", help="Load checkpoint on CPU and list available modes/layers without episode inference.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--attention-only", action="store_true",
                        help="Export attention probabilities only, without the separate CA-update L2 video.")
    args = parser.parse_args(argv)
    if args.episode is None and not args.list_layers:
        parser.error("--episode is required unless --list-layers is used")
    if args.stride < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("--stride and --max-frames must be positive")
    if not math.isfinite(args.alpha) or not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps must be positive and finite")
    low, high = args.attention_percentiles
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high <= 100):
        parser.error("--attention-percentiles must satisfy 0 <= LOW < HIGH <= 100")
    return args


def output_paths(checkpoint_path, args, capture=None):
    if args.output is None:
        mode = args.mode if capture is None else capture.mode
        layer = args.layer_index if capture is None else capture.layer_index
        query = "" if capture is None else f"_{capture.query_source}_query"
        prefix = checkpoint_path.parent.parent / f"attention_episode_{args.episode}_{mode.replace('-', '_')}_layer_{layer}{query}"
    else:
        prefix = args.output.expanduser().resolve()
        if prefix.suffix.lower() == ".mp4":
            prefix = prefix.with_suffix("")
    paths = {
        "attention": prefix.with_name(prefix.name + "_attention.mp4"),
        "raw": prefix.with_name(prefix.name + "_raw.npz"),
        "metadata": prefix.with_name(prefix.name + "_metadata.json"),
    }
    if (capture is None or capture.has_updates) and not args.attention_only:
        paths["update"] = prefix.with_name(prefix.name + "_ca_update.mp4")
    return paths


def frame_directory(path):
    return path.with_name(path.stem + "_frames")


def prepare_outputs(paths, *, overwrite, save_frames):
    # Check every output before touching any existing file, so a late collision
    # cannot delete one video while leaving the other from an older export.
    targets = list(paths.values())
    frame_dirs = [frame_directory(path) for path in paths.values() if path.suffix == ".mp4"]
    for path in targets:
        if path.is_dir():
            raise IsADirectoryError(path)
    for path in [*targets, *frame_dirs]:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}. Pass --overwrite to replace it.")
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
    for path in frame_dirs:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        if save_frames:
            path.mkdir(parents=True)


def wrench_token_labels(cfg, policy):
    count = int(policy.obs_encoder.wrench_history_steps)
    offsets = OmegaConf.select(
        cfg, f"task.dataset.windowed_lowdim_feature_groups.{policy.wrench_key}.offsets_sec",
        default=None,
    )
    if offsets is not None and len(offsets) != count:
        raise ValueError("Checkpoint wrench offsets_sec does not match its wrench token count")
    labels = []
    for index in range(count):
        suffix = f" ({1000 * float(offsets[index]):+g} ms)" if offsets is not None else ""
        labels.append(f"Wrench state {index + 1}{suffix}")
    return labels, None if offsets is None else [float(value) for value in offsets]


def percentile_bounds(records, field, percentiles=(1.0, 99.0)):
    values = np.concatenate([
        camera[field].reshape(-1)
        for record in records for camera in record["cameras"].values()
    ])
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite {field} values")
    low, high = (float(value) for value in np.percentile(values, percentiles))
    if high <= low + 1e-8:
        low, high = float(values.min()), float(values.max())
    # A constant, nonzero value still needs a meaningful shared scale. Avoid
    # the common heatmap helper falling back to a different range per camera.
    if high <= low + 1e-8:
        low, high = 0.0, max(high, 1e-6)
    return low, high


def update_bounds(records):
    return percentile_bounds(records, "update")


def attention_definition(capture):
    if capture.mode == "cross-attn":
        if capture.query_source == "image":
            return "Head mean; each image query is normalized over wrench keys. Panels select individual wrench keys; image positions are not renormalized."
        return "Head mean; panels select individual wrench queries. Softmax includes all camera patches jointly, or one camera at a time in per_camera mode."
    if capture.mode == "self-attn" and capture.query_source == "wrench":
        context = ("Earlier-time and wrench keys retain their original softmax mass."
                   if capture.temporal_layout == "flattened_observations" else
                   "Each observation is independent; all camera and wrench keys remain in the original softmax denominator.")
        return f"Wrench-query/image-key submatrix of joint SA. Head mean, latest observation rows/columns. {context}"
    if capture.mode == "dino":
        return "DINO patch-query/patch-key submatrix, averaged over heads and selected patch queries. CLS/register keys remain in the original softmax denominator."
    return "Image fusion SA, averaged over heads and selected latest-observation patch queries; all cameras remain in the original softmax denominator."


def _banner(image, text, height=28):
    width = max(image.shape[1], 280)
    result = np.zeros((image.shape[0] + height, width, 3), dtype=np.uint8)
    result[height:, :image.shape[1]] = image
    text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    scale = 0.45 * min(1.0, (width - 16) / max(1, text_width))
    cv2.putText(result, text, (8, height - 8), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def _camera_row(record, camera_keys, *, alpha, token_index=None, bounds=None):
    panels = []
    for key in camera_keys:
        camera = record["cameras"][key]
        values = camera["update"] if token_index is None else camera["attention"][token_index]
        overlay = common.overlay_attention(
            camera["image"], values, camera["grid"], alpha,
            normalization_bounds=bounds,
        )
        panels.append(_banner(overlay, key, height=24))
    height = max(panel.shape[0] for panel in panels)
    output = np.zeros((height, sum(panel.shape[1] for panel in panels), 3), dtype=np.uint8)
    offset = 0
    for panel in panels:
        output[:panel.shape[0], offset:offset + panel.shape[1]] = panel
        offset += panel.shape[1]
    return output


def _colorbar(image, bounds, title):
    width = max(image.shape[1], 360)
    result = np.zeros((image.shape[0] + 76, width, 3), dtype=np.uint8)
    result[:image.shape[0], :image.shape[1]] = image
    top = image.shape[0]
    title_width = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
    scale = 0.42 * min(1.0, (width - 24) / max(1, title_width))
    cv2.putText(result, title, (12, top + 18), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 1, cv2.LINE_AA)
    left, right = 24, width - 24
    gradient = np.linspace(0, 255, right - left, dtype=np.uint8)[None, :]
    colors = cv2.cvtColor(cv2.applyColorMap(gradient, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    result[top + 26:top + 42, left:right] = colors
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = bounds[0] + fraction * (bounds[1] - bounds[0])
        text = f"{value:.4g}"
        x = int(left + fraction * (right - left - 1))
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        x = min(max(0, x - text_width // 2), width - text_width)
        cv2.putText(result, text, (x, top + 62), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
    # yuv420p requires even dimensions, including for odd-sized camera images.
    return np.ascontiguousarray(np.pad(
        result, ((0, result.shape[0] % 2), (0, result.shape[1] % 2), (0, 0))
    ))


def render_attention(
    record, camera_keys, labels, *, episode, alpha, layer_index,
    bounds=ATTENTION_BOUNDS, mode="cross-attn", query_source="image",
    percentile_range=None,
):
    tiles = [
        _banner(_camera_row(record, camera_keys, alpha=alpha, token_index=index,
                            bounds=bounds), label)
        for index, label in enumerate(labels)
    ]
    columns = min(4, len(tiles))
    rows = math.ceil(len(tiles) / columns)
    height = max(tile.shape[0] for tile in tiles)
    width = max(tile.shape[1] for tile in tiles)
    gap = 8
    grid = np.zeros((rows * height + (rows - 1) * gap,
                     columns * width + (columns - 1) * gap, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        y, x = row * (height + gap), column * (width + gap)
        grid[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
    if mode == "cross-attn":
        relation = "Image Q -> wrench K/V" if query_source == "image" else "Wrench Q -> image K/V"
    elif mode == "dino":
        relation = "DINO patch Q -> patch K (CLS/register keys retained in softmax)"
    else:
        relation = "Joint SA: wrench Q -> image K" if query_source == "wrench" else "Image SA: patch Q -> patch K"
    grid = _banner(grid, f"{relation} | {mode} layer index {layer_index} | ep={episode} frame={record['anchor']}")
    detail = ("Encoded wrench states include temporal context."
              if query_source == "wrench" or mode == "cross-attn"
              else "Image-patch query reduction; CLS/registers are not spatial patches.")
    grid = _banner(grid, f"Head mean; latest observation; original probabilities. {detail}")
    scale = "fixed 0 to 1" if percentile_range is None else f"shared episode p{percentile_range[0]:g}-p{percentile_range[1]:g}"
    return _colorbar(grid, bounds, f"Attention weight: {scale} (all cameras and wrench positions)")


def render_update(record, camera_keys, bounds, *, episode, alpha, layer_index):
    image = _camera_row(record, camera_keys, alpha=alpha, bounds=bounds)
    image = _banner(image, f"CA update L2 | layer index {layer_index} | ep={episode} frame={record['anchor']}")
    image = _banner(image, "After output projection; before image residual and FFN")
    return _colorbar(image, bounds, "||Delta V||_2: shared episode p1-p99 scale")


def save_raw(paths, records, metadata):
    values = {"anchors": np.asarray([record["anchor"] for record in records], dtype=np.int64)}
    for index, key in enumerate(metadata["all_camera_keys"]):
        values[f"camera_{index}_attention"] = np.stack([record["cameras"][key]["attention"] for record in records])
        if "update" in records[0]["cameras"][key]:
            values[f"camera_{index}_update_l2"] = np.stack([record["cameras"][key]["update"] for record in records])
    np.savez_compressed(paths["raw"], **values)
    paths["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")


def run(args):
    checkpoint = common.choose_checkpoint(args.checkpoint, args.checkpoint_dir)
    payload, cfg = common.load_checkpoint_payload(checkpoint)
    device = torch.device("cpu") if args.list_layers else common.resolve_device(args.device, cfg)
    policy = common.instantiate_policy(payload, cfg, device, capture_attention=False)
    del payload
    capture = AttentionCapture(
        policy, mode="auto" if args.list_layers else args.mode,
        layer_index=-1 if args.list_layers else args.layer_index,
        query_mode=args.attention_query,
    )
    description = {
        "checkpoint": str(checkpoint), "architecture": capture.architecture,
        "available_modes": {mode: {"layers": count, "indices": list(range(count))}
                            for mode, count in capture.available_modes.items()},
        "selected_mode": capture.mode, "selected_layer_index": capture.layer_index,
        "query_source": capture.query_source, "temporal_layout": capture.temporal_layout,
        "checkpoint_policy_target": OmegaConf.select(cfg, "policy._target_", default=None),
        "has_spatial_ca_update": capture.has_updates,
    }
    print(json.dumps(description, indent=2, ensure_ascii=False), flush=True)
    if args.list_layers:
        return description
    layer_index = capture.layer_index
    export_updates = capture.has_updates and not args.attention_only
    camera_keys = common.display_camera_keys(policy, args.camera)
    if capture.query_source == "wrench" or capture.mode == "cross-attn":
        labels, offsets = wrench_token_labels(cfg, policy)
        all_offsets = offsets
        displayed_wrench_indices = list(range(int(policy.obs_encoder.wrench_history_steps)))
        if args.attention_query == "last":
            labels = labels[-1:]
            offsets = None if offsets is None else offsets[-1:]
            displayed_wrench_indices = displayed_wrench_indices[-1:]
    else:
        offsets, all_offsets, displayed_wrench_indices = None, None, None
        labels = ["Mean image-patch queries" if args.attention_query == "all" else "Last image-patch query"]
    paths = output_paths(checkpoint, args, capture)
    dataset = common.instantiate_episode_dataset(cfg, args.dataset_path, video_backend=args.video_backend)
    records = []
    writers = {}
    try:
        selected = common.episode_sample_indices(dataset, args.episode)[::args.stride]
        if args.max_frames is not None:
            selected = selected[:args.max_frames]
        fps = float(args.fps) if args.fps is not None else common.infer_dataset_fps(dataset)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("FPS must be positive and finite")
        prepare_outputs(paths, overwrite=args.overwrite, save_frames=args.save_frames)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        with capture:
            for frame_index, (anchor, sample_index) in enumerate(selected):
                sample = dataset[sample_index]
                obs = {key: value.unsqueeze(0).to(device) for key, value in sample["obs"].items()}
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" else contextlib.nullcontext()
                )
                with torch.inference_mode(), autocast:
                    policy._encode_context(obs, apply_curriculum_mask=False, generator=generator)
                attention = capture.attention_by_camera()
                updates = capture.update_by_camera() if export_updates else {}
                cameras = {}
                for key in capture.image_keys:
                    image = sample["obs"][key][-1]
                    cameras[key] = {
                        "image": common.image_to_uint8(image),
                        "grid": common.patch_grid(attention[key].shape[-1], *image.shape[-2:], policy.obs_encoder.backbone),
                        "attention": attention[key].numpy(),
                    }
                    if export_updates:
                        cameras[key]["update"] = updates[key].numpy()
                records.append({"anchor": anchor, "cameras": cameras})
                if frame_index == 0 or (frame_index + 1) % 50 == 0 or frame_index + 1 == len(selected):
                    print(f"Inference {frame_index + 1}/{len(selected)}", flush=True)
        attention_range = None if args.attention_normalization == "fixed" else args.attention_percentiles
        attention_bounds = ATTENTION_BOUNDS if attention_range is None else percentile_bounds(records, "attention", attention_range)
        bounds = update_bounds(records) if export_updates else None
        for index, record in enumerate(records):
            frames = {
                "attention": render_attention(record, camera_keys, labels, episode=args.episode,
                                               alpha=args.alpha, layer_index=layer_index,
                                               bounds=attention_bounds, mode=capture.mode,
                                               query_source=capture.query_source,
                                               percentile_range=attention_range),
            }
            if export_updates:
                frames["update"] = render_update(record, camera_keys, bounds, episode=args.episode,
                                                alpha=args.alpha, layer_index=layer_index)
            for key, frame in frames.items():
                if key not in writers:
                    writer = common.FFmpegVideoWriter(paths[key], frame.shape[1], frame.shape[0], fps, args.video_codec)
                    writers[key] = writer
                writers[key].write(frame)
                if args.save_frames:
                    frame_path = frame_directory(paths[key]) / f"frame_{index:06d}.png"
                    if not cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                        raise RuntimeError(f"Could not write frame: {frame_path}")
        # Close both encoders even if one fails, and verify their return codes
        # before writing metadata that advertises a successful export.
        with contextlib.ExitStack() as closing:
            for writer in writers.values():
                closing.callback(writer.close)
            writers = {}
        metadata = {
            **description,
            "checkpoint": str(checkpoint), "episode": args.episode,
            "frames": len(records), "fps": fps, "video_codec": args.video_codec,
            "layer_index": layer_index, "observation": "latest",
            "inference_scope": "observation_encoder_and_fusion_only",
            "action_generation": False, "attention_only": args.attention_only,
            "all_camera_keys": list(capture.image_keys), "display_camera_keys": camera_keys,
            "camera_patch_grids": {key: list(camera["grid"]) for key, camera in records[0]["cameras"].items()},
            "camera_image_shapes": {key: list(camera["image"].shape) for key, camera in records[0]["cameras"].items()},
            "attention_grid_columns": min(4, len(labels)),
            "attention_query": args.attention_query,
            "wrench_key": getattr(policy, "wrench_key", None), "wrench_offsets_sec": offsets,
            "all_wrench_offsets_sec": all_offsets,
            "displayed_wrench_indices": displayed_wrench_indices,
            "panel_labels": labels,
            "wrench_state_labels": labels if offsets is not None or capture.query_source == "wrench" or capture.mode == "cross-attn" else None,
            "force_temporal_encoder": getattr(policy.obs_encoder, "force_temporal_encoder_type", None),
            "attention_shape": "[frame, displayed_wrench_position_or_image_query_reduction, camera_patch]",
            "attention_bounds": list(attention_bounds),
            "attention_normalization": args.attention_normalization,
            "attention_percentiles": None if attention_range is None else list(attention_range),
            "attention_bounds_scope": "All captured cameras, wrench positions and selected episode frames.",
            "attention_definition": attention_definition(capture),
            "update_definition": "L2 norm over channels of CA output after output projection, before image residual and FFN; evaluation mode." if export_updates else None,
            "update_bounds": None if bounds is None else list(bounds), "update_percentiles": [1, 99] if export_updates else None,
            "update_bounds_scope": "All captured cameras and selected episode frames." if export_updates else None,
            "preprocessing": "Checkpoint policy observation preprocessing and normalization are preserved.",
            "interpretation": "Single-layer attention and feature update diagnostics; neither is causal action importance. Encoded wrench states include temporal context.",
            "seed": args.seed, "stride": args.stride,
            "outputs": {key: str(value) for key, value in paths.items()},
        }
        save_raw(paths, records, metadata)
    finally:
        with contextlib.ExitStack() as closing:
            closing.callback(dataset.lerobot_dataset.close)
            for writer in writers.values():
                closing.callback(writer.close)
    print(f"checkpoint: {checkpoint}")
    print(f"episode: {args.episode}; frames: {len(records)}; fps: {fps:g}; codec: {args.video_codec}")
    print(f"attention bounds: {attention_bounds} ({args.attention_normalization}); update L2 bounds: {bounds}")
    for key, path in paths.items():
        print(f"{key}: {path}")
    return paths


if __name__ == "__main__":
    run(parse_args())
