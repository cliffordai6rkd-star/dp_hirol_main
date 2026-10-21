"""Visualize final executed-action sensitivity through the full diffusion sampler.

Differentiate the final, postprocessed action chunk with respect to RGB input
pixels, keeping wrench observations and sampler noise fixed. Action channels
use the checkpoint normalizer's scale. RGB Jacobian energy is pooled over
native DINO patches, retaining separate cameras and observation times.

This is local gradient sensitivity, not attention probability or causal
contribution. No attention maps are averaged across denoising steps.
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
from omegaconf import OmegaConf

from diffusion_policy.scripts import visualize_cross_attention_episode as common
from diffusion_policy.scripts import visualize_image_query_cross_attention_episode as rendering
from diffusion_policy.scripts.full_chain_image_saliency import full_chain_image_saliency


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoints = parser.add_mutually_exclusive_group(required=True)
    checkpoints.add_argument("--checkpoint", type=pathlib.Path)
    checkpoints.add_argument("--checkpoint-dir", type=pathlib.Path)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--dataset-path", type=pathlib.Path, default=None)
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="torchcodec")
    parser.add_argument("--output", type=pathlib.Path, default=None, help="Output MP4 path or prefix.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--camera", choices=("wrist", "side", "both"), default="both")
    parser.add_argument("--observation", choices=("all", "last"), default="all",
                        help="Display each observation on its own input image (default), or only the latest. Raw values retain all observations.")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--video-codec", choices=("av1", "h264", "mpeg4"), default="av1")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--percentiles", "--attention-percentiles", nargs=2, type=float,
                        metavar=("LOW", "HIGH"), default=(1.0, 99.0),
                        help="Display percentiles shared by all captured frames, cameras and observation times.")
    parser.add_argument("--gradient-probes", type=int, default=None, metavar="N",
                        help="Opt into N random-projection gradient estimates for speed. Omit for the exact per-output Jacobian. This never averages denoising-step attention.")
    parser.add_argument("--autocast", choices=("off", "bfloat16"), default="off",
                        help="Run the differentiable forward with autocast. bfloat16 is useful on recent CUDA GPUs; it slightly changes numerical values.")
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42,
                        help="Sampler seed, reset for each frame to hold sampling noise fixed.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-frames", action="store_true")
    args = parser.parse_args(argv)
    if args.episode < 0:
        parser.error("--episode must be nonnegative")
    if args.stride < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("--stride and --max-frames must be positive")
    if args.gradient_probes is not None and args.gradient_probes < 1:
        parser.error("--gradient-probes must be positive")
    if not math.isfinite(args.alpha) or not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps must be positive and finite")
    low, high = args.percentiles
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high <= 100):
        parser.error("--percentiles must satisfy 0 <= LOW < HIGH <= 100")
    return args


def output_paths(checkpoint, args):
    prefix = (checkpoint.parent.parent / f"action_image_attribution_episode_{args.episode}"
              if args.output is None else args.output.expanduser().resolve())
    if prefix.suffix.lower() == ".mp4":
        prefix = prefix.with_suffix("")
    return {
        "video": prefix.with_name(prefix.name + ".mp4"),
        "raw": prefix.with_name(prefix.name + "_raw.npz"),
        "metadata": prefix.with_name(prefix.name + "_metadata.json"),
    }


def final_action_target(policy, prediction, obs):
    """Final controller actions, including quaternion/relative-pose processing.

    Normalizer offsets have no effect on derivatives; its learned channel
    scales balance units. Keep the actual executed chunk chosen by the policy.
    """
    return policy.normalizer["action"].normalize(prediction["action"])


def pool_pixel_sensitivity(values, backbone):
    """RMS pool [To,H,W] pixel sensitivity onto native DINO patch regions."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected finite nonnegative sensitivity [To,H,W]")
    patch = getattr(getattr(backbone, "config", None), "patch_size", None)
    if patch is None:
        raise ValueError("Checkpoint DINO config must provide patch_size")
    if isinstance(patch, (tuple, list)):
        ph, pw = map(int, patch)
    else:
        ph = pw = int(patch)
    if ph < 1 or pw < 1:
        raise ValueError("Invalid DINO patch_size")
    steps, height, width = values.shape
    if height % ph or width % pw:
        raise ValueError("Image dimensions must be divisible by DINO patch_size for pixel-region pooling")
    gh, gw = height // ph, width // pw
    squared = values.astype(np.float64) ** 2
    return np.sqrt(squared.reshape(steps, gh, ph, gw, pw).mean(axis=(2, 4))).astype(np.float32)


def shared_bounds(records, percentiles):
    values = np.concatenate([camera["saliency"].reshape(-1)
                             for record in records for camera in record["cameras"].values()])
    if not np.isfinite(values).all():
        raise ValueError("Non-finite sensitivities")
    low, high = map(float, np.percentile(values, percentiles))
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        low, high = 0.0, high if high > 0 else 1.0
    return low, high


def overlay(image, values, bounds, alpha):
    # Gradients can be arbitrarily small. Never fall back to per-panel ranges
    # merely because the shared gradient range is below an absolute epsilon.
    low, high = bounds
    scaled = np.clip((np.asarray(values, dtype=np.float64) - low) / (high - low), 0, 1)
    scaled = cv2.resize(scaled.astype(np.float32), (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_CUBIC).clip(0, 1)
    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(scaled * 255), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(np.clip((1 - alpha) * image + alpha * heatmap, 0, 255).astype(np.uint8))


def render_frame(record, camera_keys, bounds, *, episode, alpha, observation="all", percentiles=(1, 99)):
    count = record["cameras"][camera_keys[0]]["saliency"].shape[0]
    indices = range(count) if observation == "all" else [count - 1]
    rows = []
    for index in indices:
        panels = []
        for key in camera_keys:
            camera = record["cameras"][key]
            panel = overlay(camera["images"][index], camera["saliency"][index], bounds, alpha)
            suffix = "latest" if index == count - 1 else "history"
            panels.append(rendering._banner(panel, f"{key} | observation {index}: {suffix}", height=24))
        row = np.zeros((max(panel.shape[0] for panel in panels), sum(panel.shape[1] for panel in panels), 3), dtype=np.uint8)
        left = 0
        for panel in panels:
            row[:panel.shape[0], left:left + panel.shape[1]] = panel
            left += panel.shape[1]
        rows.append(row)
    image = np.concatenate(rows, axis=0)
    image = rendering._banner(image, f"Final executed-action sensitivity | ep={episode} anchor={record['anchor']}")
    image = rendering._banner(image, "RGB input gradients through the complete serial sampler; patch RMS")
    return rendering._colorbar(image, bounds, f"Sensitivity | shared episode p{percentiles[0]:g}-p{percentiles[1]:g}; not attention probability")


def save_raw(paths, records, metadata):
    arrays = {"anchors": np.asarray([record["anchor"] for record in records], dtype=np.int64),
              "action": np.stack([record["action"] for record in records]),
              "scaled_action": np.stack([record["scaled_action"] for record in records])}
    for index, key in enumerate(metadata["all_camera_keys"]):
        arrays[f"camera_{index}_sensitivity"] = np.stack([record["cameras"][key]["saliency"] for record in records])
    np.savez_compressed(paths["raw"], **arrays)
    paths["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")


def run(args):
    checkpoint = common.choose_checkpoint(args.checkpoint, args.checkpoint_dir)
    payload, cfg = common.load_checkpoint_payload(checkpoint)
    device = common.resolve_device(args.device, cfg)
    policy = common.instantiate_policy(payload, cfg, device, capture_attention=False)
    del payload
    all_camera_keys = tuple(policy.obs_encoder.image_keys)
    camera_keys = common.display_camera_keys(policy, args.camera)
    estimator = "exact" if args.gradient_probes is None else "hutchinson"
    autocast_dtype = torch.bfloat16 if args.autocast == "bfloat16" else None
    if autocast_dtype is not None and device.type != "cuda":
        print("warning: bfloat16 autocast was requested on a non-CUDA device; it may be slower or unsupported", flush=True)
    print(f"checkpoint: {checkpoint}\nmethod: final-action input Jacobian; estimator: {estimator}; autocast: {args.autocast}; device: {device}", flush=True)
    paths = output_paths(checkpoint, args)
    dataset = common.instantiate_episode_dataset(cfg, args.dataset_path, video_backend=args.video_backend)
    records, diagnostics = [], []
    writer = None
    try:
        selected = common.episode_sample_indices(dataset, args.episode)[::args.stride]
        if args.max_frames is not None:
            selected = selected[:args.max_frames]
        fps = float(args.fps) if args.fps is not None else common.infer_dataset_fps(dataset) / args.stride
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("FPS must be positive and finite")
        rendering.prepare_outputs(paths, overwrite=args.overwrite, save_frames=args.save_frames)
        for frame_index, (anchor, sample_index) in enumerate(selected):
            sample = dataset[sample_index]
            obs = {key: value.unsqueeze(0).to(device) for key, value in sample["obs"].items()}
            # Identical initial and stochastic scheduler noise for each frame;
            # probe noise is also fixed, using a separate CPU RNG in the helper.
            generator = torch.Generator(device=device).manual_seed(args.seed)
            result = full_chain_image_saliency(
                policy, obs, final_action_target, image_keys=all_camera_keys,
                estimator=estimator, num_probes=args.gradient_probes or 8,
                probe_seed=args.probe_seed, generator=generator,
                autocast_dtype=autocast_dtype,
            )
            cameras = {}
            for key in all_camera_keys:
                pixels = result.maps[key][0, :policy.n_obs_steps].numpy()
                maps = pool_pixel_sensitivity(pixels, policy.obs_encoder.backbone)
                cameras[key] = {
                    "images": np.stack([common.image_to_uint8(image) for image in sample["obs"][key][:policy.n_obs_steps]]),
                    "saliency": maps,
                }
            action = result.prediction["action"]
            records.append({"anchor": anchor, "cameras": cameras, "action": action[0].float().cpu().numpy(),
                            "scaled_action": policy.normalizer["action"].normalize(action.to(device))[0].detach().cpu().numpy()})
            diagnostics.append(result.diagnostics)
            print(f"Full-chain gradients {frame_index + 1}/{len(selected)}", flush=True)
        bounds = shared_bounds(records, args.percentiles)
        for index, record in enumerate(records):
            frame = render_frame(record, camera_keys, bounds, episode=args.episode,
                                 alpha=args.alpha, observation=args.observation, percentiles=args.percentiles)
            if writer is None:
                writer = common.FFmpegVideoWriter(paths["video"], frame.shape[1], frame.shape[0], fps, args.video_codec)
            writer.write(frame)
            if args.save_frames:
                path = rendering.frame_directory(paths["video"]) / f"frame_{index:06d}.png"
                if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError(f"Could not write frame: {path}")
        closing_writer, writer = writer, None
        closing_writer.close()
        metadata = {
            "checkpoint": str(checkpoint), "episode": args.episode, "frames": len(records),
            "checkpoint_policy_target": OmegaConf.select(cfg, "policy._target_", default=None),
            "baseline_mode": getattr(policy, "baseline_mode", None),
            "cross_attn_query": getattr(policy.obs_encoder, "cross_attn_query", None),
            "pre_cross_self_attn_layers": getattr(policy.obs_encoder, "pre_cross_self_attn_layers", None),
            "cross_attn_layers": getattr(policy.obs_encoder, "cross_attn_layers", None),
            "force_temporal_encoder": getattr(policy.obs_encoder, "force_temporal_encoder_type", None),
            "method": "Final executed-action RGB-input Jacobian RMS, pooled over DINO patch regions",
            "target": "predict_action()['action'] after all postprocessing, scaled using the checkpoint action normalizer",
            "target_shape": list(records[0]["action"].shape),
            "action_channel_scale": policy.normalizer["action"].params_dict["scale"].detach().float().cpu().tolist(),
            "action_start_index": int(policy.action_start_index),
            "relative_pose_actions": bool(getattr(policy, "relative_pose_actions", False)),
            "normalizer_note": "Learned per-channel scales balance action units; offsets do not affect derivatives. In relative-pose checkpoints these same channel scales are applied to the final absolute controller output.",
            "formula": "sqrt(mean_output(mean_pixel_in_patch(sum_RGB((d scaled_final_action / d input_pixel)^2))))",
            "estimator": estimator, "gradient_probes": args.gradient_probes,
            "autocast": args.autocast,
            "probe_seed": args.probe_seed, "sampler_seed": args.seed,
            "noise_policy": "Sampler generator and independent probe generator reset identically for every frame",
            "sampling": diagnostics,
            "all_camera_keys": list(all_camera_keys), "display_camera_keys": camera_keys,
            "observation": args.observation, "observation_steps": int(policy.n_obs_steps),
            "raw_sensitivity_shape": "[frame, observation_time, patch_row, patch_column]",
            "camera_patch_grids": {key: list(value["saliency"].shape[-2:]) for key, value in records[0]["cameras"].items()},
            "bounds": list(bounds), "percentiles": list(args.percentiles),
            "bounds_scope": "All captured frames, cameras and observation times; raw sensitivity values unchanged",
            "fps": fps, "stride": args.stride, "video_codec": args.video_codec,
            "limitations": [
                "Local sensitivity around the actual observed image and fixed sampling noise, not attention probability or causal contribution.",
                "Includes all serial denoising steps, fusion, DINO and output postprocessing; no per-step attention averaging.",
                "Wrench, action reference and other observations are held constant; this does not isolate the effect of wrench.",
                "True sampler clipping and all preprocessing are retained; saturation can give zero local gradients.",
                "Gradient energy pools action channels and execution times; it has no sign or probability-mass interpretation.",
                "Sensitivity depends on the action representation and checkpoint channel scales; independently stretched videos cannot establish cross-model magnitude differences.",
                "Rademacher probes, if explicitly requested, approximate this same full-chain Jacobian norm, not individual denoising-step maps.",
            ],
            "outputs": {key: str(path) for key, path in paths.items()},
        }
        save_raw(paths, records, metadata)
    finally:
        with contextlib.ExitStack() as closing:
            closing.callback(dataset.lerobot_dataset.close)
            if writer is not None:
                closing.callback(writer.close)
    print(f"frames: {len(records)}; shared sensitivity bounds: {bounds}")
    for key, path in paths.items():
        print(f"{key}: {path}")
    return paths


if __name__ == "__main__":
    run(parse_args())
