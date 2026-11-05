"""Visualize the intermediate stages of a patch-residual RenderNet.

This is a read-only diagnostic.  It does not alter the checkpoint or the
RenderNet implementation.  For one episode frame it writes, per camera,
the tiled residual before/after the learned 3x3 refine convolution, the
bilinear mask, gated residual, and rendered RGB.  It also compares the
trained refine path with the same checkpoint with refine bypassed.
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
import pathlib

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from diffusion_policy.scripts import visualize_cross_attention_episode as common
from diffusion_policy.scripts.visualize_render_net_episode import load_render_net


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=72,
                        help="Selected episode frame (not a raw dataset index).")
    parser.add_argument("--camera", choices=("wrist", "side", "both"), default="both")
    parser.add_argument("--dataset-path", type=pathlib.Path)
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav")
    parser.add_argument("--device")
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args(argv)


def to_rgb(x, scale=1.0):
    x = x.detach().float().cpu().numpy().transpose(1, 2, 0)
    return np.ascontiguousarray((np.clip(0.5 + scale * x, 0, 1) * 255).astype(np.uint8))


def image_rgb(x):
    x = x.detach().float().cpu().numpy().transpose(1, 2, 0)
    return np.ascontiguousarray((np.clip(x, 0, 1) * 255).astype(np.uint8))


def heatmap(x, low=None, high=None):
    x = x.detach().float().cpu().numpy()
    if x.ndim == 3:
        x = x[0]
    if low is None or high is None:
        low, high = np.percentile(x, [1, 99])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(x.min()), float(x.max() + 1e-6)
    values = np.clip((x - low) / (high - low), 0, 1)
    return cv2.cvtColor((values * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)


def label(image, text):
    bar = np.full((27, image.shape[1], 3), 18, dtype=np.uint8)
    cv2.putText(bar, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (245, 245, 245), 1, cv2.LINE_AA)
    return np.concatenate([bar, image], axis=0)


def htile(images):
    height = max(image.shape[0] for image in images)
    canvas = np.zeros((height, sum(image.shape[1] for image in images), 3), dtype=np.uint8)
    offset = 0
    for image in images:
        canvas[:image.shape[0], offset:offset + image.shape[1]] = image
        offset += image.shape[1]
    return canvas


def vtile(images):
    width = max(image.shape[1] for image in images)
    canvas = np.zeros((sum(image.shape[0] for image in images), width, 3), dtype=np.uint8)
    offset = 0
    for image in images:
        canvas[offset:offset + image.shape[0], :image.shape[1]] = image
        offset += image.shape[0]
    return canvas


def patch_stages(model, image, wrench_tokens, camera):
    """Return the exact stages used by RenderNet for one [1,3,H,W] image."""
    backbone = model.backbone
    features, _, height, width = backbone.encode(image, wrench_tokens, camera)
    batch, _, grid_h, grid_w = features.shape
    tokens = features.flatten(2).transpose(1, 2)
    raw = backbone.output(tokens).reshape(
        batch, grid_h, grid_w, 3, backbone.patch_size, backbone.patch_size,
    )
    raw = raw.permute(0, 3, 1, 4, 2, 5).reshape(
        batch, 3, grid_h * backbone.patch_size, grid_w * backbone.patch_size,
    )
    refined = backbone.refine(raw) if backbone.residual_refine else raw
    raw = raw[:, :, :height, :width]
    refined = refined[:, :, :height, :width]
    bounded = refined.tanh()
    raw_bounded = raw.tanh()

    logits = model.mask_head(features)
    mask_grid = logits.float().sigmoid()
    padded_height = height + (-height) % backbone.patch_size
    padded_width = width + (-width) % backbone.patch_size
    mask = torch.nn.functional.interpolate(
        mask_grid, size=(padded_height, padded_width), mode="bilinear", align_corners=False,
    )[:, :, :height, :width]
    nearest_mask = torch.nn.functional.interpolate(
        mask_grid, size=(padded_height, padded_width), mode="nearest",
    )[:, :, :height, :width]
    gate = torch.where(mask > model.region_threshold, mask, torch.zeros_like(mask))
    gated = bounded * gate
    gated_without_refine = raw_bounded * gate
    rendered = (image + model.residual_scale * gated).clamp(0, 1)
    rendered_without_refine = (image + model.residual_scale * gated_without_refine).clamp(0, 1)
    return {
        "raw": raw, "refined": refined, "bounded": bounded, "raw_bounded": raw_bounded,
        "mask_grid": mask_grid, "mask": mask, "nearest_mask": nearest_mask, "gate": gate,
        "gated": gated, "gated_without_refine": gated_without_refine,
        "rendered": rendered, "rendered_without_refine": rendered_without_refine,
    }


def residual_metrics(x, patch):
    """Measure pixel roughness and energy explained by repeated patch phase."""
    x = x.detach().float().cpu().numpy()[0]
    c, h, w = x.shape
    dx, dy = np.abs(np.diff(x, axis=2)), np.abs(np.diff(x, axis=1))
    bx = np.arange(1, w) % patch == 0
    by = np.arange(1, h) % patch == 0
    boundary = np.concatenate([dx[:, :, bx].ravel(), dy[:, by, :].ravel()]).mean()
    inside = np.concatenate([dx[:, :, ~bx].ravel(), dy[:, ~by, :].ravel()]).mean()
    hh, ww = (h // patch) * patch, (w // patch) * patch
    blocks = x[:, :hh, :ww].reshape(c, hh // patch, patch, ww // patch, patch)
    centered = blocks - blocks.mean(axis=(2, 4), keepdims=True)
    template = centered.mean(axis=(1, 3), keepdims=True)
    energy = np.square(centered).mean()
    return {"mean_abs": float(np.abs(x).mean()),
            "mean_abs_neighbor_difference": float((dx.mean() + dy.mean()) / 2),
            "patch_boundary_vs_interior_jump": float(boundary / max(inside, 1e-12)),
            "repeated_patch_phase_energy_fraction": float(np.square(template).mean() / max(energy, 1e-12))}


def blurred(x, size, sigma):
    array = x[0].detach().float().cpu().numpy().transpose(1, 2, 0)
    result = cv2.GaussianBlur(array, (size, size), sigma, borderType=cv2.BORDER_REFLECT_101)
    return torch.from_numpy(result.transpose(2, 0, 1).copy()).to(x).unsqueeze(0)


def run(args):
    checkpoint = common.choose_checkpoint(args.checkpoint, None)
    model, preprocessor, cfg, teacher_cfg, device, info = load_render_net(checkpoint, args.device)
    if model.rendering_mode != "patch_residual":
        raise ValueError(f"Expected patch_residual checkpoint, got {model.rendering_mode!r}")
    if model.region_selection != "soft" or model.region_activation != "sigmoid":
        raise ValueError("This diagnostic currently supports the sigmoid soft-mask renderer")
    image_keys = sorted(key for key, spec in teacher_cfg.policy.shape_meta.obs.items()
                        if spec.get("type") == "rgb")
    display_keys = [key for key in ("wrist", "side") if key in image_keys]
    display_keys.extend(key for key in image_keys if key not in display_keys)
    if args.camera != "both":
        display_keys = [args.camera]
    teacher_cfg = OmegaConf.create(OmegaConf.to_container(teacher_cfg, resolve=True))
    with open_dict(teacher_cfg):
        teacher_cfg.task.dataset.update(OmegaConf.to_container(cfg.dataset, resolve=True))
    dataset = common.instantiate_episode_dataset(teacher_cfg, args.dataset_path, args.video_backend)
    selected = common.episode_sample_indices(dataset, args.episode)
    if args.frame < 0 or args.frame >= len(selected):
        raise IndexError(f"frame={args.frame} outside [0, {len(selected)})")
    _, sample_index = selected[args.frame]
    sample = dataset[sample_index]
    obs = {key: value.unsqueeze(0).to(device) for key, value in sample["obs"].items()
           if key in image_keys or key in preprocessor.required_keys}
    images = {key: obs[key][:, :preprocessor.n_obs_steps] for key in image_keys}
    outdir = args.output or checkpoint.parent.parent / "attention_visualize" / "patch_stage_diagnostic"
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = {"checkpoint": str(checkpoint), "episode": args.episode, "frame": args.frame,
               "dataset_anchor": int(selected[args.frame][0]), "cameras": {}}
    with torch.inference_mode():
        wrench, _, _ = preprocessor(obs)
        wrench_tokens = model.encode_wrench_history(wrench)
        reference, reference_masks = model(images, wrench, return_aux=True)
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                amp_wrench, _, _ = preprocessor(obs)
                amp_reference = model(images, amp_wrench)
        else:
            amp_reference = reference
        for key in display_keys:
            image = images[key][:, -1]
            camera = model.camera_embedding.weight[model.image_keys.index(key)].reshape(1, 1, -1)
            stages = patch_stages(model, images[key].flatten(0, 1), wrench_tokens, camera)
            stages = {name: value[-1:] for name, value in stages.items()}
            reconstruction_error = float((stages["rendered"] - reference[key][:, -1]).abs().max())
            if reconstruction_error > 1e-4:
                raise RuntimeError(f"Diagnostic differs from RenderNet: {reconstruction_error}")
            stage_arrays = {name: value[0].float().cpu().numpy() for name, value in stages.items()}
            stage_arrays["original"] = image[0].float().cpu().numpy()
            np.savez_compressed(outdir / f"episode_{args.episode:04d}_frame_{args.frame:04d}_{key}_stages.npz", **stage_arrays)
            residual_scale = max(float(stages["raw"].abs().quantile(0.995)),
                                 float(stages["refined"].abs().quantile(0.995)), 1e-5)
            height, width = image.shape[-2:]
            mask_min, mask_max = float(stages["mask_grid"].min()), float(stages["mask_grid"].max())
            grid_display = cv2.resize(heatmap(stages["mask_grid"][0], 0, 1), (width, height),
                                      interpolation=cv2.INTER_NEAREST)
            panel = htile([
                label(image_rgb(image[0]), "original"),
                label(to_rgb(stages["raw"][0], 0.5 / residual_scale), f"raw tiled x{0.5 / residual_scale:.2g}"),
                label(to_rgb(stages["refined"][0], 0.5 / residual_scale), f"after refine x{0.5 / residual_scale:.2g}"),
                label(to_rgb(stages["bounded"][0], 0.5), "tanh residual | [-1,1]"),
            ])
            panel2 = htile([
                label(grid_display, f"12x16 mask [{mask_min:.3f},{mask_max:.3f}]"),
                label(heatmap(stages["mask"][0], 0, 1), "mask bilinear | black=0 white=1"),
                label(to_rgb(stages["gated"][0], 1), "gated residual x1"),
                label(to_rgb((stages["rendered"] - image)[0], 1), "actual RGB delta x1"),
            ])
            smooth3 = (image + model.residual_scale * stages["gate"] * blurred(stages["bounded"], 3, 1)).clamp(0, 1)
            smooth9 = (image + model.residual_scale * stages["gate"] * blurred(stages["bounded"], 9, 2)).clamp(0, 1)
            panel3 = htile([
                label(image_rgb(stages["rendered"][0]), "rendered with refine"),
                label(image_rgb(stages["rendered_without_refine"][0]), "rendered without refine"),
                label(image_rgb(smooth3[0]), "diagnostic: residual blur 3x3"),
                label(image_rgb(smooth9[0]), "diagnostic: residual blur 9x9"),
            ])
            result = vtile([panel, panel2, panel3])
            path = outdir / f"episode_{args.episode:04d}_frame_{args.frame:04d}_{key}.png"
            cv2.imwrite(str(path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            raw = stages["raw"]
            refined = stages["refined"]
            metrics["cameras"][key] = {
                "image_shape": list(image.shape[-2:]),
                "patch_grid": list(stages["mask_grid"].shape[-2:]),
                "patch_size": int(model.backbone.patch_size),
                "raw_p99_abs": float(raw.abs().quantile(.99)),
                "refined_p99_abs": float(refined.abs().quantile(.99)),
                "raw_to_refined_mae": float((raw - refined).abs().mean()),
                "render_refine_vs_no_refine_mae": float((stages["rendered"] - stages["rendered_without_refine"]).abs().mean()),
                "mask_grid_min": float(stages["mask_grid"].min()),
                "mask_grid_max": float(stages["mask_grid"].max()),
                "mask_min": float(stages["mask"].min()),
                "mask_max": float(stages["mask"].max()),
                "mask_bilinear_vs_nearest_mae": float((stages["mask"] - stages["nearest_mask"]).abs().mean()),
                "render_vs_original_mae": float((stages["rendered"] - image).abs().mean()),
                "fp32_vs_bfloat16_render_mae": float((reference[key][:, -1] - amp_reference[key][:, -1]).abs().mean()),
                "diagnostic_reconstruction_max_error": reconstruction_error,
                "residual_stages": {name: residual_metrics(stages[name], model.backbone.patch_size)
                                    for name in ("raw", "refined", "bounded", "gated")},
                "blur3_render_vs_original_mae": float((smooth3 - image).abs().mean()),
                "blur9_render_vs_original_mae": float((smooth9 - image).abs().mean()),
            }
    (outdir / f"episode_{args.episode:04d}_frame_{args.frame:04d}_metrics.json").write_text(
        json_dumps(metrics), encoding="utf-8")
    print(f"wrote {outdir}")
    print(json_dumps(metrics))


def json_dumps(value):
    import json
    return json.dumps(value, indent=2, sort_keys=True)


if __name__ == "__main__":
    run(parse_args())
