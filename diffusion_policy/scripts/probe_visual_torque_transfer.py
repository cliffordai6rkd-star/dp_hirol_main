"""Read-only ACT checkpoint probe: torque transfer into visual output positions.

Run on held-out episodes, in CPU FP32 by default. Gray images are an explicit
out-of-distribution intervention; nonzero effects demonstrate sensitivity,
not task improvement, complete information recovery, or curriculum causality.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import json
from pathlib import Path
import time

import dill
import hydra
import numpy as np
from omegaconf import open_dict
import torch
import torch.nn.functional as F

from diffusion_policy.model.vision.contact_curriculum import gaussian_2d_smoothing
from diffusion_policy.scripts.visualize_cross_attention_episode import (
    instantiate_policy, load_checkpoint_payload,
)
from diffusion_policy.scripts.visualize_wrench_effect_episode import make_zero_wrench_obs


def split_context(context, steps, visual_count, force_count):
    expected = steps * (visual_count + force_count)
    if context.ndim != 3 or context.shape[1] != expected:
        raise ValueError(f"Expected {expected} context tokens, got {tuple(context.shape)}")
    context = context.reshape(context.shape[0], steps, visual_count + force_count, -1)
    return context[:, :, :visual_count], context[:, :, visual_count:]


def join_context(visual, force):
    return torch.cat([visual, force], dim=2).flatten(1, 2)


def rms(value):
    return value.float().square().mean().sqrt().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    output = args.output_dir or Path("model_output/diagnostics") / (
        "visual_torque_transfer_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.resolve(strict=True)
    started = time.time()
    print(f"Loading fixed checkpoint: {checkpoint}", flush=True)
    payload, cfg = load_checkpoint_payload(checkpoint)
    meta = {}
    for key in ("epoch", "optimizer_step", "global_step", "best_val_loss"):
        if key in payload.get("pickles", {}):
            value = dill.loads(payload["pickles"][key])
            if isinstance(value, (int, float, str, bool)):
                meta[key] = value
    state_key = "ema_model" if "ema_model" in payload["state_dicts"] else "model"
    policy = instantiate_policy(payload, cfg, torch.device(args.device), capture_attention=False)
    policy.eval().requires_grad_(False)
    del payload
    if policy.action_head_backbone != "act":
        raise ValueError("This probe requires the deterministic ACT action head")
    encoder = policy.obs_encoder
    if encoder.cross_attn_layers != 0 or encoder.pre_cross_self_attn_layers < 1:
        raise ValueError("This probe requires the joint self-attention teacher")
    if getattr(encoder, "fusion_output_mode", "all") != "all":
        raise ValueError("The 2x2 context-swap probe requires separate torque output slots (fusion_output_mode=all)")
    if not getattr(encoder, "fusion_upsample", True):
        raise ValueError("This saved high-resolution probe requires fusion_upsample=true")
    steps = policy.n_obs_steps
    visual_count = sum(encoder.patch_counts.values())
    force_count = encoder.wrench_history_steps
    coarse_count = encoder.fusion_tokens_per_observation - force_count
    config = copy.deepcopy(cfg.task.dataset)
    with open_dict(config):
        config.preload_images = False
        config.image_randomer_config = None
        config.video_backend = "pyav"
        config.video_decode_device = None
        config.load_result_add = "ram"
    dataset = hydra.utils.instantiate(config).get_validation_dataset()
    if not len(dataset):
        raise ValueError("No held-out validation windows")
    labels = dataset.get_phase_labels()
    anchors = np.array([dataset._anchor_index_from_sampler_row(row) for row in dataset.indices])
    episodes = dataset.episode_index[anchors]
    selected = []
    # One middle window from every available phase in each held-out episode.
    for episode in np.unique(episodes):
        for phase in (0, 1, 2):
            indices = np.flatnonzero((episodes == episode) & (labels == phase))
            if len(indices):
                selected.append((int(indices[len(indices) // 2]), int(episode), phase))
    print(f"Selected {len(selected)} windows from held-out episodes {np.unique(episodes).tolist()}", flush=True)
    rows, examples = [], {}
    repeat_max = 0.0
    inputs = []
    handle = encoder.pre_cross_self_attention.register_forward_pre_hook(
        lambda module, values: inputs.append(values[0].detach().clone())
    )
    names = {0: "free", 1: "precontact", 2: "contact"}
    try:
        with torch.inference_mode():
            for sample_number, (index, episode, phase) in enumerate(selected):
                batch = dataset[index]
                original = {key: value.unsqueeze(0).to(args.device) for key, value in batch["obs"].items()}
                demo = batch["action"].unsqueeze(0).to(args.device)
                demo = policy.normalizer["action"].normalize(demo)
                for condition in ("clear", "blur_sigma5", "constant_gray"):
                    obs = dict(original)
                    for key in encoder.image_keys:
                        if condition == "blur_sigma5":
                            obs[key] = gaussian_2d_smoothing(original[key], 5.0)
                        elif condition == "constant_gray":
                            obs[key] = torch.full_like(original[key], 0.5)
                    zero = make_zero_wrench_obs(policy, obs)
                    inputs.clear()
                    real_context = policy._encode_context(obs, apply_curriculum_mask=False)
                    zero_context = policy._encode_context(zero, apply_curriculum_mask=False)
                    if len(inputs) != 2:
                        raise RuntimeError("Expected one joint-fusion call per forward")
                    same_image_error = (inputs[0][:, :coarse_count] - inputs[1][:, :coarse_count]).abs().max().item()
                    if same_image_error != 0.0:
                        raise RuntimeError("Paired intervention changed the pre-fusion visual input")
                    if sample_number == 0:
                        repeated = policy._encode_context(obs, apply_curriculum_mask=False)
                        repeat_max = max(repeat_max, (real_context - repeated).abs().max().item())
                    v1, t1 = split_context(real_context, steps, visual_count, force_count)
                    v0, t0 = split_context(zero_context, steps, visual_count, force_count)
                    contexts = [join_context(v1, t1), join_context(v0, t1),
                                join_context(v1, t0), join_context(v0, t0)]
                    # Keep sample order [11, 01, 10, 00]; no second context normalization.
                    actions = policy.model(torch.cat(contexts, dim=0))
                    start, stop = policy.action_start_index, policy.action_start_index + policy.n_action_steps
                    a11, a01, a10, a00 = [value[:, start:stop] for value in actions.split(1)]
                    target = demo[:, start:stop]
                    row = {
                        "episode": episode, "phase": names[phase], "validation_index": index,
                        "anchor_frame": int(anchors[index]), "condition": condition,
                        "raw_torque_rms": rms(original[policy.wrench_key]),
                        "prefusion_visual_max_abs_change": same_image_error,
                        "visual_delta_rms": rms(v1 - v0),
                        "visual_relative_l2": float((v1 - v0).norm() / v1.norm().clamp_min(1e-12)),
                        "visual_patch_cosine_mean": F.cosine_similarity(v1, v0, dim=-1).mean().item(),
                        "torque_delta_rms": rms(t1 - t0),
                        "action_visual_effect_torque_real_rms": rms(a11 - a01),
                        "action_visual_effect_torque_zero_rms": rms(a10 - a00),
                        "action_torque_effect_visual_real_rms": rms(a11 - a10),
                        "action_torque_effect_visual_zero_rms": rms(a01 - a00),
                        "action_total_effect_rms": rms(a11 - a00),
                        "action_mse_11": F.mse_loss(a11, target).item(),
                        "action_mse_01": F.mse_loss(a01, target).item(),
                        "action_mse_10": F.mse_loss(a10, target).item(),
                        "action_mse_00": F.mse_loss(a00, target).item(),
                    }
                    rows.append(row)
                    if phase == 2 and condition not in examples:
                        examples[condition] = {
                            "images": {key: obs[key][0, -1].cpu().numpy() for key in encoder.image_keys},
                            "delta": (v1 - v0)[0, -1].square().mean(-1).sqrt().cpu().numpy(),
                            "episode": episode,
                        }
                print(f"{sample_number + 1}/{len(selected)} episode={episode} phase={names[phase]} elapsed={time.time()-started:.1f}s", flush=True)
    finally:
        handle.remove()
    with (output / "windows.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = [key for key in rows[0] if key not in {
        "episode", "phase", "validation_index", "anchor_frame", "condition",
    }]
    summaries = {}
    for condition in ("clear", "blur_sigma5", "constant_gray"):
        subset = [row for row in rows if row["condition"] == condition]
        summaries[condition] = {key: {
            "mean": float(np.mean([row[key] for row in subset])),
            "min": float(np.min([row[key] for row in subset])),
            "max": float(np.max([row[key] for row in subset])),
        } for key in metrics}
    report = {
        "checkpoint": str(checkpoint), "checkpoint_state": state_key,
        "checkpoint_metadata": meta, "device": str(args.device), "dtype": "float32",
        "trained_in_this_probe": False, "window_count": len(selected),
        "held_out_episodes": np.unique(episodes).tolist(),
        "phase_counts": {names[p]: sum(phase == p for _, _, phase in selected) for p in names},
        "context_shape": [1, steps * (visual_count + force_count), encoder.n_emb],
        "same_input_repeat_max_abs_error": repeat_max,
        "action_units": "normalized action; executed chunk only",
        "limitations": ["Windows within each episode are not independent statistical samples.",
                        "Phase-balanced selected-window means are not full-validation-set means.",
                        "Gray and mixed contexts may be out of distribution.",
                        "Sensitivity is not task improvement or complete torque recovery.",
                        "This is not a checkpoint from the initial 10% of training."],
        "summaries": summaries, "elapsed_seconds": time.time() - started,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conditions = list(summaries)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    axes[0].boxplot([[row["visual_relative_l2"] for row in rows if row["condition"] == c] for c in conditions], tick_labels=["Clear", "Blur sigma=5", "Constant gray"])
    axes[0].set_ylabel("Relative visual-token change (L2)")
    x = np.arange(3)
    for offset, key, label in [(-.18, "action_visual_effect_torque_real_rms", "Visual path; torque slots fixed"),
                               (.18, "action_torque_effect_visual_real_rms", "Torque slots; visual slots fixed")]:
        axes[1].bar(x + offset, [summaries[c][key]["mean"] for c in conditions], .36, label=label)
    axes[1].set_xticks(x, ["Clear", "Blur sigma=5", "Constant gray"])
    axes[1].set_ylabel("Normalized action change (RMS)")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(f"Fixed-image torque intervention: {len(selected)} held-out windows")
    fig.tight_layout()
    fig.savefig(output / "summary.png", dpi=160)
    plt.close(fig)
    if examples:
        fig, axes = plt.subplots(3, 4, figsize=(10, 7.5))
        limit = max(float(example["delta"].max()) for example in examples.values())
        for row_index, condition in enumerate(conditions):
            example = examples[condition]
            offset = 0
            for camera_index, key in enumerate(encoder.image_keys):
                count = encoder.patch_counts[key]
                grid = encoder.patch_grid_shapes[key]
                axes[row_index, camera_index * 2].imshow(example["images"][key].transpose(1, 2, 0))
                plot = axes[row_index, camera_index * 2 + 1].imshow(example["delta"][offset:offset + count].reshape(grid), vmin=0, vmax=limit, cmap="magma")
                axes[row_index, camera_index * 2].set_title(f"{condition}: {key}", fontsize=9)
                axes[row_index, camera_index * 2 + 1].set_title("Visual-token RMS change", fontsize=9)
                offset += count
        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle(f"Same image, real vs physical-zero torque; episode {next(iter(examples.values()))['episode']}")
        fig.tight_layout()
        fig.savefig(output / "example_visual_effect.png", dpi=150)
        plt.close(fig)
    print(json.dumps({"output": str(output), "repeat_error": repeat_max,
                      "means": {c: {key: values["mean"] for key, values in summary.items()}
                                for c, summary in summaries.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
