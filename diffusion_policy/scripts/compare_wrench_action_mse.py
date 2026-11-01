"""Compare wrench ablations over one episode, with time and contact phases."""

from __future__ import annotations

if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import argparse
import contextlib
import csv
import json
import math
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from diffusion_policy.common.pose_util import normalize_quaternion_xyzw
from diffusion_policy.scripts import visualize_cross_attention_episode as common
from diffusion_policy.workspace.train_force_aware_diffusion_workspace import resolve_amp_dtype


DEFAULT_CHECKPOINT = Path(
    "data/outputs/2026.10.20/21.54.27_train_dp_image_query_lstm_insert_usb_insert_usb_force_aware"
    "/checkpoints/best.ckpt"
)
CURVES = (
    ("random", "Uniform random wrench", "red"),
    ("zero", "Zero raw wrench", "yellow"),
    ("original", "Dataset wrench", "blue"),
)
PHASES = (
    ("free", "Non-contact", "white"),
    ("precontact", "Pre-contact", "#ffe0b2"),
    ("contact", "Contact", "#ffcdd2"),
)
POSE_METRICS = ("position_mse_m2", "position_rmse_mm", "rotation_mean_deg", "rotation_mse_deg2")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episode", type=int, required=True,
                        help="Episode index (starting at 0); evaluate its windows in time order.")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav")
    parser.add_argument("--device", help="Default: checkpoint training device, with CPU fallback.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-scale", type=float, default=20.0,
                        help="Uniform noise half-width in training-set standard deviations, "
                             "centered on each wrench dimension's training mean (default: 20).")
    parser.add_argument("--stride", type=int, default=1, help="Evaluate every Nth sample.")
    parser.add_argument("--max-samples", type=int, help="Optional cap for a short evaluation.")
    parser.add_argument("--start-time", type=float, default=3.0,
                        help="Exclude earlier episode times from the plot, CSV and mean MSE "
                             "after inference (default: 3 seconds; use 0 for the full episode).")
    parser.add_argument("--output", type=Path, help="Plot filename; CSV and JSON are saved beside it.")
    parser.add_argument("--pose-metrics", action="store_true",
                        help="Also report 3D position RMSE (mm) and rotation angle (degrees); "
                             "requires xyz in meters and xyzw quaternion actions.")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.stride < 1 or args.num_workers < 0:
        parser.error("--batch-size/--stride must be positive and --num-workers must be nonnegative")
    if args.episode < 0:
        parser.error("--episode must be nonnegative")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not math.isfinite(args.noise_scale) or args.noise_scale <= 0:
        parser.error("--noise-scale must be positive and finite")
    if not math.isfinite(args.start_time) or args.start_time < 0:
        parser.error("--start-time must be nonnegative and finite")
    return args


def instantiate_dataset(cfg, args):
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.task.dataset, resolve=True))
    with open_dict(dataset_cfg):
        dataset_cfg.preload_images = False
        # Avoid building a whole-dataset image cache or initializing CUDA in workers.
        dataset_cfg.load_result_add = "ram"
        dataset_cfg.video_decode_device = None
        dataset_cfg.image_randomer_config = None
        # Compute the existing three-stage labels. DataLoader still iterates
        # chronologically: no weighted phase sampler is used for evaluation.
        dataset_cfg.phase_resampling_enabled = True
        dataset_cfg.video_backend = args.video_backend
        if args.dataset_path is not None:
            dataset_cfg.dataset_path = str(args.dataset_path.expanduser().resolve())
        dataset_cfg.val_ratio = 0.0
        dataset_cfg.max_train_episodes = None
    return hydra.utils.instantiate(dataset_cfg)


def episode_phase_spans(dataset, episode, episode_indices):
    """Use all valid episode windows so --stride cannot move phase boundaries."""
    anchors = np.array([
        dataset._anchor_index_from_sampler_row(dataset.indices[index]) for index in episode_indices
    ])
    start = dataset.episode_ranges[episode].start
    timestamps = dataset._timestamp_seconds(dataset.timestamps)
    times = timestamps[anchors] - timestamps[start]
    labels = dataset.get_phase_labels()[episode_indices]
    changes = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, len(labels)]
    spans = []
    for first, stop in zip(changes[:-1], changes[1:]):
        phase_id = int(labels[first])
        spans.append({
            "phase_id": phase_id,
            "phase": PHASES[phase_id][0],
            "start_seconds": float(times[first]),
            "end_seconds": float(times[min(stop, len(times) - 1)]),
        })
    return spans


def action_mse_per_sample(policy, prediction, target):
    """Match val_action_mse_raw: mean over executed action steps and dimensions."""
    start = int(policy.action_start_index)
    stop = start + int(policy.n_action_steps)
    prediction = prediction[:, start:stop].float()
    target = target[:, start:stop].float()
    if prediction.shape != target.shape or target.shape[1] != policy.n_action_steps:
        raise ValueError("Predicted and demonstration action chunks must have matching shapes")
    return (prediction - target).square().flatten(1).mean(1)


def pose_errors_per_sample(policy, prediction, target):
    """Executed-chunk 3D distance and sign-invariant SO(3) angle statistics."""
    start = int(policy.action_start_index)
    stop = start + int(policy.n_action_steps)
    prediction, target = prediction[:, start:stop].double(), target[:, start:stop].double()
    if prediction.shape != target.shape or prediction.shape[1:] != (policy.n_action_steps, 7):
        raise ValueError("Pose metrics require matching executed xyz + xyzw action chunks")
    position_mse = (prediction[..., :3] - target[..., :3]).square().sum(-1).mean(-1)
    predicted_q = normalize_quaternion_xyzw(prediction[..., 3:7])
    target_q = normalize_quaternion_xyzw(target[..., 3:7])
    dot = (predicted_q * target_q).sum(-1).abs().clamp(0, 1)
    angles = torch.rad2deg(2 * torch.acos(dot))
    return {
        "position_mse_m2": position_mse,
        "position_rmse_mm": position_mse.sqrt() * 1000,
        "rotation_mean_deg": angles.mean(-1),
        "rotation_mse_deg2": angles.square().mean(-1),
    }


def random_wrench_bounds(policy, noise_scale):
    """Use training statistics so force and torque have their own physical scale."""
    stats = policy.normalizer[policy.wrench_key].get_input_stats()
    mean, std = stats["mean"].detach(), stats["std"].detach()
    return mean - noise_scale * std, mean + noise_scale * std


@torch.inference_mode()
def evaluate_batch(policy, obs, target, *, seed, noise_generator, noise_scale=20.0, pose_metrics=False):
    """Change raw wrench only; pair the same diffusion noise across conditions."""
    wrench_key = policy.wrench_key
    template = obs[wrench_key]
    low, high = (bound.to(template) for bound in random_wrench_bounds(policy, noise_scale))
    random_wrench = torch.rand(template.shape, dtype=template.dtype,
                              device=template.device, generator=noise_generator)
    random_wrench = low + random_wrench * (high - low)
    conditions = {
        "random": random_wrench,
        "zero": torch.zeros_like(obs[wrench_key]),
        "original": obs[wrench_key],
    }
    scores = {}
    for name, wrench in conditions.items():
        inputs = dict(obs)
        inputs[wrench_key] = wrench
        generator = torch.Generator(device=target.device).manual_seed(int(seed))
        prediction = policy.predict_action(inputs, generator=generator)["model_action_pred"]
        scores[name] = action_mse_per_sample(policy, prediction, target)
        if pose_metrics:
            scores.update({f"{metric}_{name}": value for metric, value in
                           pose_errors_per_sample(policy, prediction, target).items()})
    return scores


def summarize_pose_metrics(records, action_steps):
    groups = [("overall", records)] + [
        (phase, [row for row in records if row["phase"] == phase]) for phase, _, _ in PHASES
    ]
    summary = []
    for group, rows in groups:
        if not rows:
            continue
        for name, _, _ in CURVES:
            summary.append({
                "phase": group, "condition": name, "windows": len(rows),
                "predicted_poses": len(rows) * action_steps,
                # Aggregate squared errors before taking the square root.
                "position_rmse_mm": float(np.sqrt(np.mean([
                    row[f"position_mse_m2_{name}"] for row in rows])) * 1000),
                "rotation_mean_deg": float(np.mean([row[f"rotation_mean_deg_{name}"] for row in rows])),
                "rotation_rmse_deg": float(np.sqrt(np.mean([row[f"rotation_mse_deg2_{name}"] for row in rows]))),
            })
    return summary


def save_pose_plot(path, records, metadata):
    from matplotlib import pyplot as plt
    from matplotlib import patheffects

    x = [row["episode_time_seconds"] for row in records]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, layout="constrained")
    for ax, metric, label in zip(axes, ("position_rmse_mm", "rotation_mean_deg"),
                                ("3D position RMSE (mm)", "Mean rotation angle error (degrees)")):
        for span in metadata["phase_spans"]:
            left, right = max(x[0], span["start_seconds"]), min(x[-1], span["end_seconds"])
            if right > left:
                ax.axvspan(left, right, color=PHASES[span["phase_id"]][2], zorder=0)
        for span in metadata["phase_spans"][1:]:
            if x[0] < span["start_seconds"] < x[-1]:
                ax.axvline(span["start_seconds"], color="#666666", linestyle="--", linewidth=1)
        for name, title, color in CURVES:
            overall = next(row for row in metadata["pose_metrics"]["summary"]
                           if row["phase"] == "overall" and row["condition"] == name)
            line, = ax.plot(x, [row[f"{metric}_{name}"] for row in records], color=color,
                            linewidth=1.1, label=f"{title} ({overall[metric]:.3f})")
            if name == "zero":
                line.set_path_effects([patheffects.withStroke(linewidth=2.2, foreground="#a88a00")])
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left")
    axes[0].set_title(f"Wrench ablation: pose errors | episode {metadata['episode']} | {len(records)} windows")
    axes[-1].set_xlabel("Episode time (s); orange = pre-contact, red = contact")
    if len(x) > 1:
        axes[-1].set_xlim(x[0], x[-1])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_results(path, records, metadata):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib import patheffects
    from matplotlib.patches import Patch

    # Filter after inference so changing the displayed interval does not change
    # batch grouping or the paired diffusion/wrench noise for retained windows.
    start_time = metadata.get("start_time_seconds", 0.0)
    metadata["evaluated_samples"] = len(records)
    records = [row for row in records if float(row["episode_time_seconds"]) >= start_time]
    if not records:
        raise ValueError(f"No samples remain at or after episode time {start_time:g} s")
    metadata["samples"] = len(records)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_path, metadata_path = path.with_suffix(".csv"), path.with_suffix(".json")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    means = {name: float(np.mean([row[f"mse_{name}"] for row in records])) for name, _, _ in CURVES}
    metadata["mean_action_mse_raw"] = means
    if "position_mse_m2_original" in records[0]:
        metadata["pose_metrics"] = {
            "position_input_unit": "meter",
            "position_rmse_mm": "1000 * sqrt(mean over windows and executed steps of squared 3D Euclidean distance); no division by 3",
            "rotation_mean_deg": "Mean of 2 * acos(abs(dot(normalized predicted quaternion, normalized target quaternion))) in degrees",
            "rotation_rmse_deg": "Root mean square of the same rotation angles",
            "weighting": "Every window and each of its executed steps has equal weight; overlapping predictions remain separate",
            "summary": summarize_pose_metrics(records, int(metadata["n_action_steps"])),
        }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    x = [row["episode_time_seconds"] for row in records]
    fig, ax = plt.subplots(figsize=(14, 5.8), layout="constrained")
    for span in metadata["phase_spans"]:
        left = max(x[0], span["start_seconds"])
        right = min(x[-1], span["end_seconds"])
        if right > left:
            ax.axvspan(left, right, color=PHASES[span["phase_id"]][2], zorder=0)
    for span in metadata["phase_spans"][1:]:
        boundary = span["start_seconds"]
        if x[0] < boundary < x[-1]:
            ax.axvline(boundary, color="#666666", linestyle="--", linewidth=1, zorder=2)
            ax.text(boundary, 0.98, f" {boundary:.2f} s", transform=ax.get_xaxis_transform(),
                    rotation=90, va="top", ha="right", fontsize=8, color="#444444")
    for name, label, color in CURVES:
        line, = ax.plot(x, [row[f"mse_{name}"] for row in records], color=color, linewidth=1.1,
                        marker="o" if len(x) == 1 else None,
                        label=f"{label} (mean action MSE={means[name]:.6g})")
        if name == "zero":
            # Keep the requested yellow visible against a white plot background.
            line.set_path_effects([patheffects.withStroke(linewidth=2.2, foreground="#a88a00")])
    ax.set(title=f"Wrench ablation: action MSE | episode {metadata['episode']} | {len(records)} samples",
           xlabel="Episode time (s)",
           ylabel="Action MSE (raw action values)")
    if len(x) > 1:
        ax.set_xlim(x[0], x[-1])
    ax.grid(alpha=0.2)
    curves_legend = ax.legend(loc="upper left")
    ax.add_artist(curves_legend)
    ax.legend(handles=[Patch(facecolor=color, edgecolor="#999999", label=label)
                       for _, label, color in PHASES], loc="upper right")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths = {"plot": path, "csv": csv_path, "metadata": metadata_path}
    if "pose_metrics" in metadata:
        paths["pose_plot"] = path.with_name(path.stem + "_pose_metrics.png")
        paths["pose_summary"] = path.with_name(path.stem + "_pose_summary.csv")
        save_pose_plot(paths["pose_plot"], records, metadata)
        summary = metadata["pose_metrics"]["summary"]
        with paths["pose_summary"].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    return paths


def run(args):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    checkpoint = common.choose_checkpoint(args.checkpoint, None)
    payload, cfg = common.load_checkpoint_payload(checkpoint)
    device = common.resolve_device(args.device, cfg)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    policy = common.instantiate_policy(payload, cfg, device, capture_attention=False).requires_grad_(False)
    state_key = "ema_model" if "ema_model" in payload["state_dicts"] else "model"
    del payload
    amp_enabled = device.type == "cuda" and bool(OmegaConf.select(cfg, "training.amp.enabled", default=False))
    amp_dtype = resolve_amp_dtype(OmegaConf.select(cfg, "training.amp.dtype", default="bfloat16"))
    allow_tf32 = bool(OmegaConf.select(cfg, "training.amp.allow_tf32", default=False))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    source = instantiate_dataset(cfg, args)
    try:
        dataset = source
        episode_indices = np.array([
            index for _, index in common.episode_sample_indices(dataset, args.episode)
        ], dtype=np.int64)
        phase_spans = episode_phase_spans(dataset, args.episode, episode_indices)
        selected = episode_indices[::args.stride]
        if args.max_samples is not None:
            selected = selected[:args.max_samples]
        anchors = [dataset._anchor_index_from_sampler_row(dataset.indices[index]) for index in selected]
        phases = dataset.get_phase_labels()[selected]
        timestamps = dataset._timestamp_seconds(dataset.timestamps)
        loader = DataLoader(Subset(dataset, selected.tolist()), batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=False, pin_memory=device.type == "cuda",
                            generator=torch.Generator().manual_seed(args.seed))
        score_keys = [name for name, _, _ in CURVES]
        if args.pose_metrics:
            score_keys += [f"{metric}_{name}" for metric in POSE_METRICS for name, _, _ in CURVES]
        scores = {name: np.empty(len(selected), dtype=np.float64) for name in score_keys}
        noise_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        noise_low, noise_high = random_wrench_bounds(policy, args.noise_scale)
        offset = 0
        print(f"Checkpoint: {checkpoint} ({state_key})\nDevice: {device}; episode={args.episode}; "
              f"samples={len(selected)}; "
              f"wrench={policy.wrench_key}; batch_size={args.batch_size}\n"
              "Only raw wrench is replaced; phase signals and RGB stay aligned with each target.\n"
              f"Red=independent uniform wrench within training mean +/- {args.noise_scale:g} std, "
              "yellow=raw wrench zero, blue=dataset wrench.\n"
              "Legend means are action MSE averages, not wrench input means.\n"
              "Background: white=non-contact, light orange=pre-contact, light red=contact.", flush=True)
        with tqdm(total=len(selected), desc="Wrench action MSE", unit="sample", mininterval=1) as progress:
            for batch in loader:
                obs = {key: value.to(device, non_blocking=True) for key, value in batch["obs"].items()}
                target = batch["action"].to(device, non_blocking=True)
                size = len(target)
                autocast = torch.autocast("cuda", dtype=amp_dtype) if amp_enabled else contextlib.nullcontext()
                with autocast:
                    values = evaluate_batch(policy, obs, target, seed=args.seed + offset,
                                            noise_generator=noise_generator, noise_scale=args.noise_scale,
                                            pose_metrics=args.pose_metrics)
                values = torch.stack([values[name] for name in score_keys]).cpu().numpy()
                if not np.isfinite(values).all():
                    raise FloatingPointError("Non-finite action MSE")
                for row, name in zip(values, score_keys):
                    scores[name][offset:offset + size] = row
                offset += size
                progress.update(size)
        records = []
        episode_start = dataset.episode_ranges[args.episode].start
        for index, (sample_index, anchor, phase) in enumerate(zip(selected, anchors, phases)):
            records.append({
                "sample_index": index, "dataset_index": int(sample_index), "episode": args.episode,
                "anchor_frame": int(anchor),
                "episode_time_seconds": float(timestamps[anchor] - timestamps[episode_start]),
                "phase_id": int(phase), "phase": PHASES[int(phase)][0],
                **{f"mse_{name}": float(scores[name][index]) for name, _, _ in CURVES},
                **{name: float(scores[name][index]) for name in score_keys if name not in {item[0] for item in CURVES}},
            })
        path = args.output or checkpoint.parent.parent / "wrench_ablation" / f"action_mse_episode_{args.episode}.png"
        metadata = {
            "checkpoint": str(checkpoint), "state_key": state_key, "dataset_path": str(dataset.dataset_path),
            "split": "episode", "episode": args.episode,
            "samples": len(selected), "stride": args.stride, "batch_size": args.batch_size,
            "start_time_seconds": args.start_time,
            "seed": args.seed, "device": str(device), "amp_enabled": amp_enabled, "amp_dtype": str(amp_dtype),
            "wrench_key": policy.wrench_key, "intervention": "raw wrench before policy preprocessing",
            "random_noise": {
                "distribution": "uniform", "noise_scale": args.noise_scale,
                "range_definition": "training_mean +/- noise_scale * training_std, per wrench dimension",
                "space": "raw physical wrench before policy preprocessing",
                "low": noise_low.cpu().tolist(), "high": noise_high.cpu().tolist(),
                "seed": args.seed + 1, "independent_wrench_elements": True,
            },
            "zero_wrench": {"raw_value": 0.0, "before_normalization": True},
            "legend_mean": "Mean action prediction MSE across evaluated windows, not mean wrench input",
            "phase_signals": "unchanged", "paired_diffusion_noise": True,
            "action_start_index": int(policy.action_start_index), "n_action_steps": int(policy.n_action_steps),
            "metric": "val_action_mse_raw: mean squared error over executed action steps and dimensions",
            "colors": {name: color for name, _, color in CURVES},
            "phase_colors": {name: color for name, _, color in PHASES},
            "phase_spans": phase_spans,
            "phase_definition": {
                "rule": "dataset three-stage labels; contact in observed/future tau windows takes precedence",
                "source_key": dataset.phase_source_key,
                "contact_threshold": dataset.contact_threshold,
                "contact_force_dims": list(dataset.contact_force_dims),
                "contact_norm": dataset.contact_norm,
                "precontact_seconds": dataset.precontact_seconds,
            },
        }
        paths = save_results(path, records, metadata)
        print("Mean action MSE: " + json.dumps(metadata["mean_action_mse_raw"]), flush=True)
        for name, output in paths.items():
            print(f"{name}: {output}", flush=True)
        return paths
    finally:
        source.lerobot_dataset.close()


if __name__ == "__main__":
    run(parse_args())
