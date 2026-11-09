"""Evaluate removing FiLM using teacher-token and demonstration-action errors.

By default, evaluate all windows in the student's original validation episodes.
Both branches use the same inputs, gates, weights and diffusion noise. Student
RGB is encoded through the frozen teacher's image-only path before its action
expert; teacher force fusion is used only for the token distillation target.
"""

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

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import default_collate

from diffusion_policy.common.pose_util import normalize_quaternion_xyzw
from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.policy.force_aware_diffusion_transformer_policy import normalize_pose_quaternions
from diffusion_policy.scripts import compare_render_net_film_episode as comparison


BRANCHES = comparison.BRANCHES


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, help="Default: the student's saved teacher checkpoint.")
    parser.add_argument("--episodes", type=int, nargs="+", help="Default: all original student validation episodes.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-windows", type=int, help="Optional cap for a short check.")
    parser.add_argument("--device")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.stride < 1 or (args.max_windows is not None and args.max_windows < 1):
        parser.error("--batch-size, --stride and --max-windows must be positive")
    if args.episodes is not None and any(episode < 0 for episode in args.episodes):
        parser.error("--episodes must be nonnegative")
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be nonnegative")
    return args


def action_metrics(policy, prediction, target):
    start = int(policy.action_start_index)
    stop = start + int(policy.n_action_steps)
    prediction, target = prediction[:, start:stop].float(), target[:, start:stop].float()
    if prediction.shape != target.shape or prediction.shape[1] != policy.n_action_steps:
        raise ValueError("Prediction and target must contain the same complete executed action chunk")
    result = {
        "action_mse": (prediction - target).square().flatten(1).mean(1),
        "normalized_action_mse": (
            policy.normalizer["action"].normalize(prediction)
            - policy.normalizer["action"].normalize(target)
        ).square().flatten(1).mean(1),
    }
    if target.shape[-1] in (7, 8):
        result["position_mse_m2"] = (prediction[..., :3] - target[..., :3]).square().sum(-1).mean(-1)
        predicted_q = normalize_quaternion_xyzw(prediction[..., 3:7].double())
        target_q = normalize_quaternion_xyzw(target[..., 3:7].double())
        dot = (predicted_q * target_q).sum(-1).abs().clamp(0, 1)
        result["rotation_mean_deg"] = torch.rad2deg(2 * torch.acos(dot)).mean(-1)
    if target.shape[-1] == 8:
        result["gripper_mse"] = (prediction[..., 7] - target[..., 7]).square().mean(-1)
    return result


def noise_generators(device, seed, sample_indices):
    # Stable per-window noise, independent of the evaluation batch size.
    return [torch.Generator(device=device).manual_seed(seed * 1_000_003 + int(index))
            for index in sample_indices]


def summarize(records):
    if not records:
        return {"num_windows": 0, "num_action_predictions": 0}
    metrics = [key[len("film_on_"):] for key in records[0] if key.startswith("film_on_")]
    values = {branch: {metric: float(np.mean([record[f"{branch}_{metric}"] for record in records]))
                       for metric in metrics} for branch in BRANCHES}
    for branch in BRANCHES:
        if "position_mse_m2" in values[branch]:
            values[branch]["position_rmse_mm"] = 1000 * math.sqrt(values[branch]["position_mse_m2"])
    delta = {metric: values["film_off"][metric] - values["film_on"][metric]
             for metric in values["film_on"]}
    change = {metric: 100 * delta[metric] / values["film_on"][metric]
              if values["film_on"][metric] > 0 else None for metric in delta}
    return {"num_windows": len({record["sample_index"] for record in records}),
            "num_action_predictions": len(records), **values,
            "off_minus_on": delta, "relative_change_percent": change}


@torch.inference_mode()
def evaluate_batch(policy, model, preprocessor, batch, indices, seeds, *, amp_enabled, amp_dtype,
                   use_teacher_ffn=False):
    device = next(policy.parameters()).device
    obs = {key: value.to(device) for key, value in batch["obs"].items()}
    target = batch["action"].to(device)
    # Match the FP32 teacher context targets used during context-effect training.
    with torch.autocast(device_type=device.type, enabled=False):
        teacher_tokens = policy._encode_context(obs, apply_curriculum_mask=False).float()
    autocast = torch.autocast("cuda", dtype=amp_dtype) if amp_enabled else contextlib.nullcontext()
    with autocast:
        _, branches = comparison.compare_observations(model, preprocessor, obs, list(model.image_keys))
        encode_kwargs = {"apply_fusion_ffn": True} if use_teacher_ffn else {}
        tokens = {branch: policy.obs_encoder.encode_image_context(
            branches[branch]["rendered"], **encode_kwargs,
        ) for branch in BRANCHES}
    token_mse = {branch: (tokens[branch].float() - teacher_tokens).square().flatten(1).mean(1)
                 for branch in BRANCHES}
    active = torch.stack([gate.flatten(1).any(-1) for gate in branches["film_on"]["gate"].values()]).any(0)
    records = []
    for seed in seeds:
        scores = {}
        for branch in BRANCHES:
            autocast = torch.autocast("cuda", dtype=amp_dtype) if amp_enabled else contextlib.nullcontext()
            with autocast:
                normalized = policy.conditional_sample(
                    (target.shape[0], policy.horizon, policy.action_dim), context=tokens[branch],
                    generator=noise_generators(device, seed, indices),
                )
                prediction = normalize_pose_quaternions(policy.normalizer["action"].unnormalize(normalized))
            scores[branch] = {"token_mse": token_mse[branch], **action_metrics(policy, prediction, target)}
        for row, index in enumerate(indices):
            records.append({"sample_index": int(index), "seed": seed, "render_active": bool(active[row]),
                            **{f"{branch}_{metric}": float(values[row]) for branch in BRANCHES
                               for metric, values in scores[branch].items()}})
    return records


def run(args):
    visual = comparison.visual
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    checkpoint = visual.common.choose_checkpoint(args.checkpoint, None)
    model, preprocessor, cfg, teacher_cfg, device, info = visual.load_render_net(checkpoint, args.device)
    if not model.spatial_modulation_enabled:
        raise ValueError("Evaluation requires a checkpoint trained with FiLM enabled")
    teacher_checkpoint = (args.teacher_checkpoint or Path(str(cfg.teacher_checkpoint))).expanduser().resolve()
    payload, policy_cfg = visual.common.load_checkpoint_payload(teacher_checkpoint)
    policy = visual.common.instantiate_policy(payload, policy_cfg, device, capture_attention=False).requires_grad_(False)
    teacher_state_key = "ema_model" if "ema_model" in payload["state_dicts"] else "model"
    del payload
    source_path = checkpoint.parent.parent / "teacher_source_config.yaml"
    source = OmegaConf.load(source_path) if source_path.is_file() else teacher_cfg
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(source, resolve=True))
    with open_dict(dataset_cfg):
        dataset_cfg.task.dataset.update(OmegaConf.to_container(cfg.dataset, resolve=True))
    val_ratio = float(dataset_cfg.task.dataset.get("val_ratio", 0.0))
    split_seed = int(dataset_cfg.task.dataset.get("seed", 42))
    dataset = visual.common.instantiate_episode_dataset(dataset_cfg, args.dataset_path, "pyav")
    output = args.output_dir.expanduser().resolve() if args.output_dir else checkpoint.parent.parent / "film_evaluation"
    paths = {"metrics": output / "paired_metrics.csv", "summary": output / "summary.json"}
    try:
        val_episodes = np.flatnonzero(get_val_mask(len(dataset.episode_ranges), val_ratio, split_seed)).tolist()
        episodes = list(dict.fromkeys(args.episodes)) if args.episodes is not None else val_episodes
        if not episodes:
            raise ValueError("No validation episodes in the saved split; specify --episodes")
        selected = [(episode, anchor, index) for episode in episodes
                    for anchor, index in visual.common.episode_sample_indices(dataset, episode)[::args.stride]]
        if args.max_windows is not None:
            selected = selected[:args.max_windows]
        for path in paths.values():
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"Output already exists: {path}; use --overwrite")
        output.mkdir(parents=True, exist_ok=True)
        amp_enabled = device.type == "cuda" and bool(cfg.training.amp.enabled)
        amp_dtype = visual.resolve_amp_dtype(cfg.training.amp.dtype)
        print(f"Student: {checkpoint} ({info['student_state_key']})\nTeacher: {teacher_checkpoint} ({teacher_state_key})\n"
              f"Episodes={episodes}; windows={len(selected)}; seeds={args.seeds}; device={device}", flush=True)
        records = []
        with paths["metrics"].open("w", newline="") as stream:
            table = None
            for offset in range(0, len(selected), args.batch_size):
                items = selected[offset:offset + args.batch_size]
                batch = default_collate([dataset[index] for _, _, index in items])
                batch_records = evaluate_batch(
                    policy, model, preprocessor, batch, [index for _, _, index in items], args.seeds,
                    amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                    use_teacher_ffn=bool(cfg.get("distillation", {}).get("use_teacher_ffn", False)),
                )
                locations = {index: (episode, anchor) for episode, anchor, index in items}
                for record in batch_records:
                    record["episode"], record["anchor"] = locations[record["sample_index"]]
                if table is None:
                    table = csv.DictWriter(stream, fieldnames=list(batch_records[0]))
                    table.writeheader()
                table.writerows(batch_records)
                records.extend(batch_records)
                completed = min(offset + args.batch_size, len(selected))
                if offset == 0 or completed % 100 < args.batch_size or completed == len(selected):
                    stream.flush()
                    recent = summarize(batch_records)
                    print(f"Evaluated {completed}/{len(selected)} windows; token MSE on/off="
                          f"{recent['film_on']['token_mse']:.6g}/{recent['film_off']['token_mse']:.6g}; "
                          f"action MSE on/off={recent['film_on']['action_mse']:.6g}/"
                          f"{recent['film_off']['action_mse']:.6g}", flush=True)
        summary = {
            "checkpoint": str(checkpoint), **info, "teacher_checkpoint": str(teacher_checkpoint),
            "teacher_state_key": teacher_state_key, "episodes": episodes,
            "original_validation_episodes": val_episodes, "split_seed": split_seed, "val_ratio": val_ratio,
            "seeds": args.seeds, "stride": args.stride, "max_windows": args.max_windows,
            "num_inference_steps": policy.num_inference_steps, "amp_enabled": amp_enabled,
            "ablation": "Bypass only decoder FiLM; retain identical inputs, weights, wrench and gates",
            "token_target": "FP32 frozen teacher fused context; student context uses the frozen image-only encoder",
            "action_target": "Dataset demonstration; executed future chunk only, same per-window noise across branches",
            "render_active": "At least one nonzero applied gate in any camera/observation; independent of measured error",
            "metric_units": {"action_mse": "raw mixed action units", "normalized_action_mse": "normalized action units",
                             "position_rmse_mm": "3D position distance RMSE in mm", "rotation_mean_deg": "sign-invariant quaternion angle in degrees",
                             "gripper_mse": "raw gripper units squared"},
            "all": summarize(records),
            "active": summarize([record for record in records if record["render_active"]]),
            "by_episode": {str(episode): summarize([record for record in records if record["episode"] == episode])
                           for episode in episodes},
            "by_seed": {str(seed): summarize([record for record in records if record["seed"] == seed])
                        for seed in args.seeds},
        }
        paths["summary"].write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: summary[key] for key in ("all", "active")}, indent=2), flush=True)
        print(f"Summary: {paths['summary']}", flush=True)
        return paths
    finally:
        dataset.lerobot_dataset.close()


if __name__ == "__main__":
    run(parse_args())
