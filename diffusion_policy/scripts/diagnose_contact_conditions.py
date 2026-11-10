"""Paired offline interventions on the deployed CIVI teacher; never executes actions.

Positive shifts replace W[t+k] by W[t+k+shift], making later force arrive earlier.
All episodes were used in training: these are diagnostic, not held-out scores.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
LIBERO = ROOT.parent / "LIBERO-FT"
sys.path.insert(1, str(LIBERO))

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from diffusion_policy.scripts.compare_wrench_action_mse import instantiate_dataset
from diffusion_policy.model.vision.wrench_preprocessing import compute_contact_alpha
from inference.fdp_inference import load_dp
from inference.cad_wm_provider import CADWMWrenchProvider


CONDITIONS = {"gt_future": 0, "wm": None, "current": None, "zero": None,
              "advance_020": 4, "advance_035": 7, "advance_050": 10,
              "delay_020": -4, "alpha_zero": None}


def metrics(pred, target):
    delta = pred - target
    p2 = np.sum(delta[..., :3] ** 2, axis=-1)
    p, q = pred[..., 3:7], target[..., 3:7]
    p = p / np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-8)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    angle = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(p * q, -1)), 0, 1)))
    return {"action_mse": np.mean(delta ** 2, axis=(1, 2)),
            "position_mse_m2": p2.mean(-1), "rotation_deg": angle.mean(-1),
            "gripper_mse": (delta[..., 7] ** 2).mean(-1),
            "gripper_sign_error": ((pred[..., 7] > 0) != (target[..., 7] > 0)).mean(-1)}


def summarize(records):
    summary = []
    for phase in ("all", "free", "precontact", "contact"):
        for condition in CONDITIONS:
            rows = [r for r in records if r["condition"] == condition and
                    (phase == "all" or r["phase"] == phase)]
            if not rows:
                continue
            summary.append({"phase": phase, "condition": condition, "windows_x_seeds": len(rows),
                **{k: float(np.mean([r[k] for r in rows])) for k in
                   ("action_mse", "rotation_deg", "gripper_mse", "gripper_sign_error", "alpha")},
                "position_rmse_mm": 1000 * float(np.sqrt(np.mean([r["position_mse_m2"] for r in rows])))})
    return summary


def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "predictions.npz").exists():
        raise FileExistsError(args.output / "predictions.npz")
    checkpoint = ROOT / "data/outputs/civi/teacher/checkpoints/optimizer_step=00180000-action_mse=0.00967494.ckpt"
    wm_checkpoint = ROOT / "cad-wm/outputs/cad_wm_lowpass2hz/checkpoints/step_00200000.pt"
    policy, cfg, key = load_dp({"repo": str(ROOT), "checkpoint": str(checkpoint),
        "state_key": "ema_model", "diffusion_steps": 8, "clip_sample": False}, torch.device("cuda:0"))
    dataset = instantiate_dataset(cfg, SimpleNamespace(video_backend="pyav", dataset_path=None))
    manifest = json.loads((Path(dataset.dataset_path) / "meta/wrench_conversion.json").read_text())
    mapping = {int(v["episode_index"]): v for v in manifest["episodes"]}
    sample_rows = dataset._sequence_indices_from_sampler_rows(dataset.indices)
    obs_rows = sample_rows[:, :policy.n_obs_steps]
    anchors = obs_rows[:, -1]
    episodes = dataset.episode_index[anchors]
    selected = []
    for ep in args.episodes:
        erange = dataset.episode_ranges[ep]
        candidates = np.flatnonzero((episodes == ep) & (obs_rows[:, 0] >= erange.start + 4)
                                   & (anchors + 18 < erange.stop))
        selected.extend(candidates[::args.stride].tolist())
    selected = np.asarray(selected, dtype=np.int64)
    if args.max_samples:
        selected = selected[:args.max_samples]
    if not len(selected):
        raise ValueError("No eligible diagnostic windows")
    phases = dataset.get_phase_labels()[selected]
    rows = obs_rows[selected]
    print(f"Selected {len(selected)} windows from episodes {args.episodes}; loading WM inputs", flush=True)
    state_cache = {}
    for ep in np.unique(episodes[selected]):
        spec = mapping[int(ep)]
        with h5py.File(spec["source"], "r") as f:
            group = f[spec["source_group"]]
            state_cache[int(ep)] = {k: group[v][:, :7].astype(np.float32) for k, v in
                                   {"q": "obs/joint_states", "dq": "obs/dq", "tau": "obs/tau"}.items()}
            assert len(group["states"]) == len(dataset.episode_ranges[ep])
    histories = {k: [] for k in ("q", "dq", "tau")}
    for ep, pair in zip(episodes[selected], rows):
        first = dataset.episode_ranges[ep].start
        for row in pair - first:
            indices = np.clip(np.arange(row - 9, row + 1), 0, None)
            for k in histories:
                histories[k].append(state_cache[int(ep)][k][indices])
    histories = {k: np.stack(v) for k, v in histories.items()}
    wm = CADWMWrenchProvider(repo=ROOT / "cad-wm", checkpoint=wm_checkpoint, device="cuda:0")
    wm_predictions = {}
    for seed in args.seeds:
        gen = torch.Generator(device="cuda:0").manual_seed(seed + 900000)
        chunks = []
        for start in range(0, len(histories["q"]), 256):
            stop = min(start + 256, len(histories["q"]))
            noise = torch.randn((stop-start, 8, 6), device="cuda:0", generator=gen)
            chunk = wm.predictor({k: v[start:stop] for k, v in histories.items()}, source_noise=noise)
            chunks.append(chunk["future_wrench"].float().cpu().numpy())
        wm_predictions[seed] = np.concatenate(chunks).reshape(len(selected), 2, 8, 6)
    del wm
    torch.cuda.empty_cache()
    loader = DataLoader(Subset(dataset, selected.tolist()), batch_size=args.batch_size,
                        num_workers=0, shuffle=False)
    predictions = {f"{condition}_{seed}": [] for seed in args.seeds for condition in CONDITIONS}
    alphas = {k: [] for k in predictions}
    all_targets, all_current, all_gt_wrench, all_measured_wrench, records = [], [], [], [], []
    offset = 0
    started = time.monotonic()
    original_encode = policy.obs_encoder._encode_image_streams
    with torch.inference_mode():
        for batch in loader:
            obs = {k: v.cuda() for k, v in batch["obs"].items()}
            target = batch["action"].numpy()
            size = len(target)
            local_rows = rows[offset:offset+size]
            truth = target[:, policy.action_start_index:policy.action_start_index+policy.n_action_steps]
            assert truth.shape[1:] == (8, 8)
            current = target[:, policy.action_start_index]
            all_targets.append(truth)
            all_current.append(current)
            all_gt_wrench.append(obs[policy.wrench_key].cpu().numpy())
            all_measured_wrench.append(obs[policy.phase_wrench_key].cpu().numpy())
            cached = []
            def cached_encode(*a, **kw):
                if not cached:
                    cached.append(original_encode(*a, **kw))
                return cached[0]
            policy.obs_encoder._encode_image_streams = cached_encode
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for seed in args.seeds:
                    for condition, shift in CONDITIONS.items():
                        inputs = dict(obs)
                        inputs.pop("contact_alpha", None)
                        if condition == "wm":
                            wrench = torch.from_numpy(wm_predictions[seed][offset:offset+size]).cuda()
                        elif condition == "current":
                            wrench = obs[policy.phase_wrench_key][:, :, -1:, :].expand(-1, -1, 8, -1)
                        elif condition == "zero":
                            wrench = torch.zeros_like(obs[policy.wrench_key])
                        elif shift is not None and shift != 0:
                            wrench = torch.from_numpy(dataset.windowed_lowdim_data[policy.wrench_key][local_rows + shift]).cuda()
                        else:
                            wrench = obs[policy.wrench_key]
                        inputs[policy.wrench_key] = wrench
                        inputs[policy.future_phase_wrench_key] = wrench
                        if condition == "alpha_zero":
                            inputs["contact_alpha"] = torch.zeros((size, 2), device="cuda:0")
                        gen = torch.Generator(device="cuda:0").manual_seed(seed + offset * 1009)
                        pred = policy.predict_action(inputs, generator=gen)["action"].float().cpu().numpy()
                        _, _, current_contact, contact_mask = policy.prepare_context_inputs_with_future_contact(inputs)
                        current_contact = current_contact[:, -1].cpu().numpy()
                        alpha = compute_contact_alpha(contact_mask)[:, -1].cpu().numpy()
                        if condition == "alpha_zero":
                            alpha[:] = 0
                        if offset == 0 and seed == args.seeds[0] and condition == "gt_future":
                            policy.obs_encoder._encode_image_streams = original_encode
                            check_gen = torch.Generator(device="cuda:0").manual_seed(seed)
                            check = policy.predict_action(inputs, generator=check_gen)["action"].float().cpu().numpy()
                            np.testing.assert_allclose(pred, check, rtol=0, atol=0)
                            policy.obs_encoder._encode_image_streams = cached_encode
                        tag = f"{condition}_{seed}"
                        predictions[tag].append(pred)
                        alphas[tag].append(alpha)
                        scores = metrics(pred, truth)
                        for j in range(size):
                            idx = offset+j
                            ep = int(episodes[selected[idx]])
                            records.append({"episode": ep, "dataset_index": int(selected[idx]),
                                "frame": int(anchors[selected[idx]]-dataset.episode_ranges[ep].start),
                                "phase": ("free", "precontact", "contact")[int(phases[idx])],
                                "seed": seed, "condition": condition, "alpha": float(alpha[j]),
                                "current_contact": bool(current_contact[j]),
                                "next_delta_y_mm": float((pred[j, 1, 1]-current[j, 1])*1000),
                                **{k: float(v[j]) for k, v in scores.items()}})
            policy.obs_encoder._encode_image_streams = original_encode
            offset += size
            elapsed = time.monotonic()-started
            print(f"{offset}/{len(selected)} windows; {elapsed:.1f}s; ETA {elapsed/offset*(len(selected)-offset):.1f}s", flush=True)
    frames = np.asarray([a-dataset.episode_ranges[e].start for a,e in zip(anchors[selected], episodes[selected])])
    np.savez_compressed(args.output / "predictions.npz", selected=selected, episode=episodes[selected],
        frame=frames, phase=phases, target=np.concatenate(all_targets), current=np.concatenate(all_current),
        gt_wrench=np.concatenate(all_gt_wrench),
        measured_wrench=np.concatenate(all_measured_wrench),
        **{f"wm_wrench_{s}": w for s,w in wm_predictions.items()},
        **{k: np.concatenate(v) for k,v in predictions.items()},
        **{f"gate_alpha_{k}": np.concatenate(v) for k,v in alphas.items()})
    with (args.output / "per_window.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = {"split": "fixed in-training diagnostic episodes; no held-out set exists for these checkpoints",
        "episodes": args.episodes, "windows": len(selected), "seeds": args.seeds,
        "checkpoint": str(checkpoint), "wm_checkpoint": str(wm_checkpoint), "state_key": key,
        "diffusion_steps": 8, "clip_sample": False, "amp": "bfloat16", "stride": args.stride,
        "batch_size": args.batch_size, "paired_diffusion_noise": True,
        "vision_cache_exact_equivalence_checked": True, "conditions": CONDITIONS,
        "current_definition": "repeat each observation's latest causally filtered measured wrench 8 times",
        "shift_definition": "W(t+k+shift) supplied at t+k; positive shift advances later force; no wrapping/padding",
        "selection": "exclude early rows missing -0.2s and late rows missing +0.5s plus 8 future tokens",
        "phase_definition": "dataset labels; contact includes history OR future contact, not proof of current contact",
        "action_offsets_s": [i*.05 for i in range(8)], "wm_offsets_s": [(i+1)*.05 for i in range(8)],
        "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "summary": summarize(records)}
    (args.output / "summary.json").write_text(json.dumps(metadata, indent=2)+"\n")
    dataset.lerobot_dataset.close()
    print(json.dumps(metadata["summary"][:9], indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, nargs="+", default=[0,5,10,15,20,25,30,35,40,45])
    p.add_argument("--seeds", type=int, nargs="+", default=[42,43,44])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-samples", type=int)
    run(p.parse_args())
