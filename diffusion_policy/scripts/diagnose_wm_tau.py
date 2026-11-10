"""Paired tau interventions on fixed CAD-WM motion histories and flow noise.

Input replacements and branch removals diagnose functional dependence; mixed
motion/torque histories are not assumed physically realizable counterfactuals.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
LIBERO = ROOT.parent / "LIBERO-FT"
sys.path.insert(1, str(LIBERO))

import h5py
import numpy as np
import torch
from inference.cad_wm_provider import CADWMWrenchProvider
from inference.wrench_filter import lowpass_wrench
from diffusion_policy.model.vision.contact_curriculum import HysteresisContactDetector
from diffusion_policy.model.vision.wrench_preprocessing import compute_contact_alpha


CONDITIONS = ("original", "identity", "tau_latest", "tau_mean", "tau_reverse",
              "tau_delay_020", "torque_attention_off", "motion_attention_off")


@contextlib.contextmanager
def remove_branch(model, branch):
    hooks = []
    def zero_update(module, inputs, output):
        return (torch.zeros_like(output[0]), *output[1:])
    if branch:
        hooks = [getattr(block, branch).register_forward_hook(zero_update)
                 for block in model.flow_blocks]
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


def evaluate(args):
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / "cad-wm/outputs/cad_wm_lowpass2hz/checkpoints/step_00200000.pt"
    manifest = json.loads((ROOT / "data/libero_open_top_drawer_put_bowl_20hz_lerobot_v3/meta/wrench_conversion.json").read_text())
    sources = {v["episode_index"]: v for v in manifest["episodes"]}
    histories = {k: [] for k in ("q", "dq", "tau")}
    delayed, targets, current_force, episode, frames = [], [], [], [], []
    for ep in args.episodes:
        spec = sources[ep]
        with h5py.File(spec["source"], "r") as f:
            group = f[spec["source_group"]]
            state = {k: group[v][:, :7].astype(np.float32) for k, v in
                     {"q": "obs/joint_states", "dq": "obs/dq", "tau": "obs/tau"}.items()}
            wrench = lowpass_wrench(group["obs/wrenches"][:], np.full(len(state["q"]), .05), 2.)
            for anchor in range(13, len(wrench)-8):
                for k in histories:
                    histories[k].append(state[k][anchor-9:anchor+1])
                delayed.append(state["tau"][anchor-13:anchor-3])
                targets.append(wrench[anchor+1:anchor+9])
                current_force.append(wrench[anchor])
                episode.append(ep)
                frames.append(anchor)
    histories = {k: np.stack(v) for k, v in histories.items()}
    delayed, targets, current_force = map(np.stack, (delayed, targets, current_force))
    episode, frames = np.asarray(episode), np.asarray(frames)
    n = len(episode)
    print(f"{n} original-history windows; conditions={CONDITIONS}; seeds={args.seeds}", flush=True)
    provider = CADWMWrenchProvider(repo=ROOT/"cad-wm", checkpoint=checkpoint, device=args.device)
    detector = HysteresisContactDetector(8., 8., norm="l2", enter_frames=5, exit_frames=5)
    forecasts, probabilities, alpha = {}, {}, {}
    for seed in args.seeds:
        generator = torch.Generator(device=args.device).manual_seed(seed)
        noise = torch.randn((n, 8, 6), device=args.device, generator=generator)
        for condition in CONDITIONS:
            p_chunks, c_chunks = [], []
            branch = {"torque_attention_off": "torque_cross_attn",
                      "motion_attention_off": "motion_cross_attn"}.get(condition)
            with remove_branch(provider.predictor.model, branch):
                for start in range(0, n, args.batch_size):
                    stop = min(start+args.batch_size, n)
                    inputs = {k: v[start:stop] for k, v in histories.items()}
                    tau = inputs["tau"]
                    if condition == "identity":
                        tau = tau.copy()
                    elif condition == "tau_latest":
                        tau = np.repeat(tau[:, -1:, :], 10, axis=1)
                    elif condition == "tau_mean":
                        tau = np.repeat(tau.mean(1, keepdims=True), 10, axis=1)
                    elif condition == "tau_reverse":
                        tau = tau[:, ::-1].copy()
                    elif condition == "tau_delay_020":
                        tau = delayed[start:stop]
                    inputs["tau"] = tau
                    result = provider.predictor(inputs, source_noise=noise[start:stop])
                    p_chunks.append(result["future_wrench"].cpu().numpy())
                    c_chunks.append(result["contact_probability"].cpu().numpy())
            tag = f"{condition}_{seed}"
            forecasts[tag] = np.concatenate(p_chunks)
            probabilities[tag] = np.concatenate(c_chunks)
            # This diagnostic uses current force >8 as the initial state, not
            # the complete teacher's observed-history debounce. Label explicitly.
            mask = detector.rollout_future(torch.from_numpy(forecasts[tag][:, None]),
                torch.from_numpy((np.linalg.norm(current_force[:, :3], axis=-1) > 8)[:, None]))
            alpha[tag] = compute_contact_alpha(mask).numpy()[:, 0]
            print(f"done {tag}", flush=True)
    for seed in args.seeds:
        np.testing.assert_array_equal(forecasts[f"original_{seed}"], forecasts[f"identity_{seed}"])
        np.testing.assert_array_equal(probabilities[f"original_{seed}"], probabilities[f"identity_{seed}"])
    np.savez_compressed(args.output/"predictions.npz", episode=episode, frame=frames, target=targets,
        current_wrench=current_force, **forecasts,
        **{f"prob_{k}": v for k, v in probabilities.items()},
        **{f"alpha_{k}": v for k, v in alpha.items()})
    current_contact = np.linalg.norm(current_force[:, :3], axis=-1) > 8
    future_contact = (np.linalg.norm(targets[..., :3], axis=-1) > 8).any(-1)
    phase_masks = {"all": np.ones(n, dtype=bool), "current_contact": current_contact,
                   "precontact": (~current_contact) & future_contact,
                   "free_now_and_future": (~current_contact) & (~future_contact)}
    rows, episode_rows = [], []
    for condition in CONDITIONS:
        pred = np.stack([forecasts[f"{condition}_{s}"] for s in args.seeds])
        base = np.stack([forecasts[f"original_{s}"] for s in args.seeds])
        prob = np.stack([probabilities[f"{condition}_{s}"] for s in args.seeds])
        base_prob = np.stack([probabilities[f"original_{s}"] for s in args.seeds])
        any_contact = (np.linalg.norm(pred[..., :3], axis=-1) > 8).any(-1)
        base_contact = (np.linalg.norm(base[..., :3], axis=-1) > 8).any(-1)
        for phase, mask in phase_masks.items():
            if not mask.any():
                continue
            def summary(selection):
                return {"force_rmse_N_per_component": float(np.sqrt(np.mean((pred[:,selection,:,:3]-targets[None,selection,:,:3])**2))),
                    "force_change_rms_N_per_component": float(np.sqrt(np.mean((pred[:,selection,:,:3]-base[:,selection,:,:3])**2))),
                    "moment_change_rms_Nm_per_component": float(np.sqrt(np.mean((pred[:,selection,:,3:]-base[:,selection,:,3:])**2))),
                    "predicted_any_force_contact_fraction": float(any_contact[:,selection].mean()),
                    "force_contact_flip_fraction": float((any_contact[:,selection]!=base_contact[:,selection]).mean()),
                    "mean_abs_contact_class2_prob_change": float(np.abs(prob[:,selection,:,2]-base_prob[:,selection,:,2]).mean()),
                    "future_class2_prob_mean": float(prob[:,selection,:,2].mean())}
            rows.append({"condition": condition, "phase": phase, "windows": int(mask.sum()), **summary(mask)})
            for ep in args.episodes:
                sel = mask & (episode == ep)
                if sel.any():
                    episode_rows.append({"episode": ep, "condition": condition, "phase": phase,
                                         "windows": int(sel.sum()), **summary(sel)})
    for name, data in (("summary.csv", rows), ("per_episode.csv", episode_rows)):
        with (args.output/name).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    report = {"checkpoint": str(checkpoint), "episodes": args.episodes, "seeds": args.seeds,
        "windows": n, "split": "training episodes; not held-out", "fixed_inputs": ["q", "dq", "flow source noise"],
        "identity_control_exact": True, "conditions": {
            "original": "unchanged 10-row native history", "identity": "tau copied unchanged",
            "tau_latest": "repeat latest measured tau over all 10 history rows",
            "tau_mean": "repeat within-window mean tau over all rows",
            "tau_reverse": "reverse real tau history order",
            "tau_delay_020": "replace tau history with real history ending 4 frames earlier",
            "torque_attention_off": "zero torque cross-attention update in all flow blocks; contact head unchanged",
            "motion_attention_off": "zero motion cross-attention update in all flow blocks; contact head unchanged"},
        "contact_probability_classes": ["free", "precontact", "contact"],
        "limits": ["Mixed histories and branch removals may be outside training distribution.",
                   "Sensitivity shows functional dependence, not a physical constraint or accurate current-contact sensing.",
                   "Branch removals affect flow generation only; zero contact-head change is expected.",
                   "Current contact grouping uses measured filtered force >8 N, not geometric contact.",
                   "Alpha array uses current threshold as initial state, not full teacher-history debounce."],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "summary": rows}
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [c for c in CONDITIONS if c not in ("identity",)]
    fig, axes = plt.subplots(1,2, figsize=(14,5), layout="constrained")
    for ax, metric, title in zip(axes,
            ("force_rmse_N_per_component", "force_change_rms_N_per_component"),
            ("Force error vs recorded future (N/component)", "Force change vs original (N/component)")):
        for j, phase in enumerate(("all", "precontact", "current_contact")):
            values = [next(r[metric] for r in rows if r["condition"]==c and r["phase"]==phase) for c in selected]
            ax.bar(np.arange(len(selected))+(j-1)*.25,values,.25,label=phase)
        ax.set_xticks(range(len(selected)), selected, rotation=35, ha="right")
        ax.set_ylabel(title); ax.legend(); ax.grid(axis="y",alpha=.2)
    fig.suptitle("Paired tau interventions: fixed q/dq and flow noise; branch ablations are out-of-distribution diagnostics")
    fig.savefig(args.output/"tau_effect.png", dpi=180)
    plt.close(fig)
    print(json.dumps([r for r in rows if r["phase"]=="all"],indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=LIBERO/"outputs/inference/offline_contact_diagnostics_20261110/tau_nominal")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0,5,10,15,20,25,30,35,40,45])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42,43,44])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    evaluate(parser.parse_args())
