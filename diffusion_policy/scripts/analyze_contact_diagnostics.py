"""Analyze paired teacher interventions without treating overlapping windows as independent.

Force contact is a measured-force threshold proxy, not a geometric/grasp label.
Positive onset advance means an intervention crosses the pull heuristic earlier.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"gt_future": "#1f77b4", "wm": "#d62728", "current": "#2ca02c", "zero": "#666666",
          "advance_020": "#ff7f0e", "advance_035": "#9467bd", "advance_050": "#e377c2",
          "delay_020": "#17becf", "alpha_zero": "#8c564b"}
LABELS = {"gt_future": "GT future", "wm": "WM prediction", "current": "Current held", "zero": "Zero",
          "advance_020": "Advance 0.20 s", "advance_035": "Advance 0.35 s", "advance_050": "Advance 0.50 s",
          "delay_020": "Delay 0.20 s", "alpha_zero": "Alpha zero"}


def first_streak(frames, positive, count):
    """Return first frame in a confirmed contiguous streak, not confirmation frame."""
    frames, positive = np.asarray(frames), np.asarray(positive, dtype=bool)
    run = 0
    for i, active in enumerate(positive):
        run = run + 1 if active and (i == 0 or frames[i] == frames[i-1] + 1) else int(active)
        if run >= count:
            return int(frames[i-count+1])
    return None


def ci_episode(values, seed=731, count=10000):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"episodes": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(count, len(values)))].mean(1)
    return {"episodes": len(values), "mean": float(values.mean()),
            "ci95_low": float(np.quantile(means, .025)), "ci95_high": float(np.quantile(means, .975))}


def write_csv(path, rows):
    if rows:
        with path.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run(directory):
    metadata = json.loads((directory / "summary.json").read_text())
    data = np.load(directory / "predictions.npz")
    seeds = metadata["seeds"]
    conditions = list(metadata["conditions"])
    episode = data["episode"]
    frames = data["frame"]
    episodes = np.unique(episode)
    prediction = {c: np.stack([data[f"{c}_{s}"] for s in seeds]) for c in conditions}
    target, current = data["target"], data["current"]
    n = len(target)
    assert all(p.shape == (len(seeds), n, 8, 8) for p in prediction.values())
    force = np.linalg.norm(data["measured_wrench"][:, -1, -1, :3], axis=-1)
    onset = {int(ep): first_streak(frames[episode == ep], force[episode == ep] > 8, 5) for ep in episodes}
    onset_rows = []
    for ep in episodes:
        contact = onset[int(ep)]
        if contact is not None:
            sel = (episode == ep) & (frames >= contact-20) & (frames <= contact+20)
            first_frame = int(frames[sel][0])
            for mode in ("next_target_minus_measured_current", "predicted_intra_chunk_step"):
                for threshold in (1., 2., 5.):
                    signals = {c: ((p[:, :, 1, 1] - (current[None, :, 1] if mode ==
                        "next_target_minus_measured_current" else p[:, :, 0, 1])) * 1000) for c,p in prediction.items()}
                    for si, seed in enumerate(seeds):
                        base = first_streak(frames[sel], signals["gt_future"][si, sel] > threshold, 3)
                        for condition in conditions:
                            found = first_streak(frames[sel], signals[condition][si, sel] > threshold, 3)
                            onset_rows.append({"episode": int(ep), "seed": int(seed), "mode": mode,
                                "threshold_mm": threshold, "condition": condition,
                                "force_onset_frame": contact, "force_confirmation_frame": contact+4,
                                "search_first_frame": first_frame,
                                "baseline_pull_frame": base, "condition_pull_frame": found,
                                "baseline_earlier_than_force": base is not None and base < contact,
                                "baseline_left_censored": base == first_frame,
                                "condition_left_censored": found == first_frame,
                                "advance_s": None if base is None or found is None else (base-found)*.05})
    write_csv(directory / "pull_onsets.csv", onset_rows)
    onset_summary = []
    for mode in ("next_target_minus_measured_current", "predicted_intra_chunk_step"):
        for threshold in (1., 2., 5.):
            for condition in conditions:
                group = [r for r in onset_rows if r["mode"] == mode and r["threshold_mm"] == threshold and r["condition"] == condition]
                valid = [r for r in group if r["advance_s"] is not None and not r["baseline_left_censored"] and not r["condition_left_censored"]]
                vals = [np.mean([r["advance_s"] for r in valid if r["episode"] == ep]) for ep in episodes
                        if any(r["episode"] == ep for r in valid)]
                found = [r for r in group if r["condition_pull_frame"] is not None]
                onset_summary.append({"mode": mode, "threshold_mm": threshold, "condition": condition,
                    "episode_seed_pairs": len(group), "both_onsets_uncensored": len(valid),
                    "baseline_already_early": sum(r["baseline_earlier_than_force"] for r in group),
                    "baseline_left_censored": sum(r["baseline_left_censored"] for r in group),
                    "condition_left_censored": sum(r["condition_left_censored"] for r in group),
                    "condition_before_force_onset": sum(r["condition_pull_frame"] < r["force_onset_frame"] for r in found),
                    "condition_onset_mean_relative_force_s": (float(np.mean([
                        (r["condition_pull_frame"]-r["force_onset_frame"])*.05 for r in found])) if found else None),
                    "earlier_pairs": sum(r["advance_s"] > 0 for r in valid),
                    "later_pairs": sum(r["advance_s"] < 0 for r in valid),
                    "unchanged_pairs": sum(r["advance_s"] == 0 for r in valid),
                    **ci_episode(vals)})
    write_csv(directory / "pull_onset_summary.csv", onset_summary)

    errors = {}
    for c,p in prediction.items():
        delta = p - target[None]
        q, qt = p[..., 3:7], target[None, ..., 3:7]
        q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
        qt = qt / np.maximum(np.linalg.norm(qt, axis=-1, keepdims=True), 1e-8)
        errors[c] = {"position_mse_m2": np.square(delta[..., :3]).sum(-1).mean(-1),
            "rotation_deg": np.degrees(2*np.arccos(np.clip(np.abs((q*qt).sum(-1)), 0, 1))).mean(-1),
            "gripper_mse": np.square(delta[..., 7]).mean(-1),
            "action_mse": np.square(delta).mean((-1,-2))}
    episode_metrics = []
    paired = []
    onset_frame = np.asarray([onset[int(ep)] if onset[int(ep)] is not None else -1000 for ep in episode])
    masks = {"all": np.ones(n, bool), "pre_force_onset_050s": (frames >= onset_frame-10) & (frames < onset_frame),
             "first_contact_transition_pm050s": np.abs(frames-onset_frame) <= 10,
             "current_force_above_8N": force > 8}
    for phase, selection in masks.items():
        for ep in episodes:
            selected = (episode == ep) & selection
            if not selected.any():
                continue
            for c in conditions:
                episode_metrics.append({"episode": int(ep), "phase": phase, "condition": c,
                    "windows": int(selected.sum()),
                    "position_rmse_mm": float(np.sqrt(errors[c]["position_mse_m2"][:, selected].mean())*1000),
                    **{k: float(v[:, selected].mean()) for k,v in errors[c].items() if k != "position_mse_m2"}})
        for c in conditions:
            if c == "gt_future":
                continue
            for metric in ("position_rmse_mm", "rotation_deg", "gripper_mse", "action_mse"):
                differences = []
                for ep in episodes:
                    a = next((r for r in episode_metrics if r["episode"] == ep and r["phase"] == phase and r["condition"] == c), None)
                    b = next((r for r in episode_metrics if r["episode"] == ep and r["phase"] == phase and r["condition"] == "gt_future"), None)
                    if a and b:
                        differences.append(a[metric]-b[metric])
                paired.append({"phase": phase, "condition": c, "metric": metric,
                               "definition": "condition minus GT; positive is worse", **ci_episode(differences)})
    write_csv(directory / "episode_metrics.csv", episode_metrics)
    write_csv(directory / "paired_episode_differences.csv", paired)
    shifts = []
    for c in ("wm", "current", "zero", "advance_020", "advance_035", "advance_050", "delay_020"):
        delta = prediction[c][:, :, 1, :3]-prediction["gt_future"][:, :, 1, :3]
        for phase, selection in masks.items():
            metric_values = {"next_target_distance_mm": np.linalg.norm(delta, axis=-1)*1000,
                             "next_target_y_change_mm": delta[..., 1]*1000}
            for name, value in metric_values.items():
                vals = [value[:, (episode == ep) & selection].mean() for ep in episodes if ((episode == ep) & selection).any()]
                shifts.append({"condition": c, "phase": phase, "metric": name, **ci_episode(vals)})
    write_csv(directory / "paired_action_changes.csv", shifts)

    primary = ["gt_future", "wm", "current", "zero"]
    fig, axes = plt.subplots(1, 3, figsize=(13,4), layout="constrained")
    for ax, metric, title in zip(axes, ("position_rmse_mm", "rotation_deg", "gripper_mse"),
                               ("3D position RMSE (mm)", "Rotation error (deg)", "Gripper MSE")):
        for i,c in enumerate(primary):
            values = [r[metric] for r in episode_metrics if r["condition"] == c and r["phase"] == "all"]
            stats = ci_episode(values)
            ax.bar(i, stats["mean"], color=COLORS[c], alpha=.8)
            ax.errorbar(i, stats["mean"], yerr=[[stats["mean"]-stats["ci95_low"]], [stats["ci95_high"]-stats["mean"]]], color="black", capsize=4)
            ax.scatter(np.full(len(values), i)+np.linspace(-.15,.15,len(values)), values, s=12, color="black", alpha=.5)
        ax.set_xticks(range(4), [LABELS[c] for c in primary], rotation=20)
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Teacher condition intervention: equal-weight episode metrics\nDots = 10 training episodes; bars = mean; error bars = episode bootstrap 95% CI")
    fig.savefig(directory / "condition_errors.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(18,7), sharex=True, layout="constrained")
    for ep, ax in zip(episodes, axes.flat):
        contact = onset[int(ep)]
        if contact is None:
            continue
        select = (episode == ep) & (np.abs(frames-contact) <= 20)
        x = (frames[select]-contact)*.05
        for c in ("gt_future", "advance_020", "advance_035", "advance_050", "delay_020"):
            signal = (prediction[c][:,select,1,1]-current[None,select,1])*1000
            ax.plot(x, signal.mean(0), color=COLORS[c], label=LABELS[c], linewidth=1.2)
        expert = (target[select,1,1]-current[select,1])*1000
        ax.plot(x, expert, "k--", label="Recorded next step", linewidth=1)
        ax.axvline(0, color="black", alpha=.3)
        ax.axhline(2, color="black", alpha=.3, linestyle=":")
        ax.set_title(f"Episode {ep}")
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.supxlabel("Time relative to first sustained measured-force onset (s)")
    fig.supylabel("Predicted next y target minus measured current y (mm)")
    fig.suptitle("Time-shift intervention (mean of 3 paired diffusion seeds)\nPositive y is drawer-pull direction; target offset is a heuristic, not executed movement")
    fig.savefig(directory / "shift_transition_curves.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(18,7), sharex=True, layout="constrained")
    for ep, ax in zip(episodes, axes.flat):
        contact = onset[int(ep)]
        if contact is None:
            continue
        select = (episode == ep) & (np.abs(frames-contact) <= 20)
        x = (frames[select]-contact)*.05
        for c in primary:
            err = np.sqrt(errors[c]["position_mse_m2"][:,select].mean(0))*1000
            ax.plot(x, err, color=COLORS[c], label=LABELS[c], linewidth=1.3)
        ax.axvline(0, color="black", alpha=.3)
        ax.set_title(f"Episode {ep}")
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.supxlabel("Time relative to first sustained measured-force onset (s)")
    fig.supylabel("Executed-chunk 3D position RMSE (mm)")
    fig.suptitle("Condition errors around first contact (3 seeds, identical observations)\nContact proxy uses causal 2 Hz measured force > 8 N for 5 frames")
    fig.savefig(directory / "condition_transition_errors.png", dpi=180)
    plt.close(fig)
    result = {"contact_proxy": "first sample of a 5-frame >8 N streak in saved causal 2 Hz measured force; confirmation is +4 frames; not geometric contact or grasp success",
        "force_onset_frames": onset, "pull_proxy": "positive y >1/2/5 mm for 3 consecutive observation frames within onset +/-1 s; no actions executed",
        "bootstrap": "10000 resamples of episode means; seeds and overlapping windows aggregated within episode; conditional on 10 training episodes",
        "limits": ["No held-out generalization estimate", "Sensitivity is not itself harmful leakage", "Force proxy can lag geometric contact", "Onset heuristic can be left-censored or absent; these pairs are excluded from timing CI"],
        "paired_error_differences": paired, "paired_action_changes": shifts, "pull_onset_summary": onset_summary}
    (directory / "analysis.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"onset": onset,
        "overall_error_differences": [r for r in paired if r["phase"] == "all" and r["condition"] in primary and r["metric"] == "position_rmse_mm"],
        "onset_2mm": [r for r in onset_summary if r["threshold_mm"] == 2 and r["condition"].startswith("advance")],
        "precontact_y_change": [r for r in shifts if r["phase"] == "pre_force_onset_050s" and r["metric"] == "next_target_y_change_mm"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    run(parser.parse_args().directory)
