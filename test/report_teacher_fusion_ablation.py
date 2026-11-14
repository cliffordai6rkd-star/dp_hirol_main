"""Build comparable tables/curves from isolated teacher ablation artifacts."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


VARIANTS = ("baseline", "film_only", "hybrid")
LABELS = ("Baseline", "FiLM-only", "Hybrid")
COLORS = ("tab:blue", "tab:orange", "tab:green")


def report(root):
    global LABELS
    histories = {}
    summaries = {}
    for variant in VARIANTS:
        path = root / variant
        log = path / "log.jsonl"
        histories[variant] = ([json.loads(line) for line in log.read_text().splitlines() if line.strip()]
                              if log.exists() else [])
        if (path / "summary.json").exists():
            summaries[variant] = json.loads((path / "summary.json").read_text())
    validation_only = any(s.get("validation_only") for s in summaries.values())
    if validation_only:
        LABELS = ("Baseline (1000 steps)", "FiLM-only (1000 steps)", "Hybrid (FiLM untrained)")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for variant, label, color in zip(VARIANTS, LABELS, COLORS):
        train = [row for row in histories[variant] if row["kind"] == "train"]
        val = [row for row in histories[variant] if row["kind"] == "validation"]
        axes[0, 0].plot([row["step"] for row in train], [row["loss"] for row in train], color=color,
                        alpha=.7, label=label)
        for ax, scope, key in ((axes[0, 1], "all", "denoising_mse"),
                               (axes[1, 0], "contact", "denoising_mse"),
                               (axes[1, 1], "contact", "shuffle_denoising_penalty")):
            rows = [row for row in val if scope in row["metrics"]]
            ax.plot([row["step"] for row in rows], [row["metrics"][scope][key] for row in rows],
                    color=color, label=label, marker="o", markersize=3)
    for ax, title in zip(axes.flat, ("Training denoising MSE (logged batches)",
                                     "Validation denoising MSE: all five episodes",
                                     "Validation denoising MSE: contact", "Contact: shuffled minus normal MSE")):
        ax.set(title=title, xlabel="Optimizer updates", ylabel="MSE")
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(root / "training_curves.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for variant, label, color in zip(VARIANTS, LABELS, COLORS):
        rows = [row for row in histories[variant] if row["kind"] == "train"]
        for ax, key in ((axes[0, 0], "cross_attention_update_rms"), (axes[0, 1], "film_update_rms")):
            for block, style in ((0, "-"), (1, "--")):
                active = [row for row in rows if f"block{block}/{key}" in row["diagnostics"]]
                if active:
                    ax.plot([row["step"] for row in active],
                            [row["diagnostics"][f"block{block}/{key}"] for row in active],
                            linestyle=style, color=color, label=f"{label} block {block+1}")
        for ax, group in ((axes[1, 0], "cross_attention"), (axes[1, 1], "film")):
            values = [row["diagnostics"][group + "/gradient_norm"] for row in rows]
            if any(values):
                ax.plot([row["step"] for row in rows], values, color=color, label=label)
                ax.set_yscale("log")
    for ax, title in zip(axes.flat, ("Cross-attention residual RMS", "FiLM residual RMS",
                                     "Cross-attention gradient L2 (before clipping)", "FiLM gradient L2 (before clipping)")):
        ax.set(title=title, xlabel="Optimizer updates")
        if ax.lines:
            ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(root / "residuals_and_gradients.png", dpi=170)
    plt.close(fig)
    if len(summaries) != 3:
        return
    for key in ("split_sha256", "normalizer_sha256", "shuffle_sha256", "common_initialization_sha256"):
        assert len({summaries[v]["manifest"][key] for v in VARIANTS}) == 1, key
    assert len({summaries[v]["optimizer_steps"] for v in VARIANTS}) == 1
    rows = []
    for variant, label in zip(VARIANTS, LABELS):
        summary = summaries[variant]
        overall, contact = summary["final"]["all"], summary["final"]["contact"]
        rows.append(dict(variant=label, val_denoising_mse=overall["denoising_mse"],
                         contact_denoising_mse=contact["denoising_mse"],
                         context_response_rms=overall["context_response_rms"],
                         contact_response_rms=contact["context_response_rms"],
                         action_prediction_difference_rms=overall["action_prediction_difference_rms"],
                         normal_action_mse=overall["action_mse"],
                         shuffled_action_mse=overall["shuffled_action_mse"],
                         shuffle_denoising_penalty=overall["shuffle_denoising_penalty"],
                         contact_shuffle_denoising_penalty=contact["shuffle_denoising_penalty"]))
    (root / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    table = "| Variant | Val denoising MSE | Contact MSE | Context response RMS | Action difference RMS | Normal action MSE | Shuffled action MSE | Shuffle denoising penalty |\n"
    table += "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for row in rows:
        table += "| " + row["variant"] + " | " + " | ".join(f"{row[key]:.6f}" for key in (
            "val_denoising_mse", "contact_denoising_mse", "context_response_rms",
            "action_prediction_difference_rms", "normal_action_mse", "shuffled_action_mse",
            "shuffle_denoising_penalty")) + " |\n"
    (root / "comparison.md").write_text(table)
    if validation_only:
        with (root / "comparison.md").open("a") as stream:
            stream.write("\nValidation only on episodes 5, 26, 39, 45, 61. Baseline and FiLM-only use the saved 1,000-step EMA checkpoints from a planned 5,000-step schedule (training stopped during sigma=5). Hybrid loads the exact Baseline checkpoint and inserts zero-initialized, untrained FiLM; it is an equivalence check and cannot establish trained Hybrid benefit. No new optimizer updates were run.\n")
    # Windows overlap within an episode. Show paired episode-level changes
    # instead of treating 1,547 neighboring windows as independent trials.
    per_sample = {v: json.loads((root / v / "per_sample_diagnostics.json").read_text()) for v in VARIANTS}
    episodes = sorted({row["episode"] for row in per_sample["baseline"]})
    assert len(episodes) == 5
    paired = []
    for episode in episodes:
        record = {"episode": episode}
        for variant in VARIANTS:
            values = [row for row in per_sample[variant] if row["episode"] == episode]
            record[variant] = {key: float(np.mean([row[key] for row in values])) for key in (
                "denoising_mse", "shuffled_denoising_mse", "action_mse", "shuffled_action_mse")}
        paired.append(record)
    (root / "paired_episode_metrics.json").write_text(json.dumps(paired, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(episodes))
    for variant, label, color in zip(VARIANTS[1:], LABELS[1:], COLORS[1:]):
        for ax, key in zip(axes, ("denoising_mse", "action_mse")):
            ax.plot(x, [row[variant][key] - row["baseline"][key] for row in paired],
                    marker="o", color=color, label=label)
    for ax, title in zip(axes, ("Denoising MSE change vs Baseline", "Raw action MSE change vs Baseline")):
        ax.axhline(0, color="gray", linewidth=1)
        ax.set(xticks=x, xticklabels=episodes, xlabel="Held-out episode", title=title)
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(root / "paired_episode_comparison.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for variant, label, color in zip(VARIANTS, LABELS, COLORS):
        final = summaries[variant]["final"]
        phase_names = [name for name in ("free", "precontact", "contact") if name in final]
        for ax, key in zip(axes, ("denoising_mse", "context_response_rms", "shuffle_denoising_penalty")):
            ax.plot(phase_names, [final[name][key] for name in phase_names], marker="o", color=color, label=label)
    for ax, title in zip(axes, ("Final EMA denoising MSE", "Normal vs zero: context RMS", "Shuffled minus normal denoising MSE")):
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(root / "phase_diagnostics.png", dpi=170)
    plt.close(fig)
    probes = {v: torch.load(root / v / "visual_probes.pt", map_location="cpu", weights_only=False) for v in VARIANTS}
    first = {v: values[0] for v, values in probes.items()}
    assert len({p["index"] for p in first.values()}) == 1
    vmax = max(float(p["response"].max()) for p in first.values())
    fig, axes = plt.subplots(3, 4, figsize=(13, 9), constrained_layout=True)
    for row, (variant, label) in enumerate(zip(VARIANTS, LABELS)):
        probe = first[variant]
        response = probe["response"].reshape(2, 2, 14, 14)[-1]
        for camera_index, key in enumerate(("side", "wrist")):
            ax = axes[row, 2 * camera_index]
            ax.imshow(probe["images"][key][-1].permute(1, 2, 0).clamp(0, 1))
            short_label = label
            ax.set_title(f"{short_label}: {key}", fontsize=9)
            ax.axis("off")
            ax = axes[row, 2 * camera_index + 1]
            im = ax.imshow(response[camera_index], vmin=0, vmax=vmax, cmap="magma")
            ax.set_title("Context response RMS", fontsize=9)
            ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=.7)
    fig.suptitle(f"Held-out contact episode {first['baseline']['episode']}; final observation")
    fig.savefig(root / "context_response_maps.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    for row, (variant, label) in enumerate(zip(VARIANTS, LABELS)):
        probe = first[variant]
        for column, channel in enumerate((0, 3, 6)):
            ax = axes[row, column]
            ax.plot(probe["ground_truth"][:, channel], "k-", label="Ground truth")
            ax.plot(probe["action"][:, channel], label="Normal torque")
            ax.plot(probe["shuffled_action"][:, channel], "--", label="Shuffled torque")
            ax.set_title(f"{label}; action channel {channel}")
            ax.grid(alpha=.2)
            if row == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "action_interventions.png", dpi=170)
    plt.close(fig)
    print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    report(parser.parse_args().run_dir)
