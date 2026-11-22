"""Paper-style analysis figures built from existing diagnostics (no training).

Run:  ~/miniconda3/envs/fdp/bin/python docs/figures/analysis/make_analysis_figures.py
Out:  docs/figures/analysis/{torque_reliance,distillation_gap}.{pdf,png}
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
REL = ROOT / "data/debug/2026.11.16/teacher_torque_reliance"

mpl.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "stixsans",
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
COND = [("zero_vals", "Torque zeroed (gate kept)", "#4C72B0"),
        ("shuffle", "Torque shuffled", "#DD8452"),
        ("bypass", "Fusion bypassed", "#8C8C8C")]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ------------------------------------------------------------------ torque reliance
def torque_reliance():
    teachers = [("A", "LIBERO\nfuture", "fit"), ("B", "Board", "val"),
                ("B_train", "Board", "fit"), ("C", "Board\nv2", "fit")]
    phases = [("pre", "Pre-contact"), ("contact", "Contact")]
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.75), sharey=True)
    w = 0.25
    for ax, (ph, ptitle) in zip(axes, phases):
        for ti, (key, name, split) in enumerate(teachers):
            m = json.load(open(REL / key / "metrics.json"))["by_phase"][ph]
            real = m["real"]["action_mse_raw"]
            for ci, (ck, _, col) in enumerate(COND):
                r = m[ck]["action_mse_raw"] / real
                ax.bar(ti + (ci - 1) * w, r, w * 0.92, color=col, lw=0, zorder=3)
        ax.axhline(1.0, color="black", lw=0.6, ls=(0, (3, 2)), zorder=2)
        ax.set_yscale("log")
        ax.set_ylim(0.7, 7.5)
        ax.set_yticks([0.8, 1, 2, 4])
        ax.set_yticklabels(["0.8", "1", "2", "4"])
        ax.minorticks_off()
        ax.set_xticks(range(len(teachers)))
        ax.set_xticklabels([f"{n}\n({s})" for _, n, s in teachers], fontsize=5.8, linespacing=1.05)
        ax.set_title(ptitle, pad=2)
        ax.grid(axis="y", lw=0.3, alpha=0.5, zorder=0)
    axes[0].set_ylabel("Action MSE / real-torque MSE")
    handles = [mpl.patches.Patch(color=c, label=l) for _, l, c in COND]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.53, 1.07), handlelength=1.0, columnspacing=0.9)
    fig.tight_layout(w_pad=0.6)
    save(fig, "torque_reliance")


# ------------------------------------------------------------------ distillation gap
RUNS = [
    ("data/outputs/2026.11.07/22.23.10_distill_gate_sft_v2/student",
     "LIBERO, future F/T (privileged)", "#8C8C8C", (0, (3, 1.5))),
    ("data/outputs/2026.11.16/06.27.19_legacy_teacher_tau_ext_b32_150k_local_5090/distillation/student",
     "Board, τ history (v2 teacher)", "#DD8452", "-"),
    ("data/outputs/2026.11.15/23.20.25_board_blur_legacy_fusion_b32_150k_local/student",
     "Board, τ history (blur teacher)", "#4C72B0", "-"),
]


def distillation_gap():
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    for path, label, col, ls in RUNS:
        rows = [json.loads(l) for l in open(ROOT / path / "logs.json.txt") if l.strip()]
        val = sorted({d["epoch"]: d for d in rows if "val_open_token_mse" in d}.items())
        ep = np.array([e for e, _ in val])
        v = np.array([d["val_open_token_mse"] for _, d in val])
        v = v / v[0]
        ax.plot(ep, v, color=col, lw=1.1, ls=ls, label=label)
        ax.plot(ep[-1], v[-1], "o", ms=2.6, color=col)
        ax.annotate(f"{v[-1]:.2f}", (ep[-1], v[-1]), xytext=(3, -1), textcoords="offset points",
                    fontsize=6, color=col, va="center")
    ax.axvspan(0, 5, color="#EEEEEE", lw=0, zorder=0)
    ax.text(9, 0.985, "gate stage", ha="left", va="center", fontsize=5.5, color="#666666")
    ax.set_xlim(0, 168)
    ax.set_ylim(0.4, 1.05)
    ax.set_xlabel("Student epoch")
    ax.set_ylabel("Remaining fusion gap\n(open-token MSE / initial)")
    ax.grid(lw=0.3, alpha=0.5)
    ax.legend(frameon=False, loc="upper right", handlelength=1.8)
    fig.tight_layout()
    save(fig, "distillation_gap")


if __name__ == "__main__":
    torque_reliance()
    distillation_gap()
    print("saved to", OUT)
