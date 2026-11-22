"""Small inline asset for panel (b): DINOv3 -> Context Token bypass with contact-gated image-token masking.

Semantics (ContactAwareImageMasker, image_mask_scope=full_context):
  * when the future window has contact, ALL DINOv3 patch tokens of every observation step are
    replaced by one learned [MASK] embedding (pixels are never changed);
  * masking probability follows the curriculum p(step): 1 on [0, 25%], linear to 0 at 75%, then 0.

Outputs (transparent background): assets/image_token_mask{,_compact}.{svg,pdf,png}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family": ["Noto Sans", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

# Palette sampled from the method figure.
BLUE_EC, BLUE_FC, BLUE_TXT = "#809DD6", "#E3EAF8", "#5178C6"
BLUE_SHADES = ["#D3DFF5", "#E3EAF8", "#C6D5F1", "#DCE5F7", "#EEF2FB"]
GRAY_FC, GRAY_EC, GRAY_TXT = "#D1D4D7", "#646A73", "#4A4F57"
PINK_FC, PINK_EC, PINK_TXT = "#FED2D9", "#E58A9A", "#B03A4E"
INK = "#333333"

OUT = Path(__file__).resolve().parent / "assets"
COLS, ROWS, CELL, GAP = 4, 3, 9.0, 2.4
GRID_W = COLS * CELL + (COLS - 1) * GAP
GRID_H = ROWS * CELL + (ROWS - 1) * GAP
STACK = [(5.0, 5.0), (2.5, 2.5), (0.0, 0.0)]  # back -> front offsets (observation steps)


def token(ax, x, y, fc, ec, text=None, lw=0.6, z=3):
    ax.add_patch(FancyBboxPatch((x, y), CELL, CELL, boxstyle="round,pad=0,rounding_size=1.6",
                                fc=fc, ec=ec, lw=lw, zorder=z))
    if text:
        ax.text(x + CELL / 2, y + CELL / 2 - 0.2, text, ha="center", va="center", fontsize=4.6,
                color=GRAY_TXT, fontweight="bold", zorder=z + 0.1)


def grid_stack(ax, x0, y0, masked):
    for depth, (dx, dy) in enumerate(STACK):
        front = depth == len(STACK) - 1
        # frame behind each layer, like the stacked image frames in the figure
        ax.add_patch(FancyBboxPatch((x0 + dx - 2.2, y0 + dy - 2.2), GRID_W + 4.4, GRID_H + 4.4,
                                    boxstyle="round,pad=0,rounding_size=2.4", fc="white",
                                    ec=GRAY_EC if masked else BLUE_EC, lw=0.6, zorder=2 + depth * 2))
        if not front:
            continue
        for r in range(ROWS):
            for c in range(COLS):
                x = x0 + dx + c * (CELL + GAP)
                y = y0 + dy + (ROWS - 1 - r) * (CELL + GAP)
                if masked:
                    token(ax, x, y, GRAY_FC, GRAY_EC, text="M", z=2 + depth * 2 + 1)
                else:
                    shade = BLUE_SHADES[(r * 7 + c * 3) % len(BLUE_SHADES)]
                    token(ax, x, y, shade, BLUE_EC, z=2 + depth * 2 + 1)


def arrow(ax, x0, x1, y, color=INK, lw=0.8):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.3", mutation_scale=9, color=color,
                                lw=lw, shrinkA=0, shrinkB=0), zorder=6)


def pill(ax, cx, cy, text, w=36, h=8.5):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}",
                                fc=PINK_FC, ec=PINK_EC, lw=0.6, zorder=6))
    ax.text(cx, cy - 0.2, text, ha="center", va="center", fontsize=4.8, color=PINK_TXT, zorder=7)


def schedule(ax, x, y, w, h):
    """Tiny curriculum curve p_mask(step): 1 -> 1 -> 0 -> 0 at 0, .25, .75, 1."""
    ax.plot([x, x, x + w], [y + h, y, y], color="#888888", lw=0.45, zorder=6, solid_capstyle="butt")
    fx = [0, .25, .75, 1.0]
    fy = [1, 1, 0, 0]
    ax.plot([x + v * w for v in fx], [y + 0.6 + v * (h - 1.6) for v in fy], color=PINK_TXT, lw=0.9,
            zorder=7, solid_joinstyle="round")
    ax.text(x - 1.2, y + h / 2, r"$p$", ha="right", va="center", fontsize=4.8, color=GRAY_TXT, zorder=7)
    ax.text(x + w / 2, y - 2.2, "train step", ha="center", va="top", fontsize=3.8, color="#777777", zorder=7)


def make(name, captions=True):
    W, H = 162, 58 if captions else 50
    fig = plt.figure(figsize=(W / 72 * 1.6, H / 72 * 1.6))  # 1 unit ~ 1.6 pt
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    gy = 14 if captions else 8
    lx, rx = 6, W - 6 - GRID_W - 5
    grid_stack(ax, lx, gy, masked=False)
    grid_stack(ax, rx, gy, masked=True)
    mid_y = gy + GRID_H / 2 + 2.5
    a0, a1 = lx + GRID_W + 8, rx - 4.5
    arrow(ax, a0, a1, mid_y)
    pill(ax, (a0 + a1) / 2, mid_y + 7.5, "future contact", w=38)
    schedule(ax, (a0 + a1) / 2 - 12, mid_y - 15.5, 25, 9.5)
    if captions:
        ax.text(lx + GRID_W / 2 + 2.5, gy - 4.2, "DINOv3 tokens", ha="center", va="top", fontsize=5.2,
                color=BLUE_TXT, style="italic")
        ax.text(rx + GRID_W / 2 + 2.5, gy - 4.2, "learned [MASK]", ha="center", va="top", fontsize=5.2,
                color=GRAY_TXT, style="italic")
    OUT.mkdir(exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", transparent=True, dpi=600 if ext == "png" else None)
    plt.close(fig)


make("image_token_mask", captions=True)
make("image_token_mask_compact", captions=False)
print("wrote", sorted(p.name for p in OUT.glob("image_token_mask*")))
