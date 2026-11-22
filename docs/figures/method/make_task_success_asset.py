"""Task-success strip for panel (d): real rollout frames (before -> contact -> after) + phase timeline.

Sources: ~/67dbba48ae486aba8dc31b42ec35b21c.mp4 (flat board, trial 1) and
         ~/bd4b88e49353023bda301acd2e77ac74.mp4 (inclined board).
Contact on/offset were read off the video frames (no force log): pre = 0.5 s before onset,
matching the phase definition in panel (a).

Outputs (transparent): assets/task_success{,_noheader}.{svg,pdf,png}
"""
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"], "pdf.fonttype": 42, "svg.fonttype": "none"})
OUT = Path(__file__).resolve().parent / "assets"

ROWS = [
    dict(label="flat board", video="~/67dbba48ae486aba8dc31b42ec35b21c.mp4",
         frames=(0, 300, 530), crop=(250, 90, 595, 320), onset=7.50, offset=17.30, end=20.40),
    dict(label="inclined board", video="~/bd4b88e49353023bda301acd2e77ac74.mp4",
         frames=(0, 330, 571), crop=(230, 70, 590, 310), onset=7.60, offset=17.00, end=19.70),
]
PRE = 0.5  # seconds before onset

# Palette (method figure).
FREE, PRE_C, CONTACT = "#E4E7EB", "#F6D97A", "#F09AA6"
BAR_EC, TXT, TXT_SOFT, LINK = "#C3C8CF", "#5F6673", "#8C929C", "#A9B0BA"
GREEN = "#38A169"
ARROW = "#9AA3AF"

TW, TH = 300.0, 200.0         # thumbnail size (units)
TGAP = 54.0                   # gap between thumbnails
RR = 16.0                     # thumbnail corner radius
BAR_DY, BAR_H = 34.0, 16.0    # bar offset below thumbnails, bar height
ROW_GAP = 40.0
LABEL_H = 30.0


def read(video, idx, crop):
    cap = cv2.VideoCapture(os.path.expanduser(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, im = cap.read()
    assert ok, (video, idx)
    x0, y0, x1, y1 = crop
    return cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2RGB), idx / fps


def shadow(ax, x, y, w, h, r, z):
    for i in range(5, 0, -1):
        g = 2.2 * i
        ax.add_patch(FancyBboxPatch((x - g, y - g + 3), w + 2 * g, h + 2 * g,
                                    boxstyle=f"round,pad=0,rounding_size={r + g}",
                                    fc="#1F2A44", ec="none", alpha=0.035, zorder=z))


def thumb(ax, img, x, y, z=2):
    shadow(ax, x, y, TW, TH, RR, z)
    art = ax.imshow(img, extent=(x, x + TW, y + TH, y), interpolation="lanczos", zorder=z + 1)
    art.set_clip_path(FancyBboxPatch((x, y), TW, TH, boxstyle=f"round,pad=0,rounding_size={RR}",
                                     transform=ax.transData, fc="none", ec="none"))
    ax.add_patch(FancyBboxPatch((x, y), TW, TH, boxstyle=f"round,pad=0,rounding_size={RR}",
                                fc="none", ec="white", lw=2.2, zorder=z + 2))


def chevron(ax, x, y):
    s = 9.0
    ax.plot([x - s * 0.55, x + s * 0.45, x - s * 0.55], [y - s, y, y + s], color=ARROW, lw=2.4,
            solid_capstyle="round", solid_joinstyle="round", zorder=5)


def check_badge(ax, cx, cy, r=24.0):
    ax.add_patch(Circle((cx, cy + 2), r + 3, fc="#1F2A44", ec="none", alpha=0.12, zorder=8))
    ax.add_patch(Circle((cx, cy), r, fc=GREEN, ec="white", lw=3.2, zorder=9))
    ax.plot([cx - 0.45 * r, cx - 0.1 * r, cx + 0.5 * r], [cy + 0.02 * r, cy + 0.38 * r, cy - 0.36 * r],
            color="white", lw=4.2, solid_capstyle="round", solid_joinstyle="round", zorder=10)


def phase_bar(ax, x, y, w, row):
    T = row["end"]
    on, off = row["onset"], row["offset"]
    segs = [(0, on - PRE, FREE), (on - PRE, on, PRE_C), (on, off, CONTACT), (off, T, FREE)]
    clip = FancyBboxPatch((x, y), w, BAR_H, boxstyle=f"round,pad=0,rounding_size={BAR_H / 2}",
                          transform=ax.transData, fc="none", ec="none")
    for a, b, c in segs:
        seg = ax.add_patch(Rectangle((x + a / T * w, y), (b - a) / T * w, BAR_H, fc=c, ec="none", zorder=3))
        seg.set_clip_path(clip)
    ax.add_patch(FancyBboxPatch((x, y), w, BAR_H, boxstyle=f"round,pad=0,rounding_size={BAR_H / 2}",
                                fc="none", ec=BAR_EC, lw=0.6, zorder=3.6))
    return T


def legend(ax, x_right, y):
    items = [("free", FREE), ("pre-contact", PRE_C), ("contact", CONTACT)]
    x = x_right
    for text, c in reversed(items):
        tw = 7.2 * len(text) + 4
        x -= tw
        ax.text(x, y, text, ha="left", va="center", fontsize=6.4, color=TXT_SOFT)
        x -= 22
        ax.add_patch(FancyBboxPatch((x, y - 6), 16, 12, boxstyle="round,pad=0,rounding_size=6",
                                    fc=c, ec=BAR_EC, lw=0.6))
        x -= 22


SUCCESS_W = 150.0             # width of the right-hand "Success" column


def success_block(ax, cx, cy, r=34.0):
    ax.add_patch(Circle((cx, cy + 3), r + 5, fc="#1F2A44", ec="none", alpha=0.10, zorder=8))
    ax.add_patch(Circle((cx, cy), r + 7, fc="#E3F4EA", ec="none", zorder=8.5))
    ax.add_patch(Circle((cx, cy), r, fc=GREEN, ec="white", lw=3.0, zorder=9))
    ax.plot([cx - 0.45 * r, cx - 0.1 * r, cx + 0.5 * r], [cy + 0.02 * r, cy + 0.38 * r, cy - 0.36 * r],
            color="white", lw=5.5, solid_capstyle="round", solid_joinstyle="round", zorder=10)
    ax.text(cx, cy + r + 30, "Success", ha="center", va="center", fontsize=10.5, color=GREEN,
            fontweight="bold", zorder=10)


def make(name, rows, header=False, labels=False, success_col=True, corner_check=False):
    strip_w = 3 * TW + 2 * TGAP
    width = strip_w + (TGAP + SUCCESS_W if success_col else 0)
    label_h = LABEL_H if labels else 0.0
    row_h = label_h + TH + BAR_DY + BAR_H + 24
    head_h = 46.0 if header else 0.0
    legend_h = 26.0
    total_h = head_h + len(rows) * row_h + (len(rows) - 1) * ROW_GAP + legend_h
    pad = 18.0
    fig = plt.figure(figsize=((width + 2 * pad) / 72 * 0.5, (total_h + 2 * pad) / 72 * 0.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-pad, width + pad); ax.set_ylim(total_h + pad, -pad)
    ax.set_aspect("equal"); ax.axis("off")

    if header:
        ax.plot([2, 2], [6, 34], color=TXT_SOFT, lw=2.2, solid_capstyle="round")
        ax.text(16, 20, "Task Success", ha="left", va="center", fontsize=11, color=TXT, fontweight="medium")

    y = head_h
    for row in rows:
        if labels:
            ax.text(0, y + label_h / 2 - 2, row["label"], ha="left", va="center", fontsize=7.6,
                    color=TXT_SOFT, style="italic")
        ty = y + label_h
        times = []
        for k, idx in enumerate(row["frames"]):
            img, t = read(row["video"], idx, row["crop"])
            times.append(t)
            tx = k * (TW + TGAP)
            thumb(ax, img, tx, ty)
            if k < 2 or success_col:
                chevron(ax, tx + TW + TGAP / 2, ty + TH / 2)
        if corner_check:
            check_badge(ax, 2 * (TW + TGAP) + TW - 10, ty + 10)
        if success_col:
            success_block(ax, strip_w + TGAP + SUCCESS_W / 2, ty + TH / 2 - 14)

        by = ty + TH + BAR_DY
        T = phase_bar(ax, 0, by, strip_w, row)
        for k, t in enumerate(times):
            tx = k * (TW + TGAP) + TW / 2
            px = min(max(t / T * strip_w, BAR_H / 2), strip_w - BAR_H / 2)
            ax.plot([tx, px], [ty + TH + 4, by - 3], color=LINK, lw=1.0, ls=(0, (2, 2)), zorder=2)
            ax.add_patch(Circle((px, by + BAR_H / 2), 5.6, fc="white", ec=TXT, lw=1.4, zorder=6))
            ha = "left" if k == 0 else ("right" if k == 2 else "center")
            lx = px - 6 if k == 0 else (px + 6 if k == 2 else px)
            ax.text(lx, by + BAR_H + 12, f"{t:.1f} s", ha=ha, va="center", fontsize=5.8,
                    color=TXT_SOFT, zorder=6)
        y += row_h + ROW_GAP
    legend(ax, strip_w, total_h - legend_h / 2)

    OUT.mkdir(exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", transparent=True, dpi=400 if ext == "png" else None)
    plt.close(fig)


FLAT, INCLINED = ROWS
make("task_success_inclined", [INCLINED])
make("task_success_flat", [FLAT])
make("task_success", ROWS, header=True, labels=True, success_col=False, corner_check=True)
make("task_success_noheader", ROWS, labels=True, success_col=False, corner_check=True)
print("wrote", sorted(p.name for p in OUT.glob("task_success*")))
