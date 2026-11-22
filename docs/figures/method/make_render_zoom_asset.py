"""Rounded-corner camera image + highlighted rendered region + magnified inset (for panel (d)).

Inputs : contact_texture_illustration_fig4_{side,wrist}_rendered.png (illustrative renders)
Outputs: assets/{side,wrist}_render_zoom.{svg,pdf,png} (transparent background)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch

plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"], "pdf.fonttype": 42, "svg.fonttype": "none"})

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "data/outputs/2026.11.16/final/student/attention_visualize"
OUT = Path(__file__).resolve().parent / "assets"

VIEWS = {  # circle around the rendered texture, in source-image pixels
    "side": dict(file="contact_texture_illustration_fig4_side_rendered.png", center=(602, 383), radius=205),
    "wrist": dict(file="contact_texture_illustration_fig4_wrist_rendered.png", center=(670, 692), radius=205),
}
# Palette taken from the method figure (RenderNet blue family).
ACCENT = "#5178C6"
ACCENT_SOFT = "#9DB2DE"
ROUND = 0.055        # corner radius, fraction of image height
ZOOM = 1.0           # inset side, fraction of image height (1.0 = top/bottom aligned)
GAP = 0.05           # gap between image and inset, fraction of image height
SHOW_BADGE = False   # optional magnification pill on the inset


def shadow(ax, x, y, w, h, radius, z, layers=6, spread=0.014, offset=0.008, alpha=0.035):
    """Soft drop shadow: stacked, slightly growing translucent rounded rects."""
    s = spread * h
    for i in range(layers, 0, -1):
        g = s * i / layers
        ax.add_patch(FancyBboxPatch((x - g, y - g + offset * h), w + 2 * g, h + 2 * g,
                                    boxstyle=f"round,pad=0,rounding_size={radius + g}",
                                    fc="#1F2A44", ec="none", alpha=alpha, zorder=z))


def rounded_image(ax, img, x, y, w, h, radius, z):
    art = ax.imshow(img, extent=(x, x + w, y + h, y), interpolation="lanczos", zorder=z)
    art.set_clip_path(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                     transform=ax.transData, fc="none", ec="none"))


def frame(ax, x, y, w, h, radius, z, inner="white", inner_lw=5, outer=ACCENT, outer_lw=2.8):
    for ec, lw, zz in ((inner, inner_lw, z), (outer, outer_lw, z + 0.1)):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                    fc="none", ec=ec, lw=lw, zorder=zz))


def tangent_point(p, c, r, upper):
    p, c = np.asarray(p, float), np.asarray(c, float)
    v = p - c
    a = np.arccos(r / np.linalg.norm(v))
    base = np.arctan2(v[1], v[0])
    pts = [c + r * np.array([np.cos(base + s * a), np.sin(base + s * a)]) for s in (1, -1)]
    return min(pts, key=lambda q: q[1]) if upper else max(pts, key=lambda q: q[1])


GROUP_GAP = 0.10     # gap between the side group and the wrist group (one-row layout)


def view_width(H, W):
    return W + GAP * H + ZOOM * H


def draw_view(ax, ox, file, center, radius):
    """Draw [image | magnified inset] with its left edge at x=ox. Returns the group width."""
    img = mpimg.imread(SRC / file)
    H, W = img.shape[:2]
    cx, cy = center[0] + ox, center[1]
    rr = ROUND * H
    Z = ZOOM * H
    X0, Y0 = ox + W + GAP * H, (H - Z) / 2
    zr = rr

    # Main image.
    shadow(ax, ox, 0, W, H, rr, z=1)
    rounded_image(ax, img, ox, 0, W, H, rr, z=2)

    # Connectors: thin solid lines, drawn under the circle and the inset.
    for upper, py in ((True, Y0 + zr), (False, Y0 + Z - zr)):
        t = tangent_point((X0, py), (cx, cy), radius, upper)
        ax.plot([t[0], X0], [t[1], py], color="white", lw=6, alpha=0.55, solid_capstyle="round", zorder=3)
        ax.plot([t[0], X0], [t[1], py], color=ACCENT, lw=2.6, alpha=0.9, solid_capstyle="round", zorder=3.1)

    # Highlight ring: soft outer glow, white ring, thin accent line.
    for lw, a in ((20, 0.06), (12, 0.09)):
        ax.add_patch(Circle((cx, cy), radius, fc="none", ec="#0B1530", lw=lw, alpha=a, zorder=4))
    ax.add_patch(Circle((cx, cy), radius, fc="none", ec="white", lw=7, zorder=4.2))
    ax.add_patch(Circle((cx, cy), radius + 4.5, fc="none", ec=ACCENT, lw=2.8, zorder=4.3))

    # Magnified inset.
    sx, sy = center
    crop = img[max(int(round(sy - radius)), 0):min(int(round(sy + radius)), H),
               max(int(round(sx - radius)), 0):min(int(round(sx + radius)), W)]
    shadow(ax, X0, Y0, Z, Z, zr, z=5)
    rounded_image(ax, crop, X0, Y0, Z, Z, zr, z=6)
    frame(ax, X0, Y0, Z, Z, zr, z=7)

    if SHOW_BADGE:
        mag = Z / (2 * radius)
        bw, bh = 0.2 * Z, 0.085 * Z
        bx, by = X0 + Z - bw - 0.05 * Z, Y0 + 0.05 * Z
        ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle=f"round,pad=0,rounding_size={bh / 2}",
                                    fc="white", ec=ACCENT_SOFT, lw=1.6, alpha=0.92, zorder=8))
        ax.text(bx + bw / 2, by + bh / 2, f"{mag:.1f}×", ha="center", va="center",
                fontsize=0.045 * Z * 0.72, color=ACCENT, fontweight="bold", zorder=9)
    return view_width(H, W), H


def render(out_name, views):
    """Lay the given views out in one row and save svg/pdf/png."""
    sizes = [mpimg.imread(SRC / v["file"]).shape[:2] for v in views]
    H = max(h for h, _ in sizes)
    widths = [view_width(h, w) for h, w in sizes]
    total_w = sum(widths) + GROUP_GAP * H * (len(views) - 1)
    pad = 0.05 * H
    fig = plt.figure(figsize=((total_w + 2 * pad) / 100, (H + 2 * pad) / 100))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-pad, total_w + pad); ax.set_ylim(H + pad, -pad)
    ax.set_aspect("equal"); ax.axis("off")
    ox = 0.0
    for v, w in zip(views, widths):
        draw_view(ax, ox, **v)
        ox += w + GROUP_GAP * H
    OUT.mkdir(exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"{out_name}.{ext}", transparent=True, dpi=100 if ext == "png" else None)
    plt.close(fig)


for name, view in VIEWS.items():
    render(f"{name}_render_zoom", [view])
render("render_zoom_row", [VIEWS["side"], VIEWS["wrist"]])
print("wrote", sorted(p.name for p in OUT.glob("*render_zoom*")))
