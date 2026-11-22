"""Vector rebuild of the CIVI method overview, with notation aligned to Sec. 3 (Method).

Raster inputs (all real data, no synthesis except the illustrative render row):
  * board dataset episode 54 torque  -> panel (a) s_t / b_t / rho_t rows
  * hand-free side & wrist frames     -> I_t thumbnails           (images/4.png in the chat; copied to assets/)
  * side_mask_overlay.png             -> M_t heatmap
  * assets/render_zoom_row.png        -> contact-informed RGB row (illustrative render)
  * method_overview_cropped.png       -> task-success frames (from the previous figure)

Outputs: method_overview_v2.{svg,pdf,png}
"""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = HERE / "assets"
plt.rcParams.update({
    "font.family": ["Noto Sans", "DejaVu Sans"],
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

# ---------------------------------------------------------------- palette
INK, SOFT, FAINT = "#262A31", "#5F6673", "#9AA1AB"
FROZEN_FC, FROZEN_EC, FROZEN_TX = "#E3EAF8", "#5178C6", "#3D63B0"
ATTN_FC, ATTN_EC = "#FEF1CE", "#D4B45B"
MOD_FC, MOD_EC = "#E2E4E8", "#6B717B"
OUT_FC, OUT_EC = "#FFFFFF", "#30343B"
RN_FC, RN_EC = "#E9EFFA", "#8EA8DC"
FREE_C, PRE_C, CONTACT_C = "#ECEEF1", "#F6D97A", "#F4A9B3"
PRE_BG, CONTACT_BG = "#FCEFC2", "#FCDDE1"
GREEN = "#38A169"
PANEL = dict(a="#62A867", b="#D6B04E", c="#D9625B", d="#5B82D0")

W, H = 504.0, 214.0          # points; 1 data unit = 1 pt
FS_TITLE, FS_BOX, FS_MATH, FS_SMALL = 6.7, 5.3, 6.0, 4.5

fig = plt.figure(figsize=(W / 72, H / 72))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal"); ax.axis("off")


# ---------------------------------------------------------------- helpers
def rbox(x, y, w, h, fc, ec, lw=0.6, r=2.4, ls="-", z=3, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                       ls=ls, zorder=z, alpha=alpha)
    ax.add_patch(p)
    return p


def text(x, y, s, fs=FS_BOX, color=INK, ha="center", va="center", z=6, **kw):
    return ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, zorder=z, **kw)


def flake(x, y, fs=5.2):
    text(x, y, "❄", fs=fs, color="#2F7BEA", family="DejaVu Sans", fontweight="bold")


def box(x, y, w, h, lines, kind="mod", fs=FS_BOX, bold=False, frozen=False, r=2.4):
    fc, ec = dict(mod=(MOD_FC, MOD_EC), attn=(ATTN_FC, ATTN_EC), frozen=(FROZEN_FC, FROZEN_EC),
                  out=(OUT_FC, OUT_EC))[kind]
    rbox(x, y, w, h, fc, ec, lw=0.7, r=r)
    lines = [lines] if isinstance(lines, str) else lines
    n = len(lines)
    gap = min(h / (n + 0.4), 6.6)
    for i, s in enumerate(lines):
        yy = y + h / 2 + (i - (n - 1) / 2) * gap
        is_math = s.startswith("$")
        text(x + w / 2 - (2.6 if frozen and i == 0 else 0), yy, s, fs=(FS_MATH if is_math else fs),
             fontweight=("bold" if bold and not is_math else "normal"))
    if frozen:
        tw = 3.0 * len(lines[0]) if not lines[0].startswith("$") else 8
        flake(x + w / 2 + tw * 0.62 + 1.5, y + h / 2 - ((n - 1) / 2) * gap)


def arrow(p0, p1, color=INK, lw=0.6, ls="-", z=4):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>,head_length=0.55,head_width=0.32", mutation_scale=6,
                                 color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0, zorder=z,
                                 joinstyle="miter", capstyle="butt"))


def path(pts, color=INK, lw=0.6, ls="-", head=True, z=4):
    pts = np.asarray(pts, float)
    if len(pts) > 2 or not head:
        upto = len(pts) - 1 if head else len(pts)
        ax.plot(pts[:upto, 0], pts[:upto, 1], color=color, lw=lw, ls=ls, zorder=z,
                solid_joinstyle="round", solid_capstyle="butt")
    if head:
        arrow(tuple(pts[-2]), tuple(pts[-1]), color=color, lw=lw, ls=ls, z=z)


def panel(key, x, y, w, h, title):
    rbox(x, y, w, h, "white", PANEL[key], lw=0.85, r=4.5, ls=(0, (3.2, 1.8)), z=1)
    t = text(x + 5, y + 8.4, title, fs=FS_TITLE, ha="left", fontweight="bold", z=6)
    return t


def load_rgb(path_or_arr):
    im = cv2.imread(str(path_or_arr), cv2.IMREAD_UNCHANGED) if not isinstance(path_or_arr, np.ndarray) else path_or_arr
    if im.ndim == 3 and im.shape[2] == 4:
        a = im[..., 3:4] / 255.0
        im = (im[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def image(img, x, y, w, h, r=2.2, edge="white", lw=0.8, z=3):
    art = ax.imshow(img, extent=(x, x + w, y + h, y), interpolation="lanczos", zorder=z)
    art.set_clip_path(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                     transform=ax.transData, fc="none", ec="none"))
    if edge:
        rbox(x, y, w, h, "none", edge, lw=lw, r=r, z=z + 0.1)


def stack(img, x, y, w, h, n=3, d=2.0, z=3):
    for k in range(n - 1, -1, -1):
        image(img, x - k * d, y - k * d, w, h, edge="white", lw=0.9, z=z + (n - k) * 0.3)
        rbox(x - k * d, y - k * d, w, h, "none", "#B9BEC6", lw=0.3, r=2.2, z=z + (n - k) * 0.3 + 0.2)


def circle_op(cx, cy, sym, r=3.6):
    ax.add_patch(Circle((cx, cy), r, fc="white", ec=INK, lw=0.6, zorder=5))
    k = r * 0.62
    if sym in ("+", "x"):
        if sym == "+":
            ax.plot([cx - k, cx + k], [cy, cy], color=INK, lw=0.55, zorder=6)
            ax.plot([cx, cx], [cy - k, cy + k], color=INK, lw=0.55, zorder=6)
        else:
            k *= 0.75
            ax.plot([cx - k, cx + k], [cy - k, cy + k], color=INK, lw=0.55, zorder=6)
            ax.plot([cx - k, cx + k], [cy + k, cy - k], color=INK, lw=0.55, zorder=6)


def chevron(x, y, s=2.6):
    ax.plot([x - s * 0.5, x + s * 0.45, x - s * 0.5], [y - s, y, y + s], color=FAINT, lw=0.8, zorder=5,
            solid_capstyle="round", solid_joinstyle="round")


# ---------------------------------------------------------------- data
ds = pd.read_parquet(ROOT / "data/board_25hz_3cam_lerobotv3_clean_ma5_l1_12p5/data/chunk-000/file-000.parquet")
EP, THR, K_PERSIST, FPS, PRE_S = 54, 12.5, 5, 25, 0.5
rho_all = np.stack(ds["observation.tau_ext_l1"].values).ravel()
rho = rho_all[ds.index[ds.episode_index == EP]]


def persistent(raw, k):
    out, state, on, off = np.zeros(len(raw), bool), False, 0, 0
    for i, r in enumerate(raw):
        on, off = (on + 1, 0) if r else (0, off + 1)
        if not state and on >= k:
            state = True
        elif state and off >= k:
            state = False
        out[i] = state
    return out


contact = persistent(rho > THR, K_PERSIST)
on_idx = int(np.argmax(contact)); off_idx = int(len(contact) - np.argmax(contact[::-1]))
pre_idx = max(on_idx - int(PRE_S * FPS), 0)
N = len(rho)

shot = cv2.imread(str(ASSETS / "fig4_source.png"))
side = cv2.cvtColor(shot[60:1020, 0:1279], cv2.COLOR_BGR2RGB)
wrist = cv2.cvtColor(shot[60:1020, 1279:2556], cv2.COLOR_BGR2RGB)
heat = load_rgb(ROOT / "docs/figures/method_reference/side_mask_overlay.png")
zoom_row = load_rgb(ASSETS / "render_zoom_row.png")
old = cv2.imread(str(HERE / "method_overview_cropped.png"))
succ_before = cv2.cvtColor(old[732:924, 1578:1866], cv2.COLOR_BGR2RGB)
succ_after = cv2.cvtColor(old[732:924, 1910:2196], cv2.COLOR_BGR2RGB)

# ================================================================ (a)
panel("a", 2, 2, 128, 110, "(a) External Torque Estimation")
ya = 32
text(7, ya, r"$q,\ \dot q,\ \Delta q$", fs=FS_MATH, ha="left")
arrow((31, ya), (37, ya))
box(37, ya - 7, 29, 14, "NEXT", kind="frozen", bold=True, frozen=True)
arrow((66, ya), (82.4, ya))
text(74, ya - 4.6, r"$\widehat{\tau}_{\mathrm{free}}$", fs=FS_MATH - 0.6)
circle_op(86, ya, None)
text(83.6, ya + 0.1, r"$-$", fs=4.6)
text(86, ya - 1.9, r"$+$", fs=4.0)
text(86, ya - 13.6, r"$\tau_m$", fs=FS_MATH)
arrow((86, ya - 10.4), (86, ya - 3.6))
arrow((89.6, ya), (97, ya))
text(98, ya, r"$\widehat{\tau}_{\mathrm{ext}}$", fs=FS_MATH, ha="left")

# phase rows
px0, px1 = 17.0, 125.0
X = lambda i: px0 + (px1 - px0) * i / (N - 1)
yb0, ybh = 44.0, 6.0
segs = [(0, pre_idx, FREE_C, "free"), (pre_idx, on_idx, PRE_C, ""), (on_idx, off_idx, CONTACT_C, "contact"),
        (off_idx, N - 1, FREE_C, "free")]
for a_, b_, c_, lab in segs:
    ax.add_patch(Rectangle((X(a_), yb0), X(b_) - X(a_), ybh, fc=c_, ec="none", zorder=3))
    if lab:
        text((X(a_) + X(b_)) / 2, yb0 + ybh / 2 + 0.2, lab, fs=3.9, color=SOFT)
ax.add_patch(Rectangle((px0, yb0), px1 - px0, ybh, fc="none", ec=FAINT, lw=0.35, zorder=3.5))
text((X(pre_idx) + X(on_idx)) / 2 - 0.6, yb0 - 2.6, "pre", fs=3.7, color="#9A7A12")
text(px0 - 3, yb0 + ybh / 2, r"$s_t$", fs=FS_MATH - 0.4, ha="right")

# b_t row
ybt0, ybt1 = 63.0, 55.0     # y for 0 and 1
bx = [px0, X(pre_idx), X(pre_idx), X(off_idx), X(off_idx), px1]
by = [ybt0, ybt0, ybt1, ybt1, ybt0, ybt0]
ax.plot(bx, by, color="#B8860B", lw=0.85, zorder=4, solid_joinstyle="miter")
ax.plot([px0, px1], [ybt0 + 0.9, ybt0 + 0.9], color="#E1E4E8", lw=0.3, zorder=3)
text(px0 - 3, (ybt0 + ybt1) / 2, r"$b_t$", fs=FS_MATH - 0.4, ha="right")

# rho_t plot
py0, py1 = 68.0, 106.0
ax.add_patch(Rectangle((X(pre_idx), py0), X(on_idx) - X(pre_idx), py1 - py0, fc=PRE_BG, ec="none", zorder=2))
ax.add_patch(Rectangle((X(on_idx), py0), X(off_idx) - X(on_idx), py1 - py0, fc=CONTACT_BG, ec="none", zorder=2))
ax.add_patch(Rectangle((px0, py0), px1 - px0, py1 - py0, fc="none", ec=FAINT, lw=0.35, zorder=3.5))
rmax = rho.max() * 1.08
Y = lambda v: py1 - (py1 - py0) * v / rmax
ax.plot([X(i) for i in range(N)], [Y(v) for v in rho], color=INK, lw=0.7, zorder=4, solid_joinstyle="round")
ax.plot([px0, px1], [Y(THR), Y(THR)], color=SOFT, lw=0.4, ls=(0, (2, 1.4)), zorder=4)
text(px0 + 1.2, Y(THR) - 2.4, "threshold", fs=3.6, color=SOFT, ha="left")
text(px0 - 3, (py0 + py1) / 2, r"$\rho_t$", fs=FS_MATH - 0.4, ha="right")

# ================================================================ (b)
panel("b", 134, 2, 196, 110, "(b) Action-Supervised Vision–Torque Fusion Teacher")
yt = 32
path([(113, yt), (143, yt)])
text(123, yt - 4.0, r"$\mathcal{T}_t$", fs=FS_MATH)
box(143, yt - 8, 23, 16, "LSTM")
arrow((166, yt), (184, yt))
text(173.2, yt - 4.4, r"$H^{\tau}_t$", fs=FS_MATH - 0.8)
rbox(180.5, yt - 12, 60.5, 24, "none", FAINT, lw=0.5, r=3, ls=(0, (2, 1.4)), z=2)
text(210, yt - 15.2, r"fusion $\mathcal{F}$ ($\times$2)", fs=FS_SMALL, color=SOFT)
box(184, yt - 8, 33, 16, "Cross-Attn", kind="attn")
arrow((217, yt), (221.5, yt))
ax.add_patch(Polygon([(221.5, yt - 9), (236.5, yt - 6), (236.5, yt + 6), (221.5, yt + 9)], closed=True,
                     fc=MOD_FC, ec=MOD_EC, lw=0.7, zorder=3))
text(229, yt, "FFN", fs=FS_BOX - 0.4, rotation=90)
arrow((241, yt), (250, yt))
box(250, yt - 8, 28, 16, ["Gate", r"$b_t$"])
arrow((278, yt), (288, yt))
box(288, yt - 8, 30, 16, ["Context", r"$\mathcal{Z}^T_t$"], kind="out")
arrow((303, yt + 8), (303, 56))
box(290, 56, 26, 18, ["DiT", r"$D_\psi$"], bold=True)
arrow((303, 74), (303, 86))
box(290, 86, 26, 16, ["Action", r"$A_t$"], kind="frozen")

stack(side, 145, 52, 36, 27)
text(161, 88.5, r"$I_t$", fs=FS_MATH + 0.8)


def token_grid(x0, y0, masked, cols=3, rows=2, c=4.6, g=1.2):
    for r in range(rows):
        for q in range(cols):
            xx, yy = x0 + q * (c + g), y0 + r * (c + g)
            if masked:
                rbox(xx, yy, c, c, MOD_FC, MOD_EC, lw=0.4, r=0.9, z=5)
                text(xx + c / 2, yy + c / 2 + 0.1, "M", fs=2.9, color=SOFT, fontweight="bold", z=6)
            else:
                rbox(xx, yy, c, c, ["#D3DFF5", "#E3EAF8", "#C6D5F1"][(r + q) % 3], FROZEN_EC, lw=0.4, r=0.9, z=5)


# contact-aware token masking (teacher training only), applied to P_t before fusion
mx, my = 229, 80
rbox(mx - 4, my - 4, 60, 26, "#FAFAFB", FAINT, lw=0.45, r=2.4, ls=(0, (2, 1.4)), z=2)
token_grid(mx, my, masked=False)
arrow((mx + 18.5, my + 5.8), (mx + 32, my + 5.8), color=SOFT, lw=0.5)
text(mx + 25.2, my + 2.2, r"$p_k$", fs=FS_SMALL, color=SOFT)
token_grid(mx + 33.5, my, masked=True)
text(mx + 26, my + 16.6, r"token mask $e_{\mathrm{M}}$ ($b_t\!=\!1$)", fs=FS_SMALL - 0.3, color=SOFT)
ax.plot([mx - 4, 202, 202], [my + 9, my + 9, 72.5], color=FAINT, lw=0.45, ls=(0, (1.6, 1.2)), zorder=3)
ax.add_patch(Circle((202, 72.5), 0.9, fc=SOFT, ec="none", zorder=5))
arrow((181, 65.5), (187, 65.5))
box(187, 58.5, 30, 14, "DINOv3", kind="frozen", frozen=True)
path([(201, 58.5), (201, yt + 8)])
text(203, 49.5, r"query $P_t$", fs=FS_SMALL, color=SOFT, ha="left")
path([(217, 65.5), (264, 65.5), (264, yt + 8)])
text(236, 62.6, r"$P_t$", fs=FS_MATH - 0.8, color=SOFT)

# ================================================================ (c)
panel("c", 2, 116, 328, 96, "(c) Image-Space Distillation Student")
rbox(56, 133, 182, 74, RN_FC, RN_EC, lw=0.8, r=4, z=2)
text(234, 139.5, r"RenderNet $\mathcal{R}_{\omega,\phi}$", fs=FS_BOX + 0.4, color=FROZEN_TX, ha="right",
     fontweight="bold", style="italic")

stack(side, 13, 149, 34, 25)
text(28, 141.2, r"$I_t$", fs=FS_MATH + 0.8)
path([(47, 161.5), (53, 161.5), (53, 147), (63, 147)])
path([(53, 161.5), (53, 180), (63, 180)], head=True)

yg = 147
box(63, yg - 7, 47, 14, [r"Spatial gate $G_\omega$"])
arrow((110, yg), (119, yg))
image(heat, 119, yg - 10, 26, 20, r=1.8, edge="white")
text(114.5, yg + 4.6, r"$M_t$", fs=FS_MATH - 0.8)
arrow((145, yg), (156, yg))
text(150.5, yg - 3.4, r"$>\!0.5$", fs=FS_SMALL - 0.3, color=SOFT)
text(157, yg, r"$g_t$", fs=FS_MATH, ha="left")
path([(165, yg), (196, yg), (196, 176.4)])
path([(171, yg), (171, 172)])
text(198.2, 161, r"$\cdot\, b_t$", fs=FS_MATH - 0.8, ha="left", color=SOFT)

yr = 180
ax.add_patch(Polygon([(63, yr - 10), (77, yr - 6), (77, yr + 6), (63, yr + 10)], closed=True,
                     fc=FROZEN_FC, ec=FROZEN_EC, lw=0.7, zorder=3))
text(70, yr, "CNN", fs=FS_BOX - 0.6, rotation=90)
arrow((77, yr), (82, yr))
box(82, yr - 7, 27, 14, "Self-Attn", kind="frozen")
arrow((109, yr), (115, yr))
box(115, yr - 7, 31, 14, "Cross-Attn", kind="attn")
box(119, 159.5, 23, 9.5, "LSTM", fs=FS_BOX - 0.6)
arrow((130.5, 169), (130.5, yr - 7))
text(110.5, 164.2, r"$\mathcal{T}_t$", fs=FS_MATH - 0.6, ha="right")
arrow((111.5, 164.2), (119, 164.2))
arrow((146, yr), (154, yr))
box(154, yr - 8, 34, 16, ["Decoder", "+ gated SFT"], fs=FS_BOX - 0.4)
arrow((188, yr), (192.4, yr))
text(190.2, yr + 4.6, r"$r_t$", fs=FS_MATH - 1.0, color=SOFT)
circle_op(196, yr, "x")
arrow((199.6, yr), (210.4, yr))
circle_op(214, yr, "+")
path([(28, 174), (28, 201), (214, 201), (214, yr + 3.6)])
text(120, 201, " skip ", fs=FS_SMALL, color=SOFT, bbox=dict(fc=RN_FC, ec="none", pad=0.4))
arrow((217.6, yr), (246, yr))
box(246, yr - 9, 34, 18, ["Rendered RGB", r"$\widetilde{I}_t$"], kind="out", fs=FS_BOX - 0.6)
arrow((280, yr), (287, yr))
box(287, yr - 9, 33, 18, ["DINOv3", r"$E_{\mathrm{vis}}$"], kind="frozen", fs=FS_BOX - 0.6, frozen=True)
path([(303.5, yr + 9), (303.5, 197.6)])
text(305.4, 193.2, r"$\mathcal{Z}^S_t$", fs=FS_MATH - 0.8, ha="left", color=SOFT)
rbox(291, 197.6, 25, 9.6, "white", INK, lw=0.6, r=4.8, z=5)
text(303.5, 202.6, r"$\mathcal{L}_{\mathrm{token}}$", fs=FS_MATH - 0.6)

# teacher targets (dashed): token target and gate target
DASH = (0, (2.2, 1.6))
path([(318, yt), (325.5, yt), (325.5, 202.4), (316, 202.4)], color=SOFT, ls=DASH, lw=0.55)
text(323.6, 150, r"$\mathrm{sg}(\mathcal{Z}^T_t)$", fs=FS_SMALL + 0.2, color=SOFT, ha="right")
path([(325.5, 128.5), (132, 128.5), (132, yg - 10)], color=SOFT, ls=DASH, lw=0.55)
text(226, 128.5, r" $\mathcal{L}_{\mathrm{gate}}$ (target $\bar m_t$) ", fs=FS_SMALL + 0.2, color=SOFT,
     bbox=dict(fc="white", ec="none", pad=0.3))

# ================================================================ (d)
panel("d", 334, 2, 168, 210, "(d) Contact-Informed Visuomotor Interface")
rbox(338, 16, 160, 46, "#F4F5F7", "none", r=3.5, z=2)
y1, y2 = 28, 50
text(341, y1, r"$q,\dot q,\Delta q,\tau_m$", fs=FS_MATH - 0.6, ha="left")
arrow((370, y1), (375, y1))
box(375, y1 - 7, 28, 14, "NEXT", kind="frozen", bold=True, frozen=True)
arrow((403, y1), (409, y1))
text(410, y1, r"$\mathcal{T}_t$", fs=FS_MATH, ha="left")
text(341, y2, r"$\mathcal{O}_{\leq t}$", fs=FS_MATH, ha="left")
arrow((356.5, y2), (361, y2))
box(361, y2 - 7, 42, 14, r"Phase pred. $C_\eta$", fs=FS_BOX - 0.4)
arrow((403, y2), (409, y2))
text(410, y2, r"$\widehat{b}_t$", fs=FS_MATH, ha="left")
image(side, 427, 21, 33, 24.7, r=1.8)
image(wrist, 463, 21, 33, 24.7, r=1.8)
text(461.5, 52.5, r"$I_t$", fs=FS_MATH)

arrow((418, 62), (418, 70))
text(422, 66.2, r"$(I_t,\mathcal{T}_t,\widehat{b}_t)$", fs=FS_SMALL, color=SOFT, ha="left")
rbox(380, 70, 76, 15, "white", FROZEN_EC, lw=0.9, r=3)
text(414, 77.6, r"RenderNet $\mathcal{R}_{\omega,\phi}$", fs=FS_BOX + 0.6, color=FROZEN_TX, fontweight="bold",
     style="italic")
flake(441.5, 77.4, fs=5.6)
arrow((418, 85), (418, 92))

ax.plot([340, 340], [94.6, 101.4], color=FAINT, lw=1.0, solid_capstyle="round", zorder=5)
text(343, 98, r"Contact-informed RGB  $\widetilde{I}_t$", fs=FS_BOX, color=SOFT, ha="left")
zh = 160 / (zoom_row.shape[1] / zoom_row.shape[0])
image(zoom_row, 338, 104, 160, zh, r=0, edge=None)
yz = 104 + zh
arrow((418, yz + 1.5), (418, yz + 8))
rbox(338, yz + 8, 160, 14, "#D8E1F2", "none", r=3)
text(418, yz + 15.2, "Downstream Visuomotor Policy", fs=FS_BOX + 1.0, color=INK, fontweight="bold")
ys = yz + 27
sw = 60; sh = sw * succ_before.shape[0] / succ_before.shape[1]
image(succ_before, 338, ys, sw, sh, r=2.4)
chevron(338 + sw + 4.5, ys + sh / 2)
image(succ_after, 338 + sw + 9, ys, sw, sh, r=2.4)
chevron(338 + 2 * sw + 13.5, ys + sh / 2)
cx_, cy_ = 486.5, ys + sh / 2 - 4
ax.add_patch(Circle((cx_, cy_), 6.8, fc="#E3F4EA", ec="none", zorder=5))
ax.add_patch(Circle((cx_, cy_), 5.4, fc=GREEN, ec="white", lw=0.8, zorder=6))
ax.plot([cx_ - 2.4, cx_ - 0.5, cx_ + 2.7], [cy_ + 0.1, cy_ + 2.0, cy_ - 1.9], color="white", lw=1.1,
        solid_capstyle="round", solid_joinstyle="round", zorder=7)
text(cx_, cy_ + 11, "Success", fs=FS_BOX, color=GREEN, fontweight="bold")

for ext in ("svg", "pdf", "png"):
    fig.savefig(HERE / f"method_overview_v2.{ext}", dpi=600 if ext == "png" else None)
print("episode", EP, "pre/on/off", pre_idx, on_idx, off_idx, "N", N)
