"""CIVI method overview figure (paper, full text width).

Run:  ~/miniconda3/envs/fdp/bin/python docs/figures/method/make_method_figure.py
Out:  docs/figures/method/civi_method.{pdf,png,svg}
Canvas units: 1 unit = 0.01 inch (figure is 7.16 x 4.20 in, sized for \\textwidth).
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

mpl.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

W, H = 716, 420
FS = 5.6  # base font size (pt)

# ---------------------------------------------------------------- palette
C = dict(
    train_fc="#FBE3C6", train_ec="#D9822B",
    frozen_fc="#DCE8F5", frozen_ec="#3F72AF",
    loss_fc="#F8D7D7", loss_ec="#C0392B",
    op_fc="#FFFFFF", op_ec="#555555",
    panel_fc="#F7F7F5", panel_ec="#BBBBBB",
    vis="#3A3A3A", tau="#7B4FA0", phase="#D08C00", sup="#C0392B",
    free="#C9C9C9", pre="#F2C14E", contact="#D1495B",
)
SNOW = "❄"

fig = plt.figure(figsize=(W / 100, H / 100))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


# ---------------------------------------------------------------- helpers
def panel(x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=4",
                                fc=C["panel_fc"], ec=C["panel_ec"], lw=0.6, zorder=0))
    ax.text(x + 5, y + h - 4, title, ha="left", va="top", fontsize=FS + 1.2,
            fontweight="bold", zorder=5)


def box(x, y, w, h, text, kind="train", fs=FS, sub=None, bold=False):
    fc, ec = {
        "train": (C["train_fc"], C["train_ec"]),
        "frozen": (C["frozen_fc"], C["frozen_ec"]),
        "loss": (C["loss_fc"], C["loss_ec"]),
        "op": (C["op_fc"], C["op_ec"]),
    }[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.5",
                                fc=fc, ec=ec, lw=0.7, zorder=3))
    cy = y + h / 2
    if sub:
        ax.text(x + w / 2, cy + 3.4, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", zorder=4)
        ax.text(x + w / 2, cy - 4.2, sub, ha="center", va="center", fontsize=fs - 1.0,
                color="#444444", zorder=4)
    else:
        ax.text(x + w / 2, cy, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", zorder=4)
    if kind == "frozen":
        ax.text(x + w - 1.8, y + h - 1.6, SNOW, ha="right", va="top",
                fontsize=fs - 0.4, color=C["frozen_ec"], fontfamily="DejaVu Sans", zorder=4)
    return dict(l=(x, cy), r=(x + w, cy), t=(x + w / 2, y + h), b=(x + w / 2, y),
                x=x, y=y, w=w, h=h)


def label(x, y, s, fs=FS, color="black", ha="center", va="center", **kw):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color, zorder=6, **kw)


def path(pts, color=C["vis"], ls="-", lw=0.75, head=True):
    pts = np.asarray(pts, float)
    if len(pts) > 2:
        ax.plot(pts[:-1, 0], pts[:-1, 1], color=color, lw=lw, ls=ls,
                solid_capstyle="butt", zorder=2)
    if head:
        ax.annotate("", xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                    mutation_scale=5.5, shrinkA=0, shrinkB=0), zorder=2)
    else:
        ax.plot(pts[-2:, 0], pts[-2:, 1], color=color, lw=lw, ls=ls, zorder=2)


def image(x, y, w, h, name, cmap=None, caption=None, ec="#333333"):
    p = ASSETS / name
    arr = np.load(p) if p.suffix == ".npy" else np.asarray(Image.open(p))
    ax.imshow(arr, extent=(x, x + w, y, y + h), cmap=cmap, vmin=0, vmax=1 if cmap else None,
              zorder=3, interpolation="lanczos")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=ec, lw=0.6, zorder=4))
    if caption:
        label(x + w / 2, y - 4.5, caption, fs=FS - 0.3)
    return dict(l=(x, y + h / 2), r=(x + w, y + h / 2), t=(x + w / 2, y + h), b=(x + w / 2, y))


def node(x, y, s, r=5.0, fs=FS):
    ax.add_patch(Circle((x, y), r, fc="white", ec=C["op_ec"], lw=0.7, zorder=3))
    label(x, y, s, fs=fs)
    return dict(l=(x - r, y), r=(x + r, y), t=(x, y + r), b=(x, y - r))


def phase_strip(x, y, w, h, segs):
    """segs: list of (t0, t1, phase) with t in [0,1]."""
    for t0, t1, ph in segs:
        ax.add_patch(Rectangle((x + t0 * w, y), (t1 - t0) * w, h, fc=C[ph], ec="none", zorder=3))
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="#666666", lw=0.4, zorder=4))


SEGS = [(0, 0.38, "free"), (0.38, 0.52, "pre"), (0.52, 0.84, "contact"), (0.84, 1, "free")]

# ====================================================================
# (a) External torque estimation & contact phases
# ====================================================================
panel(4, 212, 172, 204, "(a) External torque & contact phase")

label(34, 384, r"$[q,\dot{q},\Delta q]_{t-L+1:t}$", fs=FS)
nxt = box(70, 375, 38, 18, "NEXT", kind="frozen", bold=True)
path([(60.5, 384), nxt["l"]], C["tau"])
mn = node(128, 384, r"$-$", r=4.5)
path([nxt["r"], mn["l"]], C["tau"])
label(117, 389.5, r"$\hat{\tau}_{\mathrm{free}}$", fs=FS - 0.4)
label(128, 401, r"$\tau_m$", fs=FS)
path([(128, 397), mn["t"]], C["tau"])
label(155, 384, r"$\hat{\tau}_{\mathrm{ext},t}$", fs=FS)
path([mn["r"], (145, 384)], C["tau"])

# signed torque history (purple) and its L1 magnitude
gx, gy, gw, gh = 16, 296, 150, 58
t = np.linspace(0, 1, 400)
rng = np.random.default_rng(3)
on = 1 / (1 + np.exp(-(t - 0.53) * 90)) * (1 - 1 / (1 + np.exp(-(t - 0.84) * 90)))
taus = [on * a * (1 + 0.25 * np.sin(14 * t + p)) + 0.05 * rng.standard_normal(t.size)
        for a, p in [(0.9, 0), (-0.6, 1), (0.4, 2.2)]]
rho = np.sum(np.abs(taus), axis=0)
rho = np.convolve(rho, np.ones(9) / 9, mode="same")
ax.add_patch(Rectangle((gx, gy), gw, gh, fc="white", ec="#999999", lw=0.4, zorder=2))
for s0, s1, ph in SEGS:
    ax.add_patch(Rectangle((gx + s0 * gw, gy), (s1 - s0) * gw, gh, fc=C[ph], alpha=0.28,
                           ec="none", zorder=2))
ax.plot(gx + t * gw, gy + 4 + rho / rho.max() * (gh - 12), color=C["tau"], lw=0.8, zorder=3)
thr = gy + 4 + 0.32 * (gh - 12)
ax.plot([gx, gx + gw], [thr, thr], ls=(0, (2, 1.5)), color="#555555", lw=0.5, zorder=3)
label(gx + 3, thr + 4, "threshold + persistence", fs=FS - 1.3, ha="left", color="#555555")
label(gx + 3, gy + gh - 5, r"$\rho_t=\|\hat{\tau}_{\mathrm{ext},t}\|_1$", fs=FS - 0.3,
      ha="left", color=C["tau"])
path([(155, 377), (155, 358)], C["tau"])

# phase labels + switch
py = 272
phase_strip(gx, py, gw, 9, SEGS)
label(gx - 2, py + 4.5, r"$s_t$", fs=FS, ha="right")
label(gx + 0.19 * gw, py + 4.5, "free", fs=FS - 1)
label(gx + 0.45 * gw, py + 4.5, "pre", fs=FS - 1)
label(gx + 0.68 * gw, py + 4.5, "contact", fs=FS - 1, color="white")
label(gx + 0.92 * gw, py + 4.5, "free", fs=FS - 1)
label(gx + 0.45 * gw, py + 15, r"$0.5\,\mathrm{s}$ before onset", fs=FS - 1.3, color="#7a5a00")
by = 252
bx = [0, 0.38, 0.38, 0.84, 0.84, 1]
bv = [0, 0, 1, 1, 0, 0]
ax.plot(gx + np.array(bx) * gw, by + np.array(bv) * 10, color=C["phase"], lw=0.9, zorder=3)
ax.plot([gx, gx + gw], [by, by], color="#999999", lw=0.3, zorder=2)
label(gx - 2, by + 5, r"$b_t$", fs=FS, ha="right", color=C["phase"])
label(gx + gw / 2, 232,
      r"$b_t=\mathbf{1}[s_t\in\{\mathrm{pre},\mathrm{contact}\}]$" + "\n"
      + r"signed history $\mathcal{T}_t=[\hat{\tau}_{\mathrm{ext},t-K+1},\dots,\hat{\tau}_{\mathrm{ext},t}]$",
      fs=FS - 0.6, linespacing=1.5)

# ====================================================================
# (b) Stage 1: action-supervised visual-torque fusion (teacher)
# ====================================================================
panel(182, 212, 530, 204, "(b) Stage 1: Action-supervised visual–torque fusion (teacher)")

img = image(192, 330, 48, 48, "side_blur.png", caption=r"$I_t$ (phase-aware blur)")
# blur curriculum inset
sx, sy, sw, sh = 194, 288, 44, 18
ax.plot([sx, sx, sx + sw], [sy + sh, sy, sy], color="#777777", lw=0.4, zorder=3)
ax.plot([sx, sx + sw / 3, sx + sw], [sy + sh - 3, sy + sh - 3, sy], color=C["sup"], lw=0.8, zorder=3)
label(sx + sw / 2, sy - 4, "training progress", fs=FS - 1.6, color="#555555")
label(sx - 1.5, sy + sh / 2, r"$\sigma$", fs=FS - 0.6, ha="right", color="#555555")

dino = box(256, 344, 46, 20, "DINOv3", kind="frozen", bold=True)
proj = box(312, 344, 46, 20, "Proj. + Emb.", kind="train", fs=FS - 0.3)
path([img["r"], dino["l"]])
path([dino["r"], proj["l"]])

lstm = box(256, 248, 46, 20, r"$\mathrm{LSTM}_T$", kind="train")
label(204, 258, r"$\mathcal{T}_t$", fs=FS + 0.6, color=C["tau"])
path([(210, 258), lstm["l"]], C["tau"])

# fusion module
fx, fy, fw, fh = 382, 246, 78, 128
ax.add_patch(FancyBboxPatch((fx, fy), fw, fh, boxstyle="round,pad=0,rounding_size=3",
                            fc="#FFF6EC", ec=C["train_ec"], lw=0.7, ls=(0, (3, 1.5)), zorder=2))
label(fx + fw / 2, fy + fh - 6, r"Fusion $\mathcal{F}$", fs=FS + 0.2, fontweight="bold")
ca1 = box(fx + 8, fy + 72, fw - 16, 30, "Cross-Attn", kind="train",
          sub=r"Q: $P_t$   K,V: $H^\tau_t$")
ff1 = box(fx + 8, fy + 58, fw - 16, 11, "FFN", kind="train", fs=FS - 0.5)
ca2 = box(fx + 8, fy + 22, fw - 16, 30, "Cross-Attn", kind="train",
          sub=r"Q: tokens   K,V: $H^\tau_t$")
ff2 = box(fx + 8, fy + 8, fw - 16, 11, "FFN", kind="train", fs=FS - 0.5)
path([ca1["b"], ff1["t"]])
path([ff1["b"], ca2["t"]])
path([ca2["b"], ff2["t"]])
# visual tokens P_t
label(364, 361, r"$P_t$", fs=FS + 0.4)
path([proj["r"], (366, 354), (366, 354)], head=False)
path([(358, 354), (372, 354), (372, ca1["y"] + 20), (ca1["x"], ca1["y"] + 20)])
# torque features H
label(320, 263, r"$H^\tau_t$", fs=FS + 0.4, color=C["tau"])
path([lstm["r"], (340, 258), (340, ca1["y"] + 8), (ca1["x"], ca1["y"] + 8)], C["tau"])
path([(340, ca2["y"] + 8), (ca2["x"], ca2["y"] + 8)], C["tau"])

# phase switch: Z = LN(P + b (F - P))
swx, swy = 492, 300
ax.add_patch(Circle((swx, swy), 6.5, fc="white", ec=C["phase"], lw=0.9, zorder=3))
ax.plot([swx - 4, swx + 4], [swy - 2.5, swy + 2.5], color=C["phase"], lw=0.8, zorder=4)
ax.add_patch(Circle((swx - 4, swy - 2.5), 0.9, fc=C["phase"], ec="none", zorder=4))
path([(fx + fw / 2, fy), (fx + fw / 2, fy - 8), (470, fy - 8), (470, swy - 3), (swx - 6.5, swy - 3)])
label(476, fy - 3.5, r"$b_t\!=\!1$", fs=FS - 0.8, color=C["phase"])
path([(372, 354), (372, 386), (470, 386), (470, swy + 3), (swx - 6.5, swy + 3)])
label(421, 390.5, r"bypass ($b_t\!=\!0$)", fs=FS - 0.8, color=C["phase"])
label(swx + 9, swy - 9, r"$b_t$", fs=FS + 0.4, color=C["phase"])
ln = box(508, 291, 22, 18, "LN", kind="train")
path([(swx + 6.5, swy), ln["l"]])
label(519, 318, r"$Z^T_t$", fs=FS + 0.4)

cat = box(538, 286, 52, 28, "Concat", kind="op", sub="views × steps")
path([ln["r"], cat["l"]])
dit = box(604, 286, 44, 28, r"DiT $D_\psi$", kind="train", bold=True)
path([cat["r"], dit["l"]])
label(597, 305, r"$\mathcal{Z}^T_t$", fs=FS + 0.2)
label(626, 352, r"$A_t^{(n)}=\sqrt{\bar\alpha_n}A_t+\sqrt{1-\bar\alpha_n}\,\epsilon,\ n$", fs=FS - 0.4)
path([(626, 345), dit["t"]])
lt = box(662, 286, 44, 28, r"$\mathcal{L}_{\mathrm{teacher}}$", kind="loss",
         sub=r"$\|\hat\epsilon-\epsilon\|_2^2$")
path([dit["r"], lt["l"]], C["sup"])
label(656, 305, r"$\hat\epsilon$", fs=FS)
label(606, 232, "Blur is disabled after training; the teacher is frozen for Stage 2.",
      fs=FS - 0.8, color="#555555", style="italic")

# cross-panel: T_t and b_t from (a)
path([(170, 240), (200, 240), (200, 252)], C["tau"], head=False)
path([(swx, 222), (swx, swy - 6.5)], C["phase"])
path([(170, 222), (swx, 222)], C["phase"], head=False)

# ====================================================================
# (c) Stage 2: token-to-image distillation (student renderer)
# ====================================================================
panel(4, 20, 470, 186, "(c) Stage 2: Token-to-image distillation (student renderer)")

inp = image(12, 112, 44, 44, "side_orig.png", caption=r"clear $I_t$")
label(34, 86, r"$\mathcal{T}_t$", fs=FS + 0.6, color=C["tau"])
# input bus
path([inp["r"], (66, 134)], head=False)
path([(42, 86), (68, 86)], C["tau"], head=False)
path([(66, 134), (66, 166), (80, 166)])
path([(66, 134), (66, 76), (80, 76)])
path([(68, 86), (68, 160), (80, 160)], C["tau"])
path([(68, 86), (68, 70), (80, 70)], C["tau"])

gate = box(80, 150, 92, 30, r"Spatial gate $G_\omega$", kind="train", bold=True,
           sub="Enc · LSTM · Attn · Dec")
label(80, 186, "Step 1: gate pretraining", fs=FS - 0.7, ha="left", color="#555555",
      style="italic")
rgb = box(80, 58, 92, 30, r"RGB residual $U_\phi$", kind="train", bold=True,
          sub="Enc·LSTM·Attn·Dec + gated SFT", fs=FS - 0.2)
label(80, 50, "Step 2: freeze gate, train RGB branch", fs=FS - 0.7, ha="left",
      color="#555555", style="italic")
path([(126, 150), (126, 88)], "#999999", ls=(0, (2, 1.5)))
label(129, 119, "init", fs=FS - 1.2, ha="left", color="#777777")

mimg = image(184, 151, 28, 28, "side_mask.npy", cmap="magma")
label(198, 185, r"$M_t$", fs=FS + 0.2)
path([gate["r"], mimg["l"]])
thr = box(186, 112, 46, 16, r"$M_t\odot\mathbf{1}[M_t\!>\!0.5]$", kind="op", fs=FS - 0.6)
path([mimg["b"], (198, 128)])
label(239, 120, r"$g_t$", fs=FS + 0.4)
# gate to SFT inside U and to render
path([(209, 112), (209, 100), (150, 100), (150, 88)])
label(153, 94, r"$g_t$ (SFT)", fs=FS - 1, ha="left")

# pool + gate loss
lg = box(248, 151, 86, 28, r"$\mathcal{L}_{\mathrm{gate}}$", kind="loss",
         sub=r"SmoothL1(Pool$(M_t),\bar m_t)$ + TV", fs=FS - 0.2)
path([mimg["r"], lg["l"]], C["sup"])

teacher = box(366, 110, 98, 32, "Frozen teacher", kind="frozen", bold=True,
              sub=r"clear $I_t$, $\mathcal{T}_t$, $b_t$ $\rightarrow Z^T_t$, $Z^0_t$")
mbar = image(400, 152, 28, 28, "side_mask.npy", cmap="magma")
label(437, 189, r"$\bar m_t$: min–max of $\|Z^T_{t,j}-Z^0_{t,j}\|_2$", fs=FS - 0.8, ha="right")
path([(414, 142), mbar["b"]], C["sup"], ls=(0, (2.5, 1.5)))
path([mbar["l"], lg["r"]], C["sup"], ls=(0, (2.5, 1.5)))

# rendering
rend = box(186, 58, 64, 30, r"$\mathrm{clip}_{[0,1]}(I_t+$", kind="op", fs=FS - 0.6,
           sub=r"$b_t\,g_t\odot\tanh(r_t))$")
path([rgb["r"], rend["l"]])
label(179, 78, r"$r_t$", fs=FS)
path([(209 + 12, 112), (221, 88)])
path([(218, 36), (218, 58)], C["phase"])
label(222, 36, r"$b_t$", fs=FS, ha="left", color=C["phase"])
rimg = image(260, 51, 44, 44, "side_rendered.png")
label(282, 43, r"$\widetilde{I}_t$", fs=FS + 0.4)
path([rend["r"], rimg["l"]])
evis = box(312, 58, 54, 30, r"$E_{\mathrm{vis}}$", kind="frozen", bold=True,
           sub="DINOv3·Proj·LN")
path([rimg["r"], evis["l"]])
ltok = box(384, 58, 80, 30, r"$\mathcal{L}_{\mathrm{token}}$", kind="loss",
           sub=r"$\frac{1}{Nd}\|\mathcal{Z}^S_t-\mathrm{sg}(\mathcal{Z}^T_t)\|_F^2$")
path([evis["r"], ltok["l"]], C["sup"])
label(374, 94, r"$\mathcal{Z}^S_t$", fs=FS)
path([(423, 110), ltok["t"]], C["sup"], ls=(0, (2.5, 1.5)))
label(426, 100, r"$\mathrm{sg}(\mathcal{Z}^T_t)$", fs=FS - 0.3, ha="left")
# gradient flow note
ax.annotate("", xy=(262, 34), xytext=(360, 34),
            arrowprops=dict(arrowstyle="-|>", color=C["sup"], lw=0.6, ls=(0, (2, 1.2)),
                            mutation_scale=5))
label(311, 29, r"gradients through frozen $E_{\mathrm{vis}}$ to pixels", fs=FS - 1.2,
      color=C["sup"])

# ====================================================================
# (d) Deployment
# ====================================================================
panel(480, 20, 232, 186, "(d) Deployment: a reusable RGB interface")

label(512, 178, r"$q,\dot{q},\tau_m$", fs=FS)
n2 = box(530, 169, 34, 18, "NEXT", kind="frozen", bold=True)
path([(524, 178), n2["l"]], C["tau"])
label(512, 150, r"$\mathcal{O}_{\leq t}$", fs=FS)
ceta = box(530, 141, 34, 18, r"$C_\eta$", kind="train", bold=True)
path([(522, 150), ceta["l"]], C["phase"])
label(547, 136, "phase predictor", fs=FS - 1.3, color="#555555")
dimg = image(490, 92, 34, 34, "wrist_orig.png")
label(507, 87, r"$I_t$", fs=FS)

R = box(590, 95, 40, 92, r"$\mathcal{R}_{\omega,\phi}$", kind="frozen", bold=True, fs=FS + 0.6)
label(610, 125, "CIVI\nrenderer", fs=FS - 0.8, color="#333333", linespacing=1.2)
path([n2["r"], (590, 178)], C["tau"])
label(577, 182, r"$\mathcal{T}_t$", fs=FS - 0.2, color=C["tau"])
path([ceta["r"], (590, 150)], C["phase"])
label(577, 154, r"$\hat{b}_t$", fs=FS - 0.2, color=C["phase"])
path([dimg["r"], (590, 109)])

oimg = image(642, 120, 50, 50, "wrist_rendered.png")
label(667, 177, r"$\widetilde{I}_t$ (contact-informed)", fs=FS - 0.6)
path([(630, 145), oimg["l"]])

pol = box(548, 36, 128, 32, r"Downstream policy $\pi$", kind="train", bold=True,
          sub="own visual encoder; e.g., DP / DiT / VLA")
path([oimg["b"], (667, 82), (614, 82), pol["t"]])
label(515, 52, "state,\ninstruction", fs=FS - 1, linespacing=1.15)
path([(531, 52), pol["l"]])
path([pol["r"], (704, 52)])
label(694, 60, r"$a_{t:t+H}$", fs=FS - 0.4)

# ====================================================================
# legend
# ====================================================================
lx, ly = 8, 8
items = [("train", "trainable"), ("frozen", "frozen " + SNOW), ("loss", "objective"), ("op", "operation")]
for kind, txt in items:
    fc, ec = {"train": (C["train_fc"], C["train_ec"]), "frozen": (C["frozen_fc"], C["frozen_ec"]),
              "loss": (C["loss_fc"], C["loss_ec"]), "op": (C["op_fc"], C["op_ec"])}[kind]
    ax.add_patch(FancyBboxPatch((lx, ly - 3.5), 12, 7, boxstyle="round,pad=0,rounding_size=1.5",
                                fc=fc, ec=ec, lw=0.6))
    ax.text(lx + 15, ly, txt, va="center", fontsize=FS - 0.4,
            fontfamily=["Liberation Sans", "DejaVu Sans"])
    lx += 64
for col, txt in [(C["vis"], "visual path"), (C["tau"], "torque path"), (C["phase"], "phase switch"),
                 (C["sup"], "supervision")]:
    ax.annotate("", xy=(lx + 14, ly), xytext=(lx, ly),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8, mutation_scale=5.5))
    label(lx + 17, ly, txt, fs=FS - 0.4, ha="left")
    lx += 66
for ph, txt in [("free", "free"), ("pre", "pre-contact"), ("contact", "contact")]:
    ax.add_patch(Rectangle((lx, ly - 3), 9, 6, fc=C[ph], ec="none"))
    label(lx + 12, ly, txt, fs=FS - 0.4, ha="left")
    lx += 22 + 4.2 * len(txt)

for ext in ("pdf", "svg", "png"):
    fig.savefig(HERE / f"civi_method.{ext}", dpi=400)
print("saved", HERE / "civi_method.pdf")
