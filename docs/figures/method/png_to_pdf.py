"""Losslessly wrap a raster image in a single-page PDF (Flate, native pixels, no resampling).

usage: python png_to_pdf.py INPUT OUTPUT.pdf [--width-in 7.0] [--crop x0,y0,x1,y1]
The page is sized so the image is --width-in inches wide (7.0 in = two-column \\textwidth).
"""
import argparse

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("src"); p.add_argument("dst")
p.add_argument("--width-in", type=float, default=7.0)
p.add_argument("--crop", default=None)
a = p.parse_args()

im = cv2.imread(a.src, cv2.IMREAD_UNCHANGED)  # reads by content, so a PNG named .pdf works
if im is None:
    raise SystemExit(f"cannot read image: {a.src}")
if im.ndim == 3 and im.shape[2] == 4:
    alpha = im[..., 3:4] / 255.0
    im = (im[..., :3] * alpha + 255 * (1 - alpha)).round().astype("uint8")
if a.crop:
    x0, y0, x1, y1 = map(int, a.crop.split(","))
    im = im[y0:y1, x0:x1]
rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
h, w = rgb.shape[:2]
dpi = w / a.width_in
fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
fig.figimage(rgb, xo=0, yo=0, origin="upper", resize=False)
fig.savefig(a.dst, dpi=dpi, metadata={"Producer": "png_to_pdf.py"})
print(f"{a.dst}: {w}x{h}px, page {w / dpi * 72:.1f}x{h / dpi * 72:.1f}pt ({dpi:.0f} dpi at {a.width_in} in)")
