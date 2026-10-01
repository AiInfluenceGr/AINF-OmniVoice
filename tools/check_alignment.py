"""Numeric alignment check: is the ridge (trace brightness) higher ON the paths than 6px beside them?"""
import json, math, os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "assets", "traces.json")))
img = Image.open(os.path.join(ROOT, "assets", "board.jpg")).convert("RGB")
a = np.asarray(img).astype(float)
G = a[:, :, 1]
bg = ndi.gaussian_filter(G, 45)
r = ndi.gaussian_filter(G - bg, 2.0)
H, W = G.shape


def val(x, y):
    xi, yi = int(round(x)), int(round(y))
    return r[yi, xi] if 0 <= xi < W and 0 <= yi < H else np.nan


on, off, peak_within = [], [], []
for p in d["paths"]:
    for (x0, y0), (x1, y1) in zip(p, p[1:]):
        L = math.hypot(x1 - x0, y1 - y0)
        if L < 1:
            continue
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        nx, ny = -uy, ux
        for t in np.arange(0, L, 6):
            x, y = x0 + ux * t, y0 + uy * t
            prof = [val(x + nx * k, y + ny * k) for k in range(-12, 13)]
            if any(np.isnan(prof)):
                continue
            on.append(prof[12]); off.append((prof[0] + prof[-1]) / 2)
            # where is the local max across the trace? (offset from our line)
            win = prof[6:19]
            peak_within.append(abs(int(np.argmax(win)) - 6))
on, off, pk = np.array(on), np.array(off), np.array(peak_within)
print(f"samples={len(on)}  mean ridge on-path={on.mean():.2f}  6-12px aside={off.mean():.2f}  "
      f"on>aside: {(on > off).mean() * 100:.0f}%")
print(f"local max within 3px of line: {(pk <= 3).mean() * 100:.0f}%   within 6px: {(pk <= 6).mean() * 100:.0f}%")
