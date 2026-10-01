"""Guided trace extraction for assets/board-src.jpg (the approach that worked for the website board).

Seeds are placed on the band centres of each bus where it leaves the chip (and along the image
edges for buses that don't touch the chip); each seed is followed outward step by step with
momentum, allowing PCB-style 45-degree turns, until the trace fades, enters a shadow, or runs
into a trace that is already claimed.

    uv run --no-project --python 3.12 --with scikit-image --with pillow --with scipy python tools/trace_guided.py

Writes assets/board.jpg (3x display image), assets/traces.json, shots/trace_overlay.png.
"""
import json, math, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "assets", "board-src.jpg")
UP = 3      # display scale
S = 2       # tracing scale
E = lambda k, d: float(os.environ.get(k, d))

CHIP = (110, 152, 246, 287)   # chip body, source px (x0, y0, x1, y1)
src = Image.open(SRC).convert("RGB")
W0, H0 = src.size

disp = src.resize((W0 * UP, H0 * UP), Image.LANCZOS).filter(ImageFilter.UnsharpMask(radius=2.2, percent=70, threshold=2))
disp.save(os.path.join(ROOT, "assets", "board.jpg"), quality=82, optimize=True, progressive=True)

a = np.asarray(src.resize((W0 * S, H0 * S), Image.BICUBIC)).astype(float)
G = a[:, :, 1]
lum = a.mean(axis=2)
H, W = G.shape
bg = ndi.gaussian_filter(G, 30)
sd = np.sqrt(ndi.gaussian_filter((G - bg) ** 2, 30)) + 4
norm = (G - bg) / sd
ridge = ndi.gaussian_filter(norm, 1.4) - ndi.uniform_filter(ndi.gaussian_filter(norm, 1.4), int(E("TB_WIN", 15)))
valid = (ndi.uniform_filter(lum, 15) > E("TB_DARK", 55)) & (a[:, :, 1] > a[:, :, 0] + 4)
cx0, cy0, cx1, cy1 = [v * S for v in CHIP]
chip_mask = np.zeros_like(valid)
chip_mask[cy0 - 4:cy1 + 4, cx0 - 4:cx1 + 4] = True
valid &= ~chip_mask
EXCLUDE = [(92, 0, 150, 72), (440, 215, 601, 345), (230, 330, 470, 400), (520, 300, 601, 400), (500, 130, 545, 165)]
for x0, y0, x1, y1 in EXCLUDE:
    valid[y0 * S:y1 * S, x0 * S:x1 * S] = False
hard_stop = ~valid & (ndi.uniform_filter(lum, 15) <= 0)   # placeholder, replaced below
hard_stop = np.zeros_like(valid)
for x0, y0, x1, y1 in EXCLUDE:
    hard_stop[y0 * S:y1 * S, x0 * S:x1 * S] = True

DIRS = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]   # 0=E, counter-clockwise (y down)


def rs(x, y):
    xi, yi = int(round(x)), int(round(y))
    if 0 <= xi < W and 0 <= yi < H:
        return ridge[yi, xi]
    return -9


def ahead(x, y, d, n=5):
    dx, dy = DIRS[d]
    return sum(rs(x + dx * k, y + dy * k) for k in range(1, n + 1)) / n


occ = np.zeros((H, W), bool)
THR = E("TB_THR", 0.12)
TURN = E("TB_TURN", 0.25)


MISS = int(E("TB_MISS", 40))       # steps a trace may stay faint (shadow) while running straight


def follow(x, y, d, maxlen=4000):
    pts = [(x, y)]
    miss = 0
    since_turn = 99
    for _ in range(maxlen):
        straight = ahead(x, y, d, 7)
        nd = d
        if since_turn >= 10:
            best = (straight, d)
            for dd in (1, -1):
                cand = (d + dd) % 8
                sc = ahead(x, y, cand, 7) - TURN
                if sc > best[0]:
                    best = (sc, cand)
            # only bend when straight is genuinely losing the trace
            if best[1] != d and (straight < THR or best[0] > straight + TURN):
                nd = best[1]
        since_turn = 0 if nd != d else since_turn + 1
        d = nd
        dx, dy = DIRS[d]
        x, y = x + dx, y + dy
        px, py = -dy, dx
        nrm = math.hypot(px, py)
        px, py = px / nrm, py / nrm
        # re-centre on the band: only when the ridge is clear, at most 1px per 3 steps
        if rs(x, y) > THR and since_turn % 3 == 0:
            k = max((rs(x + px * k, y + py * k), k) for k in (-1, 0, 1))[1]
            x, y = x + px * k, y + py * k
        xi, yi = int(round(x)), int(round(y))
        if not (2 <= xi < W - 2 and 2 <= yi < H - 2):
            pts.append((x, y)); break
        if chip_mask[yi, xi] or hard_stop[yi, xi]:
            break
        if occ[yi, xi] and len(pts) > 8:
            break
        if ridge[yi, xi] < THR or not valid[yi, xi]:
            miss += 1
        else:
            miss = 0
        if miss > MISS:
            pts = pts[:-MISS]; break
        pts.append((x, y))
    return pts


def band_peaks(profile, min_sep=int(E("TB_SEP", 11)), thr=THR * 1.5):
    p = ndi.gaussian_filter1d(profile, 1.2)
    idx = [i for i in range(3, len(p) - 3) if p[i] > thr and p[i] == p[i - 3:i + 4].max()]
    out = []
    for i in idx:
        if not out or i - out[-1] >= min_sep:
            out.append(i)
    return out


def seeds_line(fixed, rng, axis, offset_dirs):
    """axis='x': vertical seed line at x=fixed, scan y in rng; returns (x, y)"""
    if axis == "x":
        prof = np.array([ridge[y, fixed - 2:fixed + 3].mean() for y in range(*rng)])
        return [(fixed, rng[0] + i) for i in band_peaks(prof)]
    prof = np.array([ridge[fixed - 2:fixed + 3, x].mean() for x in range(*rng)])
    return [(rng[0] + i, fixed) for i in band_peaks(prof)]


OFF = int(E("TB_OFF", 14))   # seed this far outside the chip edge (trace px)
seed_sets = [
    ("right", seeds_line(cx1 + OFF, (cy0, cy1), "x", None), 0),
    ("left", seeds_line(cx0 - OFF, (cy0, cy1), "x", None), 4),
    ("top", seeds_line(cy0 - OFF, (cx0, cx1), "y", None), 2),
    ("bottom", seeds_line(cy1 + OFF, (cx0, cx1), "y", None), 6),
    ("edge-right", seeds_line(W - 6, (6, H - 6), "x", None), 4),
    ("edge-top", seeds_line(6, (6, W - 6), "y", None), 6),
    ("edge-left", seeds_line(6, (6, H - 6), "x", None), 0),
    ("edge-bottom", seeds_line(H - 6, (6, W - 6), "y", None), 2),
]

paths, stats = [], {}
MINLEN = E("TB_MINLEN", 70)
for name, seeds, d in seed_sets:
    kept = 0
    for (x, y) in seeds:
        if occ[y, x]:
            continue
        fwd = follow(float(x), float(y), d)
        if name in ("right", "left", "top", "bottom"):
            back = []          # chip seeds: the chip side is the end; don't walk into the chip
        else:
            back = follow(float(x), float(y), (d + 4) % 8)  # small extension back to the edge
        pts = back[::-1] + fwd[1:]
        arr = np.array(pts)
        L = np.sum(np.hypot(*np.diff(arr, axis=0).T)) if len(arr) > 1 else 0
        if L < MINLEN:
            continue
        # claim it
        for (px, py) in pts:
            xi, yi = int(round(px)), int(round(py))
            occ[max(0, yi - 4):yi + 5, max(0, xi - 4):xi + 5] = True
        paths.append(pts)
        kept += 1
    stats[name] = (len(seeds), kept)


def octi(pts):
    """snap each segment to the nearest multiple of 45 degrees (PCB routing), keep segment lengths"""
    out = [pts[0]]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        L = math.hypot(x1 - x0, y1 - y0)
        if L < 1e-6:
            continue
        ang = round(math.atan2(y1 - y0, x1 - x0) / (math.pi / 4)) * (math.pi / 4)
        px, py = out[-1]
        out.append((px + L * math.cos(ang), py + L * math.sin(ang)))
    # merge consecutive collinear segments
    merged = [out[0]]
    for i in range(1, len(out) - 1):
        a1 = math.atan2(out[i][1] - merged[-1][1], out[i][0] - merged[-1][0])
        a2 = math.atan2(out[i + 1][1] - out[i][1], out[i + 1][0] - out[i][0])
        if abs(math.remainder(a1 - a2, 2 * math.pi)) > 1e-3:
            merged.append(out[i])
    merged.append(out[-1])
    return merged


def rdp(pts, eps):
    if len(pts) < 3:
        return pts
    (x1, y1), (x2, y2) = pts[0], pts[-1]
    L = math.hypot(x2 - x1, y2 - y1) or 1
    d = [abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / L for x, y in pts[1:-1]]
    i = int(np.argmax(d)) + 1
    if d[i - 1] > eps:
        return rdp(pts[: i + 1], eps)[:-1] + rdp(pts[i:], eps)
    return [pts[0], pts[-1]]


ccx, ccy = (cx0 + cx1) / 2, (cy0 + cy1) / 2
sc = UP / S
out = []
for p in paths:
    arr = ndi.uniform_filter1d(np.array(p, float), 5, axis=0, mode="nearest")
    simp = octi(rdp([tuple(q) for q in arr], float(E("TB_EPS", 3.0))))
    # drop tiny jogs (< 6 px) that snapping can leave behind
    simp = [simp[0]] + [q for k, q in enumerate(simp[1:-1], 1)
                        if math.hypot(q[0] - simp[k - 1][0], q[1] - simp[k - 1][1]) > 6] + [simp[-1]]
    if math.hypot(simp[0][0] - ccx, simp[0][1] - ccy) < math.hypot(simp[-1][0] - ccx, simp[-1][1] - ccy):
        simp = simp[::-1]          # end at the chip side
    out.append([[round(x * sc, 1), round(y * sc, 1)] for x, y in simp])

# ---- cleanup: drop short stubs, overlapping duplicates and self-crossing tangles ----
def seglen(p):
    return sum(math.hypot(p[i + 1][0] - p[i][0], p[i + 1][1] - p[i][1]) for i in range(len(p) - 1))


def sample(p, step=6):
    pts = []
    for (x0, y0), (x1, y1) in zip(p, p[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
        pts += [(x0 + (x1 - x0) * t / n, y0 + (y1 - y0) * t / n) for t in range(n)]
    pts.append(tuple(p[-1]))
    return np.array(pts)


def back_turns(p):
    n = 0
    for i in range(1, len(p) - 1):
        a1 = math.atan2(p[i][1] - p[i - 1][1], p[i][0] - p[i - 1][0])
        a2 = math.atan2(p[i + 1][1] - p[i][1], p[i + 1][0] - p[i][0])
        if abs(math.remainder(a2 - a1, 2 * math.pi)) > math.radians(100):
            n += 1
    return n


MIN_OUT = E("TB_MINOUT", 120)
out = [p for p in out if seglen(p) >= MIN_OUT and back_turns(p) == 0]
out.sort(key=lambda p: -seglen(p))
from scipy.spatial import cKDTree
kept, kept_pts = [], []
CLEAR = E("TB_CLEAR", 9)
for p in out:
    sp = sample(p)
    if kept_pts:
        tree = cKDTree(np.vstack(kept_pts))
        dist, _ = tree.query(sp)
        close = dist < CLEAR
        if close.mean() > 0.25:
            continue
        # trim: keep the longest run of points that are NOT on another path
        if close.any():
            runs, cur = [], []
            for q, c in zip(sp, close):
                if c:
                    if len(cur) > 1: runs.append(cur)
                    cur = []
                else:
                    cur.append(tuple(q))
            if len(cur) > 1: runs.append(cur)
            best = max(runs, key=len) if runs else []
            if len(best) * 6 < MIN_OUT:
                continue
            p = [list(q) for q in rdp(best, 1.0)]
            if math.hypot(p[0][0] - ccx * sc, p[0][1] - ccy * sc) < math.hypot(p[-1][0] - ccx * sc, p[-1][1] - ccy * sc):
                p = p[::-1]
            sp = sample(p)
    kept.append([[round(x, 1), round(y, 1)] for x, y in p])
    kept_pts.append(sp)
out = kept

data = {"size": [W0 * UP, H0 * UP], "chip": [round(ccx * sc), round(ccy * sc)], "paths": out}
with open(os.path.join(ROOT, "assets", "traces.json"), "w") as f:
    json.dump(data, f, separators=(",", ":"))

ov = disp.copy()
dr = ImageDraw.Draw(ov)
for p in out:
    dr.line([tuple(q) for q in p], fill=(255, 235, 120), width=3)
    dr.ellipse([p[-1][0] - 4, p[-1][1] - 4, p[-1][0] + 4, p[-1][1] + 4], fill=(255, 70, 70))
dr.rectangle([CHIP[0] * UP, CHIP[1] * UP, CHIP[2] * UP, CHIP[3] * UP], outline=(255, 0, 255), width=2)
os.makedirs(os.path.join(ROOT, "shots"), exist_ok=True)
ov.save(os.path.join(ROOT, "shots", "trace_overlay.png"))
lens = [sum(math.hypot(p[i + 1][0] - p[i][0], p[i + 1][1] - p[i][1]) for i in range(len(p) - 1)) for p in out]
print("seeds(found, kept):", stats)
print(f"paths={len(out)} total={sum(lens):.0f}px mean={np.mean(lens) if lens else 0:.0f}px "
      f"json={os.path.getsize(os.path.join(ROOT, 'assets', 'traces.json'))}B")
