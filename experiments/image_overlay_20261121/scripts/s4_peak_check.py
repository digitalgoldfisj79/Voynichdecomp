"""Stage 4: G2b peak-location instrument (see RESULTS pre-registration).
Usage: python3 -I s4_peak_check.py <dl_dir> <out_dir> [long_px=1000]
Writes out/s4_peak.pkl, out/g2b_peak_by_folio.csv
"""
import csv, json, os, pickle, sys
import numpy as np
from PIL import Image, ImageFilter

DL, OUT = sys.argv[1], sys.argv[2]
LONG = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
R = 16
S1 = pickle.load(open(os.path.join(OUT, "s1_boxes.pkl"), "rb"))
S2 = pickle.load(open(os.path.join(OUT, "s2_mapped.pkl"), "rb"))["mapped"]
ARMS = pickle.load(open(os.path.join(OUT, "s3_arms.pkl"), "rb"))
reg = {json.loads(l)["selected"]["folio"]: json.loads(l)["selected"] for l in open(os.path.join(DL, "reg_full_225.jsonl")) if l.strip()}
IMG = os.path.join(DL, "img%d" % LONG)

def integral(sid):
    im = Image.open(os.path.join(IMG, sid.rsplit("/", 1)[1] + ".jpg")).convert("L")
    g = np.asarray(im, np.float32)
    bg = np.asarray(im.filter(ImageFilter.GaussianBlur(radius=max(8, LONG // 40))), np.float32)
    d = np.clip(1.0 - g / np.maximum(bg, 1.0), 0, 1)
    ii = np.zeros((d.shape[0] + 1, d.shape[1] + 1), np.float64); ii[1:, 1:] = d.cumsum(0).cumsum(1)
    return ii, d.shape

def surface(ii, shape, b):
    """mean in-box darkness for every integer shift in [-R, R]^2 -> (2R+1, 2R+1) array [dy, dx]."""
    Hh, Ww = shape
    bx0, by0, bx1, by1 = (np.rint(b[:, i]).astype(int) for i in range(4))
    out = np.zeros((2 * R + 1, 2 * R + 1))
    for iy, dy in enumerate(range(-R, R + 1)):
        y0 = np.clip(by0 + dy, 0, Hh); y1 = np.clip(by1 + dy, 0, Hh)
        for ix, dx in enumerate(range(-R, R + 1)):
            x0 = np.clip(bx0 + dx, 0, Ww); x1 = np.clip(bx1 + dx, 0, Ww)
            area = ((x1 - x0) * (y1 - y0)).sum()
            out[iy, ix] = (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]).sum() / max(area, 1)
    return out

def argmax(s):
    iy, ix = np.unravel_index(np.argmax(s), s.shape)
    return ix - R, iy - R        # (dx, dy)

PLANTS = [(0, 8), (0, -8), (8, 0), (-8, 0), (6, -6)]
res = {}
for i, fo in enumerate(S1["folios"]):
    sid = reg[fo]["service_id"]; fw, fh = S2[fo]["full_wh"]
    ii, shape = integral(sid); k = max(shape) / max(fw, fh)
    row = dict(k=k)
    B = ARMS[fo]["B"] * k
    row["box_h_px"] = float(np.median(B[:, 3] - B[:, 1]))
    sB = surface(ii, shape, B); row["B_dx"], row["B_dy"] = argmax(sB)
    # sharpness: peak minus median of the surface, in SDs of the surface
    row["B_peak_sd"] = float((sB.max() - np.median(sB)) / (sB.std() + 1e-12))
    rec = 0
    for px, py in PLANTS:
        Bp = B + np.array([px, py, px, py], float)
        dx, dy = argmax(surface(ii, shape, Bp))
        if abs((dx - row["B_dx"]) + px) <= 2 and abs((dy - row["B_dy"]) + py) <= 2:
            rec += 1
    row["plants_recovered"] = rec
    for name in ("A", "N"):
        row[name + "_dx"], row[name + "_dy"] = argmax(surface(ii, shape, ARMS[fo][name] * k))
    res[fo] = row
    if i % 45 == 0:
        print(i, fo, row, flush=True)
pickle.dump(res, open(os.path.join(OUT, "s4_peak_%d.pkl" % LONG), "wb"))

rows = []
for fo in S1["folios"]:
    r = res[fo]; readable = r["plants_recovered"] >= 4
    off = float(np.hypot(r["B_dx"], r["B_dy"])); frac = off / r["box_h_px"]
    edge = max(abs(r["B_dx"]), abs(r["B_dy"])) == R
    v = "Inconclusive at this power (unreadable)" if not readable else ("CONFIRM" if frac <= 0.25 else ("FALSIFY" if frac > 0.5 else "Inconclusive"))
    rows.append(dict(folio=fo, canvas=reg[fo]["canvas_label"], n_boxes=len(ARMS[fo]["B"]), box_h_px=round(r["box_h_px"], 1),
                     px_per_legacy=round(r["k"] * np.sqrt(abs(np.linalg.det(np.array(reg[fo]["H_deriv"])[:2, :2]))) / reg[fo]["deriv_scale"], 3),
                     B_dx=r["B_dx"], B_dy=r["B_dy"], B_offset_px=round(off, 2), B_offset_frac_box_h=round(frac, 3),
                     B_offset_fullres_px=round(off / r["k"], 1), at_scan_edge=int(edge),
                     plants_recovered=r["plants_recovered"], A_dx=r["A_dx"], A_dy=r["A_dy"], N_dx=r["N_dx"], N_dy=r["N_dy"],
                     N_offset_frac_box_h=round(float(np.hypot(r["N_dx"], r["N_dy"])) / r["box_h_px"], 3), verdict_B=v))
with open(os.path.join(OUT, "g2b_peak_by_folio_%d.csv" % LONG), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

import collections
rd = [r for r in rows if r["plants_recovered"] >= 4]
print("readable folios: %d of %d (%.1f%%)" % (len(rd), len(rows), 100 * len(rd) / len(rows)))
print("plants recovered distribution:", sorted(collections.Counter(r["plants_recovered"] for r in rows).items()))
print("verdicts:", collections.Counter(r["verdict_B"] for r in rows))
for nm, key in (("B", "B_offset_frac_box_h"), ("N", "N_offset_frac_box_h")):
    v = np.array([r[key] for r in rd]); print(nm, "offset / box height (readable): median %.3f  p90 %.3f  max %.3f" % (np.median(v), np.quantile(v, .9), v.max()))
dx = np.array([r["B_dx"] / r["px_per_legacy"] for r in rd]); dy = np.array([r["B_dy"] / r["px_per_legacy"] for r in rd])
print("B argmax in legacy px: dx median %.2f (share>0 %.2f, share<0 %.2f) | dy median %.2f (share>0 %.2f, share<0 %.2f)" % (
    np.median(dx), (dx > 0).mean(), (dx < 0).mean(), np.median(dy), (dy > 0).mean(), (dy < 0).mean()))
print("B full-res offset px (readable): median %.1f p90 %.1f" % (np.median([r["B_offset_fullres_px"] for r in rd]), np.quantile([r["B_offset_fullres_px"] for r in rd], .9)))
print("FALSIFY/unreadable:", [(r["folio"], r["verdict_B"][:12], r["B_dx"], r["B_dy"], r["plants_recovered"]) for r in rows if not r["verdict_B"].startswith("CONFIRM")][:60])
