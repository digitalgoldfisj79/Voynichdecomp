"""Stage 3: G2 ink check. Arms: A runtime boxes via homography, B XML boxes via centre-fit + homography,
N naive rescale of the 636x900 frame. Null = same boxes at random offsets. Power = planted vertical shift.

Usage: python3 -I s3_ink_check.py <dl_dir> <out_dir> [long_px=1000]
Writes out/s3_ink.pkl, out/g2_ink_by_folio.csv, out/s3_arms.pkl (arm B boxes in full canvas px)
"""
import csv, io, json, os, pickle, sys, time, urllib.request
import numpy as np
from PIL import Image, ImageFilter

DL, OUT = sys.argv[1], sys.argv[2]
LONG = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
SEED = 20261121
S1 = pickle.load(open(os.path.join(OUT, "s1_boxes.pkl"), "rb"))
S2 = pickle.load(open(os.path.join(OUT, "s2_mapped.pkl"), "rb"))["mapped"]
reg = {json.loads(l)["selected"]["folio"]: json.loads(l)["selected"] for l in open(os.path.join(DL, "reg_full_225.jsonl")) if l.strip()}
IMG = os.path.join(DL, "img%d" % LONG); os.makedirs(IMG, exist_ok=True)

def persp(H, pts):
    P = np.c_[pts, np.ones(len(pts))] @ H.T
    return P[:, :2] / P[:, 2:3]

def centre_fit(X, J):
    """per-axis scale+offset XML->runtime frame from box centres, trimmed LSQ, unclamped boxes."""
    out = []
    for pi, si, lim in ((2, 4, 636), (3, 5, 900)):
        a, b = [], []
        for x, j in zip(X, J):
            if j[4] <= 0 or j[5] <= 0 or j[pi] <= 0 or j[pi] + j[si] >= lim:
                continue
            a.append(x[pi] + x[si] / 2); b.append(j[pi] + j[si] / 2)
        a, b = np.array(a, float), np.array(b, float)
        keep = np.ones(len(a), bool)
        for _ in range(5):
            s, t = np.polyfit(a[keep], b[keep], 1)
            r = b - (s * a + t); keep = np.abs(r) <= 3.0
        out.append((float(s), float(t), int(keep.sum()), float(r[keep].std())))
    return out

def arm_boxes(fo):
    r = reg[fo]; fw, fh = S2[fo]["full_wh"]
    H = np.diag([1 / r["deriv_scale"]] * 2 + [1.0]) @ np.array(r["H_deriv"], float)
    X, J = S1["xml_boxes"][fo], S1["js_boxes"][fo]
    def through(rects):
        o = []
        for x, y, w, h in rects:
            q = persp(H, np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float))
            o.append([q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()])
        return np.array(o, float).reshape(-1, 4)
    A = through([(j[2], j[3], j[4], j[5]) for j in J if j[4] > 0 and j[5] > 0])
    (sx, tx, nx, rx), (sy, ty, ny, ry) = centre_fit(X, J)
    B = through([(x[2] * sx + tx, x[3] * sy + ty, x[4] * sx, x[5] * sy) for x in X])
    N = np.array([[j[2] * fw / 636, j[3] * fh / 900, (j[2] + j[4]) * fw / 636, (j[3] + j[5]) * fh / 900]
                  for j in J if j[4] > 0 and j[5] > 0], float).reshape(-1, 4)
    return dict(A=A, B=B, N=N, fit=dict(sx=sx, tx=tx, sy=sy, ty=ty, n=min(nx, ny), resid_sd=max(rx, ry)))

def darkness(sid):
    p = os.path.join(IMG, sid.rsplit("/", 1)[1] + ".jpg")
    if not os.path.exists(p):
        req = urllib.request.Request(f"{sid}/full/!{LONG},{LONG}/0/default.jpg", headers={"User-Agent": "voynichstats-overlay/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            open(p, "wb").write(r.read())
        time.sleep(0.1)
    im = Image.open(p).convert("L")
    g = np.asarray(im, np.float32)
    bg = np.asarray(im.filter(ImageFilter.GaussianBlur(radius=max(8, LONG // 40))), np.float32)
    d = np.clip(1.0 - g / np.maximum(bg, 1.0), 0, 1)        # 0 = parchment, >0 = darker than surroundings
    ii = np.zeros((d.shape[0] + 1, d.shape[1] + 1), np.float64)
    ii[1:, 1:] = d.cumsum(0).cumsum(1)
    return ii, d.shape

def score(ii, shape, boxes, dx=0.0, dy=0.0):
    """mean darkness over the union-by-sum of boxes (px coords of this derivative)."""
    Hh, Ww = shape
    x0 = np.clip(np.rint(boxes[:, 0] + dx), 0, Ww).astype(int); x1 = np.clip(np.rint(boxes[:, 2] + dx), 0, Ww).astype(int)
    y0 = np.clip(np.rint(boxes[:, 1] + dy), 0, Hh).astype(int); y1 = np.clip(np.rint(boxes[:, 3] + dy), 0, Hh).astype(int)
    area = ((x1 - x0) * (y1 - y0)).sum()
    if area <= 0:
        return float("nan")
    s = (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]).sum()
    return float(s / area)

rng = np.random.default_rng(SEED)
ck = os.path.join(OUT, "s3_ink_%d.pkl" % LONG)
res = pickle.load(open(ck, "rb")) if os.path.exists(ck) else {}
arms_full = {}
for i, fo in enumerate(S1["folios"]):
    arms = arm_boxes(fo); arms_full[fo] = arms
    if fo in res:
        continue
    sid = reg[fo]["service_id"]; fw, fh = S2[fo]["full_wh"]
    try:
        ii, shape = darkness(sid)
    except Exception as e:
        res[fo] = dict(error=str(e)); continue
    k = max(shape) / max(fw, fh)
    row = dict(n_A=len(arms["A"]), n_B=len(arms["B"]))
    bh = float(np.median((arms["B"][:, 3] - arms["B"][:, 1]))) * k      # median box height, derivative px
    row["box_h_px"] = bh
    for name in ("A", "B", "N"):
        b = arms[name] * k
        obs = score(ii, shape, b)
        ang = rng.uniform(0, 2 * np.pi, 200); rad = rng.uniform(1.5, 6.0, 200) * bh
        null = np.array([score(ii, shape, b, float(r_ * np.cos(a_)), float(r_ * np.sin(a_))) for a_, r_ in zip(ang, rad)])
        null = null[np.isfinite(null)]
        row[name] = obs; row[name + "_null_mean"] = float(null.mean()); row[name + "_null_sd"] = float(null.std(ddof=1))
        row[name + "_z"] = (obs - row[name + "_null_mean"]) / row[name + "_null_sd"]
        if name == "B":
            for frac in (0.5, 1.0):
                sh = np.nanmean([score(ii, shape, b, 0, frac * bh), score(ii, shape, b, 0, -frac * bh)])
                row["B_drop_%g" % frac] = (obs - sh) / row["B_null_sd"]
    res[fo] = row
    if i % 25 == 0:
        pickle.dump(res, open(ck, "wb")); print(i, fo, {k_: round(v, 3) for k_, v in row.items() if k_.endswith("_z")}, flush=True)
pickle.dump(res, open(ck, "wb"))
pickle.dump(arms_full, open(os.path.join(OUT, "s3_arms.pkl"), "wb"))

cols = ["folio", "n_A", "n_B", "box_h_px", "A", "A_null_mean", "A_null_sd", "A_z", "B", "B_null_mean", "B_null_sd", "B_z",
        "N", "N_null_mean", "N_null_sd", "N_z", "B_drop_0.5", "B_drop_1", "verdict_B"]
rows = []
for fo in S1["folios"]:
    r = res[fo]
    if "error" in r:
        rows.append(dict(folio=fo, verdict_B="image error: " + r["error"])); continue
    z, drop = r["B_z"], r["B_drop_1"]
    v = "Inconclusive at this power (1-height shift < 2 null SD)" if drop < 2 else ("CONFIRM" if z >= 3 else ("FALSIFY" if z < 2 else "Inconclusive at this power"))
    rows.append({**{c: (round(r[c], 4) if isinstance(r.get(c), float) else r.get(c)) for c in cols if c in r}, "folio": fo, "verdict_B": v})
with open(os.path.join(OUT, "g2_ink_by_folio_%d.csv" % LONG), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
ok = [r for r in rows if "B_z" in r]
import collections
print("verdicts:", collections.Counter(r["verdict_B"].split(" (")[0] for r in rows))
for name in ("A", "B", "N"):
    z = np.array([r[name + "_z"] for r in ok]); print(name, "z median %.2f  min %.2f  share>=3 %.3f" % (np.median(z), z.min(), (z >= 3).mean()))
print("B beats N (z) in", sum(r["B_z"] > r["N_z"] for r in ok), "of", len(ok), "| B mean darkness > A in", sum(r["B"] > r["A"] for r in ok))
print("planted 1.0-height shift drop (null SD): median %.2f min %.2f; 0.5-height: median %.2f" % (
    np.median([r["B_drop_1"] for r in ok]), min(r["B_drop_1"] for r in ok), np.median([r["B_drop_0.5"] for r in ok])))
