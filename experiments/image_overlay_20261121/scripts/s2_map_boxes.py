"""Stage 2: G1 (XML vs runtime boxes) + map runtime boxes through the registration homography
to full Yale canvas pixels, then normalise to 0..1 of the canvas.

Usage: python3 -I s2_map_boxes.py <dl_dir> <out_dir>
Writes out/s2_mapped.pkl, out/g1_xml_vs_js.csv
"""
import csv, json, os, pickle, sys, time, urllib.request
import numpy as np

DL, OUT = sys.argv[1], sys.argv[2]
S1 = pickle.load(open(os.path.join(OUT, "s1_boxes.pkl"), "rb"))
reg = {}
for line in open(os.path.join(DL, "reg_full_225.jsonl")):
    if line.strip():
        r = json.loads(line)["selected"]
        reg[r["folio"]] = r

# ---- G1: affine fit XML corners -> JS corners
def corners(b):
    _, _, x, y, w, h = b
    return [(x, y), (x + w, y + h)]

g1_rows = []
for fo in S1["folios"]:
    X, J = S1["xml_boxes"][fo], S1["js_boxes"][fo]
    same_n = len(X) == len(J)
    same_words = same_n and all(a[1] == b[1] for a, b in zip(X, J))
    res_max = res_med = float("nan"); sx = sy = float("nan")
    if same_n and len(X) >= 3:
        A = np.array([c for b in X for c in corners(b)], float)
        B = np.array([c for b in J for c in corners(b)], float)
        M = np.c_[A, np.ones(len(A))]
        coef, *_ = np.linalg.lstsq(M, B, rcond=None)
        err = np.abs(M @ coef - B).max(axis=1)
        res_max, res_med = float(err.max()), float(np.median(err))
        sx, sy = float(coef[0, 0]), float(coef[1, 1])
    g1_rows.append(dict(folio=fo, n_xml=len(X), n_js=len(J), same_words=int(bool(same_words)),
                        affine_max_resid_px=round(res_max, 3), affine_median_resid_px=round(res_med, 3),
                        scale_x=round(sx, 4), scale_y=round(sy, 4),
                        xml_w=S1["xml_frame"][fo][0], xml_h=S1["xml_frame"][fo][1]))
with open(os.path.join(OUT, "g1_xml_vs_js.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(g1_rows[0])); w.writeheader(); w.writerows(g1_rows)

# ---- canvas sizes (info.json per service), cached
info_p = os.path.join(DL, "yale_info.json")
info = json.load(open(info_p)) if os.path.exists(info_p) else {}
for sid in sorted({r["service_id"] for r in reg.values()}):
    if sid in info:
        continue
    req = urllib.request.Request(sid + "/info.json", headers={"User-Agent": "voynichstats-overlay/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read())
    info[sid] = [d["width"], d["height"]]
    json.dump(info, open(info_p, "w"))
    time.sleep(0.1)

# ---- map
def persp(H, pts):
    P = np.c_[pts, np.ones(len(pts))] @ H.T
    return P[:, :2] / P[:, 2:3]

mapped = {}
for fo in S1["folios"]:
    r = reg[fo]
    H = np.diag([1 / r["deriv_scale"], 1 / r["deriv_scale"], 1.0]) @ np.array(r["H_deriv"], float)
    fw, fh = info[r["service_id"]]
    boxes = []
    for j, word, x, y, w, h in S1["js_boxes"][fo]:
        q = persp(H, np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float))
        x0, y0, x1, y1 = q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()
        boxes.append((j, word, float(x0), float(y0), float(x1 - x0), float(y1 - y0)))
    mapped[fo] = dict(service_id=r["service_id"], canvas_label=r["canvas_label"], full_wh=(fw, fh),
                      median_reproj_full_px=r["median_reproj_px"] / r["deriv_scale"],
                      p95_reproj_full_px=r["p95_reproj_px"] / r["deriv_scale"],
                      inliers=r["inliers"], boxes=boxes, legacy_boxes=S1["js_boxes"][fo])
pickle.dump(dict(mapped=mapped, info=info), open(os.path.join(OUT, "s2_mapped.pkl"), "wb"))

bad_words = [r["folio"] for r in g1_rows if not r["same_words"]]
bad_res = [(r["folio"], r["affine_max_resid_px"]) for r in g1_rows if not (r["affine_max_resid_px"] <= 1.5)]
print("G1 folios:", len(g1_rows), "| word-sequence mismatch:", len(bad_words), bad_words[:20])
print("G1 affine max residual > 1.5 px:", len(bad_res), bad_res[:20])
res = [r["affine_max_resid_px"] for r in g1_rows if r["affine_max_resid_px"] == r["affine_max_resid_px"]]
print("G1 residual: median of per-folio max = %.3f, worst = %.3f" % (float(np.median(res)), max(res)))
out_of_canvas = 0; total = 0
for fo, m in mapped.items():
    fw, fh = m["full_wh"]
    for b in m["boxes"]:
        total += 1
        if b[2] < 0 or b[3] < 0 or b[2] + b[4] > fw or b[3] + b[5] > fh:
            out_of_canvas += 1
print("mapped boxes:", total, "| partly outside canvas:", out_of_canvas)
print("services:", len(info))
