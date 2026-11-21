"""v4 Phase 1 proxy (DEV folios only): 2-D IoU vs Timm boxes for target glyphs that v3 flags, v3 vs v4.
Usage: python3 -I v2_dev.py <dl_dir> <out_dir> <tag> [param=value ...]  -> appends a row to out/v4/dev_variants.csv"""
import csv, json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkglyph as IG
from k3_lib import readings_for
DL, OUT, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
P = dict(IG.P0)
for kv in sys.argv[4:]:
    k, v = kv.split("="); P[k] = float(v) if "." in v else int(v)
SP = json.load(open(os.path.join(OUT, "split_v4.json"))); DEV = set(SP["dev"])
T = pickle.load(open(os.path.join(OUT, "v4", "templates.pkl"), "rb"))
J = json.load(open(os.path.join(OUT, "glyph_width_table.json")))
cases = [c for c in pickle.load(open(os.path.join(OUT, "g2_cases.pkl"), "rb")) if c["folio"] in DEV]
BOX = json.load(open(os.path.join(OUT, "voynichese-boxes.json")))["folios"]
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0
rows = []
for fo in sorted({c["folio"] for c in cases}):
    _, im = readings_for(DL, OUT, fo); rec = BOX[fo]; kx = im.size[0] / rec["cw"]; ky = im.size[1] / rec["ch"]
    ctx = None
    for c in [c for c in cases if c["folio"] == fo]:
        bi = c["box_index"]; b = rec["boxes"][bi]
        if ctx is None:
            wb = [(bb[5], (bb[0] / 1e4 * rec["cw"] * kx, bb[1] / 1e4 * rec["ch"] * ky, (bb[0] + bb[2]) / 1e4 * rec["cw"] * kx, (bb[1] + bb[3]) / 1e4 * rec["ch"] * ky)) for bb in rec["boxes"]]
            ctx = IG.analyse_page(im, [x for _, x in wb], P)
        # map the case's box (from the placed-box list) to the box-file index by geometry
        bx = c["box"]; best = min(range(len(rec["boxes"])), key=lambda i: abs(rec["boxes"][i][0] / 1e4 * rec["cw"] - bx[0]) + abs(rec["boxes"][i][1] / 1e4 * rec["ch"] - bx[1]))
        mem = ctx["members"][best]; i0, i1 = c["span"]
        g3 = IG.word_glyphs(ctx, mem, c["word"], J["widths"], J["default"], P)
        if not any(g is not None and g["est"] for g in g3[i0:i1]): continue
        g4 = IG.word_glyphs(ctx, mem, c["word"], J["widths"], J["default"], P, templates=T)
        gold = (c["gold"][0], c["gold_y"][0], c["gold"][1], c["gold_y"][1]); r = dict(folio=fo, allograph=c["allograph"], word=c["word"])
        for nm, g in (("v3", g3), ("v4", g4)):
            bb = [x["bbox"] for x in g[i0:i1] if x is not None]
            box = (min(q[0] for q in bb) / kx, min(q[1] for q in bb) / ky, max(q[2] for q in bb) / kx, max(q[3] for q in bb) / ky) if bb else None
            r[nm] = iou(box, gold) if box else 0.0
        rows.append(r)
v3 = np.array([r["v3"] for r in rows]); v4 = np.array([r["v4"] for r in rows])
row = dict(tag=TAG, params=" ".join(sys.argv[4:]), n=len(rows), v3_mean=round(v3.mean(), 4), v4_mean=round(v4.mean(), 4), diff=round((v4 - v3).mean(), 4),
           v4_better=int((v4 > v3 + 0.05).sum()), v4_worse=int((v4 < v3 - 0.05).sum()),
           k_n=sum(r["allograph"] == "k" for r in rows), k_diff=round(float(np.mean([r["v4"] - r["v3"] for r in rows if r["allograph"] == "k"] or [0])), 4))
p = os.path.join(OUT, "v4", "dev_variants.csv"); new = not os.path.exists(p)
with open(p, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row)); 
    if new: w.writeheader()
    w.writerow(row)
print(row); print(sorted(((r["allograph"], r["word"], round(r["v3"], 2), round(r["v4"], 2)) for r in rows), key=lambda t: t[3] - t[2])[:6])
