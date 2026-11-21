"""Ink glyph trial, development numbers only (DEV set: iin + k on half the folios; held-out sets untouched).
2-D IoU of the predicted glyph box vs Timm's rectangle: ink method vs width-table slabs at full box height.
Usage: python3 -I k2_dev_metric.py <dl_dir> <out_dir>  -> out/k2_dev_metric.csv"""
import csv, json, os, pickle, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkglyph as IG, glyphsplit as G
DL, OUT = sys.argv[1], sys.argv[2]
cases = [c for c in pickle.load(open(os.path.join(OUT, "g2_cases.pkl"), "rb")) if c["set"] == "DEV"]
corpus = pickle.load(open(os.path.join(OUT, "g2_corpus_profiles.pkl"), "rb"))
S2 = pickle.load(open(os.path.join(OUT, "s2_mapped.pkl"), "rb"))["mapped"]
reg = {json.loads(l)["selected"]["folio"]: json.loads(l)["selected"] for l in open(os.path.join(DL, "reg_full_225.jsonl")) if l.strip()}
J = json.load(open(os.path.join(OUT, "glyph_width_table.json")))
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0
rows = []
for fo in sorted({c["folio"] for c in cases}):
    im = Image.open(os.path.join(DL, "img2000", reg[fo]["service_id"].rsplit("/", 1)[1] + ".jpg")).convert("L")
    k = max(im.size) / max(S2[fo]["full_wh"]); idx = [i for i, r in enumerate(corpus[fo]) if r]
    words = [(corpus[fo][i][0], tuple(v * k for v in corpus[fo][i][1])) for i in idx]; pos = {i: n for n, i in enumerate(idx)}
    ctx, gl = IG.page_glyphs(im, words, J["widths"], J["default"])
    for c in [c for c in cases if c["folio"] == fo]:
        g = gl[pos[c["box_index"]]]; i0, i1 = c["span"]; gold = (c["gold"][0], c["gold_y"][0], c["gold"][1], c["gold_y"][1])
        bb = [x["bbox"] for x in g[i0:i1] if x is not None]
        ink = (min(b[0] for b in bb) / k, min(b[1] for b in bb) / k, max(b[2] for b in bb) / k, max(b[3] for b in bb) / k) if bb else None
        e = G.prior_cuts(c["word"], c["box"][0], c["box"][2], J["widths"], J["pad"]); slab = (e[i0], c["box"][1], e[i1], c["box"][3])
        gc = ((gold[0] + gold[2]) / 2, (gold[1] + gold[3]) / 2)
        rows.append(dict(folio=fo, allograph=c["allograph"], word=c["word"], ink_iou2d=round(iou(ink, gold), 4) if ink else 0.0, slab_iou2d=round(iou(slab, gold), 4),
                         ink_hit=int(bool(ink) and ink[0] <= gc[0] <= ink[2] and ink[1] <= gc[1] <= ink[3]), slab_hit=int(slab[0] <= gc[0] <= slab[2]),
                         ink_fallback=int(any(x is not None and x["est"] for x in g[i0:i1])), ink_missing=int(not bb)))
with open(os.path.join(OUT, "k2_dev_metric.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
A = lambda key, a=None: np.array([r[key] for r in rows if a is None or r["allograph"] == a], float)
for a in (None, "k", "iin"):
    print(a or "all", "n", len(A("ink_iou2d", a)), "| 2-D IoU ink mean %.3f median %.3f vs slab mean %.3f median %.3f | hit ink %.3f slab(x only) %.3f | ink IoU>=0.5: %.3f slab: %.3f | fallback %.3f missing %.3f" % (
        A("ink_iou2d", a).mean(), np.median(A("ink_iou2d", a)), A("slab_iou2d", a).mean(), np.median(A("slab_iou2d", a)), A("ink_hit", a).mean(), A("slab_hit", a).mean(),
        (A("ink_iou2d", a) >= .5).mean(), (A("slab_iou2d", a) >= .5).mean(), A("ink_fallback", a).mean(), A("ink_missing", a).mean()))
