"""Glyph stage 4: single held-out run of the frozen variant vs the naive equal split.
Usage: python3 -I g4_heldout.py <out_dir>  -> out/g4_heldout.csv"""
import csv, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g3_dev as D
OUT = sys.argv[1]; rng = np.random.default_rng(20261121)
rows = []
for name, sel in (("HELD_A", lambda c: c["set"] == "HELD_A"), ("HELD_B", lambda c: c["set"] == "HELD_B"), ("HELD_A+B", lambda c: c["set"] in ("HELD_A", "HELD_B")),
                  ("HELD_A k only", lambda c: c["set"] == "HELD_A" and c["allograph"] == "k"), ("HELD_A iin only", lambda c: c["set"] == "HELD_A" and c["allograph"] == "iin")):
    cs = [c for c in D.cases if sel(c)]
    v = D.run(cs, "table", True, None, 0.0); n = D.run(cs, "equal", False, None, 0.0)
    fo = np.array([c["folio"] for c in cs]); uf = sorted(set(fo)); idx = {f: np.where(fo == f)[0] for f in uf}
    diffs = []
    for _ in range(2000):
        pick = np.concatenate([idx[f] for f in rng.choice(uf, len(uf))]); diffs.append(v[pick, 0].mean() - n[pick, 0].mean())
    d = v[:, 0].mean() - n[:, 0].mean(); se = float(np.std(diffs, ddof=1))
    rows.append(dict(set=name, n=len(cs), folios=len(uf), table_mean_iou=round(float(v[:, 0].mean()), 4), equal_mean_iou=round(float(n[:, 0].mean()), 4),
                     diff=round(float(d), 4), boot_se=round(se, 4), ratio=round(float(d / se), 2), table_median_iou=round(float(np.median(v[:, 0])), 4),
                     table_hit=round(float(v[:, 1].mean()), 4), equal_hit=round(float(n[:, 1].mean()), 4), table_centre_err_med=round(float(np.median(v[:, 2])), 4)))
with open(os.path.join(OUT, "g4_heldout.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
