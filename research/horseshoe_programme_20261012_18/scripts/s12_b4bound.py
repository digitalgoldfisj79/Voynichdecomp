#!/usr/bin/env python3
"""Stage 12 (EXPLORATORY bounding of B4): sheet-level aggregation and content-vs-content pairs only."""
import json, os, pickle, sys
import numpy as np, pandas as pd
from scipy.stats import binomtest
ROOT = sys.argv[1]; rng = np.random.default_rng(20261012)
R = json.load(open(os.path.join(ROOT, "results_canonical.json")))
F = pd.read_csv(os.path.join(ROOT, "folio_bifolium_canonical.csv"), index_col=0)
Dt = pickle.load(open(os.path.join(ROOT, "data", "s2_textdist.pkl"), "rb"))[0]
idx = {f: i for i, f in enumerate(F.index)}
sec, sheet, scr = F.csec.values, F.sheet.values, F.scribe.astype(str).values
def pool(i, j, same_scribe=True):
    A_ = np.where(sec == sec[i])[0]; B_ = np.where(sec == sec[j])[0]
    c = [Dt[a, b] for a in A_ for b in B_ if sheet[a] != sheet[b] and a != b
         and (not same_scribe or (scr[a] == scr[i] and scr[b] == scr[j]))]
    return np.array(c) if c else pool(i, j, False)
out = {}
for tag, keep in (("all", lambda a, b: True), ("no_text_only", lambda a, b: "text-only" not in (a, b))):
    pairs = [(idx[p[0]], idx[p[1]], p[4]) for p in R["B4_pairs"] if keep(p[2], p[3])]
    by = {}
    for i, j, s in pairs: by.setdefault(s, []).append((i, j))
    rows = []
    for s, ps in by.items():
        obs = np.mean([Dt[i, j] for i, j in ps]); pl = [pool(i, j) for i, j in ps]
        rows.append((s, len(ps), round(float(obs), 4), round(float(np.mean([p.mean() for p in pl])), 4)))
    wins = sum(o < m for _, _, o, m in rows)
    # sheet-level null: one matched pair per conjugate pair, then average within sheet, then across sheets
    pools = {s: [pool(i, j) for i, j in ps] for s, ps in by.items()}
    obs_sheet = np.mean([r[2] for r in rows])
    null = np.array([np.mean([np.mean([rng.choice(p) for p in pools[s]]) for s in by]) for _ in range(5000)])
    out[tag] = dict(n_sheets=len(rows), n_pairs=len(pairs), sheets_closer=int(wins),
                    sign_p=float(binomtest(int(wins), len(rows), 0.5, alternative="greater").pvalue),
                    sheet_level_obs=round(float(obs_sheet), 4), null_mean=round(float(null.mean()), 4),
                    null_sd=round(float(null.std()), 4), z=round(float((null.mean() - obs_sheet) / null.std()), 2),
                    per_sheet=rows)
json.dump(out, open(os.path.join(ROOT, "results_b4_bounding.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
