#!/usr/bin/env python3
"""Stage 19 (EXPLORATORY bounding of F1): skin-side/legibility alternative and fragility."""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = sys.argv[1]; T = pd.read_csv(os.path.join(ROOT, "faces_per_sheet.csv")); rng = np.random.default_rng(20261012)
nc = T[~T.central]
res = {}
# (1) if skin side drives both, per-sheet text and image face effects should correlate
for k in ("D_image_raw", "D_image_geo_resid", "D_geometry_only"):
    r, p = spearmanr(nc.D_text_JS, nc[k]); res[f"rho_text_vs_{k}"] = dict(rho=round(float(r), 3), p=round(float(p), 4))
# (2) residual text effect after removing the part linearly predictable from the image face effect
A = np.c_[np.ones(len(nc)), nc.D_image_raw.values]
b, *_ = np.linalg.lstsq(A, nc.D_text_JS.values, rcond=None)
res["text_effect_at_zero_image_effect (intercept)"] = float(b[0]); res["slope_on_image"] = float(b[1])
d = nc.D_text_JS.values
# (3) fragility: leave-one-sheet-out min z, and drop the 3 largest contributions
def z(x):
    null = (x[None, :] * rng.choice([1, -1], size=(50000, len(x)))).mean(1); return float(x.mean() / null.std())
res["loo_min_z"] = min(z(np.delete(d, i)) for i in range(len(d)))
res["drop_top3_z"] = z(np.sort(d)[:-3])
res["median_D"] = float(np.median(d)); res["trimmed_mean_10pct"] = float(pd.Series(d).sort_values().iloc[3:-3].mean())
json.dump(res, open(os.path.join(ROOT, "results_faces_bounding.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
