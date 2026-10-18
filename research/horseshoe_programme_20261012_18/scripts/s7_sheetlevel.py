#!/usr/bin/env python3
"""Stage 7 (EXPLORATORY bounding): B2 at bifolium level to remove pseudo-replication.
Aggregate per sheet (mean), Spearman across sheets, permute sheet values within the sheet's modal section."""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = sys.argv[1]; rng = np.random.default_rng(20261012)
F = pd.read_csv(os.path.join(ROOT, "folio_bifolium.csv"), index_col=0)
geo = ["logw", "logh", "logasp", "wlen", "n_words"]
G = F[geo].values.astype(float); G = (G - G.mean(0)) / G.std(0)
A = np.c_[np.ones(len(G)), G]
for c in ["img_d1", "img_d2", "txt_d1"]:
    F[c + "_gr"] = F[c].values - A @ np.linalg.lstsq(A, F[c].values, rcond=None)[0]
S = F.groupby("sheet").agg(path=("path", "mean"), img_d1=("img_d1", "mean"), img_d2=("img_d2", "mean"),
                           img_d1_gr=("img_d1_gr", "mean"), img_d2_gr=("img_d2_gr", "mean"),
                           txt_d1=("txt_d1", "mean"),
                           section=("section", lambda s: s.mode().iat[0]), n=("path", "size")).dropna(subset=["path"])
def within(a, b, g, n=5000):
    obs = spearmanr(a, b)[0]; sg = np.sign(obs) or 1.0
    gi = [np.where(g == x)[0] for x in np.unique(g)]
    null = np.empty(n)
    for k in range(n):
        bp = b.copy()
        for ix in gi: bp[ix] = b[rng.permutation(ix)]
        null[k] = spearmanr(a, bp)[0] * sg
    o = obs * sg
    return dict(n_sheets=len(a), rho=round(float(obs), 4), null_mean=round(float(null.mean()), 4),
                null_sd=round(float(null.std()), 4), z=round(float((o - null.mean()) / null.std()), 2),
                p=float((np.sum(null >= o) + 1) / (n + 1)))
g = S.section.values; p = S.path.values
res = {k: within(S[k].values, p, g) for k in ["txt_d1", "img_d1", "img_d2", "img_d1_gr", "img_d2_gr"]}
res["sections_with_1_sheet"] = int((S.section.value_counts() == 1).sum())
json.dump(res, open(os.path.join(ROOT, "results_sheetlevel.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
