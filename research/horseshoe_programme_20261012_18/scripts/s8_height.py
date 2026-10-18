#!/usr/bin/env python3
"""Stage 8 (EXPLORATORY): each geometry feature vs eo777 path, sheet level (within-section null)
and page level (within section x Davis scribe null)."""
import json, os, pickle, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = sys.argv[1]; rng = np.random.default_rng(20261012)
F = pd.read_csv(os.path.join(ROOT, "folio_bifolium.csv"), index_col=0)
meta = pickle.load(open(os.path.join(ROOT, "data", "s1_words.pkl"), "rb"))["meta"]
sc = meta.assign(folio=meta.folio.astype(str), s=meta.scribe_davis.astype(str)).groupby("folio").s.agg(lambda x: x.mode().iat[0])
F["scribe"] = sc.reindex(F.index).values
def within(a, b, g, n=5000):
    ok = ~np.isnan(b); a, b, g = a[ok], b[ok], g[ok]
    obs = spearmanr(a, b)[0]; sg = np.sign(obs) or 1.0
    gi = [np.where(g == x)[0] for x in np.unique(g)]
    null = np.empty(n)
    for k in range(n):
        bp = b.copy()
        for ix in gi: bp[ix] = b[rng.permutation(ix)]
        null[k] = spearmanr(a, bp)[0] * sg
    return dict(rho=round(float(obs), 4), null_mean=round(float(null.mean()), 4),
                null_sd=round(float(null.std()), 4), z=round(float((obs * sg - null.mean()) / null.std()), 2))
geo = ["logw", "logh", "logasp", "wlen", "n_words"]
S = F.groupby("sheet").agg(**{g: (g, "mean") for g in geo}, path=("path", "mean"),
                           section=("section", lambda s: s.mode().iat[0])).dropna(subset=["path"])
res = {"sheet_level_within_section": {g: within(S[g].values, S.path.values, S.section.values) for g in geo},
       "page_level_within_section_x_scribe": {g: within(F[g].values, F.path.values.astype(float),
            (F.section.astype(str) + "|" + F.scribe.astype(str)).values) for g in geo}}
json.dump(res, open(os.path.join(ROOT, "results_geometry_vs_path.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
