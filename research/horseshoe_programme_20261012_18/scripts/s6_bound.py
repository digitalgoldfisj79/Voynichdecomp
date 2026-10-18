#!/usr/bin/env python3
"""Stage 6 (EXPLORATORY bounding, not pre-registered).
(1) B1 with a WITHIN-QUIRE null: is coherence bifolium-specific or quire-level?
(2) B2c/B2e with a WITHIN-(section x Davis scribe) null: does scribe explain the agreement?
"""
import json, os, pickle, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

ROOT = sys.argv[1]; D = os.path.join(ROOT, "data")
rng = np.random.default_rng(20261012)
F = pd.read_csv(os.path.join(ROOT, "folio_bifolium.csv"), index_col=0)
folios, C = pickle.load(open(os.path.join(D, "s2_folio.pkl"), "rb"))[1:]
Dt = pickle.load(open(os.path.join(D, "s2_textdist.pkl"), "rb"))[0]
meta = pickle.load(open(os.path.join(D, "s1_words.pkl"), "rb"))["meta"]
sc = meta.assign(folio=meta.folio.astype(str), s=meta.scribe_davis.astype(str)) \
    .groupby("folio").s.agg(lambda x: x.mode().iat[0])
F["scribe"] = sc.reindex(F.index).values
F["quire_l"] = F.sheet.str.split(":").str[0]
assert list(F.index) == folios

geo = ["logw", "logh", "logasp", "wlen", "n_words"]
G = F[geo].values.astype(float); G = (G - G.mean(0)) / G.std(0)
A = np.c_[np.ones(len(G)), G]
def resid(Y): return Y - A @ np.linalg.lstsq(A, Y, rcond=None)[0]
dists = {"image": squareform(pdist(C, "cosine")), "text": Dt,
         "geometry_only": squareform(pdist(G)), "image_geo_resid": squareform(pdist(resid(C), "cosine"))}
n = len(F); iu = np.triu_indices(n, 1)
sheet, leaf, q = F.sheet.values, F.leaf.astype(str).values, F.quire_l.values
def conj(sh): return (sh[iu[0]] == sh[iu[1]]) & (leaf[iu[0]] != leaf[iu[1]])
def perm_within(lab, grp):
    out = lab.copy()
    for g in np.unique(grp):
        ix = np.where(grp == g)[0]; out[ix] = lab[rng.permutation(ix)]
    return out
res = {"quires": int(len(np.unique(q))), "scribes": F.scribe.value_counts().to_dict()}
nulls = [perm_within(sheet, q) for _ in range(2000)]
for name, Dm in dists.items():
    d = Dm[iu]; obs = d[conj(sheet)].mean()
    null = np.array([d[conj(s)].mean() for s in nulls if conj(s).any()])
    res[f"B1q_{name}"] = dict(obs=round(float(obs), 5), null_mean=round(float(null.mean()), 5),
                              null_sd=round(float(null.std()), 5), z=round(float((null.mean() - obs) / null.std()), 2))

def within(a, b, grp, n=2000):
    ok = ~np.isnan(b); a, b, g = a[ok], b[ok], grp[ok]
    obs = spearmanr(a, b)[0]; sg = np.sign(obs) or 1.0
    gi = [np.where(g == x)[0] for x in np.unique(g)]
    null = np.empty(n)
    for k in range(n):
        bp = b.copy()
        for ix in gi: bp[ix] = b[rng.permutation(ix)]
        null[k] = spearmanr(a, bp)[0] * sg
    return dict(rho=round(float(obs), 4), null_mean=round(float(null.mean()), 4),
                null_sd=round(float(null.std()), 4), z=round(float((obs * sg - null.mean()) / null.std()), 2))
grp = (F.section.astype(str) + "|" + F.scribe.astype(str)).values
path = F.path.values.astype(float)
res["B2c_img_d2_within_section_x_scribe"] = within(F.img_d2.values, path, grp)
res["B2e_img_d2_georesid_within_section_x_scribe"] = within(resid(F.img_d2.values), path, grp)
res["B2a_text_d1_within_section_x_scribe"] = within(F.txt_d1.values, path, grp)
json.dump(res, open(os.path.join(ROOT, "results_bounding.json"), "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
