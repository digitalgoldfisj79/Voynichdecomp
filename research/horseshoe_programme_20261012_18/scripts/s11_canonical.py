#!/usr/bin/env python3
"""Stage 11 (PREREG 4): rerun section-stratified tests under the canonical section map, plus B4."""
import json, os, pickle, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

ROOT, MAP = sys.argv[1], sys.argv[2]
D = os.path.join(ROOT, "data"); rng = np.random.default_rng(20261012)
canon = json.load(open(MAP))["mapping"]
F = pd.read_csv(os.path.join(ROOT, "folio_bifolium.csv"), index_col=0)
folios, C = pickle.load(open(os.path.join(D, "s2_folio.pkl"), "rb"))[1:]
Dt = pickle.load(open(os.path.join(D, "s2_textdist.pkl"), "rb"))[0]
meta = pickle.load(open(os.path.join(D, "s1_words.pkl"), "rb"))["meta"]
m = meta.assign(folio=meta.folio.astype(str), s=meta.scribe_davis.astype(str))
F["scribe"] = m.groupby("folio").s.agg(lambda x: x.mode().iat[0]).reindex(F.index).values
F["csec"] = [canon[f] for f in F.index]
F["currier"] = F.currier.fillna("none").astype(str)
assert list(F.index) == folios
geo = ["logw", "logh", "logasp", "wlen", "n_words"]
G = F[geo].values.astype(float); G = (G - G.mean(0)) / G.std(0); A = np.c_[np.ones(len(G)), G]
def resid(Y): return Y - A @ np.linalg.lstsq(A, Y, rcond=None)[0]
n = len(F); iu = np.triu_indices(n, 1)
sheet, leaf, sec = F.sheet.values, F.leaf.astype(str).values, F.csec.values
def conj(sh): return (sh[iu[0]] == sh[iu[1]]) & (leaf[iu[0]] != leaf[iu[1]])
def perm_within(lab, grp):
    out = lab.copy()
    for g in np.unique(grp):
        ix = np.where(grp == g)[0]; out[ix] = lab[rng.permutation(ix)]
    return out
res = {"canonical_section_counts": F.csec.value_counts().to_dict()}

# descriptive: section order along text dim1, eta2
res["txt_d1_order_canonical"] = F.groupby("csec").txt_d1.mean().sort_values().round(4).to_dict()
for c in ("txt_d1", "img_d1", "img_d2"):
    eta = F.groupby("csec")[c].transform("mean")
    res[f"eta2_{c}_canonical"] = round(float(eta.var() / F[c].var()), 4)
    eta2 = F.groupby("section")[c].transform("mean")
    res[f"eta2_{c}_atlas_scheme"] = round(float(eta2.var() / F[c].var()), 4)

# Q3 (page level, as registered)
def within_rho(a, b, g, draws=2000):
    ok = ~np.isnan(b); a, b, g = a[ok], b[ok], g[ok]
    obs = spearmanr(a, b)[0]; sg = np.sign(obs) or 1.0
    gi = [np.where(g == x)[0] for x in np.unique(g)]
    null = np.empty(draws)
    for k in range(draws):
        bp = b.copy()
        for ix in gi: bp[ix] = b[rng.permutation(ix)]
        null[k] = spearmanr(a, bp)[0] * sg
    return dict(n=int(len(a)), rho=round(float(obs), 4), null_mean=round(float(null.mean()), 4),
                null_sd=round(float(null.std()), 4), z=round(float((obs * sg - null.mean()) / null.std()), 2),
                p=float((np.sum(null >= obs * sg) + 1) / (draws + 1)))
res["Q3_raw"] = within_rho(F.img_d1.values, F.txt_d1.values, sec)
res["Q3_geo"] = within_rho(resid(F.img_d1.values), F.txt_d1.values, sec)

# B1 within canonical section
dists = {"image": squareform(pdist(C, "cosine")), "text": Dt, "geometry_only": squareform(pdist(G)),
         "image_geo_resid": squareform(pdist(resid(C), "cosine"))}
def b1(strata, draws=2000):
    nulls = [perm_within(sheet, strata) for _ in range(draws)]
    out = {}
    for name, Dm in dists.items():
        d = Dm[iu]; obs = d[conj(sheet)].mean()
        null = np.array([d[conj(s)].mean() for s in nulls if conj(s).any()])
        out[name] = dict(obs=round(float(obs), 5), null_mean=round(float(null.mean()), 5),
                         null_sd=round(float(null.std()), 5), z=round(float((null.mean() - obs) / null.std()), 2),
                         pct_lower=round(float(100 * (null.mean() - obs) / null.mean()), 2))
    return out
res["B1_within_canonical_section"] = b1(sec)
res["B3c_section_x_scribe_x_currier"] = b1((F.csec + "|" + F.scribe.astype(str) + "|" + F.currier).values)

# B2 sheet level within canonical section
for c in ["img_d1", "img_d2", "txt_d1"]:
    F[c + "_gr"] = resid(F[c].values)
S = F.groupby("sheet").agg(path=("path", "mean"), **{k: (k, "mean") for k in
                           ["txt_d1", "img_d1", "img_d2", "img_d1_gr", "img_d2_gr"]},
                           csec=("csec", lambda s: s.mode().iat[0])).dropna(subset=["path"])
res["B2_sheet_level"] = {k: within_rho(S[k].values, S.path.values, S.csec.values, 5000)
                         for k in ["txt_d1", "img_d1", "img_d2", "img_d1_gr", "img_d2_gr"]}

# B4 cross-section conjugates
cm = conj(sheet)
pairs = [(i, j) for i, j, c in zip(iu[0], iu[1], cm) if c and sec[i] != sec[j]]
res["B4_n_cross_section_conjugate_pairs"] = len(pairs)
res["B4_pairs"] = [(F.index[i], F.index[j], sec[i], sec[j], F.sheet.iloc[i], round(float(Dt[i, j]), 4)) for i, j in pairs]
if pairs:
    pools = []
    for i, j in pairs:
        A_ = np.where(sec == sec[i])[0]; B_ = np.where(sec == sec[j])[0]
        cand = [(a, b) for a in A_ for b in B_ if sheet[a] != sheet[b] and a != b
                and F.scribe.values[a] == F.scribe.values[i] and F.scribe.values[b] == F.scribe.values[j]]
        fallback = not cand
        if fallback:
            cand = [(a, b) for a in A_ for b in B_ if sheet[a] != sheet[b] and a != b]
        pools.append((np.array([Dt[a, b] for a, b in cand]), fallback))
    obs = float(np.mean([Dt[i, j] for i, j in pairs]))
    null = np.array([np.mean([rng.choice(p) for p, _ in pools]) for _ in range(5000)])
    res["B4"] = dict(n=len(pairs), obs=round(obs, 5), null_mean=round(float(null.mean()), 5),
                     null_sd=round(float(null.std()), 5), z=round(float((null.mean() - obs) / null.std()), 2),
                     pct_lower=round(float(100 * (null.mean() - obs) / null.mean()), 2),
                     pairs_using_scribe_fallback=int(sum(fb for _, fb in pools)),
                     per_pair_closer_than_pool_median=int(sum(Dt[i, j] < np.median(p) for (i, j), (p, _) in zip(pairs, pools))))
F.to_csv(os.path.join(ROOT, "folio_bifolium_canonical.csv"))
json.dump(res, open(os.path.join(ROOT, "results_canonical.json"), "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
