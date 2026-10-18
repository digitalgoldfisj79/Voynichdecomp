#!/usr/bin/env python3
"""Stage 9 (pre-registered, PREREG 3): bifolium coherence of text beyond scribe."""
import json, os, pickle, sys
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform
ROOT = sys.argv[1]; D = os.path.join(ROOT, "data"); rng = np.random.default_rng(20261012)
F = pd.read_csv(os.path.join(ROOT, "folio_bifolium.csv"), index_col=0)
folios, C = pickle.load(open(os.path.join(D, "s2_folio.pkl"), "rb"))[1:]
Dt = pickle.load(open(os.path.join(D, "s2_textdist.pkl"), "rb"))[0]
meta = pickle.load(open(os.path.join(D, "s1_words.pkl"), "rb"))["meta"]
m = meta.assign(folio=meta.folio.astype(str), s=meta.scribe_davis.astype(str))
F["scribe"] = m.groupby("folio").s.agg(lambda x: x.mode().iat[0]).reindex(F.index).values
F["quire_l"] = F.sheet.str.split(":").str[0]
assert list(F.index) == folios
geo = ["logw", "logh", "logasp", "wlen", "n_words"]
G = F[geo].values.astype(float); G = (G - G.mean(0)) / G.std(0); A = np.c_[np.ones(len(G)), G]
Ci = C - A @ np.linalg.lstsq(A, C, rcond=None)[0]
dist = {"text": Dt, "image_geo_resid": squareform(pdist(Ci, "cosine"))}
n = len(F); iu = np.triu_indices(n, 1)
sheet, leaf = F.sheet.values, F.leaf.astype(str).values
def conj(sh): return (sh[iu[0]] == sh[iu[1]]) & (leaf[iu[0]] != leaf[iu[1]])
cm = conj(sheet)
res = {"n_conjugate_pairs": int(cm.sum()),
       "conjugate_pairs_same_scribe": int((F.scribe.values[iu[0]] == F.scribe.values[iu[1]])[cm].sum()),
       "conjugate_pairs_same_currier": int((F.currier.values[iu[0]] == F.currier.values[iu[1]])[cm].sum())}
strata = {"B3a_quire_x_scribe": F.quire_l + "|" + F.scribe.astype(str),
          "B3b_quire_x_scribe_x_currier": F.quire_l + "|" + F.scribe.astype(str) + "|" + F.currier.fillna("none").astype(str),
          "B3c_section_x_scribe_x_currier": F.section.astype(str) + "|" + F.scribe.astype(str) + "|" + F.currier.fillna("none").astype(str)}
for sname, st in strata.items():
    st = st.values
    # power: conjugate pairs whose (both) pages sit in a stratum containing >= 2 sheets
    nsheets = pd.Series(sheet).groupby(st).nunique()
    inf = np.array([nsheets[st[i]] >= 2 and nsheets[st[j]] >= 2 for i, j in zip(*iu)])
    res[f"{sname}_informative_pairs"] = int((cm & inf).sum())
    gi = [np.where(st == g)[0] for g in np.unique(st)]
    nulls = []
    for _ in range(5000):
        s2 = sheet.copy()
        for ix in gi: s2[ix] = sheet[rng.permutation(ix)]
        nulls.append(s2)
    for dname, Dm in dist.items():
        d = Dm[iu]; obs = d[cm].mean()
        null = np.array([d[conj(s)].mean() for s in nulls if conj(s).any()])
        res[f"{sname}__{dname}"] = dict(obs=round(float(obs), 5), null_mean=round(float(null.mean()), 5),
            null_sd=round(float(null.std()), 5), z=round(float((null.mean() - obs) / null.std()), 2),
            pct_lower=round(float(100 * (null.mean() - obs) / null.mean()), 2),
            p=float((np.sum(null <= obs) + 1) / (len(null) + 1)))
json.dump(res, open(os.path.join(ROOT, "results_scribe.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
