#!/usr/bin/env python3
"""Stage 4 (EXPLORATORY, unregistered): does image MDS dim2 carry Currier A/B beyond crop geometry?"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
ROOT = sys.argv[1]
F = pd.read_csv(os.path.join(ROOT, "folio_mds_coords.csv"), index_col=0)
rng = np.random.default_rng(20261012)
geo = ["logw", "logh", "logasp", "wlen", "n_words"]
G = F[geo].values.astype(float); G = (G - G.mean(0)) / G.std(0)
A = np.c_[np.ones(len(G)), G]
def resid(y): return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
def auc(s, lab):
    m = lab.isin(["A", "B"]).values; s, y = s[m], (lab[m] == "B").values
    r = rankdata(s); a = (r[y].sum() - y.sum()*(y.sum()+1)/2) / (y.sum()*(~y).sum())
    return max(a, 1-a)
def perm_auc(s, lab, n=2000):
    m = lab.isin(["A","B"]).values; s2 = s[m]; l2 = lab[m].values
    obs = auc(s2, pd.Series(l2)); null = np.array([auc(s2, pd.Series(rng.permutation(l2))) for _ in range(n)])
    return dict(auc=round(obs,4), null_mean=round(null.mean(),4), null_sd=round(null.std(),4),
                z=round((obs-null.mean())/null.std(),2))
out = {}
d2 = F.img_d2.values
out["img_d2_currier_raw"] = perm_auc(d2, F.currier)
out["img_d2_currier_geo_partialled"] = perm_auc(resid(d2), F.currier)
out["geometry_only_currier_auc_best_single"] = {g: round(auc(F[g].values, F.currier),4) for g in geo}
# Herbal-only (A vs B within the same section type and layout)
h = F.section.isin(["Herbal-A","Herbal-B"])
out["herbal_only_n"] = int(h.sum())
out["herbal_img_d2_currier_raw"] = perm_auc(d2[h.values], F.currier[h])
Ah = np.c_[np.ones(h.sum()), G[h.values]]
rh = d2[h.values] - Ah @ np.linalg.lstsq(Ah, d2[h.values], rcond=None)[0]
out["herbal_img_d2_currier_geo_partialled"] = perm_auc(rh, F.currier[h])
out["herbal_geometry_currier_auc"] = {g: round(auc(F[g].values[h.values], F.currier[h]),4) for g in geo}
out["text_d1_vs_img_d2_spearman"] = round(spearmanr(F.txt_d1, F.img_d2)[0], 4)
json.dump(out, open(os.path.join(ROOT, "results_dim2_exploratory.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
