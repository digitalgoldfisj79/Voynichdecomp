#!/usr/bin/env python3
"""Stage 5: bifolium coherence (B1) and eo777 sequence agreement (B2), as pre-registered.

Inputs: folio_mds_coords.csv, data/s2_folio.pkl (folio centroids), data/s2_textdist.pkl,
eo777 golf-v21-assignments.csv. Outputs: results_bifolium.json, folio_bifolium.csv,
word_bifolium_column.csv (for re-upload to Nomic).
"""
import json, os, pickle, re, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

ROOT, EO = sys.argv[1], sys.argv[2]
D = os.path.join(ROOT, "data")
rng = np.random.default_rng(20261012)
F = pd.read_csv(os.path.join(ROOT, "folio_mds_coords.csv"), index_col=0)
folios, C = pickle.load(open(os.path.join(D, "s2_folio.pkl"), "rb"))[1:]
Dt = pickle.load(open(os.path.join(D, "s2_textdist.pkl"), "rb"))[0]
assert list(F.index) == folios

e = pd.read_csv(EO)
pg = e.groupby("page").agg(sheet=("sheet", "first"), path=("page_path_position", "first"),
                           bpath=("bifolium_path_position", "first"),
                           golf=("assignment_code", lambda s: "|".join(sorted({str(v) for v in s.dropna()}))))
alias = {"f101r1": "f101r", "f101v2": "f101v"}
F["eo_page"] = [alias.get(f, f) for f in F.index]
F = F.join(pg, on="eo_page")
assert F.sheet.notna().all(), F[F.sheet.isna()].index.tolist()
F["leaf"] = [re.match(r"f(\d+)", f).group(1) for f in F.index]
print("sheets", F.sheet.nunique(), "| pages per sheet", F.sheet.value_counts().describe().round(2).to_dict())

geo = ["logw", "logh", "logasp", "wlen", "n_words"]
G = F[geo].values.astype(float); G = (G - G.mean(0)) / G.std(0)
A = np.c_[np.ones(len(G)), G]
def resid(Y): return Y - A @ np.linalg.lstsq(A, Y, rcond=None)[0]

dists = {
    "image": squareform(pdist(C, "cosine")),
    "text": Dt,
    "geometry_only": squareform(pdist(G, "euclidean")),
    "image_geo_resid": squareform(pdist(resid(C), "cosine")),
}
sheet = F.sheet.values; leaf = F.leaf.values; sec = F.section.values
n = len(F); iu = np.triu_indices(n, 1)
same_sec_pairs = sec[iu[0]] == sec[iu[1]]


def conj_mask(sh):
    return (sh[iu[0]] == sh[iu[1]]) & (leaf[iu[0]] != leaf[iu[1]])


def permute_within_section(sh):
    out = sh.copy()
    for s in np.unique(sec):
        ix = np.where(sec == s)[0]
        out[ix] = sh[rng.permutation(ix)]
    return out


nulls_sheets = [permute_within_section(sheet) for _ in range(2000)]
res = {"n_conjugate_pairs": int(conj_mask(sheet).sum()),
       "n_conjugate_pairs_same_section": int((conj_mask(sheet) & same_sec_pairs).sum())}
for name, Dm in dists.items():
    d = Dm[iu]
    obs = d[conj_mask(sheet)].mean()
    null = np.array([d[conj_mask(s)].mean() if conj_mask(s).any() else np.nan for s in nulls_sheets])
    null = null[~np.isnan(null)]
    res[f"B1_{name}"] = dict(obs=round(float(obs), 5), null_mean=round(float(null.mean()), 5),
                             null_sd=round(float(null.std()), 5),
                             z=round(float((null.mean() - obs) / null.std()), 2),
                             p=float((np.sum(null <= obs) + 1) / (len(null) + 1)))


def within_perm(a, b, n=2000):
    ok = ~np.isnan(b)
    a, b, s = a[ok], b[ok], sec[ok]
    obs = spearmanr(a, b)[0]; sg = np.sign(obs) or 1.0
    gidx = [np.where(s == g)[0] for g in np.unique(s)]
    null = np.empty(n)
    for k in range(n):
        bp = b.copy()
        for ix in gidx:
            bp[ix] = b[rng.permutation(ix)]
        null[k] = spearmanr(a, bp)[0] * sg
    o = obs * sg
    return dict(n=int(ok.sum()), rho=round(float(obs), 4), null_mean=round(float(null.mean()), 4),
                null_sd=round(float(null.std()), 4), z=round(float((o - null.mean()) / null.std()), 2),
                p=float((np.sum(null >= o) + 1) / (n + 1)))


path = F.path.values.astype(float)
ok = ~np.isnan(path)
res["B2a_text_d1_vs_path_rho"] = round(float(spearmanr(F.txt_d1[ok], path[ok])[0]), 4)
res["B2b_img_d1"] = within_perm(F.img_d1.values, path)
res["B2c_img_d2"] = within_perm(F.img_d2.values, path)
res["B2d_img_d1_geo_resid"] = within_perm(resid(F.img_d1.values), path)
res["B2e_img_d2_geo_resid"] = within_perm(resid(F.img_d2.values), path)
res["path_missing_pages"] = F.index[~ok].tolist()

F.to_csv(os.path.join(ROOT, "folio_bifolium.csv"))
json.dump(res, open(os.path.join(ROOT, "results_bifolium.json"), "w"), indent=1)
print(json.dumps(res, indent=1))

# per-word column for a future Nomic re-upload
W = pd.read_parquet(os.path.join(D, "word_level.parquet"))[["id", "folio"]]
W = W.merge(F[["sheet", "leaf", "path", "golf"]].rename(columns={
    "sheet": "bifolium", "path": "eo777_page_path", "golf": "eo777_golf"}), left_on="folio",
    right_index=True, how="left")
assert W.bifolium.notna().all()
W["quire_letter"] = W.bifolium.str.split(":").str[0]
W.to_csv(os.path.join(ROOT, "word_bifolium_column.csv"), index=False)
print("word rows", len(W))
