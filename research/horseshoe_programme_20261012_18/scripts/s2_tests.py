#!/usr/bin/env python3
"""Stage 2: folio features, MDS horseshoes (image + text replication), Q1-Q3 as pre-registered.

Each sub-stage pickles to data/ and is skipped on rerun.
"""
import json, os, pickle, sys
from collections import Counter
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

ROOT = sys.argv[1]
D = os.path.join(ROOT, "data")
SEED = 20261012
rng = np.random.default_rng(SEED)


def stage(name, fn):
    p = os.path.join(D, f"s2_{name}.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    r = fn()
    with open(p + ".tmp", "wb") as f:
        pickle.dump(r, f)
    os.replace(p + ".tmp", p)
    return r


def cmds(Dm, k=4):
    """Classical (Torgerson) MDS."""
    n = Dm.shape[0]
    Jc = np.eye(n) - np.ones((n, n)) / n
    Bm = -0.5 * Jc @ (Dm ** 2) @ Jc
    w, V = np.linalg.eigh(Bm)
    o = np.argsort(w)[::-1][:k]
    w, V = w[o], V[:, o]
    return V * np.sqrt(np.clip(w, 0, None)), w


def guttman(Y):
    """R2 of dim2 ~ dim1 (linear) and dim2 ~ dim1 + dim1^2."""
    x, y = Y[:, 0], Y[:, 1]
    def r2(A):
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ b
        return 1 - res.var() / y.var()
    one = np.ones_like(x)
    return dict(lin=r2(np.c_[one, x]), quad=r2(np.c_[one, x, x ** 2]))


S1 = pickle.load(open(os.path.join(D, "s1_words.pkl"), "rb"))
X, meta = S1["X"], S1["meta"].copy()
for _c in ("folio", "section_fixed", "currier", "quire", "word", "line_num", "word_index"):
    if _c in meta and str(meta[_c].dtype) == "category":
        meta[_c] = meta[_c].astype(str)
meta["line_num"] = pd.to_numeric(meta["line_num"], errors="coerce")
meta["word_index"] = meta["word_index"].astype(int)
meta["wlen"] = meta["word"].astype(str).str.len()
meta["logw"], meta["logh"] = np.log(meta["w"]), np.log(meta["h"])
meta["logasp"] = meta["logw"] - meta["logh"]


# ------------------------------------------------------------- folio table
def build_folio():
    g = meta.groupby("folio")
    F = pd.DataFrame({
        "n_words": g.size(),
        "logw": g.logw.mean(), "logh": g.logh.mean(), "logasp": g.logasp.mean(),
        "wlen": g.wlen.mean(),
        "section": g.section_fixed.agg(lambda s: s.mode().iat[0]),
        "currier": g.currier.agg(lambda s: s.mode().iat[0]),
        "quire": g.quire.agg(lambda s: s.mode().iat[0]),
    })
    folios = F.index.tolist()
    idx = {f: i for i, f in enumerate(folios)}
    C = np.zeros((len(folios), X.shape[1]))
    np.add.at(C, meta.folio.map(idx).values.astype(int), X)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return F, folios, C


F, folios, C = stage("folio", build_folio)


# ------------------------------------------------------------- text replication
def text_dist():
    """npcompl33t-style: per-folio glyph transitions within words (with ^ $ boundaries)
    and within lines with spaces removed; combined; Jensen-Shannon distance."""
    counts = {}
    for f, sub in meta.sort_values(["folio", "line_num", "word_index"]).groupby("folio"):
        c = Counter()
        for w in sub.word.astype(str):
            s = "^" + w + "$"
            c.update(("w", a, b) for a, b in zip(s, s[1:]))
        for _, ln in sub.groupby("line_num"):
            s = "".join(ln.word.astype(str))
            c.update(("l", a, b) for a, b in zip(s, s[1:]))
        counts[f] = c
    vocab = sorted(set().union(*counts.values()))
    vi = {v: i for i, v in enumerate(vocab)}
    M = np.zeros((len(folios), len(vocab)))
    for i, f in enumerate(folios):
        for k, n in counts[f].items():
            M[i, vi[k]] = n
    # combine word and line matrices with equal weight, then normalise
    wmask = np.array([v[0] == "w" for v in vocab])
    P = np.zeros_like(M)
    P[:, wmask] = M[:, wmask] / M[:, wmask].sum(1, keepdims=True)
    P[:, ~wmask] = M[:, ~wmask] / M[:, ~wmask].sum(1, keepdims=True)
    P = (P + 1e-9) / (P + 1e-9).sum(1, keepdims=True)
    from scipy.spatial.distance import jensenshannon
    Dm = squareform(pdist(P, lambda a, b: jensenshannon(a, b, base=2)))
    return Dm, len(vocab)


Dt, nvocab = stage("textdist", text_dist)
Di = squareform(pdist(C, "cosine"))

Yi, wi = stage("mds_img", lambda: cmds(Di))
Yt, wt = stage("mds_txt", lambda: cmds(Dt))
res = {}
res["n_folios"] = len(folios)
res["text_vocab_bigrams"] = nvocab
res["Q1_image_guttman"] = guttman(Yi)
res["Q1_text_guttman"] = guttman(Yt)
res["mds_var_share_img"] = (wi[:2] / wi[wi > 0].sum()).round(4).tolist()
res["mds_var_share_txt"] = (wt[:2] / wt[wt > 0].sum()).round(4).tolist()


# ------------------------------------------------------------- Q2 geometry
geo = ["logw", "logh", "logasp", "wlen", "n_words"]
G = F[geo].values.astype(float)
G = (G - G.mean(0)) / G.std(0)


def loo_ridge_r2(Gm, y, lam=1.0):
    pred = np.zeros_like(y)
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        A = np.c_[np.ones(m.sum()), Gm[m]]
        b = np.linalg.solve(A.T @ A + lam * np.diag([0] + [1] * Gm.shape[1]), A.T @ y[m])
        pred[i] = np.r_[1, Gm[i]] @ b
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


for tag, Y in (("image", Yi), ("text", Yt)):
    res[f"Q2_{tag}_dim1_spearman_geo"] = {g: round(spearmanr(F[g], Y[:, 0])[0], 4) for g in geo}
    res[f"Q2_{tag}_dim1_loo_ridge_R2_geo"] = round(loo_ridge_r2(G, Y[:, 0]), 4)
    res[f"Q2_{tag}_dim2_loo_ridge_R2_geo"] = round(loo_ridge_r2(G, Y[:, 1]), 4)


# word-level arc position on the public UMAP map
def umap_arc():
    from sklearn.manifold import Isomap
    U = meta[["umap_x", "umap_y"]].values
    sub = rng.choice(len(U), 6000, replace=False)
    iso = Isomap(n_neighbors=15, n_components=1).fit(U[sub])
    return iso.transform(U)[:, 0]


arc = stage("umap_arc", umap_arc)
res["Q2_wordlevel_umaparc_spearman"] = {
    g: round(spearmanr(meta[g], arc)[0], 4) for g in ["logw", "logh", "logasp", "wlen"]}


# ------------------------------------------------------------- Q3 shared gradient
def within_perm_test(a, b, groups, n=2000):
    obs = spearmanr(a, b)[0]
    sgn = np.sign(obs) if obs != 0 else 1.0
    gidx = [np.where(groups == g)[0] for g in np.unique(groups)]
    null = np.empty(n)
    for k in range(n):
        bp = b.copy()
        for ix in gidx:
            bp[ix] = b[rng.permutation(ix)]
        null[k] = spearmanr(a, bp)[0] * sgn
    o = obs * sgn
    return dict(rho=round(obs, 4), null_mean=round(null.mean(), 4), null_sd=round(null.std(), 4),
                z=round((o - null.mean()) / null.std(), 2),
                p=float((np.sum(null >= o) + 1) / (n + 1)))


sec = F["section"].values
res["Q3_raw"] = within_perm_test(Yi[:, 0], Yt[:, 0], sec)
A = np.c_[np.ones(len(G)), G]
resid = Yi[:, 0] - A @ np.linalg.lstsq(A, Yi[:, 0], rcond=None)[0]
res["Q3_geometry_partialled"] = within_perm_test(resid, Yt[:, 0], sec)


# descriptive: what each dim1 tracks
def auc(score, lab):
    m = lab.isin(["A", "B"])
    s, y = score[m.values], (lab[m] == "B").values
    from scipy.stats import rankdata
    r = rankdata(s)
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


for tag, Y in (("image", Yi), ("text", Yt)):
    a = auc(Y[:, 0], F["currier"])
    res[f"desc_{tag}_dim1_currierB_auc"] = round(max(a, 1 - a), 4)
    eta = pd.Series(Y[:, 0]).groupby(sec).transform("mean")
    res[f"desc_{tag}_dim1_section_eta2"] = round(float(eta.var() / pd.Series(Y[:, 0]).var()), 4)

out = F.copy()
out["img_d1"], out["img_d2"], out["txt_d1"], out["txt_d2"] = Yi[:, 0], Yi[:, 1], Yt[:, 0], Yt[:, 1]
out.to_csv(os.path.join(ROOT, "folio_mds_coords.csv"))
meta.assign(umap_arc=arc)[["id", "folio", "word", "umap_x", "umap_y", "logw", "logh",
                           "logasp", "wlen", "section_fixed", "currier"]].assign(umap_arc=arc) \
    .to_parquet(os.path.join(D, "word_level.parquet"))
json.dump(res, open(os.path.join(ROOT, "results_q1_q3.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
