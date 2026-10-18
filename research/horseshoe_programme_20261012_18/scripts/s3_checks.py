#!/usr/bin/env python3
"""Stage 3 (unregistered robustness checks + plots). Labelled exploratory in the results file."""
import json, os, pickle, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = sys.argv[1]; D = os.path.join(ROOT, "data")
F = pd.read_csv(os.path.join(ROOT, "folio_mds_coords.csv"), index_col=0)
W = pd.read_parquet(os.path.join(D, "word_level.parquet"))
S1 = pickle.load(open(os.path.join(D, "s1_words.pkl"), "rb"))
X = S1["X"]
C = pickle.load(open(os.path.join(D, "s2_folio.pkl"), "rb"))[2]
Dt = pickle.load(open(os.path.join(D, "s2_textdist.pkl"), "rb"))[0]
out = {}


def cmds(Dm, k=3):
    n = Dm.shape[0]; Jc = np.eye(n) - 1 / n
    Bm = -0.5 * Jc @ (Dm ** 2) @ Jc
    w, V = np.linalg.eigh(Bm); o = np.argsort(w)[::-1][:k]
    return V[:, o] * np.sqrt(np.clip(w[o], 0, None)), w[o] / w[w > 0].sum()


def quad_r2(x, y):
    A = np.c_[np.ones_like(x), x, x ** 2]
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return 1 - (y - A @ b).var() / y.var()


# 1. outliers on image dim1
z = (F.img_d1 - F.img_d1.median()) / (1.4826 * (F.img_d1 - F.img_d1.median()).abs().median())
out["img_d1_extreme_folios"] = F.assign(z=z).sort_values("z", key=abs, ascending=False) \
    .head(8)[["n_words", "section", "img_d1", "z"]].round(3).reset_index().values.tolist()

# 2. re-run image MDS excluding folios with < 30 words
keep = (F.n_words >= 30).values
Yk, vk = cmds(squareform(pdist(C[keep], "cosine")))
Fk = F[keep]
out["img_ge30_n"] = int(keep.sum())
out["img_ge30_var_share"] = vk[:2].round(4).tolist()
out["img_ge30_quadR2"] = round(quad_r2(Yk[:, 0], Yk[:, 1]), 4)
out["img_ge30_d1_rho_logw"] = round(spearmanr(Fk.logw, Yk[:, 0])[0], 4)
out["img_ge30_d1_rho_txt_d1"] = round(spearmanr(Fk.txt_d1, Yk[:, 0])[0], 4)

# 3. text horseshoe higher-order check (his diagnostic): dim3 vs dim1 cubic
Yt, vt = cmds(Dt)
A3 = np.c_[np.ones(len(Yt)), Yt[:, 0], Yt[:, 0] ** 2, Yt[:, 0] ** 3]
b, *_ = np.linalg.lstsq(A3, Yt[:, 2], rcond=None)
out["txt_dim3_cubic_R2"] = round(1 - (Yt[:, 2] - A3 @ b).var() / Yt[:, 2].var(), 4)
out["txt_var_share_3"] = vt.round(4).tolist()

# 4. face validity of the text replication vs npcompl33t's section order
sec_order = F.groupby("section").txt_d1.mean().sort_values()
out["txt_d1_section_order"] = sec_order.round(4).to_dict()
out["txt_d1_rank_f111"] = {f: int(F.txt_d1.rank()[f]) for f in F.index if f.startswith("f111")}

# 5. word-level PCA of the vectors (closest analogue of the word map Ed looked at)
rng = np.random.default_rng(20261012)
sub = rng.choice(len(X), 12000, replace=False)
Xs = X[sub] - X[sub].mean(0)
U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
P = Xs @ Vt[:2].T
Ws = W.iloc[sub]
out["word_pca_var_share"] = (s[:2] ** 2 / (s ** 2).sum()).round(4).tolist()
out["word_pca_quadR2"] = round(quad_r2(P[:, 0], P[:, 1]), 4)
out["word_pc1_rho"] = {g: round(spearmanr(Ws[g], P[:, 0])[0], 4) for g in ["logw", "logh", "logasp", "wlen"]}
out["word_pc2_rho"] = {g: round(spearmanr(Ws[g], P[:, 1])[0], 4) for g in ["logw", "logh", "logasp", "wlen"]}

# 6. how separable is the UMAP map into pieces (Isomap warned of 2 components)
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components
G = kneighbors_graph(W[["umap_x", "umap_y"]].values[sub], 15)
ncomp, lab = connected_components(G, directed=False)
out["umap_knn15_components"] = int(ncomp)
out["umap_component_sizes"] = np.bincount(lab).tolist()

json.dump(out, open(os.path.join(ROOT, "results_checks.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))

# ---------------- plots
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
sc = ax[0, 0].scatter(W.umap_x, W.umap_y, c=W.logw, s=1, cmap="viridis", rasterized=True)
ax[0, 0].set_title("Public Nomic word map (UMAP), colour = log crop width"); fig.colorbar(sc, ax=ax[0, 0])
sc = ax[0, 1].scatter(W.umap_x, W.umap_y, c=W.wlen.clip(1, 10), s=1, cmap="magma", rasterized=True)
ax[0, 1].set_title("Same map, colour = EVA word length"); fig.colorbar(sc, ax=ax[0, 1])
cmap = {"A": "tab:blue", "B": "tab:red"}
cols = [cmap.get(c, "lightgrey") for c in F.currier]
ax[1, 0].scatter(F.txt_d1, F.txt_d2, c=cols, s=18)
ax[1, 0].set_title("TEXT replication: folio MDS (JSD of glyph transitions)\nblue = Currier A, red = B, grey = unassigned")
ax[1, 1].scatter(F.img_d1, F.img_d2, c=cols, s=18)
ax[1, 1].set_title("IMAGE: folio MDS (cosine of mean word vector)\nsame colours")
for a in ax.flat:
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "horseshoes_compared.png"), dpi=110)
print("plot saved")
