#!/usr/bin/env python3
"""Stage 16 (PREREG 5): T3 CA/DCA + spectral seriation; T4 tree-likeness vs single A->B gradient model."""
import json, os, pickle, sys
from itertools import combinations
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, "/home/claude/horseshoe/scripts")
import lib_methods as L

ROOT = sys.argv[1]; D = os.path.join(ROOT, "data"); rng = np.random.default_rng(20261012)
S = pickle.load(open(os.path.join(D, "s14_features.pkl"), "rb"))
F, M, wmask = S["F"], S["M"], S["wmask"]
res = {}

def eta2(x, g):
    s = pd.Series(x); m = s.groupby(np.asarray(g)).transform("mean")
    return float(m.var() / s.var())

# ---------------- T3 CA / DCA on pages >= 30 words
keep = F.n_words.values >= 30
N = M[keep]; N = N[:, N.sum(0) > 0]
rows, _, eig = L.ca(N)
sc, lam, glen = L.dca(N)
sec = F.csec.values[keep]
res["T3"] = dict(
    n_pages=int(keep.sum()), n_features=int(N.shape[1]),
    ca_eig=[float(e) for e in eig[:3]], ca_arch_index=float(L.quad_r2(rows[:, 0], rows[:, 1])),
    dca_lam=[float(x) for x in lam], dca_lam_ratio=float(lam[1] / lam[0]),
    dca_arch_index=float(L.quad_r2(sc[:, 0], sc[:, 1])), dca_axis1_gradient_length_SD=glen,
    dca_axis2_section_eta2=eta2(sc[:, 1], sec), dca_axis1_section_eta2=eta2(sc[:, 0], sec),
    dca_axis2_scribe_eta2=eta2(sc[:, 1], F.scribe.values[keep]),
    dca_axis1_currier_eta2=eta2(sc[:, 0], F.currier.values[keep]),
    dca_axis1_vs_text_mds1_rho=float(abs(spearmanr(sc[:, 0], F.txt_d1.values[keep])[0])))
r = res["T3"]
if r["dca_lam_ratio"] < 0.25 and r["dca_axis2_section_eta2"] < 0.10: v = "ONE-DIMENSIONAL"
elif r["dca_lam_ratio"] >= 0.40 or r["dca_axis2_section_eta2"] >= 0.30: v = "TWO+ DIMENSIONS"
else: v = "NOT RESOLVED"
r["verdict"] = v

# sheet-level aggregation (shared by seriation and T4)
sheets = sorted(F.sheet.unique())
Ms = np.array([M[F.sheet.values == s].sum(0) for s in sheets])
words_s = np.array([F.n_words.values[F.sheet.values == s].sum() for s in sheets])
ok = words_s >= 100
sheets = [s for s, o in zip(sheets, ok) if o]; Ms = Ms[ok]; words_s = words_s[ok]
Ps = L.profiles(Ms, wmask); Ds = L.js_matrix(Ps)
fied = L.spectral_order(Ds)
path_s = np.array([F.path.values[F.sheet.values == s].mean() for s in sheets])
book_s = np.array([F.book_pos.values[F.sheet.values == s].min() for s in sheets])
okp = ~np.isnan(path_s)
r["seriation_n_sheets"] = len(sheets)
r["seriation_rho_vs_eo777_path"] = float(spearmanr(fied[okp], path_s[okp])[0])
r["seriation_rho_vs_book_order"] = float(spearmanr(fied, book_s)[0])
print("T3", r)

# ---------------- T4 tree-likeness vs gradient model
quart = np.array(list(combinations(range(len(sheets)), 4)))
def metrics(Dm):
    _, share = L.cmds(Dm, 2)
    return dict(delta=L.delta_mean(Dm, quart), mds1=float(share[0]), mds2=float(share[1]))
real = metrics(Ds)
Y, _ = L.cmds(Ds, 1); t_raw = Y[:, 0]
# orient so that high t = Currier B side
curr_b = np.array([(F.currier.values[F.sheet.values == s] == "B").mean() for s in sheets])
if np.corrcoef(t_raw, curr_b)[0, 1] < 0: t_raw = -t_raw
t = (pd.Series(t_raw).rank().values - 1) / (len(t_raw) - 1)
k = max(1, int(round(0.10 * len(sheets))))
lo, hi = np.argsort(t)[:k], np.argsort(t)[-k:]
blocks = [wmask, ~wmask]
pA = [Ms[lo][:, b].sum(0) / Ms[lo][:, b].sum() for b in blocks]
pB = [Ms[hi][:, b].sum(0) / Ms[hi][:, b].sum() for b in blocks]
tot = [Ms[:, b].sum(1).astype(int) for b in blocks]
sims = []
for _ in range(200):
    Msim = np.zeros_like(Ms)
    for bi, b in enumerate(blocks):
        for i in range(len(sheets)):
            p = (1 - t[i]) * pA[bi] + t[i] * pB[bi]
            Msim[i, b] = rng.multinomial(tot[bi][i], p / p.sum())
    sims.append(metrics(L.js_matrix(L.profiles(Msim, wmask))))
sims = pd.DataFrame(sims)
out = {"n_sheets": len(sheets), "n_quartets": int(len(quart)), "real": real}
verdict_flags = []
for m in ("delta", "mds1", "mds2"):
    lo_, hi_ = sims[m].quantile([0.025, 0.975]); sd = sims[m].std(); mu = sims[m].mean()
    z = (real[m] - mu) / sd
    out[m] = dict(real=real[m], sim_mean=float(mu), sim_sd=float(sd), env=[float(lo_), float(hi_)], z=float(z),
                  inside=bool(lo_ <= real[m] <= hi_))
    verdict_flags.append(abs(z) > 2)
if all(out[m]["inside"] for m in ("delta", "mds1", "mds2")): out["verdict"] = "GRADIENT SUFFICIENT"
elif any(verdict_flags): out["verdict"] = "NOT SUFFICIENT"
else: out["verdict"] = "NOT RESOLVED"
res["T4"] = out
print("T4", json.dumps(out, indent=1, default=float))
json.dump(res, open(os.path.join(ROOT, "results_t3_t4.json"), "w"), indent=1, default=float)
pickle.dump(dict(sheets=sheets, Ds=Ds, t=t, fied=fied), open(os.path.join(D, "s16_sheet.pkl"), "wb"))
