#!/usr/bin/env python3
"""Stage 17 (EXPLORATORY bounding of PREREG 5 headlines).
(a) T2 SHEET vs OPENING robustness: facing pairs that are not also conjugate; pages >= 60 words; NCD as response.
(b) T1b: excess sheet SS over its permutation null, as share of total.
(c) T4: fairer competing models: section-pooled profiles (S) and section x Currier pooled profiles (SC) + multinomial noise.
"""
import json, os, pickle, sys
from itertools import combinations
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/horseshoe/scripts")
import lib_methods as L

ROOT = sys.argv[1]; D = os.path.join(ROOT, "data"); rng = np.random.default_rng(20261012)
S = pickle.load(open(os.path.join(D, "s14_features.pkl"), "rb"))
F, M, wmask, facing = S["F"], S["M"], S["wmask"], S["facing"]
n = len(F); iu = np.triu_indices(n, 1)
Dtext = L.js_matrix(L.profiles(M, wmask))
Dz = pickle.load(open(os.path.join(D, "s15_ncd_zlib.pkl"), "rb"))
nulls = pickle.load(open(os.path.join(D, "s15_leafperms.pkl"), "rb"))
t1b = pickle.load(open(os.path.join(D, "s15_t1b.pkl"), "rb"))
sheet, leaf_id = F.sheet.values, F.leaf_id.values; leafnum = F.leaf.astype(str).values
res = {}

# (b)
tot = t1b["table"]["total"][0]
res["T1b_excess_sheet_share_of_total"] = float((t1b["SS_sheet"] - t1b["null_mean"]) / tot)
res["T1b_excess_in_null_SD_units"] = float(t1b["z"])

# (a)
def X_for(keep_pages, facing_mode):
    a, b = iu; m = keep_pages[a] & keep_pages[b]
    conj = (sheet[a] == sheet[b]) & (leafnum[a] != leafnum[b])
    fac = facing[a, b] & (~conj if facing_mode == "not_conjugate" else True)
    cols = [conj, leaf_id[a] == leaf_id[b], fac,
            np.log1p(np.abs(F.book_pos.values[a] - F.book_pos.values[b])),
            F.quire_l.values[a] == F.quire_l.values[b], F.scribe.values[a] == F.scribe.values[b],
            F.csec.values[a] == F.csec.values[b], F.currier.values[a] == F.currier.values[b],
            np.abs(np.log(F.n_words.values[a]) - np.log(F.n_words.values[b])),
            np.log(np.minimum(F.n_words.values[a], F.n_words.values[b]))]
    return np.c_[[c.astype(float) for c in cols]].T, m, conj, fac

def sheet_vs_opening(Dm, keep_pages, facing_mode, tag):
    X, m, conj, fac = X_for(keep_pages, facing_mode)
    y = Dm[iu]
    b, _ = L.mrm_fit(y[m], X[m])
    jk = []
    for s in np.unique(sheet):
        mm = m & (sheet[iu[0]] != s) & (sheet[iu[1]] != s)
        bj, _ = L.mrm_fit(y[mm], X[mm]); jk.append((bj[0], bj[2]))
    jk = np.array(jk); k = len(jk); se = np.sqrt((k - 1) / k * ((jk - jk.mean(0)) ** 2).sum(0))
    ci_c = [b[0] - 1.96 * se[0], b[0] + 1.96 * se[0]]; ci_f = [b[2] - 1.96 * se[1], b[2] + 1.96 * se[1]]
    no_overlap = ci_c[1] < ci_f[0] or ci_f[1] < ci_c[0]
    call = "SHEET > OPENING" if (abs(b[0]) > abs(b[2]) and b[0] < 0 and no_overlap) else \
           ("OPENING > SHEET" if (abs(b[2]) > abs(b[0]) and b[2] < 0 and no_overlap) else "no call")
    return dict(tag=tag, n_conj=int((conj & m).sum()), n_facing=int((fac & m).sum()),
                beta_conj=float(b[0]), ci_conj=[float(x) for x in ci_c],
                beta_facing=float(b[2]), ci_facing=[float(x) for x in ci_f], call=call)

allp = np.ones(n, bool); big = F.n_words.values >= 60
res["T2_bound"] = [sheet_vs_opening(Dtext, allp, "all", "registered spec (reproduce)"),
                   sheet_vs_opening(Dtext, allp, "not_conjugate", "facing excl. conjugate"),
                   sheet_vs_opening(Dtext, big, "not_conjugate", "pages >=60 words, facing excl. conjugate"),
                   sheet_vs_opening(Dz, allp, "not_conjugate", "NCD response, facing excl. conjugate")]
for r in res["T2_bound"]: print(r)

# (c) T4 fairer models
sh_info = pickle.load(open(os.path.join(D, "s16_sheet.pkl"), "rb"))
sheets = sh_info["sheets"]; Ds = sh_info["Ds"]
Ms = np.array([M[sheet == s].sum(0) for s in sheets])
quart = np.array(list(combinations(range(len(sheets)), 4)))
def metrics(Dm):
    _, share = L.cmds(Dm, 2)
    return dict(delta=L.delta_mean(Dm, quart), mds1=float(share[0]), mds2=float(share[1]))
real = metrics(Ds)
def modal(col):
    return np.array([pd.Series(F[col].values[sheet == s]).mode().iat[0] for s in sheets])
groups = {"S_section": modal("csec"),
          "SC_section_x_currier": np.char.add(np.char.add(modal("csec").astype(str), "|"), modal("currier").astype(str))}
blocks = [wmask, ~wmask]
for name, g in groups.items():
    pools = {}
    for lab in np.unique(g):
        pools[lab] = [Ms[g == lab][:, b].sum(0) / Ms[g == lab][:, b].sum() for b in blocks]
    sims = []
    for _ in range(200):
        Msim = np.zeros_like(Ms)
        for bi, b in enumerate(blocks):
            for i in range(len(sheets)):
                p = pools[g[i]][bi]
                Msim[i, b] = rng.multinomial(int(Ms[i, b].sum()), p / p.sum())
        sims.append(metrics(L.js_matrix(L.profiles(Msim, wmask))))
    sims = pd.DataFrame(sims)
    out = {"n_groups": int(len(pools))}
    for m in ("delta", "mds1", "mds2"):
        mu, sd = sims[m].mean(), sims[m].std(); lo, hi = sims[m].quantile([0.025, 0.975])
        out[m] = dict(real=real[m], sim_mean=float(mu), sim_sd=float(sd), z=float((real[m] - mu) / sd),
                      inside=bool(lo <= real[m] <= hi))
    res[f"T4_{name}"] = out
    print(name, json.dumps(out, default=float))
json.dump(res, open(os.path.join(ROOT, "results_bounding5.json"), "w"), indent=1, default=float)
