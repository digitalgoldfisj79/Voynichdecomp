#!/usr/bin/env python3
"""Stage 15 (PREREG 5): T1b PERMANOVA, T2 MRM, T5 NCD. Each sub-result pickled to data/s15_*.pkl."""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_methods as L

ROOT = sys.argv[1]; D = os.path.join(ROOT, "data"); rng = np.random.default_rng(20261012)
S = pickle.load(open(os.path.join(D, "s14_features.pkl"), "rb"))
F, M, wmask, strings, facing = S["F"], S["M"], S["wmask"], S["strings"], S["facing"]
n = len(F); iu = np.triu_indices(n, 1)
P = L.profiles(M, wmask)
Dtext = L.js_matrix(P)
sheet, leaf_id, stratum = F.sheet.values, F.leaf_id.values, F.stratum.values
leafnum = F.leaf.astype(str).values
NPERM = 2000


def stage(name, fn):
    p = os.path.join(D, f"s15_{name}.pkl")
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    r = fn(); pickle.dump(r, open(p + ".tmp", "wb")); os.replace(p + ".tmp", p); return r


nulls = stage("leafperms", lambda: [L.leaf_permute(sheet, leaf_id, stratum, rng) for _ in range(NPERM)])
res = {}

# ---------------- T1b PERMANOVA
def t1b():
    G = L.gower(Dtext)
    base = [("currier", F.currier.values), ("scribe", F.scribe.astype(str).values),
            ("section", F.csec.values), ("quire", F.quire_l.values)]
    o = L.permanova_seq(G, base + [("sheet", sheet), ("leaf", leaf_id)])
    obs = o["sheet"][0]
    null = np.array([L.permanova_seq(G, base + [("sheet", s)])["sheet"][0] for s in nulls])
    return dict(table={k: [round(v[0], 5), v[1]] for k, v in o.items()},
                SS_sheet=obs, null_mean=float(null.mean()), null_sd=float(null.std()),
                z=float((obs - null.mean()) / null.std()), p=float((np.sum(null >= obs) + 1) / (NPERM + 1)),
                partial_R2_sheet=float(obs / (obs + o["leaf"][0] + o["resid"][0])),
                R2_sheet_of_total=float(obs / o["total"][0]),
                df_sheet=o["sheet"][1], df_leaf=o["leaf"][1], df_resid=o["resid"][1])
res["T1b"] = stage("t1b", t1b)
print("T1b", {k: v for k, v in res["T1b"].items() if k != "table"})


# ---------------- T2 MRM
def pair_X(sh, keep=None):
    a, b = iu
    conj = (sh[a] == sh[b]) & (leafnum[a] != leafnum[b]) & (sh[a] == sh[a])
    same_leaf = leaf_id[a] == leaf_id[b]
    fac = facing[a, b]
    gap = np.log1p(np.abs(F.book_pos.values[a] - F.book_pos.values[b]))
    cols = dict(conjugate=conj, same_leaf=same_leaf, facing=fac, log_order_gap=gap,
                same_quire=F.quire_l.values[a] == F.quire_l.values[b],
                same_scribe=F.scribe.astype(str).values[a] == F.scribe.astype(str).values[b],
                same_section=F.csec.values[a] == F.csec.values[b],
                same_currier=F.currier.values[a] == F.currier.values[b],
                d_log_words=np.abs(np.log(F.n_words.values[a]) - np.log(F.n_words.values[b])),
                log_min_words=np.log(np.minimum(F.n_words.values[a], F.n_words.values[b])))
    names = list(cols); X = np.c_[[cols[k].astype(float) for k in names]].T
    return X, names


def t2():
    y = Dtext[iu]
    X, names = pair_X(sheet)
    b, r2 = L.mrm_fit(y, X)
    ci = names.index("conjugate"); fi = names.index("facing")
    null = np.array([L.mrm_fit(y, pair_X(s)[0])[0][ci] for s in nulls])
    # unique R2 (drop-one)
    uniq = {}
    for k in range(len(names)):
        _, r2k = L.mrm_fit(y, np.delete(X, k, 1)); uniq[names[k]] = float(r2 - r2k)
    # delete-one-sheet jackknife for conjugate and facing betas
    jk = []
    for s in np.unique(sheet):
        keep = sheet != s
        idx = np.where(keep)[0]; pos = {v: i for i, v in enumerate(idx)}
        m_ = keep[iu[0]] & keep[iu[1]]
        bj, _ = L.mrm_fit(y[m_], X[m_])
        jk.append((bj[ci], bj[fi]))
    jk = np.array(jk); k = len(jk)
    se = np.sqrt((k - 1) / k * ((jk - jk.mean(0)) ** 2).sum(0))
    return dict(names=names, betas=dict(zip(names, map(float, b))), R2=float(r2), unique_R2=uniq,
                conj_beta=float(b[ci]), conj_null_mean=float(null.mean()), conj_null_sd=float(null.std()),
                conj_z=float((null.mean() - b[ci]) / null.std()),
                conj_p=float((np.sum(null <= b[ci]) + 1) / (NPERM + 1)),
                n_conjugate_pairs=int(X[:, ci].sum()), n_facing_pairs=int(X[:, fi].sum()),
                jk_se_conj=float(se[0]), jk_se_facing=float(se[1]),
                ci_conj=[float(b[ci] - 1.96 * se[0]), float(b[ci] + 1.96 * se[0])],
                ci_facing=[float(b[fi] - 1.96 * se[1]), float(b[fi] + 1.96 * se[1])])
res["T2"] = stage("t2", t2)
print("T2", {k: v for k, v in res["T2"].items() if k not in ("names",)})


# ---------------- T5 NCD
def coherence(Dm, nulls_):
    d = Dm[iu]
    def mean_conj(sh):
        m = (sh[iu[0]] == sh[iu[1]]) & (leafnum[iu[0]] != leafnum[iu[1]])
        return d[m].mean()
    obs = mean_conj(sheet); null = np.array([mean_conj(s) for s in nulls_])
    return dict(obs=float(obs), null_mean=float(null.mean()), null_sd=float(null.std()),
                z=float((null.mean() - obs) / null.std()), pct_lower=float(100 * (null.mean() - obs) / null.mean()))


Dz = stage("ncd_zlib", lambda: L.ncd_matrix(strings, "zlib"))
Db = stage("ncd_bz2", lambda: L.ncd_matrix(strings, "bz2"))
def ncd60():
    mats = []
    words = [s.split() for s in strings]
    ok = np.array([len(w) >= 60 for w in words])
    for rep in range(3):
        sub = []
        for w in words:
            if len(w) < 60: sub.append(" ".join(w)); continue
            st = rng.integers(0, len(w) - 60 + 1); sub.append(" ".join(w[st:st + 60]))
        mats.append(L.ncd_matrix(sub, "zlib"))
    return np.mean(mats, 0), ok
D60, ok60 = stage("ncd60", ncd60)
res["T5_zlib"] = coherence(Dz, nulls)
res["T5_bz2"] = coherence(Db, nulls)
D60m = D60.copy(); bad = ~ok60
D60m[bad, :] = np.nan; D60m[:, bad] = np.nan
def coh_nan(Dm):
    d = Dm[iu]; v = ~np.isnan(d)
    def mc(sh):
        m = (sh[iu[0]] == sh[iu[1]]) & (leafnum[iu[0]] != leafnum[iu[1]]) & v
        return d[m].mean()
    obs = mc(sheet); null = np.array([mc(s) for s in nulls])
    return dict(n_pages=int(ok60.sum()), obs=float(obs), null_mean=float(null.mean()), null_sd=float(null.std()),
                z=float((null.mean() - obs) / null.std()), pct_lower=float(100 * (null.mean() - obs) / null.mean()))
res["T5_zlib_60words"] = coh_nan(D60m)
res["T5_text_JS_same_leaf_null_for_reference"] = coherence(Dtext, nulls)
for k in ("T5_zlib", "T5_bz2", "T5_zlib_60words", "T5_text_JS_same_leaf_null_for_reference"): print(k, res[k])
json.dump(res, open(os.path.join(ROOT, "results_t1b_t2_t5.json"), "w"), indent=1, default=float)
