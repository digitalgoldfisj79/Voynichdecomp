#!/usr/bin/env python3
"""PREREG 12: establish the within-group gradient (split-leaf reliability), identify it, and give a stability-ranked order."""
import json, os, pickle, sys
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "/home/claude/horseshoe/scripts"); import lib_methods as L
ROOT = sys.argv[1]; D = os.path.join(ROOT, "data"); rng = np.random.default_rng(20261019)
U = pickle.load(open(os.path.join(D, "s22_units.pkl"), "rb")); N = {"w": 569, "l": 451}
OUT = os.path.join(ROOT, "results_gradient.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(res, open(OUT, "w"), indent=1, default=float)

def feats(words, linelen=8):
    cc = Counter()
    for k in range(0, len(words), linelen):
        line = words[k:k + linelen]
        for w in line:
            st = "^" + w + "$"; cc.update(("w", a, b) for a, b in zip(st, st[1:]))
        st = "".join(line); cc.update(("l", a, b) for a, b in zip(st, st[1:]))
    return cc

def setup(u):
    keys = u["sheets"]; vocab = sorted(set().union(*[u["half"][(s, h)] for s in keys for h in (0, 1)]))
    vi = {v: i for i, v in enumerate(vocab)}; wmask = np.array([v[0] == "w" for v in vocab])
    arrs = {(s, h): {b: np.repeat([vi[k] for k in u["half"][(s, h)] if k[0] == b],
                                  [u["half"][(s, h)][k] for k in u["half"][(s, h)] if k[0] == b]) for b in "wl"}
            for s in keys for h in (0, 1)}
    return keys, vocab, wmask, arrs

def draw_X(keys, vocab, arrs, hs, sel):
    X = np.zeros((len(sel), len(vocab)))
    for r, i in enumerate(sel):
        for h in hs:
            for b in "wl": np.add.at(X[r], rng.choice(arrs[(keys[i], h)][b], N[b], replace=False), 1)
    return X

def mats(u, sel, draws=5):
    keys, vocab, wmask, arrs = setup(u); out = []
    for _ in range(draws):
        out.append([L.js_matrix(L.profiles(draw_X(keys, vocab, arrs, hs, sel), wmask)) for hs in ([0], [1], [0, 1])])
    return [np.mean([o[k] for o in out], 0) for k in range(3)]

def g1(DA, DB, nperm=10000):
    cA = L.spectral_order(DA)
    f = lambda Db: abs(spearmanr(cA, L.spectral_order(Db))[0])
    obs = f(DB); n = DB.shape[0]
    nl = np.array([f(DB[np.ix_(p, p)]) for p in (rng.permutation(n) for _ in range(nperm))])
    return dict(n=int(n), obs=obs, null_mean=nl.mean(), null_sd=nl.std(), z=(obs - nl.mean()) / nl.std(),
                p=(np.sum(nl >= obs) + 1) / (nperm + 1))

vms = U["vms"]; strat = vms["strat"]; strat2 = vms["strat2"]
groups = {"1|A|Herbal-A": np.where(strat == "1.0|A|Herbal-A")[0]}
for g in np.unique(strat2):
    idx = np.where(strat2 == g)[0]
    if len(idx) >= 7: groups[g] = idx
for g, idx in groups.items():
    k = f"G1_VMS_{g}"
    if k in res: continue
    DA, DB, _ = mats(vms, idx); res[k] = g1(DA, DB); save(); print(k, res[k])
for name, c in U["controls"].items():
    big = max(np.unique(c["strat"]), key=lambda x: np.sum(c["strat"] == x)); idx = np.where(c["strat"] == big)[0]
    if f"G1_{name}" not in res:
        DA, DB, _ = mats(c, idx); res[f"G1_{name}"] = g1(DA, DB, 5000); save(); print(name, res[f"G1_{name}"])
    if f"G1_SHUF_{name}" not in res:
        keys = c["sheets"]; allw = [w for s in keys for h in (0, 1) for w in c["words"][(s, h)]]; rng.shuffle(allw)
        half, pos = {}, 0
        for s in keys:
            for h in (0, 1):
                nw = len(c["words"][(s, h)]); half[(s, h)] = feats(allw[pos:pos + nw]); pos += nw
        DA, DB, _ = mats(dict(sheets=keys, half=half), idx)
        res[f"G1_SHUF_{name}"] = g1(DA, DB, 5000); save(); print("SHUF", name, res[f"G1_SHUF_{name}"])

# ---- G2 identity + G3 stability on the primary group
idx = groups["1|A|Herbal-A"]; names = [vms["sheets"][i] for i in idx]
if "G3" not in res:
    keys, vocab, wmask, arrs = setup(vms); ranks = []; ref = None
    for _ in range(200):
        c = L.spectral_order(L.js_matrix(L.profiles(draw_X(keys, vocab, arrs, [0, 1], idx), wmask)))
        if ref is None: ref = c
        if spearmanr(c, ref)[0] < 0: c = -c
        ranks.append(np.argsort(np.argsort(c)))
    ranks = np.array(ranks); mr = ranks.mean(0); sd = ranks.std(0); o = np.argsort(mr)
    res["G3"] = dict(order=[dict(sheet=names[i], mean_rank=round(float(mr[i]), 2), rank_sd=round(float(sd[i]), 2)) for i in o],
                     median_rank_sd=float(np.median(sd)), coord_mean_rank=mr.tolist()); save()
mr = np.array(res["G3"]["coord_mean_rank"])
if "G2" not in res:
    S = pickle.load(open(os.path.join(D, "s14_features.pkl"), "rb")); F, M, wm = S["F"], S["M"], S["wmask"]
    P = L.profiles(M, wm); ing = F.sheet.isin(names).values
    dirv = P[(F.currier.values == "B") & ~ing].mean(0) - P[(F.currier.values == "A") & ~ing].mean(0)
    def sheet_mean(col): return np.array([F[F.sheet == s][col].astype(float).mean() for s in names])
    score = np.array([P[(F.sheet == s).values].mean(0) @ dirv for s in names])
    def test(x):
        ok = ~np.isnan(x); r = abs(spearmanr(mr[ok], x[ok])[0])
        nl = np.array([abs(spearmanr(mr[ok], rng.permutation(x[ok]))[0]) for _ in range(10000)])
        return dict(n=int(ok.sum()), abs_rho=r, null_mean=nl.mean(), null_sd=nl.std(), z=(r - nl.mean()) / nl.std(),
                    signed_rho=spearmanr(mr[ok], x[ok])[0])
    res["G2"] = dict(a_currier_axis=test(score), b_eo777_path=test(sheet_mean("path")), c_book_pos=test(sheet_mean("book_pos")))
    save()
print(json.dumps({k: res[k] for k in ("G2",)}, indent=1, default=float))
print("G3 median rank SD", res["G3"]["median_rank_sd"]); print([ (d["sheet"], d["mean_rank"], d["rank_sd"]) for d in res["G3"]["order"]])
