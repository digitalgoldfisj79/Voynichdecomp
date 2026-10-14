#!/usr/bin/env python3
"""PREREG 14: calibrate TempSeq-like (M1) and incidence/CONOP-like (M2) ordering against Fiedler (B0) and path (B1).
Per-unit results pickled to data/s28_<unit>.pkl (atomic); summary -> results_order_methods.json
usage: python3 -I s28_order_methods.py ROOT
"""
import json, os, pickle, sys
from collections import Counter
import numpy as np
from scipy.stats import kendalltau
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import lib_methods as L
ROOT = sys.argv[1]; D = os.path.join(ROOT, "data"); rng = np.random.default_rng(20261021)
U = pickle.load(open(os.path.join(D, "s22_units.pkl"), "rb")); NG = {"w": 569, "l": 451}; DRAWS = 3

def feats(words, linelen=8):
    cc = Counter()
    for k in range(0, len(words), linelen):
        line = words[k:k + linelen]
        for w in line:
            st = "^" + w + "$"; cc.update(("w", a, b) for a, b in zip(st, st[1:]))
        st = "".join(line); cc.update(("l", a, b) for a, b in zip(st, st[1:]))
    return cc

units = {"VMS": U["vms"]}
for name, c in U["controls"].items():
    units[name] = c
    keys = c["sheets"]; allw = [w for s in keys for h in (0, 1) for w in c["words"][(s, h)]]; rng.shuffle(allw)
    half, words, pos = {}, {}, 0
    for s in keys:
        for h in (0, 1):
            nw = len(c["words"][(s, h)]); ws = allw[pos:pos + nw]; pos += nw; half[(s, h)] = feats(ws); words[(s, h)] = ws
    units["SHUF_" + name] = dict(sheets=keys, half=half, words=words, strat=c["strat"])
NW_SHEET = min(len(u["words"][(s, 0)]) + len(u["words"][(s, 1)]) for u in units.values() for s in u["sheets"])
NW_HALF = min(len(u["words"][(s, h)]) for u in units.values() for s in u["sheets"] for h in (0, 1))

# ---------------- feature draws
def glyph_X(u, hs, sel):
    keys = u["sheets"]; vocab = sorted(set().union(*[u["half"][(s, h)] for s in keys for h in (0, 1)]))
    vi = {v: i for i, v in enumerate(vocab)}; wmask = np.array([v[0] == "w" for v in vocab])
    X = np.zeros((len(sel), len(vocab)))
    for r, i in enumerate(sel):
        for h in hs:
            c = u["half"][(keys[i], h)]
            for b in "wl":
                arr = np.repeat([vi[k] for k in c if k[0] == b], [c[k] for k in c if k[0] == b])
                np.add.at(X[r], rng.choice(arr, NG[b], replace=False), 1)
    return X, wmask

def word_P(u, hs, sel, N):
    keys = u["sheets"]; docs = []
    for i in sel:
        ws = [w for h in hs for w in u["words"][(keys[i], h)]]
        docs.append(set(rng.choice(np.array(ws, dtype=object), N, replace=False)))
    vocab = sorted(set().union(*docs)); vi = {v: k for k, v in enumerate(vocab)}
    P = np.zeros((len(sel), len(vocab)), bool)
    for r, d in enumerate(docs):
        for w in d: P[r, vi[w]] = True
    k = P.sum(0); keep = (k >= 2) & (k <= 0.5 * len(sel))
    return P[:, keep]

# ---------------- objectives + SA (segment reversal moves on an open path)
def sa(n, score_fn, delta_fn=None, iters=20000, init=None, T0=None):
    o = list(init) if init is not None else list(rng.permutation(n)); cur = score_fn(o); best, bestv = list(o), cur
    T = T0 if T0 else max(1e-6, abs(cur) * 0.02)
    for it in range(iters):
        i, j = sorted(rng.choice(n, 2, replace=False))
        new = o[:i] + o[i:j + 1][::-1] + o[j + 1:]
        v = score_fn(new)
        t = T * (1 - it / iters) + 1e-9
        if v >= cur or rng.random() < np.exp((v - cur) / t):
            o, cur = new, v
            if v > bestv: best, bestv = list(o), v
    return best, bestv

def m1_order(X, restarts, init):
    F = X / X.sum(1, keepdims=True)
    keep = np.argsort(-X.sum(0))[:300]; Z = F[:, keep]; Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1)
    def score(o):
        a = Z[o[:-1]] * Z[o[1:]]; return float(np.median(a.mean(0)))
    res = [sa(len(X), score, iters=ITERS[len(X)], init=init)] + [sa(len(X), score, iters=ITERS[len(X)]) for _ in range(restarts - 1)]
    return max(res, key=lambda r: r[1])[0]

def m2_order(P, restarts, init):
    n = P.shape[0]; cnt = P.sum(0)
    def score(o):
        pos = np.empty(n); pos[o] = np.arange(n)
        pm = np.where(P, pos[:, None], np.nan)
        rngs = np.nanmax(pm, 0) - np.nanmin(pm, 0) + 1
        return -float((rngs - cnt).sum())
    res = [sa(n, score, iters=ITERS[n], init=init)] + [sa(n, score, iters=ITERS[n]) for _ in range(restarts - 1)]
    return max(res, key=lambda r: r[1])[0]

ITERS = {49: 12000, 21: 6000}
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s23_bound_chain.py")).read()
exec(src.split("OUT = os.path.join")[0].split("U = pickle.load")[1].split("\n", 1)[1])  # best_path etc.

def orders(Xg, wmask, Pw):
    Dm = L.js_matrix(L.profiles(Xg, wmask))
    b0 = list(np.argsort(L.spectral_order(Dm)))
    return dict(B0=b0, B1=best_path(Dm), M1=m1_order(Xg, 3, b0), M2=m2_order(Pw, 3, b0))

def truth_metrics(order, truth):
    o = np.array(order); t = np.asarray(truth)
    adj = np.mean([abs(t[o[k]] - t[o[k + 1]]) == 1 for k in range(len(o) - 1)])
    tau = abs(kendalltau(np.argsort(o), t)[0])
    return float(adj), float(tau)

def run_unit(name, u):
    p = os.path.join(D, f"s28_{name}.pkl")
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    n = len(u["sheets"]); strat = u["strat"]
    g = max(np.unique(strat), key=lambda x: np.sum(strat == x)); blk = np.where(strat == g)[0]
    out = dict(draws=[], reliability=None)
    if name != "VMS":
        for d in range(DRAWS):
            rec = {}
            for scope, sel in (("all", np.arange(n)), ("block", blk)):
                Xg, wm = glyph_X(u, [0, 1], sel); Pw = word_P(u, [0, 1], sel, NW_SHEET)
                od = orders(Xg, wm, Pw); truth = np.argsort(np.argsort(sel))
                rec[scope] = {m: truth_metrics(o, truth) for m, o in od.items()}
            out["draws"].append(rec); print(name, d, rec, flush=True)
    # split-leaf reliability in the block (all units)
    rel = {}
    oA = orders(*glyph_X(u, [0], blk), word_P(u, [0], blk, NW_HALF))
    oB = orders(*glyph_X(u, [1], blk), word_P(u, [1], blk, NW_HALF))
    for m in oA:
        rA = np.argsort(oA[m]); rB = np.argsort(oB[m]); obs = abs(kendalltau(rA, rB)[0])
        nl = np.array([abs(kendalltau(rA, rng.permutation(len(rA)))[0]) for _ in range(10000)])
        rel[m] = dict(obs=float(obs), null_mean=float(nl.mean()), null_sd=float(nl.std()), z=float((obs - nl.mean()) / nl.std()))
    out["reliability"] = rel; print(name, "reliability", {m: round(v["z"], 2) for m, v in rel.items()}, flush=True)
    pickle.dump(out, open(p + ".tmp", "wb")); os.replace(p + ".tmp", p)
    return out

print("NW_SHEET", NW_SHEET, "NW_HALF", NW_HALF, flush=True)
allres = {n_: run_unit(n_, u) for n_, u in units.items()}
summ = {}
for n_, r in allres.items():
    s = {"reliability_z": {m: round(v["z"], 2) for m, v in r["reliability"].items()},
         "reliability_tau": {m: round(v["obs"], 3) for m, v in r["reliability"].items()}}
    if r["draws"]:
        for scope in ("all", "block"):
            s[scope] = {m: dict(adj=round(float(np.mean([d[scope][m][0] for d in r["draws"]])), 3),
                                tau=round(float(np.mean([d[scope][m][1] for d in r["draws"]])), 3)) for m in ("B0", "B1", "M1", "M2")}
    summ[n_] = s
json.dump(summ, open(os.path.join(ROOT, "results_order_methods.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
