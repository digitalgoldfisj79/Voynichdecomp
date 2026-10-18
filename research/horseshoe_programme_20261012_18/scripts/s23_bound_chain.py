#!/usr/bin/env python3
"""PREREG 10: bounding of PREREG 9 + order-sensitive cross-validated adjacency (CVA). Uses data/s22_units.pkl."""
import json, os, pickle, sys
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "/home/claude/horseshoe/scripts"); import lib_methods as L
import importlib.util
ROOT = sys.argv[1]; D = os.path.join(ROOT, "data"); rng = np.random.default_rng(20261015)
U = pickle.load(open(os.path.join(D, "s22_units.pkl"), "rb"))

def char_mats(unit, block=None, shuffle_words=False):
    keys = unit["sheets"]; half = unit["half"]
    if block is not None:
        vocab = sorted({k for c in half.values() for k in c if k[0] == block})
    else:
        vocab = sorted(set().union(*half.values()))
    vi = {v: i for i, v in enumerate(vocab)}; wmask = np.array([v[0] == "w" for v in vocab])
    def M(hs):
        X = np.zeros((len(keys), len(vocab)))
        for r, s in enumerate(keys):
            for h in hs:
                for k, v in half[(s, h)].items():
                    if k in vi: X[r, vi[k]] += v
        if block is not None:
            P = X / X.sum(1, keepdims=True) + 1e-9; P = P / P.sum(1, keepdims=True); return L.js_matrix(P)
        return L.js_matrix(L.profiles(X, wmask))
    return M([0]), M([1]), M([0, 1])

def within_perm(strat):
    p = np.arange(len(strat))
    for g in np.unique(strat):
        idx = np.where(strat == g)[0]; p[idx] = rng.permutation(idx)
    return p

def tw_test(DA, DB, strat, pairmask=None, nperm=5000):
    n = len(strat); iu = np.triu_indices(n, 1); m = strat[iu[0]] == strat[iu[1]]
    if pairmask is not None: m = m & pairmask[iu]
    f = lambda X: np.corrcoef(DA[iu][m], X[iu][m])[0, 1]
    obs = f(DB); nl = np.array([f(DB[np.ix_(p, p)]) for p in (within_perm(strat) for _ in range(nperm))])
    return dict(obs=round(float(obs), 4), null_mean=round(float(nl.mean()), 4), null_sd=round(float(nl.std()), 4),
                z=round(float((obs - nl.mean()) / nl.std()), 2), n_pairs=int(m.sum()))

def resid_size(Dm, n_words):
    n = len(n_words); iu = np.triu_indices(n, 1); lw = np.log(n_words)
    X = np.c_[np.ones(len(iu[0])), lw[iu[0]] + lw[iu[1]], np.abs(lw[iu[0]] - lw[iu[1]]), np.minimum(lw[iu[0]], lw[iu[1]])]
    y = Dm[iu]; b = np.linalg.lstsq(X, y, rcond=None)[0]; R = np.zeros_like(Dm); R[iu] = y - X @ b; return R + R.T

def best_path(Dm, starts=20):
    n = Dm.shape[0]; best, bestL = None, np.inf
    for s in rng.choice(n, size=min(starts, n), replace=False):
        path = [s]; left = set(range(n)) - {s}
        while left:
            j = min(left, key=lambda k: Dm[path[-1], k]); path.append(j); left.remove(j)
        improved = True
        while improved:
            improved = False
            for i in range(0, n - 2):
                for j in range(i + 2, n):
                    a, b = path[i], path[i + 1]; c = path[j]; d = path[j + 1] if j + 1 < n else None
                    old = Dm[a, b] + (Dm[c, d] if d is not None else 0)
                    new = Dm[a, c] + (Dm[b, d] if d is not None else 0)
                    if new < old - 1e-12: path[i + 1:j + 1] = path[i + 1:j + 1][::-1]; improved = True
            # also allow reversing a prefix (open path start)
            for j in range(1, n):
                if Dm[path[0], path[j + 1]] < Dm[path[j], path[j + 1]] - 1e-12 if j + 1 < n else False:
                    path[:j + 1] = path[:j + 1][::-1]; improved = True
        Lp = sum(Dm[path[k], path[k + 1]] for k in range(n - 1))
        if Lp < bestL: best, bestL = list(path), Lp
    return best

def cva(DA, DB, nperm=5000):
    n = DA.shape[0]; iu = np.triu_indices(n, 1)
    def one(Dp, Ds):
        path = best_path(Dp); adj = np.zeros((n, n), bool)
        sep = np.zeros((n, n)); pos = {v: k for k, v in enumerate(path)}
        for k in range(n - 1): adj[path[k], path[k + 1]] = adj[path[k + 1], path[k]] = True
        for i in range(n):
            for j in range(n): sep[i, j] = abs(pos[i] - pos[j])
        def score(X): return X[iu].mean() - X[adj].mean()
        obs = score(Ds); nl = []
        for _ in range(nperm):
            p = rng.permutation(n); nl.append(score(Ds[np.ix_(p, p)]))
        nl = np.array(nl); decay = spearmanr(sep[iu], Ds[iu])[0]
        return obs, nl, decay
    oA, nA, dA = one(DA, DB); oB, nB, dB = one(DB, DA)
    obs = (oA + oB) / 2; nl = (nA + nB) / 2
    return dict(obs=round(float(obs), 5), null_mean=round(float(nl.mean()), 5), null_sd=round(float(nl.std()), 5),
                z=round(float((obs - nl.mean()) / nl.std()), 2), decay_rho=round(float((dA + dB) / 2), 3))

def truth_recovery(Dfull):
    n = Dfull.shape[0]; path = best_path(Dfull)
    adj_true = sum(1 for k in range(n - 1) if abs(path[k] - path[k + 1]) == 1) / (n - 1)
    return dict(adj_precision=round(adj_true, 3), chance=round(2 / n, 3),
                abs_spearman=round(abs(float(spearmanr(path, np.arange(n))[0])), 3))

OUT = os.path.join(ROOT, "results_bound_chain.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(res, open(OUT, "w"), indent=1)

vms = U["vms"]; strat = vms["strat"]; sheets = vms["sheets"]
DA, DB, Dfull = char_mats(vms)
if "B1_size" not in res:
    nA = np.array([sum(sum(1 for _ in [0]) for _ in [0]) for _ in sheets])  # placeholder replaced below
    nA = np.array([sum(v for k, v in vms["half"][(s, 0)].items() if k[0] == "w") for s in sheets], float)
    nB = np.array([sum(v for k, v in vms["half"][(s, 1)].items() if k[0] == "w") for s in sheets], float)
    res["B1_size"] = tw_test(resid_size(DA, nA), resid_size(DB, nB), strat); save(); print("B1", res["B1_size"])
if "B2_quire" not in res:
    q = np.array([s.split(":")[0] for s in sheets]); same = q[:, None] == q[None, :]
    res["B2_quire"] = dict(same_quire=tw_test(DA, DB, strat, same), diff_quire=tw_test(DA, DB, strat, ~same)); save()
    print("B2", res["B2_quire"])
if "B3_blocks" not in res:
    r = {}
    for b in ("w", "l"):
        a, bb, _ = char_mats(vms, block=b); r[b] = tw_test(a, bb, strat)
    res["B3_blocks"] = r; save(); print("B3", r)
if "B4_negctrl" not in res:
    # rebuild two control windows with words shuffled across the window, same sheet sizes
    r = {}
    for name in ("CI_start", "Galen_w2"):
        c = U["controls"][name]; keys = c["sheets"]
        allw = [w for s in keys for h in (0, 1) for w in c["words"][(s, h)]]; rng.shuffle(allw)
        half, pos = {}, 0
        for s in keys:
            for h in (0, 1):
                n = len(c["words"][(s, h)]); ws = allw[pos:pos + n]; pos += n; cc = Counter()
                for k in range(0, n, 8):
                    line = ws[k:k + 8]
                    for w in line:
                        st = "^" + w + "$"; cc.update(("w", a, b) for a, b in zip(st, st[1:]))
                    st = "".join(line); cc.update(("l", a, b) for a, b in zip(st, st[1:]))
                half[(s, h)] = cc
        a, bb, _ = char_mats(dict(sheets=keys, half=half)); r[name] = tw_test(a, bb, c["strat"])
    res["B4_negctrl"] = r; save(); print("B4", r)

# order-sensitive CVA in the largest group
def largest(strat):
    g = max(np.unique(strat), key=lambda x: np.sum(strat == x)); return np.where(strat == g)[0], g
if "CVA_VMS" not in res:
    idx, g = largest(strat)
    res["CVA_VMS"] = dict(group=str(g), n=int(len(idx)), **cva(DA[np.ix_(idx, idx)], DB[np.ix_(idx, idx)])); save()
    print("CVA VMS", res["CVA_VMS"])
for name, c in U["controls"].items():
    k = f"CVA_{name}"
    if k in res: continue
    a, bb, full = char_mats(c); idx, g = largest(c["strat"])
    res[k] = dict(n=int(len(idx)), **cva(a[np.ix_(idx, idx)], bb[np.ix_(idx, idx)]),
                  truth_block=truth_recovery(full[np.ix_(idx, idx)]), truth_all=truth_recovery(full)); save()
    print(k, res[k])
