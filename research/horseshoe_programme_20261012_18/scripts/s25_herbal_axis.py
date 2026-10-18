#!/usr/bin/env python3
"""EXPLORATORY bounding of 10b: (a) word-block-only rarefied T_w/CVA on VMS; (b) does the Herbal-A scribe-1 path
agree with present quire grouping / book position?"""
import json, os, pickle, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "/home/claude/horseshoe/scripts"); import lib_methods as L
ROOT = sys.argv[1]; D = os.path.join(ROOT, "data"); rng = np.random.default_rng(20261017)
src = open("/home/claude/horseshoe/scripts/s23_bound_chain.py").read()
exec(src.split("OUT = os.path.join")[0].split("U = pickle.load")[1].split("\n", 1)[1])
U = pickle.load(open(os.path.join(D, "s22_units.pkl"), "rb")); u = U["vms"]; N = {"w": 569, "l": 451}
keys = u["sheets"]; strat = u["strat"]
def rare(blocks, draws=5):
    vocab = sorted({k for s in keys for h in (0, 1) for k in u["half"][(s, h)] if k[0] in blocks})
    vi = {v: i for i, v in enumerate(vocab)}; wmask = np.array([v[0] == "w" for v in vocab])
    out = []
    for _ in range(draws):
        mats = []
        for hs in ([0], [1], [0, 1]):
            X = np.zeros((len(keys), len(vocab)))
            for r, s in enumerate(keys):
                for h in hs:
                    c = u["half"][(s, h)]
                    for b in blocks:
                        arr = np.repeat([vi[k] for k in c if k[0] == b], [c[k] for k in c if k[0] == b])
                        np.add.at(X[r], rng.choice(arr, N[b], replace=False), 1)
            if len(blocks) == 1:
                P = X / X.sum(1, keepdims=True) + 1e-9; mats.append(L.js_matrix(P / P.sum(1, keepdims=True)))
            else: mats.append(L.js_matrix(L.profiles(X, wmask)))
        out.append(mats)
    return [np.mean([o[k] for o in out], 0) for k in range(3)]
res = {}
g = max(np.unique(strat), key=lambda x: np.sum(strat == x)); idx = np.where(strat == g)[0]
DA, DB, Df = rare(("w",))
res["w_only"] = dict(T_w=tw_test(DA, DB, strat), CVA=cva(DA[np.ix_(idx, idx)], DB[np.ix_(idx, idx)], nperm=3000))
print("w_only", res["w_only"])
DA, DB, Df = rare(("w", "l"))
sub = Df[np.ix_(idx, idx)]; path = best_path(sub)
names = [keys[i] for i in idx]; q = np.array([n.split(":")[0] for n in names])
S = pickle.load(open(os.path.join(D, "s14_features.pkl"), "rb"))["F"]
bpos = np.array([S[S.sheet == n].book_pos.min() for n in names])
adj_same = np.mean([q[path[k]] == q[path[k + 1]] for k in range(len(path) - 1)])
nl = []
for _ in range(10000):
    p = rng.permutation(len(path)); nl.append(np.mean([q[p[k]] == q[p[k + 1]] for k in range(len(p) - 1)]))
nl = np.array(nl)
rho = spearmanr(np.arange(len(path)), bpos[path])[0]
nr = np.array([spearmanr(np.arange(len(path)), rng.permutation(bpos))[0] for _ in range(10000)])
res["herbal_path"] = dict(path=[names[i] for i in path],
    same_quire_adjacent=dict(obs=round(float(adj_same), 3), null_mean=round(float(nl.mean()), 3),
                             null_sd=round(float(nl.std()), 3), z=round(float((adj_same - nl.mean()) / nl.std()), 2)),
    book_pos_spearman=dict(obs=round(float(rho), 3), null_sd=round(float(nr.std()), 3), z=round(float(rho / nr.std()), 2)))
print(json.dumps(res["herbal_path"], indent=1))
json.dump(res, open(os.path.join(ROOT, "results_herbal_axis.json"), "w"), indent=1)
