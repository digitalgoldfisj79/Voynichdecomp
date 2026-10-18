#!/usr/bin/env python3
"""PREREG 10b: equal-N rarefied distances; T_w, CVA, truth recovery on VMS, controls, word-shuffled controls."""
import json, os, pickle, sys
from collections import Counter
import numpy as np
sys.path.insert(0, "/home/claude/horseshoe/scripts"); import lib_methods as L
from scipy.stats import spearmanr
ROOT = sys.argv[1]; D = os.path.join(ROOT, "data"); rng = np.random.default_rng(20261016)
U = pickle.load(open(os.path.join(D, "s22_units.pkl"), "rb"))
src = open("/home/claude/horseshoe/scripts/s23_bound_chain.py").read()
lib = src.split("OUT = os.path.join")[0].split("U = pickle.load")[1].split("\n", 1)[1]  # function defs only
exec(lib)  # within_perm, tw_test, resid_size, best_path, cva, truth_recovery, char_mats (uses global rng, L)

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
    half, pos = {}, 0
    for s in keys:
        for h in (0, 1):
            n = len(c["words"][(s, h)]); half[(s, h)] = feats(allw[pos:pos + n]); pos += n
    units["SHUF_" + name] = dict(sheets=keys, half=half, strat=c["strat"])

N = {b: min(sum(v for k, v in u["half"][(s, h)].items() if k[0] == b) for u in units.values()
            for s in u["sheets"] for h in (0, 1)) for b in ("w", "l")}
print("rarefaction N", N)

def rare_mats(u, draws=5):
    keys = u["sheets"]; vocab = sorted(set().union(*[u["half"][(s, h)] for s in keys for h in (0, 1)]))
    vi = {v: i for i, v in enumerate(vocab)}; wmask = np.array([v[0] == "w" for v in vocab])
    arrs = {}
    for s in keys:
        for h in (0, 1):
            c = u["half"][(s, h)]
            arrs[(s, h)] = {b: np.repeat([vi[k] for k in c if k[0] == b], [c[k] for k in c if k[0] == b]) for b in ("w", "l")}
    out = []
    for _ in range(draws):
        mats = []
        for hs in ([0], [1], [0, 1]):
            X = np.zeros((len(keys), len(vocab)))
            for r, s in enumerate(keys):
                for h in hs:
                    for b in ("w", "l"):
                        idx = rng.choice(arrs[(s, h)][b], size=N[b], replace=False); np.add.at(X[r], idx, 1)
            mats.append(L.js_matrix(L.profiles(X, wmask)))
        out.append(mats)
    return [np.mean([o[k] for o in out], 0) for k in range(3)]

OUT = os.path.join(ROOT, "results_rarefied.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {"N": N}
for name, u in units.items():
    if name in res: continue
    DA, DB, Dfull = rare_mats(u); strat = u["strat"]
    g = max(np.unique(strat), key=lambda x: np.sum(strat == x)); idx = np.where(strat == g)[0]
    r = dict(T_w=tw_test(DA, DB, strat), CVA=cva(DA[np.ix_(idx, idx)], DB[np.ix_(idx, idx)], nperm=3000))
    if name != "VMS":
        r["truth_block"] = truth_recovery(Dfull[np.ix_(idx, idx)]); r["truth_all"] = truth_recovery(Dfull)
    res[name] = r; json.dump(res, open(OUT, "w"), indent=1); print(name, json.dumps(r))
