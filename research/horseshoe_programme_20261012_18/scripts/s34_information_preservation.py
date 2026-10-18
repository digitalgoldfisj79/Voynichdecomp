#!/usr/bin/env python3
"""PREREG 20 (RUNNING_RESULTS_horseshoe.md, 2026-10-15): is 'information preserved' by a frequency-balanced
edit mapping evidence of information-preserving change, or a property of edit-space density + optimisation?

EARLY = chain stages 1a,1b,1.5a; LATE = 3c,4a,4c (eo777 chain, ZL words). Types covering 80% of tokens per period.
Mapping = min-cost transport EARLY -> LATE over edges with Levenshtein <= B; unmatched mass cost 1; tie-break 0.01*edit.
IP = I(W;T)/H(W) on matched mass; MATCH = transported fraction of EARLY mass.
Null = LATE masses permuted among LATE types (form<->frequency link broken), 200 draws.
Each solve result is pickled atomically to OUT/solves/.
usage: python3 -I s34_information_preservation.py DATA_DIR OUT_DIR
"""
import json, os, pickle, sys
from collections import Counter
import numpy as np, pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

DATA, OUT = sys.argv[1], sys.argv[2]; os.makedirs(os.path.join(OUT, "solves"), exist_ok=True)
SEED, NNULL, COVER = 20261015, 200, 0.80
W = pd.read_csv(os.path.join(DATA, "words_by_page.csv.gz"))
C = pd.read_csv(os.path.join(DATA, "eo777_chain_pages.csv"))
W["stage"] = W.folio.map(dict(zip(C.page, C.group))); W = W[W.line_num > 0]
EARLY, LATE = {"1a", "1b", "1.5a"}, {"3c", "4a", "4c"}

def top_mass(words):
    c = Counter(words); tot = sum(c.values()); out, acc = {}, 0
    for w, n in c.most_common():
        if acc / tot >= COVER: break
        out[w] = n; acc += n
    s = sum(out.values()); return {w: n / s for w, n in out.items()}

E = top_mass(W[W.stage.isin(EARLY)].word); L = top_mass(W[W.stage.isin(LATE)].word)
ew, lw = list(E), list(L); pe = np.array([E[w] for w in ew]); pl0 = np.array([L[w] for w in lw])

def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]
D = np.array([[lev(a, b) for b in lw] for a in ew])

def solve(pl, B):
    I, J = np.nonzero(D <= B); ne, nl, m = len(ew), len(lw), len(I)
    # variables: flows x_ij (m), slack_e (ne), slack_l (nl)
    c = np.concatenate([0.01 * D[I, J], np.ones(ne), np.ones(nl) * 0.0])
    rows = np.concatenate([I, np.arange(ne), ne + J, ne + np.arange(nl)])
    cols = np.concatenate([np.arange(m), m + np.arange(ne), np.arange(m), m + ne + np.arange(nl)])
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(ne + nl, m + ne + nl)).tocsr()
    b = np.concatenate([pe, pl])
    r = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    assert r.status == 0, r.message
    F = np.zeros((ne, nl)); F[I, J] = r.x[:m]
    return F

def metrics(F):
    tot = F.sum()
    if tot <= 0: return dict(IP=0.0, MATCH=0.0)
    P = F / tot; pw = P.sum(1); pt = P.sum(0)
    nz = P > 0
    mi = float((P[nz] * np.log2(P[nz] / (pw[:, None] * pt[None, :])[nz])).sum())
    hw = float(-(pw[pw > 0] * np.log2(pw[pw > 0])).sum())
    moved = F.copy(); moved[D == 0] = 0  # mass that changed form
    return dict(IP=mi / hw, MATCH=float(tot / pe.sum()), MOVED=float(moved.sum() / pe.sum()),
                n_targets_used=int((F.sum(0) > 1e-12).sum()))

def run(name, pl, B):
    p = os.path.join(OUT, "solves", f"{name}_B{B}.pkl")
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    m = metrics(solve(pl, B)); pickle.dump(m, open(p + ".tmp", "wb")); os.replace(p + ".tmp", p); return m

rng = np.random.default_rng(SEED)
res = {"n_early_types": len(ew), "n_late_types": len(lw)}
for B in (2, 1):
    real = run("real", pl0, B)
    null = [run(f"null{k:03d}", np.random.default_rng(SEED + k).permutation(pl0), B) for k in range(NNULL)]
    out = {"real": real}
    for key in ("IP", "MATCH", "MOVED"):
        v = np.array([n[key] for n in null])
        out[key] = dict(real=real[key], null_mean=float(v.mean()), null_sd=float(v.std(ddof=1)),
                        z=float((real[key] - v.mean()) / v.std(ddof=1)), null_p95=float(np.percentile(v, 95)))
    res[f"B{B}"] = out
    print(f"B={B}", json.dumps({k: {kk: round(vv, 4) for kk, vv in out[k].items()} for k in ("IP", "MATCH", "MOVED")}), flush=True)
z = res["B2"]["IP"]["z"]
res["verdict"] = ("FALSIFIES_IP_DISCRIMINATES" if abs(z) < 2 else "CONFIRMS_IP_CARRIES_REAL_SIGNAL" if z >= 2
                  else "IP_REAL_BELOW_NULL")
json.dump(res, open(os.path.join(OUT, "results_information_preservation.json"), "w"), indent=1)
print("VERDICT", res["verdict"], "types", len(ew), len(lw))
