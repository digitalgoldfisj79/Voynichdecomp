#!/usr/bin/env python3
"""PREREG 21 (RUNNING_RESULTS_horseshoe.md, 2026-10-15): does the one-edit RESPELLED-mass excess replicate
in Takahashi (TTLI) and on adjacent chain stages? Method frozen from s34 (B<=1, 80% coverage, 200 permutations).
Paragraph lines only. Each solve pickled atomically under OUT/solves/<tag>/.
usage: python3 -I s35_respelling_replication.py REPO_ROOT CHAIN_CSV OUT_DIR
"""
import json, os, pickle, sys
from collections import Counter
import numpy as np, pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.stats import norm

REPO, CHAIN, OUT = sys.argv[1:4]
SEED, NNULL, COVER, B = 20261015, 200, 0.80, 1
ORDER = ["1a", "1b", "1.5a", "2a", "3a", "3b", "3c", "4a", "4c"]
C = pd.read_csv(CHAIN); stage_of = dict(zip(C.page, C.group))
D0 = json.load(open(os.path.join(REPO, "voynich_transcriptions_slim.json")))

def stage_words(tid):
    out = {s: [] for s in ORDER}
    for fid, pg in D0["pages"].items():
        s = stage_of.get(fid)
        if s not in out: continue
        for ln, line in pg.items():
            u = line.get("u", "")
            if len(u) >= 2 and u[1] == "P":
                out[s] += [t for t in line["t"].get(tid, "").split() if t.isalpha() and t.islower()]
    return out

def top_mass(words):
    c = Counter(words); tot = sum(c.values()); out, acc = {}, 0
    for w, n in c.most_common():
        if acc / tot >= COVER: break
        out[w] = n; acc += n
    s = sum(out.values()); return {w: n / s for w, n in out.items()}

def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]

def comparison(tag, early_words, late_words):
    os.makedirs(os.path.join(OUT, "solves", tag), exist_ok=True)
    E, L = top_mass(early_words), top_mass(late_words)
    ew, lw = list(E), list(L); pe = np.array([E[w] for w in ew]); pl0 = np.array([L[w] for w in lw])
    D = np.array([[lev(a, b) for b in lw] for a in ew]); I, J = np.nonzero(D <= B); ne, nl, m = len(ew), len(lw), len(I)
    c = np.concatenate([0.01 * D[I, J], np.ones(ne), np.zeros(nl)])
    rows = np.concatenate([I, np.arange(ne), ne + J, ne + np.arange(nl)])
    cols = np.concatenate([np.arange(m), m + np.arange(ne), np.arange(m), m + ne + np.arange(nl)])
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(ne + nl, m + ne + nl)).tocsr()
    def solve(pl, name):
        p = os.path.join(OUT, "solves", tag, name + ".pkl")
        if os.path.exists(p): return pickle.load(open(p, "rb"))
        r = linprog(c, A_eq=A, b_eq=np.concatenate([pe, pl]), bounds=(0, None), method="highs"); assert r.status == 0
        x = r.x[:m]; res = dict(retained=float(x[D[I, J] == 0].sum() / pe.sum()), respelled=float(x[D[I, J] == 1].sum() / pe.sum()))
        pickle.dump(res, open(p + ".tmp", "wb")); os.replace(p + ".tmp", p); return res
    real = solve(pl0, "real")
    nulls = [solve(np.random.default_rng(SEED + k).permutation(pl0), f"null{k:03d}") for k in range(NNULL)]
    out = dict(tag=tag, n_types=[ne, nl], tokens=[len(early_words), len(late_words)])
    for k in ("respelled", "retained"):
        v = np.array([n[k] for n in nulls])
        out[k] = dict(real=real[k], null_mean=float(v.mean()), null_sd=float(v.std(ddof=1)), z=float((real[k] - v.mean()) / v.std(ddof=1)))
    print(tag, {k: round(out[k]["z"], 2) for k in ("respelled", "retained")}, "real/null respelled",
          round(out["respelled"]["real"], 4), round(out["respelled"]["null_mean"], 4), flush=True)
    return out

results = {}
for tid in ("TTLI", "ZLZI"):
    sw = stage_words(tid); r = {"tokens_per_stage": {s: len(v) for s, v in sw.items()}}
    early = sw["1a"] + sw["1b"] + sw["1.5a"]; late = sw["3c"] + sw["4a"] + sw["4c"]
    r["early_late"] = comparison(f"{tid}_early_late", early, late)
    adj = [comparison(f"{tid}_{a}_to_{b}", sw[a], sw[b]) for a, b in zip(ORDER, ORDER[1:])]
    zs = np.array([x["respelled"]["z"] for x in adj])
    r["adjacent"] = adj; r["adjacent_stouffer_z"] = float(zs.sum() / np.sqrt(len(zs)))
    r["adjacent_stouffer_p_one_sided"] = float(norm.sf(r["adjacent_stouffer_z"]))
    results[tid] = r
    print(tid, "early->late respelled z", round(r["early_late"]["respelled"]["z"], 2), "adjacent Stouffer z", round(r["adjacent_stouffer_z"], 2), flush=True)
t = results["TTLI"]; a, b = t["early_late"]["respelled"]["z"], t["adjacent_stouffer_z"]
verdict = "CONFIRMS_REPLICATION" if a >= 2 and b >= 2 else "FALSIFIES_REPLICATION" if a < 2 and b < 2 else "MIXED_NO_CLAIM"
results["verdict"] = verdict
json.dump(results, open(os.path.join(OUT, "results_respelling_replication.json"), "w"), indent=1)
print("VERDICT", verdict)
