#!/usr/bin/env python3
"""PREREG 18 (RUNNING_RESULTS_horseshoe.md, 2026-10-15): checks on npcompl33t's reply.
A: per-stage rates of chy, chey, chedy, chol, chor, cheol along the eo777 chain (two orders: eo777's, and Bio-final).
B: Torsten-style coverage asymmetry 1a vs Bio(4a) and 1a vs Stars(3c+4c) against a whole-page permutation null and an
   equal-token rarefied version. Stages with <1500 tokens are excluded (fixed before outcomes).
usage: python3 -I s32_npcompl_checks.py DATA_DIR OUT_DIR
"""
import itertools, json, os, pickle, sys
from collections import Counter
import numpy as np, pandas as pd
from scipy.stats import spearmanr

DATA, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261015)
W = pd.read_csv(os.path.join(DATA, "words_by_page.csv.gz"))
C = pd.read_csv(os.path.join(DATA, "eo777_chain_pages.csv"))
stage_of = dict(zip(C.page, C.group))
W["stage"] = W.folio.map(stage_of)
FORMS = ["chy", "chey", "chedy", "chol", "chor", "cheol"]
ORDER_EO = ["1a", "1b", "1.5a", "2a", "3a", "3b", "3c", "4a", "4c"]
ORDER_BIOFINAL = ["1a", "1b", "1.5a", "2a", "3a", "3b", "3c", "4c", "4a"]
ntok = W.groupby("stage").size()
keep = [s for s in ORDER_EO if ntok.get(s, 0) >= 1500]

# ---------- A: rates
rates = {}
for s in keep:
    ws = W[W.stage == s].word; c = Counter(ws); n = len(ws)
    rates[s] = {"tokens": n, **{f: 1000 * c[f] / n for f in FORMS}}
R = pd.DataFrame(rates).T
def exact_spearman_p(x, y):
    obs = spearmanr(x, y)[0]; perms = list(itertools.permutations(range(len(y)))) if len(y) <= 8 else None
    if perms is None:
        null = [spearmanr(x, rng.permutation(y))[0] for _ in range(20000)]
    else:
        null = [spearmanr(x, [y[i] for i in p])[0] for p in perms]
    return float(obs), float(np.mean(np.abs(null) >= abs(obs) - 1e-12))
A = {}
for name, order in (("eo777_order", ORDER_EO), ("bio_final_order", ORDER_BIOFINAL)):
    o = [s for s in order if s in keep]; Ro = R.loc[o]; pos = list(range(len(o)))
    A[name] = {"stages": o,
               "rho_stage_chey": exact_spearman_p(pos, list(Ro.chey)),
               "rho_stage_chedy": exact_spearman_p(pos, list(Ro.chedy)),
               "rho_chy_chey": exact_spearman_p(list(Ro.chy), list(Ro.chey))}

# ---------- B: coverage asymmetry with page-permutation null
def cov(g, h):  # share of g tokens whose type occurs in h
    th = set(h); return float(np.mean([w in th for w in g]))
def asym(g, h): return cov(g, h) - cov(h, g)
pages = {f: list(d.word) for f, d in W.groupby("folio")}
def groups_of(stages): return [f for f in pages if stage_of.get(f) in stages]
B = {}
for label, other in (("1a_vs_Bio", {"4a"}), ("1a_vs_Stars", {"3c", "4c"})):
    P1, P2 = groups_of({"1a"}), groups_of(other)
    g1 = [w for f in P1 for w in pages[f]]; g2 = [w for f in P2 for w in pages[f]]
    obs = asym(g1, g2)
    pool = P1 + P2; n1 = len(g1); null = []
    while len(null) < 2000:
        perm = rng.permutation(pool); acc, cut = 0, None
        for i, f in enumerate(perm):
            acc += len(pages[f])
            if acc >= n1: cut = i + 1; break
        a = [w for f in perm[:cut] for w in pages[f]]; b = [w for f in perm[cut:] for w in pages[f]]
        if abs(len(a) - n1) / n1 <= 0.10 and b: null.append(asym(a, b))
    null = np.array(null)
    m = min(len(g1), len(g2)); rar = []
    for _ in range(500):
        rar.append(asym(list(rng.choice(np.array(g1, dtype=object), m, replace=False)),
                        list(rng.choice(np.array(g2, dtype=object), m, replace=False))))
    # rarefied null: same rarefaction applied to page-permuted pools
    rnull = []
    for _ in range(500):
        perm = rng.permutation(pool); acc, cut = 0, None
        for i, f in enumerate(perm):
            acc += len(pages[f])
            if acc >= n1: cut = i + 1; break
        a = [w for f in perm[:cut] for w in pages[f]]; b = [w for f in perm[cut:] for w in pages[f]]
        mm = min(len(a), len(b))
        rnull.append(asym(list(rng.choice(np.array(a, dtype=object), mm, replace=False)),
                          list(rng.choice(np.array(b, dtype=object), mm, replace=False))))
    rnull = np.array(rnull)
    B[label] = dict(tokens=[len(g1), len(g2)], cov_1a_in_X=cov(g1, g2), cov_X_in_1a=cov(g2, g1), obs=obs,
                    null_mean=float(null.mean()), null_sd=float(null.std(ddof=1)), z=float((obs - null.mean()) / null.std(ddof=1)),
                    rarefied_mean=float(np.mean(rar)), rarefied_null_mean=float(rnull.mean()), rarefied_null_sd=float(rnull.std(ddof=1)),
                    rarefied_z=float((np.mean(rar) - rnull.mean()) / rnull.std(ddof=1)))
res = dict(prereg="PREREG 18", rates=rates, A=A, B=B)
pickle.dump(res, open(os.path.join(OUT, "s32.pkl"), "wb"))
json.dump(res, open(os.path.join(OUT, "results_npcompl_checks.json"), "w"), indent=1)
print(R.round(2).to_string()); print(json.dumps(A, indent=1)); print(json.dumps(B, indent=1))
