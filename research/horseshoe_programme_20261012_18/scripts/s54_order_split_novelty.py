#!/usr/bin/env python3
"""PREREG 38: random vs order splits for novelty/coverage within herbal scribe 1, and an early->late direction asymmetry.
usage: python3 s54_order_split_novelty.py REPLICATION_ROOT VOYNICHDECOMP OUT
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "s51_order_vs_vocabulary.py")).read().split("R = {}")[0])
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist
ATOMS = [("ckh", "K"), ("cth", "T"), ("cph", "P"), ("cfh", "F"), ("ch", "C"), ("sh", "S")]
def atoms(w):
    for a, b in ATOMS: w = w.replace(a, b)
    return w
H = np.flatnonzero(strata == "H|1"); order = H[np.argsort(pos[H])]
def pool(toks, idx): c = Counter(); [c.update(toks[i]) for i in idx]; return c
def rarefy(c, n, rng):
    arr = np.repeat(list(c.keys()), list(c.values())); return Counter(rng.choice(arr, size=n, replace=False))
def metrics(train, test):
    tv = list(train); ts = set(train); novel = [w for w in test if w not in ts]
    if not novel: return 1.0, 1.0
    D = cdist(novel, tv, scorer=Levenshtein.distance, score_cutoff=1, workers=-1); e1 = (D <= 1).any(1)
    near = dict(zip(novel, e1)); cov = sum(n for w, n in test.items() if w in ts or near.get(w, False)) / sum(test.values())
    return float(e1.mean()), float(cov)
def split_metrics(toks, A, B, rng, draws=20):
    a, b = pool(toks, A), pool(toks, B); n = min(sum(a.values()), sum(b.values())); r = []
    for _ in range(draws):
        ra, rb = rarefy(a, n, rng), rarefy(b, n, rng); r.append(metrics(ra, rb) + metrics(rb, ra))
    return np.mean(r, 0)   # E1 a->b, COV a->b, E1 b->a, COV b->a
R = {}
for label, fn in (("ZL", tokens_zl), ("TTLI", tokens_tt)):
    raw = fn(); toks = [Counter({atoms(w): n for w, n in c.items()}) if label == "ZL" else c for c in raw]
    rng = np.random.default_rng(SEED + 31)
    early, late = order[:12], order[12:]
    o = split_metrics(toks, early, late, rng)
    rand = np.array([split_metrics(toks, *np.split(rng.permutation(H), 2), rng, draws=3) for _ in range(1000)])
    rE1 = np.concatenate([rand[:, 0], rand[:, 2]]); rCOV = np.concatenate([rand[:, 1], rand[:, 3]])
    pct = lambda v, dist: float(np.mean(dist < v))
    A_obs = o[0] - o[2]
    nullA = []
    for _ in range(1000):
        perm = rng.permutation(H); m = split_metrics(toks, perm[:12], perm[12:], rng, draws=3); nullA.append(m[0] - m[2])
    nullA = np.array(nullA); zA = (A_obs - nullA.mean()) / nullA.std(ddof=1)
    R[label] = dict(E1_early_to_late=o[0], COV_early_to_late=o[1], E1_late_to_early=o[2], COV_late_to_early=o[3],
                    random_E1_pcts=[5, 25, 50, 95], random_E1=np.percentile(rE1, [5, 25, 50, 95]).tolist(), random_COV=np.percentile(rCOV, [5, 25, 50, 95]).tolist(),
                    pct_E1_fwd=pct(o[0], rE1), pct_E1_bwd=pct(o[2], rE1), pct_COV_fwd=pct(o[1], rCOV), pct_COV_bwd=pct(o[3], rCOV),
                    A=float(A_obs), A_null_mean=float(nullA.mean()), A_null_sd=float(nullA.std(ddof=1)), A_z=float(zA))
    r = R[label]
    print(label, "ORDER E1 fwd %.3f (pct %.2f) bwd %.3f (pct %.2f) | COV fwd %.3f (pct %.2f) bwd %.3f (pct %.2f)" % (o[0], r["pct_E1_fwd"], o[2], r["pct_E1_bwd"], o[1], r["pct_COV_fwd"], o[3], r["pct_COV_bwd"]))
    print(label, "RANDOM E1 5/25/50/95%%", np.round(r["random_E1"], 3), "COV", np.round(r["random_COV"], 3))
    print(label, "DIRECTION A %.4f null %.4f sd %.4f z %.2f" % (A_obs, nullA.mean(), nullA.std(ddof=1), zA), flush=True)
flat = all(R[k]["pct_E1_fwd"] < .05 and R[k]["pct_E1_bwd"] < .05 for k in R)
notflat = all(R[k]["pct_E1_fwd"] > .25 and R[k]["pct_E1_bwd"] > .25 for k in R)
R["verdict_split"] = "FLATTERED" if flat else ("NOT_FLATTERED" if notflat else "UNRESOLVED")
R["verdict_direction"] = ("DIRECTIONAL_FORWARD" if all(R[k]["A_z"] >= 2 for k in ("ZL", "TTLI")) else "DIRECTIONAL_BACKWARD" if all(R[k]["A_z"] <= -2 for k in ("ZL", "TTLI"))
                          else "NO_DIRECTION" if abs(R["ZL"]["A_z"]) < 1 else "UNRESOLVED")
json.dump(R, open(os.path.join(OUT, "results_order_split.json"), "w"), indent=1)
print("SPLIT", R["verdict_split"], "| DIRECTION", R["verdict_direction"])
