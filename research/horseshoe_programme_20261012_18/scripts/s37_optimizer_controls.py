#!/usr/bin/env python3
"""PREREG 22 (RUNNING_RESULTS_horseshoe.md, 2026-10-16).
Run npcompl33t's optimizer (eo777/voynich-optimizer, Chain class unchanged) on:
  VMS_REAL (ZL paragraph text, eo777 stages 1b..4a), VMS_PAGESHUF (pages randomly reassigned to 8 pseudo-sections),
  VULGATE (consecutive 8-passage chains, his tokenisation).
Support = identity + generic edges (unit edit distance <= k, adjacent-section rates within factor 2).
Per arm: identity TV, frequency-only LP TV (certified optimum), joint lambda search (his bounded L-BFGS-B, 2 starts).
Each arm result is pickled atomically to OUT/arms/<arm>_ed<k>.pkl.
usage: python3 -I s37_optimizer_controls.py OPT_REPO VOYNICHDECOMP CHAIN_CSV VULGATE_JSON OUT_DIR [--joint]
"""
import json, os, pickle, re, sys, time
from collections import Counter
import numpy as np, pandas as pd
from scipy.optimize import minimize
from rapidfuzz.process import cdist
from rapidfuzz.distance import Levenshtein

OPT, REPO, CHAIN, VUL, OUT = sys.argv[1:6]
JOINT = "--joint" in sys.argv
sys.path.insert(0, OPT)
from optimizer.core import Chain  # his code, unchanged
os.makedirs(os.path.join(OUT, "arms"), exist_ok=True)
KEYS = ["1b", "1.5a", "2a", "3a", "3b", "3c", "4c", "4a"]
SIZES = [2411, 1798, 3107, 2212, 1648, 7412, 4598, 6812]
SEED, NREP, NVUL, LAMBDA, ITERS = 20261016, 10, 10, 0.2, 160
ATOMS = [("ckh", "K"), ("cth", "T"), ("cph", "P"), ("cfh", "F"), ("ch", "C"), ("sh", "S")]

def atomize(w):
    for a, b in ATOMS: w = w.replace(a, b)
    return w

# ---------------- VMS pages per stage (ZL paragraph lines)
C = pd.read_csv(CHAIN); stage_of = dict(zip(C.page, C.group))
D = json.load(open(os.path.join(REPO, "voynich_transcriptions_slim.json")))
pages = {}
for fid in sorted(D["pages"], key=lambda f: (int(re.match(r"f(\d+)", f).group(1)), f)):
    if stage_of.get(fid) not in KEYS: continue
    pg = D["pages"][fid]; toks = []
    for ln in sorted(pg, key=lambda x: int(x) if x.isdigit() else 99999):
        u = pg[ln].get("u", "")
        if len(u) >= 2 and u[1] == "P":
            toks += [t for t in pg[ln]["t"].get("ZLZI", "").split() if re.fullmatch(r"[a-z]+", t)]
    if toks: pages[fid] = (stage_of[fid], toks)

def vms_real():
    secs = []
    for k, n in zip(KEYS, SIZES):
        toks = [t for f, (s, ts) in pages.items() if s == k for t in ts]; secs.append(toks[:n])
    return secs

def vms_pageshuf(r):
    order = list(pages); np.random.default_rng(SEED + r).shuffle(order)
    stream = [t for f in order for t in pages[f][1]]
    real_sizes = [len(s) for s in vms_real()]
    out, i = [], 0
    for n in real_sizes: out.append(stream[i:i + n]); i += n
    return out

V = json.load(open(VUL))["tokens"]
def vulgate(i):
    s = V[i * sum(SIZES):(i + 1) * sum(SIZES)]; out, j = [], 0
    for n in SIZES: out.append(s[j:j + n]); j += n
    return out

# ---------------- chain data in his format
def build(sections, k, unitize):
    counts = [Counter(s) for s in sections]
    words = sorted(set().union(*counts)); wi = {w: i for i, w in enumerate(words)}
    U = [unitize(w) for w in words]
    stages = [{"key": KEYS[t], "counts": dict(counts[t]), "N": sum(counts[t].values())} for t in range(len(KEYS))]
    step_edges, n_edges = {}, {}
    for t in range(len(KEYS) - 1):
        A = list(counts[t]); B = list(counts[t + 1])
        ra = np.array([counts[t][w] / stages[t]["N"] for w in A]); rb = np.array([counts[t + 1][w] / stages[t + 1]["N"] for w in B])
        dist = cdist([unitize(w) for w in A], [unitize(w) for w in B], scorer=Levenshtein.distance, score_cutoff=k, workers=-1)
        I, J = np.nonzero((dist >= 1) & (dist <= k))
        ok = (rb[J] <= 2 * ra[I]) & (ra[I] <= 2 * rb[J])
        key = KEYS[t] + ">" + KEYS[t + 1]
        step_edges[key] = [[wi[A[i]], wi[B[j]], 1.0, [], 1] for i, j in zip(I[ok], J[ok])]
        n_edges[key] = int(ok.sum())
    data = {"words": words, "stages": stages, "chain": KEYS, "menus": {"scheduled_v22": {"step_edges": step_edges}}}
    return data, n_edges

def actual(c, ks, w):
    m = c.metrics(ks); return m["mean_frequency_mismatch"] + w * (1 - m["stages"][-1]["information_retained"]), m

def run_arm(name, sections, k, unitize):
    p = os.path.join(OUT, "arms", f"{name}_ed{k}{'_joint' if JOINT else ''}.pkl")
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    t0 = time.time(); data, ne = build(sections, k, unitize); c = Chain(data, 0)
    from scipy import sparse
    ident = [sparse.csr_matrix((np.where(a == b, 1., 0.), b, ptr), shape=(c.n, c.n)) for a, b, ptr in c.edges]
    mi = c.metrics(ident); lp, cert = c.frequency_optimum(); ml = c.metrics(lp)
    r = dict(arm=name, k=k, tokens=[len(s) for s in sections], n_edges=ne, active=int(c.n), params=int(c.size),
             tv_identity=mi["mean_frequency_mismatch"], tv_lp=ml["mean_frequency_mismatch"],
             ret_lp=ml["stages"][-1]["information_retained"], ret_lp_common=ml["stages"][-1]["common_information_retained"])
    r["R_lp"] = 1 - r["tv_lp"] / r["tv_identity"]
    if JOINT:
        best = None
        for label, init in (("frequency_warm_start", lp), ("identity_start", c.kernels(c.identity_theta(31)))):
            fit = minimize(c.value_grad, c.theta(init), args=(LAMBDA,), jac=True, method="L-BFGS-B", bounds=[(-25, 25)] * c.size,
                           options={"maxiter": ITERS, "ftol": 1e-11, "gtol": 1e-7, "maxcor": 8, "maxls": 25})
            for cand in (init, c.kernels(fit.x)):
                s, m = actual(c, cand, LAMBDA)
                if best is None or s < best[0]: best = (s, m)
        for cand in (ident,):
            s, m = actual(c, cand, LAMBDA)
            if s < best[0]: best = (s, m)
        m = best[1]
        r.update(tv_joint=m["mean_frequency_mismatch"], ret_joint=m["stages"][-1]["information_retained"],
                 ret_joint_common=m["stages"][-1]["common_information_retained"])
        r["R_joint"] = 1 - r["tv_joint"] / r["tv_identity"]
    r["seconds"] = time.time() - t0
    pickle.dump(r, open(p + ".tmp", "wb")); os.replace(p + ".tmp", p)
    print(name, k, {x: (round(v, 4) if isinstance(v, float) else v) for x, v in r.items() if x not in ("n_edges", "tokens", "arm", "k")}, flush=True)
    return r

lat = lambda w: w
res = {}
for k in ([1] if JOINT else [1, 2]):
    res[f"ed{k}"] = {"VMS_REAL": run_arm("VMS_REAL", vms_real(), k, atomize),
                     "VMS_PAGESHUF": [run_arm(f"VMS_PAGESHUF{r:02d}", vms_pageshuf(r), k, atomize) for r in range(NREP)],
                     "VULGATE": [run_arm(f"VULGATE{i:02d}", vulgate(i), k, lat) for i in range(NVUL)]}
json.dump(res, open(os.path.join(OUT, f"results_optimizer_controls{'_joint' if JOINT else ''}.json"), "w"), indent=1)

def verdict(key="R_lp", k=1):
    e = res[f"ed{k}"]; v = e["VMS_REAL"][key]
    ps = np.array([x[key] for x in e["VMS_PAGESHUF"]]); vu = np.array([x[key] for x in e["VULGATE"]])
    tp, tv = ps.mean() + 2 * ps.std(ddof=1), vu.mean() + 2 * vu.std(ddof=1)
    out = dict(stat=key, k=k, vms=v, pageshuf_mean=ps.mean(), pageshuf_sd=ps.std(ddof=1), vulgate_mean=vu.mean(), vulgate_sd=vu.std(ddof=1),
               z_vs_pageshuf=(v - ps.mean()) / ps.std(ddof=1), z_vs_vulgate=(v - vu.mean()) / vu.std(ddof=1))
    out["verdict"] = ("CONFIRMS" if v > tp and v > tv else "FALSIFIES" if v <= tp else "PARTIAL")
    return out
summ = [verdict("R_lp", k) for k in ([1] if JOINT else [1, 2])]
if JOINT: summ.append(verdict("R_joint", 1))
for s in summ: print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}))
json.dump(summ, open(os.path.join(OUT, f"verdict{'_joint' if JOINT else ''}.json"), "w"), indent=1)
