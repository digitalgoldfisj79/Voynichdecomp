#!/usr/bin/env python3
"""PREREG 29 (RUNNING_RESULTS_horseshoe.md, 2026-10-16): multi-arm coupled device vs the PREREG 27 one-stream copier.
Reuses s42 (cached stages) for data, scoring and the copier. usage: python3 -I s44_multiarm_device.py VOYNICHDECOMP VULGATE_JSON OUT
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_, VUL_, OUT44 = sys.argv[1:4]
sys.argv = [sys.argv[0], REPO_, VUL_, os.path.join(os.path.dirname(OUT44.rstrip("/")), "s42_out")]
exec(open(os.path.join(HERE, "s42_two_streams.py")).read())  # cached: stage1, searches, finals of s42
OUT = OUT44; os.makedirs(os.path.join(OUT, "stages"), exist_ok=True)
from rapidfuzz.process import cdist as _cdist

def gen_device(tr, n, rng, A, N, b, q, z, r):
    W, P = tr["uni"]
    def ring_from(words, k): return [split4(w)[k] for w in words]
    disks = []
    for _ in range(A):
        cols = list(rng.choice(W, size=N, p=P)); rings = [ring_from(cols, k) for k in range(4)]
        for j in range(1, N, 2):
            if rng.random() < b:
                for k in range(4): rings[k][j] = ""
        disks.append((rings, [int(rng.integers(N))] * 4))
    out, cur = [], 0
    while len(out) < n:
        cur = (cur + 1) % A if rng.random() < q else int(rng.integers(A))
        rings, pos = disks[cur]
        x = "".join(rings[k][pos[k]] for k in range(4))
        if x: out.append(x)
        if rng.random() < z:
            base = pos[0]
            for k in range(4): pos[k] = (base + 1) % N
        else:
            k = int(rng.integers(4)); pos[k] = (pos[k] + 1) % N
        if rng.random() < r:
            k = int(rng.integers(4)); rings[k] = ring_from(list(rng.choice(W, size=N, p=P)), k)
    return out
def sample_dev(rng):
    return dict(A=int(rng.integers(1, 5)), N=int(rng.integers(12, 121)), b=float(rng.uniform(0, .8)),
                q=float(rng.uniform(0, 1)), z=float(rng.uniform(0, 1)), r=float(10 ** rng.uniform(-4, -0.5)))
def dev_cells(cfg, tr, ntok, reps, rng):
    return np.mean([all_cells(to_lines(gen_device(tr, ntok, rng, **cfg), tr["LL"], rng), tr["vocab"]) for _ in range(reps)], 0)

def search_dev():
    r2 = np.random.default_rng(SEED + 121); res = []
    for k in range(NCFG):
        cfg = sample_dev(r2); res.append((Dfun(dev_cells(cfg, tr_fit, ndev, 2, r2), dev_t, dev_se), cfg))
        if k % 25 == 0: print("DEVICE", k, round(min(x[0] for x in res), 1), flush=True)
    res.sort(key=lambda x: x[0])
    top = sorted([(Dfun(dev_cells(c_, tr_fit, ndev, 5, r2), dev_t, dev_se), c_) for _, c_ in res[:5]], key=lambda x: x[0])
    return dict(all=res, top5=top, best=top[0])
SD = stage("search_DEVICE", search_dev)
print("DEVICE best DEV", round(SD["best"][0], 1), SD["best"][1], "| ONE best DEV", round(SR["ONE"]["best"][0], 1), flush=True)

def novel_profile(stream, vocab_list, vocab):
    nov = [w for w in set(stream) if w not in vocab]
    if not nov: return dict(novel_token_share=0.0, novel_one_edit_share=None)
    D_ = _cdist(nov, vocab_list, scorer=Levenshtein.distance, score_cutoff=1, workers=-1)
    return dict(novel_token_share=float(np.mean([w not in vocab for w in stream])), novel_one_edit_share=float(np.mean((D_ <= 1).any(1))))
def final_dev(tid, pg):
    def fn():
        r3 = np.random.default_rng(SEED + (131 if tid == "ZLZI" else 132))
        fs = [f for f in FIT + DEV if f in pg]; trl = [l for f in fs for l in pg[f]]; tel = [l for f in TEST if f in pg for l in pg[f]]
        tr = make_tr(trl, fs, pg); se, Ot, Et, Pt = boot_se35(tel, tr["vocab"], r3); t = all_cells(tel, tr["vocab"], Ot, Et, Pt)
        nt = sum(len(l) for l in tel)
        g_dev = dev_cells(SD["best"][1], tr, nt, 5, r3); g_one = gen_cells(SR["ONE"]["best"][1], tr, nt, 5, r3)
        part = lambda g, sl: float(np.sum(((g[sl] - t[sl]) / np.maximum(se[sl], 1e-9)) ** 2))
        res = {k: dict(D=Dfun(g, t, se), D_repeat=part(g, slice(0, 24)), D_vocab=part(g, slice(24, 28)), D_sim=part(g, slice(28, 32)),
                       D_len=part(g, slice(32, 35)), cells=g.tolist()) for k, g in (("DEVICE", g_dev), ("ONE", g_one))}
        res["DEVICE"]["novelty"] = novel_profile(gen_device(tr, nt, r3, **SD["best"][1]), sorted(tr["vocab"]), tr["vocab"])
        diffs = []
        for _ in range(NBOOT):
            idx = r3.integers(len(tel), size=len(tel))
            tb = np.concatenate([cells([tel[i] for i in idx], tr["vocab"], Ot[idx], Et[idx]), ext_cells(Pt, idx)])
            diffs.append(Dfun(g_one, tb, se) - Dfun(g_dev, tb, se))
        return dict(models=res, target=t.tolist(), diff_mean=float(np.mean(diffs)), diff_sd=float(np.std(diffs, ddof=1)))
    return stage("final_" + tid, fn)
R2 = {"ZLZI": final_dev("ZLZI", pages), "TTLI": final_dev("TTLI", pages_tt)}
for tid, f in R2.items():
    m_ = f["models"]; f["margin"] = m_["ONE"]["D"] - m_["DEVICE"]["D"]; f["margin_sd"] = f["margin"] / f["diff_sd"]
    print(tid, {k: {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "cells"} for k, v in m_.items()}, "margin SD", round(f["margin_sd"], 2), flush=True)
verdict = "CONFIRMS_DEVICE_ADVANTAGE" if all(R2[t]["margin"] > 2 * R2[t]["diff_sd"] for t in R2) else "FALSIFIES_DEVICE_ADVANTAGE"
json.dump(dict(best_device=SD["best"], top5=SD["top5"], one_stream_best=SR["ONE"]["best"], final=R2, verdict=verdict),
          open(os.path.join(OUT, "results_multiarm_device.json"), "w"), indent=1, default=str)
print("VERDICT", verdict)
