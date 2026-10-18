#!/usr/bin/env python3
"""PREREG 27 (RUNNING_RESULTS_horseshoe.md, 2026-10-16): two interleaved streams vs one stream.
Stage 1: zig-zag diagnostic (edit-similarity and word-length covariance by lag) on VMS vs Vulgate.
Stage 2: strong one-stream vs multi-stream copy generators, fitted FIT->DEV, scored on TEST (ZL and TTLI).
usage: python3 -I s42_two_streams.py VOYNICHDECOMP VULGATE_JSON OUT
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
VULJ = sys.argv[2]; sys.argv = [sys.argv[0], sys.argv[1], sys.argv[3]]
exec(open(os.path.join(HERE, "s41_prefix.py")).read())
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist
SIM = {}
def sim(a, b):
    k = (a, b) if a <= b else (b, a)
    v = SIM.get(k)
    if v is None: v = SIM[k] = 1 - Levenshtein.distance(a, b) / max(len(a), len(b), 1)
    return v

# ---------------- extended per-line parts: 4 similarity lags + 3 length lags
def ext_parts(lines):
    S_o = np.zeros((len(lines), 4)); S_e = np.zeros((len(lines), 4)); C_o = np.zeros((len(lines), 3)); C_n = np.zeros((len(lines), 3))
    for i, l in enumerate(lines):
        n_ = len(l); M = np.array([[sim(a, b) for b in l] for a in l]); mp = (M.sum() - np.trace(M)) / (n_ * (n_ - 1))
        x = np.array([len(w) for w in l], float); xc = x - x.mean(); ss = (xc ** 2).sum()
        for j, d in enumerate(range(1, 5)):
            if n_ > d: S_o[i, j] = sum(M[k, k + d] for k in range(n_ - d)); S_e[i, j] = (n_ - d) * mp
        for j, d in enumerate(range(1, 4)):
            if n_ > d:
                C_o[i, j] = (xc[:-d] * xc[d:]).sum() + (n_ - d) * ss / (n_ * (n_ - 1))  # obs minus permutation expectation
                C_n[i, j] = (n_ - d) * ss / n_
    return S_o, S_e, C_o, C_n
def ext_cells(P, idx=None):
    S_o, S_e, C_o, C_n = P if idx is None else [a[idx] for a in P]
    return np.concatenate([S_o.sum(0) / np.maximum(S_e.sum(0), 1e-12), C_o.sum(0) / np.maximum(C_n.sum(0), 1e-12)])
def all_cells(lines, vocab, O=None, E=None, P=None):
    if P is None: P = ext_parts(lines)
    return np.concatenate([cells(lines, vocab, O, E), ext_cells(P)])
def boot_se35(lines, vocab, rng):
    O, E = line_parts(lines); P = ext_parts(lines); bs = []
    for _ in range(NBOOT):
        idx = rng.integers(len(lines), size=len(lines))
        bs.append(np.concatenate([cells([lines[i] for i in idx], vocab, O[idx], E[idx]), ext_cells(P, idx)]))
    return np.std(bs, 0, ddof=1), O, E, P

def stage(name, fn):
    p = os.path.join(OUT, "stages", name + ".pkl")
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    r_ = fn(); pickle.dump(r_, open(p + ".tmp", "wb")); os.replace(p + ".tmp", p); return r_

# ---------------- STAGE 1
def zigzag(lines, rng):
    P = ext_parts(lines); obs = ext_cells(P); bs = []
    for _ in range(NBOOT):
        idx = rng.integers(len(lines), size=len(lines)); bs.append(ext_cells(P, idx))
    bs = np.array(bs); out = {}
    for name, a, b, c in (("similarity", 0, 1, 2), ("length", 4, 5, 6)):
        d21, d23 = bs[:, b] - bs[:, a], bs[:, b] - bs[:, c]
        out[name] = dict(lag_values=(obs[0:4] if name == "similarity" else obs[4:7]).tolist(),
                         lag2_minus_lag1=[float(obs[b] - obs[a]), *np.percentile(d21, [2.5, 97.5]).tolist()],
                         lag2_minus_lag3=[float(obs[b] - obs[c]), *np.percentile(d23, [2.5, 97.5]).tolist()])
        out[name]["zigzag"] = bool(np.percentile(d21, 2.5) > 0 and np.percentile(d23, 2.5) > 0)
    return out
def stage1():
    r1 = np.random.default_rng(SEED + 61)
    vms = [l for f in pages for l in pages[f]]
    V = json.load(open(VULJ))["tokens"][:60000]; LLv = np.array([len(l) for l in vms])
    vul = to_lines(V, LLv, np.random.default_rng(SEED + 62))
    shuf = []; toks = [w for l in vms for w in l]; r1.shuffle(toks); shuf = to_lines(toks, LLv, r1)
    return {"VMS": zigzag(vms, r1), "VULGATE": zigzag(vul, r1), "VMS_TOKEN_SHUFFLE": zigzag(shuf, r1)}
Z = stage("stage1_zigzag", stage1)
for k, v in Z.items(): print("STAGE1", k, json.dumps(v), flush=True)

# ---------------- STAGE 2 generators
def page_sizes(fs, pg): return np.array([sum(len(l) for l in pg[f]) for f in fs if f in pg])
def neighbours(vocab_list):
    D_ = cdist(vocab_list, vocab_list, scorer=Levenshtein.distance, score_cutoff=1, workers=-1)
    return {w: [vocab_list[j] for j in np.nonzero(D_[i] == 1)[0]] for i, w in enumerate(vocab_list)}
def gen_streams(tr, n, rng, K, q, c, w, m, a, p):
    W, P = tr["uni"]; NB = tr["nb"]; PS = tr["PS"]; out = []; pool = rng.choice(len(W), size=n + 10, p=P); pi_ = 0
    while len(out) < n:
        size = int(rng.choice(PS)); streams = [[] for _ in range(K)]; page = []; last_stream = -1; t = 0
        while t < size and len(out) < n:
            s_ = t % K if rng.random() < q else int(rng.integers(K))
            hist = streams[s_]; x = None
            if hist and rng.random() < c:
                lag = int(rng.integers(1, min(w, len(hist)) + 1))
                if lag == 1 and s_ == last_stream and rng.random() < a and len(hist) > 1: lag = 2
                x = hist[-lag]
                if rng.random() < m and NB.get(x): x = NB[x][int(rng.integers(len(NB[x])))]
            elif page and rng.random() < p:
                x = page[int(rng.integers(len(page)))]
            else:
                x = W[int(pool[pi_ % len(pool)])]; pi_ += 1
            hist.append(x); page.append(x); out.append(x); last_stream = s_; t += 1
    return out
def sample_cfg(multi, rng):
    cfg = dict(c=float(rng.uniform(0, .7)), w=int(rng.integers(1, 9)), m=float(rng.uniform(0, .9)),
               a=float(rng.uniform(0, 1)), p=float(rng.uniform(0, .8)))
    cfg.update(K=int(rng.integers(2, 4)), q=float(rng.uniform(.5, 1))) if multi else cfg.update(K=1, q=1.0)
    return cfg
def gen_cells(cfg, tr, ntok, reps, rng):
    return np.mean([all_cells(to_lines(gen_streams(tr, ntok, rng, **cfg), tr["LL"], rng), tr["vocab"]) for _ in range(reps)], 0)
def make_tr(lines, fs, pg):
    tr = stats(lines); tr["nb"] = neighbours(sorted(tr["vocab"])); tr["PS"] = page_sizes(fs, pg); return tr

rng = np.random.default_rng(SEED + 71)
fit_l, dev_l = L(FIT), L(DEV); tr_fit = make_tr(fit_l, FIT, pages)
dev_se, _, _, _ = boot_se35(dev_l, tr_fit["vocab"], rng); dev_t = all_cells(dev_l, tr_fit["vocab"]); ndev = sum(len(l) for l in dev_l)
def search(multi):
    def fn():
        r2 = np.random.default_rng(SEED + (81 if multi else 82)); res = []
        for k in range(NCFG):
            cfg = sample_cfg(multi, r2); res.append((Dfun(gen_cells(cfg, tr_fit, ndev, 2, r2), dev_t, dev_se), cfg))
            if k % 25 == 0: print("MULTI" if multi else "ONE", k, round(min(x[0] for x in res), 1), flush=True)
        res.sort(key=lambda x: x[0])
        top = sorted([(Dfun(gen_cells(c_, tr_fit, ndev, 5, r2), dev_t, dev_se), c_) for _, c_ in res[:5]], key=lambda x: x[0])
        return dict(all=res, top5=top, best=top[0])
    return stage("search_" + ("MULTI" if multi else "ONE"), fn)
SR = {"ONE": search(False), "MULTI": search(True)}
for k, v in SR.items(): print(k, "best DEV", round(v["best"][0], 1), v["best"][1], flush=True)

def final(tid, pg):
    def fn():
        r3 = np.random.default_rng(SEED + (91 if tid == "ZLZI" else 92))
        fs = [f for f in FIT + DEV if f in pg]; trl = [l for f in fs for l in pg[f]]; tel = [l for f in TEST if f in pg for l in pg[f]]
        tr = make_tr(trl, fs, pg); se, Ot, Et, Pt = boot_se35(tel, tr["vocab"], r3); t = all_cells(tel, tr["vocab"], Ot, Et, Pt)
        nt = sum(len(l) for l in tel); gen = {k: gen_cells(SR[k]["best"][1], tr, nt, 5, r3) for k in SR}
        part = lambda g, sl: float(np.sum(((g[sl] - t[sl]) / np.maximum(se[sl], 1e-9)) ** 2))
        res = {k: dict(D=Dfun(g, t, se), D_repeat=part(g, slice(0, 24)), D_vocab=part(g, slice(24, 28)), D_sim=part(g, slice(28, 32)),
                       D_len=part(g, slice(32, 35)), cells=g.tolist()) for k, g in gen.items()}
        diffs = []
        for _ in range(NBOOT):
            idx = r3.integers(len(tel), size=len(tel))
            tb = np.concatenate([cells([tel[i] for i in idx], tr["vocab"], Ot[idx], Et[idx]), ext_cells(Pt, idx)])
            diffs.append(Dfun(gen["ONE"], tb, se) - Dfun(gen["MULTI"], tb, se))
        return dict(models=res, target=t.tolist(), diff_mean=float(np.mean(diffs)), diff_sd=float(np.std(diffs, ddof=1)))
    return stage("final_" + tid, fn)
R = {"ZLZI": final("ZLZI", pages), "TTLI": final("TTLI", pages_tt)}
for tid, f in R.items():
    m_ = f["models"]; f["margin"] = m_["ONE"]["D"] - m_["MULTI"]["D"]; f["margin_sd"] = f["margin"] / f["diff_sd"]
    print(tid, {k: {kk: round(vv, 1) for kk, vv in v.items() if kk != "cells"} for k, v in m_.items()}, "margin SD", round(f["margin_sd"], 2), flush=True)
verdict = "CONFIRMS_TWO_STREAM" if all(R[t]["margin"] > 2 * R[t]["diff_sd"] for t in R) else "FALSIFIES_TWO_STREAM"
json.dump(dict(stage1=Z, best={k: v["best"] for k, v in SR.items()}, top5={k: v["top5"] for k, v in SR.items()}, final=R, verdict=verdict),
          open(os.path.join(OUT, "results_two_streams.json"), "w"), indent=1, default=str)
print("VERDICT", verdict)
