#!/usr/bin/env python3
"""PREREG 26 (RUNNING_RESULTS_horseshoe.md, 2026-10-16): volvelle with ring wear/replacement and variable ring size
vs the PREREG 25 self-citation model. Tuned on ZL FIT->DEV; scored on the locked TEST pages in ZL and Takahashi (TTLI).
usage: python3 -I s41_volvelle_wear.py VOYNICHDECOMP OUT  (needs s41_prefix.py and s40_out results alongside)
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "s41_prefix.py")).read())  # data loading, splits, cells(), boot_se(), stats(), to_lines(), split4()

def new_ring(tr, r, N, b, rng):
    W, P = tr["uni"]; cells_ = [split4(w)[r] for w in rng.choice(W, size=N, p=P)]
    for j in range(1, N, 2):
        if rng.random() < b: cells_[j] = ""
    return cells_
def gen_vol3(tr, n, rng, s, kappa, b, V, rho, N, r):
    W, P = tr["uni"]; vols = []
    for _ in range(V):
        cols = [split4(w) for w in rng.choice(W, size=N, p=P)]
        for j in range(1, N, 2):
            if rng.random() < b: cols[j] = ["", "", "", ""]
        vols.append(([[cols[j][q] for j in range(N)] for q in range(4)], [int(rng.integers(N))] * 4))
    out, v = [], int(rng.integers(V))
    while len(out) < n:
        if rng.random() >= rho: v = int(rng.integers(V))
        rings, pos = vols[v]
        x = "".join(rings[q][pos[q]] for q in range(4))
        if x: out.append(x)
        if rng.random() < kappa:
            for q in range(4): pos[q] = (pos[q] + 1) % N
        else:
            for q in range(4):
                if rng.random() < s: pos[q] = (pos[q] + 1) % N
        if rng.random() < r:
            q = int(rng.integers(4)); rings[q] = new_ring(tr, q, N, b, rng)
    return out
exec(open(os.path.join(HERE, "s40_volvelle_refine.py")).read().split("# ---------------- generators")[1].split("GENS = ")[0])  # gen_cit
GENS = {"VOLVELLE3": gen_vol3, "SELFCITE": gen_cit}
def sample_v3(rng):
    return dict(s=float(rng.uniform(.2, 1)), kappa=float(rng.uniform(0, 1)), b=float(rng.uniform(0, .9)), V=int(rng.integers(1, 9)),
                rho=float(rng.uniform(0, .95)), N=int(rng.integers(12, 241)), r=float(10 ** rng.uniform(-4, -0.5)))
def gen_cells(fam, cfg, tr, ntok, vocab, reps, rng):
    return np.mean([cells(to_lines(GENS[fam](tr, ntok, rng, **cfg), tr["LL"], rng), vocab) for _ in range(reps)], 0)
def stage(name, fn):
    p = os.path.join(OUT, "stages", name + ".pkl")
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    r_ = fn(); pickle.dump(r_, open(p + ".tmp", "wb")); os.replace(p + ".tmp", p); return r_

rng = np.random.default_rng(SEED + 31)
fit_l, dev_l = L(FIT), L(DEV); tr_fit = stats(fit_l)
dev_se, _, _ = boot_se(dev_l, tr_fit["vocab"], rng); dev_t = cells(dev_l, tr_fit["vocab"]); ndev = sum(len(l) for l in dev_l)
def search():
    r2 = np.random.default_rng(SEED + 41); res = []
    for k in range(NCFG):
        cfg = sample_v3(r2); res.append((Dfun(gen_cells("VOLVELLE3", cfg, tr_fit, ndev, tr_fit["vocab"], 2, r2), dev_t, dev_se), cfg))
        if k % 25 == 0: print("V3", k, round(min(x[0] for x in res), 1), flush=True)
    res.sort(key=lambda x: x[0])
    top = sorted([(Dfun(gen_cells("VOLVELLE3", c, tr_fit, ndev, tr_fit["vocab"], 5, r2), dev_t, dev_se), c) for _, c in res[:5]], key=lambda x: x[0])
    return dict(all=res, top5=top, best=top[0])
SV = stage("search_V3", search)
SC = pickle.load(open(os.path.join(os.path.dirname(OUT.rstrip("/")), "s40_out", "stages", "search_SELFCITE.pkl"), "rb"))
best = {"VOLVELLE3": SV["best"][1], "SELFCITE": SC["best"][1]}
print("best V3", SV["best"], "| selfcite", SC["best"], flush=True)

def final(tid, pg):
    def fn():
        r3 = np.random.default_rng(SEED + (51 if tid == "ZLZI" else 52))
        trl = [l for f in FIT + DEV if f in pg for l in pg[f]]; tel = [l for f in TEST if f in pg for l in pg[f]]
        tr = stats(trl); se, Ot, Et = boot_se(tel, tr["vocab"], r3); t = cells(tel, tr["vocab"], Ot, Et); nt = sum(len(l) for l in tel)
        gen = {fam: gen_cells(fam, best[fam], tr, nt, tr["vocab"], 5, r3) for fam in GENS}
        part = lambda g, sl: float(np.sum(((g[sl] - t[sl]) / np.maximum(se[sl], 1e-9)) ** 2))
        res = {fam: dict(D=Dfun(g, t, se), D_repeat=part(g, slice(0, 24)), D_vocab=part(g, slice(24, 28)), vocab=g[24:].tolist()) for fam, g in gen.items()}
        diffs = []
        for _ in range(NBOOT):
            idx = r3.integers(len(tel), size=len(tel)); tb = cells([tel[i] for i in idx], tr["vocab"], Ot[idx], Et[idx])
            diffs.append(Dfun(gen["SELFCITE"], tb, se) - Dfun(gen["VOLVELLE3"], tb, se))
        return dict(models=res, target_vocab=t[24:].tolist(), diff_mean=float(np.mean(diffs)), diff_sd=float(np.std(diffs, ddof=1)))
    return stage(f"final_{tid}", fn)
R = {"ZLZI": final("ZLZI", pages), "TTLI": final("TTLI", pages_tt)}
for tid, f in R.items():
    m = f["models"]; f["margin"] = m["SELFCITE"]["D"] - m["VOLVELLE3"]["D"]; f["margin_sd"] = f["margin"] / f["diff_sd"]
    print(tid, {k: {kk: round(vv, 1) for kk, vv in v.items() if kk != "vocab"} for k, v in m.items()}, "margin SD", round(f["margin_sd"], 2), flush=True)
verdict = "CONFIRMS_VOLVELLE_ADVANTAGE" if all(R[t]["margin"] > 2 * R[t]["diff_sd"] for t in R) else "FALSIFIES_VOLVELLE_ADVANTAGE"
json.dump(dict(best=best, search_top5=SV["top5"], final=R, verdict=verdict), open(os.path.join(OUT, "results_volvelle_wear.json"), "w"), indent=1, default=str)
print("VERDICT", verdict)
