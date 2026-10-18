#!/usr/bin/env python3
"""PREREG 25 (RUNNING_RESULTS_horseshoe.md, 2026-10-16): refined volvelle vs refined self-citation, locked test set.
Stages pickled atomically to OUT/stages/. usage: python3 -I s40_volvelle_refine.py VOYNICHDECOMP OUT
"""
import json, os, pickle, re, sys
from collections import Counter
import numpy as np

REPO, OUT = sys.argv[1:3]; os.makedirs(os.path.join(OUT, "stages"), exist_ok=True)
SEED, LAGS, NBOOT, NCFG = 20261016, range(1, 7), 200, 150
ATOMS = [("ckh", "K"), ("cth", "T"), ("cph", "P"), ("cfh", "F"), ("ch", "C"), ("sh", "S")]
def atoms(w):
    for a, b in ATOMS: w = w.replace(a, b)
    return w
SLOTS = [lambda a: a, lambda a: a[:1], lambda a: a[-1:], lambda a: a[-2:]]

D = json.load(open(os.path.join(REPO, "voynich_transcriptions_slim.json")))
pages = {}
for fid in sorted(D["pages"]):
    pg = D["pages"][fid]; ls = []
    for ln in sorted(pg, key=lambda x: int(x) if x.isdigit() else 99999):
        u = pg[ln].get("u", "")
        if len(u) >= 2 and u[1] == "P":
            t = [atoms(x) for x in pg[ln]["t"].get("ZLZI", "").split() if re.fullmatch(r"[a-z]+", x)]
            if len(t) >= 2: ls.append(t)
    if ls: pages[fid] = ls
perm = list(np.random.default_rng(SEED).permutation(sorted(pages))); n = len(perm)
FIT, DEV, TEST = perm[:n // 2], perm[n // 2:3 * n // 4], perm[3 * n // 4:]
L = lambda fs: [l for f in fs for l in pages[f]]

# ---------------- per-line cell contributions (obs/exp for 24 repeat cells)
def line_parts(lines):
    O = np.zeros((len(lines), 24)); E = np.zeros((len(lines), 24))
    for i, l in enumerate(lines):
        Ln = len(l)
        for s, f in enumerate(SLOTS):
            v = [f(w) for w in l]; c = Counter(v); pe = sum(k * (k - 1) for k in c.values()) / (Ln * (Ln - 1))
            for j, d in enumerate(LAGS):
                if Ln > d:
                    O[i, s * 6 + j] = sum(v[k] == v[k + d] for k in range(Ln - d)); E[i, s * 6 + j] = (Ln - d) * pe
    return O, E
def cells(lines, vocab, O=None, E=None, ttr_n=5000):
    if O is None: O, E = line_parts(lines)
    rep = O.sum(0) / np.maximum(E.sum(0), 1e-12)
    toks = [w for l in lines for w in l]
    att = np.mean([w in vocab for w in toks]); sub = toks[:ttr_n]; ttr = len(set(sub)) / len(sub)
    ml = np.mean([len(w) for w in toks]); one = np.mean([len(w) == 1 for w in toks])
    return np.concatenate([rep, [att, ttr, ml, one]])
def boot_se(lines, vocab, rng):
    O, E = line_parts(lines); bs = []
    for _ in range(NBOOT):
        idx = rng.integers(len(lines), size=len(lines))
        bs.append(cells([lines[i] for i in idx], vocab, O[idx], E[idx]))
    return np.std(bs, 0, ddof=1), O, E
Dfun = lambda g, t, se: float(np.sum(((g - t) / np.maximum(se, 1e-9)) ** 2))

def stats(lines):
    toks = [w for l in lines for w in l]; c = Counter(toks); W = list(c); P = np.array([c[x] for x in W], float); P /= P.sum()
    ac = Counter(a for w in toks for a in w); A = list(ac); AP = np.array([ac[a] for a in A], float); AP /= AP.sum()
    return dict(uni=(W, P), atoms=(A, AP), LL=np.array([len(l) for l in lines]), vocab=set(c), ntok=len(toks))
def to_lines(stream, LL, rng):
    out, i = [], 0
    while i < len(stream):
        k = int(rng.choice(LL)); out.append(stream[i:i + k]); i += k
    return [l for l in out if len(l) >= 2]
def split4(w):
    a = list(w)
    return [a[0], "".join(a[1:-2]) if len(a) >= 4 else "", a[-2] if len(a) >= 3 else "", a[-1] if len(a) >= 2 else ""]

# ---------------- generators
def gen_vol(tr, n, rng, s, kappa, b, V, rho):
    W, P = tr["uni"]; vols = []
    for _ in range(V):
        cols = [split4(w) for w in rng.choice(W, size=24, p=P)]
        for j in range(1, 24, 2):
            if rng.random() < b: cols[j] = ["", "", "", ""]
        vols.append(([[cols[j][r] for j in range(24)] for r in range(4)], [int(rng.integers(24))] * 4))
    out, v = [], int(rng.integers(V))
    while len(out) < n:
        if rng.random() >= rho: v = int(rng.integers(V))
        rings, pos = vols[v]
        x = "".join(rings[r][pos[r]] for r in range(4))
        if x: out.append(x)
        if rng.random() < kappa:
            for r in range(4): pos[r] = (pos[r] + 1) % 24
        else:
            for r in range(4):
                if rng.random() < s: pos[r] = (pos[r] + 1) % 24
    return out
def gen_cit(tr, n, rng, c, w, shape, m, a):
    W, P = tr["uni"]; A, AP = tr["atoms"]; out = []; draws = rng.choice(W, size=n, p=P)
    for k in range(n):
        if out and rng.random() < c:
            for _ in range(5):
                lag = int(rng.integers(1, min(w, len(out)) + 1)) if shape == "uniform" else min(int(rng.geometric(1 / max(w / 2, 1))), len(out))
                if not (lag == 1 and rng.random() < a): break
            x = out[-lag]
            if rng.random() < m and x:
                i = int(rng.integers(len(x))); x = x[:i] + rng.choice(A, p=AP) + x[i + 1:]
            out.append(x)
        else:
            out.append(draws[k])
    return out
GENS = {"VOLVELLE": gen_vol, "SELFCITE": gen_cit}
def sample_cfg(fam, rng):
    if fam == "VOLVELLE":
        return dict(s=float(rng.uniform(.2, 1)), kappa=float(rng.uniform(0, 1)), b=float(rng.uniform(0, .9)),
                    V=int(rng.integers(1, 9)), rho=float(rng.uniform(0, .95)))
    return dict(c=float(rng.uniform(0, .6)), w=int(rng.integers(1, 13)), shape=str(rng.choice(["uniform", "geometric"])),
                m=float(rng.uniform(0, .8)), a=float(rng.uniform(0, 1)))

def gen_cells(fam, cfg, tr, ntok, vocab, reps, rng):
    return np.mean([cells(to_lines(GENS[fam](tr, ntok, rng, **cfg), tr["LL"], rng), vocab) for _ in range(reps)], 0)

def stage(name, fn):
    p = os.path.join(OUT, "stages", name + ".pkl")
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    r = fn(); pickle.dump(r, open(p + ".tmp", "wb")); os.replace(p + ".tmp", p); return r

rng = np.random.default_rng(SEED + 5)
fit_l, dev_l, test_l = L(FIT), L(DEV), L(TEST)
tr_fit = stats(fit_l)
dev_se, _, _ = boot_se(dev_l, tr_fit["vocab"], rng); dev_t = cells(dev_l, tr_fit["vocab"])
ndev = sum(len(l) for l in dev_l)

def search(fam):
    def fn():
        r2 = np.random.default_rng(SEED + (11 if fam == "VOLVELLE" else 12)); res = []
        for k in range(NCFG):
            cfg = sample_cfg(fam, r2); d = Dfun(gen_cells(fam, cfg, tr_fit, ndev, tr_fit["vocab"], 2, r2), dev_t, dev_se)
            res.append((d, cfg))
            if k % 25 == 0: print(fam, k, round(min(x[0] for x in res), 1), flush=True)
        res.sort(key=lambda x: x[0]); top = []
        for d0, cfg in res[:5]:
            top.append((Dfun(gen_cells(fam, cfg, tr_fit, ndev, tr_fit["vocab"], 5, r2), dev_t, dev_se), cfg))
        top.sort(key=lambda x: x[0])
        return dict(all=res, top5=top, best=top[0])
    return stage(f"search_{fam}", fn)
S = {fam: search(fam) for fam in GENS}
for fam in GENS: print(fam, "best DEV D", round(S[fam]["best"][0], 1), S[fam]["best"][1], flush=True)

def final():
    r3 = np.random.default_rng(SEED + 21); tr = stats(fit_l + dev_l)
    se, Ot, Et = boot_se(test_l, tr["vocab"], r3); t = cells(test_l, tr["vocab"], Ot, Et); ntest = sum(len(l) for l in test_l)
    gen = {fam: gen_cells(fam, S[fam]["best"][1], tr, ntest, tr["vocab"], 5, r3) for fam in GENS}
    Dt = {fam: Dfun(g, t, se) for fam, g in gen.items()}
    diffs, noise = [], []
    for _ in range(NBOOT):
        idx = r3.integers(len(test_l), size=len(test_l)); tb = cells([test_l[i] for i in idx], tr["vocab"], Ot[idx], Et[idx])
        diffs.append(Dfun(gen["SELFCITE"], tb, se) - Dfun(gen["VOLVELLE"], tb, se))
    # noise reference: real DEV vs real TEST
    Od, Ed = line_parts(dev_l); real_dev = cells(dev_l, tr["vocab"], Od, Ed)
    for _ in range(NBOOT):
        i1 = r3.integers(len(dev_l), size=len(dev_l)); i2 = r3.integers(len(test_l), size=len(test_l))
        noise.append(Dfun(cells([dev_l[i] for i in i1], tr["vocab"], Od[i1], Ed[i1]), cells([test_l[i] for i in i2], tr["vocab"], Ot[i2], Et[i2]), se))
    return dict(D_test=Dt, target=t.tolist(), se=se.tolist(), gen={k: v.tolist() for k, v in gen.items()},
                diff_mean=float(np.mean(diffs)), diff_sd=float(np.std(diffs, ddof=1)),
                D_real_dev_vs_test=Dfun(real_dev, t, se), noise_p95=float(np.percentile(noise, 95)), noise_median=float(np.median(noise)))
F = stage("final_locked_test", final)
margin = F["D_test"]["SELFCITE"] - F["D_test"]["VOLVELLE"]
verdict = "CONFIRMS_VOLVELLE_ADVANTAGE" if margin > 2 * F["diff_sd"] else "FALSIFIES_VOLVELLE_ADVANTAGE"
adequate = {fam: F["D_test"][fam] <= F["noise_p95"] for fam in GENS}
out = dict(split_sizes=[len(FIT), len(DEV), len(TEST)], search={f: dict(best=S[f]["best"], top5=S[f]["top5"]) for f in GENS},
           final=F, margin=margin, margin_over_sd=margin / F["diff_sd"], verdict=verdict, adequate=adequate)
json.dump(out, open(os.path.join(OUT, "results_volvelle_refine.json"), "w"), indent=1, default=str)
print(json.dumps({"D_test": F["D_test"], "margin_sd": round(margin / F["diff_sd"], 2), "noise_p95": round(F["noise_p95"], 1),
                  "real_dev_vs_test": round(F["D_real_dev_vs_test"], 1), "verdict": verdict, "adequate": adequate}))
