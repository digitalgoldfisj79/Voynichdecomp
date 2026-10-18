#!/usr/bin/env python3
"""PREREG 32 (RUNNING_RESULTS_horseshoe.md, 2026-10-16): salvage of the Stolfi-dial idea.
S1 Stolfi vs naive 5-slot layout; S2 mostly-aligned pair mixture (held-out cross-entropy) vs Latin and naive; S3 device + one-edit slips.
usage: python3 -I s47_salvage.py VOYNICHDECOMP VULGATE_JSON OUT
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "s46_stolfi_dials.py")).read().split('p = os.path.join(OUT, "results_stolfi_dials.json")')[0]
exec(src)
from scipy.optimize import minimize_scalar
def stage(name, fn):
    p = os.path.join(OUT, "stages", name + ".pkl"); os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    r_ = fn(); pickle.dump(r_, open(p + ".tmp", "wb")); os.replace(p + ".tmp", p); return r_
rng = np.random.default_rng(SEED + 171)
trainW = [w for l in L(FIT) + L(DEV) for w in l]; testW = [w for l in L(TEST) for w in l]
tc, te = Counter(trainW), Counter(testW); novel = [w for w in te if w not in tc]

def naive_split(w):
    n = len(w)
    if n >= 5: return (w[0], w[1], w[2:-2], w[-2], w[-1])
    if n == 4: return (w[0], w[1], "", w[2], w[3])
    if n == 3: return (w[0], w[1], "", "", w[2])
    if n == 2: return (w[0], "", "", "", w[1])
    return ("", "", "", "", w)
def naive_dev(N):
    comp = [Counter() for _ in range(5)]
    for w, c in tc.items():
        for k, x in enumerate(naive_split(w)): comp[k][x] += c
    return [{x for x, _ in comp[k].most_common(N)} for k in range(5)]
cache = {}
def cov_stolfi(w, dev): return covered(w, dev, False, "FIVE_N", cache)
def cov_naive(w, slots): return all(x in slots[k] for k, x in enumerate(naive_split(w)))
def decoys_of(ws, k):
    out = []
    for _ in range(k):
        for w in ws:
            a = list(w)
            for _ in range(10):
                rng.shuffle(a)
                if "".join(a) != w: break
            out.append("".join(a))
    return out
DEC = decoys_of(novel, 3)

def S1():
    rows = []
    for N in (24, 36, 48):
        ds, dn = learn(tc, N, False, "FIVE_N"), naive_dev(N)
        for nm, f, d in (("STOLFI", cov_stolfi, ds), ("NAIVE", cov_naive, dn)):
            tok = sum(te[w] for w in te if f(w, d)) / sum(te.values()); nov = np.mean([f(w, d) for w in novel]); dec = np.mean([f(w, d) for w in DEC])
            rows.append(dict(layout=nm, N=N, test_token_cov=tok, novel_type_cov=float(nov), decoy_cov=float(dec)))
            print("S1", nm, N, "test %.3f novel %.3f decoy %.3f" % (tok, nov, dec), flush=True)
    return rows
R1 = stage("S1_layout", S1)

def pairs_from(words, kind):
    out = []
    for w in words:
        if kind == "stolfi_crust":
            p = canonical(w, False)
            if p: out.append((p[0], p[4]))
        elif kind == "stolfi_mantle":
            p = canonical(w, False)
            if p: out.append((p[1], p[3]))
        else:
            if len(w) >= 4: out.append((w[:2], w[-2:]))
    return out
def xent(fit, dev, test, K=36):
    cf = Counter(fit); cl = Counter(l for l, _ in fit); cr = Counter(r for _, r in fit); n = len(fit)
    Vl, Vr = len(cl) + 1, len(cr) + 1
    pl = lambda l: (cl[l] + .5) / (n + .5 * Vl); pr = lambda r: (cr[r] + .5) / (n + .5 * Vr)
    top = dict(cf.most_common(K)); tt = sum(top.values()); A = lambda l, r: top.get((l, r), 0) / tt
    J = lambda l, r: cf[(l, r)] / n
    def H(test_, a, f): return -np.mean([np.log2(a * f(l, r) + (1 - a) * pl(l) * pr(r)) for l, r in test_])
    a = minimize_scalar(lambda x: H(dev, x, A), bounds=(0, .999), method="bounded").x
    b = minimize_scalar(lambda x: H(dev, x, J), bounds=(0, .999), method="bounded").x
    hi, hm, hj = H(test, 0, A), H(test, a, A), H(test, b, J)
    return dict(alpha=float(a), beta=float(b), H_indep=hi, H_mix=hm, H_joint=hj, captured=float((hi - hm) / max(hi - hj, 1e-12)), n_test=len(test))
def S2():
    out = {}
    fitW = [w for l in L(FIT) for w in l]; devW = [w for l in L(DEV) for w in l]
    for kind in ("stolfi_crust", "stolfi_mantle", "naive"):
        out["VMS_" + kind] = xent(pairs_from(fitW, kind), pairs_from(devW, kind), pairs_from(testW, kind))
    Vt = json.load(open(VULJ))["tokens"]
    out["LATIN_naive"] = xent(pairs_from(Vt[:30000], "naive"), pairs_from(Vt[30000:40000], "naive"), pairs_from(Vt[40000:50000], "naive"))
    for k, v in out.items(): print("S2", k, {kk: round(vv, 3) for kk, vv in v.items()}, flush=True)
    return out
R2 = stage("S2_alignment", S2)

ALPH = sorted({a for w in tc for a in w})
def edits1(w):
    s = set()
    for i in range(len(w) + 1):
        for a in ALPH: s.add(w[:i] + a + w[i:])
        if i < len(w):
            s.add(w[:i] + w[i + 1:])
            for a in ALPH: s.add(w[:i] + a + w[i + 1:])
    s.discard(w); s.discard(""); return s
def S3():
    res = {}
    for nm, f, d in (("STOLFI", cov_stolfi, learn(tc, 36, False, "FIVE_N")), ("NAIVE", cov_naive, naive_dev(36))):
        def close(w): return f(w, d) or any(f(v, d) for v in edits1(w))
        nov_e = float(np.mean([f(w, d) for w in novel])); nov_c = float(np.mean([close(w) for w in novel]))
        dec_e = float(np.mean([f(w, d) for w in DEC])); dec_c = float(np.mean([close(w) for w in DEC]))
        res[nm] = dict(novel_exact=nov_e, novel_closure=nov_c, decoy_exact=dec_e, decoy_closure=dec_c, solves=bool(nov_c >= .8 and dec_c <= .5 * nov_c))
        print("S3", nm, {k: round(v, 3) if isinstance(v, float) else v for k, v in res[nm].items()}, flush=True)
    return res
R3 = stage("S3_slips", S3)
json.dump(dict(S1=R1, S2=R2, S3=R3, n_novel=len(novel)), open(os.path.join(OUT, "results_salvage.json"), "w"), indent=1)
