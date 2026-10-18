#!/usr/bin/env python3
"""PREREG 33 (RUNNING_RESULTS_horseshoe.md, 2026-10-16): are local word relations dial turns (whole-segment swaps) or copy slips (one-glyph edits)?
usage: python3 -I s48_turn_vs_slip.py VOYNICHDECOMP VULGATE_JSON OUT
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "s47_salvage.py")).read().split('R1 = stage("S1_layout", S1)')[0]
exec(src)  # gives tc, learn, canonical, naive_split, stage, L, FIT/DEV/TEST, pages, pages_tt, to_lines
from rapidfuzz.distance import Levenshtein
SEGC = {}
def segs(w, scheme):
    k = (w, scheme)
    if k not in SEGC: SEGC[k] = canonical(w, False) if scheme == "STOLFI" else naive_split(w)
    return SEGC[k]
def rel(a, b, scheme):
    if a == b: return 0, 0
    e = Levenshtein.distance(a, b); e1 = int(e == 1); sw = 0
    if e >= 2:
        p, q = segs(a, scheme), segs(b, scheme)
        if p is not None and q is not None and sum(x != y for x, y in zip(p, q)) == 1: sw = 1
    return e1, sw
def line_parts(lines, scheme):
    O = np.zeros((len(lines), 2, 4)); E = np.zeros((len(lines), 2, 4))
    for i, l in enumerate(lines):
        n = len(l); M = np.zeros((2, n, n))
        for x in range(n):
            for y in range(x + 1, n):
                r = rel(l[x], l[y], scheme); M[0, x, y] = M[0, y, x] = r[0]; M[1, x, y] = M[1, y, x] = r[1]
        for k in range(2):
            pm = M[k].sum() / (n * (n - 1))
            for j, d in enumerate(range(1, 5)):
                if n > d: O[i, k, j] = sum(M[k, x, x + d] for x in range(n - d)); E[i, k, j] = (n - d) * pm
    return O, E
def summary(O, E):
    R = O.sum(0) / np.maximum(E.sum(0), 1e-12)   # [E1/SW, lag]
    return dict(E1=R[0].tolist(), SW=R[1].tolist(), D1=float(R[1, 0] / R[0, 0]), D2=float(R[1, 0] - R[1, 1]))
def measure(lines, scheme, rng, boot=True):
    O, E = line_parts(lines, scheme); s = summary(O, E)
    if boot:
        bs = []
        for _ in range(NBOOT):
            idx = rng.integers(len(lines), size=len(lines)); b = summary(O[idx], E[idx]); bs.append((b["D1"], b["D2"]))
        bs = np.array(bs); s["D1_ci"] = np.percentile(bs[:, 0], [2.5, 97.5]).tolist(); s["D2_ci"] = np.percentile(bs[:, 1], [2.5, 97.5]).tolist()
    return s

ALPH = sorted({a for w in tc for a in w})
def slip(w, rng):
    a = list(w); op = rng.integers(3); i = int(rng.integers(len(a) + (op == 0)))
    if op == 0: a.insert(i, ALPH[rng.integers(len(ALPH))])
    elif op == 1 and len(a) > 1: del a[min(i, len(a) - 1)]
    else: a[min(i, len(a) - 1)] = ALPH[rng.integers(len(ALPH))]
    return "".join(a)
dev36 = learn(tc, 36, False, "FIVE_N")
def gen_device(n, rng, t, s, m):
    dials = [list(rng.permutation(sorted(d))) for d in dev36["slots"]]; pos = [int(rng.integers(len(d))) for d in dials]; out = []
    while len(out) < n:
        w = "".join(dials[k][pos[k]] for k in range(5))
        if w:
            out.append(slip(w, rng) if rng.random() < m else w)
        for k in range(5):
            if rng.random() < t: pos[k] = (pos[k] + int(rng.choice([-1, 1])) * int(rng.integers(1, s + 1))) % len(dials[k])
    return out
W_, P_ = list(tc), np.array(list(tc.values()), float); P_ /= P_.sum()
def gen_copy(n, rng, c, w, m):
    out = []
    while len(out) < n:
        if out and rng.random() < c:
            x = out[-int(rng.integers(1, min(w, len(out)) + 1))]
            if rng.random() < m: x = slip(x, rng)
        else: x = W_[rng.choice(len(W_), p=P_)]
        out.append(x)
    return out
LL = np.array([len(l) for f in pages for l in pages[f]])
def controls(scheme):
    def fn():
        r = np.random.default_rng(SEED + 181); res = []
        for t in (.2, .4, .6):
            for s in (1, 3):
                for m in (0, .1):
                    x = measure(to_lines(gen_device(10000, r, t, s, m), LL, r), scheme, r, boot=False); x.update(fam="DEVICE", cfg=(t, s, m)); res.append(x)
        for c in (.3, .5, .7):
            for w in (4, 10):
                for m in (.2, .5):
                    x = measure(to_lines(gen_copy(10000, r, c, w, m), LL, r), scheme, r, boot=False); x.update(fam="COPY", cfg=(c, w, m)); res.append(x)
        for x in res: print("CTRL", scheme, x["fam"], x["cfg"], "D1 %.3f D2 %.3f E1(1) %.2f SW(1) %.2f" % (x["D1"], x["D2"], x["E1"][0], x["SW"][0]), flush=True)
        return res
    return stage("controls_" + scheme, fn)
out = {}
for scheme in ("STOLFI", "NAIVE"):
    C = controls(scheme); dv = [x["D1"] for x in C if x["fam"] == "DEVICE"]; cp = [x["D1"] for x in C if x["fam"] == "COPY"]
    dr, cr = (min(dv), max(dv)), (min(cp), max(cp)); void = not (dr[0] > cr[1] or cr[0] > dr[1])
    res = {"controls": C, "device_range": dr, "copy_range": cr, "VOID": void}
    for tid, pg in (("ZLZI", pages), ("TTLI", pages_tt)):
        m_ = stage(f"vms_{scheme}_{tid}", lambda: measure([l for f in pg for l in pg[f]], scheme, np.random.default_rng(SEED + 191)))
        lo, hi = m_["D1_ci"]; dev_hi = dr[0] > cr[1]
        if void: v = "VOID"
        elif dev_hi: v = "DEVICE-LIKE" if lo > cr[1] and lo >= dr[0] * 0 + min(dr[0], lo) and lo > (cr[1] + dr[0]) / 2 * 0 + cr[1] and lo >= dr[0] else ("COPY-LIKE" if hi < dr[0] and hi <= cr[1] else "UNRESOLVED")
        else: v = "DEVICE-LIKE" if hi < cr[0] and hi <= dr[1] else ("COPY-LIKE" if lo > dr[1] and lo >= cr[0] else "UNRESOLVED")
        m_["verdict"] = v; res[tid] = m_
        print(scheme, tid, "D1 %.3f CI %s D2 %.3f CI %s | device %s copy %s VOID %s -> %s" % (m_["D1"], np.round(m_["D1_ci"], 3), m_["D2"], np.round(m_["D2_ci"], 3), np.round(dr, 3), np.round(cr, 3), void, v), flush=True)
        print("   E1 by lag", np.round(m_["E1"], 3), "SW by lag", np.round(m_["SW"], 3), flush=True)
    out[scheme] = res
json.dump(out, open(os.path.join(OUT, "results_turn_vs_slip.json"), "w"), indent=1, default=str)
