#!/usr/bin/env python3
"""PREREG 26: volvelle with ring wear/replacement + variable ring size vs PREREG 25 self-citation; ZL and TTLI locked test.
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
def load_pages(tid):
    out = {}
    for fid in sorted(D["pages"]):
        pg = D["pages"][fid]; ls = []
        for ln in sorted(pg, key=lambda x: int(x) if x.isdigit() else 99999):
            u = pg[ln].get("u", "")
            if len(u) >= 2 and u[1] == "P":
                t = [atoms(x) for x in pg[ln]["t"].get(tid, "").split() if re.fullmatch(r"[a-z]+", x)]
                if len(t) >= 2: ls.append(t)
        if ls: out[fid] = ls
    return out
pages = load_pages("ZLZI"); pages_tt = load_pages("TTLI")
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

