#!/usr/bin/env python3
"""PREREG 24 (RUNNING_RESULTS_horseshoe.md, 2026-10-16): fitted generator comparison on the within-line repeat profile.
M0 independent, M1 self-citation (Timm-style family), M2 volvelle (Cham-style). Fit on half the pages, score held-out.
Per-split results pickled atomically to OUT/splits/. usage: python3 -I s39_generator_profile_fit.py VOYNICHDECOMP OUT
"""
import itertools, json, os, pickle, re, sys
from collections import Counter
import numpy as np

REPO, OUT = sys.argv[1:3]; os.makedirs(os.path.join(OUT, "splits"), exist_ok=True)
SEED, LAGS, NBOOT = 20261016, range(1, 7), 200
ATOMS = [("ckh", "K"), ("cth", "T"), ("cph", "P"), ("cfh", "F"), ("ch", "C"), ("sh", "S")]
def atoms(w):
    for a, b in ATOMS: w = w.replace(a, b)
    return w
SLOTS = [lambda a: a, lambda a: a[:1], lambda a: a[-1:], lambda a: a[-2:]]
SLOT_NAMES = ["WHOLE", "FIRST", "LAST", "LAST2"]

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

def profile_parts(lines):
    """per line: obs and exp counts for each slot x lag -> arrays (nlines, 24)"""
    O = np.zeros((len(lines), 24)); E = np.zeros((len(lines), 24))
    for i, l in enumerate(lines):
        L = len(l)
        for s, f in enumerate(SLOTS):
            v = [f(w) for w in l]; c = Counter(v); pe = sum(n * (n - 1) for n in c.values()) / (L * (L - 1))
            for j, d in enumerate(LAGS):
                if L > d:
                    O[i, s * 6 + j] = sum(v[k] == v[k + d] for k in range(L - d)); E[i, s * 6 + j] = (L - d) * pe
    return O, E
def ratio(O, E): return O.sum(0) / np.maximum(E.sum(0), 1e-12)

def to_lines(stream, LL, rng):
    out, i = [], 0
    while i < len(stream):
        n = int(rng.choice(LL)); out.append(stream[i:i + n]); i += n
    return [l for l in out if len(l) >= 2]

# ---------------- generators
def gen_m0(train, n, rng, **_):
    w, p = train["uni"]; return list(rng.choice(w, size=n, p=p))
def gen_m1(train, n, rng, c, w, m):
    W, P = train["uni"]; A, AP = train["atoms"]; out = []
    draws = rng.choice(W, size=n, p=P)
    for k in range(n):
        if out and rng.random() < c:
            x = out[-int(rng.integers(1, min(w, len(out)) + 1))]
            if rng.random() < m and x:
                i = int(rng.integers(len(x))); x = x[:i] + rng.choice(A, p=AP) + x[i + 1:]
            out.append(x)
        else:
            out.append(draws[k])
    return out
def gen_m2(train, n, rng, s, b, V):
    parts = train["parts"]; vols = []
    for _ in range(V):
        rings = []
        for r in range(4):
            vals, pr = parts[r]; cells = list(rng.choice(vals, size=24, p=pr))
            for i in range(1, 24, 2):
                if rng.random() < b: cells[i] = ""
            rings.append(cells)
        vols.append((rings, [int(rng.integers(24)) for _ in range(4)]))
    out = []
    while len(out) < n:
        rings, pos = vols[int(rng.integers(V))]
        x = "".join(rings[r][pos[r]] for r in range(4))
        if x: out.append(x)
        for r in range(4):
            if rng.random() < s: pos[r] = (pos[r] + 1) % 24
    return out

def train_stats(lines):
    toks = [w for l in lines for w in l]; c = Counter(toks)
    W = list(c); P = np.array([c[x] for x in W], float); P /= P.sum()
    ac = Counter(a for w in toks for a in w); A = list(ac); AP = np.array([ac[a] for a in A], float); AP /= AP.sum()
    pc = [Counter() for _ in range(4)]
    for w in toks:
        a = list(w)
        pc[0][a[0]] += 1; pc[3][a[-1] if len(a) >= 2 else ""] += 1
        pc[2][a[-2] if len(a) >= 3 else ""] += 1; pc[1]["".join(a[1:-2]) if len(a) >= 4 else ""] += 1
    parts = []
    for cc in pc:
        v = list(cc); p = np.array([cc[x] for x in v], float); parts.append((v, p / p.sum()))
    return dict(uni=(W, P), atoms=(A, AP), parts=parts, LL=np.array([len(l) for l in lines]), ntok=len(toks))

GRIDS = {"M0": [{}],
         "M1": [dict(c=c, w=w, m=m) for c, w, m in itertools.product([0, .05, .1, .2, .3, .45], [1, 2, 3, 5, 8], [0, .3, .6])],
         "M2": [dict(s=s, b=b, V=V) for s, b, V in itertools.product([.3, .5, .7, .85, 1.0], [0, .25, .5, .75], [1, 2, 3, 5])]}
GENS = {"M0": gen_m0, "M1": gen_m1, "M2": gen_m2}

def gen_profile(model, prm, tr, ntok, reps, rng):
    rs = []
    for _ in range(reps):
        lines = to_lines(GENS[model](tr, ntok, rng, **prm), tr["LL"], rng); rs.append(ratio(*profile_parts(lines)))
    return np.mean(rs, 0)

def run_split(name, train_pages, test_pages):
    p = os.path.join(OUT, "splits", name + ".pkl")
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    rng = np.random.default_rng(SEED + (1 if name == "A" else 2))
    trl = [l for f in train_pages for l in pages[f]]; tel = [l for f in test_pages for l in pages[f]]
    tr = train_stats(trl)
    Otr, Etr = profile_parts(trl); Ote, Ete = profile_parts(tel)
    def boot_se(O, E):
        bs = [ratio(O[i], E[i]) for i in (rng.integers(len(O), size=len(O)) for _ in range(NBOOT))]
        return np.std(bs, 0, ddof=1)
    se_tr, se_te = boot_se(Otr, Etr), boot_se(Ote, Ete)
    target_tr, target_te = ratio(Otr, Etr), ratio(Ote, Ete)
    Dfun = lambda g, t, se: float(np.sum(((g - t) / np.maximum(se, 1e-9)) ** 2))
    res = {"n_train_lines": len(trl), "n_test_lines": len(tel), "target_test": target_te.tolist(), "se_test": se_te.tolist(), "models": {}}
    gens_te = {}
    for model, grid in GRIDS.items():
        best = None
        for prm in grid:
            d = Dfun(gen_profile(model, prm, tr, tr["ntok"], 2, rng), target_tr, se_tr)
            if best is None or d < best[0]: best = (d, prm)
        g = gen_profile(model, best[1], tr, sum(len(l) for l in tel), 5, rng); gens_te[model] = g
        res["models"][model] = dict(params=best[1], D_train=best[0], D_test=Dfun(g, target_te, se_te), profile_test=g.tolist())
        print(name, model, best[1], "D_train", round(best[0], 1), "D_test", round(res["models"][model]["D_test"], 1), flush=True)
    # paired bootstrap of held-out lines for D differences
    order = sorted(res["models"], key=lambda m: res["models"][m]["D_test"])
    bd = {m: [] for m in GRIDS}
    for _ in range(NBOOT):
        idx = rng.integers(len(Ote), size=len(Ote)); t = ratio(Ote[idx], Ete[idx])
        for m in GRIDS: bd[m].append(Dfun(gens_te[m], t, se_te))
    for m in GRIDS:
        if m == "M2": continue
        diff = np.array(bd[m]) - np.array(bd["M2"])
        res["models"][m]["D_minus_M2_boot_mean"] = float(diff.mean()); res["models"][m]["D_minus_M2_boot_sd"] = float(diff.std(ddof=1))
    res["order"] = order
    pickle.dump(res, open(p + ".tmp", "wb")); os.replace(p + ".tmp", p); return res

fids = sorted(pages); r = np.random.default_rng(SEED); perm = list(r.permutation(fids)); h = len(perm) // 2
S = {"A": run_split("A", perm[:h], perm[h:]), "B": run_split("B", perm[h:], perm[:h])}
def m2_wins(s):
    o = S[s]["order"]
    if o[0] != "M2": return False
    nb = o[1]; m = S[s]["models"][nb]
    return S[s]["models"][nb]["D_test"] - S[s]["models"]["M2"]["D_test"] > 2 * m["D_minus_M2_boot_sd"]
verdict = "CONFIRMS_VOLVELLE_ADVANTAGE" if m2_wins("A") and m2_wins("B") else "FALSIFIES_VOLVELLE_ADVANTAGE"
S["verdict"] = verdict; S["slot_names"] = SLOT_NAMES; S["lags"] = list(LAGS)
json.dump(S, open(os.path.join(OUT, "results_generator_profile_fit.json"), "w"), indent=1)
print("ORDER A", S["A"]["order"], "B", S["B"]["order"], "VERDICT", verdict)
