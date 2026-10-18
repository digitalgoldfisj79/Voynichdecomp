#!/usr/bin/env python3
"""PREREG 19 (RUNNING_RESULTS_horseshoe.md, 2026-10-15): context inheritance test.
Does the later form inherit the earlier form's contexts? chy->chey (npcompl33t) vs chey->chedy (1-edit / Timm).
Context = previous + next word within line (top-150 pooled types, else OTHER; EDGE at line ends).
S(X->Y) = JSD(ctx of X in EARLY, ctx of Y in LATE), both rarefied to n occurrences, mean of 300 draws.
Null = same statistic with Y replaced by every eligible LATE control type. Results pickled per comparison.
usage: python3 -I s33_context_inheritance.py DATA_DIR OUT_DIR
"""
import json, os, pickle, sys
from collections import Counter
import numpy as np, pandas as pd

DATA, OUT = sys.argv[1], sys.argv[2]; os.makedirs(os.path.join(OUT, "cmp"), exist_ok=True)
SEED, DRAWS, TOPK = 20261015, 300, 150
W = pd.read_csv(os.path.join(DATA, "words_by_page.csv.gz"))
C = pd.read_csv(os.path.join(DATA, "eo777_chain_pages.csv"))
W["stage"] = W.folio.map(dict(zip(C.page, C.group)))
W = W[W.line_num > 0].sort_values(["folio", "line_num", "word_index"])
EARLY, LATE = {"1a", "1b", "1.5a"}, {"3c", "4a", "4c"}
W = W[W.stage.isin(EARLY | LATE)].copy()
W["period"] = np.where(W.stage.isin(EARLY), "E", "L")
top = set(w for w, _ in Counter(W.word).most_common(TOPK))
cat = lambda w: w if w in top else "OTHER"

# occurrences: (period, word) -> list of (prev_cat, next_cat)
occ = {}
for (f, l), g in W.groupby(["folio", "line_num"], sort=False):
    ws, per = list(g.word), list(g.period)
    for i, w in enumerate(ws):
        pv = cat(ws[i - 1]) if i > 0 else "EDGE"; nx = cat(ws[i + 1]) if i + 1 < len(ws) else "EDGE"
        occ.setdefault((per[i], w), []).append((pv, nx))
feats = sorted({"P:" + p for v in occ.values() for p, _ in v} | {"N:" + n for v in occ.values() for _, n in v})
fi = {f: i for i, f in enumerate(feats)}

def jsd(a, b):
    a = a / a.sum(); b = b / b.sum(); m = (a + b) / 2
    def kl(x): nz = x > 0; return float((x[nz] * np.log2(x[nz] / m[nz])).sum())
    return 0.5 * kl(a) + 0.5 * kl(b)

def vec(pairs, idx, which):
    v = np.zeros(len(feats))
    for k in idx:
        p, n = pairs[k]
        if which in ("both", "prev"): v[fi["P:" + p]] += 1
        if which in ("both", "next"): v[fi["N:" + n]] += 1
    return v

def S(x_key, y_key, n, which, rng):
    a, b = occ[x_key], occ[y_key]; vals = []
    for _ in range(DRAWS):
        ia = rng.choice(len(a), n, replace=False); ib = rng.choice(len(b), n, replace=False)
        vals.append(jsd(vec(a, ia, which), vec(b, ib, which)))
    return float(np.mean(vals))

def controls(period, n, y, chsh):
    return sorted(w for (p, w), v in occ.items() if p == period and len(v) >= n and w.endswith("y") and w != y
                  and (not chsh or w[:2] in ("ch", "sh")))

def compare(name, x, xp, y, yp, n, which="both", chsh=False):
    path = os.path.join(OUT, "cmp", name + ".pkl")
    if os.path.exists(path): return pickle.load(open(path, "rb"))
    rng = np.random.default_rng(SEED)
    ctrl = [z for z in controls(yp, n, y, chsh) if z != x or xp != yp]
    s_obs = S((xp, x), (yp, y), n, which, rng)
    s_null = {z: S((xp, x), (yp, z), n, which, rng) for z in ctrl}
    nv = np.array(list(s_null.values()))
    r = dict(name=name, x=x, x_period=xp, y=y, y_period=yp, n=n, context=which, ch_sh_controls=chsh,
             S=s_obs, null_mean=float(nv.mean()), null_sd=float(nv.std(ddof=1)), n_controls=len(nv),
             z=float((nv.mean() - s_obs) / nv.std(ddof=1)), pct=float((nv <= s_obs).mean()),
             rank=int((nv < s_obs).sum()) + 1, closest=sorted(s_null.items(), key=lambda t: t[1])[:5])
    r["supported"] = bool(r["pct"] <= 0.05 and r["z"] >= 2)
    pickle.dump(r, open(path + ".tmp", "wb")); os.replace(path + ".tmp", path)
    print(name, {k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in ("S", "null_mean", "null_sd", "z", "rank", "n_controls", "supported")}, flush=True)
    return r

R = {}
R["P1_chy_to_chey"] = compare("P1_chy_to_chey", "chy", "E", "chey", "L", 40)
R["P2_chey_to_chedy"] = compare("P2_chey_to_chedy", "chey", "E", "chedy", "L", 40)
R["P1_next_only"] = compare("P1_next_only", "chy", "E", "chey", "L", 40, "next")
R["P2_next_only"] = compare("P2_next_only", "chey", "E", "chedy", "L", 40, "next")
R["P1_chsh_controls"] = compare("P1_chsh_controls", "chy", "E", "chey", "L", 40, "both", True)
R["P2_chsh_controls"] = compare("P2_chsh_controls", "chey", "E", "chedy", "L", 40, "both", True)
R["SYN_late_chey_chedy"] = compare("SYN_late_chey_chedy", "chey", "L", "chedy", "L", 40)
R["SYN_early_chy_chey"] = compare("SYN_early_chy_chey", "chy", "E", "chey", "E", 30)
p1, p2 = R["P1_chy_to_chey"]["supported"], R["P2_chey_to_chedy"]["supported"]
verdict = ("NPCOMPL_ROUTE_PREFERRED" if p1 and not p2 else "ONE_EDIT_ROUTE_PREFERRED" if p2 and not p1
           else "BOTH_SUPPORTED_NO_DISCRIMINATION" if p1 and p2 else "NEITHER_SUPPORTED_NO_DISCRIMINATION")
json.dump(dict(prereg="PREREG 19", verdict=verdict, results=R), open(os.path.join(OUT, "results_context_inheritance.json"), "w"), indent=1)
print("VERDICT", verdict)
