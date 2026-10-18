#!/usr/bin/env python3
"""Stage 20 (PREREG 7): F1 face test and F5 conjugate coherence on every distinct transliteration."""
import json, os, pickle, re, sys
from collections import Counter
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/horseshoe/scripts")
import lib_methods as L
ROOT, SLIM = sys.argv[1], sys.argv[2]; D = os.path.join(ROOT, "data")
S = pickle.load(open(os.path.join(D, "s14_features.pkl"), "rb")); Fall = S["F"]
d = json.load(open(SLIM)); pages = d["pages"]
alias = {"f101r1": "f101r", "f101v2": "f101v"}   # our names -> slim names, if needed
def slim_name(f): return f if f in pages else alias.get(f, f)
ids = [t["id"] for t in d["transcribers"]]

def page_lines(tid, f):
    lines = pages.get(slim_name(f), {})
    out = []
    for ln in sorted(lines, key=lambda x: int(x) if x.isdigit() else 99999):
        t = lines[ln]["t"].get(tid, "")
        ws = [w for w in re.split(r"[\s.]+", t) if w]
        if ws: out.append(ws)
    return out

# dedupe exact duplicates over all folios
sig = {}
for tid in ids:
    s = tuple((f, ln, pages[f][ln]["t"].get(tid, "")) for f in sorted(pages) for ln in sorted(pages[f]))
    sig.setdefault(hash(s), []).append(tid)
groups = list(sig.values()); keep_ids = [g[0] for g in groups]
res = {"duplicate_groups": groups}

# standard non-central bifolia (from s18 table)
T = pd.read_csv(os.path.join(ROOT, "faces_per_sheet.csv")); T = T[~T.central]
rng = np.random.default_rng(20261012)

def run(tid):
    lines = {f: page_lines(tid, f) for f in Fall.index}
    have = np.array([sum(len(l) for l in lines[f]) >= 5 for f in Fall.index])
    F = Fall[have].copy(); folios = list(F.index); idx = {f: i for i, f in enumerate(folios)}
    cnts = []
    for f in folios:
        c = Counter()
        for ws in lines[f]:
            for w in ws:
                s = "^" + w + "$"; c.update(("w", a, b) for a, b in zip(s, s[1:]))
            s = "".join(ws); c.update(("l", a, b) for a, b in zip(s, s[1:]))
        cnts.append(c)
    vocab = sorted(set().union(*cnts)); vi = {v: i for i, v in enumerate(vocab)}
    M = np.zeros((len(folios), len(vocab)))
    for i, c in enumerate(cnts):
        for k, v in c.items(): M[i, vi[k]] = v
    wm = np.array([v[0] == "w" for v in vocab])
    Dm = L.js_matrix(L.profiles(M, wm))
    # F1
    Tk = T[[all(Fall.index[int(T.loc[r, c])] in idx for c in ("ar", "av", "br", "bv")) for r in T.index]]
    g = lambda c: np.array([idx[Fall.index[int(x)]] for x in Tk[c]])
    ar, av, br, bv = g("ar"), g("av"), g("br"), g("bv")
    Dd = (Dm[ar, br] + Dm[av, bv]) / 2 - (Dm[ar, bv] + Dm[av, br]) / 2
    null = (Dd[None, :] * rng.choice([1, -1], size=(100000, len(Dd)))).mean(1)
    f1 = dict(n_sheets=int(len(Dd)), mean_D=float(Dd.mean()), z=float(Dd.mean() / null.std()),
              p=float((np.sum(null >= Dd.mean()) + 1) / 100001), closer=int((Dd > 0).sum()))
    # F5 conjugate coherence with leaf permutation within quire x scribe
    n = len(F); iu = np.triu_indices(n, 1); dd = Dm[iu]
    sheet, leaf_id, strat, leafnum = F.sheet.values, F.leaf_id.values, F.stratum.values, F.leaf.astype(str).values
    def mc(sh):
        m = (sh[iu[0]] == sh[iu[1]]) & (leafnum[iu[0]] != leafnum[iu[1]]); return dd[m].mean()
    obs = mc(sheet); nul = np.array([mc(L.leaf_permute(sheet, leaf_id, strat, rng)) for _ in range(2000)])
    f5 = dict(n_pages=n, obs=float(obs), null_mean=float(nul.mean()), null_sd=float(nul.std()),
              z=float((nul.mean() - obs) / nul.std()), pct_lower=float(100 * (nul.mean() - obs) / nul.mean()))
    return dict(tokens=int(sum(sum(len(l) for l in lines[f]) for f in folios)), alphabet_size=len({c for f in folios for ws in lines[f] for w in ws for c in w}),
                F1=f1, F5=f5)

rows = {}
for tid in keep_ids:
    nsheets = sum(all(sum(len(l) for l in page_lines(tid, Fall.index[int(T.loc[r, c])])) >= 5 for c in ("ar", "av", "br", "bv")) for r in T.index)
    if nsheets < 25:
        rows[tid] = dict(skipped=f"covers {nsheets}/33 sheets"); continue
    rows[tid] = run(tid); print(tid, rows[tid])
res["per_transliteration"] = rows
ran = {k: v for k, v in rows.items() if "F1" in v}
gate = ran.get("TTVE", {}).get("F1", {})
res["gate_TTVE"] = dict(D=gate.get("mean_D"), pass_=bool(gate and gate["mean_D"] > 0 and abs(gate["mean_D"] - 0.0077) <= 0.2 * 0.0077))
Ds = np.array([v["F1"]["mean_D"] for v in ran.values()]); zs = np.array([v["F1"]["z"] for v in ran.values()])
frac = float((Ds > 0).mean()); medz = float(np.median(zs))
res["summary"] = dict(n=len(ran), frac_D_positive=frac, median_z=medz,
                      F5_median_z=float(np.median([v["F5"]["z"] for v in ran.values()])),
                      F5_frac_positive=float(np.mean([v["F5"]["z"] > 0 for v in ran.values()])))
res["verdict"] = ("VOID (gate failed)" if not res["gate_TTVE"]["pass_"] else
                  "CONSISTENT" if (frac >= 0.8 and medz >= 1.5) else
                  "INCONSISTENT" if (frac < 0.6 or medz < 0.5) else "NOT RESOLVED")
json.dump(res, open(os.path.join(ROOT, "results_translits.json"), "w"), indent=1, default=str)
print(json.dumps({k: res[k] for k in ("duplicate_groups", "gate_TTVE", "summary", "verdict")}, indent=1, default=str))
