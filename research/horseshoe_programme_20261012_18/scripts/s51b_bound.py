#!/usr/bin/env python3
"""PREREG 35b: bound the secondary result with flexible transition controls and rarefied vocabulary similarity."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "s51_order_vs_vocabulary.py")).read().split("R = {}")[0]
exec(src)
def mat(name):
    with open(os.path.join(DATA, f"pairwise/shared-{name}-js.csv")) as f: rr = list(csv.reader(f))
    c = rr[0][1:]; d = {r[0]: r[1:] for r in rr[1:]}
    return np.array([[float(d[a][c.index(b)]) for b in pid] for a in pid])
WJ, LJ = mat("word"), mat("line")
def analyse2(S, size, label):
    iu = np.triu_indices(len(ids), 1); same = strata[iu[0]] == strata[iu[1]]; a, b = iu[0][same], iu[1][same]; y = S[a, b]
    td = TD[a, b]
    X = np.column_stack([np.ones(len(a)), np.log(size[a]) + np.log(size[b]), WJ[a, b], LJ[a, b], td, td ** 2, rankdata(td)])
    res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]; rr = rankdata(res)
    stat = lambda p_: float(np.corrcoef(rankdata(np.abs(p_[a] - p_[b])), rr)[0, 1])
    obs = stat(pos); rng = np.random.default_rng(SEED); null = []
    for _ in range(NPERM):
        q = pos.copy()
        for g in np.unique(strata):
            k = np.flatnonzero(strata == g)
            if len(k) > 1: q[k] = rng.permutation(q[k])
        null.append(stat(q))
    null = np.array(null); z = (obs - null.mean()) / null.std(ddof=1)
    print(label, "rho %.3f null %.3f sd %.3f z %.2f p %.4f" % (obs, null.mean(), null.std(ddof=1), z, np.mean(null <= obs)), flush=True)
    return dict(rho=obs, null_sd=float(null.std(ddof=1)), z=float(z))
def rarefied(toks, n=120, draws=50):
    rng = np.random.default_rng(SEED + 1); acc = 0
    for _ in range(draws):
        sub = []
        for c in toks:
            arr = np.repeat(list(c.keys()), list(c.values())); sub.append(Counter(rng.choice(arr, size=n, replace=False)))
        S, _ = similarity(sub); acc = acc + S
    return acc / draws
out = {}
for label, fn in (("ZL", tokens_zl), ("TTLI", tokens_tt)):
    toks = fn(); S, size = similarity(toks)
    out[label + "_flexible"] = analyse2(S, size, label + " (a) flexible transition control")
    out[label + "_rarefied"] = analyse2(rarefied(toks), np.full(len(toks), 120.), label + " (b) + rarefied to 120")
ok = all(out[k]["z"] <= -2 for k in ("ZL_rarefied", "TTLI_rarefied"))
out["verdict"] = "SECONDARY SURVIVES BOUNDING" if ok else "SECONDARY DOES NOT SURVIVE BOUNDING"
json.dump(out, open(os.path.join(OUT, "results_bound.json"), "w"), indent=1); print(out["verdict"])
