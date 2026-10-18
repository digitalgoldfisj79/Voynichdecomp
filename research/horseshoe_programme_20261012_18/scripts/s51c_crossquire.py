#!/usr/bin/env python3
"""Dependency check for Super Grammar SG23 (post hoc, flagged): restrict PREREG 35's primary test to same-stratum pairs in DIFFERENT physical quires."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "s51_order_vs_vocabulary.py")).read().split("R = {}")[0])
quire = np.array([i.split(":")[0] for i in ids])
out = {}
for label, fn in (("ZL", tokens_zl), ("TTLI", tokens_tt)):
    S, size = similarity(fn())
    iu = np.triu_indices(len(ids), 1); m = (strata[iu[0]] == strata[iu[1]]) & (quire[iu[0]] != quire[iu[1]]); a, b = iu[0][m], iu[1][m]
    y = S[a, b]; X = np.column_stack([np.ones(len(a)), np.log(size[a]) + np.log(size[b])]); res = rankdata(y - X @ np.linalg.lstsq(X, y, rcond=None)[0])
    stat = lambda p_: float(np.corrcoef(rankdata(np.abs(p_[a] - p_[b])), res)[0, 1])
    obs = stat(pos); rng = np.random.default_rng(SEED + 3); null = []
    for _ in range(NPERM):
        q = pos.copy()
        for g in np.unique(strata):
            k = np.flatnonzero(strata == g)
            if len(k) > 1: q[k] = rng.permutation(q[k])
        null.append(stat(q))
    null = np.array(null); z = (obs - null.mean()) / null.std(ddof=1)
    out[label] = dict(pairs=int(m.sum()), rho=obs, null_mean=float(null.mean()), null_sd=float(null.std(ddof=1)), z=float(z))
    print(label, "cross-quire same-stratum pairs %d rho %.3f null %.3f sd %.3f z %.2f" % (m.sum(), obs, null.mean(), null.std(ddof=1), z), flush=True)
json.dump(out, open(os.path.join(OUT, "results_crossquire.json"), "w"), indent=1)
