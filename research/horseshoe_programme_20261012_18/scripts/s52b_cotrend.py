#!/usr/bin/env python3
"""POST HOC (flagged) follow-up to PREREG 36: do one-glyph neighbours trend in the SAME direction along the order more than far pairs?"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "s52_respelling_within_scribe.py")).read().split("R = {}\nfor label")[0])
def Sstat(t, P, thr):
    def rate(pr):
        a, b = t[pr[0]], t[pr[1]]; return np.mean(((a <= -thr) & (b <= -thr)) | ((a >= thr) & (b >= thr)))
    return rate(P["e1"]) - rate(P["far"])
out = {}
for label, fn in (("ZL", tokens_zl), ("TTLI", tokens_tt)):
    raw = fn(); toksH = [Counter({atoms(w): n for w, n in raw[i].items()}) if label == "ZL" else raw[i] for i in H]
    P = prepare_from(toksH); rng = np.random.default_rng(SEED + 21)
    for thr in (0.3, 0.4):
        obs = Sstat(trends(P["RF"], posH), P, thr); null = np.array([Sstat(trends(P["RF"], rng.permutation(posH)), P, thr) for _ in range(2000)])
        z = (obs - null.mean()) / null.std(ddof=1); out[f"{label}_{thr}"] = dict(S=float(obs), null_mean=float(null.mean()), null_sd=float(null.std(ddof=1)), z=float(z))
        print(label, thr, "same-direction excess %.4f null %.4f sd %.4f z %.2f" % (obs, null.mean(), null.std(ddof=1), z), flush=True)
json.dump(out, open(os.path.join(OUT, "results_cotrend_posthoc.json"), "w"), indent=1)
