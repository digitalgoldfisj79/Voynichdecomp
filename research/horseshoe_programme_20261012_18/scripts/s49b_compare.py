#!/usr/bin/env python3
"""PREREG 34 analysis: decisive-pair share and position spans, TEXT resampling (ours) vs his POINT ensemble (best scope, ZL pool50 l0.01)."""
import glob, json, pickle, sys
import numpy as np
from scipy.stats import spearmanr
OUT = sys.argv[1]; DATA = sys.argv[2]
d = json.load(open(DATA)); s = [s for s in d["settings"] if s["rep"] == "ZL" and s["cohort"] == "pool50" and s["penalty"] == 0.01][0]
meta = s["meta"]; ids = [m["id"] for m in meta]; n = len(ids); iu = np.triu_indices(n, 1)
def his(scope):
    sc = s["scopes"][scope]; b, t, tot = map(np.array, (sc["before"], sc["ties"], sc["together"]))
    return (b + .5 * t) / tot, np.array([q[2] - q[0] for q in sc["quantiles"]]), np.array([q[1] for q in sc["quantiles"]])
reps = [pickle.load(open(p, "rb")) for p in sorted(glob.glob(OUT + "/stages/rep_*.pkl"))]
rep0 = [r for r in reps if r["b"] == 0][0]; reps = [r for r in reps if r["b"] > 0 and r["best"] is not None]
P = np.array([r["best"]["positions"] for r in reps])                     # (B, 50)
delta = P[:, :, None] - P[:, None, :]
frac = ((delta < -1e-7).sum(0) + .5 * (np.abs(delta) <= 1e-7).sum(0)) / len(P)   # i before j
ours = frac[iu]; span_ours = np.percentile(P, 95, 0) - np.percentile(P, 5, 0)
out = dict(n_reps=len(P), n_failed=100 - len(P))
for scope in ("best", "all"):
    hf, hspan, hmed = his(scope)
    dec_h = float(np.mean(np.maximum(hf, 1 - hf) >= .95)); dec_o = float(np.mean(np.maximum(ours, 1 - ours) >= .95))
    out[scope] = dict(his_decisive_share=dec_h, text_decisive_share=dec_o, ratio=dec_o / dec_h,
                      his_median_span=float(np.median(hspan)), text_median_span=float(np.median(span_ours)),
                      reversals_vs_his_majority=float(np.mean((hf - .5) * (ours - .5) < 0)))
    print(scope, json.dumps({k: round(v, 3) for k, v in out[scope].items()}))
hf, _, hmed = his("best")
out["gate_spearman_rep0_vs_his_best_median"] = float(spearmanr(rep0["best"]["positions"], hmed)[0])
rs = [spearmanr(p, rep0["best"]["positions"])[0] for p in P]
out["rep_vs_fulldata_spearman_quantiles"] = np.percentile(rs, [5, 50, 95]).tolist()
starts = np.bincount([r["best"]["start"] for r in reps], minlength=8).tolist(); out["best_start_counts"] = starts
# group order: 3a vs 3b median, 1a vs 1b
lab = np.array([m["group"] for m in meta])
for a, b in (("3a", "3b"), ("4a", "4c")):
    out[f"{a}_before_{b}_share"] = float(np.mean([np.median(p[lab == a]) < np.median(p[lab == b]) for p in P]))
# focal: I:67-68 after each 2a pool
i = ids.index("I:67-68"); foc = {}
for j, m in enumerate(meta):
    if m["group"] == "2a" and j != i: foc[m["id"]] = float(1 - frac[i, j])   # fraction I:67-68 after j
out["focal_I6768_after_2a_text"] = foc
# per-pool table
rows = sorted([(ids[k], meta[k]["group"], float(np.median(P[:, k])), float(span_ours[k]), float(_)) for k, _ in enumerate(his("best")[1])], key=lambda r: r[2])
out["per_pool"] = rows
json.dump(out, open(OUT + "/results_text_resampled.json", "w"), indent=1)
v = out["best"]["ratio"]
print("VERDICT", "FALSIFIES (stability overstated)" if v < .8 else "CONFIRMS" if v >= .95 else "PARTIAL", "ratio %.3f" % v)
print("gate", round(out["gate_spearman_rep0_vs_his_best_median"], 3), "| rep-vs-full Spearman 5/50/95%", np.round(out["rep_vs_fulldata_spearman_quantiles"], 3), "| best starts", starts)
print("3a<3b share", out["3a_before_3b_share"], "| 4a<4c share", out["4a_before_4c_share"])
print("I:67-68 after 2a pools (text):", {k: round(v, 2) for k, v in foc.items()})
