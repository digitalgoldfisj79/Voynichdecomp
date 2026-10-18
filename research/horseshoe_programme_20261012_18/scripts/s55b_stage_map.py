#!/usr/bin/env python3
"""Stage map from combined-uncertainty refits (s55) vs his page. Writes s55_out/stage_map.json, stage_map_*.csv."""
import csv, glob, json, pickle
import numpy as np
OUT = "/home/claude/horseshoe/s55_out"
d = json.load(open("/home/claude/ext/voynich-clusters/chronology-stability/replication/solve/botanical_enrichment/team_golf/golf_rf1b_stability_20261016/individuals/data.json"))
s = [s for s in d["settings"] if s["rep"] == "ZL" and s["cohort"] == "pool50" and s["penalty"] == 0.01][0]; meta = s["meta"]; ids = [m["id"] for m in meta]; n = len(ids)
grp = [m["group"] for m in meta]; words = [m["words"] for m in meta]
def pair_frac(fits):
    bef = np.zeros((n, n)); tie = np.zeros((n, n)); tog = np.zeros((n, n))
    for t, mask in fits:
        j = mask[:, None] & mask[None, :]; dl = t[:, None] - t[None, :]
        bef += j & (dl < -1e-7); tie += j & (np.abs(dl) <= 1e-7); tog += j
    return (bef + .5 * tie) / np.maximum(tog, 1)
recs = [pickle.load(open(p, "rb")) for p in sorted(glob.glob(OUT + "/stages/rep_*.pkl"))]
best, allf = [], []
for r in recs:
    el = [f for f in r["fits"] if f["early_ok"]]
    if not el: continue
    best.append((min(el, key=lambda f: (f["objective"], f["start"]))["positions"], r["mask"]))
    allf += [(f["positions"], r["mask"]) for f in el]
def his_frac(scope):
    sc = s["scopes"][scope]; F = np.full((n, n), .5); iu = np.triu_indices(n, 1)
    b, t, tot = map(np.array, (sc["before"], sc["ties"], sc["together"])); f = (b + .5 * t) / tot
    F[iu] = f; F[(iu[1], iu[0])] = 1 - f; return F
def stages(F):
    after = np.array([np.mean([F[j, i] for j in range(n) if j != i]) for i in range(n)])   # share of others before i
    order = np.argsort(after); blocks, cur = [], [order[0]]
    for k in range(1, n):
        left, right = order[:k], order[k:]
        settled = all(F[i, j] >= .95 for i in left for j in right)
        if settled: blocks.append(cur); cur = [order[k]]
        else: cur.append(order[k])
    blocks.append(cur); return blocks
iu = np.triu_indices(n, 1); same = np.array(grp)[iu[0]] == np.array(grp)[iu[1]]
res = {}
for name, F in (("his_best", his_frac("best")), ("his_all", his_frac("all")), ("combined_best", pair_frac(best)), ("combined_all", pair_frac(allf))):
    bl = stages(F); dec = np.maximum(F[iu], 1 - F[iu]) >= .95
    res[name] = dict(n_stages=len(bl), stage_sizes=[len(b) for b in bl], decisive_all=float(dec.mean()), decisive_same_group=float(dec[same].mean()),
                     stages=[[ids[i] for i in b] for b in bl], stage_groups=[sorted({grp[i] for i in b}) for b in bl])
    print(f"{name:14s} stages {len(bl):2d} sizes {[len(b) for b in bl]} | decisive all {dec.mean():.3f} same-group {dec[same].mean():.3f}")
res["n_fits"] = dict(best=len(best), all=len(allf))
json.dump(res, open(OUT + "/stage_map.json", "w"), indent=1)
with open(OUT + "/stage_map_combined_best.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["stage", "bifolium", "group", "words"])
    for k, b in enumerate(res["combined_best"]["stages"], 1):
        for i in b: w.writerow([k, i, grp[ids.index(i)], words[ids.index(i)]])
print("fits:", res["n_fits"])
