#!/usr/bin/env python3
"""PREREG 37: Pelling's physical anchors vs npcompl33t's order within herbal scribe 1 (ranks among 24 bifolia)."""
import glob, json, pickle, sys
import numpy as np
OUT = "/home/claude/horseshoe/s50_out"
d = json.load(open("/home/claude/ext/voynich-clusters/chronology-stability/replication/solve/botanical_enrichment/team_golf/golf_rf1b_stability_20261016/individuals/data.json"))
s = [s for s in d["settings"] if s["rep"] == "ZL" and s["cohort"] == "pool50" and s["penalty"] == 0.01][0]; ids = [m["id"] for m in s["meta"]]
med = np.array([q[1] for q in s["scopes"]["best"]["quantiles"]])
H = ['A:1-8','A:2-7','A:3-6','A:4-5','B:10-15','B:11-14','B:12-13','B:9-16','C:17-24','C:18-23','C:19-22','C:20-21','D:25-32','D:27-30','D:28-29','E:35-38','E:36-37','F:42-47','F:44-45','G:49-56','G:51-54','G:52-53','O:87-90','Q:93-96']
hix = [ids.index(h) for h in H]
V = [("E:35-38","E:36-37"),("E:36-37","B:9-16"),("B:9-16","B:10-15")]
T = [("A:2-7","A:3-6"),("A:3-6","A:4-5"),("B:10-15","B:11-14"),("C:19-22","C:20-21")]
rng = np.random.default_rng(20261020)
def ranks(p): return np.argsort(np.argsort(p))
def stat(r, pairs): return float(np.mean([abs(r[H.index(a)] - r[H.index(b)]) for a, b in pairs]))
def ztest(r, pairs, n=100000):
    obs = stat(r, pairs); null = np.array([stat(rng.permutation(r), pairs) for _ in range(n)])
    return obs, float(null.mean()), float(null.std(ddof=1)), float((obs - null.mean()) / null.std(ddof=1)), float(np.mean(null <= obs))
r0 = ranks(med[hix]); out = {"ranks": {h: int(r0[i]) for i, h in enumerate(H)}}
for name, pr in (("V", V), ("T", T), ("V+T", V + T)):
    o, m, sd, z, p = ztest(r0, pr); out[name] = dict(mean_abs_rank_diff=o, null_mean=m, null_sd=sd, z=z, p=p)
    print(name, "mean|rank diff| %.2f null %.2f sd %.2f z %.2f p %.4f" % (o, m, sd, z, p), " pairs:", [(a, b, int(abs(r0[H.index(a)] - r0[H.index(b)]))) for a, b in pr])
reps = [pickle.load(open(p, "rb"))["best"]["positions"] for p in sorted(glob.glob(OUT + "/stages/rep_*.pkl"))[1:]]
zs = [ztest(ranks(np.asarray(p)[hix]), V + T, n=5000)[3] for p in reps]
out["bootstrap_share_z_le_-2"] = float(np.mean(np.array(zs) <= -2)); out["bootstrap_z_quantiles"] = np.percentile(zs, [5, 50, 95]).tolist()
print("bootstrap reps: share z <= -2 %.2f, z 5/50/95%% %s" % (out["bootstrap_share_z_le_-2"], np.round(out["bootstrap_z_quantiles"], 2)))
# power: true order with anchor pairs adjacent, noise SD 1.07 ranks
pw = []
for _ in range(500):
    true = rng.permutation(24).astype(float)
    # force anchor chains adjacent: chain E:35-38, E:36-37, B:9-16, B:10-15, B:11-14 and A:2-7,A:3-6,A:4-5 and C:19-22,C:20-21
    chains = [["E:35-38","E:36-37","B:9-16","B:10-15","B:11-14"], ["A:2-7","A:3-6","A:4-5"], ["C:19-22","C:20-21"]]
    order = [h for h in rng.permutation(H) if not any(h in c for c in chains)]
    for c in chains: order.insert(int(rng.integers(len(order) + 1)), c)
    flat = [x for item in order for x in (item if isinstance(item, list) else [item])]
    tp = np.array([flat.index(h) for h in H], float) + rng.normal(0, 1.07, 24)
    pw.append(ztest(ranks(tp), V + T, n=2000)[3])
out["power_z_le_-2"] = float(np.mean(np.array(pw) <= -2)); print("power (true adjacency, noise SD 1.07): P(z <= -2) %.2f" % out["power_z_le_-2"])
vt = out["V+T"]["z"]
out["verdict"] = "CONFIRMS" if vt <= -2 and out["bootstrap_share_z_le_-2"] >= .8 else ("FALSIFIES" if abs(vt) < 1 and out["power_z_le_-2"] >= .8 else "UNRESOLVED")
json.dump(out, open("/home/claude/horseshoe/s53_out_anchors.json", "w"), indent=1); print("VERDICT", out["verdict"])
