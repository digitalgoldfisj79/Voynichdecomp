"""Depth-matched null for eo777 thread-6299 crease adjacency. Usage: python3 -I null_depth.py <golf-v21-assignments.csv> <outdir>"""
import sys, os, json, pickle, re, numpy as np, pandas as pd
csv, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(20261113); NDRAW = 20000
e = pd.read_csv(csv)
paths = ["bifolium_path_rank","bifolium_path_2_rank","bifolium_path_3_rank","bifolium_path_4_rank"]
S = e.groupby("sheet")[paths].first().dropna()
S = S[S.index.str.match(r"^[A-H]:\d+-\d+$")]
assert len(S) == 30, len(S)
def lohi(s): q, r = s.split(":"); a, b = map(int, r.split("-")); return q, a, b
depth = {}
for s in S.index:
    q, a, b = lohi(s)
    depth[s] = sum(1 for t in S.index if lohi(t)[0] == q and lohi(t)[1] > a)
orders = {p: list(S[p].sort_values().index) for p in paths}
chart = ("A:1-8 D:27-30 F:42-47 G:49-56 A:2-7 D:25-32 C:20-21 E:35-38 A:4-5 D:28-29 B:10-15 E:36-37 B:9-16 "
         "C:19-22 B:11-14 B:12-13 C:18-23 F:44-45 G:52-53 A:3-6 C:17-24 G:51-54 H:58-65 D:26-31 H:57-66 "
         "F:41-48 F:43-46 E:34-39 E:33-40 G:50-55").split()
assert sorted(chart) == sorted(S.index); orders["chart_transcribed"] = chart
SETS = {"BOTTOM-N6": ["A:4-5","B:10-15","C:20-21","D:25-32","D:28-29","E:36-37"],
        "BOTTOM-N5": ["A:4-5","B:10-15","C:20-21","D:28-29","E:36-37"],
        "BOTTOM-L9": ["A:4-5","B:10-15","C:20-21","D:25-32","D:28-29","E:36-37","B:9-16","D:27-30","G:52-53"],
        "FORE-F4":   ["E:33-40","E:34-39","G:50-55","H:57-66"],
        "R1-CENTRES":[s for s in S.index if depth[s] == 0]}
byd = {}
for s, d in depth.items(): byd.setdefault(d, []).append(s)
def stats(r): r = np.sort(r); return r[-1]-r[0]+1, np.abs(r[:,None]-r[None,:]).sum()/(len(r)*(len(r)-1))
rows = []
for pname, order in orders.items():
    rk = {s: i+1 for i, s in enumerate(order)}; univ = np.array(list(rk.values()))
    for sname, st in SETS.items():
        obs = stats(np.array([rk[s] for s in st])); k = len(st)
        nu = np.array([stats(rng.choice(univ, k, replace=False)) for _ in range(NDRAW)])
        nd = []
        for _ in range(NDRAW):
            pick = []
            for d in set(depth[s] for s in st):
                m = sum(depth[s] == d for s in st); pick += list(rng.choice(byd[d], m, replace=False))
            nd.append(stats(np.array([rk[s] for s in pick])))
        nd = np.array(nd)
        rec = dict(path=pname, set=sname, k=k, depths="".join(str(depth[s]) for s in st),
                   obs_span=int(obs[0]), obs_mpd=round(float(obs[1]),2))
        for tag, n in (("U", nu), ("D", nd)):
            rec[f"p_{tag}"] = float(((n[:,0] <= obs[0]).sum()+1)/(NDRAW+1))
            rec[f"null{tag}_mean"] = round(float(n[:,0].mean()),2); rec[f"null{tag}_sd"] = round(float(n[:,0].std()),2)
            rec[f"z_{tag}"] = round(float((n[:,0].mean()-obs[0])/n[:,0].std()),2) if n[:,0].std()>0 else None
            rec[f"pmpd_{tag}"] = float(((n[:,1] <= obs[1]).sum()+1)/(NDRAW+1))
        rec["D_p1_span"] = int(np.percentile(nd[:,0], 1)); rec["D_min_span"] = int(nd[:,0].min())
        rec["D_p_at_perfect"] = float(((nd[:,0] <= k).sum()+1)/(NDRAW+1))
        rows.append(rec)
R = pd.DataFrame(rows); R.to_csv(f"{out}/null_depth_results.csv", index=False)
pickle.dump(dict(depth=depth, orders=orders, sets=SETS, results=R), open(f"{out}/null_depth.pkl","wb"))
pd.set_option("display.width", 250)
print(pd.Series(depth).value_counts().sort_index().to_dict(), "bifolia per depth")
print(R[["path","set","k","depths","obs_span","nullU_mean","nullU_sd","p_U","nullD_mean","nullD_sd","z_D","p_D","pmpd_D","D_p_at_perfect"]].to_string(index=False))
