"""Bounding nulls: depth x Currier, depth x LFD hand. Usage: python3 -I null_bound.py <golf csv> <profiles.csv> <outdir>"""
import sys, numpy as np, pandas as pd, pickle
csv, prof, out = sys.argv[1:4]
rng = np.random.default_rng(20261114); N = 20000
P = pd.read_csv(prof)
lab = P.groupby("bifolium").agg(cur=("currier", lambda s: "|".join(sorted({str(v) for v in s.dropna()}))),
                                hand=("lfd_hand", lambda s: "|".join(sorted({str(v) for v in s.dropna()}))))
d = pickle.load(open(f"{out}/null_depth.pkl","rb")); depth, orders, SETS = d["depth"], d["orders"], d["sets"]
U = list(depth); miss = [s for s in U if s not in lab.index]; print("missing labels", miss)
print(lab.loc[U].assign(depth=[depth[s] for s in U]).sort_values(["depth"]).to_string())
def span(r): return max(r)-min(r)+1
rows = []
for pname in ["bifolium_path_rank", "chart_transcribed"]:
    rk = {s: i+1 for i, s in enumerate(orders[pname])}
    for sname in ["BOTTOM-N6", "BOTTOM-N5", "FORE-F4"]:
        st = SETS[sname]; obs = span([rk[s] for s in st])
        for key in ["cur", "hand"]:
            stratum = {s: (depth[s], lab.at[s, key]) for s in U}
            pools = {}
            for s in U: pools.setdefault(stratum[s], []).append(s)
            sizes = [len(pools[stratum[s]]) for s in st]
            nd = []
            for _ in range(N):
                pick = []
                for g in set(stratum[s] for s in st):
                    m = sum(stratum[s] == g for s in st); pick += list(rng.choice(pools[g], m, replace=False))
                nd.append(span([rk[s] for s in pick]))
            nd = np.array(nd)
            rows.append(dict(path=pname, set=sname, null=f"depth x {key}", obs=obs, null_mean=round(nd.mean(),2),
                             null_sd=round(nd.std(),2), z=round((nd.mean()-obs)/nd.std(),2) if nd.std() else None,
                             p=((nd <= obs).sum()+1)/(N+1), p_perfect=((nd <= len(st)).sum()+1)/(N+1),
                             pool_sizes=",".join(map(str, sizes))))
R = pd.DataFrame(rows); R.to_csv(f"{out}/null_bound_results.csv", index=False); print(R.to_string(index=False))
