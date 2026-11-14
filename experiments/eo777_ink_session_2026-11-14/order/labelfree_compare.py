"""L1: compare eo777 label-free bifolium fit with published Path 1. Usage: python3 -I labelfree_compare.py <scores.csv> <golf csv> <profiles.csv> <null_depth.pkl> <out>"""
import sys, pickle, json, numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau
sc, golf, prof, pk, out = sys.argv[1:6]
S = pd.read_csv(sc).set_index("id"); e = pd.read_csv(golf)
G = e.groupby("sheet")[["bifolium_path_score","bifolium_path_2_score","bifolium_path_3_score","bifolium_path_4_score"]].first().dropna()
J = S.join(G, how="inner"); assert len(J)==50, len(J)
res={"n":len(J)}
for c in G.columns:
    res[c]={"spearman":round(float(spearmanr(J.score,J[c]).statistic),4),"kendall":round(float(kendalltau(J.score,J[c]).statistic),4)}
o_free=list(J.score.sort_values().index); o_pub=list(J.bifolium_path_score.sort_values().index)
if res["bifolium_path_score"]["spearman"]<0: o_free=o_free[::-1]
adj=lambda o:{frozenset(x) for x in zip(o,o[1:])}
res["shared_adjacencies"]=len(adj(o_free)&adj(o_pub)); res["of"]=49
# biggest movers
r1={s:i for i,s in enumerate(o_free)}; r2={s:i for i,s in enumerate(o_pub)}
mv=sorted(r1,key=lambda s:-abs(r1[s]-r2[s]))[:8]; res["largest_rank_moves"]={s:[r2[s]+1,r1[s]+1] for s in mv}
# N6 span, herbal-universe re-rank, depth x Currier null
depth=pickle.load(open(pk,"rb"))["depth"]; P=pd.read_csv(prof); P=P[P.text_kind=="para"]
lab=P.groupby("bifolium").currier.agg(lambda s:"|".join(sorted({str(v) for v in s.dropna()})))
U=[s for s in o_free if s in depth]; rk={s:i+1 for i,s in enumerate(U)}
N6=["A:4-5","B:10-15","C:20-21","D:25-32","D:28-29","E:36-37"]; span=lambda st:max(rk[s] for s in st)-min(rk[s] for s in st)+1
rng=np.random.default_rng(20261120); strat={s:(depth[s],lab[s]) for s in U}; pools={}
for s in U: pools.setdefault(strat[s],[]).append(s)
nd=[]
for _ in range(20000):
    pick=[]
    for g in set(strat[s] for s in N6):
        m=sum(strat[s]==g for s in N6); pick+=list(rng.choice(pools[g],m,replace=False))
    nd.append(span(pick))
nd=np.array(nd); ob=span(N6)
res["N6_labelfree"]={"span":ob,"null_mean":round(nd.mean(),2),"null_sd":round(nd.std(),2),"z":round((nd.mean()-ob)/nd.std(),2),"p":float(((nd<=ob).sum()+1)/20001)}
pd.DataFrame({"rank":range(1,51),"label_free":o_free,"published_path1":o_pub}).to_csv(f"{out}/labelfree_vs_published.csv",index=False)
json.dump(res,open(f"{out}/labelfree_results.json","w"),indent=1); print(json.dumps(res,indent=1))
