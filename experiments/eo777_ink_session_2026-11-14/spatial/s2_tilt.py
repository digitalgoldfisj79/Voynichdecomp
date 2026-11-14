"""S2 tilt carry. Run with golfenv python from the vms-spatial dir."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0,"/home/claude/spatial"); from common import *
from vmsspatial import compare
c=compare.Comparison(); rng=np.random.default_rng(20261122); res={}
rows=[]
for p in PM.index:
    try: d=c.drift(p)
    except Exception: continue
    if d.get("angle") is None: continue
    q=[l["angle"] for l in d["lines"] if l["qualifies"]]
    if len(q)<4: continue
    rows.append(dict(page=p,angle=d["angle"],odd=float(np.median(q[0::2])),even=float(np.median(q[1::2])),n=len(q)))
T=pd.DataFrame(rows).set_index("page"); T=T.join(PM[["leaf","side","quire","bifolium","hand","section"]])
r=float(np.corrcoef(T.odd,T.even)[0,1]); res["step0_splithalf"]={"n_pages":len(T),"r":round(r,3),"pass":bool(r>=0.7),"sd_angle_deg":round(float(T.angle.std()),3)}
# nuisance: same opening (verso n, recto n+1 consecutive leaves in the same quire) 
T["leaf"]=T.leaf.astype(int); byleaf={(int(l),s):p for p,(l,s) in T[["leaf","side"]].iterrows()}
opens=[(byleaf[(l,"v")],byleaf[(l+1,"r")]) for l in sorted(set(T.leaf)) if (l,"v") in byleaf and (l+1,"r") in byleaf and T.at[byleaf[(l,"v")],"quire"]==T.at[byleaf[(l+1,"r")],"quire"]]
ob=np.mean([abs(T.at[a,"angle"]-T.at[b,"angle"]) for a,b in opens]); nd=[]
pagesq={q:g.index.values for q,g in T.groupby("quire")}
for _ in range(10000):
    nd.append(np.mean([abs(np.subtract(*T.loc[rng.choice(pagesq[T.at[a,"quire"]],2,replace=False),"angle"].values)) for a,b in opens]))
res["nuisance_same_opening"]={"n_openings":len(opens)}|summ(ob,nd,True)
rvp=[(byleaf[(l,"r")],byleaf[(l,"v")]) for l in sorted(set(T.leaf)) if (l,"r") in byleaf and (l,"v") in byleaf]
res["recto_verso_same_leaf"]={"n":len(rvp),"r":round(float(np.corrcoef([T.at[a,"angle"] for a,b in rvp],[T.at[b,"angle"] for a,b in rvp])[0,1]),3),"mean_abs_diff":round(float(np.mean([abs(T.at[a,"angle"]-T.at[b,"angle"]) for a,b in rvp])),3)}
ST=T.groupby("bifolium").angle.median()
f=lambda a,b: abs(ST[a]-ST[b]) if a in ST.index and b in ST.index else float("nan")
ob=adj_mean(CHAIN,f); res["chain_nullA"]=summ(ob,[adj_mean(list(rng.permutation(CHAIN)),f) for _ in range(10000)],True)
pool=[s for s in HERB_A1 if s in ST.index]; res["chain_nullB"]={"pool":len(pool)}|summ(ob,[adj_mean(list(rng.choice(pool,10,replace=False)),f) for _ in range(10000)],True)
for name,o in ORDERS.items():
    o=[s for s in o if s in ST.index]; res["order_"+name]={"n":len(o)}|summ(adj_mean(o,f),[adj_mean(perm_within(o,rng),f) for _ in range(10000)],True)
T.to_csv("/home/claude/spatial/page_tilt.csv"); json.dump(res,open("/home/claude/spatial/s2_tilt_results.json","w"),indent=1); print(json.dumps(res,indent=1))
