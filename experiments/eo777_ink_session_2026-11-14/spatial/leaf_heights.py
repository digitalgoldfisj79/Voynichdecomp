"""P1: leaf heights (vms-spatial leaves.json.gz) as a physical variable. Usage: python3 -I leaf_heights.py <vms-spatial dir> <golf csv> <labelfree scores.csv> <null_depth.pkl> <out>"""
import sys, gzip, json, re, pickle, numpy as np, pandas as pd
from scipy.stats import spearmanr
vs, golf, lf, pk, out = sys.argv[1:6]; rng = np.random.default_rng(20261121)
L = json.load(gzip.open(f"{vs}/data/manuscript/leaves.json.gz"))["pages"]
PM = pd.read_csv(f"{vs}/data/manuscript/physical_map.csv").set_index("folio")
rows=[]
for p,v in L.items():
    if p in PM.index and not v.get("panel") and v.get("scale_method")=="measured" and v.get("top_method")=="measured" and v.get("leaf_height_mm"):
        rows.append(dict(page=p, leaf=int(PM.at[p,"leaf"]), side=v["side"], quire=PM.at[p,"quire"], sheet=PM.at[p,"bifolium"], h=v["leaf_height_mm"], w=v.get("visible_width_mm")))
D = pd.DataFrame(rows); D = D[D.quire.isin(list("ABCDEFGH"))]
res={"n_pages":len(D),"height_mm":{"median":float(D.h.median()),"p5":float(D.h.quantile(.05)),"p95":float(D.h.quantile(.95)),"sd":round(float(D.h.std()),2)}}
# 0a recto/verso
W = D.pivot_table(index="leaf", columns="side", values="h").dropna()
r = float(np.corrcoef(W.r, W.v)[0,1]); nr = np.array([np.corrcoef(W.r, rng.permutation(W.v.values))[0,1] for _ in range(5000)])
res["step0a"]={"n_leaves":len(W),"pearson_r":round(r,3),"null_p99":round(float(np.quantile(nr,.99)),3),"median_abs_diff_mm":round(float((W.r-W.v).abs().median()),2),"pass":bool(r>=0.5 and r>np.quantile(nr,.99))}
# leaf-level mean height
LH = D.groupby("leaf").agg(h=("h","mean"), quire=("quire","first"), sheet=("sheet","first"))
# 0b conjugates
pairs=[g.index.tolist() for _,g in LH.groupby("sheet") if len(g)==2]
obs=np.mean([abs(LH.at[a,"h"]-LH.at[b,"h"]) for a,b in pairs]); nd=[]
for _ in range(10000):
    d=[]
    for a,b in pairs:
        q=LH[LH.quire==LH.at[a,"quire"]].index.values; x,y=rng.choice(q,2,replace=False); d.append(abs(LH.at[x,"h"]-LH.at[y,"h"]))
    nd.append(np.mean(d))
nd=np.array(nd); res["step0b"]={"n_sheets":len(pairs),"mean_abs_diff_mm":round(float(obs),2),"null_mean":round(float(nd.mean()),2),"null_sd":round(float(nd.std()),2),"z":round(float((nd.mean()-obs)/nd.std()),2),"p":float(((nd<=obs).sum()+1)/10001)}
# 1 f5
qa=LH[LH.quire=="A"].h.sort_values(); res["step1_f5"]={"quireA_leaf_heights":{int(k):round(float(v),1) for k,v in qa.items()},"f5_minus_median_of_others":round(float(qa[5]-qa.drop(5).median()),1),"rv_f5":D[D.leaf==5][["page","h"]].values.tolist()}
# 2 bifolium height vs order
BH = D.groupby("sheet").agg(h=("h","mean"), quire=("quire","first"), n=("h","size"))
e=pd.read_csv(golf); G=e.groupby("sheet").bifolium_path_score.first(); F=pd.read_csv(lf).set_index("id").score
def test(score,name):
    J=BH.join(score.rename("s"),how="inner").dropna(); rho=float(spearmanr(J.h,J.s).statistic); nn=[]
    for _ in range(10000):
        hp=J.h.copy()
        for q,idx in J.groupby("quire").groups.items(): hp.loc[idx]=rng.permutation(J.loc[idx,"h"].values)
        nn.append(spearmanr(hp,J.s).statistic)
    nn=np.array(nn); res[name]={"n":len(J),"spearman":round(rho,3),"null_mean":round(float(nn.mean()),3),"null_sd":round(float(nn.std()),3),"z":round(float((rho-nn.mean())/nn.std()),2),"p_two_sided":float(((np.abs(nn-nn.mean())>=abs(rho-nn.mean())).sum()+1)/10001)}
test(G,"step2_path1"); test(F,"step2_labelfree")
# 3 N6 spread
depth=pickle.load(open(pk,"rb"))["depth"]; N6=["A:4-5","B:10-15","C:20-21","D:25-32","D:28-29","E:36-37"]; U=[s for s in BH.index if s in depth]
byd={}
for s in U: byd.setdefault(depth[s],[]).append(s)
ob=float(BH.loc[N6,"h"].std()); nd=[]
for _ in range(20000):
    pick=[]
    for d_ in set(depth[s] for s in N6):
        m=sum(depth[s]==d_ for s in N6); pick+=list(rng.choice(byd[d_],m,replace=False))
    nd.append(BH.loc[pick,"h"].std())
nd=np.array(nd); res["step3_N6"]={"heights":{s:round(float(BH.at[s,"h"]),1) for s in N6},"sd_mm":round(ob,2),"null_mean":round(float(nd.mean()),2),"null_sd":round(float(nd.std()),2),"z":round(float((nd.mean()-ob)/nd.std()),2),"p":float(((nd<=ob).sum()+1)/20001)}
BH.to_csv(f"{out}/bifolium_heights.csv"); LH.to_csv(f"{out}/leaf_heights.csv"); json.dump(res,open(f"{out}/leaf_heights_results.json","w"),indent=1); print(json.dumps(res,indent=1))
