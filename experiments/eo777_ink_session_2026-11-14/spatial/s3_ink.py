"""S3 ink features. Run with golfenv python from vms-spatial dir."""
import sys, json, gzip, numpy as np, pandas as pd
sys.path.insert(0,"/home/claude/spatial"); from common import *
from vmsspatial import ink
rng=np.random.default_rng(20261124); res={}
I=ink.load_ink(); U=pd.DataFrame(I["units"]); U["page"]=[t.split(".")[0] for t in I["token_ids"]]
LV=json.load(gzip.open(f"{VS}/data/manuscript/leaves.json.gz"))["pages"]
bad=1|2|4|8|16|256|512
U=U[(U["flags"].astype(int)&bad)==0].copy(); U["mm"]=U.page.map(lambda p:(LV.get(p) or {}).get("mm"))
U["xh_mm"]=U.xheight/U.mm; U["asc_ratio"]=(U["mean"]-U.asc_y)/U.xheight
U.loc[(U.asc_ratio<0.3)|(U.asc_ratio>4),"asc_ratio"]=np.nan
U=U[U.page.isin(PM.index)]
feats={"xh_mm":"xh_mm","angle":"angle","asc_ratio":"asc_ratio","coverage":"coverage"}
rows=[]
for p,g in U.groupby("page"):
    if len(g)<30: continue
    r=dict(page=p,n=len(g))
    for k,c in feats.items():
        v=g[c].dropna().values
        if len(v)<16: r[k]=r[k+"_odd"]=r[k+"_even"]=np.nan; continue
        r[k]=np.median(v); r[k+"_odd"]=np.median(v[0::2]); r[k+"_even"]=np.median(v[1::2])
    v=g.xh_mm.dropna().values; r["xh_spread"]=np.median(np.abs(v-np.median(v)))/np.median(v)
    a,b=v[0::2],v[1::2]; r["xh_spread_odd"]=np.median(np.abs(a-np.median(a)))/np.median(a); r["xh_spread_even"]=np.median(np.abs(b-np.median(b)))/np.median(b)
    rows.append(r)
T=pd.DataFrame(rows).set_index("page").join(PM[["leaf","side","quire","bifolium","hand","section","currier"]])
F=["xh_mm","angle","asc_ratio","coverage","xh_spread"]; keep=[]
res["step0_splithalf"]={}
for k in F:
    d=T[[k+"_odd",k+"_even"]].dropna(); r=float(np.corrcoef(d.iloc[:,0],d.iloc[:,1])[0,1]); res["step0_splithalf"][k]={"r":round(r,3),"n":len(d),"pass":bool(r>=0.7)}
    if r>=0.7: keep.append(k)
res["features_kept"]=keep; T=T.dropna(subset=keep)
Z=(T[keep]-T[keep].mean())/T[keep].std()
# S3a hands
H=T[T.hand.astype(str).isin(["1","2","3","4","5"])]; res["S3a_hands"]={}
for k in keep:
    def eta(lab):
        g=H.groupby(lab)[k]; ssb=(g.size()*(g.mean()-H[k].mean())**2).sum(); return ssb/((H[k]-H[k].mean())**2).sum()
    ob=eta(H.hand.astype(str)); nd=[eta(pd.Series(rng.permutation(H.hand.astype(str).values),index=H.index)) for _ in range(5000)]
    res["S3a_hands"][k]={"eta2":round(float(ob),3),"null_mean":round(float(np.mean(nd)),3),"p":round(float(((np.array(nd)>=ob).sum()+1)/5001),5),"by_hand_median":H.groupby(H.hand.astype(str))[k].median().round(3).to_dict()}
# also side effect (photograph nuisance)
res["side_effect"]={k:{"recto":round(float(T[T.side=="r"][k].median()),3),"verso":round(float(T[T.side=="v"][k].median()),3)} for k in keep}
# leaf-level vectors (mean of z over pages)
T["leaf"]=T.leaf.astype(int); LZ=Z.join(T[["leaf","quire","bifolium","hand"]]).groupby("leaf").agg({**{k:"mean" for k in keep},"quire":"first","bifolium":"first","hand":"first"})
def dist(a,b,cols=keep): return float(np.sqrt(((LZ.loc[a,cols].astype(float)-LZ.loc[b,cols].astype(float))**2).sum()))
def conj_test(cols,tag):
    pairs=[tuple(g.index) for _,g in LZ.groupby("bifolium") if len(g)==2 and g.hand.nunique()==1]
    ob=np.mean([dist(a,b,cols) for a,b in pairs]); nd=[]
    grp={k:g.index.values for k,g in LZ.groupby(["quire","hand"])}
    for _ in range(5000):
        d=[]
        for a,b in pairs:
            q=grp[(LZ.at[a,"quire"],LZ.at[a,"hand"])]
            if len(q)<3: continue
            x,y=rng.choice(q,2,replace=False); d.append(dist(x,y,cols))
        nd.append(np.mean(d))
    res["S3b_conjugates_"+tag]={"n_sheets":len(pairs)}|summ(ob,nd,True)
conj_test(keep,"all_kept")
nt=[k for k in keep if k!="angle"]
if nt and nt!=keep: conj_test(nt,"without_angle")
# S3c recto/verso page distance vs conjugate page distance
PZ=Z.join(T[["leaf","bifolium","quire","hand","side"]])
def pd_(a,b,cols): return float(np.sqrt(((PZ.loc[a,cols]-PZ.loc[b,cols])**2).sum()))
rv=[(g.index[0],g.index[1]) for _,g in PZ.groupby("leaf") if len(g)==2]
res["S3c_recto_verso"]={"n":len(rv),"mean_dist_all":round(float(np.mean([pd_(a,b,keep) for a,b in rv])),3),"mean_dist_without_angle":round(float(np.mean([pd_(a,b,nt) for a,b in rv])),3) if nt else None,
   "random_same_quire_hand_without_angle":None}
grpP={k:g.index.values for k,g in PZ.groupby(["quire","hand"])}
if nt:
    nd=[np.mean([pd_(*rng.choice(grpP[(PZ.at[a,"quire"],PZ.at[a,"hand"])],2,replace=False),nt) for a,b in rv if len(grpP[(PZ.at[a,"quire"],PZ.at[a,"hand"])])>2]) for _ in range(2000)]
    res["S3c_recto_verso"]|={"test_without_angle":summ(np.mean([pd_(a,b,nt) for a,b in rv]),nd,True)}
# S3d smoothness along orders, herbal A hand 1 sheets, features without angle
cols=nt or keep; SZ=Z.join(T[["bifolium"]]).groupby("bifolium")[cols].mean()
f=lambda a,b: float(np.sqrt(((SZ.loc[a]-SZ.loc[b])**2).sum())) if a in SZ.index and b in SZ.index else float("nan")
for name,o in ORDERS.items():
    oo=[s for s in o if s in set(HERB_A1) and s in SZ.index]
    res["S3d_"+name+"_herbA_hand1"]={"n":len(oo)}|summ(adj_mean(oo,f),[adj_mean(list(rng.permutation(oo)),f) for _ in range(10000)],True)
    o2=[s for s in o if s in SZ.index]; res["S3d_"+name+"_all_stratified"]={"n":len(o2)}|summ(adj_mean(o2,f),[adj_mean(perm_within(o2,rng),f) for _ in range(5000)],True)
ch=[s for s in CHAIN if s in SZ.index]; res["S3d_chain_nullA"]=summ(adj_mean(ch,f),[adj_mean(list(rng.permutation(ch)),f) for _ in range(10000)],True)
T.to_csv("/home/claude/spatial/page_ink_features.csv"); json.dump(res,open("/home/claude/spatial/s3_ink_results.json","w"),indent=1,default=str); print(json.dumps(res,indent=1,default=str))
