import sys, json, gzip, re, numpy as np, pandas as pd
sys.path.insert(0,"/home/claude/spatial"); from common import *
from vmsspatial import ink, load_bundle
rng=np.random.default_rng(20261125); res={}
I=ink.load_ink(); U=pd.DataFrame(I["units"]); U["page"]=[t.split(".")[0] for t in I["token_ids"]]
B=load_bundle("sensitivity", warp="recorded"); W=B["words"]
U["word"]=[" ".join(W[a:b+1]) for a,b in zip(U["first"],U["last"])]
bad=1|2|4|8|16|256|512; U=U[((U["flags"].astype(int)&bad)==0)&U.page.isin(PM.index)].copy()
U["w"]=U.word.str.replace(r"[^a-z]","",regex=True); U=U[U.w.str.len()>0]
chars=sorted(set("".join(U.w))); X=np.array([[len(w)]+[w.count(c) for c in chars] for w in U.w],float); X=np.c_[np.ones(len(X)),X]
beta=np.linalg.lstsq(X,U.coverage.values,rcond=None)[0]; pred=X@beta; U["resid"]=U.coverage-pred
res["model_r2"]=round(float(1-((U.resid)**2).sum()/((U.coverage-U.coverage.mean())**2).sum()),3); res["sample_words"]=U.word.head(3).tolist()
g=U.groupby("page"); T=pd.DataFrame({"resid":g.resid.median(),"cov":g.coverage.median(),"n":g.size(),"odd":g.resid.apply(lambda s:np.median(s.values[0::2])),"even":g.resid.apply(lambda s:np.median(s.values[1::2]))})
T=T[T.n>=30].join(PM[["leaf","side","quire","bifolium","hand","section"]]); res["n_pages"]=len(T)
res["splithalf_resid_r"]=round(float(np.corrcoef(T.odd,T.even)[0,1]),3)
res["side"]={"recto":round(float(T[T.side=="r"].resid.median()),4),"verso":round(float(T[T.side=="v"].resid.median()),4)}
H=T[T.hand.astype(str).isin(list("12345"))]
def eta(lab):
    gg=H.groupby(lab).resid; return float((gg.size()*(gg.mean()-H.resid.mean())**2).sum()/((H.resid-H.resid.mean())**2).sum())
ob=eta(H.hand.astype(str)); nd=np.array([eta(pd.Series(rng.permutation(H.hand.astype(str).values),index=H.index)) for _ in range(5000)])
res["hands_resid"]={"eta2":round(ob,3),"null_mean":round(float(nd.mean()),3),"p":round(float(((nd>=ob).sum()+1)/5001),5),"by_hand":H.groupby(H.hand.astype(str)).resid.median().round(4).to_dict(),"n_by_hand":H.hand.astype(str).value_counts().to_dict()}
# within section: herbal only, hand 1 vs 2
HH=T[(T.section=="H")&T.hand.astype(str).isin(["1","2"])]
d=HH[HH.hand.astype(str)=="2"].resid.mean()-HH[HH.hand.astype(str)=="1"].resid.mean(); lab=HH.hand.astype(str).values
nd=np.array([HH.resid.values[(p:=rng.permutation(lab))=="2"].mean()-HH.resid.values[p=="1"].mean() for _ in range(10000)])
res["herbal_hand2_minus_hand1_resid"]={"n1":int((lab=="1").sum()),"n2":int((lab=="2").sum())}|summ(d,nd,False)
# conjugates
T["leaf"]=T.leaf.astype(int); LZ=T.groupby("leaf").agg(r=("resid","mean"),quire=("quire","first"),bifolium=("bifolium","first"),hand=("hand","first"))
pairs=[tuple(g.index) for _,g in LZ.groupby("bifolium") if len(g)==2 and g.hand.nunique()==1]; grp={k:g.index.values for k,g in LZ.groupby(["quire","hand"])}
ob=np.mean([abs(LZ.at[a,"r"]-LZ.at[b,"r"]) for a,b in pairs]); nd=[]
for _ in range(5000):
    dd=[]
    for a,b in pairs:
        q=grp[(LZ.at[a,"quire"],LZ.at[a,"hand"])]
        if len(q)>2: x,y=rng.choice(q,2,replace=False); dd.append(abs(LZ.at[x,"r"]-LZ.at[y,"r"]))
    nd.append(np.mean(dd))
res["conjugates_resid"]={"n":len(pairs)}|summ(ob,nd,True)
SZ=T.groupby("bifolium").resid.mean(); f=lambda a,b: abs(SZ[a]-SZ[b]) if a in SZ.index and b in SZ.index else float("nan")
for name,o in ORDERS.items():
    o2=[s for s in o if s in SZ.index]; res["smooth_"+name]={"n":len(o2)}|summ(adj_mean(o2,f),[adj_mean(perm_within(o2,rng),f) for _ in range(5000)],True)
json.dump(res,open("/home/claude/spatial/b1_coverage_results.json","w"),indent=1); print(json.dumps(res,indent=1))
