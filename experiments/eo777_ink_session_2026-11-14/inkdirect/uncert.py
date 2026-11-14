import sys, json, pickle, numpy as np, pandas as pd
sys.path.insert(0,"/home/claude/spatial"); sys.path.insert(0,"/home/claude/inkdirect"); from common import *
import phase10 as p10
rng=np.random.default_rng(20261202); res={}
T,wsd=p10.T,p10.wsd; o=[s for s in ORDERS["path1"] if s in T.index]; V={s:T.loc[s].values.astype(float) for s in o}
jo=p10.JO["singulion_join"]; nlj=np.array([p10.jstat(p10.shuf(V),o,jo)[0] for _ in range(2000)]); nld=np.array([p10.dstat(p10.shuf(V),o)[0] for _ in range(2000)])
res["U1"]={f"{f}x":{"join":float(np.mean([(p10.jstat(p10.plant_walk(V,o,f),o,jo)[0]-nlj.mean())/nlj.std()>=3 for _ in range(200)])),"direction":float(np.mean([(p10.dstat(p10.plant_walk(V,o,f),o)[0]-nld.mean())/nld.std()>=3 for _ in range(200)]))} for f in (1,2,4,8,16)}
# U2
P=[pickle.load(open(f,"rb")) for f in __import__("glob").glob("/home/claude/inkdirect/pages_bein/*.pkl")]; nw={p["page"]:sum(1 for r in p["rows"] if r.get("ok")) for p in P}; rat={p["page"]:p["chk"]["ratio_halfline"] for p in P if "error" not in p}
PMs=PM[PM.panel.isna()].copy(); PMs["slot"]=np.where(PMs.leaf<PMs.partner_leaf,"a","b")+PMs.side; need=[]
for a,b in zip(o,o[1:]):
    pa=PMs[(PMs.bifolium==a)&(PMs.slot=="bv")].index; pb=PMs[(PMs.bifolium==b)&(PMs.slot=="ar")].index
    for pg in list(pa)+list(pb): need.append(pg)
def why(pg): return "ok" if nw.get(pg,0)>=20 and rat.get(pg,0)>=1.10 else ("not_measured_or_no_boxes" if pg not in nw else ("fewer_than_20_clean_words" if nw[pg]<20 else "registration_rule"))
res["U2_join_pages"]=pd.Series([why(p) for p in need]).value_counts().to_dict(); res["U2_join_pages_words_8_to_19"]=int(sum(8<=nw.get(p,0)<20 for p in need))
U=pickle.load(open("/home/claude/inkdirect/phase5.pkl","rb"))["SETS"]
# loader kept pages>=20 words; rebuild page D_net with >=8 from raw pickles
rows=[dict(page=p["page"],**r) for p in P if "error" not in p and p["chk"]["ratio_halfline"]>=1.10 and p["page"] in PM.index for r in p["rows"] if r.get("ok") and r.get("darkness",0)>0]
R=pd.DataFrame(rows); R["w"]=R.word.astype(str).str.replace(r"[^a-z]","",regex=True); R=R[R.w.str.len()>0]; ch=sorted(set("".join(R.w))); GX=np.c_[np.ones(len(R)),[[len(w)]+[w.count(c) for c in ch] for w in R.w]].astype(float)
y=np.log(R.darkness.values); R["D"]=y-GX@np.linalg.lstsq(GX,y,rcond=None)[0]; n=R.groupby("page").size(); P8=R[R.page.isin(n.index[n>=8])].groupby("page").D.mean().to_frame("D")
P8=P8[PM.loc[P8.index,"panel"].isna().values]; P8["bif"]=PM.loc[P8.index,"bifolium"].values; P8["slot"]=PMs.loc[P8.index,"slot"].values; T8=P8.pivot_table(index="bif",columns="slot",values="D").reindex(columns=p10.SL)
o8=[s for s in ORDERS["path1"] if s in T8.index]; V8={s:T8.loc[s].values.astype(float) for s in o8}; ob,nj=p10.jstat(V8,o8,jo); nl=np.array([p10.jstat(p10.shuf(V8),o8,jo)[0] for _ in range(4000)])
res["U2_variant_min8words_singulion_join_path1"]={"n_joins":nj,"gain":round(ob,4),"null_sd":round(float(nl.std()),4),"z":round(float((ob-nl.mean())/nl.std()),2)}
od,nd=p10.dstat(V8,o8); nl=np.array([p10.dstat(p10.shuf(V8),o8)[0] for _ in range(4000)]); res["U2_variant_min8words_direction_path1"]={"n_sheets":nd,"corr":round(od,3),"null_sd":round(float(nl.std()),3),"z":round(float((od-nl.mean())/nl.std()),2)}
# U3
S=pd.read_csv("/home/claude/inkdirect/phase6_sheets.csv",index_col=0); adj=lambda v: float(np.mean(np.abs(np.diff(v))))
for name in ("path1","labelfree","binding"):
    oo=[s for s in ORDERS[name] if s in S.index and SH.at[s,"stratum"]=="H|1"]; v=S.loc[oo,"D_net"].values; nl=np.array([adj(rng.permutation(v)) for _ in range(10000)])
    res[f"U3_herbal_hand1_only_{name}"]={"n_sheets":len(oo),"obs":round(adj(v),4),"null_mean":round(float(nl.mean()),4),"null_sd":round(float(nl.std()),4),"z":round(float((nl.mean()-adj(v))/nl.std()),2),"currier":SH.loc[oo,"currier"].value_counts().to_dict()}
# U4
for camp in ("y2004","y2014"):
    X=U[camp]; X=X[X.darkness>0].copy(); ch=sorted(set("".join(X.w))); GX=np.c_[np.ones(len(X)),[[len(w)]+[w.count(c) for c in ch] for w in X.w]].astype(float); y=np.log(X.darkness.values); X["D"]=y-GX@np.linalg.lstsq(GX,y,rcond=None)[0]
    Sx=X.groupby("page").D.mean().groupby(PM.bifolium).mean()
    for name in ("path1","binding"):
        oo=[s for s in ORDERS[name] if s in Sx.index]; val=Sx.to_dict(); f=lambda a,b: abs(val[a]-val[b]); obs=adj_mean(oo,f); nl=np.array([adj_mean(perm_within(oo,rng),f) for _ in range(3000)]); r=summ(obs,nl,True); r["n_sheets"]=len(oo); res[f"U4_{camp}_darkness_{name}"]=r
json.dump(res,open("/home/claude/inkdirect/uncert_results.json","w"),indent=1,default=float)
for k,v in res.items(): print(k,v)
