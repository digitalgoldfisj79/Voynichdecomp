"""Phase 2d: ink darkness on the 2014 photographs."""
import sys, pickle, json, itertools, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,"/home/claude/spatial"); from common import *
rng=np.random.default_rng(20261124); res={}; d=pickle.load(open("/home/claude/inkdirect/phase5.pkl","rb")); U=d["SETS"]["y2014"].copy(); U4=d["SETS"]["y2004"].copy()
def prep(U):
    U=U[(U.darkness>0)&(U.peak>0)&U.sigma_hat.notna()].copy(); ch=sorted(set("".join(U.w))); GX=np.c_[np.ones(len(U)),[[len(w)]+[w.count(c) for c in ch] for w in U.w]].astype(float)
    f=lambda y,X: y-X@np.linalg.lstsq(X,y,rcond=None)[0]
    U["D_net"]=f(np.log(U.darkness.values),GX); U["D_adj"]=f(U.D_net.values,np.c_[np.ones(len(U)),U.sigma_hat,np.log(U.width_px)]); U["P_net"]=f(np.log(U.peak.values),GX)
    n=U.groupby("page").size(); U=U[U.page.isin(n.index[n>=20])].copy(); U["k"]=U.groupby("page").cumcount(); return U
U=prep(U); U4=prep(U4); M=["D_net","D_adj","P_net"]; PG=U.groupby("page")[M].mean(); res["pages"]=len(PG); res["words"]=len(U)
res["darkness_median"]=round(float(U.darkness.median()),3); res["page_darkness_p5_p95"]=[round(float(x),3) for x in U.groupby("page").darkness.median().quantile([.05,.95])]
sb=lambda c:(lambda r: round(float(2*r/(1+r)),3))(np.corrcoef(U[U.k%2==0].groupby("page")[c].mean().reindex(PG.index),U[U.k%2==1].groupby("page")[c].mean().reindex(PG.index))[0,1])
res["GATE_splithalf"]={c:sb(c) for c in M}; PG4=U4.groupby("page")[M].mean(); both=PG.index.intersection(PG4.index)
res["corr_with_2004_darkness"]={c:round(float(np.corrcoef(PG.loc[both,c],PG4.loc[both,c])[0,1]),3) for c in M}
res["page_corr"]={"D_net_vs_width":round(float(PG.D_net.corr(U.groupby("page").W_net.mean())),3),"D_net_vs_inkamount":round(float(PG.D_net.corr(U.groupby("page").N_net.mean())),3),"D_net_vs_focus":round(float(PG.D_net.corr(U.groupby("page").sigma_hat.median())),3)}
def perm(v,g): return v[np.lexsort((rng.random(len(v)),g))]
zz=lambda o,nl: dict(obs=round(float(o),5),null_sd=round(float(np.std(nl)),5),z=round(float((o-np.mean(nl))/np.std(nl)),2))
# D1
A=U[U.groupby("line").line.transform("size")>=4].sort_values(["line","x"],kind="stable").reset_index(drop=True); g=A.line.values
for m in M:
    v=A[m].values-A.groupby("line")[m].transform("mean").values
    for lag in (1,2,3):
        same=g[lag:]==g[:-lag]; st=lambda x: float((x[:-lag]*x[lag:])[same].mean()/x.var()); res[f"D1_lag{lag}_{m}"]=zz(st(v),[st(perm(v,g)) for _ in range(1500)])
def parts(D,gc,order,m):
    D=D[D.groupby(gc)[gc].transform("size")>=3].sort_values(order,kind="stable").reset_index(drop=True); gid=D[gc].values; cnt=D.groupby(gc)[gc].transform("size").values
    pos=(D.groupby(gc).cumcount()/(cnt-1)).values; return D[m].values-D.groupby(gc)[m].transform("mean").values, pos-pd.Series(pos).groupby(gid).transform("mean").values, gid
sl=lambda v,pc: float((v*pc).sum()/(pc**2).sum())
for m in M:
    vr,pr,gr=parts(U[U.side=="r"],"line",["line","x"],m); vv,pv,gv=parts(U[U.side=="v"],"line",["line","x"],m); sr,sv=sl(vr,pr),sl(vv,pv)
    nl=np.array([(sl(perm(vr,gr),pr),sl(perm(vv,gv),pv)) for _ in range(2000)]); na,nb=(nl[:,0]+nl[:,1])/2,(nl[:,0]-nl[:,1])/2
    pw=float(np.mean([abs(((sl(perm(vr,gr)+np.log(1.02)*pr,pr)+sl(perm(vv,gv)+np.log(1.02)*pv,pv))/2-na.mean())/na.std())>=3 for _ in range(100)]))
    res[f"D2_along_line_{m}"]={"recto_pct":round(100*(np.exp(sr)-1),2),"verso_pct":round(100*(np.exp(sv)-1),2),"a_reading_pct":round(100*(np.exp((sr+sv)/2)-1),2),"a_null_sd":round(float(na.std()),5),"a_z":round(float(((sr+sv)/2-na.mean())/na.std()),2),"b_gutter_pct":round(100*(np.exp((sr-sv)/2)-1),2),"b_z":round(float(((sr-sv)/2-nb.mean())/nb.std()),2),"power_2pct":pw}
    Lm=U.groupby(["page","line"])[M].mean().reset_index(); Lm=Lm[Lm.groupby("page").page.transform("size")>=5]; v,pc,gg=parts(Lm,"page",["page","line"],m); o=sl(v,pc); nl=np.array([sl(perm(v,gg),pc) for _ in range(3000)])
    r=zz(o,nl); r["pct"]=round(100*(np.exp(o)-1),2); r["power_2pct"]=float(np.mean([abs((sl(perm(v,gg)+np.log(1.02)*pc,pc)-nl.mean())/nl.std())>=3 for _ in range(100)])); res[f"D3_down_page_{m}"]=r
# D4
def conj(PGv):
    T=PGv.dropna().to_frame("v"); T["leaf"]=PM.loc[T.index,"leaf"].values; T["bif"]=PM.loc[T.index,"bifolium"].values; LF=T.groupby(["bif","leaf"]).v.mean().reset_index()
    rows=[(s,x.v.values[0],x.v.values[1],SH.at[s,"quire"]) for s,x in LF.groupby("bif") if len(x)==2 and s in SH.index]; T=pd.DataFrame(rows,columns=["s","a","b","q"]); T=T[T.groupby("q").s.transform("size")>=2]
    allv=np.c_[T.a,T.b]; obs=np.abs(T.a-T.b).mean(); idx=[np.where(T.q.values==q)[0] for q in set(T.q)]; nl=[]
    for _ in range(10000):
        dd=[]
        for ix in idx: f=rng.permutation(allv[ix].ravel()); dd.append(np.abs(f[0::2]-f[1::2]))
        nl.append(np.concatenate(dd).mean())
    r=summ(obs,np.array(nl),True); r["n_sheets"]=len(T); return r
for m in M: res[f"D4_conj_all_{m}"]=conj(PG[m]); res[f"D4_conj_fore_half_{m}"]=conj(U[U.fore].groupby("page")[m].mean())
# D5
P2=PG.copy()
for c in ("leaf","side","quire","partner_leaf","panel"): P2[c]=PM.loc[P2.index,c].values
P2=P2[P2.panel.isna()]; key={(r.leaf,r.side):p for p,r in P2.iterrows()}; pairs={"recto_verso":[],"facing":[],"conj_same_face":[],"conj_opp_face":[]}
for (lf,sd),p in key.items():
    if sd=="r" and (lf,"v") in key: pairs["recto_verso"].append((p,key[(lf,"v")]))
    if sd=="v" and (lf+1,"r") in key and P2.at[key[(lf+1,"r")],"quire"]==P2.at[p,"quire"]: pairs["facing"].append((p,key[(lf+1,"r")]))
    pl=P2.at[p,"partner_leaf"]
    if pd.notna(pl) and lf<pl:
        for s2 in ("r","v"):
            if (int(pl),s2) in key: pairs["conj_same_face" if s2!=sd else "conj_opp_face"].append((p,key[(int(pl),s2)]))
byq={q:list(x.index) for q,x in P2.groupby("quire") if len(x)>=2}
for m in ("D_net","D_adj"):
    for cl,pp in pairs.items():
        pp=[(a,b) for a,b in pp if P2.at[a,"quire"] in byq]; obs=np.mean([abs(P2.at[a,m]-P2.at[b,m]) for a,b in pp])
        nl=[np.mean([abs(np.subtract(*P2.loc[rng.choice(byq[P2.at[a,"quire"]],2,replace=False),m].values)) for a,b in pp]) for _ in range(3000)]; r=summ(obs,np.array(nl),True); r["n_pairs"]=len(pp); res[f"D5_pairs_{cl}_{m}"]=r
LF=P2.groupby("leaf")[M].mean(); LF["st"]=[SH.at[PM[PM.leaf==l].bifolium.iloc[0],"stratum"] if PM[PM.leaf==l].bifolium.iloc[0] in SH.index else "NA" for l in LF.index]
LF=LF[LF.groupby("st").st.transform("size")>=4].sort_index(); st=LF.st.values; pos=np.arange(len(LF))
def sh(x):
    o=x.copy()
    for q in set(st): ix=np.where(st==q)[0]; o[ix]=rng.permutation(o[ix])
    return o
for m in M:
    v=LF[m].values-LF.groupby("st")[m].transform("mean").values; f=lambda x: float(spearmanr(pos,x).statistic); nl=np.array([f(sh(v)) for _ in range(5000)]); o=f(v)
    res[f"D5_through_book_{m}"]={"rho":round(o,3),"null_sd":round(float(nl.std()),3),"z":round(float((o-nl.mean())/nl.std()),2),"n_leaves":len(LF),"power_2SD":float(np.mean([abs((f(sh(v)+2*v.std()*np.linspace(0,1,len(v)))-nl.mean())/nl.std())>=3 for _ in range(200)])),
        "by_stratum":{q:[int((st==q).sum()),round(float(spearmanr(pos[st==q],v[st==q]).statistic),2)] for q in sorted(set(st))}}
# D6-D8 sheet level
S=PG.groupby(PM.loc[PG.index,"bifolium"].values)[M].mean(); hs=[s for s in S.index if s in SH.index and SH.at[s,"section"]=="H" and SH.at[s,"hand"] in ("1","2")]
for m in M:
    v=S.loc[hs,m].values; h=SH.loc[hs,"hand"].values; dif=lambda vv,hh: vv[hh=="2"].mean()-vv[hh=="1"].mean(); nl=np.array([dif(v,rng.permutation(h)) for _ in range(10000)]); o=dif(v,h)
    r=zz(o,nl); r["pct"]=round(100*(np.exp(o)-1),2); r["n"]={x:int((h==x).sum()) for x in ("1","2")}
    for pz in (0.05,0.10): r[f"power_{int(pz*100)}pct"]=float(np.mean([abs((dif(v+np.where((hh:=rng.permutation(h))=="2",np.log(1+pz),0),hh)-nl.mean())/nl.std())>=3 for _ in range(500)]))
    res[f"D6_hand2_vs_hand1_{m}"]=r
    val=S[m].to_dict()
    for name,order in ORDERS.items():
        o_=[s for s in order if s in S.index]; f=lambda a,b: abs(val[a]-val[b]); obs=adj_mean(o_,f); nl=np.array([adj_mean(perm_within(o_,rng),f) for _ in range(3000)]); r=summ(obs,nl,True); r["n_sheets"]=len(o_); res[f"D7_order_{name}_{m}"]=r
    SIX=[s for s in ["A:4-5","B:10-15","C:20-21","D:25-32","D:28-29","E:36-37"] if s in S.index]; pool=[s for s in S.index if s in SH.index and SH.at[s,"stratum"]=="H|1"]
    pw=lambda ss: float(np.mean([abs(S.at[a,m]-S.at[b,m]) for a,b in itertools.combinations(ss,2)])); nl=np.array([pw(rng.choice(pool,len(SIX),replace=False)) for _ in range(10000)]); r=summ(pw(SIX),nl,True); r["n"]=len(SIX); r["pool"]=len(pool); res[f"D8_crease6_{m}"]=r
json.dump(res,open("/home/claude/inkdirect/phase2d_results.json","w"),indent=1,default=float); PG.to_csv("/home/claude/inkdirect/phase2d_pages.csv"); S.to_csv("/home/claude/inkdirect/phase2d_sheets.csv")
pd.DataFrame([dict(test=k,**{a:(b if not isinstance(b,dict) else str(b)) for a,b in v.items()}) for k,v in res.items() if isinstance(v,dict)]).to_csv("/home/claude/inkdirect/phase2d_results_table.csv",index=False)
for k,v in res.items(): print(k,v)
