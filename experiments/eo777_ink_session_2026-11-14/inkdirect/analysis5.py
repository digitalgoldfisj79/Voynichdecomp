"""Phase 5: two photographic campaigns (2004 SID, 2014 Yale IIIF), one loader, registered tests P5-1..P5-5."""
import sys, glob, pickle, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,"/home/claude/spatial"); sys.path.insert(0,"/home/claude/inkdirect"); from common import *; import reg
rng=np.random.default_rng(20261123); res={}; B,_=reg.bundle(); t2l={}
for li,ln in enumerate(B["lines"]):
    for i in range(ln["start"],ln["end"]): t2l[i]=li
def load(d,pxmm_default=23.62):
    P=[pickle.load(open(f,"rb")) for f in sorted(glob.glob(f"/home/claude/inkdirect/{d}/*.pkl"))]; P=[p for p in P if "error" not in p]
    okp={p["page"] for p in P if p["chk"]["ratio_halfline"]>=1.10 and abs(p["reg"]["dx"])<=80 and abs(p["reg"]["dy"])<=80 and p["page"] in PM.index}
    U=pd.DataFrame([dict(page=p["page"],mm=p["mm"],pxmm=p.get("pxmm",pxmm_default),**r) for p in P if p["page"] in okp for r in p["rows"] if r.get("ok")])
    U["w"]=U.word.astype(str).str.replace(r"[^a-z]","",regex=True); U=U[(U.w.str.len()>0)&(U.width_px>0)&(U.n_frac>0)].copy()
    U["line"]=U.i.map(t2l); U["x"]=[B["geometry"][int(i)][0] for i in U.i]; U["logH"]=np.log(U.box_h/U.mm)
    ch=sorted(set("".join(U.w))); GX=np.c_[np.ones(len(U)),[[len(w)]+[w.count(c) for c in ch] for w in U.w]].astype(float)
    y=np.log(U.width_px/U.pxmm).values; U["W_net"]=y-GX@np.linalg.lstsq(GX,y,rcond=None)[0]
    X2=np.c_[GX,U.logH,np.log(U.box_w/U.mm)]; y=np.log(U.n_frac).values; U["N_net"]=y-X2@np.linalg.lstsq(X2,y,rcond=None)[0]
    n=U.groupby("page").size(); U=U[U.page.isin(n.index[n>=20])].copy(); U["side"]=PM.loc[U.page,"side"].values
    U["fore"]=np.where(U.side=="r",U.x>U.groupby("page").x.transform("median"),U.x<=U.groupby("page").x.transform("median")); return U,len(P),len(okp)
SETS={}
for tag,d in (("y2004","pages"),("y2014","pages_bein")):
    U,na,nk=load(d); SETS[tag]=U; res[f"{tag}_pages"]={"measured":na,"registration_ok":nk,"analysed":int(U.page.nunique()),"words":len(U),"width_mm_median":round(float(np.median(U.width_px/U.pxmm)),4),"sigma_hat_median":round(float(U.sigma_hat.median()),3)}
M=["W_net","N_net"]; PGs={t:U.groupby("page")[M].mean() for t,U in SETS.items()}; both=PGs["y2004"].index.intersection(PGs["y2014"].index); res["pages_in_both"]=len(both)
# P5-1
for m in M:
    a,b=PGs["y2004"].loc[both,m],PGs["y2014"].loc[both,m]; r=float(np.corrcoef(a,b)[0,1]); res[f"P5-1_agreement_{m}"]={"pearson":round(r,3),"spearman":round(float(spearmanr(a,b).statistic),3)}
    st=PM.loc[both,"section"].astype(str)+"|"+PM.loc[both,"hand"].astype(str); ar,br=a-a.groupby(st.values).transform("mean"),b-b.groupby(st.values).transform("mean")
    res[f"P5-1_agreement_{m}"]["pearson_within_section_hand"]=round(float(np.corrcoef(ar,br)[0,1]),3)
def perm(v,gid): return v[np.lexsort((rng.random(len(v)),gid))]
def slope_parts(D,g,order,m):
    D=D[D.groupby(g)[g].transform("size")>=3].sort_values(order,kind="stable").reset_index(drop=True); gid=D[g].values; cnt=D.groupby(g)[g].transform("size").values
    pos=(D.groupby(g).cumcount()/(cnt-1)).values; pc=pos-pd.Series(pos).groupby(gid).transform("mean").values; v=D[m].values-D.groupby(g)[m].transform("mean").values; return v,pc,gid
st_=lambda v,pc: float((v*pc).sum()/(pc**2).sum())
# P5-2
for t,U in SETS.items():
    for m in M:
        vr,pr,gr=slope_parts(U[U.side=="r"],"line",["line","x"],m); vv,pv,gv=slope_parts(U[U.side=="v"],"line",["line","x"],m); sr,sv=st_(vr,pr),st_(vv,pv)
        nl=np.array([(st_(perm(vr,gr),pr),st_(perm(vv,gv),pv)) for _ in range(2000)]); na,nb=(nl[:,0]+nl[:,1])/2,(nl[:,0]-nl[:,1])/2; a,b=(sr+sv)/2,(sr-sv)/2
        pw=float(np.mean([abs(((st_(perm(vr,gr)+np.log(1.02)*pr,pr)+st_(perm(vv,gv)+np.log(1.02)*pv,pv))/2-na.mean())/na.std())>=3 for _ in range(100)]))
        res[f"P5-2_along_line_{t}_{m}"]={"recto_pct":round(100*(np.exp(sr)-1),2),"verso_pct":round(100*(np.exp(sv)-1),2),"a_reading_pct":round(100*(np.exp(a)-1),2),"a_obs":round(a,5),"a_null_sd":round(float(na.std()),5),"a_z":round(float((a-na.mean())/na.std()),2),
            "b_gutter_pct":round(100*(np.exp(b)-1),2),"b_null_sd":round(float(nb.std()),5),"b_z":round(float((b-nb.mean())/nb.std()),2),"power_a_2pct":pw}
# P5-3
for t,U in SETS.items():
    Lm=U.groupby(["page","line"])[M].mean().reset_index(); Lm=Lm[Lm.groupby("page").page.transform("size")>=5]
    for m in M:
        v,pc,g=slope_parts(Lm,"page",["page","line"],m); o=st_(v,pc); nl=np.array([st_(perm(v,g),pc) for _ in range(3000)])
        res[f"P5-3_down_page_{t}_{m}"]={"pct":round(100*(np.exp(o)-1),2),"obs":round(o,5),"null_sd":round(float(nl.std()),5),"z":round(float((o-nl.mean())/nl.std()),2),"power_2pct":float(np.mean([abs((st_(perm(v,g)+np.log(1.02)*pc,pc)-nl.mean())/nl.std())>=3 for _ in range(100)]))}
# P5-4
def conj(PGv):
    PG=PGv.dropna().to_frame("v"); PG["leaf"]=PM.loc[PG.index,"leaf"].values; PG["bif"]=PM.loc[PG.index,"bifolium"].values; LF=PG.groupby(["bif","leaf"]).v.mean().reset_index()
    rows=[(s,g.v.values[0],g.v.values[1],SH.at[s,"quire"]) for s,g in LF.groupby("bif") if len(g)==2 and s in SH.index]; T=pd.DataFrame(rows,columns=["s","a","b","q"]); T=T[T.groupby("q").s.transform("size")>=2]
    allv=np.c_[T.a,T.b]; obs=np.abs(T.a-T.b).mean(); idx=[np.where(T.q.values==g)[0] for g in set(T.q)]; nl=[]
    for _ in range(10000):
        dd=[]
        for ix in idx: f=rng.permutation(allv[ix].ravel()); dd.append(np.abs(f[0::2]-f[1::2]))
        nl.append(np.concatenate(dd).mean())
    r=summ(obs,np.array(nl),True); r["n_sheets"]=len(T); return r
fore={t:U[U.fore].groupby("page")[M].mean() for t,U in SETS.items()}
for m in M:
    res[f"P5-4_conj_y2014_{m}"]=conj(PGs["y2014"][m]); res[f"P5-4_conj_y2004_{m}"]=conj(PGs["y2004"][m])
    res[f"P5-4_conj_mean_fore_half_{m}"]=conj(((fore["y2004"][m]+fore["y2014"][m])/2).loc[both]); res[f"P5-4_conj_mean_all_{m}"]=conj(((PGs["y2004"][m]+PGs["y2014"][m])/2).loc[both])
# P5-5 on two-campaign mean
PG=((PGs["y2004"].loc[both]+PGs["y2014"].loc[both])/2)
for c in ("leaf","side","quire","bifolium","partner_leaf","panel"): PG[c]=PM.loc[PG.index,c].values
PG=PG[PG.panel.isna()]; key={(r.leaf,r.side):p for p,r in PG.iterrows()}; pairs={"recto_verso":[],"facing":[],"conj_same_face":[],"conj_opp_face":[]}
for (lf,sd),p in key.items():
    if sd=="r" and (lf,"v") in key: pairs["recto_verso"].append((p,key[(lf,"v")]))
    if sd=="v" and (lf+1,"r") in key and PG.at[key[(lf+1,"r")],"quire"]==PG.at[p,"quire"]: pairs["facing"].append((p,key[(lf+1,"r")]))
    pl=PG.at[p,"partner_leaf"]
    if pd.notna(pl) and lf<pl:
        for s2 in ("r","v"):
            if (int(pl),s2) in key: pairs["conj_same_face" if s2!=sd else "conj_opp_face"].append((p,key[(int(pl),s2)]))
byq={q:list(g.index) for q,g in PG.groupby("quire") if len(g)>=2}
for m in M:
    for cl,pp in pairs.items():
        pp=[(a,b) for a,b in pp if PG.at[a,"quire"] in byq]; obs=np.mean([abs(PG.at[a,m]-PG.at[b,m]) for a,b in pp]); nl=[]
        for _ in range(5000): nl.append(np.mean([abs(np.subtract(*PG.loc[rng.choice(byq[PG.at[a,"quire"]],2,replace=False),m].values)) for a,b in pp]))
        r=summ(obs,np.array(nl),True); r["n_pairs"]=len(pp); res[f"P5-5_pairs_{cl}_{m}"]=r
LF=PG.groupby("leaf")[M].mean(); LF["st"]=[SH.at[PM[PM.leaf==l].bifolium.iloc[0],"stratum"] if PM[PM.leaf==l].bifolium.iloc[0] in SH.index else "NA" for l in LF.index]
LF=LF[LF.groupby("st").st.transform("size")>=4].sort_index(); st=LF.st.values; pos=np.arange(len(LF))
def sh(vv):
    o=vv.copy()
    for g in set(st): ix=np.where(st==g)[0]; o[ix]=rng.permutation(o[ix])
    return o
for m in M:
    v=LF[m].values-LF.groupby("st")[m].transform("mean").values; f=lambda x: float(spearmanr(pos,x).statistic); nl=np.array([f(sh(v)) for _ in range(5000)]); o=f(v)
    res[f"P5-5_through_book_{m}"]={"rho":round(o,3),"null_sd":round(float(nl.std()),3),"z":round(float((o-nl.mean())/nl.std()),2),"n_leaves":len(LF),"power_2SD":float(np.mean([abs((f(sh(v)+2*v.std()*np.linspace(0,1,len(v)))-nl.mean())/nl.std())>=3 for _ in range(200)]))}
json.dump(res,open("/home/claude/inkdirect/phase5_results.json","w"),indent=1,default=float); pickle.dump(dict(SETS=SETS,PGs=PGs),open("/home/claude/inkdirect/phase5.pkl","wb"))
pd.DataFrame([dict(test=k,**{a:(b if not isinstance(b,dict) else str(b)) for a,b in v.items()}) for k,v in res.items() if isinstance(v,dict)]).to_csv("/home/claude/inkdirect/phase5_results_table.csv",index=False)
for k,v in res.items(): print(k,v)
