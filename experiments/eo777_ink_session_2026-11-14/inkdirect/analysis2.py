"""Phase 2 analysis per the pre-registration in RUNNING_RESULTS_inkdirect.md. Run with golfenv python."""
import sys, glob, pickle, json, numpy as np, pandas as pd
sys.path.insert(0,"/home/claude/spatial"); from common import *
SEED=20261114; rng=np.random.default_rng(SEED); NP=10000; res={}
P=[pickle.load(open(f,"rb")) for f in sorted(glob.glob("/home/claude/inkdirect/pages/*.pkl"))]
res["pages_attempted"]=len(P); res["pages_error"]=[p["page"] for p in P if "error" in p]
pd.DataFrame([(p["page"],p["file"],p.get("sha256")) for p in P],columns=["page","file","sha256"]).to_csv("/home/claude/inkdirect/tif_sha256.csv",index=False)
meta=pd.DataFrame([dict(page=p["page"],scale=p["reg"]["scale"],dx=p["reg"]["dx"],dy=p["reg"]["dy"],ratio=p["chk"]["ratio_halfline"],mm=p["mm"]) for p in P if "error" not in p]).set_index("page")
meta["reg_ok"]=(meta.ratio>=1.10)&(meta.dx.abs()<=80)&(meta.dy.abs()<=80)
R=pd.DataFrame([dict(page=p["page"],**r) for p in P if "error" not in p for r in p["rows"]])
R=R[R.page.map(meta.reg_ok)&R.page.isin(PM.index)].copy(); res["pages_reg_fail"]=sorted(meta.index[~meta.reg_ok]); res["n_eligible_words"]=len(R)
res["G1_coverage_width"]=round(float(R.ok.mean()),3); res["coverage_darkness"]=round(float((R.ok&R.darkness.notna()).sum()/len(R)),3)
U=R[R.ok.fillna(False)&(R.width_px>0)].copy(); U["w"]=U.word.astype(str).str.replace(r"[^a-z]","",regex=True); U=U[U.w.str.len()>0].copy()
U["mm"]=U.page.map(meta.mm); U["logW"]=np.log(U.width_px/23.62); U["logH"]=np.log(U.box_h/U.mm); U["k"]=U.groupby("page").cumcount()
chars=sorted(set("".join(U.w))); GX=np.c_[np.ones(len(U)),[[len(w)]+[w.count(c) for c in chars] for w in U.w]].astype(float)
def gres(y):
    ok=np.isfinite(y); b=np.linalg.lstsq(GX[ok],y[ok],rcond=None)[0]; r=np.full(len(y),np.nan); r[ok]=y[ok]-GX[ok]@b; return r, 1-np.nansum(r**2)/np.nansum((y[ok]-y[ok].mean())**2)
U["W_net"],r2=gres(U.logW.values); res["glyph_model_r2_width"]=round(float(r2),3)
U["N_frac"],r2n=gres(np.log(U.n_frac.where(U.n_frac>0)).values); U["W_raw"]=U.logW
U["D_net"],_=gres(U.darkness.where(U.darkness>0).values.astype(float))
n=U.groupby("page").size(); keep=n.index[n>=20]; U=U[U.page.isin(keep)].copy(); res["pages_analysed"]=int(len(keep)); res["words_measured"]=int(len(U))
PG=U.groupby("page").agg(W_net=("W_net","mean"),W_raw=("W_raw","mean"),N_frac=("N_frac","mean"),D_net=("D_net","mean"),sig=("sigma_hat","median"),logH=("logH","mean"),n=("W_net","size"),width_mm=("width_px",lambda x: float(np.median(x)/23.62)))
def padj(col):
    X=np.c_[np.ones(len(PG)),PG.sig,PG.logH]; ok=np.isfinite(X).all(1)&np.isfinite(PG[col]); b=np.linalg.lstsq(X[ok],PG[col][ok],rcond=None)[0]; r=pd.Series(np.nan,PG.index); r[ok]=PG[col][ok]-X[ok]@b; return r
PG["W_adj"]=padj("W_net"); PG["D_adj"]=padj("D_net")
for c in ("bifolium","leaf","side","section","hand","quire"): PG[c]=PM.loc[PG.index,c].values
# gates
def split(col):
    a=U[U.k%2==0].groupby("page")[col].mean(); b=U[U.k%2==1].groupby("page")[col].mean(); r=np.corrcoef(a.reindex(PG.index),b.reindex(PG.index))[0,1]; return round(float(2*r/(1+r)),3)
res["G2_splithalf_SB"]={c:split(c) for c in ("W_net","W_raw","N_frac")}
res["page_summary"]={"width_mm_median":round(float(PG.width_mm.median()),4),"width_mm_p5_p95":[round(float(x),4) for x in PG.width_mm.quantile([.05,.95])],"sigma_hat_median":round(float(PG.sig.median()),3),"sigma_hat_p5_p95":[round(float(x),3) for x in PG.sig.quantile([.05,.95])],"share_pages_sigma_le_1.79":round(float((PG.sig<=1.79).mean()),3)}
res["page_corr"]={"W_net_vs_sigma":round(float(PG.W_net.corr(PG.sig)),3),"W_net_vs_logH":round(float(PG.W_net.corr(PG.logH)),3),"W_net_vs_N_frac":round(float(PG.W_net.corr(PG.N_frac)),3),"sigma_vs_logH":round(float(PG.sig.corr(PG.logH)),3)}
rv=PG.reset_index().pivot_table(index="leaf",columns="side",values="W_net"); rv=rv.dropna(); res["recto_verso_r_W_net"]=round(float(np.corrcoef(rv["r"],rv["v"])[0,1]),3) if {"r","v"}<=set(rv.columns) else None
res["gates_pass"]=bool(res["G1_coverage_width"]>=.5 and res["G2_splithalf_SB"]["W_net"]>=.8)
res["darkness_gate_pass"]=bool(res["coverage_darkness"]>=.5 and res["page_summary"]["share_pages_sigma_le_1.79"]>=.8)
MEAS=["W_net","W_adj","W_raw","N_frac"]+(["D_adj"] if res["darkness_gate_pass"] else [])
S=PG.groupby("bifolium")[MEAS].mean(); LF_=PG.groupby(["bifolium","leaf"])[MEAS].mean().reset_index()
def zp(obs,null,lower): return summ(obs,np.array(null),lower)
def ratio_note(d): d["ratio_effect_to_nullSD"]=abs(d["z"]); return d
# H1
def h1(vals,hands,rg,n=NP):
    obs=vals[hands=="2"].mean()-vals[hands=="1"].mean(); nl=[]
    for _ in range(n): h=rg.permutation(hands); nl.append(vals[h=="2"].mean()-vals[h=="1"].mean())
    nl=np.array(nl); return obs,nl
hs=[s for s in S.index if s in SH.index and SH.at[s,"section"]=="H" and SH.at[s,"hand"] in ("1","2")]
res["H1"]={"n_sheets":{h:int(sum(SH.loc[hs,"hand"]==h)) for h in ("1","2")}}
for m in MEAS:
    v=S.loc[hs,m].values; h=SH.loc[hs,"hand"].values; ok=np.isfinite(v); obs,nl=h1(v[ok],h[ok],rng)
    res["H1"][m]=dict(obs_diff=round(float(obs),4),pct=round(float(100*(np.exp(obs)-1)),2) if m!="D_adj" else None,null_sd=round(float(nl.std()),4),z=round(float((obs-nl.mean())/nl.std()),2),p_two=round(float(((np.abs(nl-nl.mean())>=abs(obs-nl.mean())).sum()+1)/(len(nl)+1)),5))
def power_h1(m,delta,plants=500,nn=400):
    v=S.loc[hs,m].values; h=SH.loc[hs,"hand"].values; det=0
    for _ in range(plants):
        hh=rng.permutation(h); vv=v+np.where(hh=="2",delta,0); obs,nl=h1(vv,hh,rng,nn); det+=abs((obs-nl.mean())/nl.std())>=3
    return det/plants
res["H1"]["power_W_net"]={f"{int(round(100*(np.exp(d)-1)))}%":power_h1("W_net",d) for d in (np.log(1.05),np.log(1.10),np.log(1.15))}
# H2
def h2stat(a,b): return float(np.mean(np.abs(a-b)))
def h2(T,m,rg,n=NP):
    obs=h2stat(T.a.values,T.b.values); nl=[]
    idx=[np.where(T.st.values==g)[0] for g in sorted(set(T.st))]
    allv=np.c_[T.a.values,T.b.values]
    for _ in range(n):
        d=[]
        for ix in idx:
            f=rg.permutation(allv[ix].ravel()); d.append(np.abs(f[0::2]-f[1::2]))
        nl.append(np.concatenate(d).mean())
    return obs,np.array(nl)
def conj(m):
    rows=[]
    for s,g in LF_.groupby("bifolium"):
        g=g.dropna(subset=[m])
        if len(g)==2 and s in SH.index: rows.append((s,g[m].values[0],g[m].values[1],SH.at[s,"stratum"]))
    return pd.DataFrame(rows,columns=["s","a","b","st"])
res["H2"]={}
for m in MEAS:
    T=conj(m); obs,nl=h2(T,m,rng); d=zp(obs,nl,True); d["n_sheets"]=len(T); res["H2"][m]=d
def power_h2(m,frac=0.5,plants=300,nn=300):
    T=conj(m); allv=np.c_[T.a,T.b]; sd=np.concatenate([allv[(T.st==g).values].ravel()-allv[(T.st==g).values].mean() for g in set(T.st)]).std(); det=0
    for _ in range(plants):
        TT=T.copy(); f=allv.copy()
        for g in set(T.st):
            ix=np.where(T.st.values==g)[0]; f[ix]=rng.permutation(allv[ix].ravel()).reshape(-1,2)
        e=rng.normal(0,frac*sd,len(T)); TT["a"]=f[:,0]+e; TT["b"]=f[:,1]+e; obs,nl=h2(TT,m,rng,nn); det+=(nl.mean()-obs)/nl.std()>=3
    return det/plants
res["H2"]["power_W_net_sheetSD_half_leafSD"]=power_h2("W_net")
# H3
res["H3"]={}
def h3(order,val,rg,n=NP):
    f=lambda a,b: abs(val[a]-val[b]); obs=adj_mean(order,f); nl=[adj_mean(perm_within(order,rg),f) for _ in range(n)]; return obs,np.array(nl)
for name,order in ORDERS.items():
    res["H3"][name]={}
    for m in MEAS:
        o=[s for s in order if s in S.index and np.isfinite(S.at[s,m])]; val=S[m].to_dict(); obs,nl=h3(o,val,rng,3000); d=zp(obs,nl,True); d["n_sheets"]=len(o); res["H3"][name][m]=d
def power_h3(name,m="W_net",amp=1.0,plants=200,nn=300):
    o=[s for s in ORDERS[name] if s in S.index and np.isfinite(S.at[s,m])]; st=SH.loc[o,"stratum"].values; v=S.loc[o,m].values
    wsd=np.concatenate([v[st==g]-v[st==g].mean() for g in set(st)]).std(); det=0
    for _ in range(plants):
        vv=v.copy()
        for g in set(st):
            ix=np.where(st==g)[0]; vv[ix]=rng.permutation(vv[ix])
        vv=vv+amp*wsd*np.linspace(0,1,len(o)); val=dict(zip(o,vv)); obs,nl=h3(o,val,rng,nn); det+=(nl.mean()-obs)/nl.std()>=3
    return det/plants,float(wsd)
for name in ORDERS:
    res["H3"][name]["power_drift_1SD"],res["H3"][name]["within_stratum_sd"]=power_h3(name)
    res["H3"][name]["power_drift_3SD"]=power_h3(name,amp=3.0)[0]
PG.to_csv("/home/claude/inkdirect/phase2_pages.csv"); S.to_csv("/home/claude/inkdirect/phase2_sheets.csv"); meta.to_csv("/home/claude/inkdirect/phase2_registration.csv")
pickle.dump(dict(U=U,PG=PG,S=S,meta=meta,res=res),open("/home/claude/inkdirect/phase2.pkl","wb"))
json.dump(res,open("/home/claude/inkdirect/phase2_results.json","w"),indent=1,default=float); print(json.dumps(res,indent=1,default=float))
