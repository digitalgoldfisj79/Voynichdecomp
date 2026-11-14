import sys, pickle, json, numpy as np, pandas as pd
sys.path.insert(0,"/home/claude/spatial"); from common import *
rng=np.random.default_rng(20261115); d=pickle.load(open("/home/claude/inkdirect/phase2.pkl","rb")); U,PG=d["U"],d["PG"]; res={}
# B1
LFm=PG.groupby(["bifolium","leaf"])[["W_net","N_frac"]].mean().reset_index()
for m in ("W_net","N_frac"):
    rows=[(s,g[m].values[0],g[m].values[1],SH.at[s,"quire"],SH.at[s,"stratum"]) for s,g in LFm.groupby("bifolium") if len(g)==2 and s in SH.index]
    T=pd.DataFrame(rows,columns=["s","a","b","q","st"]); T=T[T.groupby("q").s.transform("size")>=2]; allv=np.c_[T.a,T.b]; obs=np.abs(T.a-T.b).mean()
    for key in ("q","st"):
        idx=[np.where(T[key].values==g)[0] for g in set(T[key])]; nl=[]
        for _ in range(10000):
            dd=[]
            for ix in idx: f=rng.permutation(allv[ix].ravel()); dd.append(np.abs(f[0::2]-f[1::2]))
            nl.append(np.concatenate(dd).mean())
        r=summ(obs,np.array(nl),True); r["n_sheets"]=len(T); r["n_groups"]=len(idx); res[f"B1_{m}_null_within_{'quire' if key=='q' else 'stratum'}"]=r
# B2
U["logWc"]=np.log(U.box_w/U.mm/U.w.str.len()); pg=U.groupby("page").agg(logWc=("logWc","mean"),peak=("peak","median")); PG=PG.join(pg)
def rz(y,cols):
    X=np.c_[np.ones(len(PG))]+0
    for c in cols: X=np.c_[X,PG[c].values]
    b=np.linalg.lstsq(X,y,rcond=None)[0]; r=y-X@b; return r, 1-(r**2).sum()/((y-y.mean())**2).sum()
steps={"N0_raw":[],"Na_size":["logH","logWc"],"Nb_size_width":["logH","logWc","W_net"],"Nc_size_width_contrast":["logH","logWc","W_net","peak"]}
for k,c in steps.items(): PG[k],r2=rz(PG.N_frac.values,c) if c else (PG.N_frac.values,0.0); res[f"r2_{k}"]=round(float(r2),3)
PG["contrast"]=PG.peak; PG["sharp"]=PG.sig
S=PG.groupby("bifolium")[list(steps)+["contrast","sharp","logH"]].mean()
for name,order in ORDERS.items():
    for m in S.columns:
        o=[s for s in order if s in S.index and np.isfinite(S.at[s,m])]; val=S[m].to_dict(); f=lambda a,b: abs(val[a]-val[b])
        obs=adj_mean(o,f); nl=np.array([adj_mean(perm_within(o,rng),f) for _ in range(3000)]); res[f"B2_{name}_{m}"]=summ(obs,nl,True)
res["page_corr_Nfrac"]={c:round(float(PG.N_frac.corr(PG[c])),3) for c in ("logH","logWc","W_net","peak","sig")}
json.dump(res,open("/home/claude/inkdirect/bound2_results.json","w"),indent=1); PG.to_csv("/home/claude/inkdirect/phase2_pages_bound.csv")
for k,v in res.items(): print(k,v)
