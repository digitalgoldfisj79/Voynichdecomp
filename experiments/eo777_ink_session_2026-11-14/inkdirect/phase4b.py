import sys, json, numpy as np, pandas as pd
rng=np.random.default_rng(20261118); U=pd.read_pickle("/home/claude/inkdirect/phase4_words.pkl"); res={}
U=U[U.sigma_hat.notna()].copy()
for m in ("W_net","N_net"):
    X=np.c_[np.ones(len(U)),U.sigma_hat]; b=np.linalg.lstsq(X,U[m].values,rcond=None)[0]; U[m+"_f"]=U[m].values-X@b
U["sig"]=np.log(U.sigma_hat.clip(lower=0.2))
def perm(v,gid): return v[np.lexsort((rng.random(len(v)),gid))]
def zz(obs,nl): nl=np.asarray(nl); return dict(obs=round(float(obs),5),null_mean=round(float(nl.mean()),5),null_sd=round(float(nl.std()),5),z=round(float((obs-nl.mean())/nl.std()),2))
Lm=U.groupby(["page","line"])[["W_net","W_net_f","N_net","N_net_f","sig"]].mean().reset_index(); Lm=Lm[Lm.groupby("page").page.transform("size")>=5].sort_values(["page","line"]).reset_index(drop=True)
gid=Lm.page.values; cnt=Lm.groupby("page").page.transform("size").values; pos=(Lm.groupby("page").cumcount()/(cnt-1)).values; pc=pos-pd.Series(pos).groupby(gid).transform("mean").values
for m in ("W_net","W_net_f","N_net","N_net_f","sig"):
    v=Lm[m].values-Lm.groupby("page")[m].transform("mean").values; st=lambda vv: float((vv*pc).sum()/(pc**2).sum()); r=zz(st(v),[st(perm(v,gid)) for _ in range(3000)]); r["pct"]=round(100*(np.exp(r["obs"])-1),2); res[f"Bd1_down_page_{m}"]=r
A=U[U.groupby("line").line.transform("size")>=4].sort_values(["line","x"]).reset_index(drop=True); g=A.line.values
for m in ("W_net","W_net_f","N_net","N_net_f"):
    v=A[m].values-A.groupby("line")[m].transform("mean").values
    for lag in (1,2,3):
        same=g[lag:]==g[:-lag]; st=lambda vv: float((vv[:-lag]*vv[lag:])[same].mean()/vv.var()); res[f"Bd2_lag{lag}_{m}"]=zz(st(v),[st(perm(v,g)) for _ in range(1500)])
json.dump(res,open("/home/claude/inkdirect/phase4b_results.json","w"),indent=1)
for k,v in res.items(): print(k,v)
