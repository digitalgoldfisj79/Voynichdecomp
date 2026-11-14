"""Db1: page darkness adjusted for parchment brightness (median log background B). Run from vms-spatial dir with golfenv python. See RUNNING_RESULTS."""
import sys, pickle, json, numpy as np, pandas as pd
sys.path.insert(0,"/home/claude/spatial"); from common import *
rng=np.random.default_rng(20261125); U=pickle.load(open("/home/claude/inkdirect/phase5.pkl","rb"))["SETS"]["y2014"]; PG=pd.read_csv("/home/claude/inkdirect/phase2d_pages.csv",index_col=0); res={}
PG["logB"]=np.log(U.groupby("page").B.median()).reindex(PG.index); X=np.c_[np.ones(len(PG)),PG.logB]; PG["D_B"]=PG.D_net-X@np.linalg.lstsq(X,PG.D_net,rcond=None)[0]
def conj(PGv):
    T=PGv.dropna().to_frame("v"); T["leaf"]=PM.loc[T.index,"leaf"].values; T["bif"]=PM.loc[T.index,"bifolium"].values; LF=T.groupby(["bif","leaf"]).v.mean().reset_index()
    rows=[(s,x.v.values[0],x.v.values[1],SH.at[s,"quire"]) for s,x in LF.groupby("bif") if len(x)==2 and s in SH.index]; T=pd.DataFrame(rows,columns=["s","a","b","q"]); T=T[T.groupby("q").s.transform("size")>=2]
    allv=np.c_[T.a,T.b]; obs=np.abs(T.a-T.b).mean(); idx=[np.where(T.q.values==q)[0] for q in set(T.q)]
    nl=[np.concatenate([np.abs((f:=rng.permutation(allv[ix].ravel()))[0::2]-f[1::2]) for ix in idx]).mean() for _ in range(10000)]; return summ(obs,np.array(nl),True)
S=PG.groupby(PM.loc[PG.index,"bifolium"].values)[["D_net","D_B","logB"]].mean()
for m in ("D_B","logB"):
    res[f"conj_{m}"]=conj(PG[m]); val=S[m].to_dict()
    for name,order in ORDERS.items():
        o=[s for s in order if s in S.index]; f=lambda a,b: abs(val[a]-val[b]); res[f"order_{name}_{m}"]=summ(adj_mean(o,f),np.array([adj_mean(perm_within(o,rng),f) for _ in range(3000)]),True)
print(json.dumps(res,indent=1))
