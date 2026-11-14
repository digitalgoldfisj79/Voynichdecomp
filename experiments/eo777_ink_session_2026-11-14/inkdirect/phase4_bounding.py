"""Phase 4 corrected gradients + Bd3 (recto/verso split) + Bd4 (conjugates, fore-edge half). Consolidates the three inline runs of 2026-11-14."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0,"/home/claude/spatial"); from common import *
rng=np.random.default_rng(20261122); U=pd.read_pickle("/home/claude/inkdirect/phase4_words.pkl"); U=U[U.page.isin(PM.index)].copy(); res={}
U["side"]=PM.loc[U.page,"side"].values
def perm(v,gid): return v[np.lexsort((rng.random(len(v)),gid))]
def slope(D,g,order,m,n=3000):
    D=D[D.groupby(g)[g].transform("size")>=3].sort_values(order,kind="stable").reset_index(drop=True)
    gid=D[g].values; cnt=D.groupby(g)[g].transform("size").values; pos=(D.groupby(g).cumcount()/(cnt-1)).values
    pc=pos-pd.Series(pos).groupby(gid).transform("mean").values; v=D[m].values-D.groupby(g)[m].transform("mean").values
    st=lambda vv: float((vv*pc).sum()/(pc**2).sum()); nl=np.array([st(perm(v,gid)) for _ in range(n)]); o=st(v)
    return dict(pct=round(100*(np.exp(o)-1),2),obs=round(o,5),null_sd=round(float(nl.std()),5),z=round(float((o-nl.mean())/nl.std()),2),n=len(D))
Lm=U.groupby(["page","line"])[["W_net","N_net"]].mean().reset_index(); Lm=Lm[Lm.groupby("page").page.transform("size")>=5]
for m in ("W_net","N_net"):
    res[f"along_line_{m}"]=slope(U,"line",["line","x"],m); res[f"down_page_{m}"]=slope(Lm,"page",["page","line"],m)
Us=U.sort_values(["line","x"],kind="stable"); r=Us.groupby("line").cumcount(); c=Us.groupby("line").line.transform("size"); inner=Us[(r>0)&(r<c-1)]
for tag,D in (("recto",U[U.side=="r"]),("verso",U[U.side=="v"]),("inner_recto",inner[inner.side=="r"]),("inner_verso",inner[inner.side=="v"])):
    res[f"along_line_W_net_{tag}"]=slope(D,"line",["line","x"],"W_net")
U["fore"]=np.where(U.side=="r",U.x>U.groupby("page").x.transform("median"),U.x<=U.groupby("page").x.transform("median"))
def conj(D):
    n=D.groupby("page").size(); D=D[D.page.isin(n.index[n>=10])]; PG=D.groupby("page").W_net.mean().to_frame(); PG["leaf"]=PM.loc[PG.index,"leaf"].values; PG["bif"]=PM.loc[PG.index,"bifolium"].values
    LF=PG.groupby(["bif","leaf"]).W_net.mean().reset_index(); rows=[(s,g.W_net.values[0],g.W_net.values[1],SH.at[s,"quire"]) for s,g in LF.groupby("bif") if len(g)==2 and s in SH.index]
    T=pd.DataFrame(rows,columns=["s","a","b","q"]); T=T[T.groupby("q").s.transform("size")>=2]; allv=np.c_[T.a,T.b]; obs=np.abs(T.a-T.b).mean(); idx=[np.where(T.q.values==g)[0] for g in set(T.q)]
    nl=[np.concatenate([np.abs((f:=rng.permutation(allv[ix].ravel()))[0::2]-f[1::2]) for ix in idx]).mean() for _ in range(10000)]
    r=summ(obs,np.array(nl),True); r["n_sheets"]=len(T); return r
res["conjugates_all_words"]=conj(U); res["conjugates_fore_edge_half"]=conj(U[U.fore]); res["conjugates_gutter_half"]=conj(U[~U.fore])
json.dump(res,open("/home/claude/inkdirect/phase4_bounding_results.json","w"),indent=1,default=float)
pd.DataFrame([dict(test=k,**{a:b for a,b in v.items()}) for k,v in res.items()]).to_csv("/home/claude/inkdirect/phase4_results_table.csv",index=False)
for k,v in res.items(): print(k,{a:v[a] for a in v if a in("pct","z","null_sd","obs","n_sheets")})
