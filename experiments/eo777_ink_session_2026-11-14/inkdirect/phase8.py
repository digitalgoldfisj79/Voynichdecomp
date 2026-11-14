import sys, json, numpy as np, pandas as pd, pickle
sys.path.insert(0,"/home/claude/spatial"); from common import *
rng=np.random.default_rng(20261129); PG=pd.read_csv("/home/claude/inkdirect/phase7_pages.csv",index_col=0); PG=PG[PG.panel.isna()].copy(); res={}
U=pickle.load(open("/home/claude/inkdirect/phase5.pkl","rb"))["SETS"]["y2014"]; PG["logB"]=np.log(U.groupby("page").B.median()).reindex(PG.index); X=np.c_[np.ones(len(PG)),PG.logB]; PG["D_B"]=PG.D_net-X@np.linalg.lstsq(X,PG.D_net,rcond=None)[0]
PG["slot"]=np.where(PG.leaf<PG.partner_leaf,"a","b")+PG.side
def table(m):
    T=PG.pivot_table(index="bifolium",columns="slot",values=m).dropna(); return T[["ar","av","br","bv"]]
def stats(V):
    ar,av,br,bv=V.T; cross=(np.abs(ar-br)+np.abs(av-bv))/2
    return cross-np.abs(av-br), cross-np.abs(ar-bv), cross-(np.abs(ar-av)+np.abs(br-bv))/2
def zflip(d,n=100000):
    s=rng.choice([-1,1],(n,len(d))); nl=(s*d).mean(1); return dict(mean=round(float(d.mean()),4),null_sd=round(float(nl.std()),4),z=round(float(d.mean()/nl.std()),2),share_positive=round(float((d>0).mean()),2))
def verdict(V):
    di,do,dl=stats(V); s=rng.choice([-1,1],(4000,len(do))); z=do.mean()/(s*do).mean(1).std(); return "FACE" if z>=2 else ("SEQUENCE" if z<=-2 else "NOT RESOLVED")
for m in ("D_net","D_B"):
    T=table(m); V=T.values; wsd=float((V-V.mean(1,keepdims=True)).std()); res[f"{m}_n_sheets"]=len(T); res[f"{m}_within_sheet_sd"]=round(wsd,4)
    for f in (0.5,1.0):
        out={"seq":{},"face":{}}
        for kind in ("seq","face"):
            c={"FACE":0,"SEQUENCE":0,"NOT RESOLVED":0}
            for _ in range(300):
                W=np.array([rng.permutation(r) for r in V]); add=np.array([0,1,2,3])*f*wsd if kind=="seq" else np.array([1,-1,-1,1])*f*wsd/2*2
                c[verdict(W+add*rng.choice([-1,1],(len(W),1)))]+=1
            out[kind]={k:round(v/300,2) for k,v in c.items()}
        res[f"{m}_power_{f}x"]=out
    c={"FACE":0,"SEQUENCE":0,"NOT RESOLVED":0}
    for _ in range(300): c[verdict(np.array([rng.permutation(r) for r in V]))]+=1
    res[f"{m}_false_verdicts_no_structure"]={k:round(v/300,2) for k,v in c.items()}
    di,do,dl=stats(V); res[f"{m}_D_inner"]=zflip(di); res[f"{m}_D_outer"]=zflip(do); res[f"{m}_D_leaf"]=zflip(dl)
    ar,av,br,bv=V.T; res[f"{m}_mean_gaps"]={"d1_ar_av":round(float(np.abs(ar-av).mean()),4),"d1_av_br_inner":round(float(np.abs(av-br).mean()),4),"d1_br_bv":round(float(np.abs(br-bv).mean()),4),"d2_ar_br":round(float(np.abs(ar-br).mean()),4),"d2_av_bv":round(float(np.abs(av-bv).mean()),4),"d3_ar_bv_outer":round(float(np.abs(ar-bv).mean()),4)}
    res[f"{m}_mean_by_slot_pct_vs_sheet"]={s:round(float(100*(np.exp((T[s]-T.mean(1)).mean())-1)),2) for s in T.columns}
json.dump(res,open("/home/claude/inkdirect/phase8_results.json","w"),indent=1,default=float)
for k,v in res.items(): print(k,v)
