import sys, json, numpy as np, pandas as pd
rng=np.random.default_rng(20261128); PG=pd.read_csv("/home/claude/inkdirect/phase7_pages.csv",index_col=0); PG=PG[PG.panel.isna()]; res={}
def setup(D,key):
    D=D.dropna(subset=[key]).sort_values(key,kind="stable").reset_index(drop=True); return D.D_net.values.copy(), D.bifolium.values
def stat(v,g,k):
    sm=pd.Series(v).groupby(g).transform("mean").values; off=v-sm; n=len(v); loc=np.full(n,np.nan)
    for i in range(n):
        j=np.r_[max(0,i-k):i,i+1:min(n,i+k+1)]; j=j[g[j]!=g[i]]
        if len(j)>=2: loc[i]=off[j].mean()
    lo=loc; ok=np.isfinite(lo)&(pd.Series(g).groupby(g).transform("size").values>1)
    return float(np.corrcoef(off[ok],lo[ok])[0,1])
def wperm(v,g): return v[np.lexsort((rng.random(len(v)),g))][np.argsort(np.argsort(g,kind="stable"),kind="stable")] if False else _wp(v,g)
def _wp(v,g):
    o=v.copy()
    for s in np.unique(g):
        ix=np.where(g==s)[0]; o[ix]=rng.permutation(o[ix])
    return o
def ztest(v,g,k,n=300):
    o=stat(v,g,k); nl=np.array([stat(_wp(v,g),g,k) for _ in range(n)]); return o,float((o-nl.mean())/nl.std()),float(nl.std())
stage=sys.argv[1]; v,g=setup(PG,"prank"); wsd=float((v-pd.Series(v).groupby(g).transform("mean").values).std())
if stage=="dev":
    for k in (3,5,8):
        fa=np.mean([ztest(_wp(v,g),g,k,100)[1]>=3 for _ in range(60)]); out={"false_alarm":float(fa)}
        for f in (0.5,1.0):
            det=0
            for _ in range(60):
                x=_wp(v,g); rw=np.cumsum(rng.normal(0,1,len(v))); rw=(rw-rw.mean())/rw.std()*f*wsd; det+=ztest(x+rw,g,k,100)[1]>=3
            out[f"power_{f}x"]=det/60
        res[f"k{k}"]=out; print(k,out,flush=True)
    json.dump(res,open("/home/claude/inkdirect/phase7b_dev.json","w"),indent=1)
else:
    k=int(sys.argv[2])
    for name,key,D in (("page_path","prank",PG),("reading_order","read",PG),("page_path_herbal_hand1","prank",PG[(PG.section=="H")&(PG.hand.astype(str)=="1")]),("reading_order_herbal_hand1","read",PG[(PG.section=="H")&(PG.hand.astype(str)=="1")])):
        v,g=setup(D,key); o,z,sd=ztest(v,g,k,3000); res[name]={"n_pages":len(v),"corr":round(o,3),"null_sd":round(sd,3),"z":round(z,2)}; print(name,res[name],flush=True)
    json.dump(res,open(f"/home/claude/inkdirect/phase7b_real_k{k}.json","w"),indent=1)
