import sys, json, numpy as np, pandas as pd
sys.path.insert(0,"/home/claude/spatial"); from common import *
rng=np.random.default_rng(20261201); PG=pd.read_csv("/home/claude/inkdirect/phase7_pages.csv",index_col=0); PG=PG[PG.panel.isna()].copy(); res={}
PG["slot"]=np.where(PG.leaf<PG.partner_leaf,"a","b")+PG.side; T=PG.pivot_table(index="bifolium",columns="slot",values="D_net").reindex(columns=["ar","av","br","bv"])
wsd=float(T.sub(T.mean(axis=1),axis=0).stack().std()); SL="ar av br bv".split()
JO={"singulion_join":[("bv","ar")],"reverse_direction_join":[("ar","bv")],"nested_joins":[("av","ar"),("bv","br")]}
def shuf(V):
    W={}
    for s,v in V.items():
        ok=np.isfinite(v); w=v.copy(); w[ok]=rng.permutation(v[ok]); W[s]=w
    return W
def jstat(V,o,jo):
    out=[]
    for a,b in zip(o,o[1:]):
        G=np.abs(np.subtract.outer(V[a],V[b])); m=np.zeros((4,4),bool)
        for x,y in jo: m[SL.index(x),SL.index(y)]=True
        if np.isfinite(G[m]).all() and np.isfinite(G[~m]).sum()>=4: out.append(np.nanmean(G[~m])-G[m].mean())
    return float(np.mean(out)),len(out)
def dstat(V,o):
    x=[];y=[]
    for i in range(1,len(o)-1):
        v=V[o[i]]
        if np.isfinite(v[0]) and np.isfinite(v[3]): x.append(v[3]-v[0]); y.append(np.nanmean(V[o[i+1]])-np.nanmean(V[o[i-1]]))
    return float(np.corrcoef(x,y)[0,1]),len(x)
def plant_walk(V,o,f):
    W=shuf(V); rw=np.cumsum(rng.normal(0,1,4*len(o))); rw=(rw-rw.mean())/rw.std()*f*wsd
    return {s:W[s]+rw[4*i:4*i+4] for i,s in enumerate(o)}
def plant_equal(V,o):
    W=shuf(V)
    for a,b in zip(o,o[1:]):
        if np.isfinite(W[a][3]) and np.isfinite(W[b][0]): m=(W[a][3]+W[b][0])/2; W[a][3]=m; W[b][0]=m
    return W
def zs(obs,nl): nl=np.asarray(nl); return float((obs-nl.mean())/nl.std()),float(nl.std())
def _main():
  for name in ("path1","labelfree","binding"):
      o=[s for s in ORDERS[name] if s in T.index]; V={s:T.loc[s].values.astype(float) for s in o}
      for jn,jo in JO.items():
          obs,n=jstat(V,o,jo); nl=[jstat(shuf(V),o,jo)[0] for _ in range(4000)]; z,sd=zs(obs,nl); r=dict(n_joins=n,gain=round(obs,4),null_sd=round(sd,4),z=round(z,2))
          if jn=="singulion_join":
              r["power_join_equal"]=float(np.mean([zs(jstat(plant_equal(V,o),o,jo)[0],nl)[0]>=3 for _ in range(200)]))
              for f in (1,2): r[f"power_walk_{f}x"]=float(np.mean([zs(jstat(plant_walk(V,o,f),o,jo)[0],nl)[0]>=3 for _ in range(200)]))
          res[f"J1_{name}_{jn}"]=r
      obs,n=dstat(V,o); nl=[dstat(shuf(V),o)[0] for _ in range(4000)]; z,sd=zs(obs,nl); r=dict(n_sheets=n,corr=round(obs,3),null_sd=round(sd,3),z=round(z,2))
      for f in (1,2): r[f"power_walk_{f}x"]=float(np.mean([zs(dstat(plant_walk(V,o,f),o)[0],nl)[0]>=3 for _ in range(200)]))
      res[f"J2_direction_{name}"]=r
  res["within_sheet_sd"]=round(wsd,4); json.dump(res,open("/home/claude/inkdirect/phase10_results.json","w"),indent=1)
  pd.DataFrame([dict(test=k,**v) for k,v in res.items() if isinstance(v,dict)]).to_csv("/home/claude/inkdirect/phase10_results_table.csv",index=False)
  for k,v in res.items(): print(k,v)

if __name__=='__main__': _main()
