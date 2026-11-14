import sys, json, numpy as np, pandas as pd
sys.path.insert(0,"/home/claude/spatial"); from common import *
rng=np.random.default_rng(20261130); PG=pd.read_csv("/home/claude/inkdirect/phase7_pages.csv",index_col=0); PG=PG[PG.panel.isna()].copy(); res={}
PG["slot"]=np.where(PG.leaf<PG.partner_leaf,"a","b")+PG.side; T=PG.pivot_table(index="bifolium",columns="slot",values="D_net").reindex(columns=["ar","av","br","bv"])
S=PG.groupby("bifolium").D_net.mean(); wsd=float((T.sub(T.mean(axis=1),axis=0)).stack().std())
qs=SH.loc[[s for s in SH.index if s in S.index]].sort_values(["quire","first_leaf"]); quires=[list(g.index) for q,g in sorted(qs.groupby("quire"),key=lambda x: x[1].first_leaf.min())]
# S1
allorder=[s for q in quires for s in q]; adjq=[(a,b) for q in quires for a,b in zip(q,q[1:])]; obs=np.mean([abs(S[a]-S[b]) for a,b in adjq]); nl=[]
for _ in range(10000):
    d=[]
    for q in quires:
        p=rng.permutation(q); d+=[abs(S[a]-S[b]) for a,b in zip(p,p[1:])]
    nl.append(np.mean(d))
r=summ(obs,np.array(nl),True); r["n_adjacent_pairs"]=len(adjq); r["n_quires"]=len(quires); res["S1_within_quire_nesting_order_sheets"]=r
# S2/S3
JO={"singulion_join_bv_ar":[("bv","ar")],"reverse_stack_join":[("ar","bv")],"bound_facing_joins":[("av","ar"),("br","bv")]}
def run(pairs,tag):
    pairs=[(a,b) for a,b in pairs if a in T.index and b in T.index]
    def stat(V,jo):
        out=[]
        for a,b in pairs:
            A,B=V[a],V[b]; G=np.abs(np.subtract.outer(A,B)); m=np.zeros((4,4),bool)
            for x,y in jo: m["ar av br bv".split().index(x),"ar av br bv".split().index(y)]=True
            if np.isfinite(G[m]).all() and np.isfinite(G[~m]).sum()>=4: out.append(np.nanmean(G[~m])-G[m].mean())
        return (float(np.mean(out)),len(out)) if out else (np.nan,0)
    V={s:T.loc[s].values for s in T.index}
    def shuf(V):
        W={}
        for s,v in V.items():
            ok=np.isfinite(v); w=v.copy(); w[ok]=rng.permutation(v[ok]); W[s]=w
        return W
    for name,jo in JO.items():
        o,n=stat(V,jo); nl=np.array([stat(shuf(V),jo)[0] for _ in range(4000)]); det=0
        for _ in range(200):
            W=shuf(V)
            for a,b in pairs:
                e=rng.normal(0,wsd)
                for x,y in jo[:1]:
                    i,j="ar av br bv".split().index(x),"ar av br bv".split().index(y); mid=(W[a][i]+W[b][j])/2
                    if np.isfinite(mid): W[a]=W[a].copy(); W[b]=W[b].copy(); W[a][i]=mid+e*0.1; W[b][j]=mid-e*0.1
            det+=(stat(W,jo)[0]-nl.mean())/nl.std()>=3
        res[f"{tag}_{name}"]=dict(n_joins=n,gain=round(o,4),null_sd=round(float(nl.std()),4),z=round(float((o-nl.mean())/nl.std()),2),power_join_made_equal=det/200)
run(adjq,"S2_within_quire")
run([(q1[-1],q2[0]) for q1,q2 in zip(quires,quires[1:])],"S3_across_quires_singulion_stack")
run([(q1[0],q2[0]) for q1,q2 in zip(quires,quires[1:])],"S3_across_quires_bound_book")
json.dump(res,open("/home/claude/inkdirect/phase9_results.json","w"),indent=1,default=float)
for k,v in res.items(): print(k,v)
