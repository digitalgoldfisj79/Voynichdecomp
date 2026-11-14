import sys, pickle, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,"/home/claude/spatial"); from common import *
rng=np.random.default_rng(20261126); res={}
U=pickle.load(open("/home/claude/inkdirect/phase5.pkl","rb"))["SETS"]["y2014"]; PG=pd.read_csv("/home/claude/inkdirect/phase2d_pages.csv",index_col=0)
PG["logB"]=np.log(U.groupby("page").B.median()).reindex(PG.index); X=np.c_[np.ones(len(PG)),PG.logB]; PG["D_B"]=PG.D_net-X@np.linalg.lstsq(X,PG.D_net,rcond=None)[0]
S=PG.groupby(PM.loc[PG.index,"bifolium"].values)[["D_net","D_B"]].mean(); S.to_csv("/home/claude/inkdirect/phase6_sheets.csv")
adj=lambda v: float(np.mean(np.abs(np.diff(v))))
def strat_perm(v,st):
    o=v.copy()
    for g in set(st): ix=np.where(st==g)[0]; o[ix]=rng.permutation(o[ix])
    return o
def zlow(obs,nl): nl=np.asarray(nl); return dict(obs=round(float(obs),4),null_mean=round(float(nl.mean()),4),null_sd=round(float(nl.std()),4),z=round(float((nl.mean()-obs)/nl.std()),2))
P1=[s for s in ORDERS["path1"] if s in S.index]
# T-A
for m in ("D_net","D_B"):
    out={}
    for q in sorted(set(SH.loc[P1,"quire"])):
        o=[s for s in P1 if SH.at[s,"quire"]!=q]; v=S.loc[o,m].values; st=SH.loc[o,"stratum"].values; out[q]=[len(o),zlow(adj(v),[adj(strat_perm(v,st)) for _ in range(2000)])["z"]]
    zs=[x[1] for x in out.values()]; res[f"TA_loqo_{m}"]={"by_quire_dropped":out,"min_z":min(zs),"median_z":float(np.median(zs)),"max_z":max(zs)}
# T-B
H1=[s for s in S.index if s in SH.index and SH.at[s,"stratum"]=="H|1"]
for name,order in ORDERS.items():
    o=[s for s in order if s in H1]
    for m in ("D_net","D_B"):
        v=S.loc[o,m].values; pos=np.arange(len(o)); f=lambda x: float(spearmanr(pos,x).statistic); nl=np.array([f(rng.permutation(v)) for _ in range(10000)]); r=dict(n=len(o),rho=round(f(v),3),null_sd=round(float(nl.std()),3),z=round(float((f(v)-nl.mean())/nl.std()),2))
        for amp in (1,2,3): r[f"power_{amp}SD"]=float(np.mean([abs((f(rng.permutation(v)+amp*v.std()*np.linspace(0,1,len(v)))-nl.mean())/nl.std())>=3 for _ in range(500)]))
        res[f"TB_direction_H1_{name}_{m}"]=r
# T-C
def block_perm(v,st,k):
    o=v.copy(); b=np.arange(len(v))//k
    for g in set(zip(b,st)): ix=np.where((b==g[0])&(st==g[1]))[0]; o[ix]=rng.permutation(o[ix])
    return o
for m in ("D_net","D_B"):
    v=S.loc[P1,m].values; st=SH.loc[P1,"stratum"].values; sd=v.std()
    for k in (6,4,8):
        r=zlow(adj(v),[adj(block_perm(v,st,k)) for _ in range(5000)]); r["movable_share"]=round(float(np.mean([np.mean(block_perm(np.arange(len(v)).astype(float),st,k)!=np.arange(len(v))) for _ in range(200)])),2)
        if k==6:
            fa=det=0; n=300
            for _ in range(n):
                base=strat_perm(v,st)*0.6
                vb=base+np.repeat(rng.normal(0,sd,len(v)//k+1),k)[:len(v)]*0.8; fa+=(lambda x: (np.mean(x[1])-x[0])/np.std(x[1])>=3)((adj(vb),[adj(block_perm(vb,st,k)) for _ in range(200)]))
                w=np.cumsum(rng.normal(0,1,len(v))); w=(w-w.mean())/w.std()*sd*0.8; vs=base+w; det+=(lambda x: (np.mean(x[1])-x[0])/np.std(x[1])>=3)((adj(vs),[adj(block_perm(vs,st,k)) for _ in range(200)]))
            r["false_alarm_batch_plant"]=fa/n; r["power_sequence_plant"]=det/n
        res[f"TC_block{k}_{m}"]=r
res["path1_adjacent_same_currier_share"]=float(np.mean([SH.at[a,"currier"]==SH.at[b,"currier"] for a,b in zip(P1,P1[1:])])); res["path1_adjacent_same_quire_share"]=float(np.mean([SH.at[a,"quire"]==SH.at[b,"quire"] for a,b in zip(P1,P1[1:])]))
B1=[s for s in ORDERS["binding"] if s in S.index]; res["binding_adjacent_same_quire_share"]=float(np.mean([SH.at[a,"quire"]==SH.at[b,"quire"] for a,b in zip(B1,B1[1:])]))
json.dump(res,open("/home/claude/inkdirect/phase6_results.json","w"),indent=1,default=float)
for k,v in res.items(): print(k,v)
