import sys, pickle, json, itertools, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,"/home/claude/spatial"); from common import *
rng=np.random.default_rng(20261116); d=pickle.load(open("/home/claude/inkdirect/phase2.pkl","rb")); U=d["U"].copy(); res={}
PGb=pd.read_csv("/home/claude/inkdirect/phase2_pages_bound.csv",index_col=0)
U=U[U.n_frac>0].copy(); U["ln"]=np.log(U.n_frac); U["lw"]=np.log(U.box_w/U.mm); U["bif"]=PM.loc[U.page,"bifolium"].values
def smooth(S,order,n=3000):
    o=[s for s in order if s in S.index and np.isfinite(S[s])]; val=S.to_dict(); f=lambda a,b: abs(val[a]-val[b])
    obs=adj_mean(o,f); nl=np.array([adj_mean(perm_within(o,rng),f) for _ in range(n)]); r=summ(obs,nl,True); r["n_sheets"]=len(o); return r
# T1
top=U.w.value_counts().head(30); res["T1_types"]=top.index.tolist(); res["T1_tokens"]=int(top.sum()); T=U[U.w.isin(top.index)].copy()
def within(g):
    X=np.c_[np.ones(len(g)),g.logH,g.lw]; b=np.linalg.lstsq(X,g.ln,rcond=None)[0]; r=g.ln-X@b; return (r-r.mean())/r.std()
T["v"]=T.groupby("w",group_keys=False).apply(within)
cnt=T.groupby("page").size(); okp=cnt.index[cnt>=5]; res["T1_pages"]=int(len(okp)); res["T1_tokens_per_page_median"]=float(cnt[okp].median())
Ssw=T[T.page.isin(okp)].groupby("page").v.mean().groupby(PM.bifolium).mean()
chars=sorted(set("".join(U.w))); GX=np.c_[np.ones(len(U)),[[len(w)]+[w.count(c) for c in chars] for w in U.w],U.logH,U.lw].astype(float)
b=np.linalg.lstsq(GX,U.ln.values,rcond=None)[0]; U["allv"]=U.ln.values-GX@b
for name,order in ORDERS.items():
    res[f"T1_sameword_{name}"]=smooth(Ssw,order); zs=[]
    for k in range(20):
        sub=pd.concat([g.sample(n=min(len(g),int(cnt[p])),random_state=int(rng.integers(1e9))) for p,g in U[U.page.isin(okp)].groupby("page")])
        zs.append(smooth(sub.groupby("page").allv.mean().groupby(PM.bifolium).mean(),order,600)["z"])
    res[f"T1_control_matched_count_{name}"]={"median_z":float(np.median(zs)),"min_z":float(min(zs)),"max_z":float(max(zs))}
    res[f"T1_allword_full_{name}"]=smooth(U[U.page.isin(okp)].groupby("page").allv.mean().groupby(PM.bifolium).mean(),order)
# T2
S2=PGb.groupby("bifolium")[["W_net","Nb_size_width"]].mean()
def drift_stat(o,v,st):
    c=v.copy()
    for g in set(st): c[st==g]-=c[st==g].mean()
    return abs(spearmanr(np.arange(len(o)),c).statistic)
def drift(o,v,st,n):
    obs=drift_stat(o,v,st); nl=[]
    for _ in range(n):
        vv=v.copy()
        for g in set(st): ix=np.where(st==g)[0]; vv[ix]=rng.permutation(vv[ix])
        nl.append(drift_stat(o,vv,st))
    nl=np.array(nl); return obs,nl
for name,order in ORDERS.items():
    for m in S2.columns:
        o=[s for s in order if s in S2.index and np.isfinite(S2.at[s,m])]; v=S2.loc[o,m].values.astype(float); st=SH.loc[o,"stratum"].values
        obs,nl=drift(o,v,st,5000); r=summ(obs,nl,False); wsd=np.concatenate([v[st==g]-v[st==g].mean() for g in set(st)]).std()
        for amp in (1,2):
            det=0
            for _ in range(200):
                vv=v.copy()
                for g in set(st): ix=np.where(st==g)[0]; vv[ix]=rng.permutation(vv[ix])
                vv=vv+amp*wsd*np.linspace(0,1,len(o)); ob,n2=drift(o,vv,st,300); det+=(ob-n2.mean())/n2.std()>=3
            r[f"power_{amp}SD"]=det/200
        res[f"T2_{name}_{m}"]=r
# T3
SIX=["A:4-5","B:10-15","C:20-21","D:25-32","D:28-29","E:36-37"]
for m in S2.columns:
    have=[s for s in SIX if s in S2.index and np.isfinite(S2.at[s,m])]; pool=[s for s in S2.index if s in SH.index and np.isfinite(S2.at[s,m])]
    pw=lambda ss: float(np.mean([abs(S2.at[a,m]-S2.at[b,m]) for a,b in itertools.combinations(ss,2)])); need=SH.loc[have,"stratum"].value_counts()
    def draw(): return [x for g,k in need.items() for x in rng.choice([s for s in pool if SH.at[s,"stratum"]==g],k,replace=False)]
    obs=pw(have); nl=np.array([pw(draw()) for _ in range(10000)]); r=summ(obs,nl,True); r["have"]=have; r["strata"]=need.to_dict(); r["pool_sizes"]={g:int(sum(SH.loc[pool,"stratum"]==g)) for g in need.index}
    v=S2.loc[pool,m]; st=SH.loc[pool,"stratum"]; wsd=float((v-v.groupby(st).transform("mean")).std()); det=0
    for _ in range(300):
        ss=draw(); vals=S2[m].copy(); perm=vals[ss].values-vals[ss].mean()  # keep spread, plant by shrinking toward a shared offset
        vals[ss]=vals[ss].mean()+wsd*1.0*np.sign(rng.normal())+0.5*perm; old=S2[m]; S2[m]=vals; ob=pw(ss); n2=np.array([pw(draw()) for _ in range(300)]); S2[m]=old; det+=(n2.mean()-ob)/n2.std()>=3
    r["power_planted"]=det/300; res[f"T3_crease6_{m}"]=r
json.dump(res,open("/home/claude/inkdirect/phase3_results.json","w"),indent=1,default=float)
for k,v in res.items(): print(k,v)
