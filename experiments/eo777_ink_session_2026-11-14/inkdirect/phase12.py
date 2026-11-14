import sys, json, pickle, itertools, numpy as np, pandas as pd
sys.path.insert(0,"/home/claude/spatial"); from common import *
rng=np.random.default_rng(20261206); res={}; d=pickle.load(open("/home/claude/inkdirect/phase5.pkl","rb"))["SETS"]
SIX=["A:4-5","B:10-15","C:20-21","D:25-32","D:28-29","E:36-37"]
for camp,U in d.items():
    P=U.groupby("page").agg(logB=("B",lambda x: float(np.log(np.median(x)))),sig=("sigma_hat",lambda x: float(np.log(np.median(x))))); P["bif"]=PM.loc[P.index,"bifolium"].values; P["leaf"]=PM.loc[P.index,"leaf"].values
    S=P.groupby("bif")[["logB","sig"]].mean(); H=[s for s in S.index if s in SH.index and SH.at[s,"stratum"]=="H|1"]
    for m in ("logB","sig"):
        prs=[(a,b,abs(S.at[a,m]-S.at[b,m])) for a,b in itertools.combinations(H,2) if SH.at[a,"quire"]!=SH.at[b,"quire"]]; g=np.array([x[2] for x in prs]); o=abs(S.at["A:4-5",m]-S.at["C:20-21",m])
        res[f"M1_{camp}_{m}"]=dict(gap=round(float(o),4),median_gap=round(float(np.median(g)),4),percentile=round(float((g<o).mean()),3),n_pairs=len(g),rank=int((g<o).sum())+1)
        six=[s for s in SIX if s in H]; pw=lambda ss: float(np.mean([abs(S.at[a,m]-S.at[b,m]) for a,b in itertools.combinations(ss,2)])); nl=np.array([pw(rng.choice(H,len(six),replace=False)) for _ in range(10000)])
        res[f"M2_{camp}_{m}"]=dict(n=len(six),obs=round(pw(six),4),null_mean=round(float(nl.mean()),4),null_sd=round(float(nl.std()),4),z=round(float((nl.mean()-pw(six))/nl.std()),2))
        # M3 known answer: leaves of same sheet vs leaf pairs from different quires
        L=P[P.bif.isin(H)].groupby(["bif","leaf"])[m].mean().reset_index(); L["q"]=[SH.at[b,"quire"] for b in L.bif]
        allp=np.array([abs(a-b) for (i,a,qa),(j,b,qb) in itertools.combinations(list(zip(L.index,L[m],L.q)),2) if qa!=qb]); thr=np.quantile(allp,0.05)
        same=[abs(x[m].values[0]-x[m].values[1]) for _,x in L.groupby("bif") if len(x)==2]; res[f"M3_{camp}_{m}"]=dict(n_sheets=len(same),share_true_sheets_in_closest_5pct=round(float(np.mean(np.array(same)<thr)),2),median_true_gap=round(float(np.median(same)),4),median_other_gap=round(float(np.median(allp)),4))
    res[f"values_{camp}"]={s:{m:round(float(S.at[s,m]),4) for m in ("logB","sig")} for s in SIX if s in S.index}
json.dump(res,open("/home/claude/inkdirect/phase12_results.json","w"),indent=1)
for k,v in res.items(): print(k,v)
