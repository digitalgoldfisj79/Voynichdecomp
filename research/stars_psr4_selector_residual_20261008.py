#!/usr/bin/env python3
# STARS-PSR4 — preregistered 2026-10-08.
import collections,json,math,os,re,urllib.request
import numpy as np
from concurrent.futures import ProcessPoolExecutor

BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/7ea7f89ac447571e8340da0e75c2ae3e8fd73689/research/selector_innovation_primary_20261004.py"
src=urllib.request.urlopen(BASE,timeout=120).read().decode()
m={"__name__":"selector_innovation_primary_module"}
exec(compile(src,BASE,"exec"),m)
run_records=m["run_records"]

STAR_FOL=set(list(range(103,109))+list(range(111,115)))
CAL_SEEDS=list(range(202610081000,202610081400))
BLIND_SEEDS=list(range(202610081400,202610081500))

def role(i,n):
    if i==0:return "START"
    if i==1:return "SECOND"
    if i==n-1:return "FINAL"
    if i==n-2:return "PENULT"
    rel=(i-2)/max(1,(n-5))
    return "EARLY" if rel<1/3 else ("MIDDLE" if rel<2/3 else "LATE")
def lbin(n):
    return "L7" if n<=7 else ("L10" if n<=10 else ("L14" if n<=14 else "L15P"))

def stars_innovation_lines(observed=False,seed=None):
    ev,lines=run_records(seed=seed,latent=None,beta=0.0,observed=observed)
    out=[]
    for seq in lines:
        if not seq:continue
        fol=int(seq[0]["line"][0])
        if fol not in STAR_FOL:continue
        arr=[]
        for e in seq:
            v=-np.asarray(e["p"],float).copy()
            v[int(e["y"])]+=1.0
            arr.append(v)
        if len(arr)>=7:
            out.append((fol,np.stack(arr)))
    return out

def fit_means(lines,parity):
    cell=collections.defaultdict(list);rr=collections.defaultdict(list);allv=[]
    for fol,A in lines:
        if fol%2!=parity:continue
        n=len(A)
        for i,v in enumerate(A):
            ro=role(i,n);cell[(ro,lbin(n))].append(v);rr[ro].append(v);allv.append(v)
    return ({k:np.mean(v,0) for k,v in cell.items()},
            {k:np.mean(v,0) for k,v in rr.items()},
            np.mean(allv,0))
def residualize(lines):
    mods={0:fit_means(lines,0),1:fit_means(lines,1)};out=[]
    for fol,A in lines:
        cell,rr,g=mods[1-fol%2];n=len(A);B=[]
        for i,v in enumerate(A):
            ro=role(i,n);mu=cell.get((ro,lbin(n)),rr.get(ro,g));B.append(v-mu)
        out.append((fol,np.asarray(B)))
    return out
def crossop(lines):
    P=[];F=[]
    for fol,A in lines:
        for t in range(3,len(A)-2):
            P.append(A[t-3:t].reshape(-1));F.append(A[t:t+3].reshape(-1))
    P=np.asarray(P,float);F=np.asarray(F,float)
    P-=P.mean(0);F-=F.mean(0)
    C=P.T@F/len(P)
    U,s,Vt=np.linalg.svd(C,full_matrices=False)
    return C,U,s,Vt.T
def cv_scores(lines):
    o=[x for x in lines if x[0]%2==1];e=[x for x in lines if x[0]%2==0]
    Co,Uo,so,Vo=crossop(o);Ce,Ue,se,Ve=crossop(e)
    n=min(6,len(so),len(se));z=[]
    for i in range(n):
        z.append(.5*(float(Uo[:,i].T@Ce@Vo[:,i])+float(Ue[:,i].T@Co@Ve[:,i])))
    return z
def statistic(observed=False,seed=None):
    lines=stars_innovation_lines(observed=observed,seed=seed)
    res=residualize(lines)
    return {"scores":cv_scores(res),"lines":len(lines),"events":sum(len(a) for _,a in lines)}
def task(seed):
    x=statistic(False,seed);return seed,x["scores"]

REAL=statistic(True,None)
print("REAL",json.dumps(REAL,separators=(",",":")),flush=True)
if __name__=="__main__":
    seeds=CAL_SEEDS+BLIND_SEEDS
    with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:
        rows=list(ex.map(task,seeds,chunksize=2))
    mp={s:z for s,z in rows}
    cal=np.array([mp[s][0] for s in CAL_SEEDS],float)
    blind=np.array([mp[s][0] for s in BLIND_SEEDS],float)
    q99=float(np.quantile(cal,.99))
    blind_accept=float(np.mean(blind<=q99))
    real=float(REAL["scores"][0])
    pooled=np.r_[cal,blind]
    p_add=float((1+np.sum(pooled>=real))/(len(pooled)+1))
    if blind_accept<.90:decision="THE_METRIC_DOES_NOT_RESOLVE_CALIBRATION_FAIL"
    elif real<=q99:decision="KNOWN_COMPACT_SELECT_MECHANICS_ABSORB_LEADING_DIRECTION"
    elif p_add>.01:decision="THE_METRIC_DOES_NOT_RESOLVE_PVALUE_FAIL"
    else:decision="SURVIVES_KNOWN_SELECT_MECHANICS"
    out={"programme":"STARS-PSR4","status":"complete","real":REAL,
         "calibration":{"n":len(cal),"q99_c1":q99,"mean":float(cal.mean()),"sd":float(cal.std(ddof=1))},
         "blind":{"n":len(blind),"self_acceptance":blind_accept,"mean":float(blind.mean()),"sd":float(blind.std(ddof=1))},
         "pooled_add_one_p":p_add,"decision":decision}
    print("STARS_PSR4_JSON="+json.dumps(out,separators=(",",":")),flush=True)
