#!/usr/bin/env python3
# STARS-MEM1 — preregistered 2026-10-08.
import collections,json,math,os,re,urllib.request
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import Ridge

LAT_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/7ea7f89ac447571e8340da0e75c2ae3e8fd73689/research/latent_line_state_test_20261004.py"
m={"__name__":"latent_line_state_module"}
src=urllib.request.urlopen(LAT_URL,timeout=120).read().decode()
exec(compile(src,LAT_URL,"exec"),m)
rows=m["rows"];LINES=m["LINES"];fit_struct=m["fit_struct"];para_id=m["para_id"];K=m["K"];ALPHA=m["ALPHA"]
TR=[l for l in LINES if l["fold"] in (2,3,4)]
M,G,TH=fit_struct(TR);NG=sum(G.values())
STAR_FOL=set(f"f{n}{s}" for n in list(range(103,109))+list(range(111,115)) for s in ("r","v"))
LGRID=(1,2,3,5,8);LMAX=8;ALPHA_RIDGE=100.0
CAL_SEEDS=list(range(202610083000,202610083200))
BLIND_SEEDS=list(range(202610083200,202610083300))

def fnum(f):return int(re.search(r"\d+",str(f)).group())

def base_prob(section,piece,pagev,parav,recent):
    cc=M.get((section,piece),{});n=sum(cc.values())
    b=np.array([(cc.get(c,0)+ALPHA*((G[c]+.5)/(NG+.5*K)))/(n+ALPHA) for c in range(K)],float)
    sc=np.log(np.maximum(b,1e-15))+np.log1p(pagev)*TH[:K]+np.log1p(parav)*TH[K:2*K]+recent*TH[-1]
    sc-=sc.max();p=np.exp(sc);p/=p.sum();return p

def innovation_lines(observed=False,seed=None):
    rng=np.random.default_rng(seed if seed is not None else 12345)
    page=collections.defaultdict(lambda:np.zeros(K,float));para=collections.defaultdict(lambda:np.zeros(K,float));hist=collections.defaultdict(list);by=collections.OrderedDict()
    for r in rows:
        fol=r["folio"];ln=r["line"];lk=(fol,ln);pk=fol;q=(pk,para_id(pk,ln));yobs=int(r["start"]);pos=int(r["pos"])
        if pos==0:y=yobs
        else:
            prev_piece=hist[lk][-1][1];rc=np.zeros(K,float)
            for yy,pp in hist[lk][-6:]:rc[int(yy)]+=1
            p=base_prob(r["section"],prev_piece,page[pk],para[q],rc)
            y=yobs if observed else int(rng.choice(K,p=p))
            if fol in STAR_FOL:
                v=-p.copy();v[y]+=1.;by.setdefault(lk,[]).append(v)
        page[pk][y]+=1;para[q][y]+=1;hist[lk].append((y,r["final_piece"]))
    return [(fol,np.stack(vs)) for (fol,ln),vs in by.items() if len(vs)>=LMAX+1]

def role(i,n):
    if i==0:return "START"
    if i==1:return "SECOND"
    if i==n-1:return "FINAL"
    if i==n-2:return "PENULT"
    rel=(i-2)/max(1,n-5)
    return "EARLY" if rel<1/3 else ("MIDDLE" if rel<2/3 else "LATE")
def lbin(n):return "L7" if n<=7 else ("L10" if n<=10 else ("L14" if n<=14 else "L15P"))

def fit_means(lines,parity):
    cell=collections.defaultdict(list);rr=collections.defaultdict(list);allv=[]
    for fol,A in lines:
        if fnum(fol)%2!=parity:continue
        n=len(A)
        for i,v in enumerate(A):
            ro=role(i,n);cell[(ro,lbin(n))].append(v);rr[ro].append(v);allv.append(v)
    return ({k:np.mean(v,0) for k,v in cell.items()},{k:np.mean(v,0) for k,v in rr.items()},np.mean(allv,0))

def residualize(lines):
    mods={0:fit_means(lines,0),1:fit_means(lines,1)};out=[]
    for fol,A in lines:
        cell,rr,g=mods[1-fnum(fol)%2];n=len(A);B=[]
        for i,v in enumerate(A):
            ro=role(i,n);mu=cell.get((ro,lbin(n)),rr.get(ro,g));B.append(v-mu)
        out.append((fol,np.asarray(B)))
    return out

def dataset(lines,L,parity):
    X=[];Y=[]
    for fol,A in lines:
        if fnum(fol)%2!=parity:continue
        for t in range(LMAX,len(A)):
            X.append(A[t-L:t].reshape(-1));Y.append(A[t])
    return np.asarray(X,float),np.asarray(Y,float)

def r2_for_L(lines,L):
    total_model=0.;total_base=0.;ns=[]
    for trpar,tepar in ((1,0),(0,1)):
        Xtr,Ytr=dataset(lines,L,trpar);Xte,Yte=dataset(lines,L,tepar)
        mdl=Ridge(alpha=ALPHA_RIDGE,fit_intercept=True).fit(Xtr,Ytr)
        pred=mdl.predict(Xte);base=np.tile(Ytr.mean(0),(len(Yte),1))
        total_model+=float(np.sum((Yte-pred)**2));total_base+=float(np.sum((Yte-base)**2));ns.append(len(Yte))
    return 1.-total_model/total_base,ns

def statistic(observed=False,seed=None):
    raw=innovation_lines(observed,seed);lines=residualize(raw)
    vals={};counts=None
    for L in LGRID:
        z,n=r2_for_L(lines,L);vals[L]=float(z);counts=n
    return {"r2":vals,"lines":len(lines),"test_counts_per_parity":counts,"events_common":sum(max(0,len(A)-LMAX) for _,A in lines)}

def task(seed):
    x=statistic(False,seed);return seed,[x["r2"][L] for L in LGRID]

REAL=statistic(True,None)
print("REAL",json.dumps(REAL,separators=(",",":")),flush=True)

if __name__=="__main__":
    seeds=CAL_SEEDS+BLIND_SEEDS
    with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:
        rr=list(ex.map(task,seeds,chunksize=2))
    mp={s:np.array(v,float) for s,v in rr}
    C=np.vstack([mp[s] for s in CAL_SEEDS]);B=np.vstack([mp[s] for s in BLIND_SEEDS])
    rv=np.array([REAL["r2"][L] for L in LGRID],float)
    cmax=C.max(1);bmax=B.max(1);rmax=float(rv.max());q99=float(np.quantile(cmax,.99));accept=float(np.mean(bmax<=q99))
    pooled=np.r_[cmax,bmax];p=float((1+np.sum(pooled>=rmax))/(len(pooled)+1))
    q99L=np.quantile(C,.99,axis=0);resolved=[bool(rv[i]>q99L[i]) for i in range(len(LGRID))]
    if accept<.90:decision="THE_METRIC_DOES_NOT_RESOLVE_CALIBRATION_FAIL"
    elif rmax<=q99:decision="NO_LINEAR_RESIDUAL_MEMORY_RESOLVED"
    elif p>.01:decision="THE_METRIC_DOES_NOT_RESOLVE_PVALUE_FAIL"
    else:decision="RESIDUAL_PREDICTIVE_MEMORY_PRESENT"
    L90=None
    if decision=="RESIDUAL_PREDICTIVE_MEMORY_PRESENT":
        for i,L in enumerate(LGRID):
            if resolved[i] and rv[i]>=.9*rmax:
                L90=int(L);break
    inc_names=("2-1","3-2","5-3","8-5")
    Cinc=np.diff(C,axis=1);rinc=np.diff(rv)
    increments={inc_names[i]:{"real":float(rinc[i]),"cal_q99":float(np.quantile(Cinc[:,i],.99)),
                              "resolved_above_q99":bool(rinc[i]>np.quantile(Cinc[:,i],.99))} for i in range(4)}
    out={"programme":"STARS-MEM1","status":"complete","real":REAL,
         "calibration":{"n":len(C),"max_q99":q99,"per_L_q99":{str(L):float(q99L[i]) for i,L in enumerate(LGRID)}},
         "blind":{"n":len(B),"max_self_acceptance":accept},
         "real_max_R2":rmax,"add_one_p":p,"per_L_resolved":{str(L):resolved[i] for i,L in enumerate(LGRID)},
         "decision":decision,"L90":L90,"increments":increments}
    print("STARS_MEM1_JSON="+json.dumps(out,separators=(",",":")),flush=True)
