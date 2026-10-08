#!/usr/bin/env python3
# STARS-REL1 — preregistered 2026-10-08.
import json,math,os,urllib.request
import numpy as np
from concurrent.futures import ProcessPoolExecutor

BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/cfebbd657e31cc8a49e1513230e7e2c34514fbcc/research/stars_innov2_position_20261008.py"
src=urllib.request.urlopen(BASE,timeout=120).read().decode()
ns={"__name__":"innov2_module"}
exec(compile(src,BASE,"exec"),ns)

generate=ns["generate"];position_adjust=ns["position_adjust"];fnum=ns["fnum"];K=ns["K"]
LAGS=(1,2,3,5,10);L2=10.0
FIT_SEEDS=list(range(202610086000,202610086300))
CAL_SEEDS=list(range(202610086300,202610086400))
BLIND_SEEDS=list(range(202610086400,202610086500))

def arrays(lines,parity,lags):
    P=[];Y=[];FF=[]
    start=min(lags)
    for seq in lines:
        if not seq or fnum(seq[0]["folio"])%2!=parity:continue
        for t in range(start,len(seq)):
            # For single-lag model require that specific lag exists.
            if any(t<lag for lag in lags):
                if len(lags)==1:continue
            F=np.zeros((K,len(lags)),float)
            active=False
            for j,lag in enumerate(lags):
                if t>=lag:
                    F[int(seq[t-lag]["y"]),j]=1.;active=True
            if not active:continue
            P.append(seq[t]["p"]);Y.append(int(seq[t]["y"]));FF.append(F)
    return np.asarray(P,float),np.asarray(Y,int),np.asarray(FF,float)

def nll_grad_hess(beta,P,Y,F):
    sc=np.log(np.maximum(P,1e-15))+np.tensordot(F,beta,axes=([2],[0]))
    mx=sc.max(1,keepdims=True);Q=np.exp(sc-mx);Q/=Q.sum(1,keepdims=True)
    loss=-float(np.sum(np.log(np.maximum(Q[np.arange(len(Y)),Y],1e-300))))+.5*L2*float(np.dot(beta,beta))
    Ef=np.einsum("nk,nkj->nj",Q,F)
    Fy=F[np.arange(len(Y)),Y,:]
    grad=(Ef-Fy).sum(0)+L2*beta
    J=len(beta)
    E2=np.einsum("nk,nkj,nkl->jl",Q,F,F)
    H=np.eye(J)*L2 + E2 - Ef.T@Ef
    return loss,grad,H

def fit_beta(P,Y,F):
    J=F.shape[2];b=np.zeros(J);old=1e300
    for _ in range(20):
        loss,g,H=nll_grad_hess(b,P,Y,F)
        if np.max(np.abs(g))<1e-8:break
        step=np.linalg.solve(H+np.eye(J)*1e-10,g);t=1.
        while t>1e-6:
            c=b-t*step;nl,_,_=nll_grad_hess(c,P,Y,F)
            if nl<=loss+1e-10:b=c;old=nl;break
            t*=.5
        if t<=1e-6:break
    return b

def score(P,Y,F,beta):
    base=-np.log2(np.maximum(P[np.arange(len(Y)),Y],1e-300)).sum()
    sc=np.log(np.maximum(P,1e-15))+np.tensordot(F,beta,axes=([2],[0]))
    mx=sc.max(1,keepdims=True);Q=np.exp(sc-mx);Q/=Q.sum(1,keepdims=True)
    rel=-np.log2(np.maximum(Q[np.arange(len(Y)),Y],1e-300)).sum()
    return float(base),float(rel),len(Y)

def crossfit(lines,lags):
    B=R=N=0;betas={}
    for tr,te in ((1,0),(0,1)):
        Pt,Yt,Ft=arrays(lines,tr,lags);Pv,Yv,Fv=arrays(lines,te,lags)
        beta=fit_beta(Pt,Yt,Ft);bb,rr,nn=score(Pv,Yv,Fv,beta)
        B+=bb;R+=rr;N+=nn;betas[f"{tr}_to_{te}"]=beta.tolist()
    return {"gain":float((B-R)/N),"n":N,"betas":betas}

def statistic(observed=False,seed=None):
    lines=position_adjust(generate(observed,seed))
    full=crossfit(lines,LAGS)
    singles={str(l):crossfit(lines,(l,)) for l in LAGS}
    return {"full":full,"singles":singles,"n_lines":len(lines)}

def task(seed):
    x=statistic(False,seed)
    return seed,[x["full"]["gain"]]+[x["singles"][str(l)]["gain"] for l in LAGS]

REAL=statistic(True,None)
print("REAL",json.dumps(REAL,separators=(",",":")),flush=True)

if __name__=="__main__":
    seeds=FIT_SEEDS+CAL_SEEDS+BLIND_SEEDS
    with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:
        rr=list(ex.map(task,seeds,chunksize=1))
    mp={s:np.array(v,float) for s,v in rr}
    F=np.vstack([mp[s] for s in FIT_SEEDS]);C=np.vstack([mp[s] for s in CAL_SEEDS]);B=np.vstack([mp[s] for s in BLIND_SEEDS])
    rv=np.r_[REAL["full"]["gain"],[REAL["singles"][str(l)]["gain"] for l in LAGS]]
    # primary full gain: q99 from dedicated calibration split.
    fq=float(np.quantile(C[:,0],.99));fa=float(np.mean(B[:,0]<=fq));ref=np.r_[C[:,0],B[:,0]]
    fp=float((1+np.sum(ref>=rv[0]))/(len(ref)+1))
    if fa<.90:decision="THE_METRIC_DOES_NOT_RESOLVE_CALIBRATION_FAIL"
    elif rv[0]<=fq:decision="NO_ORDER_SPECIFIC_RETURN_AVOIDANCE_MEMORY_RESOLVED"
    elif fp>.01:decision="THE_METRIC_DOES_NOT_RESOLVE_PVALUE_FAIL"
    else:decision="ORDER_SPECIFIC_RELATIONAL_MEMORY_PRESENT"
    secondary={"opened":False}
    if decision=="ORDER_SPECIFIC_RELATIONAL_MEMORY_PRESENT":
        mu=F[:,1:].mean(0);sd=np.maximum(F[:,1:].std(0,ddof=1),1e-12)
        Cz=(C[:,1:]-mu)/sd;Bz=(B[:,1:]-mu)/sd;rz=(rv[1:]-mu)/sd
        cm=Cz.max(1);bm=Bz.max(1);rq=float(rz.max());mq=float(np.quantile(cm,.99));ma=float(np.mean(bm<=mq))
        mref=np.r_[cm,bm];mpv=float((1+np.sum(mref>=rq))/(len(mref)+1));perq=np.quantile(Cz,.99,axis=0)
        globalpass=bool(ma>=.90 and rq>mq and mpv<=.01)
        resolved={str(l):bool(globalpass and rz[i]>perq[i]) for i,l in enumerate(LAGS)}
        secondary={"opened":True,"real_Z":{str(l):float(rz[i]) for i,l in enumerate(LAGS)},
                   "per_lag_q99_Z":{str(l):float(perq[i]) for i,l in enumerate(LAGS)},
                   "maxZ_q99":mq,"blind_acceptance":ma,"add_one_p":mpv,
                   "global_pass":globalpass,"resolved_lags":resolved}
    out={"programme":"STARS-REL1","status":"complete","real":REAL,
         "primary":{"real_gain_bits":float(rv[0]),"cal_q99":fq,"blind_acceptance":fa,"add_one_p":fp},
         "decision":decision,"secondary":secondary}
    print("STARS_REL1_JSON="+json.dumps(out,separators=(",",":")),flush=True)
