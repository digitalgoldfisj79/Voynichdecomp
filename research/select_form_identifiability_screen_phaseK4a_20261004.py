#!/usr/bin/env python3
# Phase K4a: discovery screen for a parameter-only identifiability criterion
# for the frozen SELECT -> FORM ENTRY5 + ROUTE.5 socket.
# Synthetic-only. Truth/oracle used ONLY to calibrate a future difficulty criterion.
# NO P70. No real Voynich inversion.
import json,math,urllib.request,collections
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed

K0URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f/research/select_form_socket_oracle_phaseK0_20261004.py"
k0={"__name__":"k0"};exec(compile(urllib.request.urlopen(K0URL,timeout=60).read().decode(),K0URL,"exec"),k0)

K=8;D=4;R=2;ENTRY=5.;ROUTE=.5;N=4000

# Real FORM continuation-group exposure per token. This is fixed renderer information,
# not planted truth.
gw=np.zeros(8,float)
lens=[]
for z in k0["routes"]:
    ids=[k0["PID"][p] for p in z]
    lens.append(len(ids))
    for i in range(len(ids)-1):
        g=k0["G_OF"][k0["PCLASS"][ids[i]]]
        gw[g]+=1
gw/=len(k0["routes"])
REAL_MEAN=float(np.mean(lens))

def stationary(A):
    p=np.ones(K)/K
    for _ in range(10000):
        q=p@A
        if np.max(np.abs(q-p))<1e-14:break
        p=q
    p/=p.sum()
    return p

def js(p,q):
    p=np.asarray(p,float);q=np.asarray(q,float);m=.5*(p+q)
    ok=p>0;a=float(np.sum(p[ok]*np.log(p[ok]/m[ok])))
    ok=q>0;b=float(np.sum(q[ok]*np.log(q[ok]/m[ok])))
    return .5*(a+b)

def probs(U,Ve,Vr):
    be=U@Ve
    se=np.log(np.maximum(k0["P_START_CLASS"],1e-30))[None,:]+be
    se-=se.max(1,keepdims=True);Qe=np.exp(se);Qe/=Qe.sum(1,keepdims=True)
    br=U@Vr;Qr=np.zeros((K,8,12),float)
    for s in range(K):
        for g in range(8):
            ok=k0["LEGAL"][g]&(k0["P_CONT"][g]>0)
            x=np.log(np.maximum(k0["P_CONT"][g,ok],1e-30))+br[s,ok]
            x-=x.max();v=np.exp(x);v/=v.sum();Qr[s,g,ok]=v
    return Qe,Qr

def geometry(A,U,Ve,Vr):
    Qe,Qr=probs(U,Ve,Vr)
    pair=[]
    entry=[]
    route=[]
    for i in range(K):
        for j in range(i):
            de=js(Qe[i],Qe[j])
            dr=sum(float(gw[g])*js(Qr[i,g],Qr[j,g]) for g in range(8))
            entry.append(de);route.append(dr);pair.append(de+dr)
    # nearest-neighbour separation for each state
    nearest=[]
    for i in range(K):
        ds=[]
        for j in range(K):
            if i==j:continue
            de=js(Qe[i],Qe[j])
            dr=sum(float(gw[g])*js(Qr[i,g],Qr[j,g]) for g in range(8))
            ds.append(de+dr)
        nearest.append(min(ds))
    st=stationary(A)
    # transition-row distinctness is parameter-only secondary diagnostic
    atr=[]
    for i in range(K):
        for j in range(i):atr.append(js(A[i],A[j]))
    return {
      "min_pair_js":float(min(pair)),
      "q10_pair_js":float(np.quantile(pair,.10)),
      "median_pair_js":float(np.median(pair)),
      "min_entry_js":float(min(entry)),
      "min_route_js":float(min(route)),
      "min_nearest_js":float(min(nearest)),
      "median_nearest_js":float(np.median(nearest)),
      "stationary_min":float(st.min()),
      "stationary_max":float(st.max()),
      "stationary_entropy":float(-np.sum(st*np.log(np.maximum(st,1e-30)))),
      "min_transition_js":float(min(atr)),
      "median_transition_js":float(np.median(atr)),
    }

def generate(seed):
    rng=np.random.default_rng(seed);A,pi=k0["source_graph"](K,D,rng)
    U,Ve0,Vr0=k0["controls"](K,R,1.,rng,"F1");Ve=Ve0*ENTRY;Vr=Vr0*ROUTE
    geom=geometry(A,U,Ve,Vr)
    z=k0["hidden_seq"](A,pi,N,rng);obs=[]
    for x in z:
        for _ in range(100):
            t=k0["gen_token"](int(x),U,Ve,Vr,rng)
            if t is not None:obs.append(t);break
        else:raise RuntimeError("nontermination")
    E=k0["emission"](obs,U,Ve,Vr);ll,g=k0["fb"](E,A,pi);pred=g.argmax(1)
    lengths=np.array([len(t) for t in obs])
    out={
      "seed":seed,**geom,
      "oracle_nmi":k0["nmi"](z,pred),"oracle_ari":k0["ari"](z,pred),
      "oracle_ll":float(ll),
      "mean_len":float(lengths.mean()),
      "mean_len_ratio":float(lengths.mean()/REAL_MEAN),
      "q95_len":float(np.quantile(lengths,.95)),
      "occupancy_min_realized":float(np.min(np.bincount(z,minlength=K))/N)
    }
    return out

if __name__=="__main__":
    seeds=[20262001+i for i in range(96)]
    outs=[]
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(generate,s):s for s in seeds}
        for fut in as_completed(futs):
            x=fut.result();outs.append(x)
            print("K4A_SCREEN_JSON="+json.dumps(x,separators=(",",":")),flush=True)
    outs.sort(key=lambda x:x["seed"])
    keys=["min_pair_js","q10_pair_js","median_pair_js","min_entry_js","min_route_js",
          "min_nearest_js","median_nearest_js","stationary_min","stationary_max",
          "stationary_entropy","min_transition_js","median_transition_js",
          "mean_len_ratio","occupancy_min_realized"]
    y=np.array([x["oracle_nmi"] for x in outs])
    cor={}
    for k in keys:
        a=np.array([x[k] for x in outs])
        cor[k]=float(np.corrcoef(a,y)[0,1]) if np.std(a)>0 else None
    summary={
      "n":len(outs),"real_mean_len":REAL_MEAN,
      "oracle_median":float(np.median(y)),
      "oracle_q10":float(np.quantile(y,.1)),
      "oracle_ge70":int(np.sum(y>=.70)),
      "correlations_with_oracle_nmi":cor,
      "records":outs
    }
    print("SELECT_FORM_PHASEK4A_JSON="+json.dumps(summary,separators=(",",":")),flush=True)
