#!/usr/bin/env python3
# Phase L0b: oracle-context identifiability of structured sources before FORM.
import json,math,urllib.request
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score

URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/a421f5b378f691784730e4a1afd0e7d7bca75123/research/structured_source_calibration_phaseL0_20261004.py"
m={"__name__":"L0lib"};exec(compile(urllib.request.urlopen(URL,timeout=60).read().decode(),URL,"exec"),m)
V=m["V"];NSIG=m["NSIG"];NSEC=m["NSEC"];SECLEN=m["SECLEN"];SIG_OF=m["SIG_OF"]

def fb_known(sig,A,pi):
    n=len(sig)
    logA=np.log(np.maximum(A,1e-300));logpi=np.log(np.maximum(pi,1e-300))
    E=np.full((n,V),-1e300,float)
    for v in range(V): E[:,v]=np.where(sig==SIG_OF[v],0.0,-1e300)
    al=np.empty((n,V));sc=np.empty(n)
    x=logpi+E[0];mx=x.max();sc[0]=mx+np.log(np.exp(x-mx).sum());al[0]=x-sc[0]
    for t in range(1,n):
        M=al[t-1][:,None]+logA;mm=M.max(0);pr=mm+np.log(np.exp(M-mm).sum(0))
        x=pr+E[t];mx=x.max();sc[t]=mx+np.log(np.exp(x-mx).sum());al[t]=x-sc[t]
    be=np.zeros((n,V))
    for t in range(n-2,-1,-1):
        M=logA+E[t+1][None,:]+be[t+1][None,:]
        mm=M.max(1);be[t]=mm+np.log(np.exp(M-mm[:,None]).sum(1))-sc[t+1]
    lg=al+be;mm=lg.max(1,keepdims=True);g=np.exp(lg-mm);g/=g.sum(1,keepdims=True)
    # Viterbi
    dp=logpi+E[0];back=np.zeros((n,V),int)
    for t in range(1,n):
        M=dp[:,None]+logA;back[t]=M.argmax(0);dp=M.max(0)+E[t]
    z=np.empty(n,int);z[-1]=dp.argmax()
    for t in range(n-2,-1,-1):z[t]=back[t+1,z[t+1]]
    return g,z

def collision_nmi(z,pred):
    vals=[]
    for s in range(NSIG):
        ix=np.where(SIG_OF[z]==s)[0]
        if len(ix)>10:
            vals.append(normalized_mutual_info_score(z[ix],pred[ix]))
    return float(np.mean(vals))

def auc(z,g,seed):
    rng=np.random.default_rng(seed);ys=[];ss=[]
    for s in range(NSIG):
        ix=np.where(SIG_OF[z]==s)[0]
        if len(ix)<5:continue
        for _ in range(1500):
            a,b=rng.choice(ix,2,replace=False)
            ys.append(int(z[a]==z[b]));ss.append(float(np.dot(g[a],g[b])))
    return float(roc_auc_score(ys,ss))

def sec_mi(sections,sigs):
    return float(normalized_mutual_info_score(sections,sigs))

out=[]
for fam in ("LANG","NOTATION","TABLE"):
    for seed in range(20263001,20263021):
        rng=np.random.default_rng(seed)
        zs=[];sigs=[];secs=[];gammas=[];preds=[]
        for sec in range(NSEC):
            z,A=m["source_sequence"](fam,sec,SECLEN,rng)
            sig=SIG_OF[z]
            g,p=fb_known(sig,A,m["section_prior"](sec))
            zs.append(z);sigs.append(sig);secs.append(np.full(SECLEN,sec));gammas.append(g);preds.append(p)
        z=np.concatenate(zs);sig=np.concatenate(sigs);secv=np.concatenate(secs);g=np.vstack(gammas);p=np.concatenate(preds)
        r={
          "family":fam,"seed":seed,
          "source_accuracy":float(np.mean(z==p)),
          "source_nmi":float(normalized_mutual_info_score(z,p)),
          "collision_nmi":collision_nmi(z,p),
          "same_source_auc":auc(z,g,seed),
          "section_mi_signature":sec_mi(secv,sig),
          "section_mi_source":sec_mi(secv,z)
        }
        out.append(r)
        print("ORACLE_CONTEXT_REP_JSON="+json.dumps(r,separators=(",",":")),flush=True)

summary={}
for fam in ("LANG","NOTATION","TABLE"):
    rr=[r for r in out if r["family"]==fam]
    summary[fam]={k:float(np.median([x[k] for x in rr])) for k in
      ("source_accuracy","source_nmi","collision_nmi","same_source_auc","section_mi_signature","section_mi_source")}
    summary[fam]["p10_collision_nmi"]=float(np.quantile([x["collision_nmi"] for x in rr],.1))
    summary[fam]["p10_same_source_auc"]=float(np.quantile([x["same_source_auc"] for x in rr],.1))
print("ORACLE_CONTEXT_PHASEL0B_JSON="+json.dumps({"n_per_family":20,"summary":summary},separators=(",",":")),flush=True)
