#!/usr/bin/env python3
# Phase L1: true-transition oracle identifiability for structured sources.
# Synthetic-only. Tests whether source sequence context CAN resolve source items
# deliberately aliased at the SELECT socket. NO P70.
import json,math,urllib.request
import numpy as np
from sklearn.metrics import normalized_mutual_info_score,roc_auc_score
from concurrent.futures import ProcessPoolExecutor,as_completed

L0URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/a421f5b378f691784730e4a1afd0e7d7bca75123/research/structured_source_calibration_phaseL0_20261004.py"
m={"__name__":"l0lib"};exec(compile(urllib.request.urlopen(L0URL,timeout=60).read().decode(),L0URL,"exec"),m)
V=m["V"];NSIG=m["NSIG"];NSEC=m["NSEC"];SECLEN=m["SECLEN"];NFIT=m["NFIT"];NVAL=m["NVAL"];SIG_OF=m["SIG_OF"]

def fb(obs,A,pi,B):
    n=len(obs);K=A.shape[0]
    la=np.log(np.maximum(A,1e-300));lp=np.log(np.maximum(pi,1e-300))
    le=np.log(np.maximum(B[:,obs].T,1e-300))
    al=np.empty((n,K));sc=np.empty(n)
    a=lp+le[0];mx=a.max();sc[0]=mx+math.log(np.exp(a-mx).sum());al[0]=a-sc[0]
    for t in range(1,n):
        M=al[t-1][:,None]+la;mm=M.max(0);pr=mm+np.log(np.exp(M-mm).sum(0))
        a=pr+le[t];mx=a.max();sc[t]=mx+math.log(np.exp(a-mx).sum());al[t]=a-sc[t]
    be=np.zeros((n,K))
    for t in range(n-2,-1,-1):
        M=la+le[t+1][None,:]+be[t+1][None,:];mm=M.max(1)
        be[t]=mm+np.log(np.exp(M-mm[:,None]).sum(1))-sc[t+1]
    lg=al+be;mm=lg.max(1,keepdims=True);g=np.exp(lg-mm);g/=g.sum(1,keepdims=True)
    return float(sc.sum()),g

def collision_nmi(z,pred):
    vals=[]
    for s in range(NSIG):
        ix=np.where(SIG_OF[z]==s)[0]
        if len(ix)>3 and len(np.unique(z[ix]))>1:
            vals.append(normalized_mutual_info_score(z[ix],pred[ix]))
    return float(np.mean(vals)) if vals else float("nan")

def pair_auc(z,gamma,seed):
    rng=np.random.default_rng(seed);ys=[];scores=[]
    for s in range(NSIG):
        ix=np.where(SIG_OF[z]==s)[0]
        if len(ix)<3:continue
        for _ in range(1500):
            a,b=rng.choice(ix,2,replace=False)
            ys.append(int(z[a]==z[b]));scores.append(float(np.dot(gamma[a],gamma[b])))
    return float(roc_auc_score(ys,scores)) if len(set(ys))>1 else float("nan")

def deterministic_B():
    B=np.full((V,NSIG),1e-12,float)
    for v in range(V):B[v,SIG_OF[v]]=1.0
    B/=B.sum(1,keepdims=True)
    return B

def form_channel_B(ts,ds):
    # Estimate SELECT-signature -> decoded-signature channel on non-test portion only.
    C=np.ones((NSIG,NSIG),float)*.5
    ntrain=NFIT+NVAL
    for a,b in zip(ts[:ntrain],ds[:ntrain]):C[int(a),int(b)]+=1
    C/=C.sum(1,keepdims=True)
    return C[SIG_OF]

def one(fam,seed):
    rng=np.random.default_rng(seed)
    U,Ve,Vr=m["encoder"](seed)
    secs=[];allz=[];alls=[];alld=[];allsec=[]
    for sec in range(NSEC):
        z,A=m["source_sequence"](fam,sec,SECLEN,rng)
        obs,ts=m["render"](z,U,Ve,Vr,rng)
        ds,_=m["decode_sig"](obs,U,Ve,Vr)
        pi=m["section_prior"](sec)
        B0=deterministic_B();B1=form_channel_B(ts,ds)
        _,g0=fb(ts,A,pi,B0);_,g1=fb(ds,A,pi,B1)
        sl=slice(NFIT+NVAL,None);zte=z[sl]
        rec={"section":sec,
             "sig_decode_acc":float(np.mean(ds==ts)),
             "control_nmi":float(normalized_mutual_info_score(zte,g0[sl].argmax(1))),
             "control_collision_nmi":collision_nmi(zte,g0[sl].argmax(1)),
             "control_same_source_auc":pair_auc(zte,g0[sl],seed+sec*31),
             "form_nmi":float(normalized_mutual_info_score(zte,g1[sl].argmax(1))),
             "form_collision_nmi":collision_nmi(zte,g1[sl].argmax(1)),
             "form_same_source_auc":pair_auc(zte,g1[sl],seed+sec*37)}
        secs.append(rec)
        allz.extend(z.tolist());alls.extend(ts.tolist());alld.extend(ds.tolist());allsec.extend([sec]*SECLEN)
    agg=lambda k:float(np.nanmean([x[k] for x in secs]))
    return {"family":fam,"seed":seed,
            "source_section_nmi":float(normalized_mutual_info_score(allsec,allz)),
            "true_sig_section_nmi":float(normalized_mutual_info_score(allsec,alls)),
            "decoded_sig_section_nmi":float(normalized_mutual_info_score(allsec,alld)),
            "sig_decode_acc":agg("sig_decode_acc"),
            "control":{"source_nmi":agg("control_nmi"),"collision_nmi":agg("control_collision_nmi"),
                       "same_source_auc":agg("control_same_source_auc")},
            "form":{"source_nmi":agg("form_nmi"),"collision_nmi":agg("form_collision_nmi"),
                    "same_source_auc":agg("form_same_source_auc")},"sections":secs}

if __name__=="__main__":
    specs=[(f,s) for f in ("LANG","NOTATION","TABLE") for s in range(20262101,20262106)]
    out=[]
    with ProcessPoolExecutor(max_workers=15) as ex:
        fut={ex.submit(one,*x):x for x in specs}
        for q in as_completed(fut):
            r=q.result();out.append(r)
            print("STRUCTURED_SOURCE_ORACLE_REP_JSON="+json.dumps(r,separators=(",",":")),flush=True)
    summary={}
    for fam in ("LANG","NOTATION","TABLE"):
        rr=[x for x in out if x["family"]==fam]
        def med(path1,path2=None):
            v=[(x[path1] if path2 is None else x[path1][path2]) for x in rr]
            return float(np.median(v))
        summary[fam]={
          "n":len(rr),
          "source_section_nmi_median":med("source_section_nmi"),
          "sig_decode_acc_median":med("sig_decode_acc"),
          "control_source_nmi_median":med("control","source_nmi"),
          "control_collision_nmi_median":med("control","collision_nmi"),
          "control_same_source_auc_median":med("control","same_source_auc"),
          "form_source_nmi_median":med("form","source_nmi"),
          "form_collision_nmi_median":med("form","collision_nmi"),
          "form_same_source_auc_median":med("form","same_source_auc")
        }
    print("STRUCTURED_SOURCE_PHASEL1_JSON="+json.dumps({"summary":summary,"replicates":out},separators=(",",":")),flush=True)
