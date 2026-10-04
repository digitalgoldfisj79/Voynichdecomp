#!/usr/bin/env python3
# Phase L2b: blind contextual alias resolution using neighboring SELECT posteriors.
# Corrected structured source generators from L2. Synthetic-only. NO P70.
import json,urllib.request
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score,roc_auc_score
from concurrent.futures import ProcessPoolExecutor,as_completed

URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/c6384a58449b0f27f959be580c5e9613d71d83eb/research/structured_source_context_calibration_phaseL2_20261004.py"
m={"__name__":"l2lib"};exec(compile(urllib.request.urlopen(URL,timeout=60).read().decode(),URL,"exec"),m)

TRAIN=900;RADIUS=2

def softmax_rows(L):
    X=L-L.max(1,keepdims=True);Q=np.exp(X);Q/=Q.sum(1,keepdims=True);return Q

def context_features(Q,r=RADIUS):
    n,d=Q.shape;F=np.zeros((n,2*r*d),float)
    k=0
    for off in list(range(-r,0))+list(range(1,r+1)):
        if off<0:F[-off:,k*d:(k+1)*d]=Q[:n+off]
        else:F[:n-off,k*d:(k+1)*d]=Q[off:]
        k+=1
    return F

def fit_split(Q):
    cur=Q.argmax(1);F=context_features(Q)
    models={}
    for s in range(m["NSIG"]):
        ix=np.where((cur[:TRAIN]==s))[0]
        if len(ix)<20:continue
        km=KMeans(n_clusters=2,n_init=20,random_state=20261004+s,max_iter=300).fit(F[ix])
        models[s]=km
    return models,F,cur

def decode(Q,models,F,cur):
    n=len(Q);pred=np.empty(n,int);post=np.zeros((n,m["V"]),float)
    for t in range(n):
        s=int(cur[t]);km=models.get(s)
        if km is None:
            b=0;p=np.array([.5,.5])
        else:
            d=((km.cluster_centers_-F[t])**2).sum(1)
            x=-d/max(np.std(d)+1e-6,1e-3);x-=x.max();p=np.exp(x);p/=p.sum();b=int(p.argmax())
        # branch 0 -> source s, branch1 -> s+8
        pred[t]=s+8*b
        post[t,s]=p[0];post[t,s+8]=p[1]
    return pred,post

def collision_nmi(z,pred):
    vals=[]
    for s in range(m["NSIG"]):
        ix=np.where(m["SIG_OF"][z]==s)[0]
        if len(ix)>4 and len(np.unique(z[ix]))>1:
            vals.append(normalized_mutual_info_score(z[ix],pred[ix]))
    return float(np.mean(vals)) if vals else float("nan")

def pair_auc(z,post,seed):
    rng=np.random.default_rng(seed);ys=[];sc=[]
    for s in range(m["NSIG"]):
        ix=np.where(m["SIG_OF"][z]==s)[0]
        if len(ix)<3:continue
        for _ in range(1000):
            a,b=rng.choice(ix,2,replace=False)
            ys.append(int(z[a]==z[b]));sc.append(float(np.dot(post[a],post[b])))
    return float(roc_auc_score(ys,sc)) if len(set(ys))>1 else float("nan")

def metrics(z,pred,post,seed):
    return {"source_nmi":float(normalized_mutual_info_score(z,pred)),
            "collision_nmi":collision_nmi(z,pred),
            "same_source_auc":pair_auc(z,post,seed),
            "exact_source_accuracy":float(np.mean(z==pred))}

def one(fam,seed):
    rng=np.random.default_rng(seed);U,Ve,Vr=m["encoder"](seed)
    secs=[];alls=[];alld=[];allsec=[];allz=[]
    for sec in range(m["NSEC"]):
        z,A,pi=m["source_sequence"](fam,sec,m["SECLEN"],rng)
        obs,L=m["render_and_ll"](z,U,Ve,Vr,rng)
        # control: exact one-hot signature posterior
        Qc=np.zeros((len(z),m["NSIG"]),float);Qc[np.arange(len(z)),m["SIG_OF"][z]]=1.
        # FORM: exact likelihood posterior over SELECT signatures, uniform prior
        Qf=softmax_rows(L)
        rr={"section":sec}
        for ci,(name,Q) in enumerate((("CONTROL",Qc),("FORM",Qf))):
            models,F,cur=fit_split(Q);pred,post=decode(Q,models,F,cur)
            sl=slice(TRAIN,None)
            rr[name]=metrics(z[sl],pred[sl],post[sl],seed+sec*37+ci)
        rr["sig_decode_acc"]=float(np.mean(Qf.argmax(1)==m["SIG_OF"][z]))
        secs.append(rr);allz.extend(z.tolist());alls.extend(m["SIG_OF"][z].tolist());alld.extend(Qf.argmax(1).tolist());allsec.extend([sec]*len(z))
    def agg(ch,k):return float(np.nanmean([x[ch][k] for x in secs]))
    return {"family":fam,"seed":seed,
      "source_section_nmi":float(normalized_mutual_info_score(allsec,allz)),
      "signature_section_nmi":float(normalized_mutual_info_score(allsec,alls)),
      "decoded_signature_section_nmi":float(normalized_mutual_info_score(allsec,alld)),
      "sig_decode_acc":float(np.mean(np.array(alls)==np.array(alld))),
      "control":{"source_nmi":agg("CONTROL","source_nmi"),"collision_nmi":agg("CONTROL","collision_nmi"),"auc":agg("CONTROL","same_source_auc"),"accuracy":agg("CONTROL","exact_source_accuracy")},
      "form":{"source_nmi":agg("FORM","source_nmi"),"collision_nmi":agg("FORM","collision_nmi"),"auc":agg("FORM","same_source_auc"),"accuracy":agg("FORM","exact_source_accuracy")},
      "sections":secs}

if __name__=="__main__":
    specs=[(f,s) for f in ("LANG","NOTATION","TABLE") for s in (20262401,20262402,20262403,20262404,20262405)]
    out=[]
    with ProcessPoolExecutor(max_workers=15) as ex:
        fut={ex.submit(one,*x):x for x in specs}
        for q in as_completed(fut):
            r=q.result();out.append(r);print("L2B_REP_JSON="+json.dumps(r,separators=(",",":")),flush=True)
    summary={}
    for fam in ("LANG","NOTATION","TABLE"):
        rr=[x for x in out if x["family"]==fam]
        med=lambda ch,k:float(np.median([x[ch][k] for x in rr]))
        summary[fam]={"n":len(rr),
          "sig_decode_acc":float(np.median([x["sig_decode_acc"] for x in rr])),
          "source_section_nmi":float(np.median([x["source_section_nmi"] for x in rr])),
          "signature_section_nmi":float(np.median([x["signature_section_nmi"] for x in rr])),
          "control_collision_nmi":med("control","collision_nmi"),"control_auc":med("control","auc"),"control_accuracy":med("control","accuracy"),
          "form_collision_nmi":med("form","collision_nmi"),"form_auc":med("form","auc"),"form_accuracy":med("form","accuracy"),
          "form_source_nmi":med("form","source_nmi")}
    print("STRUCTURED_SOURCE_PHASEL2B_JSON="+json.dumps({"radius":RADIUS,"train":TRAIN,"summary":summary,"replicates":out},separators=(",",":")),flush=True)
