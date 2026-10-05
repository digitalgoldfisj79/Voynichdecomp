#!/usr/bin/env python3
"""
CF0c: conditional-indistinguishability calibration.
For pair A/B, predict which surface variant occurred from external context only.
Same-source random variants should have no context gain above pair-frequency prior.
"""
import urllib.request, numpy as np, collections, json, math
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/af48d2b45ad416cfac9589c47f696fa6bb0d54ff/research/context_first_lexical_equivalence_20261005.py"
m={"__name__":"cf"};exec(compile(urllib.request.urlopen(BASE).read().decode(),BASE,"exec"),m)
CS=(0.03,0.1,0.3,1.0)

def feats(r):
    d={"lp="+str(r["lp"]):1.}
    for lag in (-2,-1,1,2):
        x=r[f"n{lag:+d}"]
        d[f"L{lag}="+str(x)]=1.
    d["PAIR11="+str(r["n-1"])+"|"+str(r["n+1"])]=1.
    d["PAIR22="+str(r["n-2"])+"|"+str(r["n+2"])]=1.
    d["QUAD="+str(r["n-2"])+"|"+str(r["n-1"])+"|"+str(r["n+1"])+"|"+str(r["n+2"])]=1.
    return d

def log2p(p,y):
    q=np.where(y==1,p,1-p)
    return np.log2(np.maximum(q,1e-12))

def eval_pair(rows,a,b,C):
    tr=[r for r in rows if r["fold"] in (2,3) and r["token"] in (a,b)]
    va=[r for r in rows if r["fold"]==4 and r["token"] in (a,b)]
    if min(sum(r["token"]==a for r in tr),sum(r["token"]==b for r in tr),
           sum(r["token"]==a for r in va),sum(r["token"]==b for r in va))<2:return None
    v=DictVectorizer();X=v.fit_transform([feats(r) for r in tr]);V=v.transform([feats(r) for r in va])
    yt=np.array([r["token"]==b for r in tr],int);yv=np.array([r["token"]==b for r in va],int)
    prior=(yt.sum()+.5)/(len(yt)+1.)
    base=float(np.mean(log2p(np.full(len(yv),prior),yv)))
    try:
        md=LogisticRegression(C=C,max_iter=500,solver="liblinear").fit(X,yt)
        p=md.predict_proba(V)[:,1]; ll=float(np.mean(log2p(p,yv)))
        auc=float(roc_auc_score(yv,p)) if len(set(yv))>1 else None
    except Exception:return None
    return {"gain":ll-base,"auc":auc,"na":sum(yv==0),"nb":sum(yv==1),"ntrain":len(yt)}

def split_cross(rows,types):
    vmap,_=m["context_vocab"](rows)
    base,glob=m["fit_baseline"](rows,vmap,(2,3))
    out={}
    for fold,key in ((2,"a"),(3,"b")):
        obs,exp=m["profiles"](rows,vmap,(fold,),set(types),base,glob)
        z={}
        for t in types:
            x=np.log(np.maximum(m["multiplier"](obs[t],exp[t]),1e-9)).ravel();x-=x.mean();n=np.linalg.norm(x)
            z[t]=x/n if n else x
        out[key]=z
    return out["a"],out["b"]

def one(shuffle=False):
    rows,_=m["synth_rows"](shuffle)
    types,dc,vc,tc=m["eligible_types"](rows)
    H2,H3=split_cross(rows,types)
    rec=[]
    for i,a in enumerate(types):
      for b in types[i+1:]:
        cross=.5*(float(H2[a]@H3[b])+float(H2[b]@H3[a]))
        truth=a.split("_")[0]==b.split("_")[0]
        rec.append((a,b,truth,cross))
    # candidate surface is top 5% by discovery-only cross fingerprint
    th=float(np.quantile([x[3] for x in rec],.95))
    cand=[x for x in rec if x[3]>=th]
    result={"threshold":th,"n_candidate":len(cand),"candidate_truth_fraction":float(np.mean([x[2] for x in cand]))}
    for C in CS:
        rr=[]
        for a,b,t,cross in cand:
            e=eval_pair(rows,a,b,C)
            if e is not None:rr.append((a,b,t,cross,e))
        print("ARM",shuffle,"C",C,"candidate",len(cand),"evaluated",len(rr),flush=True)
        if not rr:
            result[str(C)]={"n":0,"status":"no_evaluable_pairs"}
            continue
        y=np.array([x[2] for x in rr],int)
        gain=np.array([x[4]["gain"] for x in rr],float)
        auc=float(roc_auc_score(y,-gain)) if len(set(y))>1 else None
        trueg=gain[y==1];falseg=gain[y==0]
        qs={}
        for q in (.10,.20,.30,.40):
            cut=float(np.quantile(gain,q))
            sel=[x for x in rr if x[4]["gain"]<=cut]
            qs[str(q)]={"cut":cut,"n":len(sel),"precision":float(np.mean([x[2] for x in sel])) if sel else None,
                        "median_gain":float(np.median([x[4]["gain"] for x in sel])) if sel else None}
        result[str(C)]={"n":len(rr),"auc_minus_gain":auc,
                        "true_gain_median":float(np.median(trueg)) if len(trueg) else None,
                        "false_gain_median":float(np.median(falseg)) if len(falseg) else None,
                        "true_gain_q90":float(np.quantile(trueg,.90)) if len(trueg) else None,
                        "false_gain_q10":float(np.quantile(falseg,.10)) if len(falseg) else None,
                        "selection":qs,
                        "lowest20":[{"a":x[0],"b":x[1],"truth":bool(x[2]),"cross":x[3],"gain":x[4]["gain"],"auc":x[4]["auc"]} for x in sorted(rr,key=lambda z:z[4]["gain"])[:20]]}
    return result

ordered=one(False); print("CF0C_ORDERED="+json.dumps(ordered,separators=(",",":")),flush=True)
shuffled=one(True); print("CF0C_SHUFFLED="+json.dumps(shuffled,separators=(",",":")),flush=True)
out={"ordered":ordered,"shuffled":shuffled}
print("CF0C="+json.dumps(out,separators=(",",":")),flush=True)
