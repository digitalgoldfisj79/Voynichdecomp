#!/usr/bin/env python3
"""
CF0d: calibrated equivalence = strong shared contextual fingerprint above shuffled-order null
AND no heldout context evidence distinguishing the two surface variants.
Synthetic truth only. No Voynich.
"""
import urllib.request, numpy as np, json
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/af48d2b45ad416cfac9589c47f696fa6bb0d54ff/research/context_first_lexical_equivalence_20261005.py"
m={"__name__":"cf"};exec(compile(urllib.request.urlopen(BASE).read().decode(),BASE,"exec"),m)
C=.1

def feats(r):
    d={"lp="+str(r["lp"]):1.}
    for lag in (-2,-1,1,2): d[f"L{lag}="+str(r[f'n{lag:+d}'])]=1.
    d["near="+str(r["n-1"])+"|"+str(r["n+1"])]=1.
    return d

def fingerprint_records(rows):
    types,dc,vc,tc=m["eligible_types"](rows)
    if len(types)<2:return [],types
    vmap,_=m["context_vocab"](rows)
    base,glob=m["fit_baseline"](rows,vmap,(2,3))
    halves={}
    for fold in (2,3):
        obs,exp=m["profiles"](rows,vmap,(fold,),set(types),base,glob)
        z={}
        for t in types:
            x=np.log(np.maximum(m["multiplier"](obs[t],exp[t]),1e-9)).ravel()
            x-=x.mean();n=np.linalg.norm(x);z[t]=x/n if n else x
        halves[fold]=z
    out=[]
    for i,a in enumerate(types):
        for b in types[i+1:]:
            cross=.5*(float(halves[2][a]@halves[3][b])+float(halves[2][b]@halves[3][a]))
            truth=(a.split("_")[0]==b.split("_")[0])
            out.append((a,b,truth,cross))
    return out,types

def discrim_gain(rows,a,b):
    tr=[r for r in rows if r["fold"] in (2,3) and r["token"] in (a,b)]
    va=[r for r in rows if r["fold"]==4 and r["token"] in (a,b)]
    na=sum(r["token"]==a for r in tr);nb=sum(r["token"]==b for r in tr)
    vaa=sum(r["token"]==a for r in va);vbb=sum(r["token"]==b for r in va)
    if min(na,nb,vaa,vbb)<2:return None
    v=DictVectorizer();X=v.fit_transform([feats(r) for r in tr]).tocsr();V=v.transform([feats(r) for r in va]).tocsr()
    X.indices=X.indices.astype(np.int32);X.indptr=X.indptr.astype(np.int32)
    V.indices=V.indices.astype(np.int32);V.indptr=V.indptr.astype(np.int32)
    y=np.array([r["token"]==b for r in tr],int);z=np.array([r["token"]==b for r in va],int)
    prior=(y.sum()+.5)/(len(y)+1)
    base=np.mean(np.log2(np.maximum(np.where(z==1,prior,1-prior),1e-12)))
    try:
        md=LogisticRegression(C=C,max_iter=500,solver="liblinear").fit(X,y)
        p=md.predict_proba(V)[:,1]
    except Exception:return None
    ll=np.mean(np.log2(np.maximum(np.where(z==1,p,1-p),1e-12)))
    return float(ll-base)

ordered,_=m["synth_rows"](False)
shuffled,_=m["synth_rows"](True)
OR,_=fingerprint_records(ordered); SR,_=fingerprint_records(shuffled)
if not OR or not SR: raise RuntimeError(("no_fingerprint_records",len(OR),len(SR)))
shuffle_cross=np.array([x[3] for x in SR],float)
results={}
for q in (.95,.975,.99,.995):
    th=float(np.quantile(shuffle_cross,q))
    cand=[x for x in OR if x[3]>th]
    ev=[]
    for a,b,t,cross in cand:
        g=discrim_gain(ordered,a,b)
        if g is not None:ev.append((a,b,t,cross,g))
    # equivalence means shared fingerprint + classifier cannot improve heldout:
    sel=[x for x in ev if x[4]<=0]
    # also report increasingly conservative nonpositive margins
    strict=[x for x in ev if x[4]<=-.005]
    results[str(q)]={
      "threshold":th,"candidates":len(cand),"evaluable":len(ev),
      "candidate_truth_fraction":float(np.mean([x[2] for x in ev])) if ev else None,
      "selected_n":len(sel),"selected_truth_fraction":float(np.mean([x[2] for x in sel])) if sel else None,
      "strict_n":len(strict),"strict_truth_fraction":float(np.mean([x[2] for x in strict])) if strict else None,
      "selected":[{"a":x[0],"b":x[1],"truth":bool(x[2]),"cross":x[3],"discrim_gain":x[4]} for x in sorted(sel,key=lambda z:-z[3])[:30]]
    }
# Shuffled arm using SAME thresholds: should not return structured equivalence above its own upper tail except nominally.
shout={}
for q in (.95,.975,.99,.995):
    th=float(np.quantile(shuffle_cross,q));cand=[x for x in SR if x[3]>th];ev=[]
    for a,b,t,cross in cand:
        g=discrim_gain(shuffled,a,b)
        if g is not None:ev.append((a,b,t,cross,g))
    sel=[x for x in ev if x[4]<=0]
    shout[str(q)]={"candidates":len(cand),"evaluable":len(ev),"selected_n":len(sel),
                   "truth_fraction":float(np.mean([x[2] for x in sel])) if sel else None}
out={"ordered":results,"shuffled":shout,
     "ordered_fingerprint_auc":None,
     "note":"Synthetic calibration only. Choose criterion for later real assay only if high precision is achieved without source labels in criterion."}
print("CF0D="+json.dumps(out,separators=(",",":")),flush=True)
