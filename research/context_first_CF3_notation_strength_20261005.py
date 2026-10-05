#!/usr/bin/env python3
import urllib.request,json,math
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/af48d2b45ad416cfac9589c47f696fa6bb0d54ff/research/context_first_lexical_equivalence_20261005.py"
m={"__name__":"cf"};exec(compile(urllib.request.urlopen(URL,timeout=120).read().decode(),URL,"exec"),m)
C=.3
def fix(X):
 X=X.tocsr();X.indices=X.indices.astype(np.int32,copy=False);X.indptr=X.indptr.astype(np.int32,copy=False);return X
def feats(r):
 d={}
 for lag in (-2,-1,1,2):d[f"L{lag}="+str(r[f"n{lag:+d}"])]=1.
 d["P11="+str(r["n-1"])+"|"+str(r["n+1"])]=1.;d["P22="+str(r["n-2"])+"|"+str(r["n+2"])]=1.
 return d
def lp(p,y):
 q=np.where(y==1,p,1-p);return np.log2(np.maximum(q,1e-12))
def val_gain(rows,a,b):
 tr=[r for r in rows if r["fold"] in (2,3) and r["token"] in (a,b)]
 va=[r for r in rows if r["fold"]==4 and r["token"] in (a,b)]
 if min(sum(r["token"]==a for r in tr),sum(r["token"]==b for r in tr),sum(r["token"]==a for r in va),sum(r["token"]==b for r in va))<2:return None
 v=DictVectorizer();X=fix(v.fit_transform([feats(r) for r in tr]));V=fix(v.transform([feats(r) for r in va]))
 y=np.array([r["token"]==b for r in tr],int);yv=np.array([r["token"]==b for r in va],int);pr=(y.sum()+.5)/(len(y)+1)
 base=float(np.mean(lp(np.full(len(yv),pr),yv)))
 try:
  md=LogisticRegression(C=C,max_iter=500,solver="liblinear").fit(X,y)
  return float(np.mean(lp(md.predict_proba(V)[:,1],yv))-base)
 except Exception:return None
def split_emb(rows,types,vmap):
 base,glob=m["fit_baseline"](rows,vmap,(2,3));out=[]
 for f in (2,3):
  o,e=m["profiles"](rows,vmap,(f,),set(types),base,glob);z={}
  for t in types:
   x=np.log(np.maximum(m["multiplier"](o[t],e[t]),1e-9)).ravel();x-=x.mean();n=np.linalg.norm(x);z[t]=x/n if n else x
  out.append(z)
 return out
def select(rows):
 types,dc,vc,tc=m["eligible_types"](rows);vmap,_=m["context_vocab"](rows);h2,h3=split_emb(rows,types,vmap)
 rec=[]
 for i,a in enumerate(types):
  for b in types[i+1:]:
   cr=.5*(float(h2[a]@h3[b])+float(h2[b]@h3[a]));rec.append([a,b,cr,None])
 th=float(np.quantile([x[2] for x in rec],.99));cand=[x for x in rec if x[2]>=th]
 for x in cand:x[3]=val_gain(rows,x[0],x[1])
 cand=[x for x in cand if x[3] is not None];cut=float(np.quantile([x[3] for x in cand],.30))
 sel=[x for x in cand if x[3]<=cut]
 sel.sort(key=lambda x:(-x[2],x[3]))
 used=set();pairs=[]
 for a,b,cr,g in sel:
  if a in used or b in used:continue
  pairs.append((a,b,g));used|={a,b}
 D={"types":types,"vmap":vmap,"pairs":pairs,"q99":th,"n_candidates":len(cand),"n_accepted_raw":len(sel)}
 return D,{"cross_cut":th,"gain_cut":cut,"raw_selected":len(sel),"disjoint_selected":len(pairs),"pairs":[{"a":a,"b":b,"gain":g} for a,b,g in pairs]}
out={}
for tid in ("ZLZI","ZLZB","TTLI"):
 rows=m["build_rows"](tid);D,meta=select(rows);ev=m["evaluate_real"](rows,tid,D,nshuffle=1000,seed=20267000+sum(map(ord,tid)))
 out[tid]={"selection":meta,"evaluation":ev};print("CF3N_"+tid+"="+json.dumps(out[tid],separators=(",",":")),flush=True)
print("FINAL="+json.dumps({"phase":"CF3_NOTATION_STRENGTH_BOUNDED","status":"complete","results":out},separators=(",",":")),flush=True)
