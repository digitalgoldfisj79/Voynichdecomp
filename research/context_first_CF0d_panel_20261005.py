#!/usr/bin/env python3
"""CF0d panel: freeze a combined split-half + conditional-indistinguishability rule on synthetic truth."""
import urllib.request, pickle, re, collections, math, json
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/af48d2b45ad416cfac9589c47f696fa6bb0d54ff/research/context_first_lexical_equivalence_20261005.py"
m={"__name__":"cf"};exec(compile(urllib.request.urlopen(BASE).read().decode(),BASE,"exec"),m)
CI_URL=m["CI_URL"];SEED=20261005
C=.3
CROSS_Q=(.90,.95,.975,.99)
GAIN_RULE=("lt0","bottom10","bottom20","bottom30")

def source_words():
 ci=pickle.loads(urllib.request.urlopen(CI_URL,timeout=120).read())
 W=[str(w).lower() for w in ci["all_words"] if re.fullmatch(r"[a-zA-Z]+",str(w)) and len(str(w))>1][:24000]
 vocab=[w for w,_ in collections.Counter(W[:16000]).most_common(24)]
 vid={w:i for i,w in enumerate(vocab)}
 return np.array([vid.get(w,24) for w in W],int)

SRC=source_words()

def synth(seed,shuffle=False):
 src=SRC.copy()
 if shuffle:src=np.random.default_rng(seed+70000).permutation(src)
 rng=np.random.default_rng(seed)
 surf=[f"V{int(s):02d}_{int(rng.integers(4))}" for s in src]
 rows=[]
 for i,t in enumerate(surf):
  f=2 if i<8000 else 3 if i<16000 else 4 if i<20000 else 0 if i<22000 else 1
  r=dict(token=t,folio="SYN",line="L",line_ord=0,pos=i,line_len=len(surf),bif=f"S{f}_{i//400:03d}",
         fold=f,section="SYN",lp=1)
  for lag in (-2,-1,1,2):
   j=i+lag;r[f"n{lag:+d}"]=surf[j] if 0<=j<len(surf) else None
  rows.append(r)
 return rows

def feats(r):
 d={}
 for lag in (-2,-1,1,2):d[f"L{lag}="+str(r[f"n{lag:+d}"])]=1.
 d["P11="+str(r["n-1"])+"|"+str(r["n+1"])]=1.
 d["P22="+str(r["n-2"])+"|"+str(r["n+2"])]=1.
 return d
def fixcsr(X):
 X=X.tocsr();X.indices=X.indices.astype(np.int32,copy=False);X.indptr=X.indptr.astype(np.int32,copy=False);return X
def log2p(p,y):
 q=np.where(y==1,p,1-p);return np.log2(np.maximum(q,1e-12))
def gain_pair(rows,a,b):
 tr=[r for r in rows if r["fold"] in (2,3) and r["token"] in (a,b)]
 va=[r for r in rows if r["fold"]==4 and r["token"] in (a,b)]
 if min(sum(r["token"]==a for r in tr),sum(r["token"]==b for r in tr),sum(r["token"]==a for r in va),sum(r["token"]==b for r in va))<2:return None
 v=DictVectorizer();X=fixcsr(v.fit_transform([feats(r) for r in tr]));V=fixcsr(v.transform([feats(r) for r in va]))
 yt=np.array([r["token"]==b for r in tr],int);yv=np.array([r["token"]==b for r in va],int)
 prior=(yt.sum()+.5)/(len(yt)+1);base=float(np.mean(log2p(np.full(len(yv),prior),yv)))
 try:
  md=LogisticRegression(C=C,max_iter=500,solver="liblinear").fit(X,yt);p=md.predict_proba(V)[:,1]
  return float(np.mean(log2p(p,yv))-base)
 except Exception:return None

def half_emb(rows,types):
 vmap,_=m["context_vocab"](rows);base,glob=m["fit_baseline"](rows,vmap,(2,3));out=[]
 for fold in (2,3):
  obs,exp=m["profiles"](rows,vmap,(fold,),set(types),base,glob);z={}
  for t in types:
   x=np.log(np.maximum(m["multiplier"](obs[t],exp[t]),1e-9)).ravel();x-=x.mean();n=np.linalg.norm(x);z[t]=x/n if n else x
  out.append(z)
 return out

def panel(seed,shuffle=False):
 rows=synth(seed,shuffle);types,dc,vc,tc=m["eligible_types"](rows);H2,H3=half_emb(rows,types)
 rec=[]
 for i,a in enumerate(types):
  for b in types[i+1:]:
   cross=.5*(float(H2[a]@H3[b])+float(H2[b]@H3[a]))
   truth=a.split("_")[0]==b.split("_")[0]
   rec.append([a,b,truth,cross,None])
 # only compute classifier for top 10% cross to save work
 th90=float(np.quantile([x[3] for x in rec],.90))
 for x in rec:
  if x[3]>=th90:x[4]=gain_pair(rows,x[0],x[1])
 out={}
 for q in CROSS_Q:
  th=float(np.quantile([x[3] for x in rec],q));cand=[x for x in rec if x[3]>=th and x[4] is not None]
  gains=np.array([x[4] for x in cand],float)
  if not len(cand):continue
  cuts={"lt0":0.,"bottom10":float(np.quantile(gains,.1)),"bottom20":float(np.quantile(gains,.2)),"bottom30":float(np.quantile(gains,.3))}
  for gr in GAIN_RULE:
   sel=[x for x in cand if x[4]<=cuts[gr]]
   out[f"q{q}_{gr}"]={"n":len(sel),"true":sum(x[2] for x in sel),"precision":float(np.mean([x[2] for x in sel])) if sel else None,
                         "cross_cut":th,"gain_cut":cuts[gr]}
 return out

seeds=[20261101+i*137 for i in range(8)]
records={"ordered":[],"shuffled":[]}
for sh,label in ((False,"ordered"),(True,"shuffled")):
 for i,seed in enumerate(seeds):
  z=panel(seed,sh);records[label].append(z);print("PANEL",label,i,json.dumps(z,separators=(",",":")),flush=True)

rules=sorted(set.intersection(*[set(x) for x in records["ordered"][:4]]))
train=[]
for rule in rules:
 rr=[x[rule] for x in records["ordered"][:4]]
 n=sum(x["n"] for x in rr);true=sum(x["true"] for x in rr);prec=true/n if n else 0
 train.append((prec,n,rule))
# require >=20 selected across four calibration seeds; choose highest precision then count
eligible=[x for x in train if x[1]>=20]
chosen=max(eligible) if eligible else max(train)
rule=chosen[2]
def summarize(label,idxs):
 rr=[records[label][i].get(rule,{"n":0,"true":0}) for i in idxs];n=sum(x["n"] for x in rr);tr=sum(x["true"] for x in rr)
 return {"rule":rule,"n":n,"true":tr,"precision":tr/n if n else None,"per_seed":rr}
out={"chosen_from_train":{"precision":chosen[0],"n":chosen[1],"rule":rule},
     "ordered_train":summarize("ordered",range(4)),
     "ordered_holdout":summarize("ordered",range(4,8)),
     "shuffled_holdout":summarize("shuffled",range(4,8))}
oh=out["ordered_holdout"];sh=out["shuffled_holdout"]
out["calibration_pass"]=bool(oh["n"]>=15 and oh["precision"] is not None and oh["precision"]>=.75 and
                              (sh["precision"] is None or sh["precision"]<=.20))
print("CF0D="+json.dumps(out,separators=(",",":")),flush=True)
