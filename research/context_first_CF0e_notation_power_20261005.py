#!/usr/bin/env python3
"""CF0e: notation-like power arm using frozen q0.99_bottom30 rule from failed prose calibration."""
import urllib.request,pickle,re,collections,json
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/af48d2b45ad416cfac9589c47f696fa6bb0d54ff/research/context_first_lexical_equivalence_20261005.py"
m={"__name__":"cf"};exec(compile(urllib.request.urlopen(BASE).read().decode(),BASE,"exec"),m)
CI_URL=m["CI_URL"];C=.3

def prior():
 ci=pickle.loads(urllib.request.urlopen(CI_URL,timeout=120).read())
 W=[str(w).lower() for w in ci["all_words"] if re.fullmatch(r"[a-zA-Z]+",str(w)) and len(str(w))>1][:16000]
 vocab=[w for w,_ in collections.Counter(W).most_common(24)];vid={w:i for i,w in enumerate(vocab)}
 z=np.array([vid.get(w,24) for w in W],int);p=np.bincount(z,minlength=25).astype(float)+1.;p/=p.sum()
 return p
P=prior()

def notation(seed,N=24000):
 A=np.tile(.10*P[None,:],(25,1))
 for i in range(25):
  A[i,(i+1)%25]+=.62;A[i,(i+5)%25]+=.18;A[i,i]+=.10
 A/=A.sum(1,keepdims=True)
 rng=np.random.default_rng(seed);z=np.empty(N,int);z[0]=rng.choice(25,p=P)
 for i in range(1,N):z[i]=rng.choice(25,p=A[z[i-1]])
 return z

def rows_for(seed,shuffle=False):
 src=notation(seed+3000)
 if shuffle:src=np.random.default_rng(seed+7000).permutation(src)
 rng=np.random.default_rng(seed+9000);surf=[f"V{int(s):02d}_{int(rng.integers(4))}" for s in src]
 rows=[]
 for i,t in enumerate(surf):
  f=2 if i<8000 else 3 if i<16000 else 4 if i<20000 else 0 if i<22000 else 1
  r=dict(token=t,folio="SYN",line="L",line_ord=0,pos=i,line_len=len(surf),bif=f"S{f}_{i//400:03d}",fold=f,section="SYN",lp=1)
  for lag in (-2,-1,1,2):
   j=i+lag;r[f"n{lag:+d}"]=surf[j] if 0<=j<len(surf) else None
  rows.append(r)
 return rows

def feats(r):
 d={}
 for lag in (-2,-1,1,2):d[f"L{lag}="+str(r[f"n{lag:+d}"])]=1.
 d["P11="+str(r["n-1"])+"|"+str(r["n+1"])]=1.;d["P22="+str(r["n-2"])+"|"+str(r["n+2"])]=1.
 return d
def fix(X):X=X.tocsr();X.indices=X.indices.astype(np.int32,copy=False);X.indptr=X.indptr.astype(np.int32,copy=False);return X
def lg(p,y):q=np.where(y==1,p,1-p);return np.log2(np.maximum(q,1e-12))
def gain(rows,a,b):
 tr=[r for r in rows if r["fold"] in (2,3) and r["token"] in (a,b)];va=[r for r in rows if r["fold"]==4 and r["token"] in (a,b)]
 if min(sum(r["token"]==a for r in tr),sum(r["token"]==b for r in tr),sum(r["token"]==a for r in va),sum(r["token"]==b for r in va))<2:return None
 v=DictVectorizer();X=fix(v.fit_transform([feats(r) for r in tr]));V=fix(v.transform([feats(r) for r in va]))
 y=np.array([r["token"]==b for r in tr],int);yv=np.array([r["token"]==b for r in va],int);pr=(y.sum()+.5)/(len(y)+1)
 base=np.mean(lg(np.full(len(yv),pr),yv))
 try:
  md=LogisticRegression(C=C,max_iter=500,solver="liblinear").fit(X,y);return float(np.mean(lg(md.predict_proba(V)[:,1],yv))-base)
 except:return None

def split(rows,types):
 vmap,_=m["context_vocab"](rows);base,glob=m["fit_baseline"](rows,vmap,(2,3));out=[]
 for f in (2,3):
  o,e=m["profiles"](rows,vmap,(f,),set(types),base,glob);zz={}
  for t in types:
   x=np.log(np.maximum(m["multiplier"](o[t],e[t]),1e-9)).ravel();x-=x.mean();n=np.linalg.norm(x);zz[t]=x/n if n else x
  out.append(zz)
 return out
def panel(seed,sh):
 rows=rows_for(seed,sh);types,dc,vc,tc=m["eligible_types"](rows);h2,h3=split(rows,types)
 rec=[]
 for i,a in enumerate(types):
  for b in types[i+1:]:
   cr=.5*(float(h2[a]@h3[b])+float(h2[b]@h3[a]));rec.append([a,b,a.split("_")[0]==b.split("_")[0],cr,None])
 th=np.quantile([x[3] for x in rec],.99);cand=[x for x in rec if x[3]>=th]
 for x in cand:x[4]=gain(rows,x[0],x[1])
 cand=[x for x in cand if x[4] is not None];gs=np.array([x[4] for x in cand]);cut=np.quantile(gs,.30)
 sel=[x for x in cand if x[4]<=cut]
 return {"n":len(sel),"true":sum(x[2] for x in sel),"precision":float(np.mean([x[2] for x in sel])) if sel else None,
         "candidate_n":len(cand),"cross_cut":float(th),"gain_cut":float(cut)}
seeds=[20262101+i*193 for i in range(8)]
out={"ordered":[],"shuffled":[]}
for sh,k in ((False,"ordered"),(True,"shuffled")):
 for i,s in enumerate(seeds):
  z=panel(s,sh);out[k].append(z);print("NOTATION",k,i,json.dumps(z,separators=(",",":")),flush=True)
for k in out:
 n=sum(x["n"] for x in out[k]);tr=sum(x["true"] for x in out[k]);out[k+"_summary"]={"n":n,"true":tr,"precision":tr/n if n else None}
out["power_pass"]=bool(out["ordered_summary"]["n"]>=30 and out["ordered_summary"]["precision"]>=.80 and out["shuffled_summary"]["precision"]<=.20)
print("CF0E="+json.dumps(out,separators=(",",":")),flush=True)
