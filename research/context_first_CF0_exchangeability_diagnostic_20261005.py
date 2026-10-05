#!/usr/bin/env python3
import urllib.request, numpy as np, collections, pickle, re, json, math
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/af48d2b45ad416cfac9589c47f696fa6bb0d54ff/research/context_first_lexical_equivalence_20261005.py"
m={"__name__":"cfmod"};exec(compile(urllib.request.urlopen(URL).read().decode(),URL,"exec"),m)
ci=pickle.loads(urllib.request.urlopen(m["CI_URL"],timeout=120).read())
W=[str(w).lower() for w in ci["all_words"] if re.fullmatch(r"[a-zA-Z]+",str(w)) and len(str(w))>1][:24000]
vocab=[w for w,_ in collections.Counter(W[:16000]).most_common(24)];vid={w:i for i,w in enumerate(vocab)}
src0=np.array([vid.get(w,24) for w in W],int)
def rows_for(seed,shuffle=False):
 src=src0.copy();rng=np.random.default_rng(seed)
 if shuffle:src=rng.permutation(src)
 surf=[f"V{int(s):02d}_{int(rng.integers(4))}" for s in src];rows=[]
 for i,t in enumerate(surf):
  f=2 if i<8000 else 3 if i<16000 else 4 if i<20000 else 0 if i<22000 else 1
  r=dict(token=t,source=int(src[i]),folio="S",line="L",pos=i,line_len=len(surf),bif=f"S{f}_{i//400}",fold=f,section="S",lp=1)
  for lag in (-2,-1,1,2):
   j=i+lag;r[f"n{lag:+d}"]=surf[j] if 0<=j<len(surf) else None
  rows.append(r)
 return rows
def normemb(types,o,e):
 out={}
 for t in types:
  x=np.log(np.maximum(m["multiplier"](o[t],e[t]),1e-9)).ravel();x-=x.mean();n=np.linalg.norm(x);out[t]=x/n if n else x
 return out
def discover(rows,seed):
 types,dc,vc,tc=m["eligible_types"](rows);vmap,_=m["context_vocab"](rows);base,glob=m["fit_baseline"](rows,vmap,(2,3))
 o2,e2=m["profiles"](rows,vmap,(2,),set(types),base,glob);o3,e3=m["profiles"](rows,vmap,(3,),set(types),base,glob)
 E2=normemb(types,o2,e2);E3=normemb(types,o3,e3)
 rec=[]
 for i,a in enumerate(types):
  for b in types[i+1:]:
   s=.25*(np.dot(E2[a],E2[b])+np.dot(E3[a],E3[b])+np.dot(E2[a],E3[b])+np.dot(E2[b],E3[a]))
   rec.append((a,b,float(s)))
 ds=m["domsec"](rows,(2,3));rng=np.random.default_rng(seed+991);rp=m["make_random_pairs"](types,dc,ds,rng,2500)
 lookup={(a,b):s for a,b,s in rec};lookup.update({(b,a):s for a,b,s in rec})
 vals=np.array([lookup[p] for p in rp if p in lookup],float);q=float(np.quantile(vals,.995))
 hits=[r for r in rec if r[2]>q];used=set();sel=[]
 for a,b,s in sorted(hits,key=lambda z:z[2],reverse=True):
  if a in used or b in used:continue
  sel.append((a,b,s));used|={a,b}
 return types,dc,ds,vmap,q,sel
def ll_event(r,mm,vmap,base,glob):
 lags=(-2,-1,1,2);li={z:i for i,z in enumerate(lags)};ll=0;n=0
 for lag in lags:
  j=m["cat_of"](r[f"n{lag:+d}"],vmap)
  if j is None:continue
  p0=base.get(m["nuis_key"](r,lag),glob[lag]);q=p0*mm[li[lag]];q=q/q.sum();ll+=math.log2(max(float(q[j]),1e-15));n+=1
 return ll/n if n else None
def pair_deficit(rows,pair,vmap,trainfolds,testfolds):
 a,b=pair;base,glob=m["fit_baseline"](rows,vmap,trainfolds);o,e=m["profiles"](rows,vmap,trainfolds,{a,b},base,glob)
 ma=m["multiplier"](o[a],e[a]);mb=m["multiplier"](o[b],e[b]);d=[];blocks=[];foldvals=collections.defaultdict(list)
 for r in rows:
  if r["fold"] not in testfolds or r["token"] not in (a,b):continue
  selfm=ma if r["token"]==a else mb;other=mb if r["token"]==a else ma
  x=ll_event(r,selfm,vmap,base,glob);y=ll_event(r,other,vmap,base,glob)
  if x is None or y is None:continue
  z=x-y;d.append(z);blocks.append(r["bif"]);foldvals[r["fold"]].append(z)
 return {"mean":float(np.mean(d)) if d else None,"n":len(d),
         "folds":{str(k):float(np.mean(v)) for k,v in foldvals.items()},
         "block":m["block_stats"](np.array(d),blocks) if d else None}
def null_def(rows,types,dc,ds,vmap,trainfolds,testfolds,seed,n=500):
 rng=np.random.default_rng(seed);rp=m["make_random_pairs"](types,dc,ds,rng,n*2);vals=[]
 for p in rp[:n]:
  z=pair_deficit(rows,p,vmap,trainfolds,testfolds)
  if z["mean"] is not None:vals.append(z["mean"])
 return np.array(vals,float)
OUT=[]
for seed in [20261005,20261006,20261007,20261008,20261009]:
 for sh in [False,True]:
  rows=rows_for(seed,sh);types,dc,ds,vmap,q,sel=discover(rows,seed)
  vr=null_def(rows,types,dc,ds,vmap,(2,3),(4,),seed+200,300)
  tr=null_def(rows,types,dc,ds,vmap,(2,3,4),(0,1),seed+300,300)
  cand=[]
  for a,b,s in sel:
   vv=pair_deficit(rows,(a,b),vmap,(2,3),(4,));tt=pair_deficit(rows,(a,b),vmap,(2,3,4),(0,1))
   vz=(vv["mean"]-float(vr.mean()))/float(vr.std(ddof=1));tz=(tt["mean"]-float(tr.mean()))/float(tr.std(ddof=1))
   cand.append({"a":a,"b":b,"truth":a.split("_")[0]==b.split("_")[0],"sim":s,"val":vv,"val_null_z":vz,"test":tt,"test_null_z":tz})
  res={"seed":seed,"shuffle":sh,"q995":q,"pairs":cand,
       "val_null":{"mean":float(vr.mean()),"sd":float(vr.std(ddof=1))},
       "test_null":{"mean":float(tr.mean()),"sd":float(tr.std(ddof=1))}}
  OUT.append(res);print("RUN="+json.dumps(res,separators=(",",":")),flush=True)
print("FINAL="+json.dumps(OUT,separators=(",",":")))
