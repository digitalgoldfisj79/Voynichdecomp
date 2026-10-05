#!/usr/bin/env python3
import urllib.request, numpy as np, collections, pickle, re, json
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
def getE(rows,types,vmap,fold,base,glob):
 o,e=m["profiles"](rows,vmap,(fold,),set(types),base,glob);return normemb(types,o,e)
def sim(a,b,Ea,Eb):return .5*(float(np.dot(Ea[a],Eb[b]))+float(np.dot(Ea[b],Eb[a])))
def discover(rows,seed):
 types,dc,vc,tc=m["eligible_types"](rows);vmap,_=m["context_vocab"](rows);base,glob=m["fit_baseline"](rows,vmap,(2,3))
 E2=getE(rows,types,vmap,2,base,glob);E3=getE(rows,types,vmap,3,base,glob)
 rec=[]
 for i,a in enumerate(types):
  for b in types[i+1:]:
   s=.25*(np.dot(E2[a],E2[b])+np.dot(E3[a],E3[b])+np.dot(E2[a],E3[b])+np.dot(E2[b],E3[a]))
   rec.append((a,b,float(s)))
 ds=m["domsec"](rows,(2,3));rng=np.random.default_rng(seed+991);rp=m["make_random_pairs"](types,dc,ds,rng,2500)
 lk={(a,b):s for a,b,s in rec};lk.update({(b,a):s for a,b,s in rec})
 vals=np.array([lk[p] for p in rp if p in lk]);q=float(np.quantile(vals,.995))
 hits=[r for r in rec if r[2]>q];used=set();sel=[]
 for a,b,s in sorted(hits,key=lambda z:z[2],reverse=True):
  if a in used or b in used:continue
  sel.append((a,b,s));used|={a,b}
 return types,dc,ds,vmap,base,glob,sel
def run(seed,sh):
 rows=rows_for(seed,sh);types,dc,ds,vmap,base,glob,sel=discover(rows,seed)
 # final fold embeddings; nuisance baseline remains discovery-fitted
 E0=getE(rows,types,vmap,0,base,glob);E1=getE(rows,types,vmap,1,base,glob)
 vals=[sim(a,b,E0,E1) for a,b,_ in sel];obs=float(np.mean(vals)) if vals else None
 rng=np.random.default_rng(seed+555);reps=[]
 for _ in range(1000):
  rp=m["make_random_pairs"](types,dc,ds,rng,max(100,len(sel)*10));rng.shuffle(rp);used=set();ss=[]
  for a,b in rp:
   if a in used or b in used:continue
   ss.append(sim(a,b,E0,E1));used|={a,b}
   if len(ss)>=len(sel):break
  if len(ss)==len(sel) and ss:reps.append(float(np.mean(ss)))
 nm=float(np.mean(reps)) if reps else None;ns=float(np.std(reps,ddof=1)) if len(reps)>1 else None
 z=(obs-nm)/ns if ns and obs is not None else None
 truth=[a.split("_")[0]==b.split("_")[0] for a,b,_ in sel]
 return {"seed":seed,"shuffle":sh,"pairs":len(sel),"true":sum(truth),"precision":sum(truth)/len(truth) if truth else None,
         "selected":[[a,b,tr,s] for (a,b,s),tr in zip(sel,truth)],"test_similarity":vals,"test_mean":obs,
         "null_mean":nm,"null_sd":ns,"test_z":z}
OUT=[]
for seed in [20261005,20261006,20261007,20261008,20261009]:
 for sh in (False,True):
  r=run(seed,sh);OUT.append(r);print("RUN="+json.dumps(r,separators=(",",":")),flush=True)
print("FINAL="+json.dumps(OUT,separators=(",",":")))
