#!/usr/bin/env python3
import urllib.request, numpy as np, collections, pickle, re, json
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/af48d2b45ad416cfac9589c47f696fa6bb0d54ff/research/context_first_lexical_equivalence_20261005.py"
m={"__name__":"cfmod"};exec(compile(urllib.request.urlopen(URL).read().decode(),URL,"exec"),m)
CI=m["CI_URL"]
ci=pickle.loads(urllib.request.urlopen(CI,timeout=120).read())
W=[str(w).lower() for w in ci["all_words"] if re.fullmatch(r"[a-zA-Z]+",str(w)) and len(str(w))>1][:24000]
vocab=[w for w,_ in collections.Counter(W[:16000]).most_common(24)];vid={w:i for i,w in enumerate(vocab)}
base_src=np.array([vid.get(w,24) for w in W],int)
def rows_for(seed,shuffle=False):
 src=base_src.copy();rng=np.random.default_rng(seed)
 if shuffle:src=rng.permutation(src)
 surf=[f"V{int(s):02d}_{int(rng.integers(4))}" for s in src];rows=[]
 for i,t in enumerate(surf):
  f=2 if i<8000 else 3 if i<16000 else 4 if i<20000 else 0 if i<22000 else 1
  r=dict(token=t,source=int(src[i]),folio="S",line="L",pos=i,line_len=len(surf),bif=f"S{f}_{i//400}",fold=f,section="S",lp=1)
  for lag in (-2,-1,1,2):
   j=i+lag;r[f"n{lag:+d}"]=surf[j] if 0<=j<len(surf) else None
  rows.append(r)
 return rows
def embs(rows,fold):
 types,dc,vc,tc=m["eligible_types"](rows);vmap,_=m["context_vocab"](rows)
 base,glob=m["fit_baseline"](rows,vmap,(2,3))
 o,e=m["profiles"](rows,vmap,(fold,),set(types),base,glob);out={}
 for t in types:
  x=np.log(np.maximum(m["multiplier"](o[t],e[t]),1e-9)).ravel();x-=x.mean();n=np.linalg.norm(x);out[t]=x/n if n else x
 return types,dc,out,vmap,base,glob
def run(seed,shuffle):
 rows=rows_for(seed,shuffle);types,dc,E2,vmap,base,glob=embs(rows,2);_,_,E3,_,_,_=embs(rows,3)
 ds=m["domsec"](rows,(2,3))
 rec=[]
 for i,a in enumerate(types):
  for b in types[i+1:]:
   s=.25*(np.dot(E2[a],E2[b])+np.dot(E3[a],E3[b])+np.dot(E2[a],E3[b])+np.dot(E2[b],E3[a]))
   rec.append((a,b,float(s),a.split("_")[0]==b.split("_")[0]))
 # truth-free matched random null; frequency-bin + domsec
 rng=np.random.default_rng(seed+991);rp=m["make_random_pairs"](types,dc,ds,rng,2000)
 lookup={(a,b):s for a,b,s,tr in rec};lookup.update({(b,a):s for a,b,s,tr in rec})
 vals=np.array([lookup[p] for p in rp if p in lookup],float)
 q99=float(np.quantile(vals,.99));q995=float(np.quantile(vals,.995))
 out={}
 for nm,q in [("q99",q99),("q995",q995)]:
  hits=[r for r in rec if r[2]>q]
  # greedy nonoverlap by similarity
  used=set();sel=[]
  for r in sorted(hits,key=lambda z:z[2],reverse=True):
   if r[0] in used or r[1] in used:continue
   sel.append(r);used|={r[0],r[1]}
  out[nm]={"threshold":q,"hits":len(hits),"hit_precision":sum(r[3] for r in hits)/len(hits) if hits else None,
           "selected":len(sel),"selected_true":sum(r[3] for r in sel),
           "selected_precision":sum(r[3] for r in sel)/len(sel) if sel else None,
           "selected_pairs":[(r[0],r[1],r[3],r[2]) for r in sel]}
 return out
OUT=[]
for seed in [20261005,20261006,20261007,20261008,20261009]:
 for sh in [False,True]:
  x=run(seed,sh);OUT.append({"seed":seed,"shuffled":sh,"result":x});print("RUN",seed,sh,json.dumps(x,separators=(",",":")),flush=True)
print("FINAL="+json.dumps(OUT,separators=(",",":")))
