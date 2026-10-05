#!/usr/bin/env python3
import urllib.request,numpy as np,collections,pickle,re,json
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/af48d2b45ad416cfac9589c47f696fa6bb0d54ff/research/context_first_lexical_equivalence_20261005.py"
m={"__name__":"cfmod"};exec(compile(urllib.request.urlopen(URL).read().decode(),URL,"exec"),m)
ci=pickle.loads(urllib.request.urlopen(m["CI_URL"],timeout=120).read())
W=[str(w).lower() for w in ci["all_words"] if re.fullmatch(r"[a-zA-Z]+",str(w)) and len(str(w))>1][:24000]
vocab=[w for w,_ in collections.Counter(W[:16000]).most_common(24)];vid={w:i for i,w in enumerate(vocab)}
src0=np.array([vid.get(w,24) for w in W],int)
def rows_for(seed,sh=False):
 src=src0.copy();rng=np.random.default_rng(seed)
 if sh:src=rng.permutation(src)
 surf=[f"V{int(s):02d}_{int(rng.integers(4))}" for s in src];rows=[]
 for i,t in enumerate(surf):
  f=2 if i<8000 else 3 if i<16000 else 4 if i<20000 else 0 if i<22000 else 1
  r=dict(token=t,source=int(src[i]),folio="S",line="L",pos=i,line_len=len(surf),bif=f"S{f}_{i//400}",fold=f,section="S",lp=1)
  for lag in (-2,-1,1,2):
   j=i+lag;r[f"n{lag:+d}"]=surf[j] if 0<=j<len(surf) else None
  rows.append(r)
 return rows
def E(rows,types,vmap,fold,base,glob):
 o,e=m["profiles"](rows,vmap,(fold,),set(types),base,glob);z={}
 for t in types:
  x=np.log(np.maximum(m["multiplier"](o[t],e[t]),1e-9)).ravel();x-=x.mean();n=np.linalg.norm(x);z[t]=x/n if n else x
 return z
def pairval(a,b,A,B=None):
 if B is None:return float(np.dot(A[a],A[b]))
 return .5*(float(np.dot(A[a],B[b]))+float(np.dot(A[b],B[a])))
def run(seed,sh):
 rows=rows_for(seed,sh);types,dc,vc,tc=m["eligible_types"](rows);vmap,_=m["context_vocab"](rows);base,glob=m["fit_baseline"](rows,vmap,(2,3))
 E2=E(rows,types,vmap,2,base,glob);E3=E(rows,types,vmap,3,base,glob);E4=E(rows,types,vmap,4,base,glob);E0=E(rows,types,vmap,0,base,glob);E1=E(rows,types,vmap,1,base,glob)
 rec=[]
 for i,a in enumerate(types):
  for b in types[i+1:]:
   d=.25*(np.dot(E2[a],E2[b])+np.dot(E3[a],E3[b])+np.dot(E2[a],E3[b])+np.dot(E2[b],E3[a]))
   v=pairval(a,b,E4);te=pairval(a,b,E0,E1);rec.append((a,b,float(d),float(v),float(te),a.split("_")[0]==b.split("_")[0]))
 ds=m["domsec"](rows,(2,3));rng=np.random.default_rng(seed+991);rp=m["make_random_pairs"](types,dc,ds,rng,3000)
 lk={(a,b):(d,v,te) for a,b,d,v,te,tr in rec};lk.update({(b,a):(d,v,te) for a,b,d,v,te,tr in rec})
 rv=np.array([lk[p] for p in rp if p in lk]);qd=float(np.quantile(rv[:,0],.99));qv=float(np.quantile(rv[:,1],.95))
 hits=[r for r in rec if r[2]>qd and r[3]>qv];used=set();sel=[]
 for r in sorted(hits,key=lambda z:(z[2]+z[3]),reverse=True):
  if r[0] in used or r[1] in used:continue
  sel.append(r);used|={r[0],r[1]}
 # final null mean of same number pairs
 reps=[]
 for _ in range(1000):
  rp2=m["make_random_pairs"](types,dc,ds,rng,max(100,len(sel)*10));rng.shuffle(rp2);used=set();ss=[]
  for a,b in rp2:
   if a in used or b in used:continue
   ss.append(pairval(a,b,E0,E1));used|={a,b}
   if len(ss)>=len(sel):break
  if len(ss)==len(sel) and ss:reps.append(float(np.mean(ss)))
 obs=float(np.mean([r[4] for r in sel])) if sel else None;nm=float(np.mean(reps)) if reps else None;ns=float(np.std(reps,ddof=1)) if len(reps)>1 else None
 return {"seed":seed,"shuffle":sh,"qd99":qd,"qv95":qv,"pairs":len(sel),"true":sum(r[5] for r in sel),
         "precision":sum(r[5] for r in sel)/len(sel) if sel else None,
         "selected":[[r[0],r[1],r[5],r[2],r[3],r[4]] for r in sel],"test_mean":obs,"null_mean":nm,"null_sd":ns,
         "test_z":(obs-nm)/ns if ns and obs is not None else None}
OUT=[]
for seed in [20261005,20261006,20261007,20261008,20261009]:
 for sh in (False,True):
  r=run(seed,sh);OUT.append(r);print("RUN="+json.dumps(r,separators=(",",":")),flush=True)
print("FINAL="+json.dumps(OUT,separators=(",",":")))
