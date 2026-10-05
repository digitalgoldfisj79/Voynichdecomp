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
def run(seed,sh):
 rows=rows_for(seed,sh);types,dc,vc,tc=m["eligible_types"](rows);vmap,_=m["context_vocab"](rows);base,glob=m["fit_baseline"](rows,vmap,(2,3))
 E2=E(rows,types,vmap,2,base,glob);E3=E(rows,types,vmap,3,base,glob);E0=E(rows,types,vmap,0,base,glob);E1=E(rows,types,vmap,1,base,glob)
 def disc(a,b):return .25*(np.dot(E2[a],E2[b])+np.dot(E3[a],E3[b])+np.dot(E2[a],E3[b])+np.dot(E2[b],E3[a]))
 def test(a,b):return .5*(np.dot(E0[a],E1[b])+np.dot(E0[b],E1[a]))
 rec=[]
 for i,a in enumerate(types):
  for b in types[i+1:]:rec.append((a,b,float(disc(a,b)),float(test(a,b)),a.split("_")[0]==b.split("_")[0]))
 ds=m["domsec"](rows,(2,3));rng=np.random.default_rng(seed+991);rp=m["make_random_pairs"](types,dc,ds,rng,4000)
 lk={(a,b):(d,t) for a,b,d,t,tr in rec};lk.update({(b,a):(d,t) for a,b,d,t,tr in rec})
 vals=np.array([lk[p][0] for p in rp if p in lk]);q=float(np.quantile(vals,.995))
 hits=[r for r in rec if r[2]>q];obs=float(np.mean([r[3] for r in hits])) if hits else None
 # random edge sets with same edge count, frequency/section matched
 null=[]
 for _ in range(1500):
  rp2=m["make_random_pairs"](types,dc,ds,rng,max(200,len(hits)*20))
  if len(rp2)>=len(hits) and hits:
   rng.shuffle(rp2);ss=[lk[p][1] for p in rp2[:len(hits)] if p in lk]
   if len(ss)==len(hits):null.append(float(np.mean(ss)))
 nm=float(np.mean(null)) if null else None;ns=float(np.std(null,ddof=1)) if len(null)>1 else None
 return {"seed":seed,"shuffle":sh,"q995":q,"edges":len(hits),"true_edges":sum(r[4] for r in hits),
         "precision":sum(r[4] for r in hits)/len(hits) if hits else None,
         "components":compstats(hits),"test_mean":obs,"null_mean":nm,"null_sd":ns,
         "test_z":(obs-nm)/ns if ns and obs is not None else None,
         "edges_detail":[[r[0],r[1],r[4],r[2],r[3]] for r in hits]}
def compstats(hits):
 adj=collections.defaultdict(set)
 for a,b,*_ in hits:adj[a].add(b);adj[b].add(a)
 seen=set();sizes=[]
 for x in adj:
  if x in seen:continue
  st=[x];seen.add(x);n=0
  while st:
   u=st.pop();n+=1
   for v in adj[u]:
    if v not in seen:seen.add(v);st.append(v)
  sizes.append(n)
 return sorted(sizes,reverse=True)
OUT=[]
for seed in [20261005,20261006,20261007,20261008,20261009]:
 for sh in (False,True):
  r=run(seed,sh);OUT.append(r);print("RUN="+json.dumps(r,separators=(",",":")),flush=True)
print("FINAL="+json.dumps(OUT,separators=(",",":")))
