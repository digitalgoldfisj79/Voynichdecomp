#!/usr/bin/env python3
"""
Post-hoc hostile control for CF3: does the frozen context-first graph retain
final-fold context similarity after expected similarity from current-token
morphology is removed? Graph is never reselected.
"""
import json, math, urllib.request, numpy as np
from sklearn.linear_model import RidgeCV

BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/23e2892e7123878fed7703c9d5f59d69562fd502/research/context_first_CF1_CF3_graph_20261005.py"
src=urllib.request.urlopen(BASE,timeout=120).read().decode();ns={"__name__":"morphctl"}
exec(compile(src.split("OUT={}")[0],BASE,"exec"),ns)
build_rows=ns["build_rows"];eligible=ns["eligible"];ctx_vocab=ns["ctx_vocab"];baseline=ns["baseline"];prof=ns["prof"]
domsec=ns["domsec"];matched_random_pairs=ns["matched_random_pairs"];sim_disc=ns["sim_disc"];sim_same=ns["sim_same"];sim_cross=ns["sim_cross"]
components=ns["components"];fbin=ns["fbin"];safe_form=ns["c"]["safe_form"]
SEED=20261005;GAL=set("fkpt")

def lev(a,b):
 p=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  q=[i]
  for j,y in enumerate(b,1):q.append(min(q[-1]+1,p[j]+1,p[j-1]+(x!=y)))
  p=q
 return p[-1]/max(1,len(a),len(b))
def sf(t):
 z=safe_form(t)
 if z is None:return None
 ps,cs=z
 return dict(entry=str(cs[0]),final=str(cs[-1]),np=len(ps),raw=len(t),g=sum(ch in GAL for ch in t),path=tuple(cs))
def features(a,b,dc,ds):
 A=sf(a);B=sf(b)
 return [1.,lev(a,b),float(A["entry"]==B["entry"]),float(A["final"]==B["final"]),
         abs(A["raw"]-B["raw"]),abs(A["np"]-B["np"]),float((A["g"]>0)==(B["g"]>0)),
         abs(A["g"]-B["g"]),float(A["path"]==B["path"]),
         math.log1p(min(dc[a],dc[b])),abs(math.log1p(dc[a])-math.log1p(dc[b])),
         float(ds.get(a)==ds.get(b))]
def relabel(edges,types,dc,ds,rng,n=2000):
 nodes=sorted(set(x for e in edges for x in e));buck={}
 from collections import defaultdict
 B=defaultdict(list)
 for t in types:B[(fbin(dc[t]),ds.get(t,"UNK"))].append(t)
 st={u:(fbin(dc[u]),ds.get(u,"UNK")) for u in nodes}
 out=[]
 for _ in range(n):
  mp={};used=set();ok=True
  for u in sorted(nodes,key=lambda x:len(B[st[x]])):
   cand=[x for x in B[st[u]] if x not in used]
   if not cand:ok=False;break
   v=str(rng.choice(cand));mp[u]=v;used.add(v)
  if ok:out.append([(mp[a],mp[b]) for a,b in edges])
 return out
def run(tid):
 rows=build_rows(tid);types,dc,vc,tc=eligible(rows);vmap=ctx_vocab(rows);base,glob=baseline(rows,vmap);ds=domsec(rows)
 E2=prof(rows,vmap,set(types),2,base,glob);E3=prof(rows,vmap,set(types),3,base,glob);E4=prof(rows,vmap,set(types),4,base,glob);E0=prof(rows,vmap,set(types),0,base,glob);E1=prof(rows,vmap,set(types),1,base,glob)
 rng=np.random.default_rng(SEED+sum(map(ord,tid)))
 rp=matched_random_pairs(types,dc,ds,rng,6000);q=float(np.quantile([sim_disc(a,b,E2,E3) for a,b in rp],.995))
 edges=[];allp=[]
 for i,a in enumerate(types):
  for b in types[i+1:]:
   p=(a,b);allp.append(p)
   if sim_disc(a,b,E2,E3)>q:edges.append(p)
 eset=set(edges)
 # fit morphology->context similarity on validation fold4, excluding graph edges
 train=[p for p in allp if p not in eset]
 X=np.array([features(a,b,dc,ds) for a,b in train]);y=np.array([sim_same(a,b,E4) for a,b in train])
 model=RidgeCV(alphas=[.1,1,10,100,1000],fit_intercept=False).fit(X,y)
 def resid(p):
  a,b=p;pred=float(model.predict(np.array(features(a,b,dc,ds))[None,:])[0]);return sim_cross(a,b,E0,E1)-pred
 obs=float(np.mean([resid(p) for p in edges]))
 null=[]
 for graph in relabel(edges,types,dc,ds,rng):
  null.append(float(np.mean([resid(p) for p in graph])))
 null=np.array(null)
 return {"tid":tid,"edges":len(edges),"ridge_alpha":float(model.alpha_),"validation_R2":float(model.score(X,y)),
         "final_morph_residual":obs,"null_mean":float(null.mean()),"null_sd":float(null.std(ddof=1)),
         "z":float((obs-null.mean())/null.std(ddof=1)),
         "edge_residuals":[[a,b,resid((a,b))] for a,b in edges]}
OUT=[run(t) for t in ("ZLZI","ZLZB","TTLI")]
print("MORPH_CONTROL="+json.dumps(OUT,separators=(",",":")),flush=True)
