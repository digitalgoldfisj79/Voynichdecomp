#!/usr/bin/env python3
"""CF0e independent-seed certification of frozen context-equivalence rule."""
import urllib.request, numpy as np, collections, json
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/research/context_first_lexical_equivalence_20261005.py"
m={"__name__":"base"};exec(compile(urllib.request.urlopen(BASE).read().decode(),BASE,"exec"),m)
MINN=30; Q=.95; C=.3; KEEP=.20
def synth(seed,shuffle=False):
 W=m["ci_words"]()[:24000]; vocab=[w for w,_ in collections.Counter(W[:16000]).most_common(24)]
 vid={w:i for i,w in enumerate(vocab)};src=np.array([vid.get(w,24) for w in W],int)
 if shuffle:src=np.random.default_rng(seed+700000).permutation(src)
 rng=np.random.default_rng(seed)
 variants={s:[f"X{s:02d}_{j}" for j in range(4)] for s in range(25)}
 surf=[variants[int(s)][int(rng.integers(4))] for s in src];rows=[]
 for i,t in enumerate(surf):
  f=2 if i<8000 else 3 if i<16000 else 4 if i<20000 else 0 if i<22000 else 1
  r=dict(token=t,source=int(src[i]),folio="SYN",line="L",line_ord=0,pos=i,line_len=len(surf),
         bif=f"S{f}_{i//400:03d}",fold=f,section="SYN",lp=1)
  for lag in (-2,-1,1,2):
   j=i+lag;r[f"n{lag:+d}"]=surf[j] if 0<=j<len(surf) else None
  rows.append(r)
 return rows
def feats(r):
 d={"lp="+str(r["lp"]):1.}
 for lag in (-2,-1,1,2):d[f"L{lag}="+str(r[f"n{lag:+d}"])]=1.
 d["PAIR11="+str(r["n-1"])+"|"+str(r["n+1"])]=1.
 d["PAIR22="+str(r["n-2"])+"|"+str(r["n+2"])]=1.
 return d
def lp(p,y):
 q=np.where(y==1,p,1-p);return np.log2(np.maximum(q,1e-12))
def evalpair(rows,a,b):
 tr=[r for r in rows if r["fold"] in (2,3) and r["token"] in (a,b)]
 va=[r for r in rows if r["fold"]==4 and r["token"] in (a,b)]
 if min(sum(r["token"]==a for r in tr),sum(r["token"]==b for r in tr),
        sum(r["token"]==a for r in va),sum(r["token"]==b for r in va))<2:return None
 v=DictVectorizer();X=v.fit_transform([feats(r) for r in tr]);V=v.transform([feats(r) for r in va])
 y=np.array([r["token"]==b for r in tr],int);yv=np.array([r["token"]==b for r in va],int)
 prior=(y.sum()+.5)/(len(y)+1.);base=float(np.mean(lp(np.full(len(yv),prior),yv)))
 md=LogisticRegression(C=C,max_iter=500,solver="liblinear").fit(X,y)
 return float(np.mean(lp(md.predict_proba(V)[:,1],yv))-base)
def half(rows,types,vmap,fold):
 base,glob=m["fit_baseline"](rows,vmap,(2,3));obs,exp=m["profiles"](rows,vmap,(fold,),set(types),base,glob);z={}
 for t in types:
  x=np.log(np.maximum(m["multiplier"](obs[t],exp[t]),1e-9)).ravel();x-=x.mean();n=np.linalg.norm(x);z[t]=x/n if n else x
 return z
def one(seed,shuffle=False):
 rows=synth(seed,shuffle);dc=collections.Counter(r["token"] for r in rows if r["fold"] in (2,3));vc=collections.Counter(r["token"] for r in rows if r["fold"]==4);tc=collections.Counter(r["token"] for r in rows if r["fold"] in (0,1))
 types=sorted(t for t,n in dc.items() if n>=MINN and vc[t]>=3 and tc[t]>=3)
 vmap,_=m["context_vocab"](rows);A=half(rows,types,vmap,2);B=half(rows,types,vmap,3)
 rec=[]
 for i,a in enumerate(types):
  for b in types[i+1:]:
   cross=.5*(float(A[a]@B[b])+float(A[b]@B[a]));rec.append([a,b,a.split("_")[0]==b.split("_")[0],cross])
 th=float(np.quantile([x[3] for x in rec],Q));cand=[x for x in rec if x[3]>=th]
 ev=[]
 for x in cand:
  g=evalpair(rows,x[0],x[1])
  if g is not None:ev.append(x+[g])
 ev.sort(key=lambda z:z[4]);nkeep=max(1,int(np.ceil(len(ev)*KEEP)));sel=ev[:nkeep]
 used=set();pairs=[]
 for a,b,truth,cross,g in sel:
  if a in used or b in used:continue
  pairs.append((a,b,g,truth));used|={a,b}
 # heldout pooling test using original machinery
 base,glob=m["fit_baseline"](rows,vmap,(2,3));obs,exp=m["profiles"](rows,vmap,(2,3),set(types),base,glob)
 D=dict(types=types,dc=dc,tc=tc,vmap=vmap,base=base,glob=glob,obs=obs,exp=exp,q99=None,
        random_val_gain=np.array([]),pairs=[(a,b,g) for a,b,g,t in pairs],n_candidates=len(cand),n_accepted_raw=len(sel))
 out=m["evaluate_real"](rows,"SYN",D)
 out.update({"seed":seed,"shuffle":shuffle,"candidate_n":len(cand),"selected_n":len(sel),
             "selected_precision":float(np.mean([x[2] for x in sel])) if sel else None,
             "pair_n_nonoverlap":len(pairs),"pair_precision_nonoverlap":float(np.mean([x[3] for x in pairs])) if pairs else None})
 return out
records=[]
for k in range(12):
 seed=20261100+k
 for sh in (False,True):
  r=one(seed,sh);records.append(r);print("REP",k,sh,r["selected_precision"],r["pair_precision_nonoverlap"],r["pooled_vs_exact"]["z"],r["gate"],flush=True)
def summ(sh):
 z=[r for r in records if r["shuffle"]==sh]
 def med(key):return float(np.median([r[key] for r in z if r.get(key) is not None]))
 return {"n":len(z),"median_selected_precision":med("selected_precision"),"min_selected_precision":float(min(r["selected_precision"] for r in z)),
         "median_nonoverlap_precision":med("pair_precision_nonoverlap"),"median_pool_block_z":float(np.median([r["pooled_vs_exact"]["z"] for r in z if r["pooled_vs_exact"]["z"] is not None])),
         "gate_rate":float(np.mean([r["gate"] for r in z]))}
out={"rule":{"min_disc":MINN,"candidate_cross_quantile":Q,"C":C,"keep_low_gain_fraction":KEEP},"ordered":summ(False),"shuffled":summ(True),"records":records}
out["certified"]=bool(out["ordered"]["median_selected_precision"]>=.8 and out["ordered"]["median_nonoverlap_precision"]>=.8 and out["shuffled"]["median_selected_precision"]<=.2)
print("CF0E="+json.dumps(out,separators=(",",":")),flush=True)
