#!/usr/bin/env python3
"""BR2 — section-conditioned transformation transfer across frozen CF neighbourhoods."""
import urllib.request,json,collections,math,numpy as np
from scipy.optimize import minimize

BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/1c68aa15881f4c2e2a86b59e4569fe80b0f3c0c5/research/BR1_cf_neighbourhood_realisation_20261006.py"
src=urllib.request.urlopen(BASE,timeout=120).read().decode();pre=src.split("\nOUT={}\n")[0]
m={"__name__":"br1"};exec(compile(pre,BASE,"exec"),m)
TIDS=("ZLZI","TTLI");SECS=("HERBAL","ASTRO","BIO","PHARMA","RECIPES","UNK");SECI={s:i for i,s in enumerate(SECS)}
L2GRID=(0.3,1.,3.,10.,30.,100.);SEED=20261006

def ndist(a,b):
 return m["lev"](a,b,4)/max(1,len(a),len(b))
def medoid(comp):
 return min(comp,key=lambda a:(sum(ndist(a,b) for b in comp),a))
def lcp(a,b):
 n=0
 for x,y in zip(a,b):
  if x!=y:break
  n+=1
 return n
def lcs(a,b):
 n=0
 for x,y in zip(a[::-1],b[::-1]):
  if x!=y:break
  n+=1
 return n
def gall(t):return sum(ch in "fkpt" for ch in t)
def desc(c,a):
 sc=m["safe_form"](c);sa=m["safe_form"](a);cc=sc[1] if sc else [];aa=sa[1] if sa else []
 d=m["lev"](a,c,4)
 z=[
  (len(c)-len(a))/6.0,(len(cc)-len(aa))/4.0,(gall(c)-gall(a))/3.0,
  float(d==1),float(d==2),float(d==3),float(d>=4),
  float(bool(cc and aa and cc[0]==aa[0])),float(bool(cc and aa and cc[-1]==aa[-1])),
  float(m["k12_start"](c)==m["k12_start"](a)),float(m["k12_final"](c)==m["k12_final"](a)),
  lcp(a,c)/max(1,max(len(a),len(c))),lcs(a,c)/max(1,max(len(a),len(c))),
  float(c.startswith(a)),float(a.startswith(c)),float(c.endswith(a)),float(a.endswith(c))
 ]
 f8=(cc[0] if cc else -1);e8=(cc[-1] if cc else -1);ks=m["k12_start"](c)
 z += [float(f8==i) for i in range(8)]
 z += [float(e8==i) for i in range(8)]
 z += [float(ks==i) for i in range(12)]
 return np.array(z,float)

def make_items(rows,tid,fitfolds):
 comps=m["COMP"][tid];mp=m["NODECOMP"][tid];prior=collections.defaultdict(collections.Counter)
 for r in rows:
  if r["fold"] in fitfolds and r["token"] in mp:prior[mp[r["token"]]][r["token"]]+=1
 anchors={ci:medoid(c) for ci,c in enumerate(comps)}
 supported={ci:tuple(x for x in c if prior[ci][x]>0) for ci,c in enumerate(comps)}
 out=[]
 for r in rows:
  ci=mp.get(r["token"])
  if r["pos"]<=0 or ci is None or len(supported[ci])<2:continue
  cand=supported[ci]
  bp=np.array([m["smooth_prob"](prior[ci],cand,c) for c in cand]);bp/=bp.sum()
  sidx=SECI.get(r["section"],len(SECS)-1);D=[]
  for c in cand:
   q=desc(c,anchors[ci]);x=np.zeros(len(SECS)*len(q));x[sidx*len(q):(sidx+1)*len(q)]=q;D.append(x)
  out.append({"r":r,"ci":ci,"cand":cand,"y":cand.index(r["token"]),"base":bp,"F":np.stack(D)})
 return out,anchors

def fit(items,l2):
 d=items[0]["F"].shape[1]
 def fg(b):
  loss=.5*l2*float(b@b);g=l2*b.copy()
  for it in items:
   u=np.log(np.maximum(it["base"],1e-15))+it["F"]@b;u-=u.max();p=np.exp(u);p/=p.sum();y=it["y"]
   loss-=math.log(max(float(p[y]),1e-300));g+=it["F"].T@p-it["F"][y]
  return loss/max(1,len(items)),g/max(1,len(items))
 rr=minimize(lambda b:fg(b),np.zeros(d),jac=True,method="L-BFGS-B",options={"maxiter":300,"ftol":1e-10})
 return rr.x
def score(items,b):
 out=[]
 for it in items:
  u=np.log(np.maximum(it["base"],1e-15))+it["F"]@b;u-=u.max();p=np.exp(u);p/=p.sum();y=it["y"]
  g=math.log2(max(float(p[y]),1e-15))-math.log2(max(float(it["base"][y]),1e-15));out.append((g,it))
 return out
def choose(tr,va):
 best=None
 for l2 in L2GRID:
  b=fit(tr,l2);s=score(va,b);g=float(np.mean([x[0] for x in s]))
  z=(-g,l2,b)
  if best is None or z[:2]<best[:2]:best=z
 return best[1],-best[0]
def stat(vals,seed):
 if not vals:return {"mean":None,"z":None,"n":0}
 by=collections.defaultdict(list)
 for g,it in vals:by[it["r"]["bif"]].append(float(g))
 bm=np.array([np.mean(v) for v in by.values()]);mu=float(np.mean([g for g,_ in vals]));rng=np.random.default_rng(seed)
 null=np.array([np.mean(bm*rng.choice((-1.,1.),len(bm))) for _ in range(10000)])
 return {"event_mean":mu,"bif_mean":float(bm.mean()),"null_mean":float(null.mean()),"null_sd":float(null.std(ddof=1)),
         "z":float((bm.mean()-null.mean())/null.std(ddof=1)),"n":len(vals),"bifs":len(bm),
         "fold0":float(np.mean([g for g,it in vals if it["r"]["fold"]==0])) if any(it["r"]["fold"]==0 for _,it in vals) else None,
         "fold1":float(np.mean([g for g,it in vals if it["r"]["fold"]==1])) if any(it["r"]["fold"]==1 for _,it in vals) else None}

OUT={}
for tid in TIDS:
 rows=m["enrich"](m["build_rows"](tid));comps=m["COMP"][tid];allsc=[];per={}
 D,_=make_items(rows,tid,(2,3));F,_=make_items(rows,tid,(2,3,4))
 for ci,c in enumerate(comps):
  tr=[x for x in D if x["r"]["fold"] in (2,3) and x["ci"]!=ci]
  va=[x for x in D if x["r"]["fold"]==4 and x["ci"]!=ci]
  # D was fit only on 2/3 priors, but includes fold4 events for validation
  if len(tr)<50 or len(va)<20:
   per["|".join(c)]={"status":"insufficient_train_or_val","n":[len(tr),len(va)]};continue
  l2,vg=choose(tr,va)
  fititems=[x for x in F if x["r"]["fold"] in (2,3,4) and x["ci"]!=ci]
  te=[x for x in F if x["r"]["fold"] in (0,1) and x["ci"]==ci]
  if not te:
   per["|".join(c)]={"status":"no_test"};continue
  b=fit(fititems,l2);sc=score(te,b);allsc.extend(sc)
  per["|".join(c)]={"status":"ok","anchor":medoid(c),"selected_l2":l2,"validation_gain":vg,"stat":stat(sc,SEED+ci+sum(map(ord,tid)))}
 OUT[tid]={"per_component":per,"aggregate_transfer":stat(allsc,SEED+9000+sum(map(ord,tid)))}
 print("BR2_"+tid+"="+json.dumps(OUT[tid],separators=(",",":")),flush=True)
print("BR2_FINAL="+json.dumps(OUT,separators=(",",":")),flush=True)
