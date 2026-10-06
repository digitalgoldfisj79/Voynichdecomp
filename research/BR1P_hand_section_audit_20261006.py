#!/usr/bin/env python3
"""BR1P — separate hand vs section inside BR1 physical arm, nested 5-fold."""
import urllib.request,json,collections,math,numpy as np
BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/1c68aa15881f4c2e2a86b59e4569fe80b0f3c0c5/research/BR1_cf_neighbourhood_realisation_20261006.py"
src=urllib.request.urlopen(BASE,timeout=120).read().decode();pre=src.split("\nOUT={}\n")[0]
m={"__name__":"br1"};exec(compile(pre,BASE,"exec"),m)
TIDS=("ZLZI","TTLI");SEED=20261006
def split_phys(items):
 for it in items:
  p=it["X"]["physical"]
  it["X"]["hand_only"]=p[:,0:1]
  it["X"]["section_only"]=p[:,1:2]
 return items
def fm(item,blocks):
 a=[item["X"][b] for b in blocks]
 return np.hstack(a) if a else np.zeros((len(item["cand"]),0))
m["fit_beta"].__globals__["feature_matrix"]=fm;m["score"].__globals__["feature_matrix"]=fm
ARMS={"HAND_ONLY":("hand_only",),"SECTION_ONLY":("section_only",),"HAND_SECTION":("hand_only","section_only")}
def choose(tr,va,blocks):
 best=None
 for l2 in m["L2GRID"]:
  b=m["fit_beta"](tr,blocks,l2);s=m["score"](va,blocks,b);g=float(np.mean([x[0] for x in s])) if s else -1e9
  z=(-g,l2,b)
  if best is None or z[:2]<best[:2]:best=z
 return best[1],-best[0]
def agg(rows,seed):
 vals=np.array([x["g"] for x in rows]);by=collections.defaultdict(list)
 for x in rows:by[x["bif"]].append(x["g"])
 bm=np.array([np.mean(v) for v in by.values()]);se=float(bm.std(ddof=1)/math.sqrt(len(bm)));rng=np.random.default_rng(seed)
 null=np.array([np.mean(bm*rng.choice((-1.,1.),len(bm))) for _ in range(10000)])
 return {"event_mean":float(vals.mean()),"bif_mean":float(bm.mean()),"se":se,"z_bif":float(bm.mean()/se),
         "null_mean":float(null.mean()),"null_sd":float(null.std(ddof=1)),
         "null_z":float((bm.mean()-null.mean())/null.std(ddof=1)),
         "fold_means":{str(f):float(np.mean([x["g"] for x in rows if x["fold"]==f])) for f in range(5)}}
OUT={}
for tid in TIDS:
 rows=m["enrich"](m["build_rows"](tid));A={k:[] for k in ARMS};meta={k:[] for k in ARMS}
 for outer in range(5):
  val=(outer+1)%5;trfolds=tuple(f for f in range(5) if f not in (outer,val))
  D=split_phys(m["make_dataset"](rows,tid,trfolds));tr=[x for x in D if x["r"]["fold"] in trfolds];va=[x for x in D if x["r"]["fold"]==val]
  fitfolds=tuple(f for f in range(5) if f!=outer);F=split_phys(m["make_dataset"](rows,tid,fitfolds));fit=[x for x in F if x["r"]["fold"] in fitfolds];te=[x for x in F if x["r"]["fold"]==outer]
  for name,blocks in ARMS.items():
   l2,vg=choose(tr,va,blocks);b=m["fit_beta"](fit,blocks,l2);sc=m["score"](te,blocks,b);meta[name].append({"outer":outer,"l2":l2,"val_gain":vg})
   for g,it in sc:A[name].append({"g":float(g),"bif":it["r"]["bif"],"fold":outer})
 r={name:{"stat":agg(A[name],SEED+i+sum(map(ord,tid))),"splits":meta[name]} for i,name in enumerate(ARMS)}
 OUT[tid]=r;print("BR1P_"+tid+"="+json.dumps(r,separators=(",",":")),flush=True)
print("BR1P_FINAL="+json.dumps(OUT,separators=(",",":")),flush=True)
