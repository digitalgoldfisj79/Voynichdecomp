#!/usr/bin/env python3
"""BR1R5 — nested five-physical-fold cross-fit robustness."""
import urllib.request,json,collections,math,numpy as np
BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/1c68aa15881f4c2e2a86b59e4569fe80b0f3c0c5/research/BR1_cf_neighbourhood_realisation_20261006.py"
src=urllib.request.urlopen(BASE,timeout=120).read().decode();pre=src.split("\nOUT={}\n")[0]
m={"__name__":"br1"};exec(compile(pre,BASE,"exec"),m)
ARMS={"M1_PHYSICAL":("physical",),"M2_RENDERER":("renderer",),"M3_SELECT_ANALOGUE":("select",),"M6_FULL_PROSPECTIVE":("physical","renderer","select","past")}
TIDS=("ZLZI","TTLI");SEED=20261006
def choose(tr,va,blocks):
 best=None
 for l2 in m["L2GRID"]:
  b=m["fit_beta"](tr,blocks,l2);s=m["score"](va,blocks,b);g=float(np.mean([x[0] for x in s])) if s else -1e9
  z=(-g,l2,b)
  if best is None or z[0:2]<best[0:2]:best=z
 return best[1],-best[0]
def agg(rows,seed):
 vals=np.array([x["g"] for x in rows]);by=collections.defaultdict(list)
 for x in rows:by[x["bif"]].append(x["g"])
 bm=np.array([np.mean(v) for v in by.values()]);mu=float(vals.mean());se=float(bm.std(ddof=1)/math.sqrt(len(bm)))
 rng=np.random.default_rng(seed);null=[float(np.mean(bm*rng.choice((-1.,1.),len(bm)))) for _ in range(10000)]
 null=np.array(null);sd=float(null.std(ddof=1));nm=float(null.mean())
 return {"mean":mu,"bif_mean":float(bm.mean()),"se":se,"z_bif":float(bm.mean()/se),"null_mean":nm,"null_sd":sd,"null_z":float((bm.mean()-nm)/sd),"n":len(vals),"bifs":len(bm),"fold_means":{str(f):float(np.mean([x["g"] for x in rows if x["fold"]==f])) for f in range(5)}}
OUT={}
for tid in TIDS:
 rows=m["enrich"](m["build_rows"](tid));A={k:[] for k in ARMS};meta={k:[] for k in ARMS}
 for outer in range(5):
  val=(outer+1)%5;trfolds=tuple(f for f in range(5) if f not in (outer,val))
  D=m["make_dataset"](rows,tid,trfolds);tr=[x for x in D if x["r"]["fold"] in trfolds];va=[x for x in D if x["r"]["fold"]==val]
  fitfolds=tuple(f for f in range(5) if f!=outer);F=m["make_dataset"](rows,tid,fitfolds);fit=[x for x in F if x["r"]["fold"] in fitfolds];te=[x for x in F if x["r"]["fold"]==outer]
  for name,blocks in ARMS.items():
   l2,vg=choose(tr,va,blocks);b=m["fit_beta"](fit,blocks,l2);sc=m["score"](te,blocks,b)
   meta[name].append({"outer":outer,"val":val,"l2":l2,"val_gain":vg})
   for g,it in sc:A[name].append({"g":float(g),"bif":it["r"]["bif"],"fold":outer})
 r={name:{"stat":agg(A[name],SEED+i+sum(map(ord,tid))),"splits":meta[name]} for i,name in enumerate(ARMS)}
 OUT[tid]=r;print("BR1R5_"+tid+"="+json.dumps(r,separators=(",",":")),flush=True)
print("BR1R5_FINAL="+json.dumps(OUT,separators=(",",":")),flush=True)
