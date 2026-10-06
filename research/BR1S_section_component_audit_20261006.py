#!/usr/bin/env python3
"""BR1S — component heterogeneity and unique hand-over-section audit."""
import urllib.request,json,collections,math,numpy as np
BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/1c68aa15881f4c2e2a86b59e4569fe80b0f3c0c5/research/BR1_cf_neighbourhood_realisation_20261006.py"
src=urllib.request.urlopen(BASE,timeout=120).read().decode();pre=src.split("\nOUT={}\n")[0]
m={"__name__":"br1"};exec(compile(pre,BASE,"exec"),m)
TIDS=("ZLZI","TTLI");SEED=20261006
def split(items):
 for it in items:
  p=it["X"]["physical"];it["X"]["hand"]=p[:,0:1];it["X"]["sectionx"]=p[:,1:2]
 return items
def fm(item,blocks):
 a=[item["X"][b] for b in blocks];return np.hstack(a) if a else np.zeros((len(item["cand"]),0))
m["fit_beta"].__globals__["feature_matrix"]=fm;m["score"].__globals__["feature_matrix"]=fm
def choose(tr,va,blocks):
 best=None
 for l2 in m["L2GRID"]:
  b=m["fit_beta"](tr,blocks,l2);s=m["score"](va,blocks,b);g=float(np.mean([x[0] for x in s])) if s else -1e9
  z=(-g,l2,b)
  if best is None or z[:2]<best[:2]:best=z
 return best[1]
def stat(vals,seed):
 if not vals:return {"mean":None,"null_sd":None,"z":None,"n":0,"bifs":0}
 by=collections.defaultdict(list)
 for g,it in vals:by[it["r"]["bif"]].append(float(g))
 bm=np.array([np.mean(v) for v in by.values()]);mu=float(np.mean([g for g,_ in vals]));rng=np.random.default_rng(seed)
 null=np.array([np.mean(bm*rng.choice((-1.,1.),len(bm))) for _ in range(10000)])
 return {"mean":mu,"bif_mean":float(bm.mean()),"null_mean":float(null.mean()),"null_sd":float(null.std(ddof=1)),
         "z":float((bm.mean()-null.mean())/null.std(ddof=1)),"n":len(vals),"bifs":len(bm)}
OUT={}
for tid in TIDS:
 rows=m["enrich"](m["build_rows"](tid));pred={"hand":[],"section":[],"both":[]}
 for outer in range(5):
  val=(outer+1)%5;trfolds=tuple(f for f in range(5) if f not in (outer,val))
  D=split(m["make_dataset"](rows,tid,trfolds));tr=[x for x in D if x["r"]["fold"] in trfolds];va=[x for x in D if x["r"]["fold"]==val]
  fitfolds=tuple(f for f in range(5) if f!=outer);F=split(m["make_dataset"](rows,tid,fitfolds));fit=[x for x in F if x["r"]["fold"] in fitfolds];te=[x for x in F if x["r"]["fold"]==outer]
  for key,blocks in {"hand":("hand",),"section":("sectionx",),"both":("hand","sectionx")}.items():
   l2=choose(tr,va,blocks);b=m["fit_beta"](fit,blocks,l2);pred[key].extend(m["score"](te,blocks,b))
 # same event order by construction
 sec=pred["section"];hand=pred["hand"];both=pred["both"]
 comp={}
 for ci,c in enumerate(m["COMP"][tid]):
  ss=[x for x in sec if x[1]["ci"]==ci];hh=[x for x in hand if x[1]["ci"]==ci];bb=[x for x in both if x[1]["ci"]==ci]
  comp["|".join(c)]={"section":stat(ss,SEED+ci+100),"hand":stat(hh,SEED+ci+200),
                      "both":stat(bb,SEED+ci+300)}
 uniq_hand=[(gb-gs,itb) for (gb,itb),(gs,its) in zip(both,sec)]
 uniq_sec=[(gb-gh,itb) for (gb,itb),(gh,ith) in zip(both,hand)]
 OUT[tid]={"components":comp,"unique_hand_over_section":stat(uniq_hand,SEED+500+sum(map(ord,tid))),
           "unique_section_over_hand":stat(uniq_sec,SEED+600+sum(map(ord,tid)))}
 print("BR1S_"+tid+"="+json.dumps(OUT[tid],separators=(",",":")),flush=True)
print("BR1S_FINAL="+json.dumps(OUT,separators=(",",":")),flush=True)
