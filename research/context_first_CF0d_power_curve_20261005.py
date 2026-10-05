#!/usr/bin/env python3
"""CF0d power curve: same estimator, vary minimum discovery occurrences."""
import urllib.request, numpy as np, json, collections
from sklearn.metrics import roc_auc_score
BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/research/context_first_CF0c_indistinguishability_20261005.py"
m={"__name__":"cfc","__file__":"cf0c"};exec(compile(urllib.request.urlopen(BASE).read().decode(),BASE,"exec"),m)
# module prints on import; now perform compact power curve
cf=__import__("types")
BASE2="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/research/context_first_lexical_equivalence_20261005.py"
b={"__name__":"base"};exec(compile(urllib.request.urlopen(BASE2).read().decode(),BASE2,"exec"),b)

def run(minn,shuffle=False,C=.3):
 rows,_=b["synth_rows"](shuffle)
 dc=collections.Counter(r["token"] for r in rows if r["fold"] in (2,3))
 vc=collections.Counter(r["token"] for r in rows if r["fold"]==4)
 types=sorted(t for t,n in dc.items() if n>=minn and vc[t]>=max(3,minn//8))
 if len(types)<4:return {"n_types":len(types)}
 H2,H3=m["split_cross"](rows,types)
 rec=[]
 for i,a in enumerate(types):
  for bb in types[i+1:]:
   cross=.5*(float(H2[a]@H3[bb])+float(H2[bb]@H3[a]))
   truth=a.split("_")[0]==bb.split("_")[0]
   rec.append((a,bb,truth,cross))
 th=float(np.quantile([x[3] for x in rec],.95));cand=[x for x in rec if x[3]>=th]
 rr=[]
 for a,bb,t,cross in cand:
  e=m["eval_pair"](rows,a,bb,C)
  if e is not None:rr.append((a,bb,t,cross,e))
 if not rr:return {"n_types":len(types),"n_cand":len(cand),"n_eval":0}
 y=np.array([x[2] for x in rr],int);gain=np.array([x[4]["gain"] for x in rr])
 auc=float(roc_auc_score(y,-gain)) if len(set(y))>1 else None
 out={"n_types":len(types),"n_cand":len(cand),"n_eval":len(rr),"base_truth":float(np.mean(y)),"auc":auc}
 for frac in (.05,.1,.15,.2,.25,.3):
  n=max(1,int(np.ceil(len(rr)*frac)));sel=sorted(rr,key=lambda z:z[4]["gain"])[:n]
  out[str(frac)]={"n":n,"precision":float(np.mean([x[2] for x in sel])),
                  "med_gain":float(np.median([x[4]["gain"] for x in sel]))}
 return out
out={}
for minn in (10,20,30,40,50,75,100,150,200):
 out[str(minn)]={"ordered":run(minn,False),"shuffled":run(minn,True)}
print("CF0D="+json.dumps(out,separators=(",",":")),flush=True)
