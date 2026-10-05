#!/usr/bin/env python3
"""
CF0b: split-half fingerprint calibration.
A true equivalence pair should share one latent context distribution:
cross-token similarity across independent halves should approach within-token reliability.
No morphology.
"""
import urllib.request, numpy as np, collections, json, math
from sklearn.metrics import roc_auc_score
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/af48d2b45ad416cfac9589c47f696fa6bb0d54ff/research/context_first_lexical_equivalence_20261005.py"
m={"__name__":"cf"};exec(compile(urllib.request.urlopen(URL).read().decode(),URL,"exec"),m)

def build_half(rows,vmap,fold):
    base,glob=m["fit_baseline"](rows,vmap,(2,3))
    types=sorted(set(r["token"] for r in rows if r["fold"] in (2,3)))
    obs,exp=m["profiles"](rows,vmap,(fold,),set(types),base,glob)
    out={}
    for t in types:
        z=np.log(np.maximum(m["multiplier"](obs[t],exp[t]),1e-9)).ravel()
        z-=z.mean();n=np.linalg.norm(z)
        out[t]=z/n if n else z
    return out

def cos(a,b):return float(np.dot(a,b))
def eval_one(shuffle=False):
    rows,_=m["synth_rows"](shuffle)
    types,dc,vc,tc=m["eligible_types"](rows)
    vmap,_=m["context_vocab"](rows)
    H2=build_half(rows,vmap,2);H3=build_half(rows,vmap,3)
    rec=[]
    for i,a in enumerate(types):
        selfa=cos(H2[a],H3[a])
        for b in types[i+1:]:
            selfb=cos(H2[b],H3[b])
            cross=.5*(cos(H2[a],H3[b])+cos(H2[b],H3[a]))
            # deattenuation proxy: cross relative to mean self-reliability.
            # additive gap is stable even when reliability <0.
            gap=cross-.5*(selfa+selfb)
            ratio=cross/max(.05,.5*(selfa+selfb)) if .5*(selfa+selfb)>0 else -9
            truth=int(a.split("_")[0]==b.split("_")[0])
            rec.append((a,b,truth,cross,selfa,selfb,gap,ratio))
    y=np.array([x[2] for x in rec]);scores={}
    for k,idx in [("cross",3),("gap",6),("ratio",7)]:
        v=np.array([x[idx] for x in rec])
        scores[k]={"auc":float(roc_auc_score(y,v))}
    # validation pooling gain from original machinery, but only among fingerprint candidates.
    base,glob=m["fit_baseline"](rows,vmap,(2,3))
    obs,exp=m["profiles"](rows,vmap,(2,3),set(types),base,glob)
    val=[r for r in rows if r["fold"]==4]
    for name,idx in [("cross",3),("gap",6)]:
        vals=np.array([x[idx] for x in rec]); # select top q by discovery only
        for q in (.95,.975,.99):
            th=float(np.quantile(vals,q))
            cand=[x for x in rec if x[idx]>=th]
            gains=[]
            for x in cand:
                g=m["pair_gain"]((x[0],x[1]),val,vmap,obs,exp,base,glob)
                if g is not None:gains.append((g,x))
            gains.sort(reverse=True,key=lambda z:z[0])
            top=gains[:max(1,min(20,len(gains)))]
            scores[f"{name}_q{q}"]={
              "n_candidate":len(cand),
              "truth_fraction_candidate":float(np.mean([x[2] for x in cand])) if cand else None,
              "positive_val_n":sum(g>0 for g,x in gains),
              "top20_truth_fraction":float(np.mean([x[1][2] for x in top])) if top else None,
              "top10":[{"a":x[1][0],"b":x[1][1],"truth":bool(x[1][2]),"disc":x[idx],"valg":g} for g,x in top[:10]]
            }
    return scores
out={"ordered":eval_one(False),"shuffled":eval_one(True)}
print("CF0B="+json.dumps(out,separators=(",",":")),flush=True)
