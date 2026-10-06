#!/usr/bin/env python3
"""BR1C — semi-synthetic identifiability calibration at observed BR1 effect scales."""
import urllib.request,json,collections,math,copy,numpy as np
from scipy.optimize import brentq

BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/1c68aa15881f4c2e2a86b59e4569fe80b0f3c0c5/research/BR1_cf_neighbourhood_realisation_20261006.py"
src=urllib.request.urlopen(BASE,timeout=120).read().decode();pre=src.split("\nOUT={}\n")[0]
m={"__name__":"br1"};exec(compile(pre,BASE,"exec"),m)

TIDS=("ZLZI","TTLI")
GENS={"PHYSICAL":("physical",),"RENDERER":("renderer",),"SELECT":("select",),"PAST":("past",)}
TARGET_BITS=(0.02,0.06)
NREP=40
SEED=20261006

def kl_bits(base,score,gamma):
    u=np.log(np.maximum(base,1e-15))+gamma*score
    u-=u.max();p=np.exp(u);p/=p.sum()
    return float(np.sum(p*np.log2(np.maximum(p,1e-15)/np.maximum(base,1e-15))))

def mean_kl(items,blocks,beta,gamma):
    z=[]
    for it in items:
        F=m["feature_matrix"](it,blocks);s=F@beta
        z.append(kl_bits(it["base"],s,gamma))
    return float(np.mean(z))

def gamma_for(items,blocks,beta,target):
    if np.linalg.norm(beta)<1e-12:return 0.0
    hi=1.0
    while mean_kl(items,blocks,beta,hi)<target and hi<256:hi*=2
    if hi>=256 and mean_kl(items,blocks,beta,hi)<target:return hi
    return float(brentq(lambda g:mean_kl(items,blocks,beta,g)-target,0.0,hi,maxiter=100))

def fit_directions(items):
    out={}
    for k,b in GENS.items():
        out[k]=m["fit_beta"](items,b,30.0)
    return out

def synth(items,blocks,beta,gamma,rng):
    out=[]
    for it in items:
        q=dict(it)
        F=m["feature_matrix"](it,blocks);u=np.log(np.maximum(it["base"],1e-15))+gamma*(F@beta)
        u-=u.max();p=np.exp(u);p/=p.sum()
        q["y"]=int(rng.choice(len(p),p=p))
        out.append(q)
    return out

def score_bits(items,blocks,beta):
    sc=m["score"](items,blocks,beta)
    return float(np.mean([x[0] for x in sc])) if sc else -1e9

def ci95(x,rng):
    a=np.asarray(x,float)
    if not len(a):return [None,None]
    vals=[]
    for _ in range(3000):
        vals.append(float(np.mean(rng.choice(a,size=len(a),replace=True))))
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]

OUT={}
for tid in TIDS:
    rows=m["enrich"](m["build_rows"](tid))
    A=m["make_dataset"](rows,tid,(2,3,4))
    tr=[x for x in A if x["r"]["fold"] in (2,3)]
    va=[x for x in A if x["r"]["fold"]==4]
    te=[x for x in A if x["r"]["fold"] in (0,1)]
    dirs=fit_directions([x for x in A if x["r"]["fold"] in (2,3,4)])
    # Freeze regularisation per classifier using REAL validation once.
    l2fix={}
    for name,blocks in GENS.items():
        best=None
        for l2 in m["L2GRID"]:
            b=m["fit_beta"](tr,blocks,l2);g=score_bits(va,blocks,b)
            z=(-g,l2)
            if best is None or z<best:best=z
        l2fix[name]=best[1]
    tout={}
    for target in TARGET_BITS:
        confusion={g:collections.Counter() for g in GENS}
        margins={g:[] for g in GENS}
        pairwins={(a,b):[] for i,a in enumerate(GENS) for b in list(GENS)[i+1:]}
        gammas={}
        for gen,gblocks in GENS.items():
            gamma=gamma_for(A,gblocks,dirs[gen],target);gammas[gen]=gamma
            for rep in range(NREP):
                rng=np.random.default_rng(SEED+int(target*10000)*100000+sum(map(ord,tid+gen))*100+rep)
                S=synth(A,gblocks,dirs[gen],gamma,rng)
                Str=[x for x in S if x["r"]["fold"] in (2,3,4)]
                Ste=[x for x in S if x["r"]["fold"] in (0,1)]
                sc={}
                for clf,cblocks in GENS.items():
                    b=m["fit_beta"](Str,cblocks,l2fix[clf]);sc[clf]=score_bits(Ste,cblocks,b)
                rank=sorted(sc,key=lambda k:(-sc[k],k))
                confusion[gen][rank[0]]+=1
                margins[gen].append(sc[gen]-max(v for k,v in sc.items() if k!=gen))
                for (a,b) in pairwins:
                    if gen==a: pairwins[(a,b)].append(float(sc[a]>sc[b]))
                    elif gen==b: pairwins[(a,b)].append(float(sc[b]>sc[a]))
        rrng=np.random.default_rng(SEED+int(target*10000)+sum(map(ord,tid)))
        diag={}
        for gen in GENS:
            vals=[1.0]*confusion[gen][gen]+[0.0]*(NREP-confusion[gen][gen])
            diag[gen]={"accuracy":float(np.mean(vals)),"ci95":ci95(vals,rrng),
                       "confusion":dict(confusion[gen]),"margin_mean":float(np.mean(margins[gen]))}
        pw={a+"__vs__"+b:{"accuracy":float(np.mean(v)) if v else None,"ci95":ci95(v,rrng)} for (a,b),v in pairwins.items()}
        tout[str(target)]={"gamma":gammas,"fixed_l2":l2fix,"diagonal":diag,"pairwise":pw}
    OUT[tid]=tout
    print("BR1C_"+tid+"="+json.dumps(tout,separators=(",",":")),flush=True)
print("BR1C_FINAL="+json.dumps(OUT,separators=(",",":")),flush=True)
