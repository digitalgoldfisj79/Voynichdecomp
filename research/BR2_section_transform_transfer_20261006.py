#!/usr/bin/env python3
"""BR2 — section-conditioned transformation transfer across frozen CF neighbourhoods."""
import urllib.request,json,collections,math,numpy as np
from scipy.optimize import minimize

BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/1c68aa15881f4c2e2a86b59e4569fe80b0f3c0c5/research/BR1_cf_neighbourhood_realisation_20261006.py"
src=urllib.request.urlopen(BASE,timeout=120).read().decode();pre=src.split("\nOUT={}\n")[0]
m={"__name__":"br1"};exec(compile(pre,BASE,"exec"),m)
TIDS=("ZLZI","TTLI");L2GRID=(0.03,0.1,0.3,1.,3.,10.,30.);SEED=20261006
SECS=("HERBAL","ASTRO","BIO","PHARMA","RECIPES","UNK")

def gall(t):return min(sum(ch in "fkpt" for ch in t),3)
def medoid(comp):
    z=[]
    for a in comp:z.append((sum(m["lev"](a,b,4) for b in comp),a))
    return min(z)[1]
def phi(c,anchor):
    sf=m["safe_form"](c);sa=m["safe_form"](anchor)
    pc=len(sf[0]) if sf else len(c);pa=len(sa[0]) if sa else len(anchor)
    ks=m["k12_start"](c);ka=m["k12_start"](anchor);kf=m["k12_final"](c);kfa=m["k12_final"](anchor)
    d=m["lev"](anchor,c,4)
    v=[float(d==j) for j in range(5)]
    v += [float(len(c)-len(anchor))/4.0,float(pc-pa)/3.0,float(gall(c)-gall(anchor))/3.0,
          float(c[:1]==anchor[:1]),float(c[-1:]==anchor[-1:]),float(ks==ka),float(kf==kfa)]
    v += [float(ks==j) for j in range(m["K12"])]
    v += [float(kf==j) for j in range(m["K12"])]
    return np.array(v,float)

def prior_counts(rows,tid,fitfolds):
    mp=m["NODECOMP"][tid];out=collections.defaultdict(collections.Counter)
    for r in rows:
        if r["fold"] in fitfolds and r["token"] in mp:out[mp[r["token"]]][r["token"]]+=1
    return out
def baseprob(counter,cands,c):
    a=.5;n=sum(counter.get(x,0) for x in cands)
    return (counter.get(c,0)+a)/(n+a*len(cands))

def feats(c,anchor,section,with_section):
    p=phi(c,anchor)
    if not with_section:return p
    s=SECS.index(section) if section in SECS else len(SECS)-1
    x=np.zeros(len(p)*(1+len(SECS)),float);x[:len(p)]=p
    x[len(p)+s*len(p):len(p)+(s+1)*len(p)]=p
    return x

def items(rows,tid,fitfolds,eventfolds,excluded_ci=None):
    pri=prior_counts(rows,tid,fitfolds);out=[];mp=m["NODECOMP"][tid]
    for r in rows:
        if r["fold"] not in eventfolds or r["pos"]==0 or r["token"] not in mp:continue
        ci=mp[r["token"]]
        if excluded_ci is not None and ci==excluded_ci:continue
        cand=tuple(x for x in m["COMP"][tid][ci] if pri[ci][x]>0)
        if len(cand)<2 or r["token"] not in cand:continue
        bp=np.array([baseprob(pri[ci],cand,c) for c in cand],float);bp/=bp.sum()
        an=medoid(m["COMP"][tid][ci])
        out.append({"r":r,"ci":ci,"cand":cand,"base":bp,"y":cand.index(r["token"]),"anchor":an})
    return out

def matrix(it,with_section):
    return np.stack([feats(c,it["anchor"],it["r"]["section"],with_section) for c in it["cand"]])

def fit(items,with_section,l2):
    d=len(feats(items[0]["cand"][0],items[0]["anchor"],items[0]["r"]["section"],with_section))
    def fg(b):
        loss=.5*l2*float(b@b);g=l2*b.copy()
        for it in items:
            F=matrix(it,with_section);u=np.log(np.maximum(it["base"],1e-15))+F@b
            u-=u.max();p=np.exp(u);p/=p.sum();y=it["y"];loss-=math.log(max(float(p[y]),1e-300));g+=F.T@p-F[y]
        return loss/max(1,len(items)),g/max(1,len(items))
    rr=minimize(lambda b:fg(b),np.zeros(d),jac=True,method="L-BFGS-B",options={"maxiter":300,"ftol":1e-10})
    return rr.x

def score(items,b,with_section):
    z=[]
    for it in items:
        F=matrix(it,with_section);u=np.log(np.maximum(it["base"],1e-15))+F@b
        u-=u.max();p=np.exp(u);p/=p.sum();y=it["y"]
        z.append((math.log2(max(float(p[y]),1e-15))-math.log2(max(float(it["base"][y]),1e-15)),it))
    return z
def choose(tr,va,with_section):
    q=[]
    for l2 in L2GRID:
        b=fit(tr,with_section,l2);s=score(va,b,with_section);g=float(np.mean([x[0] for x in s])) if s else -1e9;q.append((-g,l2))
    q.sort();return q[0][1],-q[0][0]
def stat(vals,seed):
    by=collections.defaultdict(list)
    for g,it in vals:by[it["r"]["bif"]].append(float(g))
    bm=np.array([np.mean(v) for v in by.values()],float);ev=float(np.mean([g for g,_ in vals]))
    rng=np.random.default_rng(seed);null=np.array([np.mean(bm*rng.choice((-1.,1.),len(bm))) for _ in range(10000)])
    return {"event_mean":ev,"bif_mean":float(bm.mean()),"null_mean":float(null.mean()),"null_sd":float(null.std(ddof=1)),
            "z":float((bm.mean()-null.mean())/null.std(ddof=1)),"n":len(vals),"bifs":len(bm),
            "fold0":float(np.mean([g for g,it in vals if it["r"]["fold"]==0])) if any(it["r"]["fold"]==0 for _,it in vals) else None,
            "fold1":float(np.mean([g for g,it in vals if it["r"]["fold"]==1])) if any(it["r"]["fold"]==1 for _,it in vals) else None}
OUT={}
for tid in TIDS:
    rows=m["enrich"](m["build_rows"](tid));allg=[];alls=[];rot=[]
    for hold in range(len(m["COMP"][tid])):
        tr=items(rows,tid,(2,3),(2,3),excluded_ci=hold);va=items(rows,tid,(2,3),(4,),excluded_ci=hold)
        # evaluation component gets its nuisance prior from 2/3/4, but its transformation coefficients are never trained on it.
        fititems=items(rows,tid,(2,3,4),(2,3,4),excluded_ci=hold)
        te_all=items(rows,tid,(2,3,4),(0,1),excluded_ci=None);te=[x for x in te_all if x["ci"]==hold]
        if not tr or not va or not te:continue
        l0,v0=choose(tr,va,False);l1,v1=choose(tr,va,True)
        b0=fit(fititems,False,l0);b1=fit(fititems,True,l1)
        s0=score(te,b0,False);s1=score(te,b1,True)
        # section-unique transfer = section model minus global-transform model on identical heldout component events.
        du=[(g1-g0,it1) for (g1,it1),(g0,it0) in zip(s1,s0)]
        allg.extend(s0);alls.extend(du)
        rot.append({"hold":hold,"component":list(m["COMP"][tid][hold]),"n":len(te),"l2_global":l0,"l2_section":l1,
                    "val_global":v0,"val_section":v1,"global":stat(s0,SEED+100+hold),"section_unique":stat(du,SEED+200+hold)})
    OUT[tid]={"global_transform":stat(allg,SEED+300+sum(map(ord,tid))),"section_unique_transfer":stat(alls,SEED+400+sum(map(ord,tid))),"rotations":rot}
    print("BR2_"+tid+"="+json.dumps(OUT[tid],separators=(",",":")),flush=True)
print("BR2_FINAL="+json.dumps(OUT,separators=(",",":")),flush=True)
