#!/usr/bin/env python3
# VMS-R2F: fixed 2-state PHYSICAL-BIFOLIUM-level latent regime recoverability instrument.
# Preregistered 2026-10-09 before any R2F synthetic outcome.
import argparse,collections,json,math,re,urllib.request
import numpy as np
from scipy.special import logsumexp

CORE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/a1f996f5c3e1ceaf7c561821993f6194b7fecb60/research/vms_r2_timescale_core_20261008.py"
src=urllib.request.urlopen(CORE_URL,timeout=120).read().decode()
core={"__name__":"r2core"}
exec(compile(src.split('\nif __name__=="__main__":')[0],CORE_URL,"exec"),core)

K=core["K"]; S=2; EPS=1e-15; L2=10.0; MAXITER=30; RESTARTS=2
TARGET_KL_BITS=.008
ap=argparse.ArgumentParser()
ap.add_argument("--mode",choices=["cal","blind","plant"],required=True)
ap.add_argument("--start",type=int,default=0)
ap.add_argument("--count",type=int,default=20)
ARGS=ap.parse_args(); MODE=ARGS.mode

# Frozen parent reconstruction.
lines,meta,n=core["build_parent"]()
lines,_=core["annotate"](lines)
if n!=9616: raise RuntimeError(("N",n))

# Frozen physical bifolium IDs already present in the canonical rows used to build the folds.
ROWS=core["r1"]["ROWS"]
fb=collections.defaultdict(set); ff=collections.defaultdict(set)
for r in ROWS:
    fol=str(r["folio"])
    try:
        bif=str(r["bifolium"]); fold=int(r["fold"])
    except Exception as e:
        raise RuntimeError(("BIF_META_MISSING",fol,str(e)))
    fb[fol].add(bif); ff[fol].add(fold)
bad={f:sorted(v) for f,v in fb.items() if len(v)!=1}
badf={f:sorted(v) for f,v in ff.items() if len(v)!=1}
if bad or badf: raise RuntimeError(("FOLIO_BIF_MAPPING_NONUNIQUE",bad,badf))
FOLIO_TO_BIF={f:next(iter(v)) for f,v in fb.items()}
FOLIO_TO_FOLD={f:next(iter(v)) for f,v in ff.items()}

def folio_key(f):
    m=re.match(r"^f(\d+)([rv])?(\d*)$",str(f))
    if not m:return (10**9,9,10**9,str(f))
    return (int(m.group(1)),0 if m.group(2)=="r" else 1,int(m.group(3) or 0),str(f))

# Attach immutable physical bifolium id and hard-check against q3 fold.
ann=[]
for s in lines:
    if not s: continue
    fol=str(s[0]["folio"])
    if fol not in FOLIO_TO_BIF: raise RuntimeError(("NO_BIF_FOR_FOLIO",fol))
    bif=FOLIO_TO_BIF[fol]; frozen_fold=FOLIO_TO_FOLD[fol]
    if any(int(e["fold"])!=frozen_fold for e in s):
        raise RuntimeError(("Q3_FOLD_BIF_META_DISAGREE",fol,bif,frozen_fold))
    zz=[]
    for e in s:
        x=dict(e); x["bifolium"]=bif; zz.append(x)
    ann.append(zz)
lines=sorted(ann,key=lambda s:(str(s[0]["bifolium"]),folio_key(s[0]["folio"]),int(s[0]["line_no"])))

def bifolia_from_lines(ls):
    d=collections.defaultdict(list); folds=collections.defaultdict(set); fols=collections.defaultdict(set)
    # ls is already canonical; retain order within each physical bifolium.
    for s in ls:
        if s:
            b=str(s[0]["bifolium"])
            d[b].extend(s); folds[b].add(int(s[0]["fold"])); fols[b].add(str(s[0]["folio"]))
    bad={b:sorted(v) for b,v in folds.items() if len(v)!=1}
    if bad: raise RuntimeError(("BIFOLIUM_CROSSES_FOLDS",bad))
    return [(b,d[b]) for b in sorted(d)], {b:sorted(fols[b],key=folio_key) for b in sorted(d)}

def flatten(ls):
    ev=[e for s in ls for e in s]
    P=np.vstack([np.asarray(e["p"],float) for e in ev])
    Y=np.array([int(e["y"]) for e in ev],int)
    return ev,P,Y

def tilt_probs(P,b):
    z=np.log(np.maximum(P,EPS))+b[None,:]
    z-=logsumexp(z,axis=1)[:,None]
    return np.exp(z)

def weighted_tilt(P,Y,W,b0):
    b=b0.copy(); b-=b.mean(); I=np.eye(K); logP=np.log(np.maximum(P,EPS))
    def obj(bb):
        z=logP+bb[None,:]; zz=logsumexp(z,axis=1)
        return -float(np.sum(W*(z[np.arange(len(Y)),Y]-zz)))+.5*L2*float(bb@bb)
    old=obj(b)
    for _ in range(20):
        z=logP+b[None,:]; Q=np.exp(z-logsumexp(z,axis=1)[:,None])
        D=Q.copy(); D[np.arange(len(Y)),Y]-=1.
        g=np.sum(W[:,None]*D,axis=0)+L2*b; g-=g.mean()
        if np.max(np.abs(g))<1e-8: break
        qw=W[:,None]*Q
        H=L2*I+np.diag(qw.sum(0))-Q.T@qw
        step=np.linalg.solve(H+1e-9*I,g); step-=step.mean()
        t=1.
        while t>1e-6:
            bb=b-t*step; bb-=bb.mean(); nv=obj(bb)
            if nv<=old+1e-10: b,old=bb,nv; break
            t*=.5
        if t<=1e-6: break
    return b

def group_loge(seq,b):
    out=np.zeros(S,float)
    for e in seq:
        p=np.asarray(e["p"],float); y=int(e["y"]); lp=np.log(np.maximum(p,EPS))
        den=logsumexp(lp[None,:]+b,axis=1)
        out += lp[y]+b[:,y]-den
    return out

def init(seed):
    rng=np.random.default_rng(seed)
    pi=np.ones(S)/S
    b=rng.normal(0,.08,size=(S,K)); b-=b.mean(1,keepdims=True)
    return pi,b

def fit_mix(ls,seed):
    groups,_=bifolia_from_lines(ls)
    ev,P,Y=flatten(ls)
    ranges=[];off=0
    for _,seq in groups:
        ranges.append((off,off+len(seq))); off+=len(seq)
    if off!=len(Y): raise RuntimeError(("ALIGN",off,len(Y)))
    best=None
    for rr in range(RESTARTS):
        pi,b=init(seed+rr*100003); prev=-np.inf
        for it in range(MAXITER):
            W=np.zeros((len(Y),S)); pic=np.ones(S)*.5; total=0.
            for gi,(_,seq) in enumerate(groups):
                le=group_loge(seq,b)
                lg=np.log(np.maximum(pi,EPS))+le
                ll=float(logsumexp(lg)); total+=ll
                gam=np.exp(lg-ll); pic+=gam
                a,z=ranges[gi]; W[a:z]=gam
            pi=pic/pic.sum()
            for st in range(S): b[st]=weighted_tilt(P,Y,W[:,st],b[st])
            if np.isfinite(prev) and abs(total-prev)/max(1.,abs(prev))<1e-6: break
            prev=total
        total=0.
        for _,seq in groups:
            total+=float(logsumexp(np.log(np.maximum(pi,EPS))+group_loge(seq,b)))
        cand=(total,pi.copy(),b.copy(),it+1)
        if best is None or cand[0]>best[0]: best=cand
    return {"pi":best[1],"b":best[2],"iters":best[3],"train_ll":best[0]}

def baseline_bits(ls):
    z=0.;nn=0
    for s in ls:
        for e in s:
            z += -math.log2(max(float(e["p"][int(e["y"])]),EPS)); nn+=1
    return z,nn

def eval_bits(ls,md):
    bits=0.;nn=0
    groups,_=bifolia_from_lines(ls)
    for _,seq in groups:
        post=md["pi"].copy()
        for e in seq:
            p=np.asarray(e["p"],float); y=int(e["y"])
            Q=np.vstack([tilt_probs(p[None,:],md["b"][st])[0] for st in range(S)])
            pred=post@Q
            bits += -math.log2(max(float(pred[y]),EPS)); nn+=1
            post=post*Q[:,y]; post/=post.sum()
    return bits,nn

def replace_y(ls,Y):
    out=[];k=0
    for s in ls:
        zz=[]
        for e in s:
            x=dict(e); x["y"]=int(Y[k]); zz.append(x); k+=1
        out.append(zz)
    if k!=len(Y): raise RuntimeError(("replace",k,len(Y)))
    return out

EV,PALL,Y0=flatten(lines)
rngw=np.random.default_rng(202610096000)
w=rngw.normal(size=K); w-=w.mean(); w/=np.linalg.norm(w)

def expected_kl(scale):
    vals=[]
    for sign in (-1,1):
        Q=tilt_probs(PALL,sign*scale*w)
        vals.append(np.mean(np.sum(Q*(np.log(np.maximum(Q,EPS))-np.log(np.maximum(PALL,EPS))),axis=1))/math.log(2))
    return float(np.mean(vals))
lo,hi=0.,1.
while expected_kl(hi)<TARGET_KL_BITS: hi*=2
for _ in range(50):
    m=(lo+hi)/2
    if expected_kl(m)<TARGET_KL_BITS: lo=m
    else: hi=m
PLANT_SCALE=(lo+hi)/2
PLANT_B=np.vstack([-PLANT_SCALE*w,PLANT_SCALE*w])

def sample_null(seed):
    rng=np.random.default_rng(seed)
    return np.array([rng.choice(K,p=p/p.sum()) for p in PALL],int)

def sample_plant(seed):
    rng=np.random.default_rng(seed); Y=[]
    groups,_=bifolia_from_lines(lines)
    for _,seq in groups:
        st=int(rng.integers(0,2))
        for e in seq:
            p=np.asarray(e["p"],float)
            q=tilt_probs(p[None,:],PLANT_B[st])[0]
            Y.append(int(rng.choice(K,p=q/q.sum())))
    return np.asarray(Y,int)

def cv(Y,seedbase):
    syn=replace_y(lines,Y); rows=[]; nums=[]; dens=[]
    for j in range(5):
        v=(j+1)%5
        tr=[s for s in syn if int(s[0]["fold"]) not in (j,v)]
        te=[s for s in syn if int(s[0]["fold"])==j]
        md=fit_mix(tr,seedbase+j*1000)
        b0,n0=baseline_bits(te); b2,n2=eval_bits(te,md)
        if n0!=n2: raise RuntimeError(("TESTN",n0,n2))
        gain=(b0-b2)/n0
        rows.append({"fold":j,"gain":gain,"pi":md["pi"].tolist(),"iters":md["iters"]})
        nums.append(b0-b2); dens.append(n0)
    return {"gain":float(sum(nums)/sum(dens)),
            "positive_folds":int(sum(r["gain"]>0 for r in rows)),
            "folds":rows}

ALL_SEEDS={"cal":list(range(202610096100,202610096120)),
           "blind":list(range(202610096120,202610096140)),
           "plant":list(range(202610096200,202610096220))}[MODE]
if ARGS.start<0 or ARGS.count<1 or ARGS.start+ARGS.count>20:
    raise ValueError(("SHARD",ARGS.start,ARGS.count))
seeds=ALL_SEEDS[ARGS.start:ARGS.start+ARGS.count]
groups,group_folios=bifolia_from_lines(lines)
topology={"n_bifolia":len(groups),"bifolia":group_folios,
          "fold_counts":dict(collections.Counter(int(seq[0]["fold"]) for _,seq in groups))}
outs=[]
for i,seed in enumerate(seeds):
    Y=sample_plant(seed) if MODE=="plant" else sample_null(seed)
    z=cv(Y,202610096300+seed%10000)
    z.update(rep=ARGS.start+i,seed=seed); outs.append(z)
    print("R2F_REP",MODE,i,json.dumps({"gain":z["gain"],"pos":z["positive_folds"]},separators=(",",":")),flush=True)
print("R2F_SHARD="+json.dumps({"programme":"VMS-R2F","mode":MODE,"status":"complete",
      "n_events":len(Y0),"topology":topology,"shard_start":ARGS.start,"shard_count":ARGS.count,
      "plant_scale":PLANT_SCALE,"plant_expected_kl_bits":expected_kl(PLANT_SCALE),"results":outs},
      separators=(",",":")),flush=True)
