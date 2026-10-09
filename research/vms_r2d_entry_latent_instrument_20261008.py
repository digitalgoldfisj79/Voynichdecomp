#!/usr/bin/env python3
# VMS-R2D: fixed 2-state entry-level latent regime recoverability instrument.
import argparse,collections,json,math,urllib.request
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
MODE=ap.parse_args().mode

lines,meta,n=core["build_parent"]()
lines,_=core["annotate"](lines)
lines=sorted(lines,key=lambda s:(int(s[0]["eid"]),int(s[0]["entry_line_index"])))
if n!=9616: raise RuntimeError(("N",n))

def entries_from_lines(ls):
    d=collections.defaultdict(list)
    for s in ls:
        if s: d[int(s[0]["eid"])].extend(s)
    return [(eid,d[eid]) for eid in sorted(d)]

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
            if nv<=old+1e-10:
                b,old=bb,nv; break
            t*=.5
        if t<=1e-6: break
    return b

def entry_loge(seq,b):
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
    ents=entries_from_lines(ls)
    ev,P,Y=flatten(ls)
    # canonical token ranges match entry order because ls is globally entry-sorted.
    ranges=[];off=0
    for _,seq in ents:
        ranges.append((off,off+len(seq))); off+=len(seq)
    if off!=len(Y): raise RuntimeError(("ALIGN",off,len(Y)))
    best=None
    for rr in range(RESTARTS):
        pi,b=init(seed+rr*100003); prev=-np.inf
        for it in range(MAXITER):
            W=np.zeros((len(Y),S)); pic=np.ones(S)*.5; total=0.
            for ei,(_,seq) in enumerate(ents):
                le=entry_loge(seq,b)
                lg=np.log(np.maximum(pi,EPS))+le
                ll=float(logsumexp(lg)); total+=ll
                gam=np.exp(lg-ll); pic+=gam
                a,z=ranges[ei]; W[a:z]=gam
            pi=pic/pic.sum()
            for s in range(S): b[s]=weighted_tilt(P,Y,W[:,s],b[s])
            if np.isfinite(prev) and abs(total-prev)/max(1.,abs(prev))<1e-6: break
            prev=total
        total=0.
        for _,seq in ents:
            total+=float(logsumexp(np.log(np.maximum(pi,EPS))+entry_loge(seq,b)))
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
    for _,seq in entries_from_lines(ls):
        post=md["pi"].copy()
        for e in seq:
            p=np.asarray(e["p"],float); y=int(e["y"])
            Q=np.vstack([tilt_probs(p[None,:],md["b"][s])[0] for s in range(S)])
            pred=post@Q
            bits += -math.log2(max(float(pred[y]),EPS)); nn+=1
            post=post*Q[:,y]
            post/=post.sum()
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
rngw=np.random.default_rng(202610084000)
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
    for _,seq in entries_from_lines(lines):
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

seeds={"cal":range(202610084100,202610084120),
       "blind":range(202610084120,202610084140),
       "plant":range(202610084200,202610084220)}[MODE]
outs=[]
for i,seed in enumerate(seeds):
    Y=sample_plant(seed) if MODE=="plant" else sample_null(seed)
    z=cv(Y,202610084300+seed%10000)
    z.update(rep=i,seed=seed); outs.append(z)
    print("R2D_REP",MODE,i,json.dumps({"gain":z["gain"],"pos":z["positive_folds"]},separators=(",",":")),flush=True)
print("R2D_SHARD="+json.dumps({"programme":"VMS-R2D","mode":MODE,"status":"complete",
      "n_events":len(Y0),"shard_start":ARGS.start,"shard_count":ARGS.count,"plant_scale":PLANT_SCALE,
      "plant_expected_kl_bits":expected_kl(PLANT_SCALE),"results":outs},separators=(",",":")),flush=True)
