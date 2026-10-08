#!/usr/bin/env python3
# VMS-R2C 2-state line-level HMM recoverability qualification.
import argparse,collections,json,math,random,urllib.request
import numpy as np
from scipy.special import logsumexp

CORE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/a1f996f5c3e1ceaf7c561821993f6194b7fecb60/research/vms_r2_timescale_core_20261008.py"
src=urllib.request.urlopen(CORE_URL,timeout=120).read().decode()
core={"__name__":"r2core"}
exec(compile(src.split('\nif __name__=="__main__":')[0],CORE_URL,"exec"),core)
K=core["K"]; EPS=1e-15; S=2; L2=10.0
MAXITER=30; RESTARTS=2
PLANT_A=np.array([[.85,.15],[.15,.85]],float)
PLANT_PI=np.array([.5,.5],float)
TARGET_KL_BITS=.008

ap=argparse.ArgumentParser()
ap.add_argument("--mode",choices=["cal","blind","plant"],required=True)
args=ap.parse_args()
MODE=args.mode

lines,meta,n=core["build_parent"]()
# annotate exact entry metadata only; no observed-history features needed
lines,_=core["annotate"](lines)
# Engineering alignment invariant: all token arrays, planted labels, and EM state weights\n# use one canonical physical-entry order. Scientific model/specification unchanged.\nlines=sorted(lines,key=lambda s:(int(s[0]["eid"]),int(s[0]["entry_line_index"])))\nif n!=9616: raise RuntimeError(("N",n))

# stable physical entry grouping
def entries_from_lines(ls):
    d=collections.defaultdict(list)
    for s in ls:
        if s:d[int(s[0]["eid"])].append(s)
    out=[]
    for eid,ss in d.items():
        ss=sorted(ss,key=lambda z:int(z[0]["entry_line_index"]))
        out.append((eid,ss))
    return sorted(out,key=lambda z:z[0])

def fold_lines(j):
    v=(j+1)%5
    tr=[s for s in lines if int(s[0]["fold"]) not in (j,v)]
    va=[s for s in lines if int(s[0]["fold"])==v]
    te=[s for s in lines if int(s[0]["fold"])==j]
    return tr,va,te

def flatten(ls):
    ev=[e for s in ls for e in s]
    P=np.vstack([np.asarray(e["p"],float) for e in ev])
    Y=np.array([int(e["y"]) for e in ev],int)
    return ev,P,Y

def tilt_probs(P,b):
    z=np.log(np.maximum(P,EPS))+b[None,:]
    z-=logsumexp(z,axis=1)[:,None]
    return np.exp(z)

def weighted_tilt(P,Y,W,b0=None):
    b=np.zeros(K,float) if b0 is None else b0.copy()
    b-=b.mean();I=np.eye(K);logP=np.log(np.maximum(P,EPS))
    def obj(bb):
        sc=logP+bb[None,:];zz=logsumexp(sc,axis=1)
        return -float(np.sum(W*(sc[np.arange(len(Y)),Y]-zz)))+.5*L2*float(bb@bb)
    old=obj(b)
    for _ in range(15):
        sc=logP+b[None,:];Q=np.exp(sc-logsumexp(sc,axis=1)[:,None])
        D=Q.copy();D[np.arange(len(Y)),Y]-=1.
        g=np.sum(W[:,None]*D,axis=0)+L2*b;g-=g.mean()
        if np.max(np.abs(g))<1e-7:break
        qw=W[:,None]*Q
        H=L2*I+np.diag(qw.sum(axis=0))-Q.T@qw
        step=np.linalg.solve(H+I*1e-9,g);step-=step.mean()
        t=1.
        while t>1e-5:
            bb=b-t*step;bb-=bb.mean();nv=obj(bb)
            if nv<=old+1e-10:
                b,old=bb,nv;break
            t*=.5
        if t<=1e-5:break
    return b

def line_loge(seq,b):
    # total line log likelihood under each constant state
    out=np.zeros(S,float)
    for e in seq:
        p=np.asarray(e["p"],float);y=int(e["y"]);lp=np.log(np.maximum(p,EPS))
        den=logsumexp(lp[None,:]+b,axis=1)
        out += lp[y]+b[:,y]-den
    return out

def fb_entry(ss,pi,A,b):
    L=len(ss);LE=np.vstack([line_loge(s,b) for s in ss])
    la=np.zeros((L,S));la[0]=np.log(np.maximum(pi,EPS))+LE[0]
    for i in range(1,L):
        la[i]=LE[i]+logsumexp(la[i-1][:,None]+np.log(np.maximum(A,EPS)),axis=0)
    ll=float(logsumexp(la[-1]))
    lb=np.zeros((L,S))
    for i in range(L-2,-1,-1):
        lb[i]=logsumexp(np.log(np.maximum(A,EPS))+LE[i+1][None,:]+lb[i+1][None,:],axis=1)
    gam=np.exp(la+lb-ll)
    xis=[]
    for i in range(L-1):
        lx=la[i][:,None]+np.log(np.maximum(A,EPS))+LE[i+1][None,:]+lb[i+1][None,:]-ll
        xis.append(np.exp(lx))
    return ll,gam,xis

def init(seed):
    rng=np.random.default_rng(seed)
    pi=np.ones(S)/S
    A=np.array([[.75,.25],[.25,.75]],float)+rng.random((S,S))*.05
    A/=A.sum(1,keepdims=True)
    b=rng.normal(0,.06,size=(S,K));b-=b.mean(1,keepdims=True)
    return pi,A,b

def fit_hmm(ls,seed):
    ents=entries_from_lines(ls)
    ev,P,Y=flatten(ls)
    # token ranges per line in same iteration order as ents
    line_ranges=[];off=0
    for _,ss in ents:
        rr=[]
        for s in ss:
            rr.append((off,off+len(s)));off+=len(s)
        line_ranges.append(rr)
    assert off==len(Y)
    best=None
    for rr in range(RESTARTS):
        pi,A,b=init(seed+rr*100003);prev=-np.inf
        for it in range(MAXITER):
            pic=np.ones(S)*.5;tc=np.ones((S,S))*.5
            W=np.zeros((len(Y),S),float);total=0.
            for ei,(_,ss) in enumerate(ents):
                ll,gam,xis=fb_entry(ss,pi,A,b);total+=ll;pic+=gam[0]
                for x in xis:tc+=x
                for li,(a,z) in enumerate(line_ranges[ei]):W[a:z]=gam[li]
            pi=pic/pic.sum();A=tc/tc.sum(1,keepdims=True)
            for s in range(S):b[s]=weighted_tilt(P,Y,W[:,s],b[s])
            if np.isfinite(prev) and (total-prev)/max(1.,abs(prev))<1e-6:break
            prev=total
        total=sum(fb_entry(ss,pi,A,b)[0] for _,ss in ents)
        cand=(total,pi.copy(),A.copy(),b.copy(),it+1)
        if best is None or cand[0]>best[0]:best=cand
    return {"pi":best[1],"A":best[2],"b":best[3],"iters":best[4],"train_ll":best[0]}

def eval_bits(ls,md):
    bits=0.;nn=0
    for _,ss in entries_from_lines(ls):
        prior=md["pi"].copy()
        for line in ss:
            le=line_loge(line,md["b"])
            ll=float(logsumexp(np.log(np.maximum(prior,EPS))+le))
            bits += -ll/math.log(2);nn+=len(line)
            post=np.exp(np.log(np.maximum(prior,EPS))+le-(-bits*0+logsumexp(np.log(np.maximum(prior,EPS))+le)))
            post/=post.sum()
            prior=post@md["A"]
    return bits,nn

def baseline_bits(ls):
    z=0.;n=0
    for s in ls:
        for e in s:
            z += -math.log2(max(float(e["p"][int(e["y"])]),EPS));n+=1
    return z,n

def replace_y(ls,Y):
    out=[];k=0
    for s in ls:
        zz=[]
        for e in s:
            x=dict(e);x["y"]=int(Y[k]);zz.append(x);k+=1
        out.append(zz)
    if k!=len(Y):raise RuntimeError(("replace_y",k,len(Y)))
    return out

# Global event ordering exactly lines list
EV,PALL,Y0=flatten(lines)
FOLD=np.array([int(e["fold"]) for e in EV],int)

def sample_null(seed):
    rng=np.random.default_rng(seed)
    return np.array([rng.choice(K,p=np.asarray(p,float)/np.sum(p)) for p in PALL],int)

# plant opposite state tilts with deterministic direction
rngw=np.random.default_rng(202610083500)
w=rngw.normal(0,1,size=K);w-=w.mean();w/=np.linalg.norm(w)
def expected_kl(scale):
    vals=[]
    for sign in (-1,1):
        b=sign*scale*w
        Q=tilt_probs(PALL,b)
        vals.append(np.mean(np.sum(Q*(np.log(np.maximum(Q,EPS))-np.log(np.maximum(PALL,EPS))),axis=1))/math.log(2))
    return float(np.mean(vals))
lo,hi=0.,1.
while expected_kl(hi)<TARGET_KL_BITS and hi<128:hi*=2
for _ in range(50):
    m=(lo+hi)/2
    if expected_kl(m)<TARGET_KL_BITS:lo=m
    else:hi=m
PLANT_SCALE=(lo+hi)/2
PLANT_B=np.vstack([-PLANT_SCALE*w,PLANT_SCALE*w])

def sample_plant(seed):
    rng=np.random.default_rng(seed);Y=[];states=[]
    for _,ss in entries_from_lines(lines):
        st=int(rng.choice(S,p=PLANT_PI))
        for li,line in enumerate(ss):
            states.append(st)
            for e in line:
                p=np.asarray(e["p"],float);q=tilt_probs(p[None,:],PLANT_B[st])[0]
                Y.append(int(rng.choice(K,p=q/q.sum())))
            if li<len(ss)-1:st=int(rng.choice(S,p=PLANT_A[st]))
    return np.asarray(Y,int)

def cv(Y,seedbase):
    syn=replace_y(lines,Y)
    gains=[];rows=[]
    for j in range(5):
        v=(j+1)%5
        tr=[s for s in syn if int(s[0]["fold"]) not in (j,v)]
        va=[s for s in syn if int(s[0]["fold"])==v]
        te=[s for s in syn if int(s[0]["fold"])==j]
        # fixed family: validation only adjudicates no hyperparameter; fit train once.
        md=fit_hmm(tr,seedbase+j*1000)
        b0,n0=baseline_bits(te);b2,n2=eval_bits(te,md);assert n0==n2
        gain=(b0-b2)/n0;gains.append(gain)
        rows.append({"fold":j,"gain":gain,"A":md["A"].tolist(),"pi":md["pi"].tolist(),"iters":md["iters"]})
    return {"gain":float(np.average(gains,weights=[sum(len(s) for s in syn if int(s[0]["fold"])==j) for j in range(5)])),
            "positive_folds":int(sum(g>0 for g in gains)),"folds":rows}

if MODE=="cal":
    seeds=range(202610083600,202610083620)
elif MODE=="blind":
    seeds=range(202610083620,202610083640)
else:
    seeds=range(202610083700,202610083720)

outs=[]
for i,seed in enumerate(seeds):
    Y=sample_plant(seed) if MODE=="plant" else sample_null(seed)
    z=cv(Y,202610083800+seed%10000);z["rep"]=i;z["seed"]=seed;outs.append(z)
    print("R2C_REP",MODE,i,json.dumps({"gain":z["gain"],"pos":z["positive_folds"]},separators=(",",":")),flush=True)

print("R2C_SHARD="+json.dumps({
 "programme":"VMS-R2C","mode":MODE,"status":"complete","n_events":len(Y0),
 "plant_scale":PLANT_SCALE,"plant_expected_kl_bits":expected_kl(PLANT_SCALE),
 "results":outs},separators=(",",":")),flush=True)
