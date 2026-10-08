#!/usr/bin/env python3
# VMS-RESIDREC2 Phase A — preregistered 2026-10-08.
import collections,json,math,re,urllib.request
import numpy as np
from scipy.special import logsumexp

BASE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/0677011497ff31d84ebb93c2c65a67be57ae39af/research/vms_ecology1_running_residual_20261008.py"
ns={"__name__":"eco1_import"}
src=urllib.request.urlopen(BASE_URL,timeout=120).read().decode()
# import definitions only; avoid launching ECOLOGY-1 null block
prefix=src.split('if __name__=="__main__":')[0]
exec(compile(prefix,BASE_URL,"exec"),ns)

COHORTS=ns["COHORTS"]; K=ns["K"]; REAL=ns["REAL"]
L2_EMIT=10.0
STATE_GRID=(2,3,4,6)
DEPTH_GRID=(1,2,3,5)
ALPHA_GRID=(2.0,8.0,32.0)
LAMBDA_GRID=(0.25,0.5,0.75,1.0)
EPS=1e-15

def seq_fold(seq): return int(seq[0]["fold"])

def baseline_bits(lines):
    n=0; ll=0.0
    for seq in lines:
        for e in seq:
            ll += -math.log2(max(float(e["p"][int(e["y"])]),EPS)); n+=1
    return ll,n

# ---------- M1: hierarchical variable-order observable context ----------
def fit_ctx(lines,maxL=5):
    cnt0=np.zeros(K,float)
    cnt=[collections.defaultdict(lambda:np.zeros(K,float)) for _ in range(maxL+1)]
    for seq in lines:
        if not seq: continue
        hist=[int(seq[0]["prev"])]
        for e in seq:
            y=int(e["y"]); cnt0[y]+=1
            for l in range(1,min(maxL,len(hist))+1):
                cnt[l][tuple(hist[-l:])][y]+=1
            hist.append(y)
    return cnt0,cnt

def ctx_prob(hist,cnt0,cnt,L,alpha):
    p=(cnt0+0.5); p=p/p.sum()
    for l in range(1,min(L,len(hist))+1):
        c=cnt[l].get(tuple(hist[-l:]))
        if c is None: continue
        p=(c+alpha*p)/(float(c.sum())+alpha)
    return p

def pool_prob(q,pctx,lam):
    z=(1.0-lam)*np.log(np.maximum(q,EPS))+lam*np.log(np.maximum(pctx,EPS))
    z-=z.max(); x=np.exp(z); return x/x.sum()

def eval_ctx(lines,model,hp):
    cnt0,cnt=model; L,alpha,lam=hp
    ll=0.; n=0
    for seq in lines:
        if not seq:continue
        hist=[int(seq[0]["prev"])]
        for e in seq:
            pc=ctx_prob(hist,cnt0,cnt,L,alpha)
            p=pool_prob(np.asarray(e["p"],float),pc,lam)
            y=int(e["y"]); ll += -math.log2(max(float(p[y]),EPS)); n+=1
            hist.append(y)
    return ll,n

# ---------- M2: offset GLM-HMM ----------
def weighted_tilt(P,Y,W,b0=None,l2=L2_EMIT):
    if len(Y)==0 or float(np.sum(W))<1e-9:
        return np.zeros(K,float) if b0 is None else b0.copy()
    b=np.zeros(K,float) if b0 is None else b0.copy()
    b-=b.mean(); I=np.eye(K)
    logP=np.log(np.maximum(P,EPS))
    def obj(bb):
        sc=logP+bb[None,:]
        z=logsumexp(sc,axis=1)
        return -float(np.sum(W*(sc[np.arange(len(Y)),Y]-z)))+.5*l2*float(bb@bb)
    old=obj(b)
    for _ in range(15):
        sc=logP+b[None,:]; Q=np.exp(sc-logsumexp(sc,axis=1)[:,None])
        D=Q.copy(); D[np.arange(len(Y)),Y]-=1.0
        g=np.sum(W[:,None]*D,axis=0)+l2*b; g-=g.mean()
        if np.max(np.abs(g))<1e-7: break
        H=l2*I
        for wi,q in zip(W,Q):
            if wi>1e-12: H += wi*(np.diag(q)-np.outer(q,q))
        step=np.linalg.solve(H+I*1e-9,g); step-=step.mean()
        t=1.0
        while t>1e-5:
            bb=b-t*step; bb-=bb.mean(); nv=obj(bb)
            if nv<=old+1e-10:
                b,old=bb,nv; break
            t*=0.5
        if t<=1e-5: break
    return b

def line_logE(seq,b):
    S=b.shape[0]; T=len(seq); out=np.zeros((T,S),float)
    for t,e in enumerate(seq):
        q=np.asarray(e["p"],float); y=int(e["y"]); lq=np.log(np.maximum(q,EPS))
        z=logsumexp(lq[None,:]+b,axis=1)
        out[t,:]=lq[y]+b[:,y]-z
    return out

def fb(seq,pi,A,b):
    logE=line_logE(seq,b); T,S=logE.shape
    la=np.zeros((T,S)); la[0]=np.log(np.maximum(pi,EPS))+logE[0]
    for t in range(1,T):
        la[t]=logE[t]+logsumexp(la[t-1][:,None]+np.log(np.maximum(A,EPS)),axis=0)
    ll=float(logsumexp(la[-1]))
    lb=np.zeros((T,S))
    for t in range(T-2,-1,-1):
        lb[t]=logsumexp(np.log(np.maximum(A,EPS))+logE[t+1][None,:]+lb[t+1][None,:],axis=1)
    lg=la+lb-ll; gamma=np.exp(lg)
    xis=[]
    for t in range(T-1):
        lx=la[t][:,None]+np.log(np.maximum(A,EPS))+logE[t+1][None,:]+lb[t+1][None,:]-ll
        xis.append(np.exp(lx))
    return ll,gamma,xis

def init_hmm(S,seed):
    rng=np.random.default_rng(seed)
    pi=np.ones(S)/S
    A=np.ones((S,S))*0.5
    A += np.eye(S)*2.0
    A += rng.random((S,S))*0.2
    A/=A.sum(1,keepdims=True)
    b=rng.normal(0,.08,size=(S,K)); b-=b.mean(1,keepdims=True)
    return pi,A,b

def fit_hmm(lines,S,seed,maxiter=30):
    # flatten fixed observed q/y for emission M step
    P=np.vstack([np.asarray(e["p"],float) for seq in lines for e in seq])
    Y=np.array([int(e["y"]) for seq in lines for e in seq],int)
    best=None
    for restart in range(2):
        pi,A,b=init_hmm(S,seed+restart*100003)
        prev=-np.inf
        for it in range(maxiter):
            pis=np.ones(S)*0.5; trans=np.ones((S,S))*0.5
            W=np.zeros((len(Y),S),float); off=0; total=0.
            for seq in lines:
                ll,gam,xis=fb(seq,pi,A,b); total+=ll
                pis+=gam[0]
                for x in xis: trans+=x
                W[off:off+len(seq)]=gam; off+=len(seq)
            pi=pis/pis.sum(); A=trans/trans.sum(1,keepdims=True)
            for s in range(S): b[s]=weighted_tilt(P,Y,W[:,s],b[s])
            if np.isfinite(prev) and (total-prev)/max(1.,abs(prev))<1e-6: break
            prev=total
        total=sum(fb(seq,pi,A,b)[0] for seq in lines)
        cand=(total,pi.copy(),A.copy(),b.copy(),it+1)
        if best is None or cand[0]>best[0]:best=cand
    return {"train_ll":best[0],"pi":best[1],"A":best[2],"b":best[3],"iters":best[4],"S":S}

def eval_hmm(lines,model):
    pi,A,b=model["pi"],model["A"],model["b"]; ll=0.; n=0
    for seq in lines:
        post=None
        for t,e in enumerate(seq):
            pred=pi if t==0 else post@A
            q=np.asarray(e["p"],float); y=int(e["y"]); lq=np.log(np.maximum(q,EPS))
            logden=logsumexp(lq[None,:]+b,axis=1)
            ey=np.exp(lq[y]+b[:,y]-logden)
            py=float(pred@ey); ll += -math.log2(max(py,EPS)); n+=1
            post=pred*ey; z=post.sum()
            if z<=0: post=np.ones(len(pi))/len(pi)
            else: post/=z
    return ll,n

def crossval(lines,seedbase=202610081700):
    folds={j:[s for s in lines if seq_fold(s)==j] for j in range(5)}
    foldout=[]; total={"M0":[0.,0],"M1":[0.,0],"M2":[0.,0]}
    for j in range(5):
        v=(j+1)%5
        tr=[s for k in range(5) if k not in (j,v) for s in folds[k]]
        va=folds[v]; te=folds[j]
        # M1 selection
        cm=fit_ctx(tr,5); best1=None
        for L in DEPTH_GRID:
            for a in ALPHA_GRID:
                for lam in LAMBDA_GRID:
                    bl,n=eval_ctx(va,cm,(L,a,lam)); bpe=bl/max(n,1)
                    z=(bpe,L,a,lam)
                    if best1 is None or z<best1: best1=z
        hp1=(best1[1],best1[2],best1[3])
        # M2 selection
        best2=None; fitted={}
        for S in STATE_GRID:
            md=fit_hmm(tr,S,seedbase+j*1000+S*10)
            fitted[S]=md
            bl,n=eval_hmm(va,md); bpe=bl/max(n,1)
            z=(bpe,S)
            if best2 is None or z<best2:best2=z
        Ssel=best2[1]; md=fitted[Ssel]
        # untouched test
        b0,n0=baseline_bits(te); b1,n1=eval_ctx(te,cm,hp1); b2,n2=eval_hmm(te,md)
        assert n0==n1==n2
        for k,bv in (("M0",b0),("M1",b1),("M2",b2)):
            total[k][0]+=bv; total[k][1]+=n0
        foldout.append({"test_fold":j,"validation_fold":v,"n":n0,
                        "m0_bpe":b0/n0,"m1_bpe":b1/n0,"m2_bpe":b2/n0,
                        "m1_hp":{"L":hp1[0],"alpha":hp1[1],"lambda":hp1[2]},
                        "m2_S":Ssel,"m2_train_ll":md["train_ll"],"m2_iters":md["iters"],
                        "gain_m1":(b0-b1)/n0,"gain_m2":(b0-b2)/n0,"m2_minus_m1":(b1-b2)/n0})
    bpe={k:v[0]/v[1] for k,v in total.items()}
    gm1=bpe["M0"]-bpe["M1"]; gm2=bpe["M0"]-bpe["M2"]; d=bpe["M1"]-bpe["M2"]
    pos1=sum(x["gain_m1"]>0 for x in foldout); pos2=sum(x["gain_m2"]>0 for x in foldout); beat=sum(x["m2_minus_m1"]>0 for x in foldout)
    m2cand=bool(gm2>0 and pos2>=4 and d>0.0005 and beat>=4)
    m1cand=bool(gm1>0 and pos1>=4 and bpe["M1"]<=bpe["M2"]+0.0005)
    if m2cand:dec="LATENT_STATE_CANDIDATE"
    elif m1cand:dec="OBSERVABLE_CONTEXT_EXPLANATION"
    else:dec="UNRESOLVED"
    return {"pooled_bpe":bpe,"gain_m1_vs_m0":gm1,"gain_m2_vs_m0":gm2,"m2_vs_m1_delta":d,
            "positive_folds_m1":pos1,"positive_folds_m2":pos2,"m2_beats_m1_folds":beat,
            "decision":dec,"folds":foldout}

OUT={}
for i,c in enumerate(COHORTS):
    print("RUN_COHORT",c,flush=True)
    OUT[c]=crossval(REAL[c]["lines"],202610081700+i*100000)
    print("COHORT_RESULT",c,json.dumps(OUT[c],separators=(",",":")),flush=True)

phase_gate=any(v["decision"]=="LATENT_STATE_CANDIDATE" for v in OUT.values())
print("VMS_RESIDREC2_PHASEA_JSON="+json.dumps({"programme":"VMS-RESIDREC2","phase":"A","status":"complete",
      "phaseB_null_licensed":phase_gate,"ecologies":OUT},separators=(",",":")),flush=True)
