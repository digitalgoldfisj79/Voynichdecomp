#!/usr/bin/env python3
# Phase K1: blind recovery of F0 ENTRY-only SELECT through frozen state-separated FORM.
# Synthetic-only. Search sees FORM entry classes only; truth revealed after selection.
# NO P70. No real Voynich inversion.
import json,math,urllib.request
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ProcessPoolExecutor,as_completed

K0URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f/research/select_form_socket_oracle_phaseK0_20261004.py"
k0={"__name__":"k0"};exec(compile(urllib.request.urlopen(K0URL,timeout=60).read().decode(),K0URL,"exec"),k0)

K=16;D=4;RANK=2;STRENGTH=5.0;N=4000;SEED=20261004
NFIT=2700;NVAL=500;NTR=3200
# Generate exact F0. Truth exists but is not referenced by the search routines below.
Atrue,pitrue,Utrue,Vetrue,Vrtrue,ztruth,obs=k0["generate"](K,D,RANK,STRENGTH,N,SEED,"F0")
Y=np.array([k0["PCLASS"][tok[0]] for tok in obs],dtype=np.int16)
Yfit=Y[:NFIT];Yval=Y[NFIT:NTR];Ytr=Y[:NTR];Ytest=Y[NTR:]
BASE=k0["P_START_CLASS"].copy()
LOGBASE=np.log(np.maximum(BASE,1e-30))

def emissions(U,V):
    S=LOGBASE[None,:]+U@V
    S-=S.max(1,keepdims=True);Q=np.exp(S);Q/=Q.sum(1,keepdims=True)
    return Q

def fb_counts(y,A,pi,Q):
    n=len(y);la=np.log(np.maximum(A,1e-300));lp=np.log(np.maximum(pi,1e-300));le=np.log(np.maximum(Q[:,y].T,1e-300))
    al=np.empty((n,K));sc=np.empty(n)
    a=lp+le[0];mx=a.max();sc[0]=mx+math.log(np.exp(a-mx).sum());al[0]=a-sc[0]
    for t in range(1,n):
        M=al[t-1][:,None]+la;mm=M.max(0);pr=mm+np.log(np.exp(M-mm).sum(0))
        a=pr+le[t];mx=a.max();sc[t]=mx+math.log(np.exp(a-mx).sum());al[t]=a-sc[t]
    be=np.zeros((n,K))
    for t in range(n-2,-1,-1):
        M=la+le[t+1][None,:]+be[t+1][None,:];mm=M.max(1)
        be[t]=mm+np.log(np.exp(M-mm[:,None]).sum(1))-sc[t+1]
    lg=al+be;mm=lg.max(1,keepdims=True);g=np.exp(lg-mm);g/=g.sum(1,keepdims=True)
    xi=np.zeros((K,K),float)
    for t in range(n-1):
        M=al[t][:,None]+la+le[t+1][None,:]+be[t+1][None,:]
        mm=M.max();q=np.exp(M-mm);q/=q.sum();xi+=q
    C=np.zeros((K,12),float)
    for c in range(12):C[:,c]=g[y==c].sum(0)
    return float(sc.sum()),g,xi,C

def score(y,A,pi,U,V):
    Q=emissions(U,V)
    ll,g,_,_=fb_counts(y,A,pi,Q)
    return ll,g

def fit_emission_counts(C,U0,V0,l2=.02):
    x0=np.r_[U0.ravel(),V0.ravel()]
    def fg(x):
        U=x[:K*RANK].reshape(K,RANK);V=x[K*RANK:].reshape(RANK,12)
        S=LOGBASE[None,:]+U@V;S-=S.max(1,keepdims=True);Q=np.exp(S);Q/=Q.sum(1,keepdims=True)
        nt=C.sum(1,keepdims=True);Dlt=nt*Q-C
        loss=-float(np.sum(C*np.log(np.maximum(Q,1e-300))))+.5*l2*(np.sum(U*U)+np.sum(V*V))
        gU=Dlt@V.T+l2*U;gV=U.T@Dlt+l2*V
        return loss,np.r_[gU.ravel(),gV.ravel()]
    rr=minimize(lambda x:fg(x),x0,jac=True,method="L-BFGS-B",options={"maxiter":100,"ftol":1e-10})
    x=rr.x;return x[:K*RANK].reshape(K,RANK),x[K*RANK:].reshape(RANK,12)

def dense_A(xi,alpha=.2):
    A=xi+alpha;A/=A.sum(1,keepdims=True);return A

def sparse_A(xi):
    A=np.zeros((K,K),float)
    for i in range(K):
        row=xi[i]+.05
        keep=[i]
        for j in np.argsort(row)[::-1]:
            if int(j) not in keep:keep.append(int(j))
            if len(keep)>=D:break
        w=row[keep];w/=w.sum();A[i,keep]=w
    return A

def init_model(seed):
    rng=np.random.default_rng(seed)
    U=rng.normal(0,.35,(K,RANK));V=rng.normal(0,.35,(RANK,12))
    # dense sticky random transition init to avoid premature topology lock.
    A=rng.gamma(1.0,1.0,(K,K));A[np.arange(K),np.arange(K)]+=3.
    A/=A.sum(1,keepdims=True);pi=np.ones(K)/K
    return A,pi,U,V

def fit_one(seed,y=Yfit,dense_iters=35,sparse_iters=35):
    A,pi,U,V=init_model(seed);last=-1e300;best=None
    for stage,nit in (("dense",dense_iters),("sparse",sparse_iters)):
        for it in range(nit):
            Q=emissions(U,V);ll,g,xi,C=fb_counts(y,A,pi,Q)
            A=dense_A(xi) if stage=="dense" else sparse_A(xi)
            pi=g[0]+.1;pi/=pi.sum()
            U,V=fit_emission_counts(C,U,V)
            ll2,g2,xi2,C2=fb_counts(y,A,pi,emissions(U,V))
            if best is None or ll2>best[0]:best=(ll2,A.copy(),pi.copy(),U.copy(),V.copy())
            if it>8 and abs(ll2-last)<1e-4:break
            last=ll2
        if stage=="dense":
            # restore best dense point and start sparse phase by posterior-flow pruning
            llb,A,pi,U,V=best
            _,_,xi,_=fb_counts(y,A,pi,emissions(U,V));A=sparse_A(xi);last=-1e300
    ll,A,pi,U,V=best
    return {"seed":seed,"fit_ll":float(ll),"A":A,"pi":pi,"U":U,"V":V}

def worker(seed):
    r=fit_one(seed)
    lv,_=score(Yval,r["A"],r["pi"],r["U"],r["V"])
    return {**r,"val_ll":float(lv)}

def refit_selected(md):
    # Truth-free refit on all 3200 non-test observations, initialized at selected candidate.
    A=md["A"].copy();pi=md["pi"].copy();U=md["U"].copy();V=md["V"].copy()
    last=-1e300;best=None
    for it in range(45):
        ll,g,xi,C=fb_counts(Ytr,A,pi,emissions(U,V))
        A=sparse_A(xi);pi=g[0]+.1;pi/=pi.sum();U,V=fit_emission_counts(C,U,V)
        ll2,g2,xi2,C2=fb_counts(Ytr,A,pi,emissions(U,V))
        if best is None or ll2>best[0]:best=(ll2,A.copy(),pi.copy(),U.copy(),V.copy())
        if it>8 and abs(ll2-last)<1e-4:break
        last=ll2
    ll,A,pi,U,V=best;return {"fit_ll":float(ll),"A":A,"pi":pi,"U":U,"V":V}

def nmi(a,b):return k0["nmi"](a,b)
def ari(a,b):return k0["ari"](a,b)

def edge_f1_aligned(ztrue,zpred,Aest):
    # Align predicted labels to planted labels via Hungarian confusion assignment on full test.
    C=np.zeros((K,K),int)
    for a,b in zip(ztrue,zpred):C[int(a),int(b)]+=1
    rr,cc=linear_sum_assignment(-C)
    pred_to_true={int(c):int(r) for r,c in zip(rr,cc)}
    est_edges=set()
    for i in range(K):
        for j in range(K):
            if Aest[i,j]>0:
                est_edges.add((pred_to_true.get(i,i),pred_to_true.get(j,j)))
    true_edges={(i,j) for i in range(K) for j in range(K) if Atrue[i,j]>0}
    tp=len(est_edges&true_edges);prec=tp/max(len(est_edges),1);rec=tp/max(len(true_edges),1)
    f1=2*prec*rec/max(prec+rec,1e-15)
    return {"precision":prec,"recall":rec,"f1":f1,"true_edges":len(true_edges),"est_edges":len(est_edges)}

if __name__=="__main__":
    seeds=[SEED+10000+i*137 for i in range(32)]
    outs=[]
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(worker,s):s for s in seeds}
        for fut in as_completed(futs):
            r=fut.result();outs.append(r)
            print("F0_BLIND_RESTART_JSON="+json.dumps({"seed":r["seed"],"fit_ll":r["fit_ll"],"val_ll":r["val_ll"]},separators=(",",":")),flush=True)
    # Selection uses validation likelihood only.
    outs.sort(key=lambda r:r["val_ll"],reverse=True)
    top=outs[:5]
    print("F0_BLIND_TOP_JSON="+json.dumps([{"seed":r["seed"],"fit_ll":r["fit_ll"],"val_ll":r["val_ll"]} for r in top],separators=(",",":")),flush=True)
    winner=top[0]
    pre_ll,pre_g=score(Ytest,winner["A"],winner["pi"],winner["U"],winner["V"])
    ref=refit_selected(winner)
    post_ll,post_g=score(Ytest,ref["A"],ref["pi"],ref["U"],ref["V"])
    # Truth revealed only here.
    ztest=ztruth[NTR:]
    pre=pre_g.argmax(1);post=post_g.argmax(1)
    oracleE=k0["emission"](obs[NTR:],Utrue,Vetrue,Vrtrue);or_ll,or_g=k0["fb"](oracleE,Atrue,pitrue);op=or_g.argmax(1)
    stable=[]
    # Pairwise top-solution agreement on test predictions, no truth needed.
    preds=[]
    for r in top:
        _,g=score(Ytest,r["A"],r["pi"],r["U"],r["V"]);preds.append(g.argmax(1))
    for i in range(len(preds)):
        for j in range(i):stable.append(nmi(preds[i],preds[j]))
    result={
      "family":"F0","strength":STRENGTH,"K":K,"rank":RANK,"d":D,
      "search_restarts":len(outs),"selected_seed":winner["seed"],"selected_val_ll":winner["val_ll"],
      "pre_refit":{"test_ll":float(pre_ll),"nmi":nmi(ztest,pre),"ari":ari(ztest,pre)},
      "post_refit":{"test_ll":float(post_ll),"nmi":nmi(ztest,post),"ari":ari(ztest,post),
                    "edge":edge_f1_aligned(ztest,post,ref["A"])},
      "oracle":{"test_ll":float(or_ll),"nmi":nmi(ztest,op),"ari":ari(ztest,op)},
      "top5_pairwise_nmi_median":float(np.median(stable)) if stable else None,
      "gate_single_nmi70":bool(nmi(ztest,post)>=.70)
    }
    print("SELECT_FORM_PHASEK1_JSON="+json.dumps(result,separators=(",",":")),flush=True)
