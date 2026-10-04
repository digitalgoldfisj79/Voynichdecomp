#!/usr/bin/env python3
# Phase H coarse diagnostic: exact-family basin of attraction under controlled truth corruption.
# Synthetic-only. Truth is used ONLY to construct diagnostic initializations and score AFTER the pipeline.
# NO P70.
import json,urllib.request
import numpy as np
import torch
from scipy.sparse import csr_matrix

PURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/5d46189f00e3a3fa4db3eeb9a1331b48890c8ae7/research/inverse_renderer_piece_oracle_diagnostic_20261004.py"
p={"__name__":"piece"};exec(compile(urllib.request.urlopen(PURL,timeout=60).read().decode(),PURL,"exec"),p)
p["BASE_MIX"]=.02;p["BIAS_BOUND"]=5.0
DURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/dd2c107f3335ed3d022760f67bd0b95885898ba8/research/inverse_renderer_piece_recovery_phaseD_20261004.py"
d={"__name__":"phaseD"};exec(compile(urllib.request.urlopen(DURL,timeout=60).read().decode(),DURL,"exec"),d)

K=16;DEG=4;RANK=2;STRENGTH=5.0;SEED=20261004;N=4000
NFIT=2700;NVAL=500
A0,pi0,U0,V0,z,routes=p["generate"](K,DEG,RANK,STRENGTH,N,SEED)
X=p["flatten"](routes);Xfit=X[:NFIT];Xval=X[NFIT:NFIT+NVAL];Xtest=X[NFIT+NVAL:]
zfit=z[:NFIT];zval=z[NFIT:NFIT+NVAL];ztest=z[NFIT+NVAL:]

def viterbi(E,A,pi):
    nn,kk=E.shape;la=np.log(np.maximum(A,1e-30));lp=np.log(np.maximum(pi,1e-30))
    dp=np.empty((nn,kk));bp=np.empty((nn,kk),np.int16);dp[0]=lp+E[0]
    for t in range(1,nn):
        M=dp[t-1][:,None]+la;bp[t]=np.argmax(M,0);dp[t]=M[bp[t],np.arange(kk)]+E[t]
    zz=np.empty(nn,np.int16);zz[-1]=np.argmax(dp[-1])
    for t in range(nn-2,-1,-1):zz[t]=bp[t+1,zz[t+1]]
    return zz,float(dp[-1].max())

def fit_exact(labels,Uprev=None,Vprev=None,steps=180):
    TC=np.zeros((K,K),float)
    for a,b in zip(labels[:-1],labels[1:]):TC[int(a),int(b)]+=1
    A=d["sparse_A"](TC,DEG)
    pi=np.bincount(labels[:200],minlength=K).astype(float)+1;pi/=pi.sum()
    F=csr_matrix(Xfit.reshape(NFIT,-1))
    if Uprev is None:
        Q=d["q_project_from_counts"](d["hard_counts"](F,labels,K),prior=4.0)
        Uinit,Vinit=d["lowrank"](d["bias_from_Q"](Q),RANK)
    else: Uinit,Vinit=Uprev,Vprev
    U=torch.nn.Parameter(torch.tensor(Uinit,dtype=torch.float32))
    V=torch.nn.Parameter(torch.tensor(Vinit,dtype=torch.float32))
    opt=torch.optim.Adam([U,V],lr=.025)
    idx=torch.arange(NFIT);zt=torch.tensor(labels,dtype=torch.long)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        E=d["lowrank_E_torch"](Xfit,U,V,torch.device("cpu"))
        loss=-E[idx,zt].mean()+1e-3*(U.square().mean()+V.square().mean())
        loss.backward();torch.nn.utils.clip_grad_norm_([U,V],5.0);opt.step()
    return A,pi,U.detach().numpy(),V.detach().numpy()

def score_block(Xb,A,pi,U,V):
    with torch.no_grad():
        E=d["lowrank_E_torch"](Xb,torch.tensor(U,dtype=torch.float32),torch.tensor(V,dtype=torch.float32),torch.device("cpu")).numpy().astype(float)
    ll,g,_=d["fb"](E,A,pi)
    return float(ll),g.argmax(1),E

def corrupt(frac,seed):
    rng=np.random.default_rng(seed);L=zfit.astype(np.int16).copy()
    n=int(round(frac*len(L)))
    if n:
        ix=rng.choice(len(L),n,replace=False)
        draw=rng.integers(0,K,n,dtype=np.int16)
        same=draw==L[ix];draw[same]=(draw[same]+1)%K
        L[ix]=draw
    return L

def run_one(frac,rep):
    L=corrupt(frac,SEED+10000+rep*991+int(frac*10000))
    init_nmi=d["nmi"](zfit,L)
    U=V=None;hist=[]
    for cyc in range(5):
        A,pi,U,V=fit_exact(L,U,V,180 if cyc==0 else 90)
        llfit,pfit,Efit=score_block(Xfit,A,pi,U,V)
        llval,pval,_=score_block(Xval,A,pi,U,V)
        Lnew,_=viterbi(Efit,A,pi)
        rec={"cycle":cyc,"fit_ll":llfit,"val_ll":llval,
             "label_nmi":d["nmi"](zfit,L),
             "post_fit_nmi":d["nmi"](zfit,pfit),
             "val_nmi":d["nmi"](zval,pval)}
        hist.append(rec);L=Lnew
    # blind-like selection point: cycle with best validation exact marginal likelihood.
    best=max(hist,key=lambda r:r["val_ll"])
    # reconstruct each cycle models is expensive; final model retained. For diagnostic test NMI report final only.
    llt,pt,_=score_block(Xtest,A,pi,U,V)
    return {"corruption":frac,"rep":rep,"init_nmi":init_nmi,
            "best_val_cycle":best["cycle"],"best_val_ll":best["val_ll"],"best_val_nmi_diag":best["val_nmi"],
            "final_train_label_nmi":d["nmi"](zfit,L),"final_test_ll":llt,
            "final_test_nmi":d["nmi"](ztest,pt),"final_test_ari":d["ari"](ztest,pt),
            "history":hist}

if __name__=="__main__":
    fracs=[0.0,.20,.40,.60,.80,1.0]
    outs=[]
    for frac in fracs:
        reps=[]
        for rep in range(1):
            rec=run_one(frac,rep);reps.append(rec)
            print("BASIN_RUN_JSON="+json.dumps(rec,separators=(",",":")),flush=True)
        s={"corruption":frac,
           "median_init_nmi":float(np.median([r["init_nmi"] for r in reps])),
           "median_final_test_nmi":float(np.median([r["final_test_nmi"] for r in reps])),
           "min_final_test_nmi":float(min(r["final_test_nmi"] for r in reps)),
           "reps":reps}
        print("BASIN_SUMMARY_JSON="+json.dumps(s,separators=(",",":")),flush=True);outs.append(s)
    # exact oracle on test
    E=p["emission"](Xtest,U0,V0);ll,g,_=d["fb"](E,A0,pi0);pred=g.argmax(1)
    final={"oracle_test_nmi":d["nmi"](ztest,pred),"oracle_test_ari":d["ari"](ztest,pred),"summaries":outs}
    print("INVERSE_PHASEH_BASIN_JSON="+json.dumps(final,separators=(",",":")),flush=True)
