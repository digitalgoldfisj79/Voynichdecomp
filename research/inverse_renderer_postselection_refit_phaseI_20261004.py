#!/usr/bin/env python3
# Phase I: exact-family post-selection refit diagnostic.
# Synthetic-only; truth used only for controlled initialization / diagnostic upper bounds.
# NO P70. No real Voynich.
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
NFIT=2700;NVAL=500;NTR=3200
A0,pi0,U0,V0,z,routes=p["generate"](K,DEG,RANK,STRENGTH,N,SEED)
X=p["flatten"](routes)
Xfit=X[:NFIT];Xval=X[NFIT:NTR];Xtr=X[:NTR];Xtest=X[NTR:]
zfit=z[:NFIT];zval=z[NFIT:NTR];ztr=z[:NTR];ztest=z[NTR:]

device=torch.device("cpu")

def viterbi(E,A,pi):
    nn,kk=E.shape;la=np.log(np.maximum(A,1e-30));lp=np.log(np.maximum(pi,1e-30))
    dp=np.empty((nn,kk));bp=np.empty((nn,kk),np.int16);dp[0]=lp+E[0]
    for t in range(1,nn):
        M=dp[t-1][:,None]+la;bp[t]=np.argmax(M,0);dp[t]=M[bp[t],np.arange(kk)]+E[t]
    zz=np.empty(nn,np.int16);zz[-1]=np.argmax(dp[-1])
    for t in range(nn-2,-1,-1):zz[t]=bp[t+1,zz[t+1]]
    return zz,float(dp[-1].max())

def fit_hard(Xb,labels,steps=220,Uprev=None,Vprev=None):
    n=len(Xb);F=csr_matrix(Xb.reshape(n,-1))
    TC=np.zeros((K,K),float)
    for a,b in zip(labels[:-1],labels[1:]):TC[int(a),int(b)]+=1
    A=d["sparse_A"](TC,DEG)
    pi=np.bincount(labels[:min(200,n)],minlength=K).astype(float)+1;pi/=pi.sum()
    if Uprev is None:
        Q=d["q_project_from_counts"](d["hard_counts"](F,labels,K),prior=4.0)
        Ui,Vi=d["lowrank"](d["bias_from_Q"](Q),RANK)
    else: Ui,Vi=Uprev,Vprev
    U=torch.nn.Parameter(torch.tensor(Ui,dtype=torch.float32))
    V=torch.nn.Parameter(torch.tensor(Vi,dtype=torch.float32))
    opt=torch.optim.Adam([U,V],lr=.025);idx=torch.arange(n);zt=torch.tensor(labels,dtype=torch.long)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        E=d["lowrank_E_torch"](Xb,U,V,device)
        loss=-E[idx,zt].mean()+1e-3*(U.square().mean()+V.square().mean())
        loss.backward();torch.nn.utils.clip_grad_norm_([U,V],5.0);opt.step()
    return A,pi,U.detach().numpy(),V.detach().numpy()

def score(Xb,A,pi,U,V):
    with torch.no_grad():
        E=d["lowrank_E_torch"](Xb,torch.tensor(U,dtype=torch.float32),torch.tensor(V,dtype=torch.float32),device).numpy().astype(float)
    ll,g,xi=d["fb"](E,A,pi)
    return float(ll),g,xi,E

def corrupt(frac,seed):
    rng=np.random.default_rng(seed);L=zfit.astype(np.int16).copy()
    n=int(round(frac*len(L)))
    if n:
        ix=rng.choice(len(L),n,replace=False)
        draw=rng.integers(0,K,n,dtype=np.int16)
        same=draw==L[ix];draw[same]=(draw[same]+1)%K;L[ix]=draw
    return L

def select_on_fit(frac):
    L=corrupt(frac,SEED+int(frac*10000)+10000)
    U=V=None;best=None;hist=[]
    for cyc in range(5):
        A,pi,U,V=fit_hard(Xfit,L,180 if cyc==0 else 90,U,V)
        llf,gf,xif,Ef=score(Xfit,A,pi,U,V)
        llv,gv,xiv,Ev=score(Xval,A,pi,U,V)
        Lnew,_=viterbi(Ef,A,pi)
        rec={"cycle":cyc,"fit_ll":llf,"val_ll":llv,"label_nmi_diag":d["nmi"](zfit,L),
             "fit_nmi_diag":d["nmi"](zfit,gf.argmax(1)),"val_nmi_diag":d["nmi"](zval,gv.argmax(1))}
        hist.append(rec)
        if best is None or llv>best["val_ll"]:
            best={"cycle":cyc,"val_ll":llv,"A":A.copy(),"pi":pi.copy(),"U":U.copy(),"V":V.copy(),"rec":rec.copy()}
        L=Lnew
    return best,hist

def soft_refit_exact(Xb,A,pi,U0x,V0x,max_outer=8,msteps=120):
    # Truth-free post-selection refit on fit+validation.
    U=torch.nn.Parameter(torch.tensor(U0x,dtype=torch.float32))
    V=torch.nn.Parameter(torch.tensor(V0x,dtype=torch.float32))
    opt=torch.optim.Adam([U,V],lr=.02)
    best=None;last=-1e300;hist=[]
    for outer in range(max_outer):
        with torch.no_grad():
            E=d["lowrank_E_torch"](Xb,U,V,device).numpy().astype(float)
        ll,g,xi=d["fb"](E,A,pi)
        A=d["sparse_A"](xi,DEG);pi=g[0]+.1;pi/=pi.sum()
        G=torch.tensor(g,dtype=torch.float32)
        for _ in range(msteps):
            opt.zero_grad(set_to_none=True)
            Et=d["lowrank_E_torch"](Xb,U,V,device)
            loss=-(G*Et).sum()/len(Xb)+1e-3*(U.square().mean()+V.square().mean())
            loss.backward();torch.nn.utils.clip_grad_norm_([U,V],5.0);opt.step()
        with torch.no_grad():
            E2=d["lowrank_E_torch"](Xb,U,V,device).numpy().astype(float)
        ll2,g2,xi2=d["fb"](E2,A,pi)
        hist.append({"outer":outer,"train_ll":float(ll2)})
        if best is None or ll2>best[0]:
            best=(float(ll2),A.copy(),pi.copy(),U.detach().numpy().copy(),V.detach().numpy().copy())
        if outer>2 and abs(ll2-last)<1e-3:break
        last=ll2
    return best,hist

def eval_model(A,pi,U,V):
    llt,gt,_,_=score(Xtest,A,pi,U,V);pt=gt.argmax(1)
    return {"test_ll":llt,"test_nmi":d["nmi"](ztest,pt),"test_ari":d["ari"](ztest,pt)}

def supervised_upper(labels,Xb):
    A,pi,U,V=fit_hard(Xb,labels,400)
    return eval_model(A,pi,U,V)

if __name__=="__main__":
    # Diagnostic sample-size upper bounds using truth only.
    ub2700=supervised_upper(zfit,Xfit)
    ub3200=supervised_upper(ztr,Xtr)
    Eo=p["emission"](Xtest,U0,V0);llo,go,_=d["fb"](Eo,A0,pi0);po=go.argmax(1)
    oracle={"test_ll":float(llo),"test_nmi":d["nmi"](ztest,po),"test_ari":d["ari"](ztest,po)}
    print("PHASEI_UPPER_JSON="+json.dumps({"oracle":oracle,"supervised_2700":ub2700,"supervised_3200":ub3200},separators=(",",":")),flush=True)

    outs=[]
    for frac in (0.0,.20,.40):
        best,hist=select_on_fit(frac)
        pre=eval_model(best["A"],best["pi"],best["U"],best["V"])
        ref,href=soft_refit_exact(Xtr,best["A"],best["pi"],best["U"],best["V"])
        _,Ar,pir,Ur,Vr=ref
        post=eval_model(Ar,pir,Ur,Vr)
        rec={"corruption":frac,"selected_cycle":best["cycle"],"selected_val_ll":best["val_ll"],
             "pre_refit":pre,"post_refit":post,"selection_history":hist,"refit_history":href}
        print("PHASEI_REFIT_JSON="+json.dumps(rec,separators=(",",":")),flush=True);outs.append(rec)
    print("INVERSE_PHASEI_JSON="+json.dumps({"upper":{"oracle":oracle,"supervised_2700":ub2700,"supervised_3200":ub3200},"refits":outs},separators=(",",":")),flush=True)
