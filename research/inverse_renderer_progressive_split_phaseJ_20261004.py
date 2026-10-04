#!/usr/bin/env python3
# Phase J: blind progressive exact-likelihood state-splitting search.
# Synthetic-only recoverability calibration. NO P70. Truth sealed until final outer-test scoring.
import json, math, urllib.request
import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

PURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/5d46189f00e3a3fa4db3eeb9a1331b48890c8ae7/research/inverse_renderer_piece_oracle_diagnostic_20261004.py"
p={"__name__":"piece"};exec(compile(urllib.request.urlopen(PURL,timeout=60).read().decode(),PURL,"exec"),p)
p["BASE_MIX"]=.02;p["BIAS_BOUND"]=5.0
DURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/dd2c107f3335ed3d022760f67bd0b95885898ba8/research/inverse_renderer_piece_recovery_phaseD_20261004.py"
d={"__name__":"phaseD"};exec(compile(urllib.request.urlopen(DURL,timeout=60).read().decode(),DURL,"exec"),d)

TARGET_K=16; DEG=4; RANK=2; STRENGTH=5.0; SEED=20261004; N=4000
NFIT=2700; NVAL=500; NTR=3200
Atrue,pitrue,Utrue,Vtrue,ztruth,routes=p["generate"](TARGET_K,DEG,RANK,STRENGTH,N,SEED)
X=p["flatten"](routes)
Xfit=X[:NFIT]; Xval=X[NFIT:NTR]; Xtr=X[:NTR]; Xtest=X[NTR:]
ztest=ztruth[NTR:]

device=torch.device("cpu")
torch.set_num_threads(max(1,min(8,torch.get_num_threads())))

def fb(E,A,pi):
    return d["fb"](E,A,pi)

def sparseA(xi,K):
    return d["sparse_A"](xi,min(DEG,K))

def emission(Xb,U,V):
    with torch.no_grad():
        return d["lowrank_E_torch"](Xb,torch.tensor(U,dtype=torch.float32),
                                     torch.tensor(V,dtype=torch.float32),device).numpy().astype(float)

def score(Xb,A,pi,U,V):
    E=emission(Xb,U,V); ll,g,xi=fb(E,A,pi)
    return float(ll),g,xi

def optimize_soft(Xb,A,pi,U,V,K,outer=6,msteps=60,lr=.025):
    Ut=torch.nn.Parameter(torch.tensor(U,dtype=torch.float32))
    Vt=torch.nn.Parameter(torch.tensor(V,dtype=torch.float32))
    opt=torch.optim.Adam([Ut,Vt],lr=lr)
    best=None
    for out in range(outer):
        with torch.no_grad():
            E=d["lowrank_E_torch"](Xb,Ut,Vt,device).numpy().astype(float)
        ll,g,xi=fb(E,A,pi)
        A=sparseA(xi,K)
        pi=g[0]+.1; pi/=pi.sum()
        G=torch.tensor(g,dtype=torch.float32)
        for _ in range(msteps):
            opt.zero_grad(set_to_none=True)
            Et=d["lowrank_E_torch"](Xb,Ut,Vt,device)
            loss=-(G*Et).sum()/len(Xb)+1e-3*(Ut.square().mean()+Vt.square().mean())
            loss.backward(); torch.nn.utils.clip_grad_norm_([Ut,Vt],5.0); opt.step()
        with torch.no_grad():
            E2=d["lowrank_E_torch"](Xb,Ut,Vt,device).numpy().astype(float)
        ll2,g2,xi2=fb(E2,A,pi)
        if best is None or ll2>best[0]:
            best=(float(ll2),A.copy(),pi.copy(),Ut.detach().numpy().copy(),Vt.detach().numpy().copy())
    return best

def observable_features(Xb,nc=12):
    F=csr_matrix(Xb.reshape(len(Xb),-1))
    nc=min(nc,F.shape[1]-1,len(Xb)-1)
    Z=TruncatedSVD(n_components=nc,random_state=SEED).fit_transform(F)
    return StandardScaler().fit_transform(Z)

ZF=observable_features(Xfit,12)

def hard_init(K,seed):
    labels=KMeans(n_clusters=K,n_init=12,random_state=seed,max_iter=300).fit_predict(ZF)
    F=csr_matrix(Xfit.reshape(NFIT,-1))
    TC=np.zeros((K,K),float)
    for a,b in zip(labels[:-1],labels[1:]):TC[int(a),int(b)]+=1
    A=sparseA(TC+.1,K)
    pi=np.bincount(labels[:200],minlength=K).astype(float)+1;pi/=pi.sum()
    Q=d["q_project_from_counts"](d["hard_counts"](F,labels,K),prior=4.0)
    B=d["bias_from_Q"](Q)
    U,V=d["lowrank"](B,RANK)
    return A,pi,U,V

def split_model(parent,seed,eps=.35):
    rng=np.random.default_rng(seed)
    A,pi,U,V=parent["A"],parent["pi"],parent["U"],parent["V"]
    K=A.shape[0]; K2=K*2
    # Duplicate U rows with antisymmetric perturbations.
    U2=np.zeros((K2,RANK),float)
    for k in range(K):
        delta=rng.normal(0,eps,size=RANK)
        U2[2*k]=U[k]+delta
        U2[2*k+1]=U[k]-delta
    V2=V.copy()
    # Split prior mass and parent transitions across child pairs with mild random asymmetry.
    pi2=np.zeros(K2,float)
    for k in range(K):
        q=float(rng.uniform(.40,.60)); pi2[2*k]=pi[k]*q; pi2[2*k+1]=pi[k]*(1-q)
    pi2/=pi2.sum()
    A2=np.zeros((K2,K2),float)
    for i in range(K):
        for ci in (0,1):
            row=np.zeros(K2,float)
            for j in range(K):
                mass=A[i,j]
                q=float(rng.uniform(.35,.65))
                row[2*j]+=mass*q
                row[2*j+1]+=mass*(1-q)
            # encourage child persistence differentiation without inventing impossible edges
            row[2*i+ci]+=0.05
            row/=row.sum()
            A2[2*i+ci]=row
    # Initial topology can be dense for one E step; optimize_soft sparsifies immediately.
    return A2,pi2,U2,V2

def make_rec(A,pi,U,V,K,tag):
    llv,gv,xiv=score(Xval,A,pi,U,V)
    return {"K":K,"tag":tag,"val_ll":llv,"A":A,"pi":pi,"U":U,"V":V}

def slim(r):
    return {"K":r["K"],"tag":r["tag"],"val_ll":r["val_ll"]}

# Truth is not referenced anywhere in search below.
population=[]
for s in range(6):
    A,pi,U,V=hard_init(2,SEED+101*s)
    _,A,pi,U,V=optimize_soft(Xfit,A,pi,U,V,2,outer=7,msteps=70)
    population.append(make_rec(A,pi,U,V,2,f"k2_seed{s}"))
population.sort(key=lambda r:r["val_ll"],reverse=True)
population=population[:4]
print("PHASEJ_STAGE_JSON="+json.dumps({"K":2,"kept":[slim(r) for r in population]},separators=(",",":")),flush=True)

for K2 in (4,8,16):
    cand=[]
    for pi_parent,parent in enumerate(population):
        for b in range(4):
            A,pi,U,V=split_model(parent,SEED+K2*10000+pi_parent*100+b*17,eps=.20 if K2==16 else .30)
            _,A,pi,U,V=optimize_soft(Xfit,A,pi,U,V,K2,outer=7,msteps=60)
            cand.append(make_rec(A,pi,U,V,K2,f"from{pi_parent}_b{b}"))
    cand.sort(key=lambda r:r["val_ll"],reverse=True)
    population=cand[:4]
    print("PHASEJ_STAGE_JSON="+json.dumps({"K":K2,"kept":[slim(r) for r in population]},separators=(",",":")),flush=True)

# Freeze best K16 by inner validation.
winner=population[0]
pre_ll,pre_g,_=score(Xtest,winner["A"],winner["pi"],winner["U"],winner["V"])

# Truth-free refit winner on all non-test observations, then open outer test.
_,Ar,pir,Ur,Vr=optimize_soft(Xtr,winner["A"],winner["pi"],winner["U"],winner["V"],16,outer=10,msteps=80,lr=.02)
post_ll,post_g,_=score(Xtest,Ar,pir,Ur,Vr)

# Only now reveal truth for recovery diagnostics.
pre_pred=pre_g.argmax(1); post_pred=post_g.argmax(1)
Eo=p["emission"](Xtest,Utrue,Vtrue);llo,go,_=fb(Eo,Atrue,pitrue);op=go.argmax(1)
out={
 "phase":"J_progressive_state_split",
 "selected":slim(winner),
 "pre_refit":{"test_ll":pre_ll,"test_nmi":d["nmi"](ztest,pre_pred),"test_ari":d["ari"](ztest,pre_pred)},
 "post_refit":{"test_ll":post_ll,"test_nmi":d["nmi"](ztest,post_pred),"test_ari":d["ari"](ztest,post_pred)},
 "oracle":{"test_ll":float(llo),"test_nmi":d["nmi"](ztest,op),"test_ari":d["ari"](ztest,op)},
 "gate_nmi70":bool(d["nmi"](ztest,post_pred)>=.70)
}
print("INVERSE_PHASEJ_JSON="+json.dumps(out,separators=(",",":")),flush=True)
