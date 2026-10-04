#!/usr/bin/env python3
# Phase E: collapsed combinatorial partition search + exact low-rank prospective test.
# Synthetic-only recoverability gate. NO P70.
import json,math,urllib.request
import numpy as np
import torch
from numba import njit
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/5d46189f00e3a3fa4db3eeb9a1331b48890c8ae7/research/inverse_renderer_piece_oracle_diagnostic_20261004.py"
p={"__name__":"piece"};exec(compile(urllib.request.urlopen(PURL,timeout=60).read().decode(),PURL,"exec"),p)
p["BASE_MIX"]=.02;p["BIAS_BOUND"]=5.0
DURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/dd2c107f3335ed3d022760f67bd0b95885898ba8/research/inverse_renderer_piece_recovery_phaseD_20261004.py"
dmod={"__name__":"phaseD"};exec(compile(urllib.request.urlopen(DURL,timeout=60).read().decode(),DURL,"exec"),dmod)

K=16;D=4;RANK=2;N=4000;STRENGTH=5.0;SEED=20261004
A0,pi0,U0,V0,z,routes=p["generate"](K,D,RANK,STRENGTH,N,SEED)
X=p["flatten"](routes);ntr=3200;Xtr,Xte=X[:ntr],X[ntr:];ztr,zte=z[:ntr],z[ntr:]
P0=p["P0"];SUP=p["SUPPORT"];START=p["START"];END=p["END"];C=p["NCTX"];O=p["NOPT"]

# Source-controlled decisions only: START choice and non-END continuation choices.
maxd=max(len(r) for r in routes[:ntr])
ctx=np.full((ntr,maxd),-1,np.int16);opt=np.full((ntr,maxd),-1,np.int16);nd=np.zeros(ntr,np.int16)
for t,r in enumerate(routes[:ntr]):
    arr=[(START,r[0])]+[(r[i],r[i+1]) for i in range(len(r)-1)]
    nd[t]=len(arr)
    for j,(c,o) in enumerate(arr):ctx[t,j]=c;opt[t,j]=o

# Renderer-centred base categorical over source-controlled alternatives.
PB=np.zeros((C,O),np.float64)
for c in range(C):
    allowed=SUP[c].copy();allowed[END]=False
    if allowed.any():
        v=P0[c,allowed];PB[c,allowed]=v/max(v.sum(),1e-30)

@njit
def init_counts(labels,ctx,opt,nd,K,C,O):
    ec=np.zeros((K,C,O),np.int32);et=np.zeros((K,C),np.int32)
    tc=np.zeros((K,K),np.int32);tt=np.zeros(K,np.int32)
    N=len(labels)
    for t in range(N):
        k=labels[t]
        for j in range(nd[t]):
            c=ctx[t,j];o=opt[t,j];ec[k,c,o]+=1;et[k,c]+=1
        if t+1<N:
            a=labels[t];b=labels[t+1];tc[a,b]+=1;tt[a]+=1
    return ec,et,tc,tt

@njit
def remove_token(t,labels,ctx,opt,nd,ec,et,tc,tt):
    old=labels[t];N=len(labels)
    for j in range(nd[t]):
        c=ctx[t,j];o=opt[t,j];ec[old,c,o]-=1;et[old,c]-=1
    if t>0:
        a=labels[t-1];tc[a,old]-=1;tt[a]-=1
    if t+1<N:
        b=labels[t+1];tc[old,b]-=1;tt[old]-=1
    return old

@njit
def add_token(t,k,labels,ctx,opt,nd,ec,et,tc,tt):
    N=len(labels);labels[t]=k
    for j in range(nd[t]):
        c=ctx[t,j];o=opt[t,j];ec[k,c,o]+=1;et[k,c]+=1
    if t>0:
        a=labels[t-1];tc[a,k]+=1;tt[a]+=1
    if t+1<N:
        b=labels[t+1];tc[k,b]+=1;tt[k]+=1

@njit
def cand_logp(t,k,labels,ctx,opt,nd,ec,et,tc,tt,PB,tau,alpha,K):
    lp=0.0
    # Emission posterior predictive, accounting for duplicate decisions inside token.
    # Tokens are short, scan earlier within-token decisions for local increments.
    for j in range(nd[t]):
        c=ctx[t,j];o=opt[t,j]
        addco=0;addc=0
        for h in range(j):
            if ctx[t,h]==c:
                addc+=1
                if opt[t,h]==o:addco+=1
        base=PB[c,o]
        lp+=math.log(ec[k,c,o]+tau*base+addco+1e-300)-math.log(et[k,c]+tau+addc)
    # Add prev->k then k->next sequentially under row-Dirichlet transition prior.
    N=len(labels)
    extra_row=-1;extra_col=-1
    if t>0:
        a=labels[t-1]
        lp+=math.log(tc[a,k]+alpha)-math.log(tt[a]+K*alpha)
        extra_row=a;extra_col=k
    if t+1<N:
        b=labels[t+1]
        num=tc[k,b]+alpha
        den=tt[k]+K*alpha
        if extra_row==k:
            den+=1
            if extra_col==b:num+=1
        lp+=math.log(num)-math.log(den)
    return lp

@njit
def collapsed_score(ec,et,tc,tt,PB,tau,alpha,K,C,O):
    s=0.0
    # Transition Dirichlet marginal.
    for a in range(K):
        s+=math.lgamma(K*alpha)-math.lgamma(tt[a]+K*alpha)
        for b in range(K):
            s+=math.lgamma(tc[a,b]+alpha)-math.lgamma(alpha)
    # Emission independent context marginals; only positive PB support.
    for k in range(K):
        for c in range(C):
            if et[k,c]==0:continue
            s+=math.lgamma(tau)-math.lgamma(et[k,c]+tau)
            for o in range(O):
                if PB[c,o]>0:
                    aa=tau*PB[c,o]
                    s+=math.lgamma(ec[k,c,o]+aa)-math.lgamma(aa)
    return s

@njit
def chain(labels0,ctx,opt,nd,PB,tau,alpha,sweeps,seed,K,C,O):
    np.random.seed(seed);labels=labels0.copy()
    ec,et,tc,tt=init_counts(labels,ctx,opt,nd,K,C,O)
    N=len(labels);best=labels.copy();bs=collapsed_score(ec,et,tc,tt,PB,tau,alpha,K,C,O)
    for sw in range(sweeps):
        # anneal for first 60%, then exact T=1 Gibbs-like sweep
        if sw < int(.6*sweeps):
            temp=1.8-0.8*(sw/max(int(.6*sweeps)-1,1))
        else:temp=1.0
        offset=np.random.randint(N)
        for ii in range(N):
            t=(ii+offset)%N
            old=remove_token(t,labels,ctx,opt,nd,ec,et,tc,tt)
            lps=np.empty(K,np.float64)
            mx=-1e300
            for k in range(K):
                v=cand_logp(t,k,labels,ctx,opt,nd,ec,et,tc,tt,PB,tau,alpha,K)/temp
                lps[k]=v
                if v>mx:mx=v
            sm=0.0
            for k in range(K):
                lps[k]=math.exp(lps[k]-mx);sm+=lps[k]
            u=np.random.random()*sm;acc=0.0;pick=K-1
            for k in range(K):
                acc+=lps[k]
                if u<=acc:
                    pick=k;break
            add_token(t,pick,labels,ctx,opt,nd,ec,et,tc,tt)
        sc=collapsed_score(ec,et,tc,tt,PB,tau,alpha,K,C,O)
        if sc>bs:
            bs=sc;best=labels.copy()
    return best,bs

def init_white():
    ctxn=Xtr.sum(2,keepdims=True);R0=Xtr-ctxn*P0[None,:,:];W=R0/np.sqrt(P0[None,:,:]+.01)
    F=W.reshape(ntr,-1);Z=TruncatedSVD(n_components=16,random_state=SEED).fit_transform(F)
    Z=StandardScaler().fit_transform(Z)
    return KMeans(K,n_init=20,random_state=SEED,max_iter=400).fit_predict(Z).astype(np.int16)

def fit_rank2_from_labels(labels):
    F=csr_matrix(Xtr.reshape(ntr,-1))
    # Sparse transition estimate.
    TC=np.zeros((K,K),float)
    for a,b in zip(labels[:-1],labels[1:]):TC[a,b]+=1
    A=dmod["sparse_A"](TC,D)
    pi=np.bincount(labels[:200],minlength=K).astype(float)+1;pi/=pi.sum()
    Q=dmod["q_project_from_counts"](dmod["hard_counts"](F,labels,K),prior=4.0)
    Ui,Vi=dmod["lowrank"](dmod["bias_from_Q"](Q),RANK)
    device=torch.device("cpu")
    U=torch.nn.Parameter(torch.tensor(Ui,dtype=torch.float32));V=torch.nn.Parameter(torch.tensor(Vi,dtype=torch.float32))
    optimizer=torch.optim.Adam([U,V],lr=.03);idx=torch.arange(ntr);zt=torch.tensor(labels,dtype=torch.long)
    for step in range(300):
        optimizer.zero_grad(set_to_none=True);E=dmod["lowrank_E_torch"](Xtr,U,V,device)
        loss=-E[idx,zt].mean()+1e-3*(U.square().mean()+V.square().mean())
        loss.backward();torch.nn.utils.clip_grad_norm_([U,V],5.0);optimizer.step()
    with torch.no_grad():Ete=dmod["lowrank_E_torch"](Xte,U,V,device).numpy().astype(float)
    ll,g,_=dmod["fb"](Ete,A,pi);pred=g.argmax(1)
    return {"test_ll":float(ll),"test_nmi":dmod["nmi"](zte,pred),"test_ari":dmod["ari"](zte,pred)}

if __name__=="__main__":
    base=init_white()
    rng=np.random.default_rng(SEED)
    specs=[]
    specs.append(("white",base))
    for frac in (.15,.35,.60):
        for r in range(2):
            L=base.copy();ix=rng.choice(ntr,int(frac*ntr),replace=False);L[ix]=rng.integers(0,K,len(ix));specs.append((f"white_noise{frac}_{r}",L))
    for r in range(5):specs.append((f"random{r}",rng.integers(0,K,ntr,dtype=np.int16)))
    outs=[]
    for i,(name,L0) in enumerate(specs):
        L,sc=chain(L0,ctx,opt,nd,PB,5.0,.10,180,SEED+100+i,K,C,O)
        rec={"name":name,"score":float(sc),"init_nmi":dmod["nmi"](ztr,L0),
             "final_nmi":dmod["nmi"](ztr,L),"final_ari":dmod["ari"](ztr,L)}
        print("COLLAPSED_CHAIN_JSON="+json.dumps(rec,separators=(",",":")),flush=True);outs.append((rec,L.copy()))
    # Selection uses score only, never planted labels.
    outs.sort(key=lambda x:x[0]["score"],reverse=True)
    finalists=[]
    for rec,L in outs[:3]:
        exact=fit_rank2_from_labels(L);q={**rec,**exact};finalists.append(q)
        print("COLLAPSED_FINALIST_JSON="+json.dumps(q,separators=(",",":")),flush=True)
    result={"oracle_test_nmi":0.7597784878266862,"best_by_score":finalists,
            "score_order":[x[0] for x in outs]}
    print("INVERSE_PHASEE_JSON="+json.dumps(result,separators=(",",":")),flush=True)
