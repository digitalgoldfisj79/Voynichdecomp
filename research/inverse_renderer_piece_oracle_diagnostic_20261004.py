#!/usr/bin/env python3
# Phase C: exact-piece channel oracle identifiability diagnostic. NO P70.
import json,math,urllib.request
import numpy as np
import torch

BASE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/664cd58a01a0d26bf66619fafc92945bee1ffe75/research/inverse_renderer_recoverability_phaseA_20261004.py"
m={"__name__":"phaseA"}
exec(compile(urllib.request.urlopen(BASE_URL,timeout=60).read().decode(),BASE_URL,"exec"),m)
rows=m["rows"]; segment=m["ns"]["segment"] if "ns" in m else None
if segment is None:
    # phaseA exports imported latent namespace only indirectly; recover segment from token route machinery source namespace
    LAT_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/74ae7aa0088dc07436000f777d9c5d7d705730c2/research/latent_line_state_test_20261004.py"
    z={"__name__":"latent"};exec(compile(urllib.request.urlopen(LAT_URL).read().decode(),LAT_URL,"exec"),z);segment=z["segment"]

allseq=[segment(r["token"]) for r in rows]
PIECES=sorted({p for s in allseq for p in s})
PID={p:i for i,p in enumerate(PIECES)}
NP=len(PIECES);START=NP;END=NP;NCTX=NP+1;NOPT=NP+1
BASE_MIX=.10;BIAS_BOUND=3.0

counts=np.zeros((NCTX,NOPT),float)
for s in allseq:
    if not s:continue
    ids=[PID[p] for p in s]
    counts[START,ids[0]]+=1
    for i,p in enumerate(ids):
        counts[p,END if i==len(ids)-1 else ids[i+1]]+=1
SUPPORT=counts>0
P0=(counts+0.25*SUPPORT)
P0=np.divide(P0,P0.sum(1,keepdims=True),out=np.zeros_like(P0),where=P0.sum(1,keepdims=True)>0)
LOGP0=np.where(SUPPORT,np.log(np.maximum(P0,1e-30)),-1e30)

def source_graph(K,d,rng):
    A=np.zeros((K,K),float)
    for i in range(K):
        keep={i}
        while len(keep)<d:keep.add(int(rng.integers(K)))
        js=sorted(keep);w=rng.gamma(1.5,1.0,len(js));w[js.index(i)]+=2;w/=w.sum();A[i,js]=w
    return A,np.ones(K)/K

def sample_hidden(A,pi,N,rng):
    z=np.empty(N,int);z[0]=rng.choice(len(pi),p=pi)
    for t in range(1,N):z[t]=rng.choice(len(pi),p=A[z[t-1]])
    return z

def coupling(K,rank,strength,rng):
    U=rng.normal(size=(K,rank));U-=U.mean(0);U/=np.maximum(U.std(0),1e-9)
    V=rng.normal(size=(rank,NCTX,NOPT));V-=V.mean(2,keepdims=True);V/=np.maximum(V.std(),1e-9)
    return U,V*strength

def q_numpy(Urow,V,ctx):
    b=BIAS_BOUND*np.tanh((Urow@V[:,ctx,:])/BIAS_BOUND)
    ok=SUPPORT[ctx]
    if ctx==START:
        lg=LOGP0[ctx]+b;lg[~ok]=-1e30;m=lg.max();qb=np.exp(lg-m);qb/=qb.sum()
    else:
        pend=P0[ctx,END] if SUPPORT[ctx,END] else 0.
        non=ok.copy();non[END]=False
        qb=np.zeros(NOPT);qb[END]=pend
        if non.any():
            lg=LOGP0[ctx]+b;lg[~non]=-1e30;m=lg[non].max();v=np.exp(lg[non]-m);v/=v.sum()
            qb[non]=(1-pend)*v
    q=BASE_MIX*P0[ctx]+(1-BASE_MIX)*qb;q[~ok]=0;q/=q.sum()
    return q

def sample_route(z,U,V,rng,maxlen=2048):
    ctx=START;out=[]
    for _ in range(maxlen):
        q=q_numpy(U[z],V,ctx);o=int(rng.choice(NOPT,p=q))
        if o==END:
            if out:return out
            continue
        out.append(o);ctx=o
    raise RuntimeError(("nontermination",z,maxlen))

def generate(K,d,rank,strength,N,seed):
    rng=np.random.default_rng(seed);A,pi=source_graph(K,d,rng);U,V=coupling(K,rank,strength,rng)
    z=sample_hidden(A,pi,N,rng);routes=[sample_route(int(s),U,V,rng) for s in z]
    return A,pi,U,V,z,routes

def flatten(routes):
    X=np.zeros((len(routes),NCTX,NOPT),np.float32)
    for t,s in enumerate(routes):
        X[t,START,s[0]]+=1
        for i,p in enumerate(s):X[t,p,END if i==len(s)-1 else s[i+1]]+=1
    return X

def emission(X,U,V):
    U=torch.tensor(U,dtype=torch.float32);V=torch.tensor(V,dtype=torch.float32)
    raw=torch.einsum("kr,rco->kco",U,V);bias=BIAS_BOUND*torch.tanh(raw/BIAS_BOUND)
    p0=torch.tensor(P0,dtype=torch.float32);mask=torch.tensor(SUPPORT,dtype=torch.bool)
    qs=[]
    for ctx in range(NCTX):
        if ctx==START:
            lg=torch.tensor(LOGP0[ctx],dtype=torch.float32).unsqueeze(0)+bias[:,ctx,:]
            lg=torch.where(mask[ctx].unsqueeze(0),lg,torch.tensor(-1e30));qb=torch.softmax(lg,1)
        else:
            pend=p0[ctx,END] if SUPPORT[ctx,END] else torch.tensor(0.)
            non=mask[ctx].clone();non[END]=False
            lg=torch.tensor(LOGP0[ctx],dtype=torch.float32).unsqueeze(0)+bias[:,ctx,:]
            lg=torch.where(non.unsqueeze(0),lg,torch.tensor(-1e30));qn=torch.softmax(lg,1)
            qb=(1-pend)*qn
            if SUPPORT[ctx,END]:qb[:,END]=pend
        q=BASE_MIX*p0[ctx].unsqueeze(0)+(1-BASE_MIX)*qb
        q=torch.where(mask[ctx].unsqueeze(0),q,torch.tensor(0.));q/=q.sum(1,keepdim=True)
        qs.append(q)
    logq=torch.log(torch.clamp(torch.stack(qs,1),min=1e-30))
    return torch.einsum("nco,kco->nk",torch.tensor(X),logq).numpy().astype(float)

def nmi(a,b):
    return m["nmi"](a,b)
def ari(a,b):
    return m["ari"](a,b)

if __name__=="__main__":
    out=[]
    for strength in (.5,1.,1.5,2.,2.5,3.):
        reps=[]
        for rep,seed in enumerate((20261004,20261005,20261006)):
            A,pi,U,V,z,routes=generate(16,4,2,strength,4000,seed);X=flatten(routes)
            E=emission(X,U,V);ll,g,_=m["fb_numpy"](E,A,pi);pred=g.argmax(1)
            rr={"seed":seed,"nmi":nmi(z,pred),"ari":ari(z,pred),"ll":ll,
                "mean_route_len":float(np.mean([len(x) for x in routes])),
                "max_route_len":int(max(map(len,routes)))}
            reps.append(rr)
        r={"strength":strength,"median_nmi":float(np.median([x["nmi"] for x in reps])),
           "min_nmi":float(min(x["nmi"] for x in reps])),"reps":reps}
        print("PIECE_ORACLE_STRENGTH_JSON="+json.dumps(r,separators=(",",":")),flush=True);out.append(r)
    final={"piece_count":NP,"legal_edges":int(SUPPORT.sum()),"base_mix":BASE_MIX,"bias_bound":BIAS_BOUND,
           "strengths":out}
    print("PIECE_ORACLE_DIAGNOSTIC_JSON="+json.dumps(final,separators=(",",":")),flush=True)
