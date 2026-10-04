#!/usr/bin/env python3
# Phase A v3 development: emission-first mixture warm start -> temporal HMM.
import json,math,time,urllib.request
import numpy as np, torch
from sklearn.metrics import adjusted_mutual_info_score

BASE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/83d96e02bf2149946bf849697e22217a410f42e4/research/inverse_renderer_recoverability_phaseA_20261004.py"
V2_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/1973ce55b572753fcb57fc020d423fb5ee111ad2/research/inverse_renderer_phaseA_v2_topology_explore_20261004.py"
b={"__name__":"base"};exec(compile(urllib.request.urlopen(BASE_URL,timeout=60).read().decode(),BASE_URL,"exec"),b)
v={"__name__":"v2"};exec(compile(urllib.request.urlopen(V2_URL,timeout=60).read().decode(),V2_URL,"exec"),v)

def softmax_np(A):
    m=A.max(1,keepdims=True);Q=np.exp(A-m);Q/=Q.sum(1,keepdims=True);return Q

def mixture_warm(X,K,rank,seed,device,rounds=25,msteps=15,lr=.06):
    rng=np.random.default_rng(seed)
    U=torch.nn.Parameter(torch.tensor(rng.normal(0,.45,(K,rank)),dtype=torch.float32,device=device))
    V=torch.nn.Parameter(torch.tensor(rng.normal(0,.45,(rank,b["NCTX"],b["NOPT"])),dtype=torch.float32,device=device))
    opt=torch.optim.Adam([U,V],lr=lr);w=np.ones(K)/K
    for it in range(rounds):
        with torch.no_grad():E=b["emission_loglik"](X,U,V,device).cpu().numpy().astype(np.float64)
        G=softmax_np(E+np.log(w+1e-12)[None,:])
        w=(G.sum(0)+1)/(len(X)+K)
        Gt=torch.tensor(G,dtype=torch.float32,device=device)
        for _ in range(msteps):
            opt.zero_grad(set_to_none=True);E2=b["emission_loglik"](X,U,V,device)
            loss=-(Gt*E2).sum()/len(X)+1e-3*(U.square().mean()+V.square().mean())+1e-2*U.mean(0).square().mean()
            loss.backward();opt.step()
    with torch.no_grad():E=b["emission_loglik"](X,U,V,device).cpu().numpy().astype(np.float64)
    G=softmax_np(E+np.log(w+1e-12)[None,:])
    # Soft adjacent-state counts provide a dense temporal initializer.
    xi=np.einsum("ti,tj->ij",G[:-1],G[1:])
    A=v["dense_update"](xi,.25)
    pi=G[0]+.1;pi/=pi.sum()
    return A,pi,U,V,G

def fit_v3(X,K,d,rank,seed,device,epochs=50,msteps=18,lr=.05,eta=.02):
    t0=time.time()
    A,pi,U,V,G=mixture_warm(X,K,rank,seed,device,25,15,.06)
    opt=torch.optim.Adam([U,V],lr=lr);last=-1e99;best=None
    for em in range(epochs):
        with torch.no_grad():E=b["emission_loglik"](X,U,V,device).cpu().numpy().astype(np.float64)
        ll,g,xi=b["fb_numpy"](E,A,pi)
        # keep topology exploratory for first 12 HMM rounds, then concentrate.
        A=v["explore_project"](xi,d,.06 if em<12 else eta,.05)
        pi=g[0]+.1;pi/=pi.sum()
        Gt=torch.tensor(g,dtype=torch.float32,device=device)
        for _ in range(msteps):
            opt.zero_grad(set_to_none=True);E2=b["emission_loglik"](X,U,V,device)
            loss=-(Gt*E2).sum()/len(X)+1e-3*(U.square().mean()+V.square().mean())+1e-2*U.mean(0).square().mean()
            loss.backward();opt.step()
        if ll>last:best=(ll,A.copy(),pi.copy(),U.detach().cpu().numpy().copy(),V.detach().cpu().numpy().copy(),em)
        if abs(ll-last)<1e-3 and em>20:break
        last=ll
    ll,A,pi,U0,V0,em=best
    Ah=v["hard_project"](A,d)
    with torch.no_grad():E=b["emission_loglik"](X,torch.tensor(U0,dtype=torch.float32),
                                                  torch.tensor(V0,dtype=torch.float32),device).numpy().astype(np.float64)
    llh,g,_=b["fb_numpy"](E,Ah,pi)
    return {"ll":llh,"A":Ah,"pi":pi,"U":U0,"V":V0,"gamma":g,"epochs":em+1,"seconds":time.time()-t0}

def eval_rep(rep):
    torch.set_num_threads(1);seed=20264000+rep*1009
    dat=b["generate_corpus"](16,4,2,3.5,4000,seed);X=b["flatten_decisions"](dat["routes"])
    Xtr,Xte=X[:3200],X[3200:];ztr,zte=dat["z"][:3200],dat["z"][3200:]
    runs=[]
    for j in range(4):
        f=fit_v3(Xtr,16,4,2,seed+70000+j*8191,torch.device("cpu"))
        ptr=f["gamma"].argmax(1);tr=float(adjusted_mutual_info_score(ztr,ptr))
        with torch.no_grad():E=b["emission_loglik"](Xte,torch.tensor(f["U"],dtype=torch.float32),
                    torch.tensor(f["V"],dtype=torch.float32),torch.device("cpu")).numpy().astype(np.float64)
        llte,g,_=b["fb_numpy"](E,f["A"],f["pi"]);pte=g.argmax(1);te=float(adjusted_mutual_info_score(zte,pte))
        runs.append({"j":j,"train_ll":f["ll"],"test_ll":llte,"train_ami":tr,"test_ami":te,"seconds":f["seconds"]})
    runs.sort(key=lambda x:x["train_ll"],reverse=True)
    return {"replicate":rep,"best":runs[0],"runs":runs}

if __name__=="__main__":
    # Failed and borderline v2 panel cases + one easy positive control.
    for rep in (2,6,9,18,0,3,14):
        z=eval_rep(rep);print("PHASEA_V3_DEV_JSON="+json.dumps(z,separators=(",",":")),flush=True)
