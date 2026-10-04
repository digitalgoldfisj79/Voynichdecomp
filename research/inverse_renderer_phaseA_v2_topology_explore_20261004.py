#!/usr/bin/env python3
# Phase A v2: topology-exploring EM.
import argparse,json,math,time,urllib.request
import numpy as np, torch
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/83d96e02bf2149946bf849697e22217a410f42e4/research/inverse_renderer_recoverability_phaseA_20261004.py"
b={"__name__":"phaseA_base"};exec(compile(urllib.request.urlopen(URL,timeout=60).read().decode(),URL,"exec"),b)

def dense_A(K,rng,sticky=2.0):
    A=rng.gamma(1.5,1.0,(K,K));A[np.arange(K),np.arange(K)]+=sticky
    A/=A.sum(1,keepdims=True);return A

def dense_update(xi,smooth=.25):
    A=xi+smooth;A/=A.sum(1,keepdims=True);return A

def explore_project(xi,d,eta=.03,smooth=.05):
    K=xi.shape[0];A=np.zeros_like(xi,dtype=float)
    for i in range(K):
        row=xi[i]+smooth
        order=np.argsort(row)[::-1].tolist();keep=[i]
        for j in order:
            if j not in keep:keep.append(int(j))
            if len(keep)>=d:break
        other=[j for j in range(K) if j not in keep]
        w=row[keep];w/=w.sum()
        A[i,keep]=(1-eta)*w
        if other:A[i,other]=eta/len(other)
        else:A[i,keep]=w
    return A

def hard_project(A,d):
    K=A.shape[0];B=np.zeros_like(A)
    for i in range(K):
        order=np.argsort(A[i])[::-1].tolist();keep=[i]
        for j in order:
            if j not in keep:keep.append(int(j))
            if len(keep)>=d:break
        w=A[i,keep];w/=w.sum();B[i,keep]=w
    return B

def fit_v2(X,K,d,rank,seed,device,epochs=55,msteps=20,lr=.06,warm=12,eta=.03):
    rng=np.random.default_rng(seed);A=dense_A(K,rng);pi=np.ones(K)/K
    U=torch.nn.Parameter(torch.tensor(rng.normal(0,.35,(K,rank)),dtype=torch.float32,device=device))
    V=torch.nn.Parameter(torch.tensor(rng.normal(0,.35,(rank,b["NCTX"],b["NOPT"])),dtype=torch.float32,device=device))
    opt=torch.optim.Adam([U,V],lr=lr);last=-1e99;best=None;t0=time.time()
    for em in range(epochs):
        with torch.no_grad():E=b["emission_loglik"](X,U,V,device).cpu().numpy().astype(np.float64)
        ll,g,xi=b["fb_numpy"](E,A,pi)
        A=dense_update(xi,.25) if em<warm else explore_project(xi,d,eta,.05)
        pi=g[0]+.1;pi/=pi.sum()
        G=torch.tensor(g,dtype=torch.float32,device=device)
        for _ in range(msteps):
            opt.zero_grad(set_to_none=True);E2=b["emission_loglik"](X,U,V,device)
            loss=-(G*E2).sum()/len(X)+1e-3*(U.square().mean()+V.square().mean())+1e-2*U.mean(0).square().mean()
            loss.backward();opt.step()
        if ll>last:best=(ll,A.copy(),pi.copy(),U.detach().cpu().numpy().copy(),V.detach().cpu().numpy().copy(),em)
        if abs(ll-last)<1e-3 and em>max(warm+5,20):break
        last=ll
    ll,A,pi,U0,V0,em=best
    # final hard sparse graph, then posterior
    Ah=hard_project(A,d)
    with torch.no_grad():
        E=b["emission_loglik"](X,torch.tensor(U0,dtype=torch.float32,device=device),
                               torch.tensor(V0,dtype=torch.float32,device=device),device).cpu().numpy().astype(np.float64)
    llh,g,xi=b["fb_numpy"](E,Ah,pi)
    return {"ll":llh,"A":Ah,"pi":pi,"U":U0,"V":V0,"gamma":g,"epochs":em+1,"seconds":time.time()-t0}

def edge_f1(Atrue,Ahat,d):
    K=Atrue.shape[0];tp=fp=fn=0
    for i in range(K):
        t=set(np.argsort(Atrue[i])[-d:]);h=set(np.argsort(Ahat[i])[-d:])
        tp+=len(t&h);fp+=len(h-t);fn+=len(t-h)
    p=tp/max(tp+fp,1);r=tp/max(tp+fn,1)
    return 2*p*r/max(p+r,1e-15)

def main():
    device=torch.device("cpu")
    outs=[]
    for strength in (1.25,1.75,2.5,3.5):
        dat=b["generate_corpus"](16,4,2,strength,4000,20261004);X=b["flatten_decisions"](dat["routes"])
        rr=[]
        for j in range(4):
            f=fit_v2(X,16,4,2,20263000+j*997+int(strength*100),device)
            pred=f["gamma"].argmax(1)
            rec={"restart":j,"nmi":b["nmi"](dat["z"],pred),"ari":b["ari"](dat["z"],pred),
                 "edge_f1_unaligned":edge_f1(dat["A"],f["A"],4),"ll":f["ll"],"epochs":f["epochs"],"seconds":f["seconds"]}
            print("PHASEA_V2_RESTART_JSON="+json.dumps({"strength":strength}|rec,separators=(",",":")),flush=True);rr.append(rec)
        z={"strength":strength,"best_nmi":max(x["nmi"] for x in rr),"median_nmi":float(np.median([x["nmi"] for x in rr])),
           "pass70_fraction":float(np.mean([x["nmi"]>=.70 for x in rr])),"runs":rr}
        print("PHASEA_V2_STRENGTH_JSON="+json.dumps(z,separators=(",",":")),flush=True);outs.append(z)
    print("PHASEA_V2_JSON="+json.dumps(outs,separators=(",",":")),flush=True)
if __name__=="__main__":main()
