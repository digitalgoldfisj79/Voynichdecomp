#!/usr/bin/env python3
# Phase A v4: dense transition relaxation, one-time sparse projection, sparse refit.
import concurrent.futures,json,os,time,urllib.request
import numpy as np,torch
from sklearn.metrics import adjusted_mutual_info_score
BASE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/83d96e02bf2149946bf849697e22217a410f42e4/research/inverse_renderer_recoverability_phaseA_20261004.py"
V2_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/1973ce55b572753fcb57fc020d423fb5ee111ad2/research/inverse_renderer_phaseA_v2_topology_explore_20261004.py"
b={"__name__":"base"};exec(compile(urllib.request.urlopen(BASE_URL,timeout=60).read().decode(),BASE_URL,"exec"),b)
v={"__name__":"v2"};exec(compile(urllib.request.urlopen(V2_URL,timeout=60).read().decode(),V2_URL,"exec"),v)

def fit_dense_then_sparse(X,K,d,rank,seed,device,dense_epochs=60,sparse_epochs=15,msteps=20,lr=.06):
    rng=np.random.default_rng(seed);A=v["dense_A"](K,rng);pi=np.ones(K)/K
    U=torch.nn.Parameter(torch.tensor(rng.normal(0,.30,(K,rank)),dtype=torch.float32,device=device))
    V=torch.nn.Parameter(torch.tensor(rng.normal(0,.30,(rank,b["NCTX"],b["NOPT"])),dtype=torch.float32,device=device))
    opt=torch.optim.Adam([U,V],lr=lr);last=-1e99;t0=time.time()
    for em in range(dense_epochs):
        with torch.no_grad():E=b["emission_loglik"](X,U,V,device).cpu().numpy().astype(np.float64)
        ll,g,xi=b["fb_numpy"](E,A,pi)
        # essentially unregularised dense M-step, tiny pseudocount.
        A=v["dense_update"](xi,.03);pi=g[0]+.1;pi/=pi.sum()
        G=torch.tensor(g,dtype=torch.float32,device=device)
        for _ in range(msteps):
            opt.zero_grad(set_to_none=True);E2=b["emission_loglik"](X,U,V,device)
            loss=-(G*E2).sum()/len(X)+1e-3*(U.square().mean()+V.square().mean())+1e-2*U.mean(0).square().mean()
            loss.backward();opt.step()
        if abs(ll-last)<1e-3 and em>20:break
        last=ll
    # one-time projection to required d-sparse graph.
    A=v["hard_project"](A,d)
    # fixed-support sparse refit: only existing d edges may update.
    mask=A>0
    last2=-1e99;best=None
    for se in range(sparse_epochs):
        with torch.no_grad():E=b["emission_loglik"](X,U,V,device).cpu().numpy().astype(np.float64)
        ll,g,xi=b["fb_numpy"](E,A,pi)
        AA=(xi+.03)*mask
        AA/=np.maximum(AA.sum(1,keepdims=True),1e-30);A=AA
        pi=g[0]+.1;pi/=pi.sum()
        G=torch.tensor(g,dtype=torch.float32,device=device)
        for _ in range(msteps):
            opt.zero_grad(set_to_none=True);E2=b["emission_loglik"](X,U,V,device)
            loss=-(G*E2).sum()/len(X)+1e-3*(U.square().mean()+V.square().mean())+1e-2*U.mean(0).square().mean()
            loss.backward();opt.step()
        if ll>last2:best=(ll,A.copy(),pi.copy(),U.detach().cpu().numpy().copy(),V.detach().cpu().numpy().copy(),g.copy())
        if abs(ll-last2)<1e-3 and se>5:break
        last2=ll
    ll,A,pi,U0,V0,g=best
    return {"ll":float(ll),"A":A,"pi":pi,"U":U0,"V":V0,"gamma":g,"seconds":time.time()-t0}

REPS=(2,6,9,18);NR=8
DATA={}
for rep in REPS:
    seed=20264000+rep*1009
    dat=b["generate_corpus"](16,4,2,3.5,4000,seed);X=b["flatten_decisions"](dat["routes"])
    DATA[rep]=(seed,X[:3200],X[3200:],dat["z"][:3200],dat["z"][3200:])

def task(arg):
    rep,j=arg;torch.set_num_threads(1);seed,Xtr,Xte,ztr,zte=DATA[rep]
    f=fit_dense_then_sparse(Xtr,16,4,2,seed+150000+j*8191,torch.device("cpu"))
    ptr=f["gamma"].argmax(1);tr=float(adjusted_mutual_info_score(ztr,ptr))
    with torch.no_grad():
        E=b["emission_loglik"](Xte,torch.tensor(f["U"],dtype=torch.float32),
           torch.tensor(f["V"],dtype=torch.float32),torch.device("cpu")).numpy().astype(np.float64)
    llte,g,_=b["fb_numpy"](E,f["A"],f["pi"]);pte=g.argmax(1);te=float(adjusted_mutual_info_score(zte,pte))
    return {"replicate":rep,"restart":j,"train_ll":f["ll"],"test_ll":float(llte),"train_ami":tr,"test_ami":te,"seconds":f["seconds"]}

if __name__=="__main__":
    jobs=[(r,j) for r in REPS for j in range(NR)];allr=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(28,os.cpu_count() or 8)) as ex:
        futs=[ex.submit(task,x) for x in jobs]
        for fut in concurrent.futures.as_completed(futs):
            z=fut.result();allr.append(z);print("V4_ITEM_JSON="+json.dumps(z,separators=(",",":")),flush=True)
    for rep in REPS:
        rr=[x for x in allr if x["replicate"]==rep]
        best=max(rr,key=lambda x:x["train_ll"]);diag=max(rr,key=lambda x:x["test_ami"])
        z={"replicate":rep,"selected_by_train_ll":best,"max_test_ami_diagnostic":diag["test_ami"],"n_ge70":sum(x["test_ami"]>=.70 for x in rr)}
        print("V4_REP_JSON="+json.dumps(z,separators=(",",":")),flush=True)
