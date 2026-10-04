#!/usr/bin/env python3
import json,urllib.request
import numpy as np, torch
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/83d96e02bf2149946bf849697e22217a410f42e4/research/inverse_renderer_recoverability_phaseA_20261004.py"
m={"__name__":"phaseA"};exec(compile(urllib.request.urlopen(URL,timeout=60).read().decode(),URL,"exec"),m)
device=torch.device("cpu")
for strength in (1.75,2.5):
    dat=m["generate_corpus"](16,4,2,strength,4000,20261004)
    X=m["flatten_decisions"](dat["routes"]);z=dat["z"];K=16;r=2
    # supervised emission M-step from true labels
    rng=np.random.default_rng(771+int(strength*100))
    U=torch.nn.Parameter(torch.tensor(rng.normal(0,.08,(K,r)),dtype=torch.float32))
    V=torch.nn.Parameter(torch.tensor(rng.normal(0,.08,(r,m["NCTX"],m["NOPT"])),dtype=torch.float32))
    G=torch.zeros((len(z),K),dtype=torch.float32);G[torch.arange(len(z)),torch.tensor(z)]=1.
    opt=torch.optim.Adam([U,V],lr=.05)
    for step in range(600):
        opt.zero_grad(set_to_none=True);E=m["emission_loglik"](X,U,V,device)
        loss=-(G*E).sum()/len(X)+1e-4*(U.square().mean()+V.square().mean())
        loss.backward();opt.step()
    with torch.no_grad():
        Elearn=m["emission_loglik"](X,U,V,device).numpy().astype(np.float64)
        Etrue=m["emission_loglik"](X,torch.tensor(dat["U"],dtype=torch.float32),
                                    torch.tensor(dat["V"],dtype=torch.float32),device).numpy().astype(np.float64)
    # A learned from true state transitions, topology projected exactly blind to weights
    xi=np.zeros((K,K))
    for a,b in zip(z[:-1],z[1:]):xi[a,b]+=1
    Alearn=m["project_sparse_A"](xi,4); pi=np.bincount(z[:1],minlength=K)+.1;pi=pi/pi.sum()
    ll,g,_=m["fb_numpy"](Elearn,Alearn,pi);p=g.argmax(1)
    llT,gT,_=m["fb_numpy"](Etrue,dat["A"],dat["pi"]);pT=gT.argmax(1)
    # true emission + learned/random sparse A refinement
    A0,pi0,_=m["source_graph"](K,4,np.random.default_rng(999))
    A=A0
    for it in range(20):
        llx,gx,xx=m["fb_numpy"](Etrue,A,pi0);A=m["project_sparse_A"](xx,4);pi0=gx[0]+.1;pi0/=pi0.sum()
    px=gx.argmax(1)
    out={
      "strength":strength,
      "supervised_emission_trueA":{"nmi":m["nmi"](z,p),"ari":m["ari"](z,p),"ll":ll},
      "oracle":{"nmi":m["nmi"](z,pT),"ari":m["ari"](z,pT),"ll":llT},
      "true_emission_blindA":{"nmi":m["nmi"](z,px),"ari":m["ari"](z,px),"ll":llx},
      "mean_true_state_logem_learned":float(Elearn[np.arange(len(z)),z].mean()),
      "mean_true_state_logem_oracle":float(Etrue[np.arange(len(z)),z].mean())
    }
    print("PHASEA_COMPONENT_DEBUG_JSON="+json.dumps(out,separators=(",",":")),flush=True)
