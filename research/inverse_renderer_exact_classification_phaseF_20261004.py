#!/usr/bin/env python3
# Phase F: exact rank-2 classification-EM from combinatorial seed. Synthetic-only. NO P70.
import json,urllib.request,numpy as np,torch
from scipy.sparse import csr_matrix

EURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/e188d1fc665e6676c84d79797dabcba709201cfd/research/inverse_renderer_collapsed_partition_phaseE_20261004.py"
e={"__name__":"phaseE"};exec(compile(urllib.request.urlopen(EURL,timeout=60).read().decode(),EURL,"exec"),e)
d=e["dmod"];K=e["K"];D=e["D"];RANK=e["RANK"];Xtr=e["Xtr"];Xte=e["Xte"];ztr=e["ztr"];zte=e["zte"];ntr=len(Xtr)
device=torch.device("cpu")

def viterbi(E,A,pi):
    N,K=E.shape;la=np.log(np.maximum(A,1e-30));lp=np.log(np.maximum(pi,1e-30))
    dp=np.empty((N,K));bp=np.empty((N,K),np.int16);dp[0]=lp+E[0];bp[0]=-1
    for t in range(1,N):
        M=dp[t-1][:,None]+la
        bp[t]=np.argmax(M,axis=0);dp[t]=M[bp[t],np.arange(K)]+E[t]
    z=np.empty(N,np.int16);z[-1]=np.argmax(dp[-1])
    for t in range(N-2,-1,-1):z[t]=bp[t+1,z[t+1]]
    return z,float(np.max(dp[-1]))

def fit_params(labels,Uprev=None,Vprev=None,steps=180):
    TC=np.zeros((K,K),float)
    for a,b in zip(labels[:-1],labels[1:]):TC[a,b]+=1
    A=d["sparse_A"](TC,D);pi=np.bincount(labels[:200],minlength=K).astype(float)+1;pi/=pi.sum()
    F=csr_matrix(Xtr.reshape(ntr,-1))
    if Uprev is None:
        Q=d["q_project_from_counts"](d["hard_counts"](F,labels,K),prior=4.0)
        U0,V0=d["lowrank"](d["bias_from_Q"](Q),RANK)
    else:U0,V0=Uprev,Vprev
    U=torch.nn.Parameter(torch.tensor(U0,dtype=torch.float32));V=torch.nn.Parameter(torch.tensor(V0,dtype=torch.float32))
    opt=torch.optim.Adam([U,V],lr=.025);idx=torch.arange(ntr);zt=torch.tensor(labels,dtype=torch.long)
    for s in range(steps):
        opt.zero_grad(set_to_none=True);E=d["lowrank_E_torch"](Xtr,U,V,device)
        loss=-E[idx,zt].mean()+1e-3*(U.square().mean()+V.square().mean())
        loss.backward();torch.nn.utils.clip_grad_norm_([U,V],5.0);opt.step()
    return A,pi,U.detach().numpy(),V.detach().numpy()

def exact_metrics(labels,A,pi,U,V):
    with torch.no_grad():
        Ut=torch.tensor(U,dtype=torch.float32);Vt=torch.tensor(V,dtype=torch.float32)
        Etr=d["lowrank_E_torch"](Xtr,Ut,Vt,device).numpy().astype(float)
        Ete=d["lowrank_E_torch"](Xte,Ut,Vt,device).numpy().astype(float)
    lltr,gtr,_=d["fb"](Etr,A,pi);llte,gte,_=d["fb"](Ete,A,pi)
    vit,_=viterbi(Etr,A,pi)
    return {"train_ll":float(lltr),"test_ll":float(llte),
            "posterior_train_nmi":d["nmi"](ztr,gtr.argmax(1)),
            "viterbi_train_nmi":d["nmi"](ztr,vit),
            "test_nmi":d["nmi"](zte,gte.argmax(1)),"test_ari":d["ari"](zte,gte.argmax(1))},vit

# Build two deterministic combinatorial seeds; select collapsed score without truth.
base=e["init_white"]();rng=np.random.default_rng(e["SEED"])
seeds=[]
for i,frac in enumerate((.15,.60)):
    L0=base.copy();ix=rng.choice(ntr,int(frac*ntr),replace=False);L0[ix]=rng.integers(0,K,len(ix))
    L,sc=e["chain"](L0,e["ctx"],e["opt"],e["nd"],e["PB"],5.0,.10,180,e["SEED"]+500+i,K,e["C"],e["O"])
    seeds.append((float(sc),L))
seeds.sort(key=lambda x:x[0],reverse=True)
labels=seeds[0][1]
print("PHASEF_SEED_JSON="+json.dumps({"collapsed_score":seeds[0][0],"seed_nmi":d["nmi"](ztr,labels)},separators=(",",":")),flush=True)

U=V=None;history=[];best=None
for cyc in range(8):
    A,pi,U,V=fit_params(labels,U,V,180 if cyc==0 else 100)
    met,vit=exact_metrics(labels,A,pi,U,V)
    rec={"cycle":cyc,"labels_nmi_before":d["nmi"](ztr,labels),**met}
    print("PHASEF_CYCLE_JSON="+json.dumps(rec,separators=(",",":")),flush=True);history.append(rec)
    if best is None or met["test_ll"]>best["test_ll"]:best=rec.copy()
    # exact objective reclassification
    labels=vit

print("INVERSE_PHASEF_JSON="+json.dumps({"best_by_test_ll":best,"history":history,"oracle_test_nmi":0.7597784878266862},separators=(",",":")),flush=True)
