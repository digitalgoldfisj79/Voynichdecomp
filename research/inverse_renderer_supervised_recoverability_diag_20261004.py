#!/usr/bin/env python3
import json,urllib.request,numpy as np,torch
from scipy.sparse import csr_matrix

URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/dd2c107f3335ed3d022760f67bd0b95885898ba8/research/inverse_renderer_piece_recovery_phaseD_20261004.py"
m={"__name__":"phaseD"};exec(compile(urllib.request.urlopen(URL,timeout=60).read().decode(),URL,"exec"),m)
p=m["p"];p["BASE_MIX"]=.02;p["BIAS_BOUND"]=5.0
K=16;d=4;rank=2;N=4000;seed=20261004
A0,pi0,U0,V0,z,routes=p["generate"](K,d,rank,5.0,N,seed)
X=p["flatten"](routes);ntr=int(.8*N);Xtr,Xte=X[:ntr],X[ntr:];ztr,zte=z[:ntr],z[ntr:]
Ftr=csr_matrix(Xtr.reshape(len(Xtr),-1));Fte=csr_matrix(Xte.reshape(len(Xte),-1))

# Estimate sparse transitions from planted training labels only.
C=np.zeros((K,K),float)
for a,b in zip(ztr[:-1],ztr[1:]):C[int(a),int(b)]+=1
A=m["sparse_A"](C,d);pi=np.bincount(ztr[:200],minlength=K).astype(float)+1;pi/=pi.sum()

# Dense state emissions from planted training labels.
Q=m["q_project_from_counts"](m["hard_counts"](Ftr,ztr,K),prior=4.0)
E=m["dense_E"](Fte,Q);lld,gd,_=m["fb"](E,A,pi);pd=gd.argmax(1)
dense={"ll":float(lld),"test_nmi":m["nmi"](zte,pd),"test_ari":m["ari"](zte,pd)}

# SVD rank-2 compression without optimization.
B=m["bias_from_Q"](Q);Ui,Vi=m["lowrank"](B,rank)
device=torch.device("cpu")
with torch.no_grad():
    E0=m["lowrank_E_torch"](Xte,torch.tensor(Ui,dtype=torch.float32),torch.tensor(Vi,dtype=torch.float32),device).numpy().astype(float)
ll0,g0,_=m["fb"](E0,A,pi);p0=g0.argmax(1)
svd={"ll":float(ll0),"test_nmi":m["nmi"](zte,p0),"test_ari":m["ari"](zte,p0)}

# Supervised low-rank fit on planted TRAIN labels, then unseen HMM recovery.
U=torch.nn.Parameter(torch.tensor(Ui,dtype=torch.float32));V=torch.nn.Parameter(torch.tensor(Vi,dtype=torch.float32))
opt=torch.optim.Adam([U,V],lr=.03);idx=torch.arange(len(Xtr));zt=torch.tensor(ztr,dtype=torch.long)
for step in range(400):
    opt.zero_grad(set_to_none=True);Et=m["lowrank_E_torch"](Xtr,U,V,device)
    loss=-Et[idx,zt].mean()+1e-3*(U.square().mean()+V.square().mean())
    loss.backward();torch.nn.utils.clip_grad_norm_([U,V],5.0);opt.step()
with torch.no_grad():Etst=m["lowrank_E_torch"](Xte,U,V,device).numpy().astype(float)
lls,gs,_=m["fb"](Etst,A,pi);ps=gs.argmax(1)
sup={"ll":float(lls),"test_nmi":m["nmi"](zte,ps),"test_ari":m["ari"](zte,ps),"steps":400}

# True-parameter oracle on same test subset.
with torch.no_grad():
    Etrue=p["emission"](Xte,U0,V0)
llt,gt,_=m["fb"](Etrue,A0,pi0);pt=gt.argmax(1)
oracle={"ll":float(llt),"test_nmi":m["nmi"](zte,pt),"test_ari":m["ari"](zte,pt)}
print("SUPERVISED_RECOVERABILITY_JSON="+json.dumps({"oracle":oracle,"learned_dense":dense,"svd_rank2":svd,"supervised_rank2":sup},separators=(",",":")),flush=True)
