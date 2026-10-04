#!/usr/bin/env python3
# Phase G: exact-objective population search with inner holdout.
# Synthetic-only recoverability gate. NO P70.
import json,math,os,urllib.request
import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor,as_completed

PURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/5d46189f00e3a3fa4db3eeb9a1331b48890c8ae7/research/inverse_renderer_piece_oracle_diagnostic_20261004.py"
p={"__name__":"piece"};exec(compile(urllib.request.urlopen(PURL,timeout=60).read().decode(),PURL,"exec"),p)
p["BASE_MIX"]=.02;p["BIAS_BOUND"]=5.0

DURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/dd2c107f3335ed3d022760f67bd0b95885898ba8/research/inverse_renderer_piece_recovery_phaseD_20261004.py"
d={"__name__":"phaseD"};exec(compile(urllib.request.urlopen(DURL,timeout=60).read().decode(),DURL,"exec"),d)

EURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/e188d1fc665e6676c84d79797dabcba709201cfd/research/inverse_renderer_collapsed_partition_phaseE_20261004.py"
e={"__name__":"phaseE"};exec(compile(urllib.request.urlopen(EURL,timeout=60).read().decode(),EURL,"exec"),e)

K=16;DEG=4;RANK=2;STRENGTH=5.0;SEED=20261004;N=4000
NFIT=2700;NVAL=500
A0,pi0,U0,V0,z,routes=p["generate"](K,DEG,RANK,STRENGTH,N,SEED)
X=p["flatten"](routes)
Xfit=X[:NFIT];Xval=X[NFIT:NFIT+NVAL];Xtest=X[NFIT+NVAL:]
zfit=z[:NFIT];zval=z[NFIT:NFIT+NVAL];ztest=z[NFIT+NVAL:]
P0=p["P0"];SUP=p["SUPPORT"];START=p["START"];END=p["END"]
C=p["NCTX"];O=p["NOPT"]

# Collapsed-search arrays on fit only. Source controls START and non-END continuations.
maxd=max(len(r) for r in routes[:NFIT])
ctx=np.full((NFIT,maxd),-1,np.int16);opt=np.full((NFIT,maxd),-1,np.int16);nd=np.zeros(NFIT,np.int16)
for t,r in enumerate(routes[:NFIT]):
    arr=[(START,r[0])]+[(r[i],r[i+1]) for i in range(len(r)-1)]
    nd[t]=len(arr)
    for j,(cc,oo) in enumerate(arr):ctx[t,j]=cc;opt[t,j]=oo

PB=np.zeros((C,O),np.float64)
for cc in range(C):
    allowed=SUP[cc].copy();allowed[END]=False
    if allowed.any():
        vv=P0[cc,allowed];PB[cc,allowed]=vv/max(vv.sum(),1e-30)

def white_init(seed):
    ctxn=Xfit.sum(2,keepdims=True);R0=Xfit-ctxn*P0[None,:,:]
    W=R0/np.sqrt(P0[None,:,:]+.01);F=W.reshape(NFIT,-1)
    Z=TruncatedSVD(n_components=16,random_state=seed).fit_transform(F)
    Z=StandardScaler().fit_transform(Z)
    return KMeans(K,n_init=8,random_state=seed,max_iter=300).fit_predict(Z).astype(np.int16)

def viterbi(E,A,pi):
    nn,kk=E.shape;la=np.log(np.maximum(A,1e-30));lp=np.log(np.maximum(pi,1e-30))
    dp=np.empty((nn,kk));bp=np.empty((nn,kk),np.int16);dp[0]=lp+E[0]
    for t in range(1,nn):
        M=dp[t-1][:,None]+la;bp[t]=np.argmax(M,0);dp[t]=M[bp[t],np.arange(kk)]+E[t]
    zz=np.empty(nn,np.int16);zz[-1]=np.argmax(dp[-1])
    for t in range(nn-2,-1,-1):zz[t]=bp[t+1,zz[t+1]]
    return zz,float(dp[-1].max())

def fit_exact(labels,Uprev=None,Vprev=None,steps=100):
    TC=np.zeros((K,K),float)
    for a,b in zip(labels[:-1],labels[1:]):TC[int(a),int(b)]+=1
    A=d["sparse_A"](TC,DEG)
    pi=np.bincount(labels[:200],minlength=K).astype(float)+1;pi/=pi.sum()
    F=csr_matrix(Xfit.reshape(NFIT,-1))
    if Uprev is None:
        Q=d["q_project_from_counts"](d["hard_counts"](F,labels,K),prior=4.0)
        U0,V0=d["lowrank"](d["bias_from_Q"](Q),RANK)
    else:U0,V0=Uprev,Vprev
    U=torch.nn.Parameter(torch.tensor(U0,dtype=torch.float32))
    V=torch.nn.Parameter(torch.tensor(V0,dtype=torch.float32))
    optm=torch.optim.Adam([U,V],lr=.025);idx=torch.arange(NFIT);zt=torch.tensor(labels,dtype=torch.long)
    for _ in range(steps):
        optm.zero_grad(set_to_none=True);Em=d["lowrank_E_torch"](Xfit,U,V,torch.device("cpu"))
        loss=-Em[idx,zt].mean()+1e-3*(U.square().mean()+V.square().mean())
        loss.backward();torch.nn.utils.clip_grad_norm_([U,V],5.0);optm.step()
    return A,pi,U.detach().numpy(),V.detach().numpy()

def score_block(Xb,A,pi,U,V):
    with torch.no_grad():
        E=d["lowrank_E_torch"](Xb,torch.tensor(U,dtype=torch.float32),torch.tensor(V,dtype=torch.float32),torch.device("cpu")).numpy().astype(float)
    ll,g,_=d["fb"](E,A,pi)
    return float(ll),g.argmax(1),E

def particle(pid):
    torch.set_num_threads(1)
    rng=np.random.default_rng(SEED+pid*1009)
    if pid<8:
        L0=white_init(SEED+pid)
        frac=(0.10,0.25,0.45,0.70)[pid%4]
        ix=rng.choice(NFIT,int(frac*NFIT),replace=False);L0[ix]=rng.integers(0,K,len(ix))
    else:
        L0=rng.integers(0,K,NFIT,dtype=np.int16)
    L,collapsed=e["chain"](L0,ctx,opt,nd,PB,5.0,.10,120,SEED+5000+pid,K,C,O)
    U=V=None;hist=[]
    for cyc in range(4):
        A,pi,U,V=fit_exact(L,U,V,120 if cyc==0 else 70)
        llfit,pfit,Efit=score_block(Xfit,A,pi,U,V)
        llval,pval,_=score_block(Xval,A,pi,U,V)
        Lnew,_=viterbi(Efit,A,pi)
        hist.append({"cycle":cyc,"fit_ll":llfit,"val_ll":llval,
                     "fit_nmi_diag":d["nmi"](zfit,pfit),"val_nmi_diag":d["nmi"](zval,pval),
                     "label_nmi_diag":d["nmi"](zfit,L)})
        L=Lnew
    # particle selection point = cycle with best inner validation likelihood.
    bestcy=max(range(len(hist)),key=lambda j:hist[j]["val_ll"])
    # Refit from current best labels is not reconstructable from history; final current state retained.
    # Selection across particles uses final val likelihood after 4 cycles.
    llval,pval,_=score_block(Xval,A,pi,U,V)
    return {"pid":pid,"collapsed_score":float(collapsed),"history":hist,
            "final_val_ll":llval,"final_fit_nmi_diag":d["nmi"](zfit,L),
            "final_val_nmi_diag":d["nmi"](zval,pval),
            "A":A.tolist(),"pi":pi.tolist(),"U":U.tolist(),"V":V.tolist()}

if __name__=="__main__":
    # Warm numba JIT before forking.
    warm=np.zeros(NFIT,np.int16)
    e["chain"](warm,ctx,opt,nd,PB,5.0,.10,1,SEED,K,C,O)
    parts=[]
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(particle,i):i for i in range(24)}
        for fut in as_completed(futs):
            rec=fut.result();parts.append(rec)
            slim={k:v for k,v in rec.items() if k not in ("A","pi","U","V")}
            print("POP_PARTICLE_JSON="+json.dumps(slim,separators=(",",":")),flush=True)
    parts.sort(key=lambda r:r["final_val_ll"],reverse=True)
    # IMPORTANT: only now open outer test for top 3 by inner val likelihood.
    finalists=[]
    for r in parts[:3]:
        A=np.array(r["A"]);pi=np.array(r["pi"]);U=np.array(r["U"]);V=np.array(r["V"])
        llt,pt,_=score_block(Xtest,A,pi,U,V)
        q={"pid":r["pid"],"val_ll":r["final_val_ll"],"outer_test_ll":llt,
           "outer_test_nmi_diag":d["nmi"](ztest,pt),"outer_test_ari_diag":d["ari"](ztest,pt),
           "inner_val_nmi_diag":r["final_val_nmi_diag"]}
        finalists.append(q);print("POP_FINALIST_JSON="+json.dumps(q,separators=(",",":")),flush=True)
    # Diagnostic correlation uses planted labels only after selection logic defined.
    vals=np.array([r["final_val_ll"] for r in parts]);nmis=np.array([r["final_val_nmi_diag"] for r in parts])
    corr=float(np.corrcoef(vals,nmis)[0,1])
    out={"population":24,"fit_n":NFIT,"val_n":NVAL,"test_n":len(Xtest),
         "best_by_val":finalists,"val_ll_vs_val_nmi_corr_diag":corr,
         "top10_selection":[{"pid":r["pid"],"val_ll":r["final_val_ll"],"val_nmi_diag":r["final_val_nmi_diag"]} for r in parts[:10]],
         "oracle_outer_nmi_diag":float(d["nmi"](ztest,d["fb"](p["emission"](Xtest,U0,V0),A0,pi0)[1].argmax(1)))}
    print("INVERSE_PHASEG_JSON="+json.dumps(out,separators=(",",":")),flush=True)
