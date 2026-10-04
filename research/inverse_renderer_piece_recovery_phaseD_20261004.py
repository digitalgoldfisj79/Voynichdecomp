#!/usr/bin/env python3
# Phase D: exact-piece planted-source recovery with dense-emission teacher -> low-rank compression.
# NO P70.
import argparse,json,math,time,urllib.request
import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PIECE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/5d46189f00e3a3fa4db3eeb9a1331b48890c8ae7/research/inverse_renderer_piece_oracle_diagnostic_20261004.py"
p={"__name__":"piece"}
exec(compile(urllib.request.urlopen(PIECE_URL,timeout=60).read().decode(),PIECE_URL,"exec"),p)
# Frozen recoverable synthetic regime.
p["BASE_MIX"]=.02
p["BIAS_BOUND"]=5.0
BASE_MIX=.02;BIAS_BOUND=5.0
P0=p["P0"];SUPPORT=p["SUPPORT"];LOGP0=p["LOGP0"]
NCTX=p["NCTX"];NOPT=p["NOPT"];START=p["START"];END=p["END"]
fb=p["m"]["fb_numpy"]; sparse_A=p["m"]["project_sparse_A"]
nmi=p["nmi"];ari=p["ari"]

def q_project_from_counts(C,prior=8.0):
    # C K,C,O -> valid dense state-conditioned Q with exact frozen END hazard and renderer floor.
    K=C.shape[0];Q=np.zeros_like(C,dtype=float)
    for z in range(K):
        for c in range(NCTX):
            ok=SUPPORT[c]
            raw=C[z,c]+prior*P0[c]
            if c==START:
                qb=np.zeros(NOPT);qb[ok]=raw[ok]/max(raw[ok].sum(),1e-30)
            else:
                pend=P0[c,END] if SUPPORT[c,END] else 0.
                non=ok.copy();non[END]=False
                qb=np.zeros(NOPT);qb[END]=pend
                if non.any():
                    v=raw[non];v=v/max(v.sum(),1e-30);qb[non]=(1-pend)*v
            q=BASE_MIX*P0[c]+(1-BASE_MIX)*qb
            q[~ok]=0;q/=q.sum();Q[z,c]=q
    return Q

def dense_E(F,Q):
    L=np.log(np.maximum(Q.reshape(Q.shape[0],-1),1e-30))
    return np.asarray(F @ L.T,dtype=float)

def labels_to_A(labels,K,smooth=.25):
    A=np.full((K,K),smooth,float)
    for a,b in zip(labels[:-1],labels[1:]):A[int(a),int(b)]+=1
    A/=A.sum(1,keepdims=True)
    pi=np.bincount(labels[:min(200,len(labels))],minlength=K).astype(float)+1;pi/=pi.sum()
    return A,pi

def hard_counts(F,labels,K):
    H=np.zeros((K,len(labels)),float);H[labels,np.arange(len(labels))]=1
    return np.asarray(H @ F).reshape(K,NCTX,NOPT)

def teacher_fit(X,K,d,seed,dense_epochs=35,sparse_epochs=15,prior=8.0):
    t0=time.time();F=csr_matrix(X.reshape(len(X),-1))
    # Spectral observable initializer.
    nc=min(32,max(4,K*2),F.shape[1]-1)
    Z=TruncatedSVD(n_components=nc,random_state=seed).fit_transform(F)
    Z=StandardScaler().fit_transform(Z)
    labels=KMeans(n_clusters=K,n_init=20,random_state=seed,max_iter=400).fit_predict(Z)
    A,pi=labels_to_A(labels,K)
    Q=q_project_from_counts(hard_counts(F,labels,K),prior)
    best=None;last=-1e99
    for stage,ne in (("dense",dense_epochs),("sparse",sparse_epochs)):
        for it in range(ne):
            E=dense_E(F,Q);ll,g,xi=fb(E,A,pi)
            A=(xi+.10);A/=A.sum(1,keepdims=True)
            if stage=="sparse":A=sparse_A(xi,d)
            pi=g[0]+.1;pi/=pi.sum()
            C=np.asarray(g.T @ F).reshape(K,NCTX,NOPT)
            Q=q_project_from_counts(C,prior)
            E2=dense_E(F,Q);ll2,g2,xi2=fb(E2,A,pi)
            if best is None or ll2>best[0]:best=(ll2,A.copy(),pi.copy(),Q.copy(),g2.copy())
            if abs(ll2-last)<1e-4 and it>6:break
            last=ll2
        if stage=="dense":
            ll,A,pi,Q,g=best
            E=dense_E(F,Q);_,_,xi=fb(E,A,pi);A=sparse_A(xi,d);last=-1e99
    ll,A,pi,Q,g=best
    return {"ll":float(ll),"A":A,"pi":pi,"Q":Q,"gamma":g,"seconds":time.time()-t0}

def bias_from_Q(Q):
    K=Q.shape[0];B=np.zeros_like(Q)
    for z in range(K):
        for c in range(NCTX):
            ok=SUPPORT[c]
            # remove frozen renderer mixture
            qb=(Q[z,c]-BASE_MIX*P0[c])/(1-BASE_MIX)
            qb=np.maximum(qb,1e-12);qb[~ok]=0
            if c==START:
                qb[ok]/=qb[ok].sum()
                b=np.zeros(NOPT);b[ok]=np.log(qb[ok])-np.log(np.maximum(P0[c,ok],1e-30));b[ok]-=b[ok].mean()
            else:
                non=ok.copy();non[END]=False;b=np.zeros(NOPT)
                if non.any():
                    x=qb[non];x/=x.sum()
                    y=P0[c,non];y/=y.sum()
                    b[non]=np.log(np.maximum(x,1e-30))-np.log(np.maximum(y,1e-30));b[non]-=b[non].mean()
            B[z,c]=np.clip(b,-BIAS_BOUND,BIAS_BOUND)
    B-=B.mean(0,keepdims=True)
    return B

def lowrank(B,rank):
    K=B.shape[0];M=B.reshape(K,-1);u,s,vt=np.linalg.svd(M,full_matrices=False)
    U=u[:,:rank]*np.sqrt(s[:rank])[None,:]
    V=(np.sqrt(s[:rank])[:,None]*vt[:rank]).reshape(rank,NCTX,NOPT)
    return U,V

def lowrank_E_torch(X,U,V,device):
    raw=torch.einsum("kr,rco->kco",U,V);bias=BIAS_BOUND*torch.tanh(raw/BIAS_BOUND)
    p0=torch.tensor(P0,dtype=torch.float32,device=device);mask=torch.tensor(SUPPORT,dtype=torch.bool,device=device)
    qs=[]
    for c in range(NCTX):
        if c==START:
            lg=torch.tensor(LOGP0[c],dtype=torch.float32,device=device).unsqueeze(0)+bias[:,c,:]
            lg=torch.where(mask[c].unsqueeze(0),lg,torch.tensor(-1e30,device=device));qb=torch.softmax(lg,1)
        else:
            pend=p0[c,END] if SUPPORT[c,END] else torch.tensor(0.,device=device)
            non=mask[c].clone();non[END]=False
            lg=torch.tensor(LOGP0[c],dtype=torch.float32,device=device).unsqueeze(0)+bias[:,c,:]
            lg=torch.where(non.unsqueeze(0),lg,torch.tensor(-1e30,device=device));qn=torch.softmax(lg,1)
            qb=(1-pend)*qn
            if SUPPORT[c,END]:qb[:,END]=pend
        q=BASE_MIX*p0[c].unsqueeze(0)+(1-BASE_MIX)*qb
        q=torch.where(mask[c].unsqueeze(0),q,torch.tensor(0.,device=device));q/=q.sum(1,keepdim=True);qs.append(q)
    logq=torch.log(torch.clamp(torch.stack(qs,1),min=1e-30))
    return torch.einsum("nco,kco->nk",torch.tensor(X,dtype=torch.float32,device=device),logq)

def lowrank_refine(X,A,pi,Q,rank,d,device,steps_outer=35,msteps=35,lr=.025):
    U0,V0=lowrank(bias_from_Q(Q),rank)
    U=torch.nn.Parameter(torch.tensor(U0,dtype=torch.float32,device=device))
    V=torch.nn.Parameter(torch.tensor(V0,dtype=torch.float32,device=device))
    opt=torch.optim.Adam([U,V],lr=lr);best=None;last=-1e99
    for outer in range(steps_outer):
        with torch.no_grad():E=lowrank_E_torch(X,U,V,device).cpu().numpy().astype(float)
        ll,g,xi=fb(E,A,pi);A=sparse_A(xi,d);pi=g[0]+.1;pi/=pi.sum()
        G=torch.tensor(g,dtype=torch.float32,device=device)
        for _ in range(msteps):
            opt.zero_grad(set_to_none=True);Et=lowrank_E_torch(X,U,V,device)
            loss=-(G*Et).sum()/len(X)+1e-3*(U.square().mean()+V.square().mean())
            loss.backward();torch.nn.utils.clip_grad_norm_([U,V],5.0);opt.step()
        with torch.no_grad():E2=lowrank_E_torch(X,U,V,device).cpu().numpy().astype(float)
        ll2,g2,xi2=fb(E2,A,pi)
        if best is None or ll2>best[0]:best=(ll2,A.copy(),pi.copy(),U.detach().cpu().numpy().copy(),V.detach().cpu().numpy().copy(),g2.copy())
        if abs(ll2-last)<1e-4 and outer>8:break
        last=ll2
    ll,A,pi,U,V,g=best
    return {"ll":float(ll),"A":A,"pi":pi,"U":U,"V":V,"gamma":g}

def oracle(dat,X):
    E=p["emission"](X,dat["U"],dat["V"]);ll,g,_=fb(E,dat["A"],dat["pi"]);pred=g.argmax(1)
    return {"ll":float(ll),"nmi":nmi(dat["z"],pred),"ari":ari(dat["z"],pred)}

def run(args):
    device=torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")
    # p.generate returns tuple A,pi,U,V,z,routes
    A0,pi0,U0,V0,z,routes=p["generate"](args.K,args.d,args.rank,args.strength,args.N,args.seed)
    dat={"A":A0,"pi":pi0,"U":U0,"V":V0,"z":z};X=p["flatten"](routes)
    orc=oracle(dat,X);ntr=int(.8*args.N);Xtr,Xte=X[:ntr],X[ntr:];ztr,zte=z[:ntr],z[ntr:]
    results=[]
    for r in range(args.restarts):
        teach=teacher_fit(Xtr,args.K,args.d,args.seed+r*101,args.dense_epochs,args.sparse_epochs,args.prior)
        tp=teach["gamma"].argmax(1)
        lr=lowrank_refine(Xtr,teach["A"],teach["pi"],teach["Q"],args.rank,args.d,device,args.outer,args.msteps,args.lr)
        lp=lr["gamma"].argmax(1)
        with torch.no_grad():
            U=torch.tensor(lr["U"],dtype=torch.float32,device=device);V=torch.tensor(lr["V"],dtype=torch.float32,device=device)
            Ete=lowrank_E_torch(Xte,U,V,device).cpu().numpy().astype(float)
        llte,gte,_=fb(Ete,lr["A"],lr["pi"]);pt=gte.argmax(1)
        rec={"restart":r,"teacher_ll":teach["ll"],"teacher_train_nmi":nmi(ztr,tp),
             "lowrank_train_ll":lr["ll"],"lowrank_train_nmi":nmi(ztr,lp),
             "test_ll":float(llte),"test_nmi":nmi(zte,pt),"test_ari":ari(zte,pt),
             "teacher_seconds":teach["seconds"]}
        print("PHASED_RESTART_JSON="+json.dumps(rec,separators=(",",":")),flush=True);results.append(rec)
    out={"phase":"D_piece_teacher_lowrank","K":args.K,"d":args.d,"rank":args.rank,"strength":args.strength,
         "mix":BASE_MIX,"bound":BIAS_BOUND,"oracle":orc,"restarts":results,
         "median_test_nmi":float(np.median([x["test_nmi"] for x in results])),
         "best_test_nmi":float(max(x["test_nmi"] for x in results)),
         "fraction_nmi70":float(np.mean([x["test_nmi"]>=.70 for x in results]))}
    print("INVERSE_PHASED_JSON="+json.dumps(out,separators=(",",":")),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--K",type=int,default=16);ap.add_argument("--d",type=int,default=4)
    ap.add_argument("--rank",type=int,default=2);ap.add_argument("--strength",type=float,default=5.0);ap.add_argument("--N",type=int,default=4000)
    ap.add_argument("--seed",type=int,default=20261004);ap.add_argument("--restarts",type=int,default=3)
    ap.add_argument("--dense-epochs",type=int,default=35);ap.add_argument("--sparse-epochs",type=int,default=15);ap.add_argument("--prior",type=float,default=8.)
    ap.add_argument("--outer",type=int,default=35);ap.add_argument("--msteps",type=int,default=35);ap.add_argument("--lr",type=float,default=.025)
    ap.add_argument("--device",choices=["cpu","cuda"],default="cpu");run(ap.parse_args())
