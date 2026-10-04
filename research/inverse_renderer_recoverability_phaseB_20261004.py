#!/usr/bin/env python3
# Phase B recoverability solver: termination-safe emissions + clustering/dense-init + annealed EM.
# NO P70.
import argparse,json,math,time,urllib.request
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/664cd58a01a0d26bf66619fafc92945bee1ffe75/research/inverse_renderer_recoverability_phaseA_20261004.py"
m={"__name__":"phaseA"}
exec(compile(urllib.request.urlopen(BASE_URL,timeout=60).read().decode(),BASE_URL,"exec"),m)

P0=m["P0"]; SUPPORT=m["SUPPORT"]; LOGP0=m["LOGP0"]; NCTX=m["NCTX"]; NOPT=m["NOPT"]; END=m["END"]
KFORM=m["KFORM"]; rows=m["rows"]
FLOOR_MIX=0.25  # frozen renderer share in every legal continuation
BIAS_BOUND=2.0   # maximum absolute source logit perturbation before renderer mixing

def q_state_numpy(Urow,V,ctx):
    # Exact generator family: bounded source bias + frozen renderer mixture.
    b=BIAS_BOUND*np.tanh((Urow@V[:,ctx,:])/BIAS_BOUND)
    ok=SUPPORT[ctx].copy()
    if ctx==0:
        logits=LOGP0[ctx].copy()+b
        logits[~ok]=-1e30
        mx=logits.max();qb=np.exp(logits-mx);qb/=qb.sum()
        q=FLOOR_MIX*P0[ctx]+(1-FLOOR_MIX)*qb
        q[~ok]=0;q/=q.sum()
        return q
    pend=float(P0[ctx,END]) if SUPPORT[ctx,END] else 0.0
    non=ok.copy();non[END]=False
    qb=np.zeros(NOPT,float);qb[END]=pend
    if non.any():
        logits=LOGP0[ctx].copy()+b
        logits[~non]=-1e30
        mx=logits[non].max();z=np.exp(logits[non]-mx);z/=z.sum()
        qb[non]=(1.0-pend)*z
    q=FLOOR_MIX*P0[ctx]+(1-FLOOR_MIX)*qb
    q[~ok]=0;q/=q.sum()
    return q

def sample_route_safe(z,U,V,rng,maxlen=2048):
    route=[];ctx=0
    for _ in range(maxlen):
        q=q_state_numpy(U[z],V,ctx)
        opt=int(rng.choice(NOPT,p=q))
        if opt==END:
            if route:return route
            continue
        route.append(opt);ctx=1+opt
    raise RuntimeError(("termination_floor_failed",z,maxlen,FLOOR_MIX))

def generate_corpus(K,d,rank,strength,N,seed):
    rng=np.random.default_rng(seed)
    A,pi,edges=m["source_graph"](K,d,rng)
    U,V=m["make_coupling"](K,rank,strength,rng)
    z=m["sample_hidden"](A,pi,N,rng)
    routes=[sample_route_safe(int(zz),U,V,rng) for zz in z]
    return {"A":A,"pi":pi,"edges":edges,"U":U,"V":V,"z":z,"routes":routes}

def emission_loglik_mix(X,U,V,device):
    # Exact same family as generator: frozen END hazard, source biases non-END choices only.
    raw=torch.einsum("kr,rco->kco",U,V)
    bias=BIAS_BOUND*torch.tanh(raw/BIAS_BOUND)
    p0=torch.tensor(P0,dtype=torch.float32,device=device)
    mask=torch.tensor(SUPPORT,dtype=torch.bool,device=device)
    K=U.shape[0]
    qs=[]
    for ctx in range(NCTX):
        if ctx==0:
            logits=torch.tensor(LOGP0[ctx],dtype=torch.float32,device=device).unsqueeze(0)+bias[:,ctx,:]
            logits=torch.where(mask[ctx].unsqueeze(0),logits,torch.tensor(-1e30,device=device))
            qs.append(torch.softmax(logits,dim=1))
        else:
            pend=p0[ctx,END] if SUPPORT[ctx,END] else torch.tensor(0.,device=device)
            non=mask[ctx].clone();non[END]=False
            logits=torch.tensor(LOGP0[ctx],dtype=torch.float32,device=device).unsqueeze(0)+bias[:,ctx,:]
            logits=torch.where(non.unsqueeze(0),logits,torch.tensor(-1e30,device=device))
            qn=torch.softmax(logits,dim=1)
            q=(1-pend)*qn
            if SUPPORT[ctx,END]:
                q[:,END]=pend
            qs.append(q)
    q=torch.stack(qs,dim=1)
    p0all=torch.tensor(P0,dtype=torch.float32,device=device).unsqueeze(0)
    q=FLOOR_MIX*p0all+(1-FLOOR_MIX)*q
    q=torch.where(torch.tensor(SUPPORT,dtype=torch.bool,device=device).unsqueeze(0),q,torch.tensor(0.,device=device))
    q=q/torch.clamp(q.sum(dim=2,keepdim=True),min=1e-30)
    logq=torch.log(torch.clamp(q,min=1e-30))
    Xt=torch.tensor(X,dtype=torch.float32,device=device)
    return torch.einsum("nco,kco->nk",Xt,logq)

def token_features(X):
    # Relative route features: counts + residual against expected baseline.
    N=len(X)
    total=X.sum((1,2),keepdims=True)
    raw=X.reshape(N,-1)
    # Binary path occurrence helps prevent long tokens dominating.
    binary=(X>0).astype(np.float32).reshape(N,-1)
    length=total.reshape(N,1)
    F=np.concatenate([raw,binary,length],axis=1)
    return StandardScaler().fit_transform(F)

def labels_to_dense_A(labels,K,smooth=.5):
    C=np.full((K,K),smooth,float)
    for a,b in zip(labels[:-1],labels[1:]):C[int(a),int(b)]+=1
    C/=C.sum(1,keepdims=True)
    pi=np.bincount(labels[:max(10,min(100,len(labels)))],minlength=K).astype(float)+1
    pi/=pi.sum()
    return C,pi

def cluster_bias_matrix(X,labels,K):
    # State-specific log ratio over each context, with observed support only.
    B=np.zeros((K,NCTX,NOPT),float)
    for z in range(K):
        ix=np.where(labels==z)[0]
        if not len(ix):continue
        C=X[ix].sum(0)+0.5*SUPPORT
        Q=np.divide(C,C.sum(1,keepdims=True),out=np.zeros_like(C),where=C.sum(1,keepdims=True)>0)
        for c in range(NCTX):
            ok=SUPPORT[c]
            lr=np.zeros(NOPT,float)
            lr[ok]=np.log(np.maximum(Q[c,ok],1e-12))-np.log(np.maximum(P0[c,ok],1e-12))
            lr[ok]-=lr[ok].mean()
            B[z,c]=lr
    B-=B.mean(0,keepdims=True)
    return B

def lowrank_from_B(B,rank):
    K=B.shape[0];M=B.reshape(K,-1)
    u,s,vt=np.linalg.svd(M,full_matrices=False)
    rr=min(rank,len(s))
    U=u[:,:rr]*np.sqrt(s[:rr])[None,:]
    V=(np.sqrt(s[:rr])[:,None]*vt[:rr]).reshape(rr,NCTX,NOPT)
    if rr<rank:
        U=np.pad(U,((0,0),(0,rank-rr)))
        V=np.pad(V,((0,rank-rr),(0,0),(0,0)))
    return U,V

def dense_A_from_xi(xi,smooth=.1):
    A=xi+smooth
    A/=A.sum(1,keepdims=True)
    return A

def entropy_rows(A):
    return float(np.mean(-np.sum(np.where(A>0,A*np.log2(np.maximum(A,1e-30)),0),axis=1)))

def fit_better(X,K,d,rank,seed,device,epochs_dense=45,epochs_sparse=20,msteps=25,lr=.04):
    rng=np.random.default_rng(seed)
    t0=time.time()
    # 1) deterministic-ish clustering init on observable route features.
    feats=token_features(X)
    km=KMeans(n_clusters=K,n_init=10,random_state=seed,max_iter=300)
    labels=km.fit_predict(feats)
    A,pi=labels_to_dense_A(labels,K)
    B0=cluster_bias_matrix(X,labels,K)
    U0,V0=lowrank_from_B(B0,rank)
    # Add tiny restart perturbation.
    U0+=rng.normal(0,.01,U0.shape);V0+=rng.normal(0,.01,V0.shape)
    U=torch.nn.Parameter(torch.tensor(U0,dtype=torch.float32,device=device))
    V=torch.nn.Parameter(torch.tensor(V0,dtype=torch.float32,device=device))
    opt=torch.optim.Adam([U,V],lr=lr)
    best=None;last=-1e99

    def one_stage(nepoch,sparse=False):
        nonlocal A,pi,best,last
        # deterministic annealing: hotter early, then T->1.
        for em in range(nepoch):
            frac=em/max(nepoch-1,1)
            T=1.35-(.35*frac)
            with torch.no_grad():
                E=emission_loglik_mix(X,U,V,device).cpu().numpy().astype(np.float64)/T
            ll,g,xi=m["fb_numpy"](E,A,pi)
            A=m["project_sparse_A"](xi,d) if sparse else dense_A_from_xi(xi,.1)
            pi=(g[0]+.1);pi/=pi.sum()
            G=torch.tensor(g,dtype=torch.float32,device=device)
            for _ in range(msteps):
                opt.zero_grad(set_to_none=True)
                E2=emission_loglik_mix(X,U,V,device)
                loss=-(G*E2).sum()/len(X)
                loss=loss+2e-3*(U.square().mean()+V.square().mean())+2e-2*U.mean(0).square().mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([U,V],5.0)
                opt.step()
            # assess at T=1
            with torch.no_grad():
                Etrue=emission_loglik_mix(X,U,V,device).cpu().numpy().astype(np.float64)
            lltrue,gtrue,xitrue=m["fb_numpy"](Etrue,A,pi)
            if best is None or lltrue>best[0]:
                best=(lltrue,A.copy(),pi.copy(),U.detach().cpu().numpy().copy(),
                      V.detach().cpu().numpy().copy(),gtrue.copy(),em,sparse)
            if abs(lltrue-last)<1e-3 and em>8:break
            last=lltrue

    one_stage(epochs_dense,False)
    # Restore best dense point, then sparsify topology by posterior flow and refine.
    ll,A,pi,Ux,Vx,g,em,_=best
    U.data.copy_(torch.tensor(Ux,dtype=torch.float32,device=device))
    V.data.copy_(torch.tensor(Vx,dtype=torch.float32,device=device))
    with torch.no_grad():
        E=emission_loglik_mix(X,U,V,device).cpu().numpy().astype(np.float64)
    _,_,xi=m["fb_numpy"](E,A,pi)
    A=m["project_sparse_A"](xi,d)
    last=-1e99
    one_stage(epochs_sparse,True)

    ll,A,pi,Ux,Vx,g,em,sparse=best
    return {"ll":float(ll),"A":A,"pi":pi,"U":Ux,"V":Vx,"gamma":g,
            "seconds":time.time()-t0,"row_entropy":entropy_rows(A)}

def oracle(dat,X,device):
    U=torch.tensor(dat["U"],dtype=torch.float32,device=device)
    V=torch.tensor(dat["V"],dtype=torch.float32,device=device)
    with torch.no_grad():E=emission_loglik_mix(X,U,V,device).cpu().numpy().astype(np.float64)
    ll,g,_=m["fb_numpy"](E,dat["A"],dat["pi"])
    p=g.argmax(1)
    return {"ll":float(ll),"nmi":m["nmi"](dat["z"],p),"ari":m["ari"](dat["z"],p)}

def run(args):
    device=torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")
    dat=generate_corpus(args.K,args.d,args.rank,args.strength,args.N,args.seed)
    X=m["flatten_decisions"](dat["routes"])
    orc=oracle(dat,X,device)
    ntr=int(.8*args.N)
    Xtr,Xte=X[:ntr],X[ntr:];ztr,zte=dat["z"][:ntr],dat["z"][ntr:]
    recs=[]
    for r in range(args.restarts):
        fit=fit_better(Xtr,args.K,args.d,args.rank,args.seed+1000+r*97,device,
                       args.dense_epochs,args.sparse_epochs,args.msteps,args.lr)
        p=fit["gamma"].argmax(1)
        with torch.no_grad():
            U=torch.tensor(fit["U"],dtype=torch.float32,device=device)
            V=torch.tensor(fit["V"],dtype=torch.float32,device=device)
            Ete=emission_loglik_mix(Xte,U,V,device).cpu().numpy().astype(np.float64)
        llte,gte,_=m["fb_numpy"](Ete,fit["A"],fit["pi"])
        pt=gte.argmax(1)
        rec={"restart":r,"train_ll":fit["ll"],"test_ll":float(llte),
             "train_nmi":m["nmi"](ztr,p),"test_nmi":m["nmi"](zte,pt),
             "train_ari":m["ari"](ztr,p),"test_ari":m["ari"](zte,pt),
             "seconds":fit["seconds"],"row_entropy":fit["row_entropy"]}
        print("PHASEB_RESTART_JSON="+json.dumps(rec,separators=(",",":")),flush=True);recs.append(rec)
    out={"phase":"B_solver_recovery","floor_mix":FLOOR_MIX,"bias_bound":BIAS_BOUND,"device":str(device),
         "gpu":torch.cuda.get_device_name(0) if device.type=="cuda" else None,
         "N":args.N,"K":args.K,"d":args.d,"rank":args.rank,"strength":args.strength,
         "oracle":orc,"restarts":recs,
         "median_test_nmi":float(np.median([x["test_nmi"] for x in recs])),
         "best_test_nmi":float(max(x["test_nmi"] for x in recs)),
         "fraction_nmi70":float(np.mean([x["test_nmi"]>=.70 for x in recs])),
         "corpus_rows":len(rows)}
    print("INVERSE_PHASEB_JSON="+json.dumps(out,separators=(",",":")),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--N",type=int,default=4000)
    ap.add_argument("--K",type=int,default=16);ap.add_argument("--d",type=int,default=4)
    ap.add_argument("--rank",type=int,default=2);ap.add_argument("--strength",type=float,default=1.75)
    ap.add_argument("--seed",type=int,default=20261004);ap.add_argument("--restarts",type=int,default=4)
    ap.add_argument("--dense-epochs",type=int,default=45);ap.add_argument("--sparse-epochs",type=int,default=20)
    ap.add_argument("--msteps",type=int,default=25);ap.add_argument("--lr",type=float,default=.04)
    ap.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    run(ap.parse_args())
