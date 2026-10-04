#!/usr/bin/env python3
# Voynich inverse-renderer Phase A: planted-source recoverability.
# NO P70. Uses canonical K12/chunk FORM route grammar only.
import argparse,collections,json,math,os,time,urllib.request
import numpy as np
import torch

BASE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/74ae7aa0088dc07436000f777d9c5d7d705730c2/research/latent_line_state_test_20261004.py"
ns={"__name__":"latent_line_base"}
exec(compile(urllib.request.urlopen(BASE_URL,timeout=60).read().decode(),BASE_URL,"exec"),ns)
rows=ns["rows"]; folds=ns["folds"]; ST=ns["ST"]; KFORM=12

END=12
NOPT=13
NCTX=13 # 0 START, 1..12 previous FORM class
EPS=1e-7

def route_of_token(tok):
    pcs=ns["segment"](tok)
    return [ST[p] for p in pcs]

def build_route_baseline():
    counts=np.zeros((NCTX,NOPT),dtype=np.float64)
    # START -> first class; current class -> next class or END.
    for r in rows:
        route=route_of_token(r["token"])
        if not route: continue
        counts[0,route[0]]+=1
        for i,c in enumerate(route):
            ctx=1+c
            nxt=END if i==len(route)-1 else route[i+1]
            counts[ctx,nxt]+=1
    support=counts>0
    # Smooth only genuinely observed legal transitions; do not invent edges.
    probs=(counts+0.25*support)
    probs=np.divide(probs,probs.sum(1,keepdims=True),out=np.zeros_like(probs),where=probs.sum(1,keepdims=True)>0)
    return probs,support

P0,SUPPORT=build_route_baseline()
LOGP0=np.where(SUPPORT,np.log(np.maximum(P0,1e-30)),-1e30)

def source_graph(K,d,rng):
    A=np.zeros((K,K),float)
    edges=[]
    for i in range(K):
        mandatory={i}
        while len(mandatory)<d:
            mandatory.add(int(rng.integers(K)))
        js=sorted(mandatory)
        w=rng.gamma(shape=1.5,scale=1.0,size=len(js))
        # mild stickiness
        w[js.index(i)] += 2.0
        w/=w.sum()
        A[i,js]=w
        edges.append(js)
    pi=np.ones(K)/K
    return A,pi,edges

def sample_hidden(A,pi,N,rng):
    K=len(pi);z=np.empty(N,dtype=np.int64)
    z[0]=rng.choice(K,p=pi)
    for t in range(1,N): z[t]=rng.choice(K,p=A[z[t-1]])
    return z

def make_coupling(K,rank,strength,rng):
    U=rng.normal(0,1,(K,rank))
    U-=U.mean(0,keepdims=True)
    U/=np.maximum(U.std(0,keepdims=True),1e-9)
    V=rng.normal(0,1,(rank,NCTX,NOPT))
    V-=V.mean(2,keepdims=True)
    V/=np.maximum(V.std(),1e-9)
    return U,V*strength

def sample_route(z,U,V,rng,maxlen=16,max_attempts=100):
    for attempt in range(max_attempts):
        route=[];ctx=0
        for step in range(maxlen):
            logits=LOGP0[ctx].copy()+U[z]@V[:,ctx,:]
            logits[~SUPPORT[ctx]]=-1e30
            m=np.max(logits);q=np.exp(logits-m);q/=q.sum()
            opt=int(rng.choice(NOPT,p=q))
            if opt==END:
                if route:return route
                continue
            route.append(opt);ctx=1+opt
        # Do not forge a termination event: discard this attempt and resample.
    raise RuntimeError(("synthetic_route_failed_to_terminate",z,maxlen,max_attempts))

def generate_corpus(K,d,rank,strength,N,seed):
    rng=np.random.default_rng(seed)
    A,pi,edges=source_graph(K,d,rng)
    U,V=make_coupling(K,rank,strength,rng)
    z=sample_hidden(A,pi,N,rng)
    routes=[sample_route(int(zz),U,V,rng) for zz in z]
    return {"A":A,"pi":pi,"edges":edges,"U":U,"V":V,"z":z,"routes":routes}

def flatten_decisions(routes):
    # token -> list (ctx,opt), padded into dense feature counts for each ctx,opt.
    X=np.zeros((len(routes),NCTX,NOPT),dtype=np.float32)
    for t,route in enumerate(routes):
        if not route:continue
        X[t,0,route[0]]+=1
        for i,c in enumerate(route):
            ctx=1+c;nxt=END if i==len(route)-1 else route[i+1]
            X[t,ctx,nxt]+=1
    return X

def emission_loglik(X,U,V,device):
    # X [N,C,O], U [K,R], V [R,C,O]
    # bias K,C,O; normalized over opt within context.
    bias=torch.einsum("kr,rco->kco",U,V)
    lp0=torch.tensor(LOGP0,dtype=torch.float32,device=device)
    logits=lp0.unsqueeze(0)+bias
    mask=torch.tensor(SUPPORT,dtype=torch.bool,device=device).unsqueeze(0)
    logits=torch.where(mask,logits,torch.tensor(-1e30,device=device))
    logq=torch.log_softmax(logits,dim=2)
    Xt=torch.tensor(X,dtype=torch.float32,device=device)
    # N,K
    return torch.einsum("nco,kco->nk",Xt,logq)

def fb_numpy(E,A,pi):
    # E N,K log emissions. Stable scaled forward-backward + xi sums.
    N,K=E.shape
    la=np.log(np.maximum(A,1e-30));lpi=np.log(np.maximum(pi,1e-30))
    alpha=np.empty((N,K),np.float64);scale=np.empty(N,np.float64)
    alpha[0]=lpi+E[0];m=alpha[0].max();scale[0]=m+math.log(np.exp(alpha[0]-m).sum());alpha[0]-=scale[0]
    for t in range(1,N):
        M=alpha[t-1][:,None]+la
        mm=M.max(0);pred=mm+np.log(np.exp(M-mm).sum(0))
        a=pred+E[t];m=a.max();scale[t]=m+math.log(np.exp(a-m).sum());alpha[t]=a-scale[t]
    beta=np.zeros((N,K),np.float64)
    for t in range(N-2,-1,-1):
        M=la+E[t+1][None,:]+beta[t+1][None,:]
        mm=M.max(1);b=mm+np.log(np.exp(M-mm[:,None]).sum(1))
        beta[t]=b-scale[t+1]
    lg=alpha+beta;mm=lg.max(1,keepdims=True);g=np.exp(lg-mm);g/=g.sum(1,keepdims=True)
    xi=np.zeros((K,K),np.float64)
    for t in range(N-1):
        M=alpha[t][:,None]+la+E[t+1][None,:]+beta[t+1][None,:]
        mm=M.max();Q=np.exp(M-mm);Q/=Q.sum();xi+=Q
    return float(scale.sum()),g,xi

def project_sparse_A(xi,d,eps=1e-4):
    K=xi.shape[0];A=np.zeros_like(xi)
    for i in range(K):
        row=xi[i]+0.05
        # always permit self, plus highest other d-1.
        idx=np.argsort(row)[::-1].tolist()
        keep=[i]
        for j in idx:
            if j not in keep:keep.append(int(j))
            if len(keep)>=d:break
        w=row[keep]+eps;w/=w.sum();A[i,keep]=w
    return A

def nmi(a,b):
    a=np.asarray(a);b=np.asarray(b);n=len(a)
    ua,ia=np.unique(a,return_inverse=True);ub,ib=np.unique(b,return_inverse=True)
    C=np.zeros((len(ua),len(ub)),float)
    np.add.at(C,(ia,ib),1)
    P=C/n;pa=P.sum(1);pb=P.sum(0)
    mi=0.
    for i in range(len(pa)):
        for j in range(len(pb)):
            if P[i,j]>0:mi+=P[i,j]*math.log(P[i,j]/(pa[i]*pb[j])+1e-300)
    ha=-sum(x*math.log(x+1e-300) for x in pa if x>0)
    hb=-sum(x*math.log(x+1e-300) for x in pb if x>0)
    return float(mi/max((ha+hb)/2,1e-15))

def ari(a,b):
    # adjusted Rand index, permutation invariant, no sklearn dependency.
    a=np.asarray(a);b=np.asarray(b);n=len(a)
    ua,ia=np.unique(a,return_inverse=True);ub,ib=np.unique(b,return_inverse=True)
    C=np.zeros((len(ua),len(ub)),dtype=np.int64);np.add.at(C,(ia,ib),1)
    comb=lambda x:x*(x-1)/2
    sumij=comb(C).sum();suma=comb(C.sum(1)).sum();sumb=comb(C.sum(0)).sum();tot=comb(n)
    exp=suma*sumb/max(tot,1);mx=.5*(suma+sumb)
    return float((sumij-exp)/max(mx-exp,1e-15))

def fit_one(X,K,d,rank,seed,device,epochs=35,msteps=20,lr=.06):
    rng=np.random.default_rng(seed)
    # sparse random initial graph
    A,pi,_=source_graph(K,d,rng)
    U=torch.nn.Parameter(torch.tensor(rng.normal(0,.08,(K,rank)),dtype=torch.float32,device=device))
    V=torch.nn.Parameter(torch.tensor(rng.normal(0,.08,(rank,NCTX,NOPT)),dtype=torch.float32,device=device))
    opt=torch.optim.Adam([U,V],lr=lr)
    best=None;last=-1e99
    t0=time.time()
    for em in range(epochs):
        with torch.no_grad():
            E=emission_loglik(X,U,V,device).cpu().numpy().astype(np.float64)
        ll,g,xi=fb_numpy(E,A,pi)
        A=project_sparse_A(xi,d)
        pi=(g[0]+.1);pi/=pi.sum()
        G=torch.tensor(g,dtype=torch.float32,device=device)
        for _ in range(msteps):
            opt.zero_grad(set_to_none=True)
            E2=emission_loglik(X,U,V,device)
            loss=-(G*E2).sum()/len(X)
            # weak shrinkage + zero-mean source gauge
            loss=loss+1e-3*(U.square().mean()+V.square().mean())+1e-2*U.mean(0).square().mean()
            loss.backward();opt.step()
        if ll>last:best=(ll,A.copy(),pi.copy(),U.detach().cpu().numpy().copy(),V.detach().cpu().numpy().copy(),g.copy(),em)
        if abs(ll-last)<1e-3 and em>5:break
        last=ll
    ll,A,pi,U0,V0,g,em=best
    return {"ll":float(ll),"A":A,"pi":pi,"U":U0,"V":V0,"gamma":g,"epochs":int(em+1),"seconds":time.time()-t0}

def run(args):
    device=torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")
    dat=generate_corpus(args.K,args.d,args.rank,args.strength,args.N,args.seed)
    X=flatten_decisions(dat["routes"])
    # train/test split is contiguous and prospective for evaluation.
    ntr=int(.8*args.N);Xtr=X[:ntr];Xte=X[ntr:];ztr=dat["z"][:ntr];zte=dat["z"][ntr:]
    results=[]
    for r in range(args.restarts):
        fit=fit_one(Xtr,args.K,args.d,args.rank,args.seed+10000+r*997,device,args.epochs,args.msteps,args.lr)
        # recover train labels
        pred=fit["gamma"].argmax(1)
        tr_nmi=nmi(ztr,pred);tr_ari=ari(ztr,pred)
        # prospective test filtering with frozen learned params
        with torch.no_grad():
            Ut=torch.tensor(fit["U"],dtype=torch.float32,device=device)
            Vt=torch.tensor(fit["V"],dtype=torch.float32,device=device)
            Ete=emission_loglik(Xte,Ut,Vt,device).cpu().numpy().astype(np.float64)
        llte,gte,_=fb_numpy(Ete,fit["A"],fit["pi"])
        tepred=gte.argmax(1);te_nmi=nmi(zte,tepred);te_ari=ari(zte,tepred)
        rec={"restart":r,"train_ll":fit["ll"],"test_ll":llte,"train_nmi":tr_nmi,"train_ari":tr_ari,
             "test_nmi":te_nmi,"test_ari":te_ari,"epochs":fit["epochs"],"seconds":fit["seconds"]}
        print("RECOVERY_RESTART_JSON="+json.dumps(rec,separators=(",",":")),flush=True)
        results.append(rec)
    best=max(results,key=lambda x:x["train_ll"])
    # throughput benchmark: repeated emission kernel using best model-sized tensors.
    rng=np.random.default_rng(args.seed+777)
    U=torch.tensor(rng.normal(size=(args.K,args.rank)),dtype=torch.float32,device=device)
    V=torch.tensor(rng.normal(size=(args.rank,NCTX,NOPT)),dtype=torch.float32,device=device)
    if device.type=="cuda":torch.cuda.synchronize()
    t0=time.time();loops=args.bench_loops
    for _ in range(loops):
        E=emission_loglik(X,U,V,device)
        _=E.sum()
    if device.type=="cuda":torch.cuda.synchronize()
    dt=time.time()-t0
    decisions=float(X.sum())
    evals=loops*args.N*args.K*max(decisions/args.N,1.0)
    out={
      "phase":"A_recoverability_canary","device":str(device),
      "gpu":torch.cuda.get_device_name(0) if device.type=="cuda" else None,
      "torch":torch.__version__,"N":args.N,"K":args.K,"d":args.d,"rank":args.rank,"strength":args.strength,
      "restarts":args.restarts,"best":best,
      "median_test_nmi":float(np.median([x["test_nmi"] for x in results])),
      "median_test_ari":float(np.median([x["test_ari"] for x in results])),
      "stable_fraction_nmi70":float(np.mean([x["test_nmi"]>=.70 for x in results])),
      "benchmark":{"loops":loops,"seconds":dt,"approx_state_decision_evals":evals,"evals_per_sec":evals/dt},
      "route_legal_edges":int(SUPPORT.sum()),"route_baseline_sha_like":float(P0.sum()),
      "corpus_rows":len(rows)
    }
    print("INVERSE_RECOVERABILITY_JSON="+json.dumps(out,separators=(",",":")),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--N",type=int,default=4000)
    ap.add_argument("--K",type=int,default=16)
    ap.add_argument("--d",type=int,default=4)
    ap.add_argument("--rank",type=int,default=2)
    ap.add_argument("--strength",type=float,default=.85)
    ap.add_argument("--seed",type=int,default=20261004)
    ap.add_argument("--restarts",type=int,default=3)
    ap.add_argument("--epochs",type=int,default=30)
    ap.add_argument("--msteps",type=int,default=15)
    ap.add_argument("--lr",type=float,default=.06)
    ap.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    ap.add_argument("--bench-loops",type=int,default=10)
    args=ap.parse_args()
    run(args)
