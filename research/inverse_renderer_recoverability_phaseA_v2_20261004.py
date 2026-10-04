#!/usr/bin/env python3
# Inverse renderer Phase A v2: termination-safe planted source + informed inference.
# NO P70.
import argparse,json,math,time,urllib.request,collections
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

BASE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/664cd58a01a0d26bf66619fafc92945bee1ffe75/research/inverse_renderer_recoverability_phaseA_20261004.py"
b={"__name__":"phaseA_base"}
exec(compile(urllib.request.urlopen(BASE_URL,timeout=60).read().decode(),BASE_URL,"exec"),b)

P0=b["P0"]; SUPPORT=b["SUPPORT"]; LOGP0=b["LOGP0"]; rows=b["rows"]
NCTX=b["NCTX"];NOPT=b["NOPT"];END=b["END"]
source_graph=b["source_graph"];sample_hidden=b["sample_hidden"];flatten_decisions=b["flatten_decisions"]
fb_numpy=b["fb_numpy"];nmi=b["nmi"];ari=b["ari"]

HAZARD_MIX=.10
BIAS_CAP=2.5

def safe_bias(U,V):
    # source can strongly alter legal FORM choices, but not arbitrarily.
    raw=np.einsum("kr,rco->kco",U,V)
    B=BIAS_CAP*np.tanh(raw/BIAS_CAP)
    B[:,:,END]=0.0
    B[:,~SUPPORT]=0.0
    return B

def safe_q_numpy(z,U,V,ctx):
    B=safe_bias(U,V)[z,ctx]
    logits=LOGP0[ctx]+B
    logits[~SUPPORT[ctx]]=-1e30
    m=logits.max(); qb=np.exp(logits-m);qb/=qb.sum()
    q=(1-HAZARD_MIX)*qb+HAZARD_MIX*P0[ctx]
    q[~SUPPORT[ctx]]=0.;q/=q.sum()
    return q

def make_coupling(K,rank,strength,rng):
    U=rng.normal(size=(K,rank));U-=U.mean(0,keepdims=True);U/=np.maximum(U.std(0,keepdims=True),1e-9)
    V=rng.normal(size=(rank,NCTX,NOPT));V[:,:,END]=0
    # normalize only legal non-END entries
    z=V[:,SUPPORT].std()
    V/=max(z,1e-9);V*=strength
    return U,V

def sample_route(z,U,V,rng,maxlen=64):
    route=[];ctx=0
    for step in range(maxlen):
        q=safe_q_numpy(z,U,V,ctx)
        opt=int(rng.choice(NOPT,p=q))
        if opt==END:
            if route:return route
            continue
        route.append(opt);ctx=1+opt
    # With hazard mixture this should be extraordinarily rare; reject rather than forge END.
    return None

def generate_corpus(K,d,rank,strength,N,seed):
    rng=np.random.default_rng(seed);A,pi,edges=source_graph(K,d,rng);U,V=make_coupling(K,rank,strength,rng)
    z=sample_hidden(A,pi,N,rng);routes=[]
    for zz in z:
        rt=None
        for _ in range(100):
            rt=sample_route(int(zz),U,V,rng)
            if rt is not None:break
        if rt is None:raise RuntimeError(("proper_route_generation_failed",int(zz)))
        routes.append(rt)
    return {"A":A,"pi":pi,"edges":edges,"U":U,"V":V,"z":z,"routes":routes}

def emission_loglik(X,U,V,device):
    # torch exact counterpart of safe_q_numpy
    raw=torch.einsum("kr,rco->kco",U,V)
    B=BIAS_CAP*torch.tanh(raw/BIAS_CAP)
    B=B.clone();B[:,:,END]=0.
    lp0=torch.tensor(LOGP0,dtype=torch.float32,device=device)
    mask=torch.tensor(SUPPORT,dtype=torch.bool,device=device).unsqueeze(0)
    logits=lp0.unsqueeze(0)+B
    logits=torch.where(mask,logits,torch.tensor(-1e30,device=device))
    qb=torch.softmax(logits,dim=2)
    p0=torch.tensor(P0,dtype=torch.float32,device=device).unsqueeze(0)
    q=(1-HAZARD_MIX)*qb+HAZARD_MIX*p0
    logq=torch.log(torch.clamp(q,min=1e-30))
    Xt=torch.tensor(X,dtype=torch.float32,device=device)
    return torch.einsum("nco,kco->nk",Xt,logq)

def dense_A_from_labels(lab,K):
    C=np.full((K,K),.25,float)
    for a,c in zip(lab[:-1],lab[1:]):C[int(a),int(c)]+=1
    return C/C.sum(1,keepdims=True)

def init_from_kmeans(X,K,rank,seed):
    # Use only observed legal decision cells; length-normalize to avoid clustering only by token length.
    feat=X[:,SUPPORT].astype(np.float64)
    den=np.maximum(feat.sum(1,keepdims=True),1)
    F=feat/den
    # SVD denoising before kmeans.
    ncomp=min(max(8,2*K),F.shape[1]-1,K*4)
    if ncomp>=2:
        Z=TruncatedSVD(n_components=ncomp,random_state=seed).fit_transform(F)
    else:Z=F
    lab=KMeans(n_clusters=K,n_init=10,random_state=seed,max_iter=300).fit_predict(Z)
    A=dense_A_from_labels(lab,K)
    pi=np.bincount(lab,minlength=K).astype(float)+.25;pi/=pi.sum()

    # Empirical per-cluster log tilt relative to frozen renderer.
    B=np.zeros((K,NCTX,NOPT),float)
    for k in range(K):
        ix=np.where(lab==k)[0]
        Xk=X[ix].sum(0) if len(ix) else np.zeros((NCTX,NOPT))
        for ctx in range(NCTX):
            supp=SUPPORT[ctx]
            prior=4.0*P0[ctx]
            q=(Xk[ctx]+prior);q=np.where(supp,q,0)
            if q.sum()>0:q/=q.sum()
            vals=np.zeros(NOPT)
            vals[supp]=np.log(np.maximum(q[supp],1e-12))-np.log(np.maximum(P0[ctx,supp],1e-12))
            vals[END]=0.
            B[k,ctx]=np.clip(vals,-BIAS_CAP,BIAS_CAP)
    M=B.reshape(K,-1)
    # factor to requested rank
    uu,ss,vt=np.linalg.svd(M,full_matrices=False)
    rr=min(rank,len(ss));sroot=np.sqrt(np.maximum(ss[:rr],0))
    U=uu[:,:rr]*sroot
    Vf=(sroot[:,None]*vt[:rr])
    V=Vf.reshape(rr,NCTX,NOPT)
    if rr<rank:
        U=np.pad(U,((0,0),(0,rank-rr)))
        V=np.pad(V,((0,rank-rr),(0,0),(0,0)))
    V[:,:,END]=0.
    return A,pi,U,V,lab

def annealed_fb(E,A,pi,temp):
    if temp!=1.0:
        E=E/temp
        # flatten transition differences early too
        At=np.power(np.maximum(A,1e-30),1/temp);At/=At.sum(1,keepdims=True)
        pit=np.power(np.maximum(pi,1e-30),1/temp);pit/=pit.sum()
        return fb_numpy(E,At,pit)
    return fb_numpy(E,A,pi)

def project_sparse_A(xi,d):
    K=xi.shape[0];A=np.zeros_like(xi)
    for i in range(K):
        row=xi[i]+.10
        idx=np.argsort(row)[::-1]
        keep=[]
        # self is allowed but not forcibly retained if unsupported
        for j in idx:
            keep.append(int(j))
            if len(keep)>=d:break
        w=row[keep];w/=w.sum();A[i,keep]=w
    return A

def fit_one(X,K,d,rank,seed,device,epochs=70,msteps=25,lr=.04,dense_epochs=25):
    rng=np.random.default_rng(seed)
    A,pi,U0,V0,lab=init_from_kmeans(X,K,rank,seed)
    U=torch.nn.Parameter(torch.tensor(U0,dtype=torch.float32,device=device))
    V=torch.nn.Parameter(torch.tensor(V0,dtype=torch.float32,device=device))
    opt=torch.optim.Adam([U,V],lr=lr)
    best=None;last=-1e99;t0=time.time()
    for em in range(epochs):
        # deterministic annealing 1.8 -> 1 across first 30 epochs
        temp=max(1.0,1.8-0.8*min(em,30)/30)
        with torch.no_grad():E=emission_loglik(X,U,V,device).cpu().numpy().astype(np.float64)
        ll,g,xi=annealed_fb(E,A,pi,temp)
        # Keep transition dense until emissions/state partition settle.
        if em<dense_epochs:
            A=(xi+.10);A/=A.sum(1,keepdims=True)
        else:
            A=project_sparse_A(xi,d)
        pi=(g[0]+.1);pi/=pi.sum()
        G=torch.tensor(g,dtype=torch.float32,device=device)
        for _ in range(msteps):
            opt.zero_grad(set_to_none=True);E2=emission_loglik(X,U,V,device)
            loss=-(G*E2).sum()/len(X)
            loss=loss+5e-4*(U.square().mean()+V.square().mean())+1e-2*U.mean(0).square().mean()
            loss.backward()
            # END bias fixed by model; stabilize remaining coupling
            torch.nn.utils.clip_grad_norm_([U,V],5.0)
            opt.step()
            with torch.no_grad():V[:,:,END]=0.
        # evaluate true unannealed likelihood for model selection
        with torch.no_grad():Er=emission_loglik(X,U,V,device).cpu().numpy().astype(np.float64)
        llr,gr,_=fb_numpy(Er,A,pi)
        if best is None or llr>best[0]:
            best=(llr,A.copy(),pi.copy(),U.detach().cpu().numpy().copy(),V.detach().cpu().numpy().copy(),gr.copy(),em,lab.copy())
        if em>=dense_epochs+8 and abs(llr-last)<1e-3:break
        last=llr
    ll,A,pi,U0,V0,g,em,lab=best
    return {"ll":float(ll),"A":A,"pi":pi,"U":U0,"V":V0,"gamma":g,"epochs":int(em+1),
            "seconds":time.time()-t0,"init_labels":lab}

def oracle(dat,X,device):
    U=torch.tensor(dat["U"],dtype=torch.float32,device=device);V=torch.tensor(dat["V"],dtype=torch.float32,device=device)
    with torch.no_grad():E=emission_loglik(X,U,V,device).cpu().numpy().astype(np.float64)
    ll,g,_=fb_numpy(E,dat["A"],dat["pi"]);p=g.argmax(1)
    return {"ll":float(ll),"nmi":nmi(dat["z"],p),"ari":ari(dat["z"],p)}

def one_strength(strength,args,device):
    dat=generate_corpus(args.K,args.d,args.rank,strength,args.N,args.seed)
    X=flatten_decisions(dat["routes"]);orc=oracle(dat,X,device);fits=[]
    for r in range(args.restarts):
        f=fit_one(X,args.K,args.d,args.rank,args.seed+1000+r*997,device,args.epochs,args.msteps,args.lr,args.dense_epochs)
        pred=f["gamma"].argmax(1)
        rec={"restart":r,"ll":f["ll"],"nmi":nmi(dat["z"],pred),"ari":ari(dat["z"],pred),
             "init_nmi":nmi(dat["z"],f["init_labels"]),"epochs":f["epochs"],"seconds":f["seconds"]}
        fits.append(rec);print("V2_RESTART_JSON="+json.dumps({"strength":strength,**rec},separators=(",",":")),flush=True)
    return {"strength":strength,"oracle":orc,"fits":fits,"median_nmi":float(np.median([x["nmi"] for x in fits])),
            "best_nmi":max(x["nmi"] for x in fits),"stable_fraction_nmi70":float(np.mean([x["nmi"]>=.70 for x in fits]))}

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--N",type=int,default=4000);ap.add_argument("--K",type=int,default=16);ap.add_argument("--d",type=int,default=4)
    ap.add_argument("--rank",type=int,default=2);ap.add_argument("--seed",type=int,default=20261004)
    ap.add_argument("--strengths",default="1.25,1.75,2.5");ap.add_argument("--restarts",type=int,default=4)
    ap.add_argument("--epochs",type=int,default=70);ap.add_argument("--dense-epochs",type=int,default=25)
    ap.add_argument("--msteps",type=int,default=25);ap.add_argument("--lr",type=float,default=.04)
    ap.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    args=ap.parse_args();device=torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")
    out=[]
    for s in [float(x) for x in args.strengths.split(",")]:
        z=one_strength(s,args,device);out.append(z);print("V2_STRENGTH_JSON="+json.dumps(z,separators=(",",":")),flush=True)
    final={"phase":"A_v2","device":str(device),"gpu":torch.cuda.get_device_name(0) if device.type=="cuda" else None,
           "hazard_mix":HAZARD_MIX,"bias_cap":BIAS_CAP,"N":args.N,"K":args.K,"d":args.d,"rank":args.rank,
           "results":out,"gate_pass":bool(all(x["oracle"]["nmi"]>=.70 and x["median_nmi"]>=.70 for x in out if x["strength"]>=1.75))}
    print("INVERSE_PHASEA_V2_JSON="+json.dumps(final,separators=(",",":")),flush=True)
