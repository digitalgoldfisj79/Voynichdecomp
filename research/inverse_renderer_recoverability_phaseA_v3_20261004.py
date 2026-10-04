#!/usr/bin/env python3
# Inverse-renderer Phase A v3: categorical route-HMM state discovery -> low-rank renderer model.
# NO P70.
import argparse,collections,json,math,time,urllib.request
import numpy as np, torch

BASE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/664cd58a01a0d26bf66619fafc92945bee1ffe75/research/inverse_renderer_recoverability_phaseA_20261004.py"
b={"__name__":"phaseA_base"}
exec(compile(urllib.request.urlopen(BASE_URL,timeout=60).read().decode(),BASE_URL,"exec"),b)
P0=b["P0"];SUPPORT=b["SUPPORT"];LOGP0=b["LOGP0"];rows=b["rows"]
NCTX=b["NCTX"];NOPT=b["NOPT"];END=b["END"]
source_graph=b["source_graph"];sample_hidden=b["sample_hidden"];flatten_decisions=b["flatten_decisions"]
fb_numpy=b["fb_numpy"];nmi=b["nmi"];ari=b["ari"]

HAZARD_MIX=.05
BIAS_CAP=4.0

def bias_np(U,V):
    raw=np.einsum("kr,rco->kco",U,V)
    B=BIAS_CAP*np.tanh(raw/BIAS_CAP);B[:,:,END]=0.;B[:,~SUPPORT]=0.
    return B
def q_np(z,U,V,ctx):
    B=bias_np(U,V)[z,ctx];lg=LOGP0[ctx]+B;lg[~SUPPORT[ctx]]=-1e30
    m=lg.max();qb=np.exp(lg-m);qb/=qb.sum()
    q=(1-HAZARD_MIX)*qb+HAZARD_MIX*P0[ctx];q[~SUPPORT[ctx]]=0.;q/=q.sum()
    return q
def make_coupling(K,rank,strength,rng):
    U=rng.normal(size=(K,rank));U-=U.mean(0);U/=np.maximum(U.std(0),1e-9)
    V=rng.normal(size=(rank,NCTX,NOPT));V[:,:,END]=0.
    V/=max(V[:,SUPPORT].std(),1e-9);V*=strength
    return U,V
def sample_route(z,U,V,rng,maxlen=96):
    route=[];ctx=0
    for _ in range(maxlen):
        opt=int(rng.choice(NOPT,p=q_np(z,U,V,ctx)))
        if opt==END:
            if route:return route
            continue
        route.append(opt);ctx=1+opt
    return None
def generate(K,d,rank,strength,N,seed):
    rng=np.random.default_rng(seed);A,pi,_=source_graph(K,d,rng);U,V=make_coupling(K,rank,strength,rng);z=sample_hidden(A,pi,N,rng)
    rr=[]
    for zz in z:
        rt=None
        for _ in range(100):
            rt=sample_route(int(zz),U,V,rng)
            if rt is not None:break
        if rt is None:raise RuntimeError("nontermination")
        rr.append(rt)
    return {"A":A,"pi":pi,"U":U,"V":V,"z":z,"routes":rr}

def emission_ll(X,U,V,device):
    raw=torch.einsum("kr,rco->kco",U,V)
    B=BIAS_CAP*torch.tanh(raw/BIAS_CAP);B=B.clone();B[:,:,END]=0.
    lp0=torch.tensor(LOGP0,dtype=torch.float32,device=device)
    mask=torch.tensor(SUPPORT,dtype=torch.bool,device=device).unsqueeze(0)
    lg=torch.where(mask,lp0.unsqueeze(0)+B,torch.tensor(-1e30,device=device))
    qb=torch.softmax(lg,dim=2);p0=torch.tensor(P0,dtype=torch.float32,device=device).unsqueeze(0)
    q=(1-HAZARD_MIX)*qb+HAZARD_MIX*p0
    lq=torch.log(torch.clamp(q,min=1e-30));Xt=torch.tensor(X,dtype=torch.float32,device=device)
    return torch.einsum("nco,kco->nk",Xt,lq)

def route_ids(routes):
    mp={};obs=np.empty(len(routes),int)
    for i,r in enumerate(routes):
        key=tuple(r)
        if key not in mp:mp[key]=len(mp)
        obs[i]=mp[key]
    return obs,len(mp)

def cat_hmm(obs,K,M,seed,epochs=80):
    rng=np.random.default_rng(seed);N=len(obs)
    # global route frequencies as a stabilizing prior
    gf=np.bincount(obs,minlength=M).astype(float)+.25;gf/=gf.sum()
    best=None
    for restart in range(4):
        # sticky dense transition
        A=rng.gamma(1.,1.,(K,K))+.3*np.eye(K);A/=A.sum(1,keepdims=True)
        pi=np.ones(K)/K
        # perturb global categorical distribution
        E=rng.gamma(.7,1.,(K,M))*gf[None,:]+.03*gf[None,:]
        E/=E.sum(1,keepdims=True)
        last=-1e99
        for ep in range(epochs):
            loge=np.log(np.maximum(E[:,obs].T,1e-30))
            ll,g,xi=fb_numpy(loge,A,pi)
            A=xi+.15;A/=A.sum(1,keepdims=True)
            pi=g[0]+.1;pi/=pi.sum()
            C=np.tile(.75*gf,(K,1))
            for k in range(K):np.add.at(C[k],obs,g[:,k])
            E=C/C.sum(1,keepdims=True)
            if abs(ll-last)<1e-3 and ep>10:break
            last=ll
        if best is None or ll>best[0]:best=(ll,A.copy(),pi.copy(),E.copy(),g.copy(),ep+1)
    return best

def init_lowrank_from_gamma(X,g,rank):
    K=g.shape[1];B=np.zeros((K,NCTX,NOPT))
    for k in range(K):
        C=np.einsum("n,nco->co",g[:,k],X)
        for ctx in range(NCTX):
            q=C[ctx]+4*P0[ctx];q=np.where(SUPPORT[ctx],q,0)
            if q.sum()>0:q/=q.sum()
            z=np.zeros(NOPT);s=SUPPORT[ctx]
            z[s]=np.log(np.maximum(q[s],1e-12))-np.log(np.maximum(P0[ctx,s],1e-12))
            z[END]=0.;B[k,ctx]=np.clip(z,-BIAS_CAP,BIAS_CAP)
    M=B.reshape(K,-1);uu,ss,vt=np.linalg.svd(M,full_matrices=False);rr=min(rank,len(ss));sq=np.sqrt(np.maximum(ss[:rr],0))
    U=uu[:,:rr]*sq;V=(sq[:,None]*vt[:rr]).reshape(rr,NCTX,NOPT)
    if rr<rank:
        U=np.pad(U,((0,0),(0,rank-rr)));V=np.pad(V,((0,rank-rr),(0,0),(0,0)))
    V[:,:,END]=0.
    return U,V

def sparse_A(xi,d):
    K=len(xi);A=np.zeros_like(xi)
    for i in range(K):
        row=xi[i]+.1;keep=np.argsort(row)[::-1][:d];w=row[keep];w/=w.sum();A[i,keep]=w
    return A

def refine(X,A,pi,U0,V0,d,device,epochs=70,msteps=25,lr=.035,dense_epochs=25):
    U=torch.nn.Parameter(torch.tensor(U0,dtype=torch.float32,device=device));V=torch.nn.Parameter(torch.tensor(V0,dtype=torch.float32,device=device))
    opt=torch.optim.Adam([U,V],lr=lr);best=None;last=-1e99;t0=time.time()
    for ep in range(epochs):
        with torch.no_grad():E=emission_ll(X,U,V,device).cpu().numpy().astype(float)
        ll,g,xi=fb_numpy(E,A,pi)
        if ep<dense_epochs:
            A=xi+.1;A/=A.sum(1,keepdims=True)
        else:A=sparse_A(xi,d)
        pi=g[0]+.1;pi/=pi.sum();G=torch.tensor(g,dtype=torch.float32,device=device)
        for _ in range(msteps):
            opt.zero_grad(set_to_none=True);E2=emission_ll(X,U,V,device)
            loss=-(G*E2).sum()/len(X)+5e-4*(U.square().mean()+V.square().mean())+1e-2*U.mean(0).square().mean()
            loss.backward();torch.nn.utils.clip_grad_norm_([U,V],5.);opt.step()
            with torch.no_grad():V[:,:,END]=0.
        with torch.no_grad():Er=emission_ll(X,U,V,device).cpu().numpy().astype(float)
        llr,gr,_=fb_numpy(Er,A,pi)
        if best is None or llr>best[0]:best=(llr,A.copy(),pi.copy(),U.detach().cpu().numpy().copy(),V.detach().cpu().numpy().copy(),gr.copy(),ep+1)
        if ep>dense_epochs+8 and abs(llr-last)<1e-3:break
        last=llr
    return best+(time.time()-t0,)

def oracle(dat,X,device):
    U=torch.tensor(dat["U"],dtype=torch.float32,device=device);V=torch.tensor(dat["V"],dtype=torch.float32,device=device)
    with torch.no_grad():E=emission_ll(X,U,V,device).cpu().numpy().astype(float)
    ll,g,_=fb_numpy(E,dat["A"],dat["pi"]);p=g.argmax(1)
    return {"ll":ll,"nmi":nmi(dat["z"],p),"ari":ari(dat["z"],p)}

def run_strength(s,args,device):
    dat=generate(args.K,args.d,args.rank,s,args.N,args.seed);X=flatten_decisions(dat["routes"]);obs,M=route_ids(dat["routes"])
    orc=oracle(dat,X,device);runs=[]
    for r in range(args.restarts):
        cll,A,pi,CE,g0,cep=cat_hmm(obs,args.K,M,args.seed+100*r)
        catpred=g0.argmax(1);cn=nmi(dat["z"],catpred)
        U0,V0=init_lowrank_from_gamma(X,g0,args.rank)
        ll,A2,pi2,U,V,g,ep,sec=refine(X,A,pi,U0,V0,args.d,device,args.epochs,args.msteps,args.lr,args.dense_epochs)
        pred=g.argmax(1)
        z={"restart":r,"cat_ll":cll,"cat_nmi":cn,"cat_ari":ari(dat["z"],catpred),"cat_epochs":cep,
           "final_ll":ll,"final_nmi":nmi(dat["z"],pred),"final_ari":ari(dat["z"],pred),"final_epochs":ep,"seconds":sec}
        runs.append(z);print("V3_RESTART_JSON="+json.dumps({"strength":s,**z},separators=(",",":")),flush=True)
    return {"strength":s,"oracle":orc,"unique_routes":M,"runs":runs,
            "median_cat_nmi":float(np.median([x["cat_nmi"] for x in runs])),
            "median_final_nmi":float(np.median([x["final_nmi"] for x in runs])),
            "best_final_nmi":max(x["final_nmi"] for x in runs),
            "stable_fraction_nmi70":float(np.mean([x["final_nmi"]>=.70 for x in runs]))}

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--N",type=int,default=4000);ap.add_argument("--K",type=int,default=16)
    ap.add_argument("--d",type=int,default=4);ap.add_argument("--rank",type=int,default=2);ap.add_argument("--seed",type=int,default=20261004)
    ap.add_argument("--strengths",default="1.5,2.5,4.0");ap.add_argument("--restarts",type=int,default=3)
    ap.add_argument("--epochs",type=int,default=70);ap.add_argument("--dense-epochs",type=int,default=25);ap.add_argument("--msteps",type=int,default=25)
    ap.add_argument("--lr",type=float,default=.035);ap.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    args=ap.parse_args();device=torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")
    out=[]
    for s in map(float,args.strengths.split(",")):
        z=run_strength(s,args,device);out.append(z);print("V3_STRENGTH_JSON="+json.dumps(z,separators=(",",":")),flush=True)
    print("INVERSE_PHASEA_V3_JSON="+json.dumps({"phase":"A_v3","device":str(device),"hazard_mix":HAZARD_MIX,"bias_cap":BIAS_CAP,
      "N":args.N,"K":args.K,"d":args.d,"rank":args.rank,"results":out,
      "gate_pass":bool(any(x["oracle"]["nmi"]>=.70 and x["median_final_nmi"]>=.70 and x["stable_fraction_nmi70"]>=.67 for x in out))},separators=(",",":")),flush=True)
