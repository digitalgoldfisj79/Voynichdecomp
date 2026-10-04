#!/usr/bin/env python3
# Phase K3b: blind asymmetric F1 recovery on the state-separated FORM socket.
# ENTRY strength 5, ROUTE strength .5. Exact legal route family only.
# Synthetic-only. NO P70. No real Voynich inversion.
import json,math,urllib.request
import numpy as np
from scipy.optimize import minimize,linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor,as_completed

K0URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f/research/select_form_socket_oracle_phaseK0_20261004.py"
k0={"__name__":"k0"};exec(compile(urllib.request.urlopen(K0URL,timeout=60).read().decode(),K0URL,"exec"),k0)

K=8;D=4;R=2;ENTRY=5.;ROUTE=.5;N=4000;SEED=20261004
NFIT=2700;NTR=3200

def generate(seed):
    rng=np.random.default_rng(seed);A,pi=k0["source_graph"](K,D,rng)
    U,Ve0,Vr0=k0["controls"](K,R,1.,rng,"F1");Ve=Ve0*ENTRY;Vr=Vr0*ROUTE
    z=k0["hidden_seq"](A,pi,N,rng);obs=[]
    for x in z:
        for _ in range(100):
            t=k0["gen_token"](int(x),U,Ve,Vr,rng)
            if t is not None:obs.append(t);break
        else:raise RuntimeError("nontermination")
    return A,pi,U,Ve,Vr,z,obs

Atrue,pitrue,Utrue,Vetrue,Vrtrue,ztruth,obs=generate(SEED)

def features(obs):
    S=np.zeros((len(obs),12),float);C=np.zeros((len(obs),8,12),float)
    for t,tok in enumerate(obs):
        S[t,k0["PCLASS"][tok[0]]]=1.
        for i in range(len(tok)-1):
            g=k0["G_OF"][k0["PCLASS"][tok[i]]];dc=k0["PCLASS"][tok[i+1]]
            C[t,g,dc]+=1.
    return S,C
S,C=features(obs)
Sf,Sv,Str,Ste=S[:NFIT],S[NFIT:NTR],S[:NTR],S[NTR:]
Cf,Cv,Ctr,Cte=C[:NFIT],C[NFIT:NTR],C[:NTR],C[NTR:]
BASEE=k0["P_START_CLASS"];BASER=k0["P_CONT"];LEGAL=k0["LEGAL"]
LE=np.log(np.maximum(BASEE,1e-30))

def probs(U,Ve,Vr):
    se=LE[None,:]+U@Ve;se-=se.max(1,keepdims=True);Qe=np.exp(se);Qe/=Qe.sum(1,keepdims=True)
    Qr=np.zeros((K,8,12),float)
    b=U@Vr
    for z in range(K):
        for g in range(8):
            ok=LEGAL[g]& (BASER[g]>0)
            x=np.log(np.maximum(BASER[g,ok],1e-30))+b[z,ok];x-=x.max();v=np.exp(x);v/=v.sum();Qr[z,g,ok]=v
    return Qe,Qr

def emissions(Sb,Cb,U,Ve,Vr):
    Qe,Qr=probs(U,Ve,Vr)
    E=Sb@np.log(np.maximum(Qe,1e-300)).T
    # token route-count dot state/group/dest log probs
    for g in range(8):
        E+=Cb[:,g,:]@np.log(np.maximum(Qr[:,g,:],1e-300)).T
    return E

def fb(E,A,pi):
    n=len(E);la=np.log(np.maximum(A,1e-300));lp=np.log(np.maximum(pi,1e-300))
    al=np.empty((n,K));sc=np.empty(n)
    a=lp+E[0];mx=a.max();sc[0]=mx+math.log(np.exp(a-mx).sum());al[0]=a-sc[0]
    for t in range(1,n):
        M=al[t-1][:,None]+la;mm=M.max(0);pr=mm+np.log(np.exp(M-mm).sum(0))
        a=pr+E[t];mx=a.max();sc[t]=mx+math.log(np.exp(a-mx).sum());al[t]=a-sc[t]
    be=np.zeros((n,K))
    for t in range(n-2,-1,-1):
        M=la+E[t+1][None,:]+be[t+1][None,:];mm=M.max(1)
        be[t]=mm+np.log(np.exp(M-mm[:,None]).sum(1))-sc[t+1]
    lg=al+be;mm=lg.max(1,keepdims=True);gm=np.exp(lg-mm);gm/=gm.sum(1,keepdims=True)
    xi=np.zeros((K,K),float)
    for t in range(n-1):
        M=al[t][:,None]+la+E[t+1][None,:]+be[t+1][None,:];mm=M.max();q=np.exp(M-mm);q/=q.sum();xi+=q
    return float(sc.sum()),gm,xi

def sparse_A(xi):
    A=np.zeros((K,K),float)
    for i in range(K):
        row=xi[i]+.05;keep=[i]
        for j in np.argsort(row)[::-1]:
            if int(j) not in keep:keep.append(int(j))
            if len(keep)>=D:break
        w=row[keep];w/=w.sum();A[i,keep]=w
    return A

def weighted_counts(Sb,Cb,gamma):
    Ce=gamma.T@Sb
    Cr=np.einsum("tk,tgc->kgc",gamma,Cb)
    return Ce,Cr

def fit_controls(Ce,Cr,U0,Ve0,Vr0,l2=.02):
    x0=np.r_[U0.ravel(),Ve0.ravel(),Vr0.ravel()]
    nU=K*R;nV=R*12
    def fg(x):
        U=x[:nU].reshape(K,R);Ve=x[nU:nU+nV].reshape(R,12);Vr=x[nU+nV:].reshape(R,12)
        Qe,Qr=probs(U,Ve,Vr)
        ne=Ce.sum(1,keepdims=True);De=ne*Qe-Ce
        loss=-float(np.sum(Ce*np.log(np.maximum(Qe,1e-300))))
        gU=De@Ve.T;gVe=U.T@De;gVr=np.zeros_like(Vr)
        for g in range(8):
            nr=Cr[:,g,:].sum(1,keepdims=True);Dr=nr*Qr[:,g,:]-Cr[:,g,:]
            loss-=float(np.sum(Cr[:,g,:]*np.log(np.maximum(Qr[:,g,:],1e-300))))
            gU+=Dr@Vr.T;gVr+=U.T@Dr
        loss+=.5*l2*(np.sum(U*U)+np.sum(Ve*Ve)+np.sum(Vr*Vr))
        gU+=l2*U;gVe+=l2*Ve;gVr+=l2*Vr
        return loss,np.r_[gU.ravel(),gVe.ravel(),gVr.ravel()]
    rr=minimize(lambda x:fg(x),x0,jac=True,method="L-BFGS-B",options={"maxiter":120,"ftol":1e-10})
    x=rr.x;return x[:nU].reshape(K,R),x[nU:nU+nV].reshape(R,12),x[nU+nV:].reshape(R,12)

def labels_init(seed):
    rng=np.random.default_rng(seed)
    # Observable socket features only; no truth.
    F=np.c_[S,C.reshape(len(C),-1)]
    # stabilize very sparse route columns; add tiny deterministic jitter by seed
    Z=StandardScaler().fit_transform(F[:NFIT])
    Z=Z+rng.normal(0,.01,Z.shape)
    labels=KMeans(K,n_init=8,random_state=seed,max_iter=300).fit_predict(Z)
    TC=np.full((K,K),.1,float)
    for a,b in zip(labels[:-1],labels[1:]):TC[a,b]+=1
    A=TC/TC.sum(1,keepdims=True);pi=np.bincount(labels[:200],minlength=K).astype(float)+1;pi/=pi.sum()
    H=np.zeros((NFIT,K),float);H[np.arange(NFIT),labels]=1
    Ce,Cr=weighted_counts(Sf,Cf,H)
    U=rng.normal(0,.2,(K,R));Ve=rng.normal(0,.2,(R,12));Vr=rng.normal(0,.1,(R,12))
    U,Ve,Vr=fit_controls(Ce,Cr,U,Ve,Vr)
    return A,pi,U,Ve,Vr

def fit_one(seed):
    A,pi,U,Ve,Vr=labels_init(seed);best=None;last=-1e300
    for stage,nit in (("dense",18),("sparse",25)):
        for it in range(nit):
            E=emissions(Sf,Cf,U,Ve,Vr);ll,g,xi=fb(E,A,pi)
            if stage=="dense":
                A=xi+.15;A/=A.sum(1,keepdims=True)
            else:A=sparse_A(xi)
            pi=g[0]+.1;pi/=pi.sum()
            Ce,Cr=weighted_counts(Sf,Cf,g);U,Ve,Vr=fit_controls(Ce,Cr,U,Ve,Vr)
            ll2,g2,xi2=fb(emissions(Sf,Cf,U,Ve,Vr),A,pi)
            if best is None or ll2>best[0]:best=(ll2,A.copy(),pi.copy(),U.copy(),Ve.copy(),Vr.copy())
            if it>8 and abs(ll2-last)<1e-4:break
            last=ll2
        if stage=="dense":
            ll,A,pi,U,Ve,Vr=best
            _,_,xi=fb(emissions(Sf,Cf,U,Ve,Vr),A,pi);A=sparse_A(xi);last=-1e300
    ll,A,pi,U,Ve,Vr=best
    lv,_,_=fb(emissions(Sv,Cv,U,Ve,Vr),A,pi)
    return {"seed":seed,"fit_ll":float(ll),"val_ll":float(lv),"A":A,"pi":pi,"U":U,"Ve":Ve,"Vr":Vr}

def refit(md):
    A=md["A"].copy();pi=md["pi"].copy();U=md["U"].copy();Ve=md["Ve"].copy();Vr=md["Vr"].copy()
    best=None;last=-1e300
    for it in range(35):
        ll,g,xi=fb(emissions(Str,Ctr,U,Ve,Vr),A,pi);A=sparse_A(xi);pi=g[0]+.1;pi/=pi.sum()
        Ce,Cr=weighted_counts(Str,Ctr,g);U,Ve,Vr=fit_controls(Ce,Cr,U,Ve,Vr)
        ll2,g2,xi2=fb(emissions(Str,Ctr,U,Ve,Vr),A,pi)
        if best is None or ll2>best[0]:best=(ll2,A.copy(),pi.copy(),U.copy(),Ve.copy(),Vr.copy())
        if it>8 and abs(ll2-last)<1e-4:break
        last=ll2
    return best

def nmi(a,b):return k0["nmi"](a,b)
def ari(a,b):return k0["ari"](a,b)
def edge_f1(ztrue,zpred,Aest):
    X=np.zeros((K,K),int)
    for a,b in zip(ztrue,zpred):X[int(a),int(b)]+=1
    rr,cc=linear_sum_assignment(-X);mp={int(c):int(r) for r,c in zip(rr,cc)}
    ee={(mp.get(i,i),mp.get(j,j)) for i in range(K) for j in range(K) if Aest[i,j]>0}
    te={(i,j) for i in range(K) for j in range(K) if Atrue[i,j]>0}
    tp=len(ee&te);p=tp/len(ee);r=tp/len(te);return 2*p*r/(p+r) if p+r else 0.

if __name__=="__main__":
    seeds=[SEED+30000+i*211 for i in range(12)];outs=[]
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs={ex.submit(fit_one,s):s for s in seeds}
        for fut in as_completed(futs):
            q=fut.result();outs.append(q)
            print("K3B_RESTART_JSON="+json.dumps({"seed":q["seed"],"fit_ll":q["fit_ll"],"val_ll":q["val_ll"]},separators=(",",":")),flush=True)
    outs.sort(key=lambda x:x["val_ll"],reverse=True);top=outs[:5];w=top[0]
    pre_ll,pre_g,_=fb(emissions(Ste,Cte,w["U"],w["Ve"],w["Vr"]),w["A"],w["pi"])
    ll,A,pi,U,Ve,Vr=refit(w);post_ll,post_g,_=fb(emissions(Ste,Cte,U,Ve,Vr),A,pi)
    # comparable exact oracle socket likelihood
    or_ll,or_g,_=fb(emissions(Ste,Cte,Utrue,Vetrue,Vrtrue),Atrue,pitrue)
    ztest=ztruth[NTR:];pre=pre_g.argmax(1);post=post_g.argmax(1);op=or_g.argmax(1)
    pair=[];pred=[]
    for q in top:
        _,g,_=fb(emissions(Ste,Cte,q["U"],q["Ve"],q["Vr"]),q["A"],q["pi"]);pred.append(g.argmax(1))
    for i in range(len(pred)):
        for j in range(i):pair.append(nmi(pred[i],pred[j]))
    out={"phase":"K3b_asym_F1_blind","K":K,"d":D,"rank":R,"entry_strength":ENTRY,"route_strength":ROUTE,
         "restarts":len(outs),"selected_seed":w["seed"],"selected_val_ll":w["val_ll"],
         "pre_refit":{"ll":pre_ll,"nmi":nmi(ztest,pre),"ari":ari(ztest,pre)},
         "post_refit":{"ll":post_ll,"nmi":nmi(ztest,post),"ari":ari(ztest,post),"edge_f1":edge_f1(ztest,post,A)},
         "oracle":{"ll":or_ll,"nmi":nmi(ztest,op),"ari":ari(ztest,op)},
         "top5_pairwise_nmi_median":float(np.median(pair)) if pair else None,
         "gate_nmi70":bool(nmi(ztest,post)>=.70)}
    print("SELECT_FORM_PHASEK3B_JSON="+json.dumps(out,separators=(",",":")),flush=True)
