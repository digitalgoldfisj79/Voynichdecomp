#!/usr/bin/env python3
# Phase K2: full-categorical HMM proposals -> exact rank-2 F0 ENTRY family.
# Synthetic-only. Proposal objective may be richer; final selection/scoring exact F0 only.
# NO P70. No real Voynich inversion.
import json,math,urllib.request
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ProcessPoolExecutor,as_completed

K0URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f/research/select_form_socket_oracle_phaseK0_20261004.py"
k0={"__name__":"k0"};exec(compile(urllib.request.urlopen(K0URL,timeout=60).read().decode(),K0URL,"exec"),k0)

K=8;D=4;RANK=2;STRENGTH=5.;N=4000;SEED=20261004
NFIT=2700;NTR=3200
Atrue,pitrue,Utrue,Vetrue,Vrtrue,ztruth,obs=k0["generate"](K,D,RANK,STRENGTH,N,SEED,"F0")
Y=np.array([k0["PCLASS"][tok[0]] for tok in obs],dtype=np.int16)
Yfit=Y[:NFIT];Yval=Y[NFIT:NTR];Ytr=Y[:NTR];Ytest=Y[NTR:]
BASE=k0["P_START_CLASS"].copy();LOGBASE=np.log(np.maximum(BASE,1e-30))

def fb_counts(y,A,pi,Q):
    n=len(y);la=np.log(np.maximum(A,1e-300));lp=np.log(np.maximum(pi,1e-300));le=np.log(np.maximum(Q[:,y].T,1e-300))
    al=np.empty((n,K));sc=np.empty(n)
    a=lp+le[0];mx=a.max();sc[0]=mx+math.log(np.exp(a-mx).sum());al[0]=a-sc[0]
    for t in range(1,n):
        M=al[t-1][:,None]+la;mm=M.max(0);pr=mm+np.log(np.exp(M-mm).sum(0))
        a=pr+le[t];mx=a.max();sc[t]=mx+math.log(np.exp(a-mx).sum());al[t]=a-sc[t]
    be=np.zeros((n,K))
    for t in range(n-2,-1,-1):
        M=la+le[t+1][None,:]+be[t+1][None,:];mm=M.max(1)
        be[t]=mm+np.log(np.exp(M-mm[:,None]).sum(1))-sc[t+1]
    lg=al+be;mm=lg.max(1,keepdims=True);g=np.exp(lg-mm);g/=g.sum(1,keepdims=True)
    xi=np.zeros((K,K),float)
    for t in range(n-1):
        M=al[t][:,None]+la+le[t+1][None,:]+be[t+1][None,:];mm=M.max();q=np.exp(M-mm);q/=q.sum();xi+=q
    C=np.zeros((K,12),float)
    for c in range(12):C[:,c]=g[y==c].sum(0)
    return float(sc.sum()),g,xi,C

def sparse_A(xi):
    A=np.zeros((K,K),float)
    for i in range(K):
        row=xi[i]+.05;keep=[i]
        for j in np.argsort(row)[::-1]:
            if int(j) not in keep:keep.append(int(j))
            if len(keep)>=D:break
        w=row[keep];w/=w.sum();A[i,keep]=w
    return A

def proposal(seed):
    rng=np.random.default_rng(seed)
    A=rng.gamma(1.,1.,(K,K));A[np.arange(K),np.arange(K)]+=2.;A/=A.sum(1,keepdims=True)
    # Rich categorical emission proposal around the known FORM entry prior.
    B=np.empty((K,12))
    for z in range(K):
        w=BASE*np.exp(rng.normal(0,1.0,12));B[z]=w/w.sum()
    pi=np.ones(K)/K;last=-1e300;best=None
    for stage,nit in (("dense",35),("sparse",25)):
        for it in range(nit):
            ll,g,xi,C=fb_counts(Yfit,A,pi,B)
            if stage=="dense":
                A=xi+.15;A/=A.sum(1,keepdims=True)
            else:A=sparse_A(xi)
            pi=g[0]+.1;pi/=pi.sum()
            B=C+2.0*BASE[None,:];B/=B.sum(1,keepdims=True)
            ll2,g2,xi2,C2=fb_counts(Yfit,A,pi,B)
            if best is None or ll2>best[0]:best=(ll2,A.copy(),pi.copy(),B.copy())
            if it>8 and abs(ll2-last)<1e-4:break
            last=ll2
        if stage=="dense":
            ll,A,pi,B=best;_,_,xi,_=fb_counts(Yfit,A,pi,B);A=sparse_A(xi);last=-1e300
    ll,A,pi,B=best
    vl,_,_,_=fb_counts(Yval,A,pi,B)
    return {"seed":seed,"proposal_fit_ll":float(ll),"proposal_val_ll":float(vl),"A":A,"pi":pi,"B":B}

def project_rank2(B):
    R=np.log(np.maximum(B,1e-12))-LOGBASE[None,:]
    R-=R.mean(1,keepdims=True)
    u,s,vt=np.linalg.svd(R,full_matrices=False)
    U=u[:,:RANK]*np.sqrt(s[:RANK])[None,:]
    V=np.sqrt(s[:RANK])[:,None]*vt[:RANK]
    return U,V

def exact_Q(U,V):
    S=LOGBASE[None,:]+U@V;S-=S.max(1,keepdims=True);Q=np.exp(S);Q/=Q.sum(1,keepdims=True);return Q

def fit_emission(C,U0,V0,l2=.02):
    x0=np.r_[U0.ravel(),V0.ravel()]
    def fg(x):
        U=x[:K*RANK].reshape(K,RANK);V=x[K*RANK:].reshape(RANK,12)
        Q=exact_Q(U,V);nt=C.sum(1,keepdims=True);Dlt=nt*Q-C
        loss=-float(np.sum(C*np.log(np.maximum(Q,1e-300))))+.5*l2*(np.sum(U*U)+np.sum(V*V))
        return loss,np.r_[(Dlt@V.T+l2*U).ravel(),(U.T@Dlt+l2*V).ravel()]
    rr=minimize(lambda x:fg(x),x0,jac=True,method="L-BFGS-B",options={"maxiter":120,"ftol":1e-10})
    x=rr.x;return x[:K*RANK].reshape(K,RANK),x[K*RANK:].reshape(RANK,12)

def exact_refine(y,A,pi,U,V,nit=35):
    best=None;last=-1e300
    for it in range(nit):
        ll,g,xi,C=fb_counts(y,A,pi,exact_Q(U,V))
        A=sparse_A(xi);pi=g[0]+.1;pi/=pi.sum();U,V=fit_emission(C,U,V)
        ll2,g2,xi2,C2=fb_counts(y,A,pi,exact_Q(U,V))
        if best is None or ll2>best[0]:best=(ll2,A.copy(),pi.copy(),U.copy(),V.copy())
        if it>8 and abs(ll2-last)<1e-4:break
        last=ll2
    return best

def exact_score(y,A,pi,U,V):
    ll,g,_,_=fb_counts(y,A,pi,exact_Q(U,V));return float(ll),g

def nmi(a,b):return k0["nmi"](a,b)
def ari(a,b):return k0["ari"](a,b)

def edge_f1(ztrue,zpred,Aest):
    C=np.zeros((K,K),int)
    for a,b in zip(ztrue,zpred):C[int(a),int(b)]+=1
    rr,cc=linear_sum_assignment(-C);mp={int(c):int(r) for r,c in zip(rr,cc)}
    ee={(mp.get(i,i),mp.get(j,j)) for i in range(K) for j in range(K) if Aest[i,j]>0}
    te={(i,j) for i in range(K) for j in range(K) if Atrue[i,j]>0}
    tp=len(ee&te);p=tp/max(len(ee),1);r=tp/max(len(te),1);return 2*p*r/max(p+r,1e-15)

if __name__=="__main__":
    seeds=[SEED+20000+i*193 for i in range(16)]
    props=[]
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(proposal,s):s for s in seeds}
        for fut in as_completed(futs):
            q=fut.result();props.append(q)
            print("K2_PROPOSAL_JSON="+json.dumps({"seed":q["seed"],"fit_ll":q["proposal_fit_ll"],"val_ll":q["proposal_val_ll"]},separators=(",",":")),flush=True)
    # Rich proposal score only narrows candidates; never final scientific selector.
    props.sort(key=lambda x:x["proposal_val_ll"],reverse=True);props=props[:8]
    exact=[]
    for q in props:
        U,V=project_rank2(q["B"])
        ll,A,pi,U,V=exact_refine(Yfit,q["A"].copy(),q["pi"].copy(),U,V,35)
        vl,_=exact_score(Yval,A,pi,U,V)
        rec={"seed":q["seed"],"exact_fit_ll":float(ll),"exact_val_ll":float(vl),"A":A,"pi":pi,"U":U,"V":V}
        exact.append(rec);print("K2_EXACT_JSON="+json.dumps({"seed":rec["seed"],"fit_ll":rec["exact_fit_ll"],"val_ll":rec["exact_val_ll"]},separators=(",",":")),flush=True)
    # Final selection strictly by exact F0 validation likelihood.
    exact.sort(key=lambda x:x["exact_val_ll"],reverse=True);winner=exact[0]
    pre_ll,pre_g=exact_score(Ytest,winner["A"],winner["pi"],winner["U"],winner["V"])
    ll,A,pi,U,V=exact_refine(Ytr,winner["A"].copy(),winner["pi"].copy(),winner["U"].copy(),winner["V"].copy(),45)
    post_ll,post_g=exact_score(Ytest,A,pi,U,V)
    # Comparable oracle ENTRY-only likelihood.
    Qtrue=k0["row_soft"]
    E=np.empty((len(Ytest),K),float)
    for x in range(K):
        q=Qtrue(BASE,Utrue[x]@Vetrue,np.ones(12,bool));E[:,x]=np.log(np.maximum(q[Ytest],1e-300))
    or_ll,or_g=k0["fb"](E,Atrue,pitrue)
    # reveal truth only now
    ztest=ztruth[NTR:];pre=pre_g.argmax(1);post=post_g.argmax(1);op=or_g.argmax(1)
    pair=[]
    pred=[]
    for r in exact[:5]:
        _,g=exact_score(Ytest,r["A"],r["pi"],r["U"],r["V"]);pred.append(g.argmax(1))
    for i in range(len(pred)):
        for j in range(i):pair.append(nmi(pred[i],pred[j]))
    out={"phase":"K2_F0_categorical_proposal","K":K,"d":D,"rank":RANK,"strength":STRENGTH,
         "proposal_starts":16,"exact_finalists":len(exact),"selected_seed":winner["seed"],
         "pre_refit":{"ll":pre_ll,"nmi":nmi(ztest,pre),"ari":ari(ztest,pre)},
         "post_refit":{"ll":post_ll,"nmi":nmi(ztest,post),"ari":ari(ztest,post),"edge_f1":edge_f1(ztest,post,A)},
         "oracle":{"ll":or_ll,"nmi":nmi(ztest,op),"ari":ari(ztest,op)},
         "top5_pairwise_nmi_median":float(np.median(pair)) if pair else None,
         "gate_nmi70":bool(nmi(ztest,post)>=.70)}
    print("SELECT_FORM_PHASEK2_JSON="+json.dumps(out,separators=(",",":")),flush=True)
