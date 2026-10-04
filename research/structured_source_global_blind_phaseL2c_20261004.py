#!/usr/bin/env python3
# Phase L2c: global blind structured-source recovery on corrected L2 generator.
# Blind contextual initializers -> exact full-sequence fixed-emission HMM EM.
# Synthetic-only. NO P70.
import json,math,urllib.request,numpy as np
from numba import njit
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score,roc_auc_score
from concurrent.futures import ProcessPoolExecutor,as_completed

URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/c6384a58449b0f27f959be580c5e9613d71d83eb/research/structured_source_context_calibration_phaseL2_20261004.py"
m={"__name__":"l2lib"};exec(compile(urllib.request.urlopen(URL,timeout=60).read().decode(),URL,"exec"),m)
V=m["V"];NSIG=m["NSIG"];SIG_OF=m["SIG_OF"];NFIT=m["NFIT"];NVAL=m["NVAL"];NSEC=m["NSEC"];SECLEN=m["SECLEN"]

@njit
def fb(E,A,pi,want_xi=True):
    n,K=E.shape
    B=np.empty((n,K));shift=np.empty(n)
    for t in range(n):
        mx=E[t,0]
        for k in range(1,K):
            if E[t,k]>mx:mx=E[t,k]
        shift[t]=mx
        for k in range(K):B[t,k]=math.exp(E[t,k]-mx)
    al=np.empty((n,K));c=np.empty(n)
    s=0.
    for k in range(K):al[0,k]=pi[k]*B[0,k];s+=al[0,k]
    c[0]=max(s,1e-300)
    for k in range(K):al[0,k]/=c[0]
    for t in range(1,n):
        s=0.
        for j in range(K):
            q=0.
            for i in range(K):q+=al[t-1,i]*A[i,j]
            al[t,j]=q*B[t,j];s+=al[t,j]
        c[t]=max(s,1e-300)
        for j in range(K):al[t,j]/=c[t]
    be=np.ones((n,K))
    for t in range(n-2,-1,-1):
        for i in range(K):
            q=0.
            for j in range(K):q+=A[i,j]*B[t+1,j]*be[t+1,j]
            be[t,i]=q/c[t+1]
    g=np.empty((n,K))
    for t in range(n):
        s=0.
        for k in range(K):g[t,k]=al[t,k]*be[t,k];s+=g[t,k]
        s=max(s,1e-300)
        for k in range(K):g[t,k]/=s
    xi=np.zeros((K,K))
    if want_xi:
        for t in range(n-1):
            den=0.
            for i in range(K):
                for j in range(K):den+=al[t,i]*A[i,j]*B[t+1,j]*be[t+1,j]
            den=max(den,1e-300)
            for i in range(K):
                for j in range(K):xi[i,j]+=al[t,i]*A[i,j]*B[t+1,j]*be[t+1,j]/den
    ll=0.
    for t in range(n):ll+=math.log(c[t])+shift[t]
    return ll,g,xi

@njit
def em(E,A,pi,iters,alpha):
    last=-1e300
    for it in range(iters):
        ll,g,xi=fb(E,A,pi,True)
        for i in range(V):
            s=0.
            for j in range(V):
                A[i,j]=xi[i,j]+alpha;s+=A[i,j]
            for j in range(V):A[i,j]/=s
        s=0.
        for k in range(V):
            pi[k]=g[0,k]+.05;s+=pi[k]
        for k in range(V):pi[k]/=s
        if it>10 and abs(ll-last)<1e-6:break
        last=ll
    return A,pi

def softmax_rows(L):
    X=L-L.max(1,keepdims=True);Q=np.exp(X);Q/=Q.sum(1,keepdims=True);return Q

def context_features(Q,r):
    n,d=Q.shape;offs=list(range(-r,0))+list(range(1,r+1));F=np.zeros((n,len(offs)*d))
    for k,off in enumerate(offs):
        if off<0:F[-off:,k*d:(k+1)*d]=Q[:n+off]
        else:F[:n-off,k*d:(k+1)*d]=Q[off:]
    return F

def init_context(E,seed,radius):
    # derive blind signature posterior from the 8 unique emission columns
    Es=np.stack([E[:,s] for s in range(NSIG)],axis=1)
    Q=softmax_rows(Es);cur=Q.argmax(1);F=context_features(Q,radius)
    labels=np.empty(len(E),int)
    for s in range(NSIG):
        ix=np.where(cur==s)[0]
        if len(ix)<10:
            labels[ix]=s
            continue
        km=KMeans(n_clusters=2,n_init=8,random_state=seed+s,max_iter=300).fit(F[ix])
        labels[ix]=s+NSIG*km.labels_
    A=np.full((V,V),.15,float)
    for a,b in zip(labels[:-1],labels[1:]):A[a,b]+=1
    A/=A.sum(1,keepdims=True)
    pi=np.bincount(labels[:min(100,len(labels))],minlength=V).astype(float)+.1;pi/=pi.sum()
    return A,pi

def init_random(seed):
    rng=np.random.default_rng(seed);A=rng.gamma(1.,1.,(V,V));A[np.arange(V),np.arange(V)]+=1.;A/=A.sum(1,keepdims=True);pi=rng.dirichlet(np.ones(V));return A,pi

def fit_select(Efit,Eval,seed):
    inits=[]
    for r in (1,2,3,4):
        for rep in range(3):
            inits.append(init_context(Efit,seed+1000*r+37*rep,r))
    for rep in range(6):inits.append(init_random(seed+90000+rep*137))
    cand=[]
    for ii,(A,pi) in enumerate(inits):
        A,pi=em(Efit,A.copy(),pi.copy(),90,.05)
        lv=fb(Eval,A,pi,False)[0]
        cand.append((float(lv),ii,A.copy(),pi.copy()))
    cand.sort(key=lambda x:x[0],reverse=True)
    return cand

def collision_nmi(z,pred):
    vals=[]
    for s in range(NSIG):
        ix=np.where(SIG_OF[z]==s)[0]
        if len(ix)>4 and len(np.unique(z[ix]))>1:vals.append(normalized_mutual_info_score(z[ix],pred[ix]))
    return float(np.mean(vals)) if vals else float("nan")

def pair_auc(z,g,seed):
    rng=np.random.default_rng(seed);ys=[];sc=[]
    for s in range(NSIG):
        ix=np.where(SIG_OF[z]==s)[0]
        if len(ix)<3:continue
        for _ in range(800):
            a,b=rng.choice(ix,2,replace=False);ys.append(int(z[a]==z[b]));sc.append(float(np.dot(g[a],g[b])))
    return float(roc_auc_score(ys,sc)) if len(set(ys))>1 else float("nan")

def metrics(z,g,seed):
    p=g.argmax(1)
    return {"source_nmi":float(normalized_mutual_info_score(z,p)),
            "collision_nmi":collision_nmi(z,p),
            "same_source_auc":pair_auc(z,g,seed)}

def one_section(fam,sec,seed,U,Ve,Vr,rng):
    z,At,pt=m["source_sequence"](fam,sec,SECLEN,rng)
    obs,L=m["render_and_ll"](z,U,Ve,Vr,rng)
    channels={"CONTROL":m["E_control"](z),"FORM":m["E_form"](L)}
    out={"section":sec,"sig_decode_acc":float(np.mean(L.argmax(1)==SIG_OF[z]))}
    for ci,(name,E) in enumerate(channels.items()):
        ef=E[:NFIT];ev=E[NFIT:NFIT+NVAL];etr=E[:NFIT+NVAL];ete=E[NFIT+NVAL:];zt=z[NFIT+NVAL:]
        cand=fit_select(ef,ev,seed+sec*10000+ci*500000)
        lv,ii,A,pi=cand[0]
        A,pi=em(etr,A.copy(),pi.copy(),100,.05)
        ll,g,_=fb(ete,A,pi,False)
        # oracle on full prefix so test prior is contextual
        _,go,_=fb(E,At,pt,False)
        out[name]={"selected_val_ll":lv,"selected_init":ii,"test_ll":float(ll),
                   "blind":metrics(zt,g,seed+sec*31+ci),
                   "oracle":metrics(zt,go[NFIT+NVAL:],seed+sec*41+ci),
                   "top3_val_gap":float(cand[0][0]-cand[min(2,len(cand)-1)][0])}
    return z,L,out

def one_dataset(fam,seed):
    rng=np.random.default_rng(seed);U,Ve,Vr=m["encoder"](seed)
    secs=[];alls=[];alld=[];allsec=[];allz=[]
    for sec in range(NSEC):
        z,L,r=one_section(fam,sec,seed,U,Ve,Vr,rng);secs.append(r)
        allz.extend(z.tolist());alls.extend(SIG_OF[z].tolist());alld.extend(L.argmax(1).tolist());allsec.extend([sec]*len(z))
    def agg(ch,kind,key):return float(np.nanmean([x[ch][kind][key] for x in secs]))
    return {"family":fam,"seed":seed,
      "source_section_nmi":float(normalized_mutual_info_score(allsec,allz)),
      "signature_section_nmi":float(normalized_mutual_info_score(allsec,alls)),
      "decoded_signature_section_nmi":float(normalized_mutual_info_score(allsec,alld)),
      "sig_decode_acc":float(np.mean(np.array(alls)==np.array(alld))),
      "control":{"blind_collision_nmi":agg("CONTROL","blind","collision_nmi"),"blind_auc":agg("CONTROL","blind","same_source_auc"),"oracle_collision_nmi":agg("CONTROL","oracle","collision_nmi"),"oracle_auc":agg("CONTROL","oracle","same_source_auc")},
      "form":{"blind_collision_nmi":agg("FORM","blind","collision_nmi"),"blind_auc":agg("FORM","blind","same_source_auc"),"oracle_collision_nmi":agg("FORM","oracle","collision_nmi"),"oracle_auc":agg("FORM","oracle","same_source_auc")},
      "sections":secs}

if __name__=="__main__":
    specs=[(f,s) for f in ("LANG","NOTATION","TABLE") for s in (20262601,20262602,20262603)]
    out=[]
    with ProcessPoolExecutor(max_workers=9) as ex:
        fut={ex.submit(one_dataset,*x):x for x in specs}
        for q in as_completed(fut):
            r=q.result();out.append(r);print("L2C_REP_JSON="+json.dumps(r,separators=(",",":")),flush=True)
    summary={}
    for fam in ("LANG","NOTATION","TABLE"):
        rr=[x for x in out if x["family"]==fam]
        med=lambda ch,k:float(np.median([x[ch][k] for x in rr]))
        summary[fam]={"n":len(rr),
          "source_section_nmi":float(np.median([x["source_section_nmi"] for x in rr])),
          "signature_section_nmi":float(np.median([x["signature_section_nmi"] for x in rr])),
          "sig_decode_acc":float(np.median([x["sig_decode_acc"] for x in rr])),
          "control_blind_collision_nmi":med("control","blind_collision_nmi"),"control_blind_auc":med("control","blind_auc"),
          "control_oracle_collision_nmi":med("control","oracle_collision_nmi"),"control_oracle_auc":med("control","oracle_auc"),
          "form_blind_collision_nmi":med("form","blind_collision_nmi"),"form_blind_auc":med("form","blind_auc"),
          "form_oracle_collision_nmi":med("form","oracle_collision_nmi"),"form_oracle_auc":med("form","oracle_auc")}
    print("STRUCTURED_SOURCE_PHASEL2C_JSON="+json.dumps({"summary":summary,"replicates":out},separators=(",",":")),flush=True)
