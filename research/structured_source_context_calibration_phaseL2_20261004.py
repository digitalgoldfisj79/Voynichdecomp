#!/usr/bin/env python3
# Phase L2: corrected structured-source calibration.
# Current source items collide at SELECT output; structured sources differ in context.
# Uses full exact per-signature FORM token likelihood. Synthetic-only. NO P70.
import json,math,urllib.request
import numpy as np
from sklearn.metrics import normalized_mutual_info_score,roc_auc_score
from concurrent.futures import ProcessPoolExecutor,as_completed

K0URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f/research/select_form_socket_oracle_phaseK0_20261004.py"
k0={"__name__":"k0"};exec(compile(urllib.request.urlopen(K0URL,timeout=60).read().decode(),K0URL,"exec"),k0)

V=16;NSIG=8;NSEC=4;SECLEN=1200
NFIT=700;NVAL=200;NTEST=300
ENTRY=5.;ROUTE=.5
SIG_OF=np.arange(V)%NSIG

def state_prior(sec):
    # Explicit section/register conditioning at source and SELECT-signature level.
    w=np.ones(V,float)
    fav={(2*sec)%8,(2*sec+1)%8}
    for v in range(V):
        if SIG_OF[v] in fav:w[v]*=4.0
        # small lexical skew within signature, shared across sections
        w[v]*=1.0/(1.0+0.04*v)
    return w/w.sum()

def add_sig_mass(row,sig,mass,branch_pref):
    a=sig;b=sig+8
    if branch_pref==0:
        row[a]+=mass*.82;row[b]+=mass*.18
    else:
        row[a]+=mass*.18;row[b]+=mass*.82

def A_family(fam,sec):
    prior=state_prior(sec)
    A=np.zeros((V,V),float)
    if fam=="TABLE":
        A[:]=prior
        return A
    for v in range(V):
        s=v%8;br=v//8
        row=.30*prior
        if fam=="LANG":
            # Same current SELECT signature, different lexical context by alias branch.
            if br==0:
                p1=(2*s+1+sec)%8;p2=(3*s+2+sec)%8;p3=(s+5)%8
            else:
                p1=(2*s+4+sec)%8;p2=(5*s+3+sec)%8;p3=(s+2)%8
            add_sig_mass(row,p1,.32,br)
            add_sig_mass(row,p2,.22,1-br)
            add_sig_mass(row,p3,.10,br)
            row[v]+=.06
        elif fam=="NOTATION":
            # Two distinct motif roles share each visible control signature.
            if br==0:
                p1=(s+1)%8;p2=(s+4+sec)%8
            else:
                p1=(s+3)%8;p2=(s+6+sec)%8
            add_sig_mass(row,p1,.52,br)
            add_sig_mass(row,p2,.12,br)
            # rare branch switch preserves ambiguity but context is strongly diagnostic
            add_sig_mass(row,s,.04,1-br)
        else:raise ValueError(fam)
        row+=1e-5
        A[v]=row/row.sum()
    return A

def source_sequence(fam,sec,n,rng):
    A=A_family(fam,sec);pi=state_prior(sec)
    z=np.empty(n,int);z[0]=rng.choice(V,p=pi)
    for t in range(1,n):z[t]=rng.choice(V,p=A[z[t-1]])
    return z,A,pi

def encoder(seed):
    rng=np.random.default_rng(seed+44771)
    U,Ve0,Vr0=k0["controls"](NSIG,2,1.,rng,"F1")
    return U,Ve0*ENTRY,Vr0*ROUTE

def render_and_ll(z,U,Ve,Vr,rng):
    obs=[];L=[]
    for v in z:
        sig=int(SIG_OF[v])
        for _ in range(100):
            tok=k0["gen_token"](sig,U,Ve,Vr,rng)
            if tok is not None:break
        else:raise RuntimeError("nontermination")
        obs.append(tok)
        q=np.array([k0["token_loglik"](tok,s,U,Ve,Vr) for s in range(NSIG)],float)
        q-=q.max();L.append(q)
    return obs,np.stack(L)

def E_control(z):
    sig=SIG_OF[z];E=np.full((len(z),V),-80.,float)
    for v in range(V):E[:,v]=np.where(sig==SIG_OF[v],0.,-80.)
    return E

def E_form(Lsig):
    return Lsig[:,SIG_OF]

def logsumexp(a,axis=None):
    m=np.max(a,axis=axis,keepdims=True);v=m+np.log(np.sum(np.exp(a-m),axis=axis,keepdims=True))
    return np.squeeze(v,axis=axis)

def fb(E,A,pi,want_xi=True):
    n,K=E.shape;la=np.log(np.maximum(A,1e-300));lp=np.log(np.maximum(pi,1e-300))
    al=np.empty((n,K));sc=np.empty(n)
    a=lp+E[0];sc[0]=logsumexp(a);al[0]=a-sc[0]
    for t in range(1,n):
        pr=logsumexp(al[t-1][:,None]+la,axis=0)
        a=pr+E[t];sc[t]=logsumexp(a);al[t]=a-sc[t]
    be=np.zeros((n,K))
    for t in range(n-2,-1,-1):
        be[t]=logsumexp(la+E[t+1][None,:]+be[t+1][None,:],axis=1)-sc[t+1]
    lg=al+be;lg-=logsumexp(lg,axis=1)[:,None];g=np.exp(lg)
    xi=None
    if want_xi:
        xi=np.zeros((K,K),float)
        for t in range(n-1):
            M=al[t][:,None]+la+E[t+1][None,:]+be[t+1][None,:]
            M-=logsumexp(M);xi+=np.exp(M)
    return float(sc.sum()),g,xi

def init_A(seed):
    rng=np.random.default_rng(seed)
    A=rng.gamma(1.,1.,(V,V));A[np.arange(V),np.arange(V)]+=1.
    A/=A.sum(1,keepdims=True)
    return A,np.ones(V)/V

def em_fit(E,seed,iters=70,A0=None,pi0=None):
    A,pi=(init_A(seed) if A0 is None else (A0.copy(),pi0.copy()))
    best=None;last=-1e300
    for it in range(iters):
        ll,g,xi=fb(E,A,pi,True)
        A=(xi+.05);A/=A.sum(1,keepdims=True)
        pi=g[0]+.05;pi/=pi.sum()
        ll2,_,_=fb(E,A,pi,False)
        if best is None or ll2>best[0]:best=(ll2,A.copy(),pi.copy(),g[-1].copy())
        if it>10 and abs(ll2-last)<1e-5:break
        last=ll2
    return best

def fit_select(Efit,Eval,seed,restarts=12):
    out=[]
    for r in range(restarts):
        q=em_fit(Efit,seed+137*r)
        ll,A,pi,g_end=q
        pval=g_end@A;pval/=pval.sum()
        lv,_,_=fb(Eval,A,pval,False)
        out.append((lv,A,pi))
    out.sort(key=lambda x:x[0],reverse=True)
    return out

def refit(Etr,A,pi,seed):
    return em_fit(Etr,seed,iters=80,A0=A,pi0=pi)

def collision_nmi(z,pred):
    vals=[]
    for s in range(NSIG):
        ix=np.where(SIG_OF[z]==s)[0]
        if len(ix)>4 and len(np.unique(z[ix]))>1:
            vals.append(normalized_mutual_info_score(z[ix],pred[ix]))
    return float(np.mean(vals)) if vals else float("nan")

def pair_auc(z,g,seed):
    rng=np.random.default_rng(seed);ys=[];scores=[]
    for s in range(NSIG):
        ix=np.where(SIG_OF[z]==s)[0]
        if len(ix)<3:continue
        for _ in range(1000):
            a,b=rng.choice(ix,2,replace=False)
            ys.append(int(z[a]==z[b]));scores.append(float(np.dot(g[a],g[b])))
    return float(roc_auc_score(ys,scores)) if len(set(ys))>1 else float("nan")

def metrics(z,g,seed):
    pred=g.argmax(1)
    return {"source_nmi":float(normalized_mutual_info_score(z,pred)),
            "collision_nmi":collision_nmi(z,pred),
            "same_source_auc":pair_auc(z,g,seed),
            "coarse_signature_nmi":float(normalized_mutual_info_score(z,SIG_OF[z]))}

def one_section(fam,sec,seed,U,Ve,Vr,rng):
    z,Atrue,pitrue=source_sequence(fam,sec,SECLEN,rng)
    obs,Lsig=render_and_ll(z,U,Ve,Vr,rng)
    channels={"CONTROL":E_control(z),"FORM":E_form(Lsig)}
    rec={}
    for ci,(name,E) in enumerate(channels.items()):
        ef=E[:NFIT];ev=E[NFIT:NFIT+NVAL];etr=E[:NFIT+NVAL];ete=E[NFIT+NVAL:]
        cand=fit_select(ef,ev,seed+sec*10000+ci*500000,12)
        lv,A,pi=cand[0]
        q=refit(etr,A,pi,seed+sec*17000+ci*700000)
        _,Ar,pir,g_end=q
        ptest=g_end@Ar;ptest/=ptest.sum()
        ll,g,_=fb(ete,Ar,ptest,False)
        # true-transition oracle on identical observation channel
        # run through full prefix to carry the correct contextual prior into test.
        _,go,_=fb(E,Atrue,pitrue,False)
        sl=slice(NFIT+NVAL,None)
        rec[name]={"selected_val_ll":float(lv),"test_ll":ll,
                   "blind":metrics(z[sl],g,seed+ci+sec*11),
                   "oracle":metrics(z[sl],go[sl],seed+ci+sec*13)}
    # exact SELECT signature classification from renderer likelihood
    ds=Lsig.argmax(1)
    return z,Lsig,{"section":sec,"sig_decode_acc":float(np.mean(ds==SIG_OF[z])),**rec}

def one_dataset(fam,seed):
    rng=np.random.default_rng(seed);U,Ve,Vr=encoder(seed)
    secs=[];allz=[];allsec=[];allsig=[];alldec=[]
    for sec in range(NSEC):
        z,L,r=one_section(fam,sec,seed,U,Ve,Vr,rng);secs.append(r)
        allz.extend(z.tolist());allsec.extend([sec]*len(z));allsig.extend(SIG_OF[z].tolist());alldec.extend(L.argmax(1).tolist())
    def agg(ch,kind,key):
        return float(np.nanmean([x[ch][kind][key] for x in secs]))
    out={"family":fam,"seed":seed,
         "source_section_nmi":float(normalized_mutual_info_score(allsec,allz)),
         "signature_section_nmi":float(normalized_mutual_info_score(allsec,allsig)),
         "decoded_signature_section_nmi":float(normalized_mutual_info_score(allsec,alldec)),
         "sig_decode_acc":float(np.mean(np.array(allsig)==np.array(alldec))),
         "control":{"blind_source_nmi":agg("CONTROL","blind","source_nmi"),
                    "blind_collision_nmi":agg("CONTROL","blind","collision_nmi"),
                    "blind_auc":agg("CONTROL","blind","same_source_auc"),
                    "oracle_source_nmi":agg("CONTROL","oracle","source_nmi"),
                    "oracle_collision_nmi":agg("CONTROL","oracle","collision_nmi"),
                    "oracle_auc":agg("CONTROL","oracle","same_source_auc")},
         "form":{"blind_source_nmi":agg("FORM","blind","source_nmi"),
                 "blind_collision_nmi":agg("FORM","blind","collision_nmi"),
                 "blind_auc":agg("FORM","blind","same_source_auc"),
                 "oracle_source_nmi":agg("FORM","oracle","source_nmi"),
                 "oracle_collision_nmi":agg("FORM","oracle","collision_nmi"),
                 "oracle_auc":agg("FORM","oracle","same_source_auc")},
         "sections":secs}
    return out

if __name__=="__main__":
    specs=[(f,s) for f in ("LANG","NOTATION","TABLE") for s in (20262201,20262202,20262203)]
    out=[]
    with ProcessPoolExecutor(max_workers=9) as ex:
        fut={ex.submit(one_dataset,*x):x for x in specs}
        for q in as_completed(fut):
            r=q.result();out.append(r)
            print("STRUCTURED_SOURCE_L2_REP_JSON="+json.dumps(r,separators=(",",":")),flush=True)
    summary={}
    for fam in ("LANG","NOTATION","TABLE"):
        rr=[x for x in out if x["family"]==fam]
        def med1(k):return float(np.median([x[k] for x in rr]))
        def med2(ch,k):return float(np.median([x[ch][k] for x in rr]))
        summary[fam]={"n":len(rr),"source_section_nmi":med1("source_section_nmi"),
          "signature_section_nmi":med1("signature_section_nmi"),"sig_decode_acc":med1("sig_decode_acc"),
          "control_blind_collision_nmi":med2("control","blind_collision_nmi"),
          "control_oracle_collision_nmi":med2("control","oracle_collision_nmi"),
          "form_blind_collision_nmi":med2("form","blind_collision_nmi"),
          "form_oracle_collision_nmi":med2("form","oracle_collision_nmi"),
          "control_blind_auc":med2("control","blind_auc"),"control_oracle_auc":med2("control","oracle_auc"),
          "form_blind_auc":med2("form","blind_auc"),"form_oracle_auc":med2("form","oracle_auc"),
          "form_blind_source_nmi":med2("form","blind_source_nmi"),
          "form_oracle_source_nmi":med2("form","oracle_source_nmi")}
    print("STRUCTURED_SOURCE_PHASEL2_JSON="+json.dumps({"summary":summary,"replicates":out},separators=(",",":")),flush=True)
