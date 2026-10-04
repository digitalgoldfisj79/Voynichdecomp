#!/usr/bin/env python3
# Phase L1: corrected structured-source calibration through frozen Mauro/state-separated FORM.
# Paired source items share instantaneous SELECT signature but differ in FUTURE observable signature behavior.
# Synthetic-only. NO P70. No semantic decoding.
import json,math,urllib.request
import numpy as np
from sklearn.metrics import normalized_mutual_info_score,roc_auc_score

K0URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f/research/select_form_socket_oracle_phaseK0_20261004.py"
k0={"__name__":"k0"};exec(compile(urllib.request.urlopen(K0URL,timeout=60).read().decode(),K0URL,"exec"),k0)

V=16;NSIG=8;NSEC=4;SECLEN=1600
NFIT=900;NVAL=250;NTEST=450
ENTRY=5.0;ROUTE=.5
SIG_OF=np.arange(V)%NSIG
EPS_EM=1e-3

def sig_prior(sec):
    w=np.ones(NSIG,float)
    fav=[(2*sec)%NSIG,(2*sec+1)%NSIG]
    w[fav]*=math.exp(1.45)
    return w/w.sum()

def section_prior(sec):
    ps=sig_prior(sec);p=np.zeros(V,float)
    # modest section-conditioned within-signature lexical skew, never exclusive.
    for s in range(NSIG):
        vh=.58 if (s+sec)%2==0 else .42
        p[s]=ps[s]*(1-vh);p[s+NSIG]=ps[s]*vh
    return p/p.sum()

def distribute_sig(qsig,h,sec,persist):
    row=np.zeros(V,float)
    for s,q in enumerate(qsig):
        # target hidden-variant persistence is source-side syntax, not SELECT.
        same=min(.95,max(.05,persist + (.04 if (s+sec)%2==h else -.04)))
        row[s + h*NSIG]+=q*same
        row[s + (1-h)*NSIG]+=q*(1-same)
    return row/row.sum()

def A_family(fam,sec):
    ps=sig_prior(sec);A=np.zeros((V,V),float)
    if fam=="TABLE":
        A[:]=section_prior(sec)
        return A
    for i in range(V):
        s=i%NSIG;h=i//NSIG
        if fam=="LANG":
            q=.55*ps
            if h==0:
                prefs=[((s+1+sec)%8,.20),((s+3)%8,.12),((s+6)%8,.08), (s,.05)]
            else:
                prefs=[((s+2+sec)%8,.20),((s+5)%8,.12),((s+7)%8,.08), (s,.05)]
            for j,w in prefs:q[j]+=w
            q/=q.sum()
            A[i]=distribute_sig(q,h,sec,.68)
        elif fam=="NOTATION":
            q=.50*ps
            if h==0:
                prefs=[((s+1)%8,.335),((s+3+sec)%8,.10),(s,.065)]
            else:
                prefs=[((s+2)%8,.335),((s+6+sec)%8,.10),(s,.065)]
            for j,w in prefs:q[j]+=w
            q/=q.sum()
            A[i]=distribute_sig(q,h,sec,.84)
        else:raise ValueError(fam)
    return A

def source_sequence(fam,sec,n,rng):
    A=A_family(fam,sec);pi=section_prior(sec)
    z=np.empty(n,int);z[0]=rng.choice(V,p=pi)
    for t in range(1,n):z[t]=rng.choice(V,p=A[z[t-1]])
    return z,A,pi

def encoder(seed):
    rng=np.random.default_rng(seed+77113)
    U,Ve0,Vr0=k0["controls"](NSIG,2,1.,rng,"F1")
    return U,Ve0*ENTRY,Vr0*ROUTE

def render(z,U,Ve,Vr,rng):
    obs=[];sig=SIG_OF[z]
    for s in sig:
        for _ in range(100):
            tok=k0["gen_token"](int(s),U,Ve,Vr,rng)
            if tok is not None:obs.append(tok);break
        else:raise RuntimeError("nontermination")
    return obs,sig

def control_E(sig):
    n=len(sig);E=np.full((n,V),math.log(EPS_EM/(NSIG-1)),float)
    hit=math.log(1-EPS_EM)
    for v in range(V):E[:,v]=np.where(sig==SIG_OF[v],hit,E[:,v])
    return E

def form_E(obs,U,Ve,Vr):
    # state pairs share exactly the same SELECT emission; only context can split them.
    ES=np.empty((len(obs),NSIG),float)
    for t,tok in enumerate(obs):
        for s in range(NSIG):ES[t,s]=k0["token_loglik"](tok,s,U,Ve,Vr)
    return ES[:,SIG_OF]

def fb(E,A,pi,want_xi=True):
    n,K=E.shape;la=np.log(np.maximum(A,1e-300));lp=np.log(np.maximum(pi,1e-300))
    al=np.empty((n,K));sc=np.empty(n)
    x=lp+E[0];mx=x.max();sc[0]=mx+math.log(np.exp(x-mx).sum());al[0]=x-sc[0]
    for t in range(1,n):
        M=al[t-1][:,None]+la;mm=M.max(0);pr=mm+np.log(np.exp(M-mm).sum(0))
        x=pr+E[t];mx=x.max();sc[t]=mx+math.log(np.exp(x-mx).sum());al[t]=x-sc[t]
    be=np.zeros((n,K))
    for t in range(n-2,-1,-1):
        M=la+E[t+1][None,:]+be[t+1][None,:];mm=M.max(1)
        be[t]=mm+np.log(np.exp(M-mm[:,None]).sum(1))-sc[t+1]
    lg=al+be;mm=lg.max(1,keepdims=True);g=np.exp(lg-mm);g/=g.sum(1,keepdims=True)
    xi=None
    if want_xi:
        xi=np.zeros((K,K),float)
        for t in range(n-1):
            M=al[t][:,None]+la+E[t+1][None,:]+be[t+1][None,:]
            mm=M.max();q=np.exp(M-mm);q/=q.sum();xi+=q
    return float(sc.sum()),g,xi

def viterbi(E,A,pi):
    n,K=E.shape;la=np.log(np.maximum(A,1e-300));dp=np.log(np.maximum(pi,1e-300))+E[0]
    back=np.zeros((n,K),np.int16)
    for t in range(1,n):
        M=dp[:,None]+la;back[t]=M.argmax(0);dp=M.max(0)+E[t]
    z=np.empty(n,np.int16);z[-1]=dp.argmax()
    for t in range(n-2,-1,-1):z[t]=back[t+1,z[t+1]]
    return z.astype(int)

def fit_blind(Efit,Eval,seed,restarts=3):
    best=None
    for rr in range(restarts):
        rng=np.random.default_rng(seed+rr*173)
        # preserve emission grouping but make transition hypotheses completely blind.
        A=rng.gamma(1.0,1.0,(V,V));A[np.arange(V),np.arange(V)]+=1.0;A/=A.sum(1,keepdims=True)
        pi=rng.dirichlet(np.ones(V))
        last=-1e300
        for it in range(100):
            ll,g,xi=fb(Efit,A,pi,True)
            A=xi+.15;A/=A.sum(1,keepdims=True)
            pi=g[0]+.1;pi/=pi.sum()
            ll2,_,_=fb(Efit,A,pi,False)
            if it>12 and abs(ll2-last)<1e-5:break
            last=ll2
        lv,_,_=fb(Eval,A,pi,False)
        if best is None or lv>best[0]:best=(lv,A.copy(),pi.copy())
    return best

def collision_nmi(z,pred):
    vals=[]
    for s in range(NSIG):
        ix=np.where(SIG_OF[z]==s)[0]
        if len(ix)>10:vals.append(normalized_mutual_info_score(z[ix],pred[ix]))
    return float(np.mean(vals))

def same_auc(z,g,seed):
    rng=np.random.default_rng(seed);ys=[];sc=[]
    for s in range(NSIG):
        ix=np.where(SIG_OF[z]==s)[0]
        if len(ix)<8:continue
        for _ in range(1200):
            a,b=rng.choice(ix,2,replace=False)
            ys.append(int(z[a]==z[b]));sc.append(float(np.dot(g[a],g[b])))
    return float(roc_auc_score(ys,sc))

def eval_model(E,z,A,pi,seed):
    ll,g,_=fb(E,A,pi,False);p=viterbi(E,A,pi)
    return {"ll":ll,"source_nmi":float(normalized_mutual_info_score(z,p)),
            "collision_nmi":collision_nmi(z,p),
            "same_source_auc":same_auc(z,g,seed)}

def one(fam,seed):
    rng=np.random.default_rng(seed);U,Ve,Vr=encoder(seed)
    secout=[];allsec=[];allsig=[];all_form_sig=[]
    for sec in range(NSEC):
        z,Atrue,pitrue=source_sequence(fam,sec,SECLEN,rng)
        obs,sig=render(z,U,Ve,Vr,rng)
        EC=control_E(sig);EF=form_E(obs,U,Ve,Vr)
        # posterior signature prediction from raw FORM emission alone
        S=EF[:,:NSIG].copy() # first eight source states correspond signatures 0..7
        Sm=S-S.max(1,keepdims=True);Q=np.exp(Sm);Q/=Q.sum(1,keepdims=True)
        fsig=Q.argmax(1)
        allsec.extend([sec]*SECLEN);allsig.extend(sig.tolist());all_form_sig.extend(fsig.tolist())
        for ch,E in (("CONTROL",EC),("FORM",EF)):
            b=fit_blind(E[:NFIT],E[NFIT:NFIT+NVAL],seed+sec*1000+(0 if ch=="CONTROL" else 50000))
            lv,A,pi=b
            test=eval_model(E[NFIT+NVAL:],z[NFIT+NVAL:],A,pi,seed+sec*41)
            oracle=eval_model(E[NFIT+NVAL:],z[NFIT+NVAL:],Atrue,pitrue,seed+sec*43)
            secout.append({"section":sec,"channel":ch,"val_ll":lv,"blind":test,"oracle":oracle})
    def ag(ch,which,key):
        return float(np.median([x[which][key] for x in secout if x["channel"]==ch]))
    secarr=np.array(allsec);sigarr=np.array(allsig);farr=np.array(all_form_sig)
    return {
      "family":fam,"seed":seed,
      "section_mi_signature":float(normalized_mutual_info_score(secarr,sigarr)),
      "section_mi_form_signature":float(normalized_mutual_info_score(secarr,farr)),
      "form_signature_accuracy":float(np.mean(sigarr==farr)),
      "control":{"blind_collision_nmi":ag("CONTROL","blind","collision_nmi"),
                 "blind_same_auc":ag("CONTROL","blind","same_source_auc"),
                 "oracle_collision_nmi":ag("CONTROL","oracle","collision_nmi"),
                 "oracle_same_auc":ag("CONTROL","oracle","same_source_auc")},
      "form":{"blind_collision_nmi":ag("FORM","blind","collision_nmi"),
              "blind_same_auc":ag("FORM","blind","same_source_auc"),
              "oracle_collision_nmi":ag("FORM","oracle","collision_nmi"),
              "oracle_same_auc":ag("FORM","oracle","same_source_auc")},
      "sections":secout
    }

if __name__=="__main__":
    out=[]
    for fam,seed in [("LANG",20264001),("NOTATION",20264001),("TABLE",20264001)]:
        r=one(fam,seed);out.append(r)
        print("STRUCTURED_L1S_CANARY_REP_JSON="+json.dumps(r,separators=(",",":")),flush=True)
    print("STRUCTURED_L1S_CANARY_JSON="+json.dumps(out,separators=(",",":")),flush=True)
