#!/usr/bin/env python3
# Phase L0: Q13/Mauro-informed structured-source calibration through frozen FORM.
# Synthetic-only. NO P70. No real Voynich inversion.
import json,math,urllib.request,warnings
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score
from hmmlearn.hmm import CategoricalHMM
warnings.filterwarnings("ignore")

K0URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f/research/select_form_socket_oracle_phaseK0_20261004.py"
k0={"__name__":"k0"};exec(compile(urllib.request.urlopen(K0URL,timeout=60).read().decode(),K0URL,"exec"),k0)

V=16;NSIG=8;NSEC=4;SECLEN=1200
NFIT=650;NVAL=200;NTEST=350
ENTRY=5.0;ROUTE=.5
assert NFIT+NVAL+NTEST==SECLEN
SIG_OF=np.arange(V)%NSIG

def section_prior(sec):
    # Zipf-like, with observed register/section modulation but all types retained.
    r=np.arange(1,V+1,dtype=float)
    base=1/(r**.75)
    # rotate which quarter is section-favoured; moderate rather than exclusive.
    fav=np.zeros(V);idx=np.arange(sec*4,(sec+1)*4)%V;fav[idx]=1
    w=base*np.exp(.8*fav)
    return w/w.sum()

def A_family(fam,sec):
    prior=section_prior(sec)
    A=np.zeros((V,V),float)
    if fam=="TABLE":
        A[:]=prior
    elif fam=="LANG":
        for i in range(V):
            # Moderate lexical sequential structure: each source item has distinct
            # preferred followers on top of section-conditioned Zipf vocabulary.
            row=.38*prior
            prefs=[(3*i+1+sec)%V,(5*i+7)%V,(i+1)%V]
            ws=[.30,.20,.12]
            for j,w in zip(prefs,ws):row[j]+=w
            row+=.02/V
            A[i]=row/row.sum()
    elif fam=="NOTATION":
        for i in range(V):
            row=.08*prior
            row[(i+1)%V]+=.60
            row[(i+4+sec)%V]+=.20
            row[(i//4)*4]+=.08
            row[(i+8)%V]+=.04
            A[i]=row/row.sum()
    else:raise ValueError(fam)
    return A

def source_sequence(fam,sec,n,rng):
    A=A_family(fam,sec);pi=section_prior(sec)
    z=np.empty(n,int);z[0]=rng.choice(V,p=pi)
    for t in range(1,n):z[t]=rng.choice(V,p=A[z[t-1]])
    return z,A

def encoder(seed):
    rng=np.random.default_rng(seed+99173)
    U,Ve0,Vr0=k0["controls"](NSIG,2,1.,rng,"F1")
    return U,Ve0*ENTRY,Vr0*ROUTE

def render(z,U,Ve,Vr,rng):
    obs=[];true_sig=SIG_OF[z]
    for s in true_sig:
        for _ in range(100):
            tok=k0["gen_token"](int(s),U,Ve,Vr,rng)
            if tok is not None:obs.append(tok);break
        else:raise RuntimeError("nontermination")
    return obs,true_sig

def decode_sig(obs,U,Ve,Vr):
    pred=[];post=[]
    for tok in obs:
        ll=np.array([k0["token_loglik"](tok,s,U,Ve,Vr) for s in range(NSIG)],float)
        ll-=ll.max();q=np.exp(ll);q/=q.sum()
        pred.append(int(q.argmax()));post.append(q)
    return np.array(pred,int),np.stack(post)

def init_model(seed):
    rng=np.random.default_rng(seed)
    m=CategoricalHMM(n_components=V,n_iter=120,tol=1e-4,algorithm="viterbi",
                     random_state=seed,init_params="",params="ste",
                     implementation="log")
    # two hidden source states per observed SELECT signature.
    start=rng.dirichlet(np.ones(V))
    T=rng.gamma(1.,1.,(V,V));T[np.arange(V),np.arange(V)]+=1.5;T/=T.sum(1,keepdims=True)
    E=np.full((V,NSIG),.10/(NSIG-1),float)
    for v in range(V):E[v,v%NSIG]=.90
    E+=rng.uniform(0,.01,E.shape);E/=E.sum(1,keepdims=True)
    m.startprob_=start;m.transmat_=T;m.emissionprob_=E;m.n_features=NSIG
    return m

def fit_hmm(train,val,seed0=0,restarts=8):
    best=None
    Xtr=train.reshape(-1,1);Xv=val.reshape(-1,1)
    for r in range(restarts):
        seed=seed0+101*r
        m=init_model(seed)
        try:
            m.fit(Xtr)
            lv=float(m.score(Xv))
        except Exception:
            continue
        if best is None or lv>best[0]:best=(lv,m)
    if best is None:raise RuntimeError("all HMM restarts failed")
    return best

def collision_nmi(z,pred):
    vals=[]
    for s in range(NSIG):
        ix=np.where(SIG_OF[z]==s)[0]
        if len(ix)>3 and len(np.unique(z[ix]))>1:
            vals.append(normalized_mutual_info_score(z[ix],pred[ix]))
    return float(np.mean(vals)) if vals else float("nan")

def pair_auc(z,gamma,seed):
    rng=np.random.default_rng(seed)
    bysig={s:np.where(SIG_OF[z]==s)[0] for s in range(NSIG)}
    ys=[];scores=[]
    for _ in range(12000):
        s=int(rng.integers(NSIG));ix=bysig[s]
        if len(ix)<2:continue
        a,b=rng.choice(ix,2,replace=False)
        ys.append(int(z[a]==z[b]));scores.append(float(np.dot(gamma[a],gamma[b])))
    if len(set(ys))<2:return float("nan")
    return float(roc_auc_score(ys,scores))

def section_mi(sections,sigs):
    return float(normalized_mutual_info_score(sections,sigs))

def one_dataset(fam,seed):
    rng=np.random.default_rng(seed)
    U,Ve,Vr=encoder(seed)
    sec_records=[]
    allsec=[];allsig=[];alldec=[]
    for sec in range(NSEC):
        z,A=source_sequence(fam,sec,SECLEN,rng)
        obs,ts=render(z,U,Ve,Vr,rng)
        ds,sp=decode_sig(obs,U,Ve,Vr)
        allsec.extend([sec]*SECLEN);allsig.extend(ts.tolist());alldec.extend(ds.tolist())
        sec_records.append((z,ts,ds,sp,A))
    out_sections=[]
    for sec,(z,ts,ds,sp,Atrue) in enumerate(sec_records):
        for channel,ob in (("CONTROL",ts),("FORM",ds)):
            tr=ob[:NFIT];va=ob[NFIT:NFIT+NVAL];te=ob[NFIT+NVAL:]
            lv,m=fit_hmm(tr,va,seed0=seed+sec*1000+(0 if channel=="CONTROL" else 50000),restarts=8)
            pred=m.predict(te.reshape(-1,1));gam=m.predict_proba(te.reshape(-1,1))
            zte=z[NFIT+NVAL:]
            nmi=float(normalized_mutual_info_score(zte,pred))
            cnmi=collision_nmi(zte,pred)
            auc=pair_auc(zte,gam,seed+sec*37)
            out_sections.append({
                "section":sec,"channel":channel,"val_ll":lv,"source_nmi":nmi,
                "collision_nmi":cnmi,"same_source_auc":auc,
                "sig_decode_acc":float(np.mean(ds==ts))
            })
    def agg(ch,key):
        v=[x[key] for x in out_sections if x["channel"]==ch]
        return float(np.nanmean(v))
    result={
      "family":fam,"seed":seed,
      "signature_decode_accuracy":float(np.mean(np.array(alldec)==np.array(allsig))),
      "section_mi_true_control":section_mi(np.array(allsec),np.array(allsig)),
      "section_mi_form_decoded":section_mi(np.array(allsec),np.array(alldec)),
      "control":{"source_nmi":agg("CONTROL","source_nmi"),
                 "collision_nmi":agg("CONTROL","collision_nmi"),
                 "same_source_auc":agg("CONTROL","same_source_auc")},
      "form":{"source_nmi":agg("FORM","source_nmi"),
              "collision_nmi":agg("FORM","collision_nmi"),
              "same_source_auc":agg("FORM","same_source_auc")},
      "sections":out_sections
    }
    return result

if __name__=="__main__":
    allr=[]
    for fam in ("LANG","NOTATION","TABLE"):
        for seed in (20262001,20262002,20262003):
            r=one_dataset(fam,seed);allr.append(r)
            print("STRUCTURED_SOURCE_REP_JSON="+json.dumps(r,separators=(",",":")),flush=True)
    summary={}
    for fam in ("LANG","NOTATION","TABLE"):
        rr=[x for x in allr if x["family"]==fam]
        summary[fam]={
          "n":len(rr),
          "sig_decode_acc_median":float(np.median([x["signature_decode_accuracy"] for x in rr])),
          "section_mi_surface_median":float(np.median([x["section_mi_form_decoded"] for x in rr])),
          "control_source_nmi_median":float(np.median([x["control"]["source_nmi"] for x in rr])),
          "control_collision_nmi_median":float(np.median([x["control"]["collision_nmi"] for x in rr])),
          "control_same_source_auc_median":float(np.median([x["control"]["same_source_auc"] for x in rr])),
          "form_source_nmi_median":float(np.median([x["form"]["source_nmi"] for x in rr])),
          "form_collision_nmi_median":float(np.median([x["form"]["collision_nmi"] for x in rr])),
          "form_same_source_auc_median":float(np.median([x["form"]["same_source_auc"] for x in rr]))
        }
    print("STRUCTURED_SOURCE_PHASEL0_JSON="+json.dumps({"design":{"V":V,"signatures":NSIG,"sections":NSEC,"section_len":SECLEN,"entry":ENTRY,"route":ROUTE},"summary":summary,"replicates":allr},separators=(",",":")),flush=True)
