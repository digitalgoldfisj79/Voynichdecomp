#!/usr/bin/env python3
# Phase L3: real structured plaintext survival through frozen SELECT->FORM.
# Uses Circa Instans real word sequence from repo; synthetic notation + shuffled repertoire controls.
# Calibration only. NO P70. No real Voynich inversion.
import io,pickle,re,urllib.request,json,math,hashlib,collections
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score

K0URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f/research/select_form_socket_oracle_phaseK0_20261004.py"
k0={"__name__":"k0"};exec(compile(urllib.request.urlopen(K0URL,timeout=60).read().decode(),K0URL,"exec"),k0)
CIURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/67a73f80da2caefd6788de43c003701225833825/Paper/Cipher_paper/ci_corpus_parsed.pkl"
ci=pickle.loads(urllib.request.urlopen(CIURL,timeout=60).read())
words=[str(w).lower() for w in ci["all_words"] if re.fullmatch(r"[a-zA-Z]+",str(w)) and len(str(w))>1]

NSIG=8;NSEC=4;ENTRY=5.;ROUTE=.5;SEED=20261004
rng=np.random.default_rng(SEED)

# 4 contiguous real-text sections, equal length, cap to keep calibration manageable.
L=min(9000,len(words)//NSEC)
sections=[words[i*(len(words)//NSEC):i*(len(words)//NSEC)+L] for i in range(NSEC)]

def section_vocab(seq,V=48):
    c=collections.Counter(seq)
    return [w for w,n in c.most_common(V)]

def balanced_map(vocab,sec):
    # Frequency-agnostic deterministic mapping: many words collide by construction.
    out={}
    for w in vocab:
        h=int(hashlib.sha256(f"{sec}|{w}".encode()).hexdigest()[:16],16)
        out[w]=h%NSIG
    return out

# fixed global signature-to-FORM controls
U,Ve0,Vr0=k0["controls"](NSIG,2,1.,np.random.default_rng(SEED+77),"F1")
Ve=Ve0*ENTRY;Vr=Vr0*ROUTE

def render_sig_seq(sigseq,seed):
    rr=np.random.default_rng(seed);obs=[];soft=[]
    for s in sigseq:
        for _ in range(100):
            tok=k0["gen_token"](int(s),U,Ve,Vr,rr)
            if tok is not None: break
        ll=np.array([k0["token_loglik"](tok,j,U,Ve,Vr) for j in range(NSIG)],float)
        ll-=ll.max();q=np.exp(ll);q/=q.sum()
        obs.append(tok);soft.append(q)
    return obs,np.asarray(soft)

def build_states(seq,vocab):
    idx={w:i for i,w in enumerate(vocab)}
    other=len(vocab)
    z=np.array([idx.get(w,other) for w in seq],int)
    labels=vocab+["<OTHER>"]
    return z,labels

def fit_trans(z,nstate,alpha=.15):
    pi=np.bincount(z[:max(50,len(z)//20)],minlength=nstate).astype(float)+alpha
    pi/=pi.sum()
    C=np.full((nstate,nstate),alpha,float)
    for a,b in zip(z[:-1],z[1:]): C[a,b]+=1
    # interpolated with unigram to reduce sparse bigram overfit
    uni=np.bincount(z,minlength=nstate).astype(float)+alpha
    uni/=uni.sum()
    A=C/C.sum(1,keepdims=True)
    A=.85*A+.15*uni[None,:]
    A/=A.sum(1,keepdims=True)
    return pi,A

def fb_decode(em,A,pi):
    n,K=em.shape
    la=np.log(np.maximum(A,1e-300));lp=np.log(np.maximum(pi,1e-300))
    E=np.log(np.maximum(em,1e-300))
    al=np.empty((n,K));sc=np.empty(n)
    x=lp+E[0];mx=x.max();sc[0]=mx+np.log(np.exp(x-mx).sum());al[0]=x-sc[0]
    for t in range(1,n):
        M=al[t-1][:,None]+la;mm=M.max(0);pr=mm+np.log(np.exp(M-mm).sum(0))
        x=pr+E[t];mx=x.max();sc[t]=mx+np.log(np.exp(x-mx).sum());al[t]=x-sc[t]
    be=np.zeros((n,K))
    for t in range(n-2,-1,-1):
        M=la+E[t+1][None,:]+be[t+1][None,:];mm=M.max(1)
        be[t]=mm+np.log(np.exp(M-mm[:,None]).sum(1))-sc[t+1]
    lg=al+be;mm=lg.max(1,keepdims=True);g=np.exp(lg-mm);g/=g.sum(1,keepdims=True)
    return g,g.argmax(1)

def emissions_from_signature(sig_or_soft,state_sig,is_soft):
    n=len(sig_or_soft);K=len(state_sig)
    E=np.zeros((n,K),float)
    if is_soft:
        for k,s in enumerate(state_sig): E[:,k]=np.maximum(sig_or_soft[:,s],1e-12)
    else:
        for k,s in enumerate(state_sig): E[:,k]=np.where(sig_or_soft==s,1.0,1e-12)
    return E

def same_source_auc(z,g,state_sig,seed):
    rr=np.random.default_rng(seed);ys=[];ss=[]
    # restrict pairs to states with same instantaneous signature and exclude OTHER-heavy trivialities
    sigz=np.array([state_sig[x] for x in z])
    by={s:np.where(sigz==s)[0] for s in range(NSIG)}
    for s,ix in by.items():
        if len(ix)<10:continue
        for _ in range(1800):
            a,b=rr.choice(ix,2,replace=False)
            ys.append(int(z[a]==z[b]));ss.append(float(np.dot(g[a],g[b])))
    return float(roc_auc_score(ys,ss)) if len(set(ys))>1 else float("nan")

def recurrence_top1(z,g,state_sig):
    # For each test token, retrieve most posterior-similar earlier occurrence with same signature;
    # score whether it is the same source state.
    hit=tot=0
    for i in range(20,len(z)):
        cand=np.where(np.array(state_sig)[z[:i]]==state_sig[z[i]])[0]
        if len(cand)==0:continue
        sims=g[cand]@g[i]
        j=cand[int(np.argmax(sims))]
        hit+=int(z[j]==z[i]);tot+=1
    return hit/max(tot,1)

def eval_sequence(seq,sec,family,seed):
    vocab=section_vocab(seq[:int(.65*len(seq))],48)
    z,labels=build_states(seq,vocab);K=len(labels)
    smap=balanced_map(vocab,sec)
    # OTHER gets a fixed extra signature per section
    state_sig=np.array([smap[w] for w in vocab]+[(sec*3+5)%NSIG],int)
    sig=state_sig[z]
    ntr=int(.65*len(seq));nval=int(.10*len(seq));nt=ntr+nval
    pi,A=fit_trans(z[:nt],K)
    _,soft=render_sig_seq(sig,seed+sec*100)
    out={}
    for ch,data,issoft in (("CONTROL",sig,False),("FORM",soft,True)):
        E=emissions_from_signature(data[nt:],state_sig,issoft)
        g,p=fb_decode(E,A,pi);zt=z[nt:]
        mask=zt<len(vocab)  # evaluate identifiable lexical states, not OTHER
        nmi=float(normalized_mutual_info_score(zt[mask],p[mask])) if mask.sum()>10 else float("nan")
        acc=float(np.mean(zt[mask]==p[mask])) if mask.sum()>10 else float("nan")
        auc=same_source_auc(zt,g,state_sig,seed+sec*31)
        rec=recurrence_top1(zt,g,state_sig)
        out[ch]={"source_nmi":nmi,"source_acc":acc,"same_source_auc":auc,"recurrence_top1":rec}
    return out, float(np.mean(np.argmax(soft,1)==sig)), len(vocab), len(seq)

def make_notation(n,sec,seed):
    rr=np.random.default_rng(seed+sec);V=49
    # motif/event system with repeated phrases and section-specific transitions
    motifs=[]
    for m in range(12):
        base=(m*5+sec*7)%V
        motifs.append([(base+j*j+3*m)%V for j in range(5+(m%4))])
    seq=[]
    cur=sec%12
    while len(seq)<n:
        if rr.random()<.72:cur=(cur+(1 if rr.random()<.65 else 3))%12
        else:cur=int(rr.integers(12))
        seq.extend(motifs[cur])
    return [f"N{x}" for x in seq[:n]]

def shuffled_repertoire(seq,sec,seed):
    rr=np.random.default_rng(seed+sec);x=list(seq);rr.shuffle(x);return x

records=[]
for sec,real in enumerate(sections):
    variants=[
      ("PLAINTEXT",real),
      ("NOTATION",make_notation(len(real),sec,SEED+1000)),
      ("REPERTOIRE",shuffled_repertoire(real,sec,SEED+2000))
    ]
    for fam,seq in variants:
        r,sa,Vv,N=eval_sequence(seq,sec,fam,SEED+sec*10000+{"PLAINTEXT":0,"NOTATION":100,"REPERTOIRE":200}[fam])
        rec={"family":fam,"section":sec,"signature_decode_acc":sa,"vocab":Vv,"n":N,**r}
        records.append(rec)
        print("REAL_SOURCE_REP_JSON="+json.dumps(rec,separators=(",",":")),flush=True)

summary={}
for fam in ("PLAINTEXT","NOTATION","REPERTOIRE"):
    rr=[x for x in records if x["family"]==fam]
    summary[fam]={}
    for ch in ("CONTROL","FORM"):
        for met in ("source_nmi","source_acc","same_source_auc","recurrence_top1"):
            summary[fam][f"{ch.lower()}_{met}_median"]=float(np.nanmedian([x[ch][met] for x in rr]))
    summary[fam]["signature_decode_acc_median"]=float(np.median([x["signature_decode_acc"] for x in rr]))
print("REAL_SOURCE_PHASEL3_JSON="+json.dumps({"source":"Circa Instans real sequence","sections":NSEC,"section_len":L,"summary":summary,"records":records},separators=(",",":")),flush=True)
