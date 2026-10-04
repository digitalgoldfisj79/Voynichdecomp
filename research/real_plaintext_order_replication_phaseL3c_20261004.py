#!/usr/bin/env python3
# Phase L3c: expanded real-plaintext order-survival calibration.
# 12 contiguous Circa Instans blocks x 2 independent many-to-one mappings.
# Same FORM trace within each replicate; true-order source model vs unigram and shuffled-order nulls.
# Calibration only. NO P70. No real Voynich inversion.
import io,pickle,re,urllib.request,json,math,hashlib,collections
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score
from concurrent.futures import ProcessPoolExecutor,as_completed

K0URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f/research/select_form_socket_oracle_phaseK0_20261004.py"
k0={"__name__":"k0"};exec(compile(urllib.request.urlopen(K0URL,timeout=60).read().decode(),K0URL,"exec"),k0)
CIURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/67a73f80da2caefd6788de43c003701225833825/Paper/Cipher_paper/ci_corpus_parsed.pkl"
ci=pickle.loads(urllib.request.urlopen(CIURL,timeout=60).read())
WORDS=[str(w).lower() for w in ci["all_words"] if re.fullmatch(r"[a-zA-Z]+",str(w)) and len(str(w))>1]

NSIG=8; ENTRY=5.; ROUTE=.5; BASESEED=20261004
BLOCK=4000; NBLOCK=12; MAPSEEDS=(17,53); NNULL=20
NTRAIN=2600; NVAL=400; NTEST=1000
assert NTRAIN+NVAL+NTEST==BLOCK

# Fixed global SELECT->FORM controls for all replicates.
U,Ve0,Vr0=k0["controls"](NSIG,2,1.,np.random.default_rng(BASESEED+77),"F1")
Ve=Ve0*ENTRY; Vr=Vr0*ROUTE

def top_vocab(seq,n=40):
    c=collections.Counter(seq[:NTRAIN])
    return [w for w,_ in c.most_common(n)]

def mapping(vocab,block_id,mapseed):
    out={}
    for w in vocab:
        h=int(hashlib.sha256(f"{block_id}|{mapseed}|{w}".encode()).hexdigest()[:16],16)
        out[w]=h%NSIG
    return out

def encode(seq,vocab,block_id,mapseed):
    idx={w:i for i,w in enumerate(vocab)}
    other=len(vocab)
    z=np.array([idx.get(w,other) for w in seq],dtype=np.int16)
    mp=mapping(vocab,block_id,mapseed)
    state_sig=np.array([mp[w] for w in vocab]+[((block_id*5+mapseed*3)+7)%NSIG],dtype=np.int8)
    sig=state_sig[z]
    return z,state_sig,sig

def render_soft(sig,seed):
    rr=np.random.default_rng(seed)
    soft=np.empty((len(sig),NSIG),float)
    for i,s in enumerate(sig):
        tok=None
        for _ in range(100):
            tok=k0["gen_token"](int(s),U,Ve,Vr,rr)
            if tok is not None:break
        if tok is None:raise RuntimeError("FORM nontermination")
        ll=np.array([k0["token_loglik"](tok,j,U,Ve,Vr) for j in range(NSIG)],float)
        ll-=ll.max();q=np.exp(ll);q/=q.sum()
        soft[i]=q
    return soft

def uni_and_A(z,K,shuffle_rng=None):
    train=z[:NTRAIN+NVAL].copy()
    if shuffle_rng is not None: shuffle_rng.shuffle(train)
    alpha=.15
    uni=np.bincount(z[:NTRAIN+NVAL],minlength=K).astype(float)+alpha
    uni/=uni.sum()
    C=np.full((K,K),alpha,float)
    for a,b in zip(train[:-1],train[1:]):C[a,b]+=1
    A=C/C.sum(1,keepdims=True)
    # modest interpolation protects against sparse lexical bigrams.
    A=.85*A+.15*uni[None,:]
    A/=A.sum(1,keepdims=True)
    A0=np.tile(uni[None,:],(K,1))
    return uni,A0,A

def emissions(soft,state_sig):
    return np.maximum(soft[:,state_sig],1e-15)

def fb(E,A,pi):
    n,K=E.shape
    logA=np.log(np.maximum(A,1e-300)); logpi=np.log(np.maximum(pi,1e-300)); le=np.log(np.maximum(E,1e-300))
    al=np.empty((n,K));sc=np.empty(n)
    x=logpi+le[0];mx=x.max();sc[0]=mx+np.log(np.exp(x-mx).sum());al[0]=x-sc[0]
    for t in range(1,n):
        M=al[t-1][:,None]+logA
        mm=M.max(0);pr=mm+np.log(np.exp(M-mm).sum(0))
        x=pr+le[t];mx=x.max();sc[t]=mx+np.log(np.exp(x-mx).sum());al[t]=x-sc[t]
    be=np.zeros((n,K))
    for t in range(n-2,-1,-1):
        M=logA+le[t+1][None,:]+be[t+1][None,:]
        mm=M.max(1);be[t]=mm+np.log(np.exp(M-mm[:,None]).sum(1))-sc[t+1]
    lg=al+be;mm=lg.max(1,keepdims=True);g=np.exp(lg-mm);g/=g.sum(1,keepdims=True)
    return float(sc.sum()),g,g.argmax(1)

def make_pairs(z,state_sig,seed):
    rr=np.random.default_rng(seed)
    sigz=state_sig[z]
    ia=[];ib=[];yy=[]
    for s in range(NSIG):
        ix=np.where(sigz==s)[0]
        if len(ix)<10:continue
        for _ in range(1200):
            a,b=rr.choice(ix,2,replace=False)
            ia.append(int(a));ib.append(int(b));yy.append(int(z[a]==z[b]))
    return np.array(ia),np.array(ib),np.array(yy)

def auc_from_pairs(g,pairs):
    a,b,y=pairs
    if len(np.unique(y))<2:return float("nan")
    scores=np.sum(g[a]*g[b],axis=1)
    return float(roc_auc_score(y,scores))

def metrics(z,g,p,ll,pairs,Klex):
    mask=z<Klex
    ix=np.where(mask)[0]
    nmi=float(normalized_mutual_info_score(z[mask],p[mask]))
    acc=float(np.mean(z[mask]==p[mask]))
    auc=auc_from_pairs(g,pairs)
    tb=float(np.mean(np.log2(np.maximum(g[ix,z[ix]],1e-300))))
    return {"nmi":nmi,"acc":acc,"auc":auc,"true_logpost_bits":tb,"obs_ll_bits_per_tok":float(ll/len(z)/math.log(2))}

def one(rep):
    block_id,mapseed=rep
    seq=WORDS[block_id*BLOCK:(block_id+1)*BLOCK]
    if len(seq)!=BLOCK:raise RuntimeError("short block")
    vocab=top_vocab(seq,40);K=len(vocab)+1
    z,state_sig,sig=encode(seq,vocab,block_id,mapseed)
    soft=render_soft(sig,BASESEED+block_id*1009+mapseed*101)
    uni,A0,A=uni_and_A(z,K)
    # isolate transition order: same unigram initial distribution in all models.
    pi=uni.copy()
    zt=z[NTRAIN+NVAL:];Et=emissions(soft[NTRAIN+NVAL:],state_sig)
    pairs=make_pairs(zt,state_sig,BASESEED+block_id*71+mapseed)
    ll0,g0,p0=fb(Et,A0,pi);m0=metrics(zt,g0,p0,ll0,pairs,len(vocab))
    ll1,g1,p1=fb(Et,A,pi);m1=metrics(zt,g1,p1,ll1,pairs,len(vocab))
    delta={k:m1[k]-m0[k] for k in m0}
    null=[]
    for j in range(NNULL):
        rr=np.random.default_rng(BASESEED+block_id*100000+mapseed*1000+j)
        _,_,An=uni_and_A(z,K,shuffle_rng=rr)
        lln,gn,pn=fb(Et,An,pi);mn=metrics(zt,gn,pn,lln,pairs,len(vocab))
        null.append({k:mn[k]-m0[k] for k in m0})
    zscore={}
    for k in delta:
        v=np.array([x[k] for x in null],float)
        sd=float(v.std(ddof=1))
        zscore[k]=float((delta[k]-v.mean())/sd) if sd>0 else float("nan")
    out={"block":block_id,"mapseed":mapseed,"vocab":len(vocab),
         "signature_decode_acc":float(np.mean(np.argmax(soft,1)==sig)),
         "unigram":m0,"sequential":m1,"delta":delta,"z_vs_shuffle":zscore,"null":null}
    print("PLAINTEXT_ORDER_REP_JSON="+json.dumps(out,separators=(",",":")),flush=True)
    return out

if __name__=="__main__":
    reps=[(b,m) for b in range(NBLOCK) for m in MAPSEEDS]
    out=[]
    with ProcessPoolExecutor(max_workers=8) as ex:
        fut={ex.submit(one,r):r for r in reps}
        for f in as_completed(fut):out.append(f.result())
    keys=("nmi","acc","auc","true_logpost_bits","obs_ll_bits_per_tok")
    summary={"n":len(out)}
    for k in keys:
        obs=np.array([r["delta"][k] for r in out],float)
        summary[f"delta_{k}_median"]=float(np.median(obs))
        summary[f"delta_{k}_mean"]=float(np.mean(obs))
        summary[f"frac_positive_{k}"]=float(np.mean(obs>0))
        summary[f"rep_z_{k}_median"]=float(np.nanmedian([r["z_vs_shuffle"][k] for r in out]))
        # matched panel null: null index j gives one null panel draw across all reps.
        nv=np.array([[r["null"][j][k] for r in out] for j in range(NNULL)],float)
        null_panel=nv.mean(1)
        sd=float(null_panel.std(ddof=1))
        summary[f"panel_z_{k}"]=float((obs.mean()-null_panel.mean())/sd) if sd>0 else float("nan")
        summary[f"null_panel_mean_{k}"]=float(null_panel.mean())
        summary[f"null_panel_sd_{k}"]=sd
    print("PLAINTEXT_ORDER_PHASEL3C_JSON="+json.dumps({
      "source":"Circa Instans","blocks":NBLOCK,"block_len":BLOCK,"mapseeds":list(MAPSEEDS),
      "nnull":NNULL,"summary":summary,"records":out
    },separators=(",",":")),flush=True)
