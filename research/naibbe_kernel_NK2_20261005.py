#!/usr/bin/env python3
# NK2 — source-shape calibration through stripped SOURCE -> FAMILY -> frozen FORM channel.
# A real Circa Instans order, B notation-like structured order, C exact-repertoire shuffled hostile control.
import collections, hashlib, json, math, pickle, re, urllib.request
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, roc_auc_score

SEED=20261005; BLOCK=4000; NBLOCK=6; NTRAIN=2600; NVAL=400; NTEST=1000
TOPV=24; NSIG=8; KFAM=4; NB=6; ALPHA=.70; PRIOR=8.; MC=3500; NNULL=20; RHO=.25
K0URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f/research/select_form_socket_oracle_phaseK0_20261004.py"
CIURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/67a73f80da2caefd6788de43c003701225833825/Paper/Cipher_paper/ci_corpus_parsed.pkl"
k0={"__name__":"k0"};exec(compile(urllib.request.urlopen(K0URL,timeout=60).read().decode(),K0URL,"exec"),k0)
ci=pickle.loads(urllib.request.urlopen(CIURL,timeout=60).read())
WORDS=[str(w).lower() for w in ci["all_words"] if re.fullmatch(r"[a-zA-Z]+",str(w)) and len(str(w))>1]
vocab=[w for w,_ in collections.Counter(WORDS[:NBLOCK*BLOCK]).most_common(TOPV)]
VID={w:i for i,w in enumerate(vocab)};OTHER=len(vocab);K=OTHER+1

# Global many-to-one source -> SELECT signature mapping, fixed before all block outcomes.
state_sig=[]
for i,w in enumerate(vocab+["<OTHER>"]):
    h=int(hashlib.sha256(("NK2SIG|"+w).encode()).hexdigest()[:16],16);state_sig.append(h%NSIG)
state_sig=np.array(state_sig,np.int8)

# Frozen FORM controls on 8 SELECT signatures.
U,Ve0,Vr0=k0["controls"](NSIG,2,1.,np.random.default_rng(SEED+77),"F1")
Ve=Ve0*5.;Vr=Vr0*.5
PCLASS=k0["PCLASS"];PSC=k0["P_START_CLASS"];PSP=k0["P_START_PIECE"];PC=k0["P_CONT"];LEGAL=k0["LEGAL"];PN=k0["P_NEXT"];PSTOP=k0["P_STOP"]
G_OF=k0["G_OF"];START=k0["START_IN"];K12=k0["K12"];NP=k0["NP"];PIECES=k0["PIECES"]
GALS=set("fkpt");PG=np.array([any(c in GALS for c in p) for p in PIECES],bool)

def row_soft(base,bias,mask):
    q=np.zeros_like(base,float);ok=mask&(base>0);z=np.log(np.maximum(base[ok],1e-300))+bias[ok];z-=z.max();q[ok]=np.exp(z);q/=q.sum();return q
def gen_sig(s,rng):
    be=U[s]@Ve;br=U[s]@Vr;q=row_soft(PSC,be,np.ones(K12,bool));c=int(rng.choice(K12,p=q))
    pp=int(rng.choice(NP,p=PSP[c]));out=[pp];inc=START;used=bool(PG[pp])
    for d in range(30):
        if rng.random()<PSTOP[pp,min(d,3),inc]:return out
        g=G_OF[PCLASS[pp]];qd=row_soft(PC[g],br,LEGAL[g]);dc=int(rng.choice(K12,p=qd));qn=PN[g,dc].copy()
        if used: qn[PG]*=RHO; qn/=max(qn.sum(),1e-300)
        nxt=int(rng.choice(NP,p=qn));inc=PCLASS[pp];pp=nxt;out.append(pp);used=used or bool(PG[pp])
    return None
def sample_sig(s,rng):
    for _ in range(200):
        t=gen_sig(int(s),rng)
        if t is not None:return t
    raise RuntimeError("FORM nontermination")
def ll_sig(t,s):
    be=U[s]@Ve;br=U[s]@Vr;c0=PCLASS[t[0]];q=row_soft(PSC,be,np.ones(K12,bool))
    lp=math.log(max(q[c0],1e-300))+math.log(max(PSP[c0,t[0]],1e-300));inc=START;used=bool(PG[t[0]])
    for i,pp in enumerate(t):
        ps=PSTOP[pp,min(i,3),inc]
        if i==len(t)-1:lp+=math.log(max(ps,1e-300));break
        lp+=math.log(max(1-ps,1e-300));g=G_OF[PCLASS[pp]];dc=PCLASS[t[i+1]]
        qd=row_soft(PC[g],br,LEGAL[g]);lp+=math.log(max(qd[dc],1e-300))
        qn=PN[g,dc].copy()
        if used:qn[PG]*=RHO;qn/=max(qn.sum(),1e-300)
        lp+=math.log(max(qn[t[i+1]],1e-300));inc=PCLASS[pp];used=used or bool(PG[t[i+1]])
    return lp
def fam(t):
    cls=[int(PCLASS[p]) for p in t]
    s=",".join(map(str,cls[1:]))+"|F"+str(cls[-1])
    return int.from_bytes(hashlib.sha256(("NK1FAM|"+s).encode()).digest()[:8],"big")%KFAM
def lb(t): return min(len(t),NB)-1
def surf(t):return "".join(PIECES[p] for p in t)

# Frozen baseline p(family | SELECT signature, length bucket), keeping impossible cells at zero.
rr=np.random.default_rng(SEED+100);raw=np.zeros((NSIG,NB,KFAM),float)
for s in range(NSIG):
    for _ in range(MC):
        t=sample_sig(s,rr);raw[s,lb(t),fam(t)]+=1
SUP=raw>0;PB=np.zeros_like(raw)
for s in range(NSIG):
    for b in range(NB):
        ok=SUP[s,b]
        if ok.any():
            v=raw[s,b]+.5*ok;PB[s,b]=v/v.sum()

# Global rank-2 source family preference, independent of target tokens.
rp=np.random.default_rng(SEED+200);Us=rp.normal(size=(K,2));Us=(Us-Us.mean(0))/np.maximum(Us.std(0),1e-9)
Vf=rp.normal(size=(2,KFAM));Vf-=Vf.mean(1,keepdims=True);Vf/=np.maximum(Vf.std(),1e-9);SCORE=Us@Vf
QTRUE=np.zeros((K,NB,KFAM),float)
for x in range(K):
    s=int(state_sig[x])
    for b in range(NB):
        ok=SUP[s,b]
        if not ok.any():continue
        z=np.log(np.maximum(PB[s,b,ok],1e-12))+ALPHA*SCORE[x,ok];z-=z.max();q=np.exp(z);QTRUE[x,b,ok]=q/q.sum()

def emit_one(x,rng):
    s=int(state_sig[x]);seed=sample_sig(s,rng);b=lb(seed)
    if QTRUE[x,b].sum()<=0:return seed
    ff=int(rng.choice(KFAM,p=QTRUE[x,b]))
    if fam(seed)==ff:return seed
    for _ in range(6000):
        t=sample_sig(s,rng)
        if lb(t)==b and fam(t)==ff:return t
    raise RuntimeError(("NK2 rejection",x,s,b,ff,PB[s,b].tolist()))

def source_A(block):
    seq=WORDS[block*BLOCK:(block+1)*BLOCK]
    return np.array([VID.get(w,OTHER) for w in seq],np.int16)
def source_C(z,block):
    q=z.copy();np.random.default_rng(SEED+9000+block).shuffle(q);return q
def source_B(z,block):
    # notation-like: same state repertoire and training unigram prior, but strong cyclic/motif transitions
    p=np.bincount(z[:NTRAIN+NVAL],minlength=K).astype(float)+1.;p/=p.sum()
    A=np.tile(.10*p[None,:],(K,1))
    for i in range(K):
        A[i,(i+1)%K]+=.62
        A[i,(i+5)%K]+=.18
        A[i,i]+=.10
    A/=A.sum(1,keepdims=True)
    r=np.random.default_rng(SEED+10000+block);out=np.empty(BLOCK,np.int16);out[0]=int(r.choice(K,p=p))
    for i in range(1,BLOCK):out[i]=int(r.choice(K,p=A[out[i-1]]))
    return out

def learn_family(z,toks):
    C=np.zeros((K,NB,KFAM),float)
    for x,t in zip(z[:NTRAIN+NVAL],toks[:NTRAIN+NVAL]):C[int(x),lb(t),fam(t)]+=1
    Q=np.zeros_like(C)
    for x in range(K):
        s=int(state_sig[x])
        for b in range(NB):
            base=PB[s,b]
            v=C[x,b]+PRIOR*base
            if v.sum()>0:Q[x,b]=v/v.sum()
    return Q

def build_logE(z,toks,Q,use_family):
    n=len(toks);E=np.empty((n,K),float)
    for i,t in enumerate(toks):
        b=lb(t);f=fam(t)
        for x in range(K):
            s=int(state_sig[x]);v=ll_sig(t,s)
            if use_family:
                den=PB[s,b,f];num=Q[x,b,f]
                if den<=0 or num<=0:v=-1e6
                else:v+=math.log(num/den)
            E[i,x]=v
        E[i]-=E[i].max()
    return E

def trans(z,shuffle_rng=None):
    q=z[:NTRAIN+NVAL].copy()
    if shuffle_rng is not None:shuffle_rng.shuffle(q)
    alpha=.15
    uni=np.bincount(z[:NTRAIN+NVAL],minlength=K).astype(float)+alpha;uni/=uni.sum()
    C=np.full((K,K),alpha,float)
    for a,b in zip(q[:-1],q[1:]):C[a,b]+=1
    A=C/C.sum(1,keepdims=True);A=.85*A+.15*uni[None,:];A/=A.sum(1,keepdims=True)
    return uni,np.tile(uni[None,:],(K,1)),A

def fb(logE,A,pi):
    n,kk=logE.shape;la=np.log(np.maximum(A,1e-300));lp=np.log(np.maximum(pi,1e-300))
    al=np.empty((n,kk));sc=np.empty(n);filt=np.empty((n,kk))
    x=lp+logE[0];m=x.max();sc[0]=m+np.log(np.exp(x-m).sum());al[0]=x-sc[0];filt[0]=np.exp(al[0])
    for t in range(1,n):
        M=al[t-1][:,None]+la;mm=M.max(0);pr=mm+np.log(np.exp(M-mm).sum(0))
        x=pr+logE[t];m=x.max();sc[t]=m+np.log(np.exp(x-m).sum());al[t]=x-sc[t];filt[t]=np.exp(al[t])
    be=np.zeros((n,kk))
    for t in range(n-2,-1,-1):
        M=la+logE[t+1][None,:]+be[t+1][None,:];mm=M.max(1);be[t]=mm+np.log(np.exp(M-mm[:,None]).sum(1))-sc[t+1]
    lg=al+be;mm=lg.max(1,keepdims=True);g=np.exp(lg-mm);g/=g.sum(1,keepdims=True)
    return float(sc.sum()),g,g.argmax(1),filt

def pairs(z,famobs,seed,crossfam=False):
    r=np.random.default_rng(seed);a=[];b=[];y=[];sg=state_sig[z]
    for s in range(NSIG):
        ix=np.where(sg==s)[0]
        if len(ix)<20:continue
        for _ in range(800):
            i,j=r.choice(ix,2,replace=False);same=(z[i]==z[j])
            if crossfam and same and famobs[i]==famobs[j]:continue
            a.append(i);b.append(j);y.append(int(same))
    return np.array(a),np.array(b),np.array(y)
def auc(g,p):
    a,b,y=p
    if len(y)==0 or len(np.unique(y))<2:return float("nan")
    return float(roc_auc_score(y,np.sum(g[a]*g[b],axis=1)))
def metrics(zt,g,pred,filt,A,pi,pair,paircf):
    mask=zt<OTHER;ix=np.where(mask)[0]
    nmi=float(normalized_mutual_info_score(zt[mask],pred[mask]));ari=float(adjusted_rand_score(zt[mask],pred[mask]))
    acc=float(np.mean(zt[mask]==pred[mask]));tb=float(np.mean(np.log2(np.maximum(g[ix,zt[ix]],1e-300))))
    nxt=[]
    for t in range(len(zt)-1):
        pr=filt[t]@A;nxt.append(math.log2(max(pr[int(zt[t+1])],1e-300)))
    return {"nmi":nmi,"ari":ari,"acc":acc,"auc":auc(g,pair),"cross_family_auc":auc(g,paircf),
            "true_logpost_bits":tb,"next_source_logprob_bits":float(np.mean(nxt))}
def cmi_block_family_given_source(blocks,zs,fs):
    # plug-in conditional MI I(B;F|Z), weighted over source states
    tot=sum(len(x) for x in zs);res=0.
    for x in range(K):
        triples=[]
        for b,(z,f) in enumerate(zip(zs,fs)):
            for zz,ff in zip(z,f):
                if zz==x:triples.append((b,int(ff)))
        n=len(triples)
        if n<2:continue
        cb=collections.Counter(b for b,f in triples);cf=collections.Counter(f for b,f in triples);cbf=collections.Counter(triples)
        mi=0.
        for (b,f),nn in cbf.items():
            p=nn/n;mi+=p*math.log2(p/((cb[b]/n)*(cf[f]/n)))
        res+=(n/tot)*mi
    return res

def run_arm(z,block,arm):
    rng=np.random.default_rng(SEED+block*1000+{"A":11,"B":22,"C":33}[arm])
    toks=[emit_one(int(x),rng) for x in z]
    Q=learn_family(z,toks)
    sl=slice(NTRAIN+NVAL,None);zt=z[sl];tt=toks[sl]
    famobs=np.array([fam(t) for t in tt],np.int8)
    uni,A0,A=trans(z);pi=uni.copy()
    pair=pairs(zt,famobs,SEED+block*17+ord(arm),False);paircf=pairs(zt,famobs,SEED+block*19+ord(arm),True)
    out={}
    for label,usefam,TA in [("base_unigram",False,A0),("family_unigram",True,A0),("family_sequential",True,A)]:
        E=build_logE(zt,tt,Q,usefam);ll,g,p,filt=fb(E,TA,pi);m=metrics(zt,g,p,filt,TA,pi,pair,paircf);m["obs_ll_nats"]=ll;out[label]=m
    out["family_gain"]={k:out["family_unigram"][k]-out["base_unigram"][k] for k in out["base_unigram"] if k!="obs_ll_nats"}
    out["order_gain"]={k:out["family_sequential"][k]-out["family_unigram"][k] for k in out["family_unigram"] if k!="obs_ll_nats"}
    null=[]
    E=build_logE(zt,tt,Q,True)
    for j in range(NNULL):
        _,_,An=trans(z,np.random.default_rng(SEED+block*100000+ord(arm)*100+j))
        ll,g,p,filt=fb(E,An,pi);m=metrics(zt,g,p,filt,An,pi,pair,paircf)
        null.append({k:m[k]-out["family_unigram"][k] for k in m})
    zsc={}
    for k,v in out["order_gain"].items():
        nv=np.array([n[k] for n in null],float);sd=float(np.nanstd(nv,ddof=1));mu=float(np.nanmean(nv))
        zsc[k]={"effect":float(v-mu),"null_mean":mu,"null_sd":sd,"z":float((v-mu)/sd) if sd>0 else None}
    trains=set(surf(t) for t in toks[:NTRAIN+NVAL]);testsurf=[surf(t) for t in tt]
    out.update({"order_vs_shuffle":zsc,"order_null":null,"test_unique_types":len(set(testsurf)),
                "test_unseen_share":float(np.mean([s not in trains for s in testsurf])),
                "test_family":famobs.tolist(),"test_source":zt.tolist()})
    return out

records=[]
A_zs=[];A_fs=[]
for b in range(NBLOCK):
    za=source_A(b);seqs={"A":za,"B":source_B(za,b),"C":source_C(za,b)}
    rec={"block":b,"arms":{}}
    for arm,z in seqs.items():
        x=run_arm(z,b,arm);rec["arms"][arm]=x
        if arm=="A":A_zs.append(x["test_source"]);A_fs.append(x["test_family"])
    records.append(rec)
    print("NK2_BLOCK_JSON="+json.dumps(rec,separators=(",",":")),flush=True)

# panel summaries across blocks
keys=["nmi","ari","acc","auc","cross_family_auc","true_logpost_bits","next_source_logprob_bits"]
summary={}
for arm in "ABC":
    summary[arm]={}
    for k in keys:
        fg=np.array([r["arms"][arm]["family_gain"][k] for r in records],float)
        og=np.array([r["arms"][arm]["order_gain"][k] for r in records],float)
        null_mu=np.array([r["arms"][arm]["order_vs_shuffle"][k]["null_mean"] for r in records],float)
        exc=og-null_mu
        nv=np.array([[r["arms"][arm]["order_null"][j][k] for r in records] for j in range(NNULL)],float)
        panel_null=np.nanmean(nv,axis=1)
        pnm=float(np.nanmean(panel_null));pns=float(np.nanstd(panel_null,ddof=1))
        pe=float(np.nanmean(og)-pnm)
        summary[arm][k]={"family_gain_mean":float(np.nanmean(fg)),
                         "order_gain_mean":float(np.nanmean(og)),
                         "order_excess_over_shuffle_mean":float(np.nanmean(exc)),
                         "block_sd_excess":float(np.nanstd(exc,ddof=1)),
                         "block_z":float(np.nanmean(exc)/np.nanstd(exc,ddof=1)) if np.nanstd(exc,ddof=1)>0 else None,
                         "panel_null_mean":pnm,"panel_null_sd":pns,"panel_effect":pe,
                         "panel_z":float(pe/pns) if pns>0 else None,
                         "median_within_block_shuffle_z":float(np.nanmedian([r["arms"][arm]["order_vs_shuffle"][k]["z"] for r in records]))}
    summary[arm]["diversity"]={"mean_unique_types":float(np.mean([r["arms"][arm]["test_unique_types"] for r in records])),
                               "mean_unseen_share":float(np.mean([r["arms"][arm]["test_unseen_share"] for r in records]))}

cmi=cmi_block_family_given_source(list(range(NBLOCK)),[np.array(z) for z in A_zs],[np.array(f) for f in A_fs])
out={"phase":"NK2","status":"complete","nblocks":NBLOCK,"block":BLOCK,"test_per_block":NTEST,"source_states":K,
     "channel":{"select_signatures":NSIG,"support_families":KFAM,"length_neutral":True,"no_exact_token_lookup":True},
     "arms":{"A":"real Circa Instans order","B":"notation-like structured order","C":"exact-repertoire shuffled hostile control"},
     "summary":summary,"A_conditional_MI_block_family_given_source_bits":cmi,
     "prior_L3c_reference":{"nmi_order_gain":0.02774,"same_source_auc_gain":0.01196,"true_source_posterior_gain_bits":0.1647,
                            "note":"pre-existing expanded real-plaintext order calibration; not recomputed here"},
     "closure_note":"Stolfi/Mauro closure is not used as an NK2 promotion statistic because the family coordinate is deliberately planted/random. Classical real-Voynich closure is reserved for NK4 where it is evidential rather than tautological.",
     "records":records}
print("NAIBBE_NK2_JSON="+json.dumps(out,separators=(",",":")),flush=True)
