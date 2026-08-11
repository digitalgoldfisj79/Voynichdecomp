# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations

import argparse, collections, concurrent.futures, hashlib, json, math, re, statistics, urllib.request
from dataclasses import dataclass
from typing import Any
import numpy as np
from numba import njit
from unidecode import unidecode

NS="AMADIRESIDUALV1"
PLAIN="abcdefghilmnopqrstu"; SURF="acdefghiklmnopqrsty"; K=19
R12="aceilmnorstu"; L12=len(R12)
assert len(PLAIN)==len(set(PLAIN))==len(SURF)==len(set(SURF))==K and L12==12
P2I={c:i for i,c in enumerate(PLAIN)}; S2I={c:i for i,c in enumerate(SURF)}; R2I={c:i for i,c in enumerate(R12)}
RF_URL="https://www.voynich.nu/data/RF1b-er.txt"; RF_SHA="eb857a1f353b18983fbc25b954e1bbce227a26d99cefabfda9206ff9b57644d2"
HEADERS={"User-Agent":"Mozilla/5.0","Referer":"https://www.voynich.nu/transcr.html"}
LANGS=["latin","italian","german","french","greek","hebrew","arabic","spanish"]
URLS={
"latin":"https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-ITTB/master/la_ittb-ud-train.conllu",
"italian":"https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master/it_isdt-ud-train.conllu",
"german":"https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-train.conllu",
"french":"https://raw.githubusercontent.com/UniversalDependencies/UD_French-GSD/master/fr_gsd-ud-train.conllu",
"greek":"https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-Perseus/master/grc_perseus-ud-train.conllu",
"hebrew":"https://raw.githubusercontent.com/UniversalDependencies/UD_Hebrew-HTB/master/he_htb-ud-train.conllu",
"arabic":"https://raw.githubusercontent.com/UniversalDependencies/UD_Arabic-PADT/master/ar_padt-ud-train.conllu",
"spanish":"https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-AnCora/master/es_ancora-ud-train.conllu"}
TRAIN_RES={0,1,3,4,6,8}; CTRL_RES={2,5,7,9}; VOW=set("aeiou")
PWA_RULES=[2,3,4,5]

# Frozen optimizer budget before Q1.
PROPOSALS=30000; MAX_RESTARTS=12; BATCH=4
SMOKE_PROPOSALS=1200; SMOKE_RESTARTS=2

def seed(*x): return int.from_bytes(hashlib.sha256("::".join(map(str,x)).encode()).digest()[:8],"big") & 0x7fffffff

def getb(url,headers=None):
    req=urllib.request.Request(url,headers=headers or {"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req,timeout=120) as r: return r.read()

def norm_std(raw:str)->list[int]:
    s=unidecode(raw).lower().replace("j","i").replace("v","u").replace("w","u").replace("y","i").replace("x","s").replace("z","s")
    return [P2I[c] for c in s if c in P2I]

def norm_r12(raw:str)->list[int]:
    s=unidecode(raw).lower(); out=[]
    for c in s:
        if not ("a"<=c<="z"): continue
        if c=="j": c="i"
        if c=="b": c="u"
        elif c=="d": c="t"
        elif c in "fhp": continue
        elif c=="g": c="i"
        elif c=="q": c="c"
        elif c=="v": c="o"
        elif c=="w": c="u"
        elif c=="y": c="i"
        elif c in "xz": c="s"
        if c not in R2I: return []  # e.g. modern foreign k: exclude whole word, do not invent a rule
        out.append(R2I[c])
    return out

def vc_word(w:list[int])->list[int]:
    return [x for x in w if PLAIN[x] not in VOW]+[x for x in w if PLAIN[x] in VOW]

def parse_ud(raw:bytes)->list[list[str]]:
    sents=[]; cur=[]
    for line in raw.decode("utf-8","replace").splitlines():
        if not line:
            if cur: sents.append(cur); cur=[]
            continue
        if line.startswith("#"): continue
        c=line.split("\t")
        if len(c)>=2 and c[0].isdigit(): cur.append(c[1])
    if cur: sents.append(cur)
    return sents

@dataclass
class LM:
    name:str; alph:str; logtri:np.ndarray; freq:np.ndarray; control_words:list[list[int]]; meta:dict[str,Any]

def build_lm(name:str,sents:list[list[str]],kind:str)->LM:
    alph=R12 if kind=="r12" else PLAIN; A=len(alph); B=A
    C=np.full((A+1,A+1,A+1),0.25,dtype=np.float64); F=np.full(A,0.25,dtype=np.float64)
    ctr=[]; nw=nl=0
    for i,s in enumerate(sents):
        istrain=i%10 in TRAIN_RES; isctrl=i%10 in CTRL_RES
        for raw in s:
            if kind=="r12": w=norm_r12(raw)
            else:
                w=norm_std(raw)
                if kind=="vc": w=vc_word(w)
            if not w: continue
            if isctrl: ctr.append(w)
            if not istrain: continue
            nw+=1; nl+=len(w)
            for x in w: F[x]+=1
            z=[B,B]+w+[B,B]
            for a,b,c in zip(z,z[1:],z[2:]): C[a,b,c]+=1
    C/=C.sum(axis=2,keepdims=True); F/=F.sum()
    return LM(name,alph,np.log(C),F,ctr,{"kind":kind,"train_words":nw,"train_letters":nl,"control_words":len(ctr),"control_letters":sum(map(len,ctr))})

def load_lms(smoke=False):
    names=LANGS if not smoke else ["latin","italian","german"]
    std={}; vc={}; rawsets={}
    for n in names:
        s=parse_ud(getb(URLS[n])); rawsets[n]=s; std[n]=build_lm(n,s,"std"); vc[n]=build_lm(n,s,"vc")
        print("LM",n,json.dumps(std[n].meta,sort_keys=True),flush=True)
    r12=build_lm("italian",rawsets["italian"],"r12")
    return std,vc,r12

def span(words,tag,fitn,holdn):
    st=seed(NS,"span",tag)%len(words); fit=[]; hold=[]; nf=nh=0; j=0
    while nh<holdn:
        w=words[(st+j)%len(words)]; j+=1
        if nf<fitn: fit.append(w); nf+=len(w)
        else: hold.append(w); nh+=len(w)
        if j>len(words)*30: raise RuntimeError("control span exhausted")
    return fit,hold

@dataclass
class TStats:
    a:np.ndarray; b:np.ndarray; c:np.ndarray; n:np.ndarray; offsets:np.ndarray; adj:np.ndarray; tfreq:np.ndarray; chars:int; nstate:int; boundary:int

def state_words(words:list[list[int]],mode:str,rule:int|None,meta:list[int]|None=None):
    out=[]
    for wi,w in enumerate(words):
        if mode in ("M0","VC"): st=[0]*len(w)
        elif mode=="PWA": st=[j%int(rule) for j in range(len(w))]
        elif mode=="GH": st=[int(meta[wi])]*len(w)
        else: raise ValueError(mode)
        out.append(st)
    return out

def make_stats(words,states,nstate):
    bd=nstate*K; cnt=collections.Counter(); tf=np.zeros(bd+1,dtype=np.int64); chars=0
    for w,ss in zip(words,states):
        toks=[bd,bd]+[s*K+x for x,s in zip(w,ss)]+[bd,bd]; chars+=len(w)
        for t in toks:
            if t!=bd: tf[t]+=1
        for x,y,z in zip(toks,toks[1:],toks[2:]): cnt[(x,y,z)]+=1
    ks=list(cnt); a=np.array([x[0] for x in ks],np.int32); b=np.array([x[1] for x in ks],np.int32); c=np.array([x[2] for x in ks],np.int32); n=np.array([cnt[x] for x in ks],np.int64)
    lists=[[] for _ in range(bd)]
    for i,(x,y,z) in enumerate(ks):
        for t in set((x,y,z)):
            if t!=bd: lists[t].append(i)
    off=[0]; flat=[]
    for q in lists: flat.extend(q); off.append(len(flat))
    return TStats(a,b,c,n,np.array(off,np.int32),np.array(flat,np.int32),tf,chars,nstate,bd)

@njit(nogil=True,cache=False)
def score_sparse(a,b,c,n,dec,logp):
    z=0.0
    for i in range(len(n)): z+=n[i]*logp[dec[a[i]],dec[b[i]],dec[c[i]]]
    return z

@njit(nogil=True,cache=False)
def dswap(a,b,c,n,dec,off,adj,ta,tb,logp):
    va=dec[ta]; vb=dec[tb]; d=0.0
    for jj in range(off[ta],off[ta+1]):
        i=adj[jj]; x=a[i]; y=b[i]; z=c[i]
        ox=dec[x]; oy=dec[y]; oz=dec[z]
        nx=vb if x==ta else (va if x==tb else ox); ny=vb if y==ta else (va if y==tb else oy); nz=vb if z==ta else (va if z==tb else oz)
        d+=n[i]*(logp[nx,ny,nz]-logp[ox,oy,oz])
    for jj in range(off[tb],off[tb+1]):
        i=adj[jj]; x=a[i]; y=b[i]; z=c[i]
        if x==ta or y==ta or z==ta: continue
        ox=dec[x]; oy=dec[y]; oz=dec[z]
        nx=va if x==tb else ox; ny=va if y==tb else oy; nz=va if z==tb else oz
        d+=n[i]*(logp[nx,ny,nz]-logp[ox,oy,oz])
    return d

@njit(nogil=True,cache=False)
def done(a,b,c,n,dec,off,adj,t,newv,logp):
    old=dec[t]; d=0.0
    for jj in range(off[t],off[t+1]):
        i=adj[jj]; x=a[i]; y=b[i]; z=c[i]
        ox=dec[x]; oy=dec[y]; oz=dec[z]
        nx=newv if x==t else ox; ny=newv if y==t else oy; nz=newv if z==t else oz
        d+=n[i]*(logp[nx,ny,nz]-logp[ox,oy,oz])
    return d

@njit(nogil=True,cache=False)
def rngstep(s):
    s^=s>>np.uint64(12); s^=s<<np.uint64(25); s^=s>>np.uint64(27); return s*np.uint64(2685821657736338717)
@njit(nogil=True,cache=False)
def rint(s,u): s=rngstep(s); return s,int(s%np.uint64(u))
@njit(nogil=True,cache=False)
def rfloat(s): s=rngstep(s); return s,float(s>>np.uint64(11))*(1.0/9007199254740992.0)

@njit(nogil=True,cache=False)
def anneal_bij(a,b,c,n,off,adj,tf,init,logp,nstate,props,sd):
    dec=init.copy(); bd=nstate*K; dec[bd]=logp.shape[0]-1; sc=score_sparse(a,b,c,n,dec,logp); best=sc; bestd=dec.copy(); state=np.uint64(sd or 1)
    ma=0.0; nn=0
    for _ in range(64):
        state,s=rint(state,nstate); state,x=rint(state,K); state,y=rint(state,K)
        if x==y: continue
        dd=dswap(a,b,c,n,dec,off,adj,s*K+x,s*K+y,logp); ma+=abs(dd); nn+=1
    ma/=max(1,nn); t0=max(0.03,2.5*ma); te=max(0.0003,0.01*ma); cool=math.exp(math.log(te/t0)/max(1,props)); temp=t0
    for _ in range(props):
        state,s=rint(state,nstate); state,x=rint(state,K); state,y=rint(state,K)
        if x==y: temp*=cool; continue
        ta=s*K+x; tb=s*K+y; dd=dswap(a,b,c,n,dec,off,adj,ta,tb,logp); ok=dd>=0
        if not ok: state,u=rfloat(state); ok=u<math.exp(dd/max(temp,1e-12))
        if ok:
            q=dec[ta]; dec[ta]=dec[tb]; dec[tb]=q; sc+=dd
            if sc>best: best=sc; bestd=dec.copy()
        temp*=cool
    dec=bestd.copy(); sc=best
    for _ in range(3):
        imp=False
        for s in range(nstate):
            for x in range(K-1):
                for y in range(x+1,K):
                    ta=s*K+x; tb=s*K+y; dd=dswap(a,b,c,n,dec,off,adj,ta,tb,logp)
                    if dd>1e-8:
                        q=dec[ta]; dec[ta]=dec[tb]; dec[tb]=q; sc+=dd; imp=True
        if not imp: break
    return dec,sc

@njit(nogil=True,cache=False)
def anneal_hom(a,b,c,n,off,adj,tf,init,logp,props,sd):
    dec=init.copy(); bd=K; dec[bd]=L12; counts=np.zeros(L12,np.int32)
    for t in range(K): counts[dec[t]]+=1
    sc=score_sparse(a,b,c,n,dec,logp); best=sc; bestd=dec.copy(); state=np.uint64(sd or 1)
    ma=0.0; nn=0
    for _ in range(64):
        state,t=rint(state,K); state,v=rint(state,L12)
        if v==dec[t] or counts[dec[t]]<=1: continue
        dd=done(a,b,c,n,dec,off,adj,t,v,logp); ma+=abs(dd); nn+=1
    ma/=max(1,nn); t0=max(0.03,2.5*ma); te=max(0.0003,0.01*ma); cool=math.exp(math.log(te/t0)/max(1,props)); temp=t0
    for _ in range(props):
        state,t=rint(state,K); state,v=rint(state,L12); old=dec[t]
        if v==old or counts[old]<=1: temp*=cool; continue
        dd=done(a,b,c,n,dec,off,adj,t,v,logp); ok=dd>=0
        if not ok: state,u=rfloat(state); ok=u<math.exp(dd/max(temp,1e-12))
        if ok:
            counts[old]-=1; counts[v]+=1; dec[t]=v; sc+=dd
            if sc>best: best=sc; bestd=dec.copy()
        temp*=cool
    dec=bestd.copy(); sc=best; counts[:]=0
    for t in range(K): counts[dec[t]]+=1
    for _ in range(3):
        imp=False
        for t in range(K):
            old=dec[t]
            if counts[old]<=1: continue
            for v in range(L12):
                if v==old: continue
                dd=done(a,b,c,n,dec,off,adj,t,v,logp)
                if dd>1e-8:
                    counts[old]-=1; counts[v]+=1; dec[t]=v; sc+=dd; old=v; imp=True
        if not imp: break
    return dec,sc

def init_bij(st:TStats,lm:LM,tag,random=False):
    dec=np.empty(st.boundary+1,np.int32); dec[st.boundary]=len(lm.alph)
    for s in range(st.nstate):
        if random:
            rg=np.random.default_rng(seed(NS,"initbij",tag,s)); p=rg.permutation(K)
            for x in range(K): dec[s*K+x]=p[x]
        else:
            obs=sorted(range(K),key=lambda x:(-int(st.tfreq[s*K+x]),x)); pl=sorted(range(K),key=lambda x:(-float(lm.freq[x]),x))
            for x,p in zip(obs,pl): dec[s*K+x]=p
    return dec

def init_hom(st:TStats,lm:LM,tag,random=False):
    dec=np.empty(K+1,np.int32); dec[K]=L12
    rg=np.random.default_rng(seed(NS,"inithom",tag))
    if random:
        obs=list(rg.permutation(K)); lat=list(rg.permutation(L12))
    else:
        obs=sorted(range(K),key=lambda x:(-int(st.tfreq[x]),x)); lat=sorted(range(L12),key=lambda x:(-float(lm.freq[x]),x))
    for j,t in enumerate(obs[:L12]): dec[t]=lat[j]
    for j,t in enumerate(obs[L12:]): dec[t]=lat[j%L12] if not random else int(rg.integers(0,L12))
    return dec

def agree(d1,d2,tf,boundary):
    den=max(1,int(tf[:boundary].sum())); return float(tf[:boundary][d1[:boundary]==d2[:boundary]].sum()/den)

def state_agree(d1,d2,tf,nstate):
    out=[]
    for s in range(nstate):
        lo=s*K; hi=lo+K; den=max(1,int(tf[lo:hi].sum())); out.append(float(tf[lo:hi][d1[lo:hi]==d2[lo:hi]].sum()/den))
    return out

def solve_bij(st,lm,tag,smoke=False):
    props=SMOKE_PROPOSALS if smoke else PROPOSALS; mx=SMOKE_RESTARTS if smoke else MAX_RESTARTS; batch=2 if smoke else BATCH
    bestd=[None,None]; best=[-1e300,-1e300]; used=[0,0]; conv=False; ag=0.0
    for end in range(batch,mx+1,batch):
        for e in (0,1):
            for r in range(used[e],end):
                init=init_bij(st,lm,f"{tag}:{e}:{r}",random=(r>0))
                d,sc=anneal_bij(st.a,st.b,st.c,st.n,st.offsets,st.adj,st.tfreq,init,lm.logtri,st.nstate,props,seed(NS,"annbij",tag,e,r))
                if sc>best[e]: best[e]=float(sc); bestd[e]=d.copy()
            used[e]=end
        ag=agree(bestd[0],bestd[1],st.tfreq,st.boundary); diff=abs(best[0]-best[1])/max(1,st.chars)
        if diff<=1e-7 and ag>=0.95: conv=True; break
    win=0 if best[0]>=best[1] else 1
    return {"dec":bestd[win],"fit_score":best[win]/max(1,st.chars),"agreement":ag,"state_agreement":state_agree(bestd[0],bestd[1],st.tfreq,st.nstate),"converged":conv,"score_diff":abs(best[0]-best[1])/max(1,st.chars),"restarts_each":used[0]}

def solve_hom(st,lm,tag,smoke=False):
    props=SMOKE_PROPOSALS if smoke else PROPOSALS; mx=SMOKE_RESTARTS if smoke else MAX_RESTARTS; batch=2 if smoke else BATCH
    bestd=[None,None]; best=[-1e300,-1e300]; used=[0,0]; conv=False; ag=0.0
    for end in range(batch,mx+1,batch):
        for e in (0,1):
            for r in range(used[e],end):
                init=init_hom(st,lm,f"{tag}:{e}:{r}",random=(r>0)); d,sc=anneal_hom(st.a,st.b,st.c,st.n,st.offsets,st.adj,st.tfreq,init,lm.logtri,props,seed(NS,"annhom",tag,e,r))
                if sc>best[e]: best[e]=float(sc); bestd[e]=d.copy()
            used[e]=end
        ag=agree(bestd[0],bestd[1],st.tfreq,st.boundary); diff=abs(best[0]-best[1])/max(1,st.chars)
        if diff<=1e-7 and ag>=0.95: conv=True; break
    win=0 if best[0]>=best[1] else 1
    return {"dec":bestd[win],"fit_score":best[win]/max(1,st.chars),"agreement":ag,"converged":conv,"score_diff":abs(best[0]-best[1])/max(1,st.chars),"restarts_each":used[0]}

def fixed_score(st,lm,dec): return float(score_sparse(st.a,st.b,st.c,st.n,dec,lm.logtri)/max(1,st.chars))

def decode(words,states,dec): return [[int(dec[s*K+x]) for x,s in zip(w,ss)] for w,ss in zip(words,states)]
def acc(truth,pred):
    a=[x for w in truth for x in w]; b=[x for w in pred for x in w]
    return 0.0 if len(a)!=len(b) else (sum(x==y for x,y in zip(a,b))/max(1,len(a)))

def metadata(n,tag):
    rg=np.random.default_rng(seed(NS,"meta",tag)); return [int(x) for x in rg.integers(0,5,n)]

def encrypt_bij(words,mode,rule,meta,tag):
    ns=1 if mode in ("M0","VC") else (int(rule) if mode=="PWA" else 5); rg=np.random.default_rng(seed(NS,"key",tag,mode,rule)); p2c=np.array([rg.permutation(K) for _ in range(ns)],np.int32)
    states=state_words(words,mode,rule,meta); out=[]; inv=np.empty_like(p2c)
    for s in range(ns):
        for p,c in enumerate(p2c[s]): inv[s,int(c)]=p
    for w,ss in zip(words,states): out.append([int(p2c[s,x]) for x,s in zip(w,ss)])
    return out,states,inv

def encrypt_hom(words,tag):
    rg=np.random.default_rng(seed(NS,"r12key",tag)); obs=list(rg.permutation(K)); assignment=np.empty(K,np.int32)
    for j,t in enumerate(obs[:L12]): assignment[t]=j
    for t in obs[L12:]: assignment[t]=int(rg.integers(0,L12))
    rev=[[] for _ in range(L12)]
    for o,v in enumerate(assignment): rev[int(v)].append(o)
    out=[]
    for wi,w in enumerate(words):
        r=np.random.default_rng(seed(NS,"r12emit",tag,wi)); out.append([int(rev[x][int(r.integers(0,len(rev[x])))]) for x in w])
    return out,assignment

def make_control(std,vc,r12,fam,rule,lang,rep,stage,fitn=1600,holdn=1600):
    lm=r12 if fam=="R12H" else (vc[lang] if fam=="VC" else std[lang]); fw,hw=span(lm.control_words,f"{stage}:{fam}:{rule}:{lang}:{rep}",fitn,holdn); mfit=metadata(len(fw),f"{stage}:{fam}:{rule}:{lang}:{rep}:fit"); mhold=metadata(len(hw),f"{stage}:{fam}:{rule}:{lang}:{rep}:hold")
    if fam=="R12H":
        fc,true=encrypt_hom(fw,f"{stage}:{fam}:{rule}:{lang}:{rep}:fitkey"); hc,_=encrypt_hom(hw,f"{stage}:{fam}:{rule}:{lang}:{rep}:holdkey")
        # Same mapping must be document-global: re-encrypt hold using fit assignment.
        rev=[[] for _ in range(L12)]
        for o,v in enumerate(true): rev[int(v)].append(o)
        hc=[]
        for wi,w in enumerate(hw):
            rg=np.random.default_rng(seed(NS,"r12holdemit",stage,fam,rule,lang,rep,wi)); hc.append([int(rev[x][int(rg.integers(0,len(rev[x])))]) for x in w])
        return {"fit_plain":fw,"hold_plain":hw,"fit":fc,"hold":hc,"mfit":mfit,"mhold":mhold,"truth":true}
    mode=fam; fc,fs,true=encrypt_bij(fw,mode,rule,mfit,f"{stage}:{fam}:{rule}:{lang}:{rep}")
    # apply same p2c to hold
    ns=true.shape[0]; p2c=np.empty_like(true)
    for s in range(ns):
        for c,p in enumerate(true[s]): p2c[s,p]=c
    hs=state_words(hw,mode,rule,mhold); hc=[[int(p2c[s,x]) for x,s in zip(w,ss)] for w,ss in zip(hw,hs)]
    return {"fit_plain":fw,"hold_plain":hw,"fit":fc,"hold":hc,"mfit":mfit,"mhold":mhold,"truth":true}

def cand_fit(ctl,std,vc,r12,fam,rule,lang,tag,smoke=False):
    if fam=="R12H":
        fs=make_stats(ctl["fit"],state_words(ctl["fit"],"M0",None),1); hs=make_stats(ctl["hold"],state_words(ctl["hold"],"M0",None),1); sol=solve_hom(fs,r12,tag,smoke); sc=fixed_score(hs,r12,sol["dec"]); pred=[[int(sol["dec"][x]) for x in w] for w in ctl["hold"]]
        return sol|{"hold_score":sc,"recovery":acc(ctl["hold_plain"],pred)}
    lm=vc[lang] if fam=="VC" else std[lang]; fstates=state_words(ctl["fit"],fam,rule,ctl["mfit"]); hstates=state_words(ctl["hold"],fam,rule,ctl["mhold"]); ns=1 if fam=="VC" else (int(rule) if fam=="PWA" else 5); fs=make_stats(ctl["fit"],fstates,ns); hs=make_stats(ctl["hold"],hstates,ns); sol=solve_bij(fs,lm,tag,smoke); sc=fixed_score(hs,lm,sol["dec"]); return sol|{"hold_score":sc,"recovery":acc(ctl["hold_plain"],decode(ctl["hold"],hstates,sol["dec"]))}

def baseline(ctl,std,lang,tag,smoke=False):
    fs=make_stats(ctl["fit"],state_words(ctl["fit"],"M0",None),1); hs=make_stats(ctl["hold"],state_words(ctl["hold"],"M0",None),1); sol=solve_bij(fs,std[lang],tag,smoke); return sol|{"hold_score":fixed_score(hs,std[lang],sol["dec"])}

def q0():
    ex={"non":"nno","staro":"strao","discorere":"dscrioee","differentia":"dffrntieia","riputatione":"rpttnuaioe","competitori":"cmpttroeioi","et":"te","il":"li","splendore":"splndreoe","uestitto":"stttueio"}
    vcok={}
    for p,e in ex.items(): vcok[p]="".join(PLAIN[x] for x in vc_word(norm_std(p)))==e
    rex={"labro":"lauro","grande":"grante","felice":"elice","gioue":"ioue","pietro":"ietro","quando":"cuando","mouendo":"mooendo"}; rok={}
    for p,e in rex.items(): rok[p]="".join(R12[x] for x in norm_r12(p))==e
    return {"VC_END":vcok,"R12_V1_024":rok,"pass":all(vcok.values()) and all(rok.values())}

def q1(std,vc,r12,smoke=False,workers=1):
    jobs=[("R12H",1,"italian",r) for r in range(3)]+[("VC",1,l,r) for l in (["latin"] if smoke else ["latin","italian","german"]) for r in range(3)]+[("PWA",k,l,r) for k in PWA_RULES for l in (["latin"] if smoke else ["latin","italian","german"]) for r in range(3)]+[("GH",5,l,r) for l in (["latin"] if smoke else ["latin","italian","german"]) for r in range(3)]
    if smoke: jobs=jobs[:1]+jobs[3:4]+jobs[12:13]+jobs[-1:]
    def one(j):
        fam,rule,lang,rep=j; ctl=make_control(std,vc,r12,fam,rule,lang,rep,"Q1",320 if smoke else 1200,320 if smoke else 1200); r=cand_fit(ctl,std,vc,r12,fam,rule,lang,f"Q1:{fam}:{rule}:{lang}:{rep}",smoke)
        return {"family":fam,"rule":rule,"language":lang,"rep":rep,"recovery":r["recovery"],"agreement":r["agreement"],"state_agreement":r.get("state_agreement"),"converged":r["converged"],"hold_score":r["hold_score"]}
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(one,jobs): rows.append(r); print("Q1",json.dumps(r,sort_keys=True),flush=True)
    gates={}
    for fam in ["R12H","VC","PWA","GH"]:
        z=[x for x in rows if x["family"]==fam]; rules=sorted(set(x["rule"] for x in z)); ok=bool(z)
        for rr in rules:
            q=[x for x in z if x["rule"]==rr]; rec=[x["recovery"] for x in q]; agr=[x["agreement"] for x in q]; ok &= all(x["converged"] for x in q) and statistics.median(rec)>=.95 and min(rec)>=.85 and statistics.median(agr)>=.95 and min(agr)>=.90
        gates[fam]=bool(ok)
    return rows,gates

def q2(std,vc,r12,gates,smoke=False,workers=1):
    rows=[]
    # R12 separate, as frozen protocol permits.
    if gates.get("R12H"):
        for rep in range(2 if smoke else 8):
            ctl=make_control(std,vc,r12,"R12H",1,"italian",rep,"Q2R",400 if smoke else 1500,400 if smoke else 1500); r=cand_fit(ctl,std,vc,r12,"R12H",1,"italian",f"Q2R:{rep}",smoke); rows.append({"truth_family":"R12H","selected_family":"R12H","truth_rule":1,"selected_rule":1,"truth_language":"italian","selected_language":"italian","recovery":r["recovery"],"converged":r["converged"]})
    fams=[f for f in ["VC","PWA","GH"] if gates.get(f)]; controls=[]
    for fam in fams:
        langs=(["latin","italian"] if smoke else LANGS)
        for i,lang in enumerate(langs):
            rule=(PWA_RULES[i%4] if fam=="PWA" else (5 if fam=="GH" else 1)); controls.append((fam,rule,lang,i))
    def one(j):
        tf,tr,tl,rep=j; ctl=make_control(std,vc,r12,tf,tr,tl,rep,"Q2",400 if smoke else 1500,400 if smoke else 1500); cand=[]
        for fam in fams:
            rules=PWA_RULES if fam=="PWA" else ([5] if fam=="GH" else [1])
            for rule in rules:
                for lang in list(std):
                    r=cand_fit(ctl,std,vc,r12,fam,rule,lang,f"Q2:{tf}:{tr}:{tl}:{rep}:{fam}:{rule}:{lang}",smoke); cand.append((r["hold_score"],fam,rule,lang,r))
        cand.sort(key=lambda x:(-x[0],x[1],x[2],x[3])); x=cand[0]
        return {"truth_family":tf,"selected_family":x[1],"truth_rule":tr,"selected_rule":x[2],"truth_language":tl,"selected_language":x[3],"recovery":x[4]["recovery"],"converged":x[4]["converged"],"top_score":x[0]}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(one,controls): rows.append(r); print("Q2",json.dumps(r,sort_keys=True),flush=True)
    z=[r for r in rows if r["truth_family"]!="R12H"]; rz=[r for r in rows if r["truth_family"]=="R12H"]
    famacc=sum(r["truth_family"]==r["selected_family"] for r in z)/max(1,len(z)); langacc=sum(r["truth_language"]==r["selected_language"] for r in z)/max(1,len(z)); pw=[r for r in z if r["truth_family"]=="PWA"]; ruleacc=sum(r["truth_rule"]==r["selected_rule"] for r in pw)/max(1,len(pw)); med=statistics.median([r["recovery"] for r in z]) if z else 0
    perlang={l:(sum(r["selected_language"]==l for r in z if r["truth_language"]==l),sum(1 for r in z if r["truth_language"]==l)) for l in list(std)}
    ok=(famacc>=.90 and ruleacc>=.85 and langacc>=.90 and med>=.90 and all(n<4 or c/n>=.75 for c,n in perlang.values())) if z else False
    rok=bool(rz) and all(r["converged"] and r["recovery"]>=.85 for r in rz) and statistics.median(r["recovery"] for r in rz)>=.95
    return rows,{"multifamily_pass":bool(ok),"family_accuracy":famacc,"pwa_rule_accuracy":ruleacc,"language_accuracy":langacc,"median_recovery":med,"per_language":perlang,"R12H_pass":rok}

def p5(v): return float(np.quantile(np.array(v,float),.05,method="linear"))

def phase_shuffle_states(words,k,tag):
    out=[]
    for i,w in enumerate(words):
        ph=seed(NS,"phase",tag,i)%k; out.append([(j+ph)%k for j in range(len(w))])
    return out

def q3(std,vc,r12,gates,q2res,smoke=False,workers=1):
    active=[]
    if gates.get("R12H") and q2res.get("R12H_pass"): active.append("R12H")
    if q2res.get("multifamily_pass"):
        active += [f for f in ["VC","PWA","GH"] if gates.get(f)]
    jobs=[]
    for fam in active:
        langs=["italian"] if fam=="R12H" else list(std)
        for lang in langs:
            for rep in range(2 if smoke else 8): jobs.append((fam,lang,rep))
    def one(j):
        fam,lang,rep=j; true_rule=(PWA_RULES[rep%4] if fam=="PWA" else (5 if fam=="GH" else 1)); ctl=make_control(std,vc,r12,fam,true_rule,lang,rep,"Q3",500 if smoke else 1800,500 if smoke else 1800)
        if fam=="PWA":
            fits=[(cand_fit(ctl,std,vc,r12,fam,k,lang,f"Q3:{fam}:{lang}:{rep}:{k}",smoke),k) for k in PWA_RULES]; r,k=max(fits,key=lambda q:q[0]["fit_score"])
        else: r=cand_fit(ctl,std,vc,r12,fam,true_rule,lang,f"Q3:{fam}:{lang}:{rep}",smoke); k=true_rule
        b=baseline(ctl,std,lang,f"Q3BASE:{fam}:{lang}:{rep}",smoke); out={"family":fam,"language":lang,"rep":rep,"truth_rule":true_rule,"selected_rule":k,"score":r["hold_score"],"baseline":b["hold_score"],"delta":r["hold_score"]-b["hold_score"],"recovery":r["recovery"],"converged":r["converged"]}
        if fam=="PWA":
            hs=make_stats(ctl["hold"],phase_shuffle_states(ctl["hold"],k,f"Q3:{lang}:{rep}:{k}"),k); out["phase_score"]=fixed_score(hs,std[lang],r["dec"]); out["reset_delta"]=r["hold_score"]-out["phase_score"]
        return out
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(one,jobs): rows.append(r); print("Q3",json.dumps(r,sort_keys=True),flush=True)
    cells={}
    for fam in active:
        langs=["italian"] if fam=="R12H" else list(std)
        for lang in langs:
            z=[r for r in rows if r["family"]==fam and r["language"]==lang]
            if not z: continue
            d={"ABS_FLOOR":p5([x["score"] for x in z]),"DELTA_FLOOR":p5([x["delta"] for x in z]),"median_recovery":statistics.median(x["recovery"] for x in z),"all_converged":all(x["converged"] for x in z)}
            if fam=="PWA": d["RESET_DELTA_FLOOR"]=p5([x["reset_delta"] for x in z]); d["rule_accuracy"]=sum(x["truth_rule"]==x["selected_rule"] for x in z)/len(z)
            cells[f"{fam}:{lang}"]=d
    return rows,{"active":active,"cells":cells}

def neg_words(std,kind,rep,total=4200):
    lm=std["italian"]; rg=np.random.default_rng(seed(NS,"neg",kind,rep)); lens=np.array([len(w) for w in lm.control_words if 1<=len(w)<=16],int); freq=lm.freq/lm.freq.sum(); out=[]; n=0; prev1=prev2=int(rg.choice(K,p=freq))
    # transition extracted from order-3 LM but successor rows are deterministically permuted to avoid natural-language identity.
    while n<total:
        L=int(lens[int(rg.integers(0,len(lens)))]); w=[]
        if kind=="iid": w=[int(x) for x in rg.choice(K,L,p=freq)]
        elif kind=="markov2":
            for j in range(L):
                base=np.exp(lm.logtri[prev2,prev1,:K]); base=base/base.sum(); sh=(rep*7+j*3+5)%K; p=np.roll(base,sh); x=int(rg.choice(K,p=p)); w.append(x); prev2,prev1=prev1,x
        elif kind=="motif":
            mot=[int(x) for x in rg.choice(K,max(1,min(4,L)),p=freq)]; w=(mot*((L+len(mot)-1)//len(mot)))[:L]
            for j in range(L):
                if rg.random()<.12: w[j]=int(rg.choice(K,p=freq))
        elif kind=="copy":
            if out and rg.random()<.75:
                src=list(out[int(rg.integers(0,len(out)))]); w=(src+[int(rg.choice(K,p=freq))]*L)[:L]
                for j in range(len(w)):
                    if rg.random()<.16: w[j]=int(rg.choice(K,p=freq))
            else: w=[int(x) for x in rg.choice(K,L,p=freq)]
        elif kind=="slot":
            classes=[list(range(0,5)),list(range(5,10)),list(range(10,15)),list(range(15,19))]
            w=[int(rg.choice(classes[j%4])) for j in range(L)]
        out.append(w); n+=L
    return out

def neg_eval(std,vc,r12,q3res,kind,rep,smoke=False):
    words=neg_words(std,kind,rep,1200 if smoke else 4200); cut=len(words)//2; ctl={"fit":words[:cut],"hold":words[cut:],"fit_plain":[],"hold_plain":[],"mfit":metadata(cut,f"N:{kind}:{rep}:f"),"mhold":metadata(len(words)-cut,f"N:{kind}:{rep}:h")}; flags=[]; detail=[]
    for fam in q3res["active"]:
        langs=["italian"] if fam=="R12H" else list(std); best=None
        for lang in langs:
            rules=PWA_RULES if fam=="PWA" else ([5] if fam=="GH" else [1])
            for rule in rules:
                r=cand_fit(ctl,std,vc,r12,fam,rule,lang,f"Q4:{kind}:{rep}:{fam}:{lang}:{rule}",smoke)
                if best is None or r["fit_score"]>best[0]: best=(r["fit_score"],fam,rule,lang,r)
        _,fam,rule,lang,r=best; b=baseline(ctl,std,lang,f"Q4BASE:{kind}:{rep}:{fam}:{lang}",smoke); cell=q3res["cells"][f"{fam}:{lang}"]; ok=r["hold_score"]>=cell["ABS_FLOOR"] and r["hold_score"]-b["hold_score"]>=cell["DELTA_FLOOR"]
        if fam=="PWA":
            hs=make_stats(ctl["hold"],phase_shuffle_states(ctl["hold"],rule,f"Q4:{kind}:{rep}:{rule}"),rule); rd=r["hold_score"]-fixed_score(hs,std[lang],r["dec"]); ok &= rd>=cell["RESET_DELTA_FLOOR"]
        detail.append({"family":fam,"rule":rule,"language":lang,"score":r["hold_score"],"delta":r["hold_score"]-b["hold_score"],"positive":bool(ok)}); flags.append(bool(ok))
    return {"kind":kind,"rep":rep,"positive":any(flags),"details":detail}

def q4(std,vc,r12,q3res,smoke=False,workers=1):
    kinds=["iid","markov2","motif","copy","slot"]; jobs=[(k,r) for k in kinds for r in range(2 if smoke else 16)]
    rows=[]
    def one(x): return neg_eval(std,vc,r12,q3res,x[0],x[1],smoke)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(one,jobs): rows.append(r); print("Q4",json.dumps({"kind":r["kind"],"rep":r["rep"],"positive":r["positive"]},sort_keys=True),flush=True)
    fp=sum(x["positive"] for x in rows); by={k:sum(x["positive"] for x in rows if x["kind"]==k) for k in kinds}; ok=fp<=(1 if smoke else 2) and all(v<=1 for v in by.values())
    return rows,{"pass":bool(ok),"false_positives":fp,"by_generator":by,"trials":len(rows)}

def parse_rf():
    b=getb(RF_URL,HEADERS); sha=hashlib.sha256(b).hexdigest()
    if sha!=RF_SHA: raise RuntimeError("RF hash mismatch")
    pages=collections.defaultdict(list); total=ret=rawc=retw=unc=rare=0
    for line in b.decode("utf-8","replace").splitlines():
        if not line.startswith("<") or ">" not in line: continue
        lab,rhs=line.split(">",1)
        if "." not in lab or "<!" in rhs: continue
        pg=lab[1:].split(".",1)[0]; rhs=re.sub(r"<(?:-|~)>",".",rhs); rhs=re.sub(r"<[^>]*>",".",rhs); rhs=rhs.replace(",","")
        for rw in rhs.split("."):
            rw=rw.strip()
            if not rw: continue
            rawc+=1; letters=[c for c in rw.lower() if "a"<=c<="z"]; total+=len(letters)
            if "[" in rw or "]" in rw or "?" in rw: unc+=1; continue
            ch=[c for c in rw.replace("{","").replace("}","").lower() if "a"<=c<="z"]
            if not ch: continue
            if any(c not in S2I for c in ch): rare+=1; continue
            w=[S2I[c] for c in ch]; pages[pg].append(w); retw+=1; ret+=len(w)
    return dict(pages),{"sha256":sha,"pages":len(pages),"raw_words":rawc,"retained_words":retw,"total_alpha":total,"retained_alpha":ret,"coverage":ret/max(1,total),"uncertain_words":unc,"rare_words":rare}

def target_split(pages):
    fs=sorted(pages,key=lambda f:hashlib.sha256(f"CIPHERCLOSEV1split::{f}".encode()).digest()); n=len(fs); nt=round(.60*n); nh=round(.20*n); T=fs[:nt]; H=fs[nt:nt+nh]; C1=fs[nt+nh:]
    c=sorted(C1,key=lambda f:hashlib.sha256(f"{NS}::{f}".encode()).digest()); h2=c[:len(c)//2]; c2=c[len(c)//2:]; return T,H,C1,h2,c2

def combine(pages,fs): return [w for f in fs for w in pages[f]]

def manifest():
    pages,meta=parse_rf(); T,H,C1,H2,C2=target_split(pages); chars=lambda q:sum(len(x) for f in q for x in pages[f]); return {"source":meta,"FIT_A":{"folios":T+H,"count":len(T)+len(H),"chars":chars(T+H)},"prior_C1":{"folios":C1,"count":len(C1),"chars":chars(C1)},"H2":{"folios":H2,"count":len(H2),"chars":chars(H2)},"C2":{"folios":C2,"count":len(C2),"chars":chars(C2)}}

def target_extract_gh(words):
    gl={S2I[x]:i for i,x in enumerate("ktpf")}; out=[]; meta=[]; amb=0; empty=0
    for w in words:
        pos=[j for j,x in enumerate(w) if x in gl]
        if not pos: out.append(list(w)); meta.append(4); continue
        if len(pos)>1: amb+=1
        j=pos[0]; meta.append(gl[w[j]]); q=w[:j]+w[j+1:]; out.append(q); empty+=int(not q)
    return out,meta,{"multi_gallows_ambiguous":amb,"empty_payload":empty,"words":len(words),"selector_counts":dict(collections.Counter(meta))}

def target_fit_family(fitw,h2w,std,vc,r12,fam,q3res,smoke=False):
    cells=q3res["cells"]; best=None; fitmeta=metadata(len(fitw),"TARGET-DUMMY-FIT"); hmeta=metadata(len(h2w),"TARGET-DUMMY-H2")
    if fam=="GH": fitx,fm,fcen=target_extract_gh(fitw); h2x,hm,hcen=target_extract_gh(h2w)
    else: fitx,h2x,fm,hm=fitw,h2w,fitmeta,hmeta; fcen=hcen=None
    langs=["italian"] if fam=="R12H" else list(std); rules=PWA_RULES if fam=="PWA" else ([5] if fam=="GH" else [1])
    for lang in langs:
        for rule in rules:
            ctl={"fit":fitx,"hold":h2x,"mfit":fm,"mhold":hm,"fit_plain":[],"hold_plain":[]}
            # target candidate solve without recovery
            if fam=="R12H":
                fs=make_stats(fitx,state_words(fitx,"M0",None),1); hs=make_stats(h2x,state_words(h2x,"M0",None),1); sol=solve_hom(fs,r12,f"TARGET:{fam}:{lang}",smoke); hsco=fixed_score(hs,r12,sol["dec"])
            else:
                lmo=vc[lang] if fam=="VC" else std[lang]; fss=state_words(fitx,fam,rule,fm); hss=state_words(h2x,fam,rule,hm); ns=1 if fam=="VC" else (rule if fam=="PWA" else 5); fs=make_stats(fitx,fss,ns); hs=make_stats(h2x,hss,ns); sol=solve_bij(fs,lmo,f"TARGET:{fam}:{rule}:{lang}",smoke); hsco=fixed_score(hs,lmo,sol["dec"])
            q=(sol["fit_score"],lang,rule,sol,hsco,ctl)
            if best is None or q[0]>best[0]: best=q
    fit_score,lang,rule,sol,hsco,ctl=best; b=baseline(ctl,std,lang,f"TARGETBASE:{fam}:{lang}",smoke); cell=cells[f"{fam}:{lang}"]; delta=hsco-b["hold_score"]; out={"family":fam,"language":lang,"rule":rule,"fit_score":fit_score,"H2_score":hsco,"baseline_H2":b["hold_score"],"delta":delta,"ABS_FLOOR":cell["ABS_FLOOR"],"DELTA_FLOOR":cell["DELTA_FLOOR"],"agreement":sol["agreement"],"state_agreement":sol.get("state_agreement"),"converged":sol["converged"],"source_census_fit":fcen,"source_census_H2":hcen}
    out["abs_pass"]=hsco>=cell["ABS_FLOOR"]; out["delta_pass"]=delta>=cell["DELTA_FLOOR"]
    if fam=="PWA":
        hs=make_stats(ctl["hold"],phase_shuffle_states(ctl["hold"],rule,f"TARGET:{rule}"),rule); rd=hsco-fixed_score(hs,std[lang],sol["dec"]); out["reset_delta"]=rd; out["RESET_DELTA_FLOOR"]=cell["RESET_DELTA_FLOOR"]; out["reset_pass"]=rd>=cell["RESET_DELTA_FLOOR"]
    return out,best

def gh_permutation_gate(best,std):
    _,lang,rule,sol,real,ctl=best; hm=list(ctl["mhold"]); words=ctl["hold"]; realdiff=[]
    # Deterministic within-pseudo-folio blocks of 64 words; target driver substitutes true folio ordering via stable blocks.
    # This preserves class counts locally and is pre-frozen before H2.
    for rep in range(256):
        pm=hm.copy()
        for st in range(0,len(pm),64):
            rg=np.random.default_rng(seed(NS,"ghperm",rep,st)); q=np.array(pm[st:st+64]); rg.shuffle(q); pm[st:st+64]=[int(x) for x in q]
        hs=make_stats(words,state_words(words,"GH",5,pm),5); realdiff.append(fixed_score(hs,std[lang],sol["dec"]))
    p99=float(np.quantile(np.array(realdiff),.99,method="linear")); return {"real_score":real,"permuted_p99":p99,"pass":real>p99,"permuted_max":max(realdiff),"n":256}

def run_target(std,vc,r12,qual,smoke=False):
    man=manifest(); pages,_=parse_rf(); fitfs=man["FIT_A"]["folios"]; h2fs=man["H2"]["folios"]; fitw=combine(pages,fitfs); h2w=combine(pages,h2fs); out={"manifest":man,"families":{},"C2_opened":False}
    q3r=qual["q3_summary"]; active=q3r["active"] if qual["q4_summary"]["pass"] else []
    for fam in active:
        r,best=target_fit_family(fitw,h2w,std,vc,r12,fam,q3r,smoke); r["q4_pass"]=qual["q4_summary"]["pass"]
        if fam=="GH" and r["abs_pass"] and r["delta_pass"]: r["selector_permutation_gate"]=gh_permutation_gate(best,std)
        positive=r["abs_pass"] and r["delta_pass"] and r["converged"] and qual["q4_summary"]["pass"]
        if fam=="PWA": positive &= r.get("reset_pass",False)
        if fam=="GH": positive &= r.get("selector_permutation_gate",{}).get("pass",False)
        r["verdict"]="H2_CANDIDATE" if positive else ("CLOSED_NEGATIVE_INCOMPATIBLE_V1" if not r["abs_pass"] and r["converged"] else "COMPATIBLE_NONSPECIFIC")
        out["families"][fam]=r
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=["preflight","smoke","qualify","target"],required=True); ap.add_argument("--workers",type=int,default=8); ap.add_argument("--qual-url"); a=ap.parse_args()
    if a.mode=="preflight": print("RESULT_JSON",json.dumps({"q0":q0(),"manifest":manifest()},sort_keys=True)); return
    smoke=a.mode=="smoke"; std,vc,r12=load_lms(smoke)
    if a.mode in ("smoke","qualify"):
        q0r=q0();
        if not q0r["pass"]: raise RuntimeError(("Q0 fail",q0r))
        q1r,g=q1(std,vc,r12,smoke,a.workers); q2r,q2s=q2(std,vc,r12,g,smoke,a.workers); q3r,q3s=q3(std,vc,r12,g,q2s,smoke,a.workers)
        if not q3s["active"]: res={"q0":q0r,"q1":q1r,"q1_gates":g,"q2":q2r,"q2_summary":q2s,"q3":q3r,"q3_summary":q3s,"q4":[],"q4_summary":{"pass":False,"reason":"NO_ACTIVE_FAMILIES"}}
        else:
            q4r,q4s=q4(std,vc,r12,q3s,smoke,a.workers); res={"q0":q0r,"q1":q1r,"q1_gates":g,"q2":q2r,"q2_summary":q2s,"q3":q3r,"q3_summary":q3s,"q4":q4r,"q4_summary":q4s}
        print("RESULT_JSON",json.dumps(res,sort_keys=True)); return
    if not a.qual_url: raise SystemExit("--qual-url required")
    qual=json.loads(getb(a.qual_url).decode()); print("RESULT_JSON",json.dumps(run_target(std,vc,r12,qual,False),sort_keys=True))
if __name__=="__main__": main()
