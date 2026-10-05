#!/usr/bin/env python3
# NK1 — strip Naibbe to a non-circular source -> complete-form-family kernel.
# Synthetic architecture calibration only. No Naibbe lookup strings. No semantic inversion.
import collections, hashlib, json, math, pickle, re, urllib.request
import numpy as np

SEED=20261005
NTRAIN=16000; NVAL=4000; NTEST=8000; NTOT=NTRAIN+NVAL+NTEST
TOPV=24; KFAM=4; ALPHA=0.80; MC_PER_SOURCE=4000; NNULL=500; NGENREP=8
RHO_GALLOWS=.25
K0URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f/research/select_form_socket_oracle_phaseK0_20261004.py"
CIURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/67a73f80da2caefd6788de43c003701225833825/Paper/Cipher_paper/ci_corpus_parsed.pkl"

k0={"__name__":"k0"}; exec(compile(urllib.request.urlopen(K0URL,timeout=60).read().decode(),K0URL,"exec"),k0)
ci=pickle.loads(urllib.request.urlopen(CIURL,timeout=60).read())
WORDS=[str(w).lower() for w in ci["all_words"] if re.fullmatch(r"[a-zA-Z]+",str(w)) and len(str(w))>1][:NTOT]
if len(WORDS)<NTOT: raise RuntimeError("short Circa Instans sequence")

# External source vocabulary fixed from training only.
vc=collections.Counter(WORDS[:NTRAIN]); vocab=[w for w,_ in vc.most_common(TOPV)]
VID={w:i for i,w in enumerate(vocab)}; OTHER=len(vocab); KSRC=OTHER+1
X=np.array([VID.get(w,OTHER) for w in WORDS],dtype=np.int16)

# Frozen FORM socket. Source controls use exactly the calibrated ENTRY/ROUTE strengths.
U,Ve0,Vr0=k0["controls"](KSRC,2,1.,np.random.default_rng(SEED+77),"F1")
Ve=Ve0*5.0; Vr=Vr0*.5
Ve_entry=Ve0*5.0; Vr_zero=np.zeros_like(Vr0)
PCLASS=k0["PCLASS"]; P_START_CLASS=k0["P_START_CLASS"]; P_START_PIECE=k0["P_START_PIECE"]
P_CONT=k0["P_CONT"]; LEGAL=k0["LEGAL"]; P_NEXT=k0["P_NEXT"]; P_STOP=k0["P_STOP"]
G_OF=k0["G_OF"]; START_IN=k0["START_IN"]; K12=k0["K12"]; NP=k0["NP"]; PIECES=k0["PIECES"]
GALS=set("fkpt")
PIECE_G=np.array([any(ch in GALS for ch in p) for p in PIECES],bool)

def row_soft(base,bias,mask):
    q=np.zeros_like(base,float);ok=mask & (base>0)
    x=np.log(np.maximum(base[ok],1e-300))+bias[ok];mx=x.max();v=np.exp(x-mx);v/=v.sum();q[ok]=v
    return q

def gen_token(x,VeX,VrX,rng,maxlen=30):
    be=U[x]@VeX;br=U[x]@VrX
    q=row_soft(P_START_CLASS,be,np.ones(K12,bool));c=int(rng.choice(K12,p=q))
    pp=int(rng.choice(NP,p=P_START_PIECE[c])); out=[pp];inc=START_IN; used=bool(PIECE_G[pp])
    for depth in range(maxlen):
        dep=min(depth,3)
        if rng.random()<P_STOP[pp,dep,inc]: return out
        g=G_OF[PCLASS[pp]]
        qd=row_soft(P_CONT[g],br,LEGAL[g]);dc=int(rng.choice(K12,p=qd))
        qn=P_NEXT[g,dc].copy()
        if used:
            qn[PIECE_G]*=RHO_GALLOWS
            s=qn.sum()
            if s<=0: return None
            qn/=s
        npp=int(rng.choice(NP,p=qn))
        inc=PCLASS[pp];pp=npp;out.append(pp);used=used or bool(PIECE_G[pp])
    return None

def token_loglik(tok,x,VeX,VrX):
    be=U[x]@VeX;br=U[x]@VrX
    c0=PCLASS[tok[0]]; q=row_soft(P_START_CLASS,be,np.ones(K12,bool))
    lp=math.log(max(q[c0],1e-300))+math.log(max(P_START_PIECE[c0,tok[0]],1e-300))
    inc=START_IN; used=bool(PIECE_G[tok[0]])
    for i,pp in enumerate(tok):
        dep=min(i,3);ps=P_STOP[pp,dep,inc]
        if i==len(tok)-1:
            lp+=math.log(max(ps,1e-300)); break
        lp+=math.log(max(1-ps,1e-300))
        g=G_OF[PCLASS[pp]];dc=PCLASS[tok[i+1]]
        qd=row_soft(P_CONT[g],br,LEGAL[g])
        lp+=math.log(max(qd[dc],1e-300))
        qn=P_NEXT[g,dc].copy()
        if used:
            qn[PIECE_G]*=RHO_GALLOWS
            qn/=max(qn.sum(),1e-300)
        lp+=math.log(max(qn[tok[i+1]],1e-300))
        inc=PCLASS[pp];used=used or bool(PIECE_G[tok[i+1]])
    return lp

def surf(tok):
    return "".join(PIECES[i] for i in tok)

def fam(tok):
    # One complete-token support coordinate defined only on frozen K12 path.
    # First class excluded so it cannot collapse to ENTRY. No exact-piece/token identity enters.
    cls=[int(PCLASS[p]) for p in tok]
    s=",".join(map(str,cls[1:]))+"|F"+str(cls[-1])
    h=hashlib.sha256(("NK1FAM|"+s).encode()).digest()
    return int.from_bytes(h[:8],"big")%KFAM

def sample_base(x,rng):
    for _ in range(200):
        t=gen_token(int(x),Ve,Vr,rng)
        if t is not None:return t
    raise RuntimeError("FORM nontermination")

# Estimate baseline family normalizers under ENTRY+ROUTE using Monte Carlo.
rng=np.random.default_rng(SEED+100)
PB=np.zeros((KSRC,KFAM),float)
for x in range(KSRC):
    cc=np.ones(KFAM)*.5
    for _ in range(MC_PER_SOURCE):
        cc[fam(sample_base(x,rng))]+=1
    PB[x]=cc/cc.sum()

# Externally generated low-dimensional source-family preference: random rank-2, not learned from target tokens.
rngp=np.random.default_rng(SEED+200)
Us=rngp.normal(size=(KSRC,2)); Us=(Us-Us.mean(0))/np.maximum(Us.std(0),1e-9)
Vf=rngp.normal(size=(2,KFAM)); Vf-=Vf.mean(1,keepdims=True); Vf/=np.maximum(Vf.std(),1e-9)
SCORE=Us@Vf
QTRUE=np.zeros_like(PB)
for x in range(KSRC):
    z=np.log(np.maximum(PB[x],1e-12))+ALPHA*SCORE[x];z-=z.max();q=np.exp(z);QTRUE[x]=q/q.sum()

def sample_target(x,rng):
    # sample complete support family, then realize locally with frozen FORM until it lands in that family
    ff=int(rng.choice(KFAM,p=QTRUE[x]))
    for _ in range(300):
        t=sample_base(x,rng)
        if fam(t)==ff:return t
    raise RuntimeError(("support rejection failed",x,ff,PB[x].tolist()))

rngt=np.random.default_rng(SEED+300)
TOK=[sample_target(int(x),rngt) for x in X]
F=np.array([fam(t) for t in TOK],dtype=np.int8)
SURF=[surf(t) for t in TOK]

# Fit only a 4-way family distribution per source from training occurrences, shrunk to frozen baseline PB.
C=np.zeros((KSRC,KFAM),float)
for x,f in zip(X[:NTRAIN],F[:NTRAIN]):C[x,f]+=1
PRIOR=8.0
QH=(C+PRIOR*PB);QH/=QH.sum(1,keepdims=True)

def lp_model(i,model,qh=QH):
    x=int(X[i]);tok=TOK[i]
    if model=="ENTRY":
        return token_loglik(tok,x,Ve_entry,Vr_zero)/math.log(2)
    b=token_loglik(tok,x,Ve,Vr)/math.log(2)
    if model=="ROUTE":return b
    f=int(F[i])
    return b+math.log2(max(qh[x,f],1e-300)/max(PB[x,f],1e-300))

test_ix=np.arange(NTRAIN+NVAL,NTOT)
val_ix=np.arange(NTRAIN,NTRAIN+NVAL)
L={m:np.array([lp_model(int(i),m) for i in test_ix]) for m in ("ENTRY","ROUTE","FAMILY")}
LV={m:np.array([lp_model(int(i),m) for i in val_ix]) for m in ("ENTRY","ROUTE","FAMILY")}

# Main effects as bits/token gain (higher log likelihood = fewer bits).
gain_route=L["ROUTE"]-L["ENTRY"]; gain_fam=L["FAMILY"]-L["ROUTE"]
val_gain_route=LV["ROUTE"]-LV["ENTRY"]; val_gain_fam=LV["FAMILY"]-LV["ROUTE"]

# Source-label permutation matched null for family gain: train family outcomes fixed, source labels permuted.
rngn=np.random.default_rng(SEED+400)
null=np.empty(NNULL,float)
for b in range(NNULL):
    xp=X[:NTRAIN].copy();rngn.shuffle(xp)
    cn=np.zeros((KSRC,KFAM),float)
    for x,f in zip(xp,F[:NTRAIN]):cn[x,f]+=1
    qn=cn+PRIOR*PB;qn/=qn.sum(1,keepdims=True)
    vv=[]
    for i in test_ix:
        x=int(X[i]);ff=int(F[i])
        vv.append(math.log2(max(qn[x,ff],1e-300)/max(PB[x,ff],1e-300)))
    null[b]=np.mean(vv)
null_mu=float(null.mean());null_sd=float(null.std(ddof=1));fam_mean=float(gain_fam.mean())
znull=(fam_mean-null_mu)/null_sd

# Block SD and novel-token restriction.
BS=400
blocks=[gain_fam[i:i+BS].mean() for i in range(0,len(gain_fam),BS)]
train_types=set(SURF[:NTRAIN]); novmask=np.array([SURF[int(i)] not in train_types for i in test_ix])
nov=gain_fam[novmask]
nov_blocks=[]
for st in range(0,len(test_ix),BS):
    z=gain_fam[st:st+BS];m=novmask[st:st+BS]
    if m.any():nov_blocks.append(z[m].mean())

# Matched null for novel restriction too.
null_nov=np.empty(NNULL,float)
for b in range(NNULL):
    xp=X[:NTRAIN].copy();rngn.shuffle(xp)
    cn=np.zeros((KSRC,KFAM),float)
    for x,f in zip(xp,F[:NTRAIN]):cn[x,f]+=1
    qn=cn+PRIOR*PB;qn/=qn.sum(1,keepdims=True)
    vv=[]
    for j,i in enumerate(test_ix):
        if not novmask[j]:continue
        x=int(X[i]);ff=int(F[i])
        vv.append(math.log2(max(qn[x,ff],1e-300)/max(PB[x,ff],1e-300)))
    null_nov[b]=np.mean(vv)
nmu=float(null_nov.mean());nsd=float(null_nov.std(ddof=1));nmean=float(nov.mean());nz=(nmean-nmu)/nsd

# Baseline-vs-target morphology preservation.
rngb=np.random.default_rng(SEED+500)
BT=[sample_base(int(x),rngb) for x in X[NTRAIN+NVAL:]]
def hist_len(ts):
    h=np.zeros(8,float)
    for t in ts:h[min(len(t),8)-1]+=1
    return h/h.sum()
ht=hist_len([TOK[i] for i in test_ix]); hb=hist_len(BT)
len_tv=.5*float(np.abs(ht-hb).sum())
mean_target=float(np.mean([len(TOK[i]) for i in test_ix]));mean_base=float(np.mean([len(t) for t in BT]))
def multi_g(ts):
    return float(np.mean([sum(any(ch in GALS for ch in PIECES[p]) for p in t)>=2 for t in ts]))
mg_t=multi_g([TOK[i] for i in test_ix]);mg_b=multi_g(BT)

# Generate from learned family kernel; compare diversity/unseen/length with target.
rep=[]
for rr in range(NGENREP):
    rg=np.random.default_rng(SEED+600+rr);gg=[]
    for x in X[NTRAIN+NVAL:]:
        ff=int(rg.choice(KFAM,p=QH[int(x)]))
        for _ in range(300):
            t=sample_base(int(x),rg)
            if fam(t)==ff:gg.append(t);break
        else:raise RuntimeError("learned generation rejection")
    ss=[surf(t) for t in gg];hh=hist_len(gg)
    rep.append({
      "types":len(set(ss)),"unseen_share":float(np.mean([s not in train_types for s in ss])),
      "mean_piece_len":float(np.mean([len(t) for t in gg])),
      "length_tv_vs_target":.5*float(np.abs(hh-ht).sum()),
      "multi_gallows":multi_g(gg)
    })

target_types=len(set(SURF[i] for i in test_ix));target_unseen=float(novmask.mean())
out={
 "phase":"NK1","status":"complete","source":"Circa Instans external word identities",
 "n":{"train":NTRAIN,"val":NVAL,"test":NTEST,"source_vocab":KSRC},
 "socket":{"entry_strength":5.0,"route_strength":.5,"gallows_rho":RHO_GALLOWS},
 "support":{"kind":"4-way hash of complete frozen K12 path excluding first class","k":KFAM,"alpha_planted":ALPHA,
            "baseline_mc_per_source":MC_PER_SOURCE,"fit_prior":PRIOR,
            "no_exact_token_lookup":True,"no_naibbe_strings":True},
 "validation":{"route_gain_bits_per_token":float(val_gain_route.mean()),"family_gain_bits_per_token":float(val_gain_fam.mean())},
 "test":{
   "entry_bits_per_token":float(-L["ENTRY"].mean()),"route_bits_per_token":float(-L["ROUTE"].mean()),
   "family_bits_per_token":float(-L["FAMILY"].mean()),
   "route_gain_bits_per_token":float(gain_route.mean()),
   "family_gain_bits_per_token":fam_mean,
   "family_block_sd":float(np.std(blocks,ddof=1)),"family_block_z0":float(fam_mean/np.std(blocks,ddof=1)),
   "family_shuffle_null_mean":null_mu,"family_shuffle_null_sd":null_sd,"family_z_vs_shuffle":float(znull),
   "novel_n":int(novmask.sum()),"novel_share":target_unseen,"novel_family_gain_bits_per_token":nmean,
   "novel_shuffle_null_mean":nmu,"novel_shuffle_null_sd":nsd,"novel_z_vs_shuffle":float(nz),
   "novel_block_sd":float(np.std(nov_blocks,ddof=1)) if len(nov_blocks)>1 else None
 },
 "preservation":{
   "target_mean_piece_len":mean_target,"baseline_mean_piece_len":mean_base,"delta_mean_piece_len":mean_target-mean_base,
   "length_distribution_tv":len_tv,"target_multi_gallows":mg_t,"baseline_multi_gallows":mg_b,
   "all_tokens_form_legal_by_construction":True
 },
 "generation":{
   "target_types":target_types,"target_unseen_share":target_unseen,
   "replicates":rep,
   "mean_types":float(np.mean([r["types"] for r in rep])),
   "mean_unseen_share":float(np.mean([r["unseen_share"] for r in rep])),
   "mean_length_tv_vs_target":float(np.mean([r["length_tv_vs_target"] for r in rep]))
 },
 "gates":{
   "family_vs_shuffle_gt2":bool(znull>2),
   "novel_family_vs_shuffle_gt2":bool(nz>2),
   "length_tv_le_0_03":bool(len_tv<=.03),
   "mean_piece_len_abs_delta_le_0_10":bool(abs(mean_target-mean_base)<=.10)
 }
}
print("NAIBBE_NK1_JSON="+json.dumps(out,separators=(",",":")),flush=True)
