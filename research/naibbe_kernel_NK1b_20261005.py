#!/usr/bin/env python3
# NK1b — length-preserving correction after NK1a failed length-TV gate.
# Fresh final source block 36k:44k was untouched by NK1a.
import collections, hashlib, json, math, pickle, re, urllib.request
import numpy as np
SEED=20261005
NTRAIN=16000; NVAL=4000; TEST0=36000; NTEST=8000; TOPV=24
KFAM=4; ALPHA=.8; MC_PER_SOURCE=5000; NNULL=300; NGENREP=3; PRIOR=8.; RHO=.25
K0URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f/research/select_form_socket_oracle_phaseK0_20261004.py"
CIURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/67a73f80da2caefd6788de43c003701225833825/Paper/Cipher_paper/ci_corpus_parsed.pkl"
k0={"__name__":"k0"};exec(compile(urllib.request.urlopen(K0URL,timeout=60).read().decode(),K0URL,"exec"),k0)
ci=pickle.loads(urllib.request.urlopen(CIURL,timeout=60).read())
W=[str(w).lower() for w in ci["all_words"] if re.fullmatch(r"[a-zA-Z]+",str(w)) and len(str(w))>1]
if len(W)<TEST0+NTEST:raise RuntimeError("short source")
vocab=[w for w,_ in collections.Counter(W[:NTRAIN]).most_common(TOPV)];VID={w:i for i,w in enumerate(vocab)}
OTHER=len(vocab);KSRC=OTHER+1
XTR=np.array([VID.get(w,OTHER) for w in W[:NTRAIN]],np.int16)
XVA=np.array([VID.get(w,OTHER) for w in W[NTRAIN:NTRAIN+NVAL]],np.int16)
XTE=np.array([VID.get(w,OTHER) for w in W[TEST0:TEST0+NTEST]],np.int16)

U,Ve0,Vr0=k0["controls"](KSRC,2,1.,np.random.default_rng(SEED+77),"F1")
Ve=Ve0*5.;Vr=Vr0*.5;VeE=Ve0*5.;Vr0z=np.zeros_like(Vr0)
PCLASS=k0["PCLASS"];PSC=k0["P_START_CLASS"];PSP=k0["P_START_PIECE"];PC=k0["P_CONT"];LEGAL=k0["LEGAL"];PN=k0["P_NEXT"];PSTOP=k0["P_STOP"]
G_OF=k0["G_OF"];START=k0["START_IN"];K12=k0["K12"];NP=k0["NP"];PIECES=k0["PIECES"]
GALS=set("fkpt");PG=np.array([any(c in GALS for c in p) for p in PIECES],bool)
def row_soft(base,bias,mask):
 q=np.zeros_like(base,float);ok=mask&(base>0);z=np.log(np.maximum(base[ok],1e-300))+bias[ok];z-=z.max();q[ok]=np.exp(z);q/=q.sum();return q
def gen(x,VEX,VRX,rng):
 be=U[x]@VEX;br=U[x]@VRX
 q=row_soft(PSC,be,np.ones(K12,bool));c=int(rng.choice(K12,p=q));pp=int(rng.choice(NP,p=PSP[c]));out=[pp];inc=START;used=bool(PG[pp])
 for d in range(30):
  if rng.random()<PSTOP[pp,min(d,3),inc]:return out
  g=G_OF[PCLASS[pp]];qd=row_soft(PC[g],br,LEGAL[g]);dc=int(rng.choice(K12,p=qd));qn=PN[g,dc].copy()
  if used:qn[PG]*=RHO;qn/=max(qn.sum(),1e-300)
  npid=int(rng.choice(NP,p=qn));inc=PCLASS[pp];pp=npid;out.append(pp);used=used or bool(PG[pp])
 return None
def sample(x,rng,VEX=Ve,VRX=Vr):
 for _ in range(200):
  t=gen(int(x),VEX,VRX,rng)
  if t is not None:return t
 raise RuntimeError("nonterm")
def ll(tok,x,VEX,VRX):
 be=U[x]@VEX;br=U[x]@VRX;c0=PCLASS[tok[0]];q=row_soft(PSC,be,np.ones(K12,bool))
 z=math.log(max(q[c0],1e-300))+math.log(max(PSP[c0,tok[0]],1e-300));inc=START;used=bool(PG[tok[0]])
 for i,pp in enumerate(tok):
  ps=PSTOP[pp,min(i,3),inc]
  if i==len(tok)-1:z+=math.log(max(ps,1e-300));break
  z+=math.log(max(1-ps,1e-300));g=G_OF[PCLASS[pp]];dc=PCLASS[tok[i+1]];qd=row_soft(PC[g],br,LEGAL[g]);z+=math.log(max(qd[dc],1e-300))
  qn=PN[g,dc].copy()
  if used:qn[PG]*=RHO;qn/=max(qn.sum(),1e-300)
  z+=math.log(max(qn[tok[i+1]],1e-300));inc=PCLASS[pp];used=used or bool(PG[tok[i+1]])
 return z
def sf(t):return "".join(PIECES[p] for p in t)
def fam(t):
 cls=[int(PCLASS[p]) for p in t];s=",".join(map(str,cls[1:]))+"|F"+str(cls[-1])
 return int.from_bytes(hashlib.sha256(("NK1FAM|"+s).encode()).digest()[:8],"big")%KFAM
def lb(t):return min(len(t),7)-1 # 1..6 exact, 7+
NB=7

# Baseline conditional family probabilities p(f | source,length-bucket).
rng=np.random.default_rng(SEED+100);cnt=np.ones((KSRC,NB,KFAM),float)*.5
for x in range(KSRC):
 for _ in range(MC_PER_SOURCE):
  t=sample(x,rng);cnt[x,lb(t),fam(t)]+=1
PB=cnt/cnt.sum(2,keepdims=True)

# Same rank-2 source family preference as NK1a; normalization is now inside frozen length bucket.
rp=np.random.default_rng(SEED+200);Us=rp.normal(size=(KSRC,2));Us=(Us-Us.mean(0))/np.maximum(Us.std(0),1e-9)
Vf=rp.normal(size=(2,KFAM));Vf-=Vf.mean(1,keepdims=True);Vf/=np.maximum(Vf.std(),1e-9);S=Us@Vf
QT=np.empty_like(PB)
for x in range(KSRC):
 for b in range(NB):
  z=np.log(np.maximum(PB[x,b],1e-12))+ALPHA*S[x];z-=z.max();q=np.exp(z);QT[x,b]=q/q.sum()

def target(x,rng):
 seed=sample(x,rng);b=lb(seed);ff=int(rng.choice(KFAM,p=QT[x,b]))
 if fam(seed)==ff:return seed
 for _ in range(800):
  t=sample(x,rng)
  if lb(t)==b and fam(t)==ff:return t
 raise RuntimeError(("conditional rejection",int(x),b,ff,PB[int(x),b].tolist()))

def make(xs,seed):
 r=np.random.default_rng(seed);return [target(int(x),r) for x in xs]
TTR=make(XTR,SEED+300);TVA=make(XVA,SEED+301);TTE=make(XTE,SEED+302)
FTR=np.array([fam(t) for t in TTR],np.int8);FVA=np.array([fam(t) for t in TVA],np.int8);FTE=np.array([fam(t) for t in TTE],np.int8)
BTR=np.array([lb(t) for t in TTR],np.int8);BVA=np.array([lb(t) for t in TVA],np.int8);BTE=np.array([lb(t) for t in TTE],np.int8)

C=np.zeros((KSRC,NB,KFAM),float)
for x,b,f in zip(XTR,BTR,FTR):C[x,b,f]+=1
QH=C+PRIOR*PB;QH/=QH.sum(2,keepdims=True)
def lps(ts,xs,bs,fs,qh):
 A=[];R=[];F=[]
 for t,x,b,f in zip(ts,xs,bs,fs):
  a=ll(t,int(x),VeE,Vr0z)/math.log(2);r=ll(t,int(x),Ve,Vr)/math.log(2)
  A.append(a);R.append(r);F.append(r+math.log2(max(qh[int(x),int(b),int(f)],1e-300)/max(PB[int(x),int(b),int(f)],1e-300)))
 return np.array(A),np.array(R),np.array(F)
va=lps(TVA,XVA,BVA,FVA,QH);te=lps(TTE,XTE,BTE,FTE,QH)
gv=va[2]-va[1];gr=te[1]-te[0];gf=te[2]-te[1]

# matched null: permute training source labels within observed length bucket.
rn=np.random.default_rng(SEED+400);null=np.empty(NNULL)
for j in range(NNULL):
 xp=XTR.copy()
 for b in range(NB):
  ix=np.where(BTR==b)[0]
  if len(ix)>1:xp[ix]=rn.permutation(xp[ix])
 cn=np.zeros((KSRC,NB,KFAM),float)
 for x,b,f in zip(xp,BTR,FTR):cn[x,b,f]+=1
 q=cn+PRIOR*PB;q/=q.sum(2,keepdims=True)
 null[j]=np.mean([math.log2(max(q[int(x),int(b),int(f)],1e-300)/max(PB[int(x),int(b),int(f)],1e-300)) for x,b,f in zip(XTE,BTE,FTE)])
mu=float(null.mean());sd=float(null.std(ddof=1));mean=float(gf.mean())

strtr=set(sf(t) for t in TTR);nov=np.array([sf(t) not in strtr for t in TTE]);ng=gf[nov]
nulln=np.empty(NNULL)
for j in range(NNULL):
 xp=XTR.copy()
 for b in range(NB):
  ix=np.where(BTR==b)[0]
  if len(ix)>1:xp[ix]=rn.permutation(xp[ix])
 cn=np.zeros((KSRC,NB,KFAM),float)
 for x,b,f in zip(xp,BTR,FTR):cn[x,b,f]+=1
 q=cn+PRIOR*PB;q/=q.sum(2,keepdims=True)
 nulln[j]=np.mean([math.log2(max(q[int(x),int(b),int(f)],1e-300)/max(PB[int(x),int(b),int(f)],1e-300)) for x,b,f,m in zip(XTE,BTE,FTE,nov) if m])
nmu=float(nulln.mean());nsd=float(nulln.std(ddof=1));nmean=float(ng.mean())

# preservation against unconditioned frozen B on same fresh source block.
rb=np.random.default_rng(SEED+500);BT=[sample(int(x),rb) for x in XTE]
def hist(ts):
 h=np.zeros(10,float)
 for t in ts:h[min(len(t),10)-1]+=1
 return h/h.sum()
ht=hist(TTE);hb=hist(BT);tv=.5*float(np.abs(ht-hb).sum())
def mg(ts):return float(np.mean([sum(bool(PG[p]) for p in t)>=2 for t in ts]))
# generative replication of learned kernel.
reps=[]
for rr in range(NGENREP):
 rg=np.random.default_rng(SEED+600+rr);gg=[]
 for x in XTE:
  seed=sample(int(x),rg);b=lb(seed);ff=int(rg.choice(KFAM,p=QH[int(x),b]))
  if fam(seed)==ff:gg.append(seed);continue
  for _ in range(800):
   t=sample(int(x),rg)
   if lb(t)==b and fam(t)==ff:gg.append(t);break
  else:raise RuntimeError("rep reject")
 hh=hist(gg);ss=[sf(t) for t in gg]
 reps.append({"types":len(set(ss)),"unseen_share":float(np.mean([s not in strtr for s in ss])),
              "mean_piece_len":float(np.mean([len(t) for t in gg])),
              "length_tv_vs_target":.5*float(np.abs(hh-ht).sum()),"multi_gallows":mg(gg)})
blocks=[gf[i:i+400].mean() for i in range(0,NTEST,400)]
nb=[]
for i in range(0,NTEST,400):
 m=nov[i:i+400];z=gf[i:i+400]
 if m.any():nb.append(z[m].mean())
out={"phase":"NK1","attempt":"b_length_preserving_fresh_final","status":"complete",
 "source":{"name":"Circa Instans","train":[0,NTRAIN],"validation":[NTRAIN,NTRAIN+NVAL],"fresh_final":[TEST0,TEST0+NTEST],"vocab":KSRC},
 "support":{"k":KFAM,"definition":"complete frozen K12 path hash excluding first class","family_bias_rank":2,"alpha":ALPHA,
            "normalization":"conditional within frozen piece-length bucket; length prior not controlled by family","no_exact_token_lookup":True,"no_naibbe_strings":True},
 "validation":{"family_gain_bits_per_token":float(gv.mean())},
 "test":{"entry_bits_per_token":float(-te[0].mean()),"route_bits_per_token":float(-te[1].mean()),"family_bits_per_token":float(-te[2].mean()),
         "route_gain_bits_per_token":float(gr.mean()),"family_gain_bits_per_token":mean,
         "family_block_sd":float(np.std(blocks,ddof=1)),"family_block_z0":float(mean/np.std(blocks,ddof=1)),
         "shuffle_null_mean":mu,"shuffle_null_sd":sd,"z_vs_shuffle":float((mean-mu)/sd),
         "novel_n":int(nov.sum()),"novel_share":float(nov.mean()),"novel_gain_bits_per_token":nmean,
         "novel_null_mean":nmu,"novel_null_sd":nsd,"novel_z_vs_shuffle":float((nmean-nmu)/nsd),
         "novel_block_sd":float(np.std(nb,ddof=1))},
 "preservation":{"target_mean_piece_len":float(np.mean([len(t) for t in TTE])),"baseline_mean_piece_len":float(np.mean([len(t) for t in BT])),
                 "delta_mean_piece_len":float(np.mean([len(t) for t in TTE])-np.mean([len(t) for t in BT])),
                 "length_distribution_tv":tv,"target_multi_gallows":mg(TTE),"baseline_multi_gallows":mg(BT),"all_form_legal":True},
 "generation":{"target_types":len(set(sf(t) for t in TTE)),"target_unseen_share":float(nov.mean()),"replicates":reps,
               "mean_types":float(np.mean([r["types"] for r in reps])),"mean_unseen_share":float(np.mean([r["unseen_share"] for r in reps])),
               "mean_length_tv_vs_target":float(np.mean([r["length_tv_vs_target"] for r in reps]))},
 "gates":{"family_vs_shuffle_gt2":bool((mean-mu)/sd>2),"novel_vs_shuffle_gt2":bool((nmean-nmu)/nsd>2),
          "length_tv_le_0_03":bool(tv<=.03),"mean_len_abs_delta_le_0_10":bool(abs(np.mean([len(t) for t in TTE])-np.mean([len(t) for t in BT]))<=.10)}
}
print("NAIBBE_NK1B_JSON="+json.dumps(out,separators=(",",":")),flush=True)
