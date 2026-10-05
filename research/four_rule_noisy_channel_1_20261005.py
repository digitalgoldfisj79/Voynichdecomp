#!/usr/bin/env python3
"""
4R-NOISY-CHANNEL-1 (NC1)
Literature-grounded homophonic word/state decipherment calibration + gated Voynich test.

Method family:
- fixed plaintext language model + unknown homophonic substitution channel
- hard decipherment search with many independent simulated-annealing trajectories
- source LM never fitted to Voynich
- positive-control recoverability required before any real-Voynich inference

The plaintext states are exact historical-German grammatical/lexical cells:
    lemma | HiTS POS | morphology
not coarse POS labels.

The ciphertext alphabet is the most frequent exact Voynich token types plus OTHER.
Within-token FORM is not refitted: surface token types are treated as opaque outputs
of the already-frozen renderer. LINE_ENTRY is suppressed by excluding the first two
ordinary tokens of every physical line. Thus NC1 probes the upstream source->target
lexical/state bridge and CONNECT ordering, not FORM or LINE_ENTRY.

CALIBRATION:
- real ReF 15th-c Alemannic manuscript state sequences are plaintext
- real Voynich discovery token-frequency shape supplies only cipher-symbol frequencies
- hidden sparse homophonic codebook (avg ~4 symbols/state)
- solver sees only ciphertext + a Bavarian ReF15 source LM
- 16 independent synthetic channels
- rho ladder chosen only by recovery
- gate must pass before real Voynich stage

REAL:
- codebook learned from Voynich physical folds2/3 against Bavarian LM
- no tuning on fold4; fold4 diagnostic only
- frozen codebook opened on folds0/1
- primary confirmation uses independently held-out Alemannic LM
- controls: Alemannic iid/unigram, within-line order shuffles, and state-label permutations
- physical-bifolium block consistency reported

No daiin/ein/sein hypothesis enters fitting.
No PGCS, P70, ED, DINO, Stolfi/Mauro or character substitution enters fitting.
"""
import collections, io, json, math, re, tarfile, urllib.request, hashlib
import xml.etree.ElementTree as ET
import numpy as np
from numba import njit

SEED=20261005
S=48                   # 47 shared lexical/morph cells + OTHER
O=192                  # 191 exact VMS token types + OTHER
MIN_CELL_BOTH=25
LM_ALPHA=0.25
RHO_GRID=(0.25,0.5,1.0,2.0)
CAL_REPS=16
CAL_RESTARTS=12
CAL_STEPS=10000
REAL_RESTARTS=192      # prior programme showed restart-sensitive homophonic recovery
REAL_STEPS=22000
NNULL=500

LAT_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/74ae7aa0088dc07436000f777d9c5d7d705730c2/research/latent_line_state_test_20261004.py"
REF_URL="https://zenodo.org/api/records/5793616/files/ReF-v1.0.2.tar.gz/content"

# ---------- utilities ----------
def child(el,name):
    for x in el:
        if x.tag.split("}")[-1]==name:return x.attrib.get("tag","")
    return ""
def rrng(s):
    if not s:return None,None
    return tuple(s.split("..",1)) if ".." in s else (s,s)
def header(root):
    h=next((x for x in root.iter() if x.tag.split("}")[-1]=="header"),None);d={}
    for line in ((h.text or "") if h is not None else "").splitlines():
        if ":" in line:
            k,v=line.split(":",1);d[k.strip().lower()]=v.strip()
    return d
def canon_cell(le,pos,mor):
    return f"{le}|{pos}|{mor or '--'}"
def transition(lines,nstate=S,alpha=LM_ALPHA):
    C=np.zeros((nstate,nstate),np.float64);F=np.zeros(nstate,np.float64);pairs=0
    for seq in lines:
        for i,x in enumerate(seq):
            F[x]+=1
            if i:
                C[seq[i-1],x]+=1;pairs+=1
    pi=(F+alpha)/(F.sum()+alpha*nstate)
    T=(C+alpha*pi[None,:])/(C.sum(1,keepdims=True)+alpha)
    return T,pi,C,F,pairs
def iid_T(pi):
    return np.repeat(pi[None,:],len(pi),axis=0)
def bits_for_lines(lines,T):
    n=0;z=0.
    for seq in lines:
        for i in range(1,len(seq)):
            z-=math.log2(max(T[seq[i-1],seq[i]],1e-300));n+=1
    return z/n if n else float("nan"),n

# ---------- VMS frozen physical data ----------
ns={"__name__":"nc1_latent"}
exec(compile(urllib.request.urlopen(LAT_URL,timeout=120).read().decode(),LAT_URL,"exec"),ns)
vrows=ns["rows"];vfolds=ns["folds"]
FOLD_SHA=ns["canon_sha"](vfolds)

# Strict frozen ordinary rows from canonical programme. Exclude first two physical-line tokens.
vg=collections.OrderedDict()
for r in vrows:
    k=(r["folio"],int(r["line"]))
    vg.setdefault(k,[]).append(r)
vlines=[]
for k,arr in vg.items():
    arr=sorted(arr,key=lambda x:int(x["pos"]))
    if len(arr)<=2:continue
    tail=arr[2:]
    vlines.append({"folio":arr[0]["folio"],"line":int(arr[0]["line"]),
                   "bif":arr[0]["bifolium"],"fold":int(arr[0]["fold"]),
                   "tokens":[x["token"] for x in tail]})

# Cipher alphabet frozen from discovery only.
vf=collections.Counter(t for L in vlines if L["fold"] in (2,3) for t in L["tokens"])
VOC=[t for t,n in vf.most_common(O-1)]
OID={t:i for i,t in enumerate(VOC)}
OTHER_O=O-1
def oid(t):return OID.get(t,OTHER_O)
def vobs(fset):
    out=[]
    for L in vlines:
        if L["fold"] not in fset:continue
        xs=[oid(t) for t in L["tokens"]]
        if len(xs)>=2:out.append((L["bif"],xs))
    return out
VD=vobs({2,3});VV=vobs({4});VT=vobs({0,1})
def obs_stats(lines):
    c=np.zeros(O,np.int64);B=np.zeros((O,O),np.int64);pairs=0
    for _,seq in lines:
        for i,x in enumerate(seq):
            c[x]+=1
            if i:B[seq[i-1],x]+=1;pairs+=1
    return B,c,pairs
VB_D,VC_D,VNP_D=obs_stats(VD)
VB_V,VC_V,VNP_V=obs_stats(VV)
VB_T,VC_T,VNP_T=obs_stats(VT)
OBS_BASE=(VC_D+0.5)/(VC_D.sum()+0.5*O)
print("NC1_VMS_AUDIT",json.dumps({"fold_sha":FOLD_SHA,"vocab_exact":len(VOC),
      "disc_tokens":int(VC_D.sum()),"disc_pairs":VNP_D,"val_tokens":int(VC_V.sum()),
      "test_tokens":int(VC_T.sum()),"other_share_disc":float(VC_D[OTHER_O]/VC_D.sum())}),flush=True)

# ---------- ReF 15th-c manuscript plaintext ----------
print("NC1_REF_DOWNLOAD",flush=True)
rb=urllib.request.urlopen(REF_URL,timeout=300).read()
tar=tarfile.open(fileobj=io.BytesIO(rb),mode="r:gz")
raw_by={"BAV":[],"ALEM":[]}
doc_by={"BAV":collections.defaultdict(list),"ALEM":collections.defaultdict(list)}
for name in [n for n in tar.getnames() if n.endswith(".xml")]:
    try:root=ET.fromstring(tar.extractfile(name).read())
    except:continue
    md=header(root);med=md.get("medium","").lower();tm=md.get("time","").lower();area=md.get("language-area","").lower()
    if "handschrift" not in med or not tm.startswith("15,"):continue
    bav=(("bair" in area or "bayr" in area or "österreich" in area or "oesterreich" in area) and "alemann" not in area)
    alem=(("alemann" in area or "schwäb" in area or "elsäss" in area) and "bair" not in area and "bayr" not in area)
    if not (bav or alem):continue
    side="BAV" if bav else "ALEM"
    starts=set();ends=set()
    for x in root.iter():
        if x.tag.split("}")[-1]=="line":
            a,b=rrng(x.attrib.get("range",""))
            if a:starts.add(a)
            if b:ends.add(b)
    lines=[];cur=[]
    for tok in root.iter():
        if tok.tag.split("}")[-1]!="token":continue
        ds=[x for x in tok if x.tag.split("}")[-1] in ("tok_dipl","dipl")]
        ms=[x for x in tok if x.tag.split("}")[-1] in ("tok_anno","mod")]
        if not ms:continue
        first=ds[0].attrib.get("id") if ds else None;last=ds[-1].attrib.get("id") if ds else None
        for mi,x in enumerate(ms):
            le=child(x,"lemma");pos=child(x,"pos");mor=child(x,"morph") or child(x,"inflection") or "--"
            valid=bool(le and le not in ("--","[!]") and pos and not pos.startswith("$"))
            if mi==0 and first in starts and cur:lines.append(cur);cur=[]
            if valid:cur.append(canon_cell(le,pos,mor))
            if mi==len(ms)-1 and last in ends:
                if cur:lines.append(cur);cur=[]
    if cur:lines.append(cur)
    # Match VMS boundary treatment: first two tokens are not source-order evidence.
    lines=[x[2:] for x in lines if len(x)>3]
    raw_by[side].extend(lines)
    doc_by[side][name].extend(lines)

# Common source-state inventory fixed before VMS decipherment.
cb=collections.Counter(x for line in raw_by["BAV"] for x in line)
ca=collections.Counter(x for line in raw_by["ALEM"] for x in line)
eligible=[x for x in cb if cb[x]>=MIN_CELL_BOTH and ca[x]>=MIN_CELL_BOTH]
eligible.sort(key=lambda x:(-min(cb[x],ca[x]),-(cb[x]+ca[x]),x))
CELLS=eligible[:S-1]
CID={x:i for i,x in enumerate(CELLS)}
OTHER_S=S-1
def sid(x):return CID.get(x,OTHER_S)
src={}
for side in ("BAV","ALEM"):
    src[side]=[[sid(x) for x in line] for line in raw_by[side] if len(line)>=2]
TB,PIB,CB,FB,NPB=transition(src["BAV"])
TA,PIA,CA,FA,NPA=transition(src["ALEM"])
print("NC1_SOURCE_AUDIT",json.dumps({"states_exact":len(CELLS),"bav_lines":len(src["BAV"]),
      "alem_lines":len(src["ALEM"]),"bav_pairs":NPB,"alem_pairs":NPA,
      "other_bav":float(FB[OTHER_S]/FB.sum()),"other_alem":float(FA[OTHER_S]/FA.sum()),
      "top_cells":CELLS[:12]}),flush=True)

# ---------- homophonic mapper ----------
@njit(cache=True)
def obj_full(B,c,m,logT,pi,rho):
    O_=len(m);ll=0.;N=float(c.sum())
    for i in range(O_):
        mi=m[i]
        for j in range(O_):
            z=B[i,j]
            if z:ll += z*logT[mi,m[j]]
    mass=np.zeros(len(pi),np.float64)
    for i in range(O_):mass[m[i]]+=c[i]
    if N>0:
        for s in range(len(pi)):
            q=mass[s]/N
            if q>0: ll -= rho*N*q*math.log(max(q/pi[s],1e-300))
    return ll

@njit(cache=True)
def delta_move(B,c,m,logT,pi,rho,i,new):
    old=m[i]
    if old==new:return 0.
    O_=len(m);d=0.
    for j in range(O_):
        if j==i:continue
        z=B[i,j]
        if z:d += z*(logT[new,m[j]]-logT[old,m[j]])
        z=B[j,i]
        if z:d += z*(logT[m[j],new]-logT[m[j],old])
    z=B[i,i]
    if z:d += z*(logT[new,new]-logT[old,old])
    N=float(c.sum())
    if N>0:
        mo=0.;mn=0.
        for k in range(O_):
            if m[k]==old:mo+=c[k]
            if m[k]==new:mn+=c[k]
        def term(x,p):
            if x<=0:return 0.
            q=x/N
            return -rho*N*q*math.log(max(q/p,1e-300))
        d += term(mo-c[i],pi[old])+term(mn+c[i],pi[new])-term(mo,pi[old])-term(mn,pi[new])
    return d

@njit(cache=True)
def anneal(B,c,logT,pi,rho,init,steps,seed):
    np.random.seed(seed)
    m=init.copy();cur=obj_full(B,c,m,logT,pi,rho);best=cur;bm=m.copy()
    npairs=max(1.0,float(B.sum()))
    t0=0.010;t1=0.00002
    for st in range(steps):
        frac=st/max(1,steps-1)
        temp=t0*((t1/t0)**frac)
        i=np.random.randint(0,len(m));new=np.random.randint(0,len(pi))
        if new==m[i]:continue
        d=delta_move(B,c,m,logT,pi,rho,i,new)
        dn=d/npairs
        if d>=0 or math.log(max(np.random.random(),1e-300)) < dn/temp:
            m[i]=new;cur+=d
            if cur>best:
                best=cur;bm=m.copy()
    return best,bm

def freq_init(c,pi,rng,jitter=False):
    order=np.argsort(-c);target=pi*c.sum();mass=np.zeros(len(pi),float);m=np.zeros(len(c),np.int64)
    if jitter:
        # seeded random starting point still frequency-aware
        mass[:]=0
        for i in order:
            deficit=np.maximum(target-mass,1e-9)
            p=deficit/deficit.sum();s=int(rng.choice(len(pi),p=p));m[i]=s;mass[s]+=c[i]
    else:
        for i in order:
            s=int(np.argmax(target-mass));m[i]=s;mass[s]+=c[i]
    return m

def fit_mapping(B,c,T,pi,rho,restarts,steps,seed,keep=12):
    logT=np.log(np.maximum(T,1e-300));rng=np.random.default_rng(seed);arr=[]
    for r in range(restarts):
        init=freq_init(c,pi,rng,jitter=(r>0))
        val,mp=anneal(B,c,logT,pi,float(rho),init,int(steps),int(seed+7919*r+17))
        arr.append((float(val),mp.copy()))
    arr.sort(key=lambda x:-x[0])
    return arr[:min(keep,len(arr))]

def map_accuracy(mp,hidden,c):
    return float(np.sum(c*(mp==hidden))/max(1,c.sum()))
def decode_lines(obs_lines,mp):
    return [[int(mp[x]) for x in seq] for _,seq in obs_lines]
def encode_plain(lines,hidden,base,rng):
    pools=[np.where(hidden==s)[0] for s in range(S)]
    probs=[]
    for p in pools:
        if len(p)==0:probs.append(None)
        else:
            z=base[p];z=z/z.sum();probs.append(z)
    out=[]
    for seq in lines:
        oo=[]
        for s in seq:
            p=pools[s]
            if len(p)==0:
                # hidden-map construction should prevent this.
                oo.append(int(rng.integers(0,O)))
            else:oo.append(int(rng.choice(p,p=probs[s])))
        if len(oo)>=2:out.append(("SYN",oo))
    return out

def hidden_map(pi,base,rng):
    # unique cipher-symbol -> plaintext-state; every state gets >=1 symbol.
    h=np.full(O,-1,np.int64);mass=np.zeros(S,float)
    syms=list(np.argsort(-base));states=list(np.argsort(-pi))
    for s,o in zip(states,syms[:S]):
        h[o]=s;mass[s]+=base[o]
    for o in syms[S:]:
        deficit=pi-mass
        if deficit.max()>0:s=int(np.argmax(deficit))
        else:s=int(np.argmin(mass/np.maximum(pi,1e-12)))
        h[o]=s;mass[s]+=base[o]
    return h

def sample_disjoint(lines,ntrain,ntest,rng):
    ix=rng.permutation(len(lines));tr=[];te=[];a=b=0
    for ii in ix:
        x=lines[int(ii)]
        if a<ntrain:tr.append(x);a+=len(x)
        elif b<ntest:te.append(x);b+=len(x)
        if a>=ntrain and b>=ntest:break
    return tr,te

# ---------- CALIBRATION ----------
# The positive control plaintext is real 15c Alemannic; only the channel is synthetic.
rng=np.random.default_rng(SEED)
cal={rho:[] for rho in RHO_GRID}
for rep in range(CAL_REPS):
    ptr,pte=sample_disjoint(src["ALEM"],int(VC_D.sum()),int(VC_T.sum()),rng)
    h=hidden_map(PIA,OBS_BASE,rng)
    otr=encode_plain(ptr,h,OBS_BASE,rng);ote=encode_plain(pte,h,OBS_BASE,rng)
    B,c,_=obs_stats(otr)
    B2,c2,_=obs_stats(ote)
    for rho in RHO_GRID:
        fits=fit_mapping(B,c,TB,PIB,rho,CAL_RESTARTS,CAL_STEPS,SEED+100000*rep+int(rho*1000),keep=1)
        mp=fits[0][1]
        train_acc=map_accuracy(mp,h,c)
        test_acc=map_accuracy(mp,h,c2)
        decoded=decode_lines(ote,mp)
        truebits,_=bits_for_lines(decoded,TA)
        iidbits,_=bits_for_lines(decoded,iid_T(PIA))
        rec={"rep":rep,"rho":rho,"train_acc":train_acc,"test_acc":test_acc,
             "alem_real_bits":truebits,"alem_iid_bits":iidbits,"order_gain":iidbits-truebits}
        cal[rho].append(rec)
        print("NC1_CAL_REP",json.dumps(rec,separators=(",",":")),flush=True)

def med(xs):return float(np.median(np.asarray(xs,float)))
cal_summary={}
for rho,rr in cal.items():
    cal_summary[str(rho)]={"median_train_acc":med([x["train_acc"] for x in rr]),
                          "median_test_acc":med([x["test_acc"] for x in rr]),
                          "frac_test_ge_060":float(np.mean([x["test_acc"]>=.60 for x in rr])),
                          "median_order_gain":med([x["order_gain"] for x in rr])}
best_rho=max(RHO_GRID,key=lambda r:(cal_summary[str(r)]["median_test_acc"],
                                   cal_summary[str(r)]["frac_test_ge_060"]))
cs=cal_summary[str(best_rho)]
chance=float(PIA.max())
CAL_PASS=bool(cs["median_test_acc"]>=.70 and cs["frac_test_ge_060"]>=.75 and
              cs["median_test_acc"]>=chance+.30 and cs["median_order_gain"]>0)
cal_out={"selected_rho":best_rho,"source_majority_chance":chance,"by_rho":cal_summary,
         "gate":{"median_test_acc_ge_070":bool(cs["median_test_acc"]>=.70),
                 "frac_ge060_ge075":bool(cs["frac_test_ge_060"]>=.75),
                 "beats_majority_plus030":bool(cs["median_test_acc"]>=chance+.30),
                 "heldout_real_order_beats_iid":bool(cs["median_order_gain"]>0),
                 "pass":CAL_PASS}}
print("NC1_CALIBRATION_JSON="+json.dumps(cal_out,separators=(",",":")),flush=True)

if not CAL_PASS:
    out={"phase":"4R_NOISY_CHANNEL_1","status":"calibration_failed",
         "calibration":cal_out,
         "decision":"NON-INFERENTIAL. Real Voynich stage not run because known 15th-century German was not reliably recoverable.",
         "constants":{"S":S,"O":O,"cal_reps":CAL_REPS,"cal_restarts":CAL_RESTARTS,"cal_steps":CAL_STEPS}}
    print("NC1_RESULT_JSON="+json.dumps(out,separators=(",",":")),flush=True)
    raise SystemExit(0)

# ---------- REAL VOYNICH, only after calibration passes ----------
rho=float(best_rho)
fits=fit_mapping(VB_D,VC_D,TB,PIB,rho,REAL_RESTARTS,REAL_STEPS,SEED+700000,keep=20)
best_obj,mp=fits[0]
# restart stability weighted by discovery token mass
stab=[]
for _,m2 in fits[1:]:
    stab.append(float(np.sum(VC_D*(m2==mp))/VC_D.sum()))
stability=float(np.median(stab)) if stab else 1.0

def decode_v(lines,mp):
    return [[int(mp[x]) for x in seq] for _,seq in lines]
DV=decode_v(VV,mp);DT=decode_v(VT,mp)
val_A,val_n=bits_for_lines(DV,TA);val_iid,_=bits_for_lines(DV,iid_T(PIA))
test_A,test_n=bits_for_lines(DT,TA);test_iid,_=bits_for_lines(DT,iid_T(PIA))
test_B,_=bits_for_lines(DT,TB)

# independent heldout-source order null: shuffle Alemannic state order within lines.
null=[];rngn=np.random.default_rng(SEED+900001)
for b in range(NNULL):
    z=[]
    for seq in src["ALEM"]:
        q=list(seq);rngn.shuffle(q);z.append(q)
    Tn,pn,_,_,_=transition(z)
    bb,_=bits_for_lines(DT,Tn);null.append(bb)
null=np.asarray(null)
order_gain=float(np.mean(null)-test_A)
order_z=float(order_gain/np.std(null,ddof=1))

# cross-dialect state-identity null: permute ALEM state labels.
pn=[]
for b in range(NNULL):
    p=rngn.permutation(S)
    Tp=TA[np.ix_(p,p)]
    bb,_=bits_for_lines(DT,Tp);pn.append(bb)
pn=np.asarray(pn)
perm_gain=float(np.mean(pn)-test_A);perm_z=float(perm_gain/np.std(pn,ddof=1))

# physical-bifolium real-vs-iid block effect
blk=[]
for bif,seq in VT:
    ds=[int(mp[x]) for x in seq]
    if len(ds)<2:continue
    br,_=bits_for_lines([ds],TA);bi,_=bits_for_lines([ds],iid_T(PIA))
    blk.append((bif,bi-br,len(ds)-1))
byb=collections.defaultdict(lambda:[0.,0])
for bif,g,n in blk:
    byb[bif][0]+=g*n;byb[bif][1]+=n
bg=[v[0]/v[1] for v in byb.values() if v[1]]
bsd=float(np.std(bg,ddof=1)) if len(bg)>1 else 0.
block_z=float(np.mean(bg)/bsd) if bsd else None

# decode diagnostics, not evidence unless gates pass
def cellname(s):
    return CELLS[s] if s<OTHER_S else "__OTHER__"
diag={}
for t in ["daiin","aiin","dain","chedy","qokeedy","ol"]:
    if t in OID:diag[t]=cellname(int(mp[OID[t]]))

key_cost_bits=float(O*math.log2(S))
key_cost_per_test_pair=key_cost_bits/max(1,test_n)
real_gate=bool((test_iid-test_A)>0 and order_z>2 and perm_z>2 and
               sum(x>0 for x in bg)>len(bg)/2)

out={"phase":"4R_NOISY_CHANNEL_1","status":"complete",
 "method":"fixed historical-German bigram LM + unknown deterministic homophonic codebook; multi-restart annealed hard decipherment",
 "firewall":{"plaintext_states":"ReF15 lemma|HiTS POS|morph; 47 shared cells + OTHER",
             "cipher_symbols":"191 exact VMS discovery token types + OTHER",
             "line_entry":"first two tokens of every line excluded",
             "form":"opaque/frozen; no within-token features used",
             "voynich_tuning":"none before calibration gate; mapping folds2/3 only",
             "final":"folds0/1 scored against independent Alemannic LM"},
 "calibration":cal_out,
 "real":{"rho":rho,"best_discovery_objective":best_obj,"top_restart_weighted_agreement_median":stability,
         "validation":{"alem_bits":val_A,"iid_bits":val_iid,"gain":val_iid-val_A,"pairs":val_n},
         "final":{"alem_bits":test_A,"bav_bits_descriptive":test_B,"iid_bits":test_iid,
                  "alem_vs_iid_gain_bits":test_iid-test_A,"pairs":test_n},
         "order_shuffle":{"null_mean_bits":float(np.mean(null)),"null_sd_bits":float(np.std(null,ddof=1)),
                          "real_advantage_bits":order_gain,"z":order_z},
         "state_permutation":{"null_mean_bits":float(np.mean(pn)),"null_sd_bits":float(np.std(pn,ddof=1)),
                              "real_advantage_bits":perm_gain,"z":perm_z},
         "physical_blocks":{"n":len(bg),"positive":int(sum(x>0 for x in bg)),
                            "mean_gain":float(np.mean(bg)),"sd":bsd,"mean_over_sd":block_z},
         "mdl":{"codebook_bits":key_cost_bits,"codebook_bits_per_test_pair":key_cost_per_test_pair},
         "diagnostic_decodes":diag,
         "gate_pass":real_gate},
 "constants":{"S":S,"O":O,"cal_reps":CAL_REPS,"cal_restarts":CAL_RESTARTS,
              "cal_steps":CAL_STEPS,"real_restarts":REAL_RESTARTS,"real_steps":REAL_STEPS,
              "nnull":NNULL}}
print("NC1_RESULT_JSON="+json.dumps(out,separators=(",",":")),flush=True)
