#!/usr/bin/env python3
"""
4R-GRAMMAR-1 — external German morphosyntax bridge into frozen four-rule Voynich architecture.

Preregistered logic:
Rule 1 SELECT: frozen prospective K12 selector + frozen K4 opener-conditioned line repertoire.
Rule 2 FORM: untouched.
Rule 3 CONNECT: ONLY new term is an external German morphosyntactic transition prior.
Rule 4 LINE_ENTRY: physical line opener excluded; first two ordinary tokens remain identification-only
                   exactly as in the frozen K4 line-state baseline.

Firewall:
- Voynich grammar classes inferred from CONNECT distributions only on physical folds 2/3.
- NO token spelling, ED, FORM pieces, DINO, Stolfi/Mauro labels, German words, or daiin/ein labels used in clustering.
- German state taxonomy fixed from HiTS POS before observing alignment.
- Mapping Voynich clusters -> German states fitted only on Voynich folds2/3 + Bavarian ReF15 manuscripts.
- Lambda selected only on Voynich fold4 using Bavarian transitions.
- Final folds0/1 opened once using ALEMANNIC ReF15 transitions under the frozen mapping/lambda.
- Existing SELECT baseline refits in the canonical way; grammar coordinate does not refit on fold4/final.
"""

import collections, hashlib, io, json, math, re, tarfile, urllib.request
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

SEED=20261005
KGRAM=12
MIN_TYPE=5
NEIGH_VOCAB=256
NNULL=500

LAT_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/74ae7aa0088dc07436000f777d9c5d7d705730c2/research/latent_line_state_test_20261004.py"
REF_URL="https://zenodo.org/api/records/5793616/files/ReF-v1.0.2.tar.gz/content"

# ---------------- frozen 4R baseline ----------------
ns={"__name__":"grammar1_latent"}
exec(compile(urllib.request.urlopen(LAT_URL,timeout=120).read().decode(),LAT_URL,"exec"),ns)
rows,folds=ns["rows"],ns["folds"]
fit_struct,attach,fit_mix=ns["fit_struct"],ns["attach"],ns["fit_mix"]
LINES=ns["LINES"]

# exact token sequence aligned to frozen LINES
toklines=collections.defaultdict(list)
for r in rows:
    toklines[(r["folio"],r["line"])].append((int(r["pos"]),r["token"],int(r["start"]),r["bifolium"],int(r["fold"])))
for k in toklines:
    toklines[k].sort()

# ---------------- CONNECT-only Voynich type embedding ----------------
def discovery_sequences(fset):
    out=[]
    for (fol,line),arr in toklines.items():
        if not arr or arr[0][4] not in fset: continue
        out.append([x[1] for x in arr])
    return out

Dseq=discovery_sequences({2,3})
dfreq=collections.Counter(t for seq in Dseq for t in seq)
types=sorted([t for t,n in dfreq.items() if n>=MIN_TYPE],key=lambda t:(-dfreq[t],t))
if len(types)<KGRAM: raise RuntimeError("Too few discovery types")
tid={t:i for i,t in enumerate(types)}
nv=[t for t,n in dfreq.most_common(NEIGH_VOCAB)]
nid={t:i for i,t in enumerate(nv)}
X=np.zeros((len(types),2*len(nv)),float)
for seq in Dseq:
    for i,t in enumerate(seq):
        if t not in tid: continue
        a=tid[t]
        if i and seq[i-1] in nid: X[a,nid[seq[i-1]]]+=1
        if i+1<len(seq) and seq[i+1] in nid: X[a,len(nv)+nid[seq[i+1]]]+=1

# PPMI independently of spelling.
tot=X.sum()
rs=X.sum(1,keepdims=True);cs=X.sum(0,keepdims=True)
den=rs@cs
with np.errstate(divide="ignore",invalid="ignore"):
    P=np.where(X>0,np.log2(np.maximum(X*tot,1e-300)/np.maximum(den,1e-300)),0.0)
P=np.maximum(P,0.0)
dim=min(32,max(2,min(P.shape)-1))
Z=TruncatedSVD(n_components=dim,random_state=SEED).fit_transform(P)
lab=KMeans(n_clusters=KGRAM,n_init=50,random_state=SEED).fit_predict(Z)
TYPECL={t:int(lab[tid[t]]) for t in types}
cluster_counts=collections.Counter(TYPECL[t] for t in types for _ in range(dfreq[t]))
print("G1_CLUSTER_AUDIT",json.dumps({"n_types":len(types),"coverage_tokens":sum(dfreq[t] for t in types),
      "discovery_tokens":sum(dfreq.values()),"cluster_token_counts":cluster_counts},default=int),flush=True)

def cluster_transition(fset):
    C=np.zeros((KGRAM,KGRAM),float);freq=np.zeros(KGRAM,float);eligible=0;total_pairs=0
    for seq in discovery_sequences(fset):
        for i,t in enumerate(seq):
            if t in TYPECL:freq[TYPECL[t]]+=1
            if i:
                total_pairs+=1
                a,b=seq[i-1],t
                if a in TYPECL and b in TYPECL:
                    C[TYPECL[a],TYPECL[b]]+=1;eligible+=1
    return C,freq,eligible,total_pairs

VC_D,VF_D,_,_=cluster_transition({2,3})
VC_V,VF_V,_,_=cluster_transition({4})
VC_T,VF_T,Velig,Vpairs=cluster_transition({0,1})

def rowprob(C,alpha=1.0):
    return (C+alpha)/(C.sum(1,keepdims=True)+alpha*C.shape[1])
def fprob(v,alpha=1.0):
    return (v+alpha)/(v.sum()+alpha*len(v))
def jsd(p,q):
    p=np.asarray(p,float);q=np.asarray(q,float);p/=p.sum();q/=q.sum();m=(p+q)/2
    z=p>0; a=np.sum(p[z]*np.log2(p[z]/m[z]))
    z=q>0; b=np.sum(q[z]*np.log2(q[z]/m[z]))
    return float(.5*(a+b))

# ---------------- fixed German 12-state taxonomy ----------------
GSTATES=["DET","PRON","AUX","MODAL","VERB_FIN","VERB_NONFIN","NOUN","ADJ","ADV","PREP","CONJ","OTHER"]
GID={x:i for i,x in enumerate(GSTATES)}
def gstate(pos):
    p=(pos or "").upper()
    if p.startswith("D"): return GID["DET"]
    if p.startswith("P"): return GID["PRON"]
    if p.startswith("VA"): return GID["AUX"]
    if p.startswith("VM"): return GID["MODAL"]
    if p=="VVFIN": return GID["VERB_FIN"]
    if p.startswith("VV"): return GID["VERB_NONFIN"]
    if p=="NA" or p.startswith("N"): return GID["NOUN"]
    if p.startswith("ADJ"): return GID["ADJ"]
    if p in ("ADV","AVD") or p.startswith("ADV"): return GID["ADV"]
    if p.startswith("AP"): return GID["PREP"]
    if p.startswith("KO") or p=="KON": return GID["CONJ"]
    return GID["OTHER"]

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

print("G1_REF_DOWNLOAD",flush=True)
rb=urllib.request.urlopen(REF_URL,timeout=300).read()
tar=tarfile.open(fileobj=io.BytesIO(rb),mode="r:gz")
gseq={"BAV":[],"ALEM":[]}
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
    cur=[];lines=[]
    for tok in root.iter():
        if tok.tag.split("}")[-1]!="token":continue
        ds=[x for x in tok if x.tag.split("}")[-1] in ("tok_dipl","dipl")]
        ms=[x for x in tok if x.tag.split("}")[-1] in ("tok_anno","mod")]
        if not ms:continue
        first=ds[0].attrib.get("id") if ds else None;last=ds[-1].attrib.get("id") if ds else None
        for mi,x in enumerate(ms):
            pos=child(x,"pos")
            valid=bool(pos and not pos.startswith("$"))
            if mi==0 and first in starts and cur:lines.append(cur);cur=[]
            if valid:cur.append(gstate(pos))
            if mi==len(ms)-1 and last in ends:
                if cur:lines.append(cur);cur=[]
    if cur:lines.append(cur)
    gseq[side].extend([x for x in lines if len(x)>=2])

def Gmat(lines):
    C=np.zeros((KGRAM,KGRAM),float);F=np.zeros(KGRAM,float)
    for seq in lines:
        for i,s in enumerate(seq):
            F[s]+=1
            if i:C[seq[i-1],s]+=1
    return C,F
GC_B,GF_B=Gmat(gseq["BAV"]);GC_A,GF_A=Gmat(gseq["ALEM"])
print("G1_GERMAN_AUDIT",json.dumps({"bav_lines":len(gseq["BAV"]),"alem_lines":len(gseq["ALEM"]),
      "bav_state_counts":GF_B.tolist(),"alem_state_counts":GF_A.tolist()}),flush=True)

# ---------------- permutation graph alignment: Voynich discovery -> Bavarian ----------------
PV=rowprob(VC_D);FV=fprob(VF_D);PB=rowprob(GC_B);FB=fprob(GF_B)
def align_obj(perm,Pg=PB,Fg=FB):
    # perm[i] = German state assigned to Voynich cluster i
    perm=np.asarray(perm,int)
    d=0.0
    for i in range(KGRAM):
        # reorder German destination dimensions back into Voynich-cluster order
        grow=np.array([Pg[perm[i],perm[j]] for j in range(KGRAM)])
        d+=FV[i]*jsd(PV[i],grow)
    gf=np.array([Fg[perm[i]] for i in range(KGRAM)])
    d+=0.5*jsd(FV,gf)
    return float(d)

rng=np.random.default_rng(SEED)
best=None
for restart in range(256):
    p=np.arange(KGRAM) if restart==0 else rng.permutation(KGRAM)
    cur=align_obj(p)
    improved=True
    while improved:
        improved=False;bi=bj=None;bv=cur
        for i in range(KGRAM):
            for j in range(i+1,KGRAM):
                q=p.copy();q[i],q[j]=q[j],q[i];v=align_obj(q)
                if v < bv-1e-12: bv=v;bi=i;bj=j
        if bi is not None:
            p[bi],p[bj]=p[bj],p[bi];cur=bv;improved=True
    if best is None or cur<best[0]:best=(cur,p.copy())
ALIGN_DISC,bperm=best
print("G1_ALIGNMENT",json.dumps({"discovery_objective":ALIGN_DISC,"mapping":{str(i):GSTATES[int(bperm[i])] for i in range(KGRAM)}}),flush=True)

def mapped_trans(C,F,perm):
    P=rowprob(C);Fg=fprob(F)
    # transition expressed in Voynich-cluster coordinate
    Q=np.zeros((KGRAM,KGRAM),float)
    qf=np.zeros(KGRAM,float)
    for i in range(KGRAM):
        qf[i]=Fg[perm[i]]
        for j in range(KGRAM):Q[i,j]=P[perm[i],perm[j]]
    return Q,qf
QB,QFB=mapped_trans(GC_B,GF_B,bperm)
QA,QFA=mapped_trans(GC_A,GF_A,bperm)

# independent CONNECT network endpoint
def netdist(C,F,Q,qf):
    P=rowprob(C);f=fprob(F);return float(sum(f[i]*jsd(P[i],Q[i]) for i in range(KGRAM))+0.5*jsd(f,qf))
NET_VAL_B=netdist(VC_V,VF_V,QB,QFB)
NET_FINAL_A=netdist(VC_T,VF_T,QA,QFA)

# ---------------- cluster -> K12 emission from discovery only ----------------
E=np.ones((KGRAM,12),float) # Laplace
cf=np.ones(KGRAM,float)
for r in rows:
    if int(r["fold"]) not in (2,3):continue
    t=r["token"]
    if t in TYPECL:
        c=TYPECL[t];E[c,int(r["start"])]+=1;cf[c]+=1
E/=E.sum(1,keepdims=True)
cf/=cf.sum()
qmarg=cf@E

# frozen baseline event predictions + German grammar tilt
def baseline_event_probs(train_lines,test_lines):
    base=fit_struct(train_lines);trp=attach(train_lines,base);tep=attach(test_lines,base)
    md=fit_mix(trp,4,3.0)
    out=[]
    for l in tep:
        P,Y=l["P"],l["Y"];q=2
        prior=md["hp"][l["house"]]
        A=np.log(prior+1e-15)
        for z in range(4):
            sc=np.log(np.maximum(P[:q],1e-15))+md["B"][z]
            mx=sc.max(1,keepdims=True);QQ=np.exp(sc-mx);QQ/=QQ.sum(1,keepdims=True)
            A[z]+=np.log(np.maximum(QQ[np.arange(q),Y[:q]],1e-300)).sum()
        mx=A.max();post=np.exp(A-mx);post/=post.sum()
        arr=toklines[(l["folio"],l["line"])]
        toks=[x[1] for x in arr]
        # event i corresponds exact token toks[i+1]; scoring i>=2.
        for i in range(q,len(Y)):
            pm=np.zeros(12,float)
            for z in range(4):
                sc=np.log(np.maximum(P[i],1e-15))+md["B"][z];sc-=sc.max()
                qq=np.exp(sc);qq/=qq.sum();pm+=post[z]*qq
            prevtok=toks[i] if i < len(toks) else None
            out.append({"P":pm,"y":int(Y[i]),"prev":prevtok,"bif":l["bif"],"folio":l["folio"],"line":l["line"]})
    return out

D=[l for l in LINES if l["fold"] in (2,3)]
VAL=[l for l in LINES if l["fold"]==4]
TR=[l for l in LINES if l["fold"] in (2,3,4)]
TE=[l for l in LINES if l["fold"] in (0,1)]
EVV=baseline_event_probs(D,VAL)

def score_events(events,Q,lam,return_blocks=False):
    b=m=0.0;n=0;eligible=0;blk=collections.defaultdict(lambda:[0.,0.,0])
    for e in events:
        p=e["P"];y=e["y"];pb=max(float(p[y]),1e-300)
        pa=p.copy()
        if e["prev"] in TYPECL and lam>0:
            eligible+=1;c=TYPECL[e["prev"]];q=Q[c]@E
            ratio=np.maximum(q,1e-12)/np.maximum(qmarg,1e-12)
            pa=pa*np.power(ratio,lam);pa/=pa.sum()
        pg=max(float(pa[y]),1e-300)
        bb=-math.log2(pb);mm=-math.log2(pg);b+=bb;m+=mm;n+=1
        z=blk[e["bif"]];z[0]+=bb;z[1]+=mm;z[2]+=1
    ret={"n":n,"eligible":eligible,"coverage":eligible/n if n else 0.,
         "base_bits":b/n,"model_bits":m/n,"gain_bits":(b-m)/n}
    if return_blocks:
        gs=[(x[0]-x[1])/x[2] for x in blk.values() if x[2]]
        ret.update({"blocks":len(gs),"blocks_positive":sum(g>0 for g in gs),
                    "block_mean":float(np.mean(gs)),"block_sd":float(np.std(gs,ddof=1)),
                    "block_mean_over_sd":float(np.mean(gs)/np.std(gs,ddof=1)) if len(gs)>1 and np.std(gs,ddof=1)>0 else None})
    return ret

LAMBDAS=[0.,0.125,0.25,0.5,1.0]
val=[{"lambda":x,**score_events(EVV,QB,x)} for x in LAMBDAS]
sel=max(val,key=lambda z:z["gain_bits"])
LAMBDA=float(sel["lambda"])
print("G1_VALIDATION",json.dumps({"grid":val,"selected_lambda":LAMBDA}),flush=True)

# final baseline refit canonically, grammar coordinate/mapping remain discovery-only
EVT=baseline_event_probs(TR,TE)
FINAL_A=score_events(EVT,QA,LAMBDA,True)
FINAL_B=score_events(EVT,QB,LAMBDA,True)

# ---------------- hostile nulls on final Alemannic ----------------
def shuffle_lines(lines,rng):
    out=[]
    for s in lines:
        x=list(s);rng.shuffle(x);out.append(x)
    return out

null_gains=[];null_net=[]
rng2=np.random.default_rng(SEED+991)
for rep in range(NNULL):
    sc,sf=Gmat(shuffle_lines(gseq["ALEM"],rng2))
    q,qf=mapped_trans(sc,sf,bperm)
    null_gains.append(score_events(EVT,q,LAMBDA)["gain_bits"])
    null_net.append(netdist(VC_T,VF_T,q,qf))

ng=np.array(null_gains);nn=np.array(null_net)
obs=FINAL_A["gain_bits"];gmean=float(ng.mean());gsd=float(ng.std(ddof=1));gz=(obs-gmean)/gsd if gsd else None
nmean=float(nn.mean());nsd=float(nn.std(ddof=1));nz=(nmean-NET_FINAL_A)/nsd if nsd else None

# mapping-permutation null using real Alemannic order.
mapg=[]
for rep in range(NNULL):
    pp=rng2.permutation(KGRAM);q,qf=mapped_trans(GC_A,GF_A,pp)
    mapg.append(score_events(EVT,q,LAMBDA)["gain_bits"])
mp=np.array(mapg);mpmean=float(mp.mean());mpsd=float(mp.std(ddof=1));mpz=(obs-mpmean)/mpsd if mpsd else None

out={
 "phase":"4R_GRAMMAR_1","status":"complete",
 "firewall":{"voynich_cluster_input":"CONNECT distributions only","min_type":MIN_TYPE,"n_types":len(types),
             "german_taxonomy":GSTATES,"mapping_fit":"Voynich folds2/3 + Bavarian ReF15",
             "lambda_select":"Voynich fold4 + Bavarian ReF15","final":"Voynich folds0/1 + Alemannic ReF15"},
 "alignment":{"discovery_objective":ALIGN_DISC,"mapping":[GSTATES[int(bperm[i])] for i in range(KGRAM)],
              "validation_network_distance_bav":NET_VAL_B,"final_network_distance_alem":NET_FINAL_A,
              "final_network_shuffle_null_mean":nmean,"final_network_shuffle_null_sd":nsd,"final_network_advantage_z":nz},
 "validation":{"grid":val,"selected_lambda":LAMBDA},
 "final":{"alemannic_primary":FINAL_A,"bavarian_descriptive":FINAL_B,
          "order_shuffle_null_mean_gain":gmean,"order_shuffle_null_sd":gsd,"order_shuffle_advantage_z":gz,
          "mapping_perm_null_mean_gain":mpmean,"mapping_perm_null_sd":mpsd,"mapping_perm_advantage_z":mpz},
 "coverage":{"final_connect_pairs_eligible":Velig,"final_connect_pairs_total":Vpairs}
}
print("GRAMMAR1_RESULT_JSON="+json.dumps(out,separators=(",",":")),flush=True)
