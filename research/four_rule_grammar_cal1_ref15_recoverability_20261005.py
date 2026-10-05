#!/usr/bin/env python3
"""
4R-GRAMMAR-CAL1 — recoverability calibration on real 15th-century German.

Purpose:
Before interpreting 4R-GRAMMAR-1 on Voynich, establish that its blind CONNECT-clustering
and graph-alignment machinery can recover genuine German morphosyntax at Voynich-like
sample sizes.

Data:
ReF 1.0.2, 15th-century manuscripts only.
Discovery: Bavarian/Austrian.
Heldout: Alemannic.
No Voynich data used except target sample sizes and hyperparameters fixed by 4R-GRAMMAR-1.

Per replicate:
1. sample Bavarian physical lines to ~14,600 tokens;
2. hide POS labels and cluster exact word types from left/right token context only,
   exactly K=12, min type count=5, neighbour vocab=256, PPMI->SVD->KMeans;
3. graph-align blind clusters to the true Bavarian 12-state POS transition network
   using ONLY network structure/frequencies, not token POS labels;
4. measure discovery token-weighted mapping accuracy against the hidden true POS labels;
5. transfer the frozen type->cluster mapping and cluster->POS mapping to an independent
   Alemannic sample (~12,000 tokens);
6. measure heldout token-level POS accuracy/NMI and transition-network distance;
7. compare heldout real German order with within-line POS-order shuffles preserving
   state frequencies and line lengths.

Primary calibration gate:
median heldout network advantage >2 null SD across replicates AND
median heldout mapped POS accuracy materially exceeds majority-class baseline AND
>=80% replicates have positive real-vs-shuffle advantage.
If this fails, 4R-GRAMMAR-1 is NON-INFERENTIAL regarding German.
"""
import collections, io, json, math, re, tarfile, urllib.request
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

SEED=20261005
K=12
MIN_TYPE=5
NEIGH_VOCAB=256
N_DISC=14600
N_TEST=12000
REPS=40
NSHUFF=100
REF_URL="https://zenodo.org/api/records/5793616/files/ReF-v1.0.2.tar.gz/content"
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
def rowprob(C,a=1.):
    return (C+a)/(C.sum(1,keepdims=True)+a*C.shape[1])
def fprob(v,a=1.):
    return (v+a)/(v.sum()+a*len(v))
def jsd(p,q):
    p=np.asarray(p,float);q=np.asarray(q,float);p/=p.sum();q/=q.sum();m=(p+q)/2
    za=p>0;zb=q>0
    return float(.5*np.sum(p[za]*np.log2(p[za]/m[za]))+.5*np.sum(q[zb]*np.log2(q[zb]/m[zb])))

print("CAL1_DOWNLOAD",flush=True)
rb=urllib.request.urlopen(REF_URL,timeout=300).read()
tar=tarfile.open(fileobj=io.BytesIO(rb),mode="r:gz")
by={"BAV":[],"ALEM":[]}
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
            pos=child(x,"pos")
            norm=(x.attrib.get("ascii") or x.attrib.get("utf") or x.attrib.get("trans") or "").strip().lower()
            valid=bool(pos and not pos.startswith("$") and norm)
            if mi==0 and first in starts and cur:lines.append(cur);cur=[]
            if valid:cur.append((norm,gstate(pos)))
            if mi==len(ms)-1 and last in ends:
                if cur:lines.append(cur);cur=[]
    if cur:lines.append(cur)
    by[side].extend([x for x in lines if len(x)>=2])

print("CAL1_CORPUS",json.dumps({"bav_lines":len(by["BAV"]),"alem_lines":len(by["ALEM"]),
  "bav_tokens":sum(len(x) for x in by["BAV"]),"alem_tokens":sum(len(x) for x in by["ALEM"])}),flush=True)

def sample_lines(lines,target,rng):
    idx=rng.permutation(len(lines));out=[];n=0
    for i in idx:
        out.append(lines[int(i)]);n+=len(lines[int(i)])
        if n>=target:break
    return out

def blind_cluster(lines,seed):
    freq=collections.Counter(w for line in lines for w,s in line)
    types=sorted([w for w,n in freq.items() if n>=MIN_TYPE],key=lambda w:(-freq[w],w))
    if len(types)<K:return None
    tid={w:i for i,w in enumerate(types)}
    nv=[w for w,n in freq.most_common(NEIGH_VOCAB)];nid={w:i for i,w in enumerate(nv)}
    X=np.zeros((len(types),2*len(nv)),float)
    truth=collections.defaultdict(collections.Counter)
    for line in lines:
        for i,(w,s) in enumerate(line):
            if w in tid:
                truth[w][s]+=1
                a=tid[w]
                if i and line[i-1][0] in nid:X[a,nid[line[i-1][0]]]+=1
                if i+1<len(line) and line[i+1][0] in nid:X[a,len(nv)+nid[line[i+1][0]]]+=1
    tot=X.sum();rs=X.sum(1,keepdims=True);cs=X.sum(0,keepdims=True);den=rs@cs
    with np.errstate(divide="ignore",invalid="ignore"):
        P=np.where(X>0,np.log2(np.maximum(X*tot,1e-300)/np.maximum(den,1e-300)),0.)
    P=np.maximum(P,0.)
    dim=min(32,max(2,min(P.shape)-1))
    Z=TruncatedSVD(n_components=dim,random_state=seed).fit_transform(P)
    lab=KMeans(n_clusters=K,n_init=30,random_state=seed).fit_predict(Z)
    tcl={w:int(lab[tid[w]]) for w in types}
    return tcl,freq,truth

def cluster_net(lines,tcl):
    C=np.zeros((K,K),float);F=np.zeros(K,float);eligible=0;pairs=0
    for line in lines:
        for i,(w,s) in enumerate(line):
            if w in tcl:F[tcl[w]]+=1
            if i:
                pairs+=1
                a=line[i-1][0];b=w
                if a in tcl and b in tcl:
                    C[tcl[a],tcl[b]]+=1;eligible+=1
    return C,F,eligible,pairs

def pos_net(lines,shuffle=False,rng=None):
    C=np.zeros((K,K),float);F=np.zeros(K,float)
    for line in lines:
        states=[s for w,s in line]
        if shuffle:
            states=list(states);rng.shuffle(states)
        for i,s in enumerate(states):
            F[s]+=1
            if i:C[states[i-1],s]+=1
    return C,F

def align(Cv,Fv,Cg,Fg):
    PV=rowprob(Cv);FV=fprob(Fv);PG=rowprob(Cg);FG=fprob(Fg)
    def obj(p):
        p=np.asarray(p,int);d=0.
        for i in range(K):
            grow=np.array([PG[p[i],p[j]] for j in range(K)])
            d+=FV[i]*jsd(PV[i],grow)
        gf=np.array([FG[p[i]] for i in range(K)])
        return float(d+.5*jsd(FV,gf))
    rng=np.random.default_rng(SEED+777)
    best=None
    for rr in range(128):
        p=np.arange(K) if rr==0 else rng.permutation(K);cur=obj(p)
        while True:
            bv=cur;bij=None
            for i in range(K):
                for j in range(i+1,K):
                    q=p.copy();q[i],q[j]=q[j],q[i];v=obj(q)
                    if v<bv-1e-12:bv=v;bij=(i,j)
            if bij is None:break
            i,j=bij;p[i],p[j]=p[j],p[i];cur=bv
        if best is None or cur<best[0]:best=(cur,p.copy())
    return best

def netdist(Cv,Fv,Cg,Fg,p):
    PV=rowprob(Cv);FV=fprob(Fv);PG=rowprob(Cg);FG=fprob(Fg);d=0.
    for i in range(K):
        grow=np.array([PG[p[i],p[j]] for j in range(K)])
        d+=FV[i]*jsd(PV[i],grow)
    gf=np.array([FG[p[i]] for i in range(K)])
    return float(d+.5*jsd(FV,gf))

rng=np.random.default_rng(SEED)
results=[]
for rep in range(REPS):
    D=sample_lines(by["BAV"],N_DISC,rng)
    T=sample_lines(by["ALEM"],N_TEST,rng)
    bc=blind_cluster(D,SEED+rep)
    if bc is None:continue
    tcl,freq,truth=bc
    CV,FV,_,_=cluster_net(D,tcl)
    GB,FB=pos_net(D)
    aobj,p=align(CV,FV,GB,FB)

    # discovery hidden-label accuracy/NMI, token weighted
    yt=[];yp=[]
    for line in D:
        for w,s in line:
            if w in tcl:
                yt.append(s);yp.append(int(p[tcl[w]]))
    disc_acc=float(np.mean(np.array(yt)==np.array(yp)))
    disc_nmi=float(normalized_mutual_info_score(yt,yp))
    disc_ari=float(adjusted_rand_score(yt,yp))

    # heldout shared-type classification and blind cluster network
    yt2=[];yp2=[]
    for line in T:
        for w,s in line:
            if w in tcl:
                yt2.append(s);yp2.append(int(p[tcl[w]]))
    test_acc=float(np.mean(np.array(yt2)==np.array(yp2))) if yt2 else 0.
    test_nmi=float(normalized_mutual_info_score(yt2,yp2)) if yt2 else 0.
    test_ari=float(adjusted_rand_score(yt2,yp2)) if yt2 else 0.
    majority=max(collections.Counter(s for line in T for w,s in line).values())/sum(len(x) for x in T)

    CT,FT,elig,pairs=cluster_net(T,tcl)
    GA,FA=pos_net(T)
    real=netdist(CT,FT,GA,FA,p)
    nul=[]
    for b in range(NSHUFF):
        GS,FS=pos_net(T,True,rng)
        nul.append(netdist(CT,FT,GS,FS,p))
    nm=float(np.mean(nul));ns=float(np.std(nul,ddof=1));z=(nm-real)/ns if ns else None
    results.append({"rep":rep,"disc_tokens":sum(len(x) for x in D),"test_tokens":sum(len(x) for x in T),
      "n_types":len(tcl),"test_shared_tokens":len(yt2),"test_coverage":len(yt2)/sum(len(x) for x in T),
      "disc_acc":disc_acc,"disc_nmi":disc_nmi,"disc_ari":disc_ari,
      "test_acc":test_acc,"test_nmi":test_nmi,"test_ari":test_ari,"majority":majority,
      "network_real":real,"network_null_mean":nm,"network_null_sd":ns,"network_advantage_z":z,
      "align_obj":aobj})
    print("CAL1_REP",json.dumps(results[-1],separators=(",",":")),flush=True)

def med(k):return float(np.median([x[k] for x in results]))
def mean(k):return float(np.mean([x[k] for x in results]))
summary={"phase":"4R_GRAMMAR_CAL1","status":"complete","reps":len(results),
 "median_disc_acc":med("disc_acc"),"median_disc_nmi":med("disc_nmi"),"median_disc_ari":med("disc_ari"),
 "median_test_acc":med("test_acc"),"median_test_nmi":med("test_nmi"),"median_test_ari":med("test_ari"),
 "median_majority":med("majority"),"median_test_coverage":med("test_coverage"),
 "median_network_advantage_z":med("network_advantage_z"),
 "mean_network_advantage_z":mean("network_advantage_z"),
 "fraction_z_positive":float(np.mean([x["network_advantage_z"]>0 for x in results])),
 "fraction_z_gt2":float(np.mean([x["network_advantage_z"]>2 for x in results])),
 "gate":{"median_z_gt2":med("network_advantage_z")>2,
         "accuracy_beats_majority":med("test_acc")>med("majority"),
         "positive_80pct":np.mean([x["network_advantage_z"]>0 for x in results])>=.8},
 "results":results}
summary["gate"]={k:bool(v) for k,v in summary["gate"].items()}
summary["gate"]["pass"]=bool(all(summary["gate"].values()))
print("CAL1_RESULT_JSON="+json.dumps(summary,separators=(",",":")),flush=True)
