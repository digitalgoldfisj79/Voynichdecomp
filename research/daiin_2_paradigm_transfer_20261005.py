#!/usr/bin/env python3
"""
DAIIN-2: paradigm-transfer test for the promoted EIN-like hypothesis.

Post-selection confirmatory test (not pristine prospective discovery):
- Voynich forms frozen by spelling only: top 8 ED1 neighbours of daiin.
- Voynich discovery = physical folds 2/3; final = folds 0/1. Fold 4 unused.
- ReF 15th-c manuscripts only.
- Learn an 8-way Voynich-form <-> German grammatical-cell assignment on Bavarian ReF
  using only language-agnostic node features:
    within-paradigm frequency, line-start, line-end, exact self-repeat,
    mean contextual distinctiveness (mean pairwise neighbour JSD).
- Freeze that assignment.
- Test same grammatical-cell identities in Alemannic ReF and held-out Voynich folds.
- Rank EIN against all ReF lemmas with enough cells and matched total frequency.
- German words/lemmas are NEVER used to construct Voynich form classes.
"""
import collections, hashlib, io, json, math, re, tarfile, urllib.request
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import linear_sum_assignment

V_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/92ec41cb26d233a388b6f65fa1a4b7c45d7ad8c5/voynich_transcriptions_slim.json"
V_SHA="26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f"
REF_URL="https://zenodo.org/api/records/5793616/files/ReF-v1.0.2.tar.gz/content"
FORMS=["daiin","aiin","dain","saiin","kaiin","raiin","odaiin","taiin"]
PAIRS=[(1,8),(2,7),(3,6),(4,5),(9,16),(10,15),(11,14),(17,24),(18,23),(19,22),(20,21),
(25,32),(26,31),(27,30),(28,29),(33,40),(34,39),(35,38),(36,37),(41,48),(42,47),
(43,46),(44,45),(49,56),(50,55),(51,54),(52,53),(57,66),(58,65),(67,68),(69,70),
(71,72),(75,84),(76,83),(77,82),(78,81),(79,80),(85,86),(87,90),(88,89),(93,96),
(94,95),(99,102),(100,101),(103,116),(104,115),(105,114),(106,113),(107,112),(108,111)]
BIF={n:f"B{a:03d}_{b:03d}" for a,b in PAIRS for n in (a,b)}
FOLD={b:i%5 for i,b in enumerate(sorted(set(BIF.values())))}

def fnum(f):
    m=re.match(r"f(\d+)",str(f));return int(m.group(1)) if m else None
def jsd(a,b):
    ks=set(a)|set(b)
    if not ks:return 0.
    x=np.array([a.get(k,0.) for k in ks],float);y=np.array([b.get(k,0.) for k in ks],float)
    if x.sum()==0 or y.sum()==0:return 1.
    x/=x.sum();y/=y.sum();m=(x+y)/2
    def kl(p,q):z=p>0;return float(np.sum(p[z]*np.log2(p[z]/q[z])))
    return .5*kl(x,m)+.5*kl(y,m)
def zscore_cols(X):
    X=np.array(X,float);mu=X.mean(0);sd=X.std(0);sd[sd<1e-9]=1.
    return (X-mu)/sd,mu,sd
def apply_z(X,mu,sd):return (np.array(X,float)-mu)/sd
def child(el,name):
    for x in el:
        if x.tag.split("}")[-1]==name:return x.attrib.get("tag","")
    return ""
def rrng(s):
    if not s:return (None,None)
    if ".." in s:return tuple(s.split("..",1))
    return s,s
def headmeta(root):
    h=next((x for x in root.iter() if x.tag.split("}")[-1]=="header"),None)
    out={}
    for line in ((h.text or "") if h is not None else "").splitlines():
        if ":" in line:
            k,v=line.split(":",1);out[k.strip().lower()]=v.strip()
    return out

# -------- Voynich --------
raw=urllib.request.urlopen(V_URL,timeout=120).read()
if hashlib.sha256(raw).hexdigest()!=V_SHA:raise RuntimeError("Voynich SHA mismatch")
V=json.loads(raw)
vrows=[]
for fol,ld in V["pages"].items():
    n=fnum(fol)
    if n not in BIF:continue
    fold=FOLD[BIF[n]]
    for lid,rec in ld.items():
        if str(rec.get("u",""))!="+P0":continue
        xs=[x.lower() for x in rec.get("t",{}).get("ZLZI","").split() if re.fullmatch(r"[a-z]+",x.lower())]
        if not xs:continue
        for i,t in enumerate(xs):
            vrows.append({"fold":fold,"t":t,"start":i==0,"end":i==len(xs)-1,
                          "left":xs[i-1] if i else None,"right":xs[i+1] if i+1<len(xs) else None})

def vfeatures(folds):
    rr=[r for r in vrows if r["fold"] in folds]
    cnt=collections.Counter(r["t"] for r in rr if r["t"] in FORMS)
    ng={t:collections.Counter() for t in FORMS};st=collections.Counter();en=collections.Counter();selfn=collections.Counter()
    for r in rr:
        t=r["t"]
        if t not in FORMS:continue
        st[t]+=r["start"];en[t]+=r["end"]
        if r["left"]:ng[t]["L:"+r["left"]]+=1
        if r["right"]:
            ng[t]["R:"+r["right"]]+=1
            selfn[t]+=r["right"]==t
    total=sum(cnt.values())
    D=np.zeros((len(FORMS),len(FORMS)))
    for i,a in enumerate(FORMS):
        for j,b in enumerate(FORMS):D[i,j]=jsd(ng[a],ng[b])
    feat=[]
    for i,t in enumerate(FORMS):
        n=cnt[t]
        feat.append([math.log((n+0.5)/(total+0.5*len(FORMS))),
                     st[t]/n if n else 0.,en[t]/n if n else 0.,selfn[t]/n if n else 0.,
                     float(np.mean([D[i,j] for j in range(len(FORMS)) if j!=i]))])
    return {"counts":cnt,"features":np.array(feat),"D":D}
VD=vfeatures({2,3});VT=vfeatures({0,1})
print("DAIIN2_V_COUNTS",json.dumps({"disc":dict(VD["counts"]),"final":dict(VT["counts"])}),flush=True)

# -------- ReF 15c manuscripts --------
print("DAIIN2_REF_DOWNLOAD",flush=True)
rb=urllib.request.urlopen(REF_URL,timeout=300).read();tar=tarfile.open(fileobj=io.BytesIO(rb),mode="r:gz")
docs=[]
for name in [n for n in tar.getnames() if n.endswith(".xml")]:
    try:root=ET.fromstring(tar.extractfile(name).read())
    except:continue
    md=headmeta(root);med=md.get("medium","").lower();tm=md.get("time","").lower()
    if "handschrift" not in med or not tm.startswith("15,"):continue
    reg=md.get("language-region","").lower();area=md.get("language-area","").lower()
    bav=(("bair" in area or "bayr" in area or "österreich" in area or "oesterreich" in area) and "alemann" not in area)
    alem=(("alemann" in area or "schwäb" in area or "elsäss" in area) and "bair" not in area and "bayr" not in area)
    if not (bav or alem):continue
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
        fd=ds[0].attrib.get("id") if ds else None;ld=ds[-1].attrib.get("id") if ds else None
        for mi,m in enumerate(ms):
            le=child(m,"lemma");po=child(m,"pos");morph=child(m,"morph") or child(m,"inflection") or "--"
            norm=(m.attrib.get("ascii") or m.attrib.get("utf") or m.attrib.get("trans") or "").strip().lower()
            valid=bool(le and le not in ("--","[!]") and po and not po.startswith("$") and norm)
            if mi==0 and fd in starts and cur:lines.append(cur);cur=[]
            if valid:cur.append({"cell":(le,po,morph),"norm":norm})
            if mi==len(ms)-1 and ld in ends:
                if cur:lines.append(cur);cur=[]
    if cur:lines.append(cur)
    if lines:docs.append({"bav":bav,"alem":alem,"lines":lines,"name":name})
print("DAIIN2_REF_DOCS",len(docs),collections.Counter(("BAV" if d["bav"] else "ALEM") for d in docs),flush=True)

def corpus_stats(which):
    cc=collections.Counter();lc=collections.Counter();st=collections.Counter();en=collections.Counter()
    selfn=collections.Counter();ng=collections.defaultdict(collections.Counter)
    for d in docs:
        if which=="BAV" and not d["bav"]:continue
        if which=="ALEM" and not d["alem"]:continue
        for line in d["lines"]:
            for i,x in enumerate(line):
                c=x["cell"];cc[c]+=1;lc[c[0]]+=1
                st[c]+=i==0;en[c]+=i==len(line)-1
                if i:ng[c]["L:"+line[i-1]["norm"]]+=1
                if i+1<len(line):
                    ng[c]["R:"+line[i+1]["norm"]]+=1
                    selfn[c]+=line[i+1]["cell"]==c
    return cc,lc,st,en,selfn,ng
GS={k:corpus_stats(k) for k in ("BAV","ALEM")}

def lemma_bundle(which,lemma,k=8):
    cc,lc,st,en,selfn,ng=GS[which]
    cells=[(c,n) for c,n in cc.items() if c[0]==lemma and n>=10]
    cells.sort(key=lambda z:(-z[1],z[0]))
    if len(cells)<k:return None
    cells=cells[:k];tot=sum(n for _,n in cells)
    D=np.zeros((k,k))
    for i,(a,_) in enumerate(cells):
        for j,(b,_) in enumerate(cells):D[i,j]=jsd(ng[a],ng[b])
    feat=[]
    for i,(c,n) in enumerate(cells):
        feat.append([math.log((n+0.5)/(tot+0.5*k)),st[c]/n,en[c]/n,selfn[c]/n,
                     float(np.mean([D[i,j] for j in range(k) if j!=i]))])
    return {"cells":[c for c,n in cells],"counts":[n for c,n in cells],"features":np.array(feat),"D":D}

# Candidate lemmas fixed from Bavarian corpus by richness/frequency, not by POS.
ccB,lcB,*_=GS["BAV"]
vshare=sum(VD["counts"].values())/max(1,len([r for r in vrows if r["fold"] in (2,3)]))
cands=[]
for lemma,total in lcB.items():
    b=lemma_bundle("BAV",lemma,8)
    a=lemma_bundle("ALEM",lemma,8)
    if b is None or a is None:continue
    # loose frequency match so highly implausible content lemmas do not dominate
    # target here is family share; factor 8 is intentionally permissive.
    corpusN=sum(ccB.values());share=total/corpusN
    if share<=0 or abs(math.log((share+1e-12)/(vshare+1e-12)))>math.log(8):continue
    cands.append(lemma)
print("DAIIN2_CANDIDATES",len(cands),flush=True)

# Assignment uses standardized node features jointly on Voynich discovery + German Bav.
# Feature weights frozen here before candidate ranking.
W=np.array([2.0,0.75,0.75,1.0,2.0])
def learn_assignment(gb):
    X=np.vstack([VD["features"],gb["features"]]);Z,mu,sd=zscore_cols(X)
    VZ=Z[:8];GZ=Z[8:]
    C=((VZ[:,None,:]-GZ[None,:,:])**2*W).sum(2)
    ri,ci=linear_sum_assignment(C)
    mp={int(i):int(j) for i,j in zip(ri,ci)}
    return mp,mu,sd,float(C[ri,ci].mean())

def transfer_loss(gb,ga,mp,mu,sd):
    Vfin=apply_z(VT["features"],mu,sd);AZ=apply_z(ga["features"],mu,sd)
    node=np.mean([np.sum(W*(Vfin[i]-AZ[mp[i]])**2) for i in range(8)])
    # relation term: does the pairwise context-distance geometry transfer?
    pairs=[]
    for i in range(8):
        for j in range(i+1,8):
            pairs.append((VT["D"][i,j]-ga["D"][mp[i],mp[j]])**2)
    rel=float(np.mean(pairs))
    return float(node+2.0*rel),float(node),rel

rows=[]
for lemma in cands:
    gb=lemma_bundle("BAV",lemma,8);ga=lemma_bundle("ALEM",lemma,8)
    mp,mu,sd,train=learn_assignment(gb)
    test,node,rel=transfer_loss(gb,ga,mp,mu,sd)
    rows.append({"lemma":lemma,"train_loss":train,"test_loss":test,"test_node":node,"test_relation":rel,
                 "mapping":[{"v":FORMS[i],"cell_bav":gb["cells"][mp[i]],"cell_alem_same_index":ga["cells"][mp[i]],
                             "bav_count":gb["counts"][mp[i]],"alem_count":ga["counts"][mp[i]]} for i in range(8)]})
rows.sort(key=lambda x:(x["test_loss"],x["train_loss"]))
for i,x in enumerate(rows,1):x["rank"]=i

ein=[x for x in rows if x["lemma"] in ("ein","èin")]
named={}
for q in ["ein","èin","der","dër","sein","sîn","haben","ich","er","ër","mein","unser","kein"]:
    h=next((x for x in rows if x["lemma"]==q),None)
    if h:named[q]={"rank":h["rank"],"train_loss":h["train_loss"],"test_loss":h["test_loss"],"mapping":h["mapping"]}

# Hostile null: permute Voynich form labels relative to discovery node features.
# Mapping algorithm can reassign, so null perturbs form-specific heldout correspondence:
# shuffle final-form rows while keeping discovery assignment fixed.
rng=np.random.default_rng(20261005)
einrow=ein[0] if ein else None
null=[]
if einrow:
    gb=lemma_bundle("BAV",einrow["lemma"],8);ga=lemma_bundle("ALEM",einrow["lemma"],8)
    mp,mu,sd,_=learn_assignment(gb);AZ=apply_z(ga["features"],mu,sd)
    for _ in range(500):
        perm=rng.permutation(8);Vfin=apply_z(VT["features"][perm],mu,sd)
        node=np.mean([np.sum(W*(Vfin[i]-AZ[mp[i]])**2) for i in range(8)])
        pairs=[]
        for i in range(8):
            for j in range(i+1,8):
                pairs.append((VT["D"][perm[i],perm[j]]-ga["D"][mp[i],mp[j]])**2)
        null.append(float(node+2*np.mean(pairs)))
    obs=einrow["test_loss"];nm=float(np.mean(null));ns=float(np.std(null,ddof=1));z=(nm-obs)/ns if ns else None
else:
    obs=nm=ns=z=None

out={"phase":"DAIIN_2","status":"complete","forms":FORMS,
     "design":"assignment learned Bavarian ReF15 from Voynich folds2/3; frozen mapping evaluated on Alemannic ReF15 + Voynich folds0/1",
     "candidate_count":len(rows),"top":[x for x in rows[:30]],"named":named,
     "ein_null":{"observed_loss":obs,"null_mean":nm,"null_sd":ns,"advantage_z":z,"n":len(null)}}
print("DAIIN2_TOP="+json.dumps([{"rank":x["rank"],"lemma":x["lemma"],"train":x["train_loss"],"test":x["test_loss"]} for x in rows[:20]],ensure_ascii=False,separators=(",",":")),flush=True)
print("DAIIN2_NAMED="+json.dumps(named,ensure_ascii=False,separators=(",",":")),flush=True)
print("DAIIN2_EIN_NULL="+json.dumps(out["ein_null"],separators=(",",":")),flush=True)
print("DAIIN2_RESULT_JSON="+json.dumps(out,ensure_ascii=False,separators=(",",":")),flush=True)
