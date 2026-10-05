#!/usr/bin/env python3
"""
DAIIN-2b — corrected cross-dialect paradigm transfer.

Supersedes DAIIN-2 commit 2dc4... because that run:
(1) matched Bavarian/Alemannic cells by within-dialect frequency index rather than exact grammatical identity;
(2) reconstructed physical folds by modulo instead of using the frozen bifolium folds.

Corrected design:
- strict +P0 ZLZI
- frozen physical folds imported from frozen FORM commit
- Voynich forms fixed before German fitting:
  daiin aiin dain saiin kaiin raiin odaiin taiin
- candidate paradigm = (lemma, paradigm-class), preventing homonymous POS mixing
  e.g. ein|DIART excludes ein|APPR; ihr|PPER excludes ihr|DPOSA.
- candidate must have >=8 exact grammatical cells present in BOTH Bavarian and Alemannic ReF15 manuscripts
- choose the 8 cells by Bavarian discovery count only among cross-dialect-attested cells
- exact same cell identities are used in transfer
- symmetric directions:
    V discovery + Bavarian -> V final + Alemannic
    V discovery + Alemannic -> V final + Bavarian
- score node-feature transfer + pairwise contextual-distance geometry
- hostile null shuffles held-out Voynich form identities with mapping fixed.
"""
import collections, hashlib, io, json, math, re, tarfile, urllib.request
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import linear_sum_assignment

SEED=20261005
V_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/92ec41cb26d233a388b6f65fa1a4b7c45d7ad8c5/voynich_transcriptions_slim.json"
V_SHA="26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f"
REF_URL="https://zenodo.org/api/records/5793616/files/ReF-v1.0.2.tar.gz/content"
LAT_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/74ae7aa0088dc07436000f777d9c5d7d705730c2/research/latent_line_state_test_20261004.py"
FORMS=["daiin","aiin","dain","saiin","kaiin","raiin","odaiin","taiin"]

m={"__name__":"daiin2b_latent"}
exec(compile(urllib.request.urlopen(LAT_URL,timeout=120).read().decode(),LAT_URL,"exec"),m)
folds=m["folds"]

PAIRS=[(1,8),(2,7),(3,6),(4,5),(9,16),(10,15),(11,14),(17,24),(18,23),(19,22),(20,21),
(25,32),(26,31),(27,30),(28,29),(33,40),(34,39),(35,38),(36,37),(41,48),(42,47),
(43,46),(44,45),(49,56),(50,55),(51,54),(52,53),(57,66),(58,65),(67,68),(69,70),
(71,72),(75,84),(76,83),(77,82),(78,81),(79,80),(85,86),(87,90),(88,89),(93,96),
(94,95),(99,102),(100,101),(103,116),(104,115),(105,114),(106,113),(107,112),(108,111)]
BIF={n:f"B{a:03d}_{b:03d}" for a,b in PAIRS for n in (a,b)}

def fnum(f):
    z=re.match(r"f(\d+)",str(f));return int(z.group(1)) if z else None
def jsd(a,b):
    ks=set(a)|set(b)
    if not ks:return 0.
    x=np.array([a.get(k,0.) for k in ks],float);y=np.array([b.get(k,0.) for k in ks],float)
    if x.sum()==0 or y.sum()==0:return 1.
    x/=x.sum();y/=y.sum();mm=(x+y)/2
    def kl(p,q):
        z=p>0;return float(np.sum(p[z]*np.log2(p[z]/q[z])))
    return .5*kl(x,mm)+.5*kl(y,mm)
def zfit(X):
    X=np.asarray(X,float);mu=X.mean(0);sd=X.std(0);sd[sd<1e-9]=1.
    return (X-mu)/sd,mu,sd
def zap(X,mu,sd):return (np.asarray(X,float)-mu)/sd
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
def pclass(pos):
    # Keep grammatical systems separate; unify inflectional verb forms only.
    if pos.startswith("V"): return "VERB"
    return pos

# Voynich
raw=urllib.request.urlopen(V_URL,timeout=120).read()
if hashlib.sha256(raw).hexdigest()!=V_SHA:raise RuntimeError("Voynich SHA")
V=json.loads(raw);vr=[]
for fol,ld in V["pages"].items():
    n=fnum(fol)
    if n not in BIF or BIF[n] not in folds:continue
    fd=int(folds[BIF[n]])
    for lid,rec in ld.items():
        if str(rec.get("u",""))!="+P0":continue
        xs=[x.lower() for x in rec.get("t",{}).get("ZLZI","").split() if re.fullmatch(r"[a-z]+",x.lower())]
        for i,t in enumerate(xs):
            vr.append({"fold":fd,"t":t,"start":i==0,"end":i==len(xs)-1,
                       "left":xs[i-1] if i else None,"right":xs[i+1] if i+1<len(xs) else None})

def vf(fs):
    rr=[r for r in vr if r["fold"] in fs];cnt=collections.Counter(r["t"] for r in rr if r["t"] in FORMS)
    st=collections.Counter();en=collections.Counter();sn=collections.Counter();ng={t:collections.Counter() for t in FORMS}
    for r in rr:
        t=r["t"]
        if t not in FORMS:continue
        st[t]+=r["start"];en[t]+=r["end"]
        if r["left"]:ng[t]["L:"+r["left"]]+=1
        if r["right"]:
            ng[t]["R:"+r["right"]]+=1;sn[t]+=r["right"]==t
    total=sum(cnt.values());D=np.zeros((8,8))
    for i,a in enumerate(FORMS):
        for j,b in enumerate(FORMS):D[i,j]=jsd(ng[a],ng[b])
    F=[]
    for i,t in enumerate(FORMS):
        n=cnt[t]
        F.append([math.log((n+.5)/(total+4.0)),st[t]/n if n else 0.,en[t]/n if n else 0.,
                  sn[t]/n if n else 0.,float(np.mean([D[i,j] for j in range(8) if j!=i]))])
    return {"counts":cnt,"features":np.array(F),"D":D}
VD=vf({2,3});VF=vf({0,1})
print("D2B_VCOUNTS",json.dumps({"disc":dict(VD["counts"]),"final":dict(VF["counts"])}),flush=True)

# ReF15 manuscripts
print("D2B_REF_DOWNLOAD",flush=True)
rb=urllib.request.urlopen(REF_URL,timeout=300).read();tar=tarfile.open(fileobj=io.BytesIO(rb),mode="r:gz");docs=[]
for name in [n for n in tar.getnames() if n.endswith(".xml")]:
    try:root=ET.fromstring(tar.extractfile(name).read())
    except:continue
    md=header(root);med=md.get("medium","").lower();tm=md.get("time","").lower()
    if "handschrift" not in med or not tm.startswith("15,"):continue
    area=md.get("language-area","").lower()
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
        first=ds[0].attrib.get("id") if ds else None;last=ds[-1].attrib.get("id") if ds else None
        for mi,x in enumerate(ms):
            le=child(x,"lemma");pos=child(x,"pos");mor=child(x,"morph") or child(x,"inflection") or "--"
            norm=(x.attrib.get("ascii") or x.attrib.get("utf") or x.attrib.get("trans") or "").strip().lower()
            valid=bool(le and le not in ("--","[!]") and pos and not pos.startswith("$") and norm)
            if mi==0 and first in starts and cur:lines.append(cur);cur=[]
            if valid:cur.append({"cell":(le,pos,mor),"group":(le,pclass(pos)),"norm":norm})
            if mi==len(ms)-1 and last in ends:
                if cur:lines.append(cur);cur=[]
    if cur:lines.append(cur)
    if lines:docs.append({"side":"BAV" if bav else "ALEM","lines":lines,"name":name})
print("D2B_DOCS",collections.Counter(d["side"] for d in docs),flush=True)

def stats(side):
    cc=collections.Counter();gc=collections.Counter();st=collections.Counter();en=collections.Counter();sn=collections.Counter();ng=collections.defaultdict(collections.Counter)
    for d in docs:
        if d["side"]!=side:continue
        for line in d["lines"]:
            for i,x in enumerate(line):
                c=x["cell"];g=x["group"];cc[c]+=1;gc[g]+=1
                st[c]+=i==0;en[c]+=i==len(line)-1
                if i:ng[c]["L:"+line[i-1]["norm"]]+=1
                if i+1<len(line):
                    ng[c]["R:"+line[i+1]["norm"]]+=1;sn[c]+=line[i+1]["cell"]==c
    return {"cc":cc,"gc":gc,"st":st,"en":en,"sn":sn,"ng":ng}
S={x:stats(x) for x in ("BAV","ALEM")}

def common_bundle(group,k=8):
    # exact cells shared with >=10 attestations in BOTH dialects.
    cb=S["BAV"]["cc"];ca=S["ALEM"]["cc"]
    cs=[c for c,n in cb.items() if (c[0],pclass(c[1]))==group and n>=10 and ca.get(c,0)>=10]
    cs.sort(key=lambda c:(-cb[c],c)) # select by Bavarian counts only
    if len(cs)<k:return None
    cells=cs[:k]
    out={}
    for side in ("BAV","ALEM"):
        ss=S[side];cnt=[ss["cc"][c] for c in cells];tot=sum(cnt);D=np.zeros((k,k))
        for i,a in enumerate(cells):
            for j,b in enumerate(cells):D[i,j]=jsd(ss["ng"][a],ss["ng"][b])
        feat=[]
        for i,(c,n) in enumerate(zip(cells,cnt)):
            feat.append([math.log((n+.5)/(tot+.5*k)),ss["st"][c]/n,ss["en"][c]/n,ss["sn"][c]/n,
                         float(np.mean([D[i,j] for j in range(k) if j!=i]))])
        out[side]={"cells":cells,"counts":cnt,"features":np.array(feat),"D":D}
    return out

# Candidate groups: enough shared cells and rough family-frequency match.
vN=sum(VD["counts"].values());vTot=sum(1 for r in vr if r["fold"] in (2,3));vshare=vN/vTot
groups=[]
for g,total in S["BAV"]["gc"].items():
    b=common_bundle(g,8)
    if b is None:continue
    sh=total/sum(S["BAV"]["cc"].values())
    if sh>0 and abs(math.log((sh+1e-12)/(vshare+1e-12)))<=math.log(8):groups.append(g)
print("D2B_CANDIDATES",len(groups),flush=True)

W=np.array([2.,.75,.75,1.,2.])
def learn(gfeat):
    X=np.vstack([VD["features"],gfeat]);Z,mu,sd=zfit(X);A=Z[:8];B=Z[8:]
    C=((A[:,None,:]-B[None,:,:])**2*W).sum(2);ri,ci=linear_sum_assignment(C)
    return {int(i):int(j) for i,j in zip(ri,ci)},mu,sd,float(C[ri,ci].mean())

def test(target,mp,mu,sd,perm=None):
    Vfeat=VF["features"] if perm is None else VF["features"][perm]
    VDmat=VF["D"] if perm is None else VF["D"][np.ix_(perm,perm)]
    A=zap(Vfeat,mu,sd);B=zap(target["features"],mu,sd)
    node=float(np.mean([np.sum(W*(A[i]-B[mp[i]])**2) for i in range(8)]))
    rel=float(np.mean([(VDmat[i,j]-target["D"][mp[i],mp[j]])**2 for i in range(8) for j in range(i+1,8)]))
    return node+2*rel,node,rel

rows=[]
for g in groups:
    bun=common_bundle(g,8)
    mpBA,muBA,sdBA,trBA=learn(bun["BAV"]["features"]);teBA,nodeBA,relBA=test(bun["ALEM"],mpBA,muBA,sdBA)
    mpAB,muAB,sdAB,trAB=learn(bun["ALEM"]["features"]);teAB,nodeAB,relAB=test(bun["BAV"],mpAB,muAB,sdAB)
    # mapping agreement compares exact grammatical cell identities.
    invBA={FORMS[i]:bun["BAV"]["cells"][mpBA[i]] for i in range(8)}
    invAB={FORMS[i]:bun["ALEM"]["cells"][mpAB[i]] for i in range(8)}
    agree=sum(invBA[v]==invAB[v] for v in FORMS)/8
    rows.append({"group":g,"test_mean":(teBA+teAB)/2,"train_mean":(trBA+trAB)/2,
                 "BAV_to_ALEM":{"train":trBA,"test":teBA,"node":nodeBA,"relation":relBA},
                 "ALEM_to_BAV":{"train":trAB,"test":teAB,"node":nodeAB,"relation":relAB},
                 "assignment_agreement":agree,
                 "mapping_BAV":[{"v":v,"cell":invBA[v]} for v in FORMS],
                 "mapping_ALEM":[{"v":v,"cell":invAB[v]} for v in FORMS]})
rows.sort(key=lambda x:(x["test_mean"],x["train_mean"]))
for i,x in enumerate(rows,1):x["rank"]=i

# Nulls for EIN|DIART and the top candidate, each direction, fixed maps.
rng=np.random.default_rng(SEED)
def null_for(row,n=1000):
    g=tuple(row["group"]);bun=common_bundle(g,8);vals=[];obs=[]
    for src,tgt in (("BAV","ALEM"),("ALEM","BAV")):
        mp,mu,sd,_=learn(bun[src]["features"]);o,_,_=test(bun[tgt],mp,mu,sd);obs.append(o)
        z=[]
        for _ in range(n):
            p=rng.permutation(8);q,_,_=test(bun[tgt],mp,mu,sd,p);z.append(q)
        vals.append(z)
    null=np.mean(np.array(vals),axis=0);o=float(np.mean(obs));nm=float(np.mean(null));ns=float(np.std(null,ddof=1))
    return {"observed":o,"null_mean":nm,"null_sd":ns,"advantage_z":(nm-o)/ns if ns else None,"n":n}
einrow=next((x for x in rows if tuple(x["group"])==("ein","DIART")),None)
toprow=rows[0] if rows else None
nulls={}
if einrow:nulls["ein_DIART"]=null_for(einrow)
if toprow:nulls["top"]=null_for(toprow)

out={"phase":"DAIIN_2B","status":"complete","forms":FORMS,
     "corrections":["frozen physical folds","exact same grammatical cells across dialects","paradigm-class prevents POS-homonym mixing","symmetric dialect transfer"],
     "candidate_count":len(rows),"top":rows[:30],
     "ein":einrow,"top_nulls":nulls}
print("D2B_TOP="+json.dumps([{"rank":x["rank"],"group":x["group"],"test":x["test_mean"],"train":x["train_mean"],"agree":x["assignment_agreement"]} for x in rows[:20]],ensure_ascii=False,separators=(",",":")),flush=True)
print("D2B_EIN="+json.dumps(einrow,ensure_ascii=False,separators=(",",":")),flush=True)
print("D2B_NULLS="+json.dumps(nulls,separators=(",",":")),flush=True)
print("DAIIN2B_RESULT_JSON="+json.dumps(out,ensure_ascii=False,separators=(",",":")),flush=True)
