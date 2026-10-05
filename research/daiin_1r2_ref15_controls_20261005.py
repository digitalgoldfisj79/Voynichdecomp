#!/usr/bin/env python3
"""DAIIN-1R2: 15th-century ReF manuscript/dialect holdout + cross-token controls.
Frozen ED1-only DAIIN-1 score. No retuning after ReM.
"""
import collections, hashlib, io, json, math, re, tarfile, urllib.request
from itertools import combinations
import xml.etree.ElementTree as ET
import numpy as np

V_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/92ec41cb26d233a388b6f65fa1a4b7c45d7ad8c5/voynich_transcriptions_slim.json"
V_SHA="26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f"
REF_URL="https://zenodo.org/api/records/5793616/files/ReF-v1.0.2.tar.gz/content"

def lev(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]
def simpson(c):
    n=sum(c.values())
    if not n:return 1.
    p=np.array(list(c.values()),float)/n;return float((p*p).sum())
def jsdc(a,b):
    ks=set(a)|set(b)
    if not ks:return 0.
    x=np.array([a.get(k,0.) for k in ks],float);y=np.array([b.get(k,0.) for k in ks],float)
    if x.sum()==0 or y.sum()==0:return 1.
    x/=x.sum();y/=y.sum();m=(x+y)/2
    def kl(p,q):z=p>0;return float(np.sum(p[z]*np.log2(p[z]/q[z])))
    return .5*kl(x,m)+.5*kl(y,m)
def jsdv(a,b):
    n=max(len(a),len(b));x=np.zeros(n);y=np.zeros(n);x[:len(a)]=a;y[:len(b)]=b
    if x.sum()==0 or y.sum()==0:return 1.
    x/=x.sum();y/=y.sum();m=(x+y)/2
    def kl(p,q):z=p>0;return float(np.sum(p[z]*np.log2(p[z]/q[z])))
    return .5*kl(x,m)+.5*kl(y,m)
def pct(v):
    o=np.argsort(v,kind="mergesort");r=np.empty(len(v),float)
    for k,i in enumerate(o):r[i]=k/max(1,len(v)-1)
    return r
def child(el,name):
    for x in el:
        if x.tag.split("}")[-1]==name:return x.attrib.get("tag","")
    return ""
def rrng(s):
    if not s:return None,None
    if ".." in s:return tuple(s.split("..",1))
    return s,s
def headmeta(root):
    h=next((x for x in root.iter() if x.tag.split("}")[-1]=="header"),None)
    txt=(h.text or "") if h is not None else ""
    d={}
    for line in txt.splitlines():
        if ":" in line:
            k,v=line.split(":",1);d[k.strip().lower()]=v.strip()
    return d

# Voynich targets/fingerprints
raw=urllib.request.urlopen(V_URL,timeout=120).read()
if hashlib.sha256(raw).hexdigest()!=V_SHA:raise RuntimeError("V SHA")
V=json.loads(raw);vlines=[]
for fol,ld in V["pages"].items():
    for lid,rec in ld.items():
        if str(rec.get("u",""))!="+P0":continue
        xs=[x.lower() for x in rec.get("t",{}).get("ZLZI","").split() if re.fullmatch(r"[a-z]+",x.lower())]
        if xs:vlines.append(xs)
vf=collections.Counter(t for xs in vlines for t in xs);VN=sum(vf.values())
TARGETS=[t for t,n in vf.most_common(20)]

def tfp(t):
    left=collections.Counter();right=collections.Counter();cnt=st=en=selfn=0;cache={}
    def ctx(q):
        if q in cache:return cache[q]
        c=collections.Counter()
        for xs in vlines:
            for i,z in enumerate(xs):
                if z!=q:continue
                if i:c[xs[i-1]]+=1
                if i+1<len(xs):c[xs[i+1]]+=1
        cache[q]=c;return c
    for xs in vlines:
        for i,z in enumerate(xs):
            if z!=t:continue
            cnt+=1;st+=i==0;en+=i==len(xs)-1
            if i:left[xs[i-1]]+=1
            if i+1<len(xs):
                right[xs[i+1]]+=1;selfn+=xs[i+1]==t
    fs=[q for q,n in vf.items() if n>=3 and lev(q,t)<=1]
    fs.sort(key=lambda q:(-vf[q],q));cs=np.array([vf[q] for q in fs],float)
    prof=(cs/cs.sum()).tolist();top=fs[:min(8,len(fs))]
    pp=[jsdc(ctx(a),ctx(b)) for i,a in enumerate(top) for b in top[i+1:]]
    return {"share":cnt/VN,"start":st/cnt,"end":en/cnt,"self":selfn/cnt,"sp":simpson(left+right),
            "within":vf[t]/cs.sum(),"profile":prof,"sep":float(np.mean(pp)) if pp else 0.}
VFP={t:tfp(t) for t in TARGETS}
print("R2_TARGETS",json.dumps([(t,vf[t]) for t in TARGETS]),flush=True)

# ReF parse, retaining panel metadata.
print("R2_REF_DOWNLOAD",flush=True)
rb=urllib.request.urlopen(REF_URL,timeout=300).read();tar=tarfile.open(fileobj=io.BytesIO(rb),mode="r:gz")
docs=[]
for fi,name in enumerate([n for n in tar.getnames() if n.endswith(".xml")]):
    try:root=ET.fromstring(tar.extractfile(name).read())
    except:continue
    md=headmeta(root)
    med=md.get("medium","").lower();tm=md.get("time","").lower();reg=md.get("language-region","").lower();area=md.get("language-area","").lower()
    isms="handschrift" in med
    is15=tm.startswith("15,") or tm=="15"
    h1=tm.startswith("15,1")
    east="ostoberdeutsch" in reg
    west="westoberdeutsch" in reg
    bav=(("bair" in area or "bayr" in area or "österreich" in area or "oesterreich" in area) and "alemann" not in area)
    alem=(("alemann" in area or "schwäb" in area or "elsäss" in area) and "bair" not in area and "bayr" not in area)
    if not (isms and is15):continue
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
            le=child(m,"lemma");po=child(m,"pos");inf=child(m,"morph") or child(m,"inflection") or "--"
            norm=(m.attrib.get("ascii") or m.attrib.get("utf") or m.attrib.get("trans") or "").strip().lower()
            valid=bool(le and le not in ("--","[!]") and po and not po.startswith("$") and norm)
            if mi==0 and fd in starts and cur:lines.append(cur);cur=[]
            if valid:cur.append({"cell":(le,po,inf),"norm":norm})
            if mi==len(ms)-1 and ld in ends:
                if cur:lines.append(cur);cur=[]
    if cur:lines.append(cur)
    if lines:docs.append({"name":name,"h1":h1,"east":east,"west":west,"bav":bav,"alem":alem,"area":area,"lines":lines})
    if (fi+1)%100==0:print("R2_PARSE",fi+1,"docs",len(docs),flush=True)
print("R2_DOCS",len(docs),collections.Counter((d["h1"],d["east"],d["bav"],d["alem"]) for d in docs).most_common(12),flush=True)

PANELS={
 "REF15_ALL_MS":lambda d:True,
 "REF15_EAST_MS":lambda d:d["east"],
 "REF15_BAV_MS":lambda d:d["bav"],
 "REF15_ALEM_MS":lambda d:d["alem"],
 "REF15H1_BAV_MS":lambda d:d["h1"] and d["bav"],
 "REF15H1_ALEM_MS":lambda d:d["h1"] and d["alem"],
}
MET=["df","ds","de","dself","dsp","dc","dp","dx"]
W={"df":2.,"ds":.5,"de":.5,"dself":1.,"dsp":1.,"dc":1.,"dp":2.,"dx":2.}

def german(panel):
    pred=PANELS[panel];cc=collections.Counter();lc=collections.Counter();forms=collections.defaultdict(collections.Counter)
    st=collections.Counter();en=collections.Counter();sn=collections.Counter();ng=collections.defaultdict(collections.Counter);N=0;nd=0
    for d in docs:
        if not pred(d):continue
        nd+=1
        for line in d["lines"]:
            for i,x in enumerate(line):
                c=x["cell"];N+=1;cc[c]+=1;lc[c[0]]+=1;forms[c][x["norm"]]+=1
                if i==0:st[c]+=1
                if i==len(line)-1:en[c]+=1
                if i:ng[c][line[i-1]["norm"]]+=1
                if i+1<len(line):
                    ng[c][line[i+1]["norm"]]+=1
                    if line[i+1]["cell"]==c:sn[c]+=1
    if N<5000:return N,nd,[]
    by=collections.defaultdict(list)
    for c,n in cc.items():by[c[0]].append((c,n))
    prof={};sep={}
    for le,a in by.items():
        a.sort(key=lambda z:(-z[1],z[0]));tot=sum(n for _,n in a);prof[le]=[n/tot for _,n in a]
        top=[c for c,n in a[:8] if n>=3];pp=[jsdc(ng[x],ng[y]) for i,x in enumerate(top) for y in top[i+1:]]
        sep[le]=float(np.mean(pp)) if pp else 0.
    mn=max(20,int(N*.0001));rows=[]
    for c,n in cc.items():
        if n<mn:continue
        le=c[0];rows.append({"cell":c,"share":n/N,"start":st[c]/n,"end":en[c]/n,"self":sn[c]/n,"sp":simpson(ng[c]),
          "within":n/lc[le],"profile":prof[le],"sep":sep[le],"cells":len(by[le]),"forms":forms[c].most_common(8)})
    return N,nd,rows

def score(rows,v):
    rr=[]
    for g in rows:
        x=dict(g);x["df"]=abs(math.log((g["share"]+1e-12)/(v["share"]+1e-12)));x["ds"]=abs(g["start"]-v["start"]);x["de"]=abs(g["end"]-v["end"])
        x["dself"]=abs(math.log((g["self"]+1e-4)/(v["self"]+1e-4)));x["dsp"]=abs(g["sp"]-v["sp"]);x["dc"]=abs(g["within"]-v["within"])
        x["dp"]=jsdv(g["profile"],v["profile"]);x["dx"]=abs(g["sep"]-v["sep"])
        if g["cells"]<2:x["dc"]+=1;x["dp"]+=1;x["dx"]+=1
        rr.append(x)
    for m in MET:
        ps=pct(np.array([x[m] for x in rr],float))
        for x,p in zip(rr,ps):x["p"+m]=p
    ws=sum(W.values())
    for x in rr:x["score"]=sum(W[m]*x["p"+m] for m in MET)/ws
    rr.sort(key=lambda x:x["score"]);return rr

OUT={}
for panel in PANELS:
    N,nd,rows=german(panel)
    if not rows:
        OUT[panel]={"N":N,"docs":nd,"status":"too_small"};continue
    po={}
    for t in TARGETS:
        rr=score(rows,VFP[t])
        def srank(surf):
            for i,x in enumerate(rr):
                if any(f==surf for f,n in x["forms"]):return i+1,x
            return None,None
        rw,xw=srank("was");re,xe=srank("ein");ri,xi=srank("ist")
        po[t]={"was_rank":rw,"ein_rank":re,"ist_rank":ri,"top":[{"cell":x["cell"],"score":x["score"],"forms":x["forms"][:3]} for x in rr[:5]]}
    OUT[panel]={"N":N,"docs":nd,"status":"ok","targets":po}
    print("R2_PANEL",panel,"N",N,"DOCS",nd,"DAIIN",json.dumps(po["daiin"],ensure_ascii=False,separators=(",",":")),flush=True)

# specificity ordering for each surface per panel
for panel,p in OUT.items():
    if p.get("status")!="ok":continue
    for surf,key in [("was","was_rank"),("ein","ein_rank"),("ist","ist_rank")]:
        a=sorted((v.get(key) or 999999,t) for t,v in p["targets"].items())
        print("R2_SPEC",panel,surf,json.dumps([{"target":t,"rank":r} for r,t in a],ensure_ascii=False,separators=(",",":")),flush=True)
res={"phase":"DAIIN_1R2","status":"complete","design":"frozen ED1-only score; ReF 15th-c manuscript holdout; top20 specificity controls","targets":TARGETS,"panels":OUT}
print("DAIIN1R2_RESULT_JSON="+json.dumps(res,ensure_ascii=False,separators=(",",":")),flush=True)
