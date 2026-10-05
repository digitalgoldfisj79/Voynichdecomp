#!/usr/bin/env python3
"""
DAIIN-1 temporal validation: ReF 1.0.2 (1350-1650).

IMPORTANT FIREWALL:
This file freezes the Voynich target and scoring rule established BEFORE ReM discovery output.
Do not modify weights/target after inspecting ReM rankings.

Candidate unit: (lemma, HiTS POS, inflection).
Primary validation window: manuscript texts with ReF time metadata 14,2 or 15,1
(~1350-1450), with dialect subpanels.
Full 1350-1650 panels are robustness only.
"""
import collections, io, json, math, re, tarfile, urllib.request
from itertools import combinations
import numpy as np
import xml.etree.ElementTree as ET

REF_URL="https://zenodo.org/api/records/5793616/files/ReF-v1.0.2.tar.gz/content"

# Frozen Voynich strict +P0 target.
VT={"share":0.02399870277282309,"start":0.20270270270270271,
    "end":0.1608108108108108,"self_next":0.013513513513513514,
    "simpson":0.005339848525864955}
V1_COUNTS=np.array([740,416,191,111,75,58,55,43,19,17,16,15,12,7,6,5,4,3,3,3],float)
V1_PROFILE=(V1_COUNTS/V1_COUNTS.sum()).tolist()
V1_WITHIN=0.4113396331295164
V1_SEP=0.6871803110693901
# ED2 is diagnostic only, never used in score.
V2_COUNTS=np.array([740,416,199,191,133,111,97,77,75,58,56,55,46,45,43,43,40,40,37,37,33,28,27,21,19,19,19,17,17,16,16,15,14,13,12,12,12,11,9,8,7,7,6,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],float)
V2_PROFILE=(V2_COUNTS/V2_COUNTS.sum()).tolist()
V2_WITHIN=0.24478994376447238
V2_SEP=0.6361092810438288

def entropy_norm(c):
    n=sum(c.values())
    if n<=0 or len(c)<=1:return 0.0
    p=np.array(list(c.values()),float)/n
    return float(-(p*np.log2(p)).sum()/math.log2(len(p)))
def simpson(c):
    n=sum(c.values())
    if n<=0:return 1.0
    p=np.array(list(c.values()),float)/n
    return float((p*p).sum())
def jsd_counter(a,b):
    keys=set(a)|set(b)
    if not keys:return 0.0
    pa=np.array([a.get(k,0.0) for k in keys],float);pb=np.array([b.get(k,0.0) for k in keys],float)
    if pa.sum()==0 or pb.sum()==0:return 1.0
    pa/=pa.sum();pb/=pb.sum();m=(pa+pb)/2
    def kl(p,q):
        z=p>0;return float(np.sum(p[z]*np.log2(p[z]/q[z])))
    return .5*kl(pa,m)+.5*kl(pb,m)
def jsd_vec(a,b):
    n=max(len(a),len(b));x=np.zeros(n);y=np.zeros(n);x[:len(a)]=a;y[:len(b)]=b
    if x.sum()==0 or y.sum()==0:return 1.0
    x/=x.sum();y/=y.sum();m=(x+y)/2
    def kl(p,q):
        z=p>0;return float(np.sum(p[z]*np.log2(p[z]/q[z])))
    return .5*kl(x,m)+.5*kl(y,m)
def percentile_ranks(vals):
    order=np.argsort(vals,kind="mergesort");out=np.empty(len(vals),float)
    for rank,i in enumerate(order):out[i]=rank/max(1,len(vals)-1)
    return out

def parse_header(root):
    h=root.find("header")
    txt="".join(h.itertext()) if h is not None else ""
    md={}
    for line in txt.splitlines():
        if ":" in line:
            k,v=line.split(":",1);md[k.strip()]=v.strip()
    return md

def flags(md):
    region=md.get("language-region","").lower();area=md.get("language-area","").lower();med=md.get("medium","").lower()
    ms="handschrift" in med
    east="ostoberdeutsch" in region;west="westoberdeutsch" in region;north="nordoberdeutsch" in region
    bair=("bair" in area or "bayr" in area or "österreich" in area or "oesterreich" in area) and "alemann" not in area
    alem=("alemann" in area or "schwäb" in area or "elsäss" in area) and "bair" not in area and "bayr" not in area
    time=md.get("time","").strip()
    early=time in {"14,2","15,1"} or time.startswith("14,2") or time.startswith("15,1")
    return ms,east,west,north,bair,alem,early

class Acc:
    def __init__(self):
        self.N=0;self.docs=0
        self.cellcnt=collections.Counter();self.lemmacnt=collections.Counter();self.forms=collections.defaultdict(collections.Counter)
        self.starts=collections.Counter();self.ends=collections.Counter();self.selfn=collections.Counter()
        self.neigh=collections.defaultdict(collections.Counter);self.docset=collections.defaultdict(set)
    def add_doc(self,docid,lines):
        self.docs+=1
        for line in lines:
            for i,x in enumerate(line):
                c=x["cell"];lemma=c[0];self.N+=1;self.cellcnt[c]+=1;self.lemmacnt[lemma]+=1;self.forms[c][x["norm"]]+=1;self.docset[c].add(docid)
                if i==0:self.starts[c]+=1
                if i==len(line)-1:self.ends[c]+=1
                if i>0:self.neigh[c][line[i-1]["norm"]]+=1
                if i+1<len(line):
                    self.neigh[c][line[i+1]["norm"]]+=1
                    if line[i+1]["cell"]==c:self.selfn[c]+=1

panel_names=["EARLY_ALL_MS","EARLY_EAST_UPPER_MS","EARLY_BAVARIAN_STRICT_MS","EARLY_WEST_UPPER_MS","EARLY_ALEMANNIC_STRICT_MS",
             "FULL_ALL_MS","FULL_EAST_UPPER_MS","FULL_BAVARIAN_STRICT_MS","FULL_WEST_UPPER_MS","FULL_ALEMANNIC_STRICT_MS"]
A={p:Acc() for p in panel_names}
meta=collections.Counter()

print("REF_DOWNLOAD_BEGIN",flush=True)
rb=urllib.request.urlopen(REF_URL,timeout=360).read();print("REF_DOWNLOAD_BYTES",len(rb),flush=True)
tar=tarfile.open(fileobj=io.BytesIO(rb),mode="r:gz")
names=[n for n in tar.getnames() if n.lower().endswith(".xml") and ("/ref-mlu/" in n.lower() or "/ref-rub/" in n.lower())]
print("REF_MORPH_XML_FILES",len(names),flush=True)

for fi,n in enumerate(names):
    root=ET.fromstring(tar.extractfile(n).read())
    md=parse_header(root)
    ms,east,west,north,bair,alem,early=flags(md)
    docid=root.attrib.get("id",n)
    meta[(md.get("time",""),md.get("language-region",""),md.get("language-area",""),md.get("medium",""))]+=1
    # Map parent virtual token number -> valid annotated grammatical tokens.
    byv=collections.defaultdict(list)
    order=0
    for tok in root.findall("token"):
        mt=re.match(r"t(\d+)",tok.attrib.get("id",""));vi=int(mt.group(1)) if mt else None
        if vi is None:continue
        for an in tok.findall("tok_anno"):
            lemma_el=an.find("lemma");pos_el=an.find("pos")
            if lemma_el is None or pos_el is None:continue
            lemma=lemma_el.attrib.get("tag","").strip();pos=pos_el.attrib.get("tag","").strip()
            if not lemma or not pos or lemma in {"--","[!]"} or pos in {"--","FM","$_"}:continue
            infl_el=an.find("inflection")
            if infl_el is None:infl_el=an.find("infl")
            infl=infl_el.attrib.get("tag","--").strip() if infl_el is not None else "--"
            norm=(an.attrib.get("ascii") or an.attrib.get("utf") or an.attrib.get("trans") or "").lower()
            if not norm:continue
            byv[vi].append((order,{"cell":(lemma,pos,infl),"norm":norm}));order+=1
    lines=[]
    lay=root.find("layoutinfo")
    if lay is not None:
        for L in lay.findall("line"):
            rg=L.attrib.get("range","")
            nums=[int(x) for x in re.findall(r"t(\d+)_d\d+",rg)]
            if not nums:continue
            a,b=nums[0],nums[-1];q=[]
            for vi in range(a,b+1):q.extend(byv.get(vi,[]))
            if q:lines.append([x for _,x in sorted(q,key=lambda t:t[0])])
    if not lines:continue
    targets=[]
    if ms:
        targets.append("FULL_ALL_MS")
        if east:targets.append("FULL_EAST_UPPER_MS")
        if bair:targets.append("FULL_BAVARIAN_STRICT_MS")
        if west:targets.append("FULL_WEST_UPPER_MS")
        if alem:targets.append("FULL_ALEMANNIC_STRICT_MS")
        if early:
            targets.append("EARLY_ALL_MS")
            if east:targets.append("EARLY_EAST_UPPER_MS")
            if bair:targets.append("EARLY_BAVARIAN_STRICT_MS")
            if west:targets.append("EARLY_WEST_UPPER_MS")
            if alem:targets.append("EARLY_ALEMANNIC_STRICT_MS")
    for p in targets:A[p].add_doc(docid,lines)
    if fi%25==0:print("REF_PARSE",fi,n,flush=True)

print("REF_META_TOP",json.dumps(meta.most_common(30),ensure_ascii=False),flush=True)

NAMED=["dër","ein","sîn","haben","werden","in","ze","und","ich","ër","wir","dû"]
def analyse(label,a):
    N=a.N
    if N<5000:return {"status":"too_small","N":N,"docs":a.docs}
    rankmap={c:i+1 for i,(c,n) in enumerate(a.cellcnt.most_common())}
    bylemma=collections.defaultdict(list)
    for c,n in a.cellcnt.items():bylemma[c[0]].append((c,n))
    lprof={};lsep={}
    for lemma,arr in bylemma.items():
        arr.sort(key=lambda z:(-z[1],z[0]));tot=sum(n for _,n in arr);lprof[lemma]=[n/tot for _,n in arr]
        top=[c for c,n in arr[:8] if n>=3]
        pp=[jsd_counter(a.neigh[x],a.neigh[y]) for x,y in combinations(top,2)]
        lsep[lemma]=float(np.mean(pp)) if pp else 0.0
    mincount=max(20,int(N*.0001));cand=[]
    for c,n in a.cellcnt.items():
        if n<mincount:continue
        lemma,pos,infl=c;arr=bylemma[lemma];within=n/a.lemmacnt[lemma];prof=lprof[lemma]
        f={"cell":c,"count":n,"share":n/N,"rank":rankmap[c],"start":a.starts[c]/n,"end":a.ends[c]/n,
           "self_next":a.selfn[c]/n,"entropy":entropy_norm(a.neigh[c]),"simpson":simpson(a.neigh[c]),
           "lemma_share":a.lemmacnt[lemma]/N,"within_lemma_share":within,"lemma_cells":len(arr),
           "paradigm_sep":lsep[lemma],"profile":prof,"forms":a.forms[c].most_common(8),"docs":len(a.docset[c])}
        f["d_logfreq"]=abs(math.log((f["share"]+1e-12)/(VT["share"]+1e-12)))
        f["d_start"]=abs(f["start"]-VT["start"]);f["d_end"]=abs(f["end"]-VT["end"])
        f["d_self"]=abs(math.log((f["self_next"]+1e-4)/(VT["self_next"]+1e-4)))
        f["d_simpson"]=abs(f["simpson"]-VT["simpson"])
        f["d_cellshare1"]=abs(within-V1_WITHIN);f["d_profile1"]=jsd_vec(prof,V1_PROFILE);f["d_sep1"]=abs(f["paradigm_sep"]-V1_SEP)
        # diagnostic ED2 only
        f["d_cellshare2"]=abs(within-V2_WITHIN);f["d_profile2"]=jsd_vec(prof,V2_PROFILE);f["d_sep2"]=abs(f["paradigm_sep"]-V2_SEP)
        if len(arr)<2:
            f["d_cellshare1"]+=1;f["d_profile1"]+=1;f["d_sep1"]+=1
            f["d_cellshare2"]+=1;f["d_profile2"]+=1;f["d_sep2"]+=1
        cand.append(f)
    metrics=["d_logfreq","d_start","d_end","d_self","d_simpson","d_cellshare1","d_profile1","d_sep1"]
    weights={"d_logfreq":2.0,"d_start":0.5,"d_end":0.5,"d_self":1.0,"d_simpson":1.0,"d_cellshare1":1.0,"d_profile1":2.0,"d_sep1":2.0}
    for m in metrics:
        p=percentile_ranks(np.array([x[m] for x in cand]))
        for x,v in zip(cand,p):x["pct_"+m]=float(v)
    ws=sum(weights.values())
    for x in cand:x["score"]=sum(weights[m]*x["pct_"+m] for m in metrics)/ws
    cand.sort(key=lambda x:(x["score"],x["d_logfreq"],-x["count"]))
    def pack(x):
        keep=["cell","count","share","rank","start","end","self_next","entropy","simpson","lemma_share","within_lemma_share","lemma_cells","paradigm_sep","forms","docs","score","d_logfreq","d_profile1","d_sep1","d_profile2","d_sep2"]
        return {k:x[k] for k in keep}
    freq=sorted(cand,key=lambda x:x["d_logfreq"])[:25]
    named={}
    for lemma in NAMED:
        h=[x for x in cand if x["cell"][0]==lemma]
        if h:
            q=min(h,key=lambda x:x["score"]);named[lemma]={"rank":cand.index(q)+1,"candidate":pack(q)}
    return {"status":"ok","N":N,"docs":a.docs,"top_ranked":[pack(x) for x in cand[:40]],"frequency_only":[pack(x) for x in freq],
            "named_probe_ranks":named}

OUT={}
for p in panel_names:
    OUT[p]=analyse(p,A[p])
    print("DAIIN1_REF_PANEL",p,json.dumps(OUT[p] if OUT[p].get("status")!="ok" else {"N":OUT[p]["N"],"docs":OUT[p]["docs"],"top":OUT[p]["top_ranked"][:15],"freq":OUT[p]["frequency_only"][:15],"named":OUT[p]["named_probe_ranks"]},ensure_ascii=False,separators=(",",":")),flush=True)
print("DAIIN1_REF_RESULT_JSON="+json.dumps({"phase":"DAIIN_1_REF_VALIDATION","status":"complete","panels":OUT,
 "firewall":"Voynich target and ED1-only weights frozen before ReM discovery output; ReF not used for retuning."},ensure_ascii=False,separators=(",",":")),flush=True)
