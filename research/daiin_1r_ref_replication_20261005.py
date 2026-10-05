#!/usr/bin/env python3
"""
DAIIN-1R: frozen-weight temporal replication in ReF 1.0.2 (1350–1650).

This script is written after seeing ReM results, so it does NOT tune weights,
candidate classes, or the Voynich target. It applies the DAIIN-1 ReM metric
unchanged to ReF.RUB/MLU CorA-XML cells that carry lemma+POS+morphology.

Primary replication question:
Does a cell whose dominant diplomatic/modernized form is WAS (past 3sg of
'wesen/sein') remain near the top of the blind ranking?
"""
import collections, io, json, math, re, tarfile, urllib.request
import xml.etree.ElementTree as ET
import numpy as np

REF_URL="https://zenodo.org/api/records/5793616/files/ReF-v1.0.2.tar.gz/content"

# Frozen from strict +P0 ZLZI DAIIN-1.
VT={"share":0.02399870277282309,"start":0.20270270270270271,
    "end":0.1608108108108108,"self_next":0.013513513513513514,
    "simpson":0.005339848525864955}
VN1={"profile":[740,416,191,111,75,58,55,43,19,17,16,15,12,7,6,5,4,3,3,3],
     "target_within_share":0.4113396331295164,"context_sep":0.6871803110693901}
VN2={"profile":[740,416,199,191,133,111,97,77,75,58,56,55,46,45,43,43,40,40,37,37,33,28,27,21,19,19,19,17,17,16,16,15,14,13,12,12,12,11,9,8,7,7,6,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
     "target_within_share":0.24478994376447238,"context_sep":0.6361092810438288}
for d in (VN1,VN2):
    a=np.array(d["profile"],float);d["profile"]=(a/a.sum()).tolist()

def simpson(c):
    n=sum(c.values())
    if n<=0:return 1.0
    p=np.array(list(c.values()),float)/n
    return float((p*p).sum())

def jsd_counter(a,b):
    keys=set(a)|set(b)
    if not keys:return 0.0
    pa=np.array([a.get(k,0.) for k in keys],float); pb=np.array([b.get(k,0.) for k in keys],float)
    if pa.sum()==0 or pb.sum()==0:return 1.0
    pa/=pa.sum();pb/=pb.sum();m=(pa+pb)/2
    def kl(p,q):
        z=p>0
        return float(np.sum(p[z]*np.log2(p[z]/q[z])))
    return .5*kl(pa,m)+.5*kl(pb,m)

def jsd_vec(a,b):
    n=max(len(a),len(b));x=np.zeros(n);y=np.zeros(n)
    x[:len(a)]=a;y[:len(b)]=b
    if x.sum()==0 or y.sum()==0:return 1.0
    x/=x.sum();y/=y.sum();m=(x+y)/2
    def kl(p,q):
        z=p>0
        return float(np.sum(p[z]*np.log2(p[z]/q[z])))
    return .5*kl(x,m)+.5*kl(y,m)

def percentile_ranks(vals):
    order=np.argsort(vals,kind="mergesort"); out=np.empty(len(vals),float)
    for r,i in enumerate(order):out[i]=r/max(1,len(vals)-1)
    return out

def child_tag(el,name):
    for ch in el:
        if ch.tag.split("}")[-1]==name:
            return ch.attrib.get("tag","")
    return ""

def range_ends(s):
    if not s:return (None,None)
    if ".." in s:
        a,b=s.split("..",1);return a,b
    return s,s

print("REF_DOWNLOAD_BEGIN",flush=True)
raw=urllib.request.urlopen(REF_URL,timeout=300).read()
print("REF_DOWNLOAD_BYTES",len(raw),flush=True)
tar=tarfile.open(fileobj=io.BytesIO(raw),mode="r:gz")
names=[n for n in tar.getnames() if n.endswith(".xml")]
print("REF_XML_FILES",len(names),flush=True)

cellcnt=collections.Counter(); lemmacnt=collections.Counter(); forms=collections.defaultdict(collections.Counter)
starts=collections.Counter(); ends=collections.Counter(); selfn=collections.Counter()
neigh=collections.defaultdict(collections.Counter); docset=collections.defaultdict(set)
N=0; ndocs=0; skipped=collections.Counter(); header_samples=[]

for fi,name in enumerate(names):
    try:
        root=ET.fromstring(tar.extractfile(name).read())
    except Exception:
        skipped["xml_parse"]+=1;continue
    # Only documents with usable lemma layers contribute.
    # Build diplomatic line boundary IDs.
    line_start_d=set();line_end_d=set()
    for el in root.iter():
        if el.tag.split("}")[-1]=="line":
            a,b=range_ends(el.attrib.get("range",""))
            if a:line_start_d.add(a)
            if b:line_end_d.add(b)
    # header sample for later geographic/date audit, not used in score
    if len(header_samples)<30:
        hs=[]
        for el in root.iter():
            if el.tag.split("}")[-1] in ("header","cora-header"):
                hs.append((el.text or "")[:1000])
                if el.attrib:hs.append(json.dumps(el.attrib,ensure_ascii=False))
        header_samples.append({"file":name,"header":"\n".join(hs)[:1500]})
    lines=[];cur=[]
    for tok in root.iter():
        if tok.tag.split("}")[-1]!="token":continue
        dipls=[x for x in tok if x.tag.split("}")[-1] in ("dipl","tok_dipl")]
        mods=[x for x in tok if x.tag.split("}")[-1] in ("mod","tok_anno")]
        if not mods:continue
        first_d=dipls[0].attrib.get("id") if dipls else None
        last_d=dipls[-1].attrib.get("id") if dipls else None
        for mi,mod in enumerate(mods):
            lemma=child_tag(mod,"lemma")
            pos=child_tag(mod,"pos")
            infl=child_tag(mod,"inflection") or child_tag(mod,"morph")
            if not infl:
                # Some CorA exports concatenate morphology into a dedicated tag
                infl=child_tag(mod,"infl")
            norm=(mod.attrib.get("ascii") or mod.attrib.get("utf") or mod.attrib.get("trans") or "").strip().lower()
            if not lemma or lemma in ("--","[!]") or not pos or not norm:
                continue
            # punctuation / foreign-style nonlexical cells
            if pos.startswith("$"):continue
            cell=(lemma,pos,infl or "--")
            x={"cell":cell,"norm":norm,
               "start":bool(mi==0 and first_d in line_start_d),
               "end":bool(mi==len(mods)-1 and last_d in line_end_d)}
            if x["start"] and cur:
                lines.append(cur);cur=[]
            cur.append(x)
            if x["end"]:
                lines.append(cur);cur=[]
    if cur:lines.append(cur)
    if not lines:
        skipped["no_annotated_lines"]+=1;continue
    ndocs+=1
    docid=name
    for line in lines:
        for i,x in enumerate(line):
            c=x["cell"];lemma=c[0];N+=1;cellcnt[c]+=1;lemmacnt[lemma]+=1;forms[c][x["norm"]]+=1;docset[c].add(docid)
            if i==0:starts[c]+=1
            if i==len(line)-1:ends[c]+=1
            if i>0:neigh[c][line[i-1]["norm"]]+=1
            if i+1<len(line):
                neigh[c][line[i+1]["norm"]]+=1
                if line[i+1]["cell"]==c:selfn[c]+=1
    if (fi+1)%100==0:print("REF_PARSE_PROGRESS",fi+1,"N",N,"docs",ndocs,flush=True)

print("REF_CORPUS",json.dumps({"N":N,"docs":ndocs,"skipped":skipped,"cells":len(cellcnt),"lemmas":len(lemmacnt)}),flush=True)
print("REF_HEADER_SAMPLES="+json.dumps(header_samples[:8],ensure_ascii=False,separators=(",",":")),flush=True)
if N<10000:raise RuntimeError("Too few annotated ReF tokens parsed")

bylemma=collections.defaultdict(list)
for c,n in cellcnt.items():bylemma[c[0]].append((c,n))
lprof={};lsep={}
for lemma,arr in bylemma.items():
    arr.sort(key=lambda z:(-z[1],z[0]));tot=sum(n for _,n in arr)
    lprof[lemma]=[n/tot for _,n in arr]
    top=[c for c,n in arr[:8] if n>=3]
    pp=[jsd_counter(neigh[a],neigh[b]) for i,a in enumerate(top) for b in top[i+1:]]
    lsep[lemma]=float(np.mean(pp)) if pp else 0.0

ranked=[c for c,n in cellcnt.most_common()];rankmap={c:i+1 for i,c in enumerate(ranked)}
mincount=max(20,int(N*.0001));cand=[]
for c,n in cellcnt.items():
    if n<mincount:continue
    lemma,pos,infl=c;arr=bylemma[lemma];prof=lprof[lemma]
    f={"cell":c,"count":n,"share":n/N,"rank":rankmap[c],
       "start":starts[c]/n,"end":ends[c]/n,"self_next":selfn[c]/n,
       "simpson":simpson(neigh[c]),"lemma_share":lemmacnt[lemma]/N,
       "within_lemma_share":n/lemmacnt[lemma],"lemma_cells":len(arr),
       "paradigm_sep":lsep[lemma],"profile":prof,"forms":forms[c].most_common(10),"docs":len(docset[c])}
    f["d_logfreq"]=abs(math.log((f["share"]+1e-12)/(VT["share"]+1e-12)))
    f["d_start"]=abs(f["start"]-VT["start"]);f["d_end"]=abs(f["end"]-VT["end"])
    f["d_self"]=abs(math.log((f["self_next"]+1e-4)/(VT["self_next"]+1e-4)))
    f["d_simpson"]=abs(f["simpson"]-VT["simpson"])
    for k,vn in [(1,VN1),(2,VN2)]:
        f[f"d_cellshare{k}"]=abs(f["within_lemma_share"]-vn["target_within_share"])
        f[f"d_profile{k}"]=jsd_vec(prof,vn["profile"])
        f[f"d_sep{k}"]=abs(f["paradigm_sep"]-vn["context_sep"])
    if len(arr)<2:
        for k in (1,2):
            f[f"d_cellshare{k}"]+=1;f[f"d_profile{k}"]+=1;f[f"d_sep{k}"]+=1
    cand.append(f)

metrics=["d_logfreq","d_start","d_end","d_self","d_simpson","d_cellshare1","d_cellshare2","d_profile1","d_profile2","d_sep1","d_sep2"]
weights={"d_logfreq":2.0,"d_start":0.5,"d_end":0.5,"d_self":1.0,"d_simpson":1.0,
         "d_cellshare1":0.75,"d_cellshare2":0.75,"d_profile1":1.25,"d_profile2":1.25,"d_sep1":1.0,"d_sep2":1.0}
for m in metrics:
    pr=percentile_ranks(np.array([x[m] for x in cand],float))
    for x,p in zip(cand,pr):x["pct_"+m]=float(p)
ws=sum(weights.values())
for x in cand:x["score"]=sum(weights[m]*x["pct_"+m] for m in metrics)/ws
cand.sort(key=lambda x:(x["score"],x["d_logfreq"],-x["count"]))

def pack(x):
    ks=["cell","count","share","rank","start","end","self_next","simpson","lemma_share","within_lemma_share","lemma_cells","paradigm_sep","forms","docs","score","d_logfreq","d_profile1","d_profile2","d_sep1","d_sep2"]
    return {k:x[k] for k in ks}

# Blind top + locate surface was/ist/ein/er and common grammatical probes AFTER ranking.
top=[pack(x) for x in cand[:50]]
surface_probes={}
for surf in ["was","ist","ein","er","der","ich","hat","hât","sind","sint"]:
    hits=[]
    for i,x in enumerate(cand):
        if any(form==surf for form,n in x["forms"]):
            hits.append((i+1,x))
    if hits:
        surface_probes[surf]={"best_rank":hits[0][0],"candidate":pack(hits[0][1])}
lemma_probes={}
for q in ["sein","sîn","wesen","wësen","ein","èin","der","dër","haben","ich","er","ër"]:
    hits=[(i+1,x) for i,x in enumerate(cand) if x["cell"][0]==q]
    if hits:lemma_probes[q]={"best_rank":hits[0][0],"candidate":pack(hits[0][1])}

fq=sorted(cand,key=lambda x:x["d_logfreq"])[:30]
out={"phase":"DAIIN_1R_REF","status":"complete","frozen_from":"DAIIN-1 ReM before ReF inspection",
     "corpus":"ReF 1.0.2 1350-1650","N":N,"docs":ndocs,"min_candidate_count":mincount,
     "top_ranked":top,"frequency_only":[pack(x) for x in fq],
     "surface_probe_ranks":surface_probes,"lemma_probe_ranks":lemma_probes}
print("DAIIN1R_TOP="+json.dumps(top[:20],ensure_ascii=False,separators=(",",":")),flush=True)
print("DAIIN1R_PROBES="+json.dumps({"surface":surface_probes,"lemma":lemma_probes},ensure_ascii=False,separators=(",",":")),flush=True)
print("DAIIN1R_RESULT_JSON="+json.dumps(out,ensure_ascii=False,separators=(",",":")),flush=True)
