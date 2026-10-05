#!/usr/bin/env python3
"""
DAIIN-1: blind historical-German paradigm search.

Discovery question:
Does the behavioural fingerprint of Voynich ZLZI 'daiin' resemble a high-frequency
Middle High German inflectional/grammatical cell, and does the same candidate survive
dialect stratification?

No candidate lemma is privileged in scoring.

Voynich:
- strict +P0 running text
- target exact token daiin
- ED1 and ED2 spelling neighbourhoods define two independent candidate paradigm proxies

German:
- ReM 2.1 Tabular JSON, full annotations
- candidate unit = (lemma_gen, pos_hits, infl) grammatical cell
- context neighbour identity = normalized wordform
- primary panels are manuscript texts; dialect panels are metadata-defined

Outputs:
- Voynich target fingerprint
- frequency-only closest German cells
- multi-feature blind ranking per panel
- named-probe ranks only after blind ranking
"""
import collections, hashlib, io, json, math, re, statistics, urllib.request, zipfile
from itertools import combinations

import numpy as np

SEED=20261005
V_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/92ec41cb26d233a388b6f65fa1a4b7c45d7ad8c5/voynich_transcriptions_slim.json"
V_SHA="26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f"
REM_URL="https://zenodo.org/api/records/13982324/files/ReM-v2.1_json.zip/content"
TARGET="daiin"

def lev(a,b):
    if a==b:return 0
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):
            cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]

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
    pa=np.array([a.get(k,0.0) for k in keys],float)
    pb=np.array([b.get(k,0.0) for k in keys],float)
    if pa.sum()==0 or pb.sum()==0:return 1.0
    pa/=pa.sum();pb/=pb.sum();m=(pa+pb)/2
    def kl(p,q):
        z=p>0
        return float(np.sum(p[z]*np.log2(p[z]/q[z])))
    return 0.5*kl(pa,m)+0.5*kl(pb,m)

def jsd_vec(a,b):
    n=max(len(a),len(b));x=np.zeros(n);y=np.zeros(n)
    x[:len(a)]=a;y[:len(b)]=b
    if x.sum()==0 or y.sum()==0:return 1.0
    x/=x.sum();y/=y.sum();m=(x+y)/2
    def kl(p,q):
        z=p>0
        return float(np.sum(p[z]*np.log2(p[z]/q[z])))
    return 0.5*kl(x,m)+0.5*kl(y,m)

def percentile_ranks(vals):
    # 0 = best/smallest difference
    order=np.argsort(vals,kind="mergesort")
    out=np.empty(len(vals),float)
    for rank,i in enumerate(order):out[i]=rank/max(1,len(vals)-1)
    return out

# ---------------- Voynich target ----------------
raw=urllib.request.urlopen(V_URL,timeout=120).read()
if hashlib.sha256(raw).hexdigest()!=V_SHA:raise RuntimeError("Voynich corpus SHA mismatch")
V=json.loads(raw)
vlines=[]
for fol,ld in V["pages"].items():
    for lid,rec in ld.items():
        if str(rec.get("u",""))!="+P0":continue
        txt=rec.get("t",{}).get("ZLZI","")
        toks=[x.lower() for x in txt.split() if re.fullmatch(r"[a-z]+",x.lower())]
        if toks:vlines.append((fol,str(lid),toks))
vfreq=collections.Counter(t for _,_,xs in vlines for t in xs)
VN=sum(vfreq.values())
if TARGET not in vfreq:raise RuntimeError("target absent")

def v_form_context(form):
    cnt=0;start=end=0;selfn=0;left=collections.Counter();right=collections.Counter()
    for _,_,xs in vlines:
        for i,t in enumerate(xs):
            if t!=form:continue
            cnt+=1
            if i==0:start+=1
            if i==len(xs)-1:end+=1
            if i>0:left[xs[i-1]]+=1
            if i+1<len(xs):
                right[xs[i+1]]+=1
                if xs[i+1]==form:selfn+=1
    both=left+right
    return dict(count=cnt,share=cnt/VN,start=start/cnt if cnt else 0,end=end/cnt if cnt else 0,
                self_next=selfn/cnt if cnt else 0,entropy=entropy_norm(both),simpson=simpson(both),
                left=left,right=right,both=both)

vt=v_form_context(TARGET)
vctx={t:v_form_context(t) for t in vfreq if vfreq[t]>=3}

def v_neigh(maxed):
    forms=[t for t,n in vfreq.items() if n>=3 and lev(t,TARGET)<=maxed]
    forms.sort(key=lambda t:(-vfreq[t],t))
    counts=np.array([vfreq[t] for t in forms],float)
    profile=(counts/counts.sum()).tolist()
    cellshare=vfreq[TARGET]/counts.sum()
    top=forms[:min(8,len(forms))]
    pairs=[]
    for a,b in combinations(top,2):
        pairs.append(jsd_counter(vctx[a]["both"],vctx[b]["both"]))
    sep=float(np.mean(pairs)) if pairs else 0.0
    return {"forms":forms,"counts":[vfreq[t] for t in forms],"profile":profile,
            "target_within_share":cellshare,"context_sep":sep}
VN1=v_neigh(1);VN2=v_neigh(2)
vprint={
 "strict_P0_tokens":VN,"target_count":vfreq[TARGET],"target_share":vt["share"],
 "line_start_rate":vt["start"],"line_end_rate":vt["end"],"self_next_rate":vt["self_next"],
 "neighbor_entropy_norm":vt["entropy"],"neighbor_simpson":vt["simpson"],
 "ED1":{k:v for k,v in VN1.items() if k!="profile"},
 "ED2":{k:v for k,v in VN2.items() if k!="profile"}
}
print("DAIIN1_VOYNICH="+json.dumps(vprint,ensure_ascii=False,separators=(",",":")),flush=True)

# ---------------- ReM ingestion ----------------
print("REM_DOWNLOAD_BEGIN",flush=True)
rb=urllib.request.urlopen(REM_URL,timeout=300).read()
print("REM_DOWNLOAD_BYTES",len(rb),flush=True)
z=zipfile.ZipFile(io.BytesIO(rb))
names=[n for n in z.namelist() if n.endswith(".json")]

def dialect_flags(md):
    region=str(md.get("language-region","")).lower()
    area=str(md.get("language-area","")).lower()
    typ=str(md.get("language-type","")).lower()
    med=str(md.get("medium","")).lower()
    ms="handschrift" in med
    east=region=="ostoberdeutsch" or "ostoberdeutsch" in region
    west=region=="westoberdeutsch" or "westoberdeutsch" in region
    north=region=="nordoberdeutsch" or "nordoberdeutsch" in region
    bair=("bair" in area or "bayr" in area or "österreich" in area or "oesterreich" in area) and "alemann" not in area
    alem=("alemann" in area or "schwäb" in area or "elsäss" in area) and "bair" not in area and "bayr" not in area
    return ms,east,west,north,bair,alem

# occurrence tuple: cell, norm, lemma, line_start, line_end, docid
docs=[]
meta_counts=collections.Counter(); line_span_fallback_docs=0
for n in names:
    d=json.loads(z.read(n))
    md=d.get("metadata",{})
    if str(md.get("language","")).lower()!="mhd":continue
    ms,east,west,north,bair,alem=dialect_flags(md)
    docid=str(md.get("id") or d.get("id") or n)
    meta_counts[(str(md.get("language-region","")),str(md.get("language-area","")),str(md.get("medium","")))] += 1
    # Build physical lines from ReM's explicit virtual-token begin/end spans.
    # Filtering punctuation/foreign material happens inside each raw line so
    # invalid material at a line edge cannot accidentally merge adjacent lines.
    rawt=[]
    for order,tok in enumerate(d.get("token",[])):
        vm=str(tok.get("virttok",""))
        mm=re.match(r"t(\d+)",vm)
        vi=int(mm.group(1)) if mm else None
        lemma=str(tok.get("lemma_gen","--"))
        pos=str(tok.get("pos_hits","--"))
        infl=str(tok.get("infl","--"))
        norm=str(tok.get("norm","--")).lower()
        valid=not (lemma in ("--","[!]","") or pos in ("--","$_","FM") or norm in ("--","[!]","") or tok.get("pos_upos")=="PUNCT")
        rawt.append((order,vi,{"cell":(lemma,pos,infl),"norm":norm} if valid else None))
    byvi=collections.defaultdict(list)
    for order,vi,x in rawt:
        if vi is not None and x is not None:byvi[vi].append((order,x))
    lines=[]
    for L in d.get("line",[]):
        try:a0=int(L["begin"]);b0=int(L["end"])
        except Exception:continue
        line=[]
        for vi in range(a0,b0+1):
            for order,x in byvi.get(vi,[]):line.append((order,x))
        line=[x for _,x in sorted(line,key=lambda q:q[0])]
        if line:lines.append(line)
    # Rare files without usable line spans: one-document sequence fallback.
    if not lines:
        line_span_fallback_docs+=1
        line=[x for _,_,x in rawt if x is not None]
        if line:lines=[line]
    if not lines:continue
    docs.append({"id":docid,"ms":ms,"east":east,"west":west,"north":north,"bair":bair,"alem":alem,
                 "region":str(md.get("language-region","")),"area":str(md.get("language-area","")),
                 "topic":str(md.get("topic","")),"lines":lines})

print("REM_DOCS",len(docs),"LINE_SPAN_FALLBACK_DOCS",line_span_fallback_docs,"META_TOP",json.dumps(meta_counts.most_common(20),ensure_ascii=False),flush=True)

PANELS={
 "ALL_MHG":lambda d:True,
 "ALL_MANUSCRIPT":lambda d:d["ms"],
 "EAST_UPPER_MS":lambda d:d["ms"] and d["east"],
 "BAVARIAN_STRICT_MS":lambda d:d["ms"] and d["bair"],
 "WEST_UPPER_MS":lambda d:d["ms"] and d["west"],
 "ALEMANNIC_STRICT_MS":lambda d:d["ms"] and d["alem"],
 "NORTH_UPPER_MS":lambda d:d["ms"] and d["north"],
}

NAMED=["dër","ein","sîn","haben","werden","in","ze","und","ich","ër","wir","dû","ër","ër/sie/ëz"]

def panel_analysis(label,pred):
    cellcnt=collections.Counter();lemmacnt=collections.Counter();forms=collections.defaultdict(collections.Counter)
    starts=collections.Counter();ends=collections.Counter();selfn=collections.Counter()
    neigh=collections.defaultdict(collections.Counter);docset=collections.defaultdict(set)
    N=0;ndocs=0
    for d in docs:
        if not pred(d):continue
        ndocs+=1
        for line in d["lines"]:
            for i,x in enumerate(line):
                c=x["cell"];lemma=c[0];N+=1;cellcnt[c]+=1;lemmacnt[lemma]+=1;forms[c][x["norm"]]+=1;docset[c].add(d["id"])
                if i==0:starts[c]+=1
                if i==len(line)-1:ends[c]+=1
                if i>0:neigh[c][line[i-1]["norm"]]+=1
                if i+1<len(line):
                    neigh[c][line[i+1]["norm"]]+=1
                    if line[i+1]["cell"]==c:selfn[c]+=1
    if N<5000:return {"status":"too_small","N":N,"docs":ndocs}
    ranked=[c for c,n in cellcnt.most_common()]
    rankmap={c:i+1 for i,c in enumerate(ranked)}
    # lemma cell profiles and context separation
    bylemma=collections.defaultdict(list)
    for c,n in cellcnt.items():bylemma[c[0]].append((c,n))
    lprof={};lsep={}
    for lemma,arr in bylemma.items():
        arr.sort(key=lambda x:(-x[1],x[0]))
        total=sum(n for _,n in arr)
        lprof[lemma]=[n/total for _,n in arr]
        top=[c for c,n in arr[:8] if n>=3]
        pp=[jsd_counter(neigh[a],neigh[b]) for a,b in combinations(top,2)]
        lsep[lemma]=float(np.mean(pp)) if pp else 0.0
    mincount=max(20,int(N*0.0001))
    cand=[]
    for c,n in cellcnt.items():
        lemma,pos,infl=c
        if n<mincount:continue
        # require at least 2 attested cells for paradigm metrics, but retain uninflected as fallback with penalty
        arr=bylemma[lemma]
        cellshare=n/lemmacnt[lemma]
        prof=lprof[lemma]
        f={
          "cell":c,"count":n,"share":n/N,"rank":rankmap[c],
          "start":starts[c]/n,"end":ends[c]/n,"self_next":selfn[c]/n,
          "entropy":entropy_norm(neigh[c]),"simpson":simpson(neigh[c]),
          "lemma_share":lemmacnt[lemma]/N,"within_lemma_share":cellshare,
          "lemma_cells":len(arr),"paradigm_sep":lsep[lemma],
          "profile":prof,"forms":forms[c].most_common(8),"docs":len(docset[c]),
        }
        # blind differences
        f["d_logfreq"]=abs(math.log((f["share"]+1e-12)/(vt["share"]+1e-12)))
        f["d_start"]=abs(f["start"]-vt["start"])
        f["d_end"]=abs(f["end"]-vt["end"])
        f["d_self"]=abs(math.log((f["self_next"]+1e-4)/(vt["self_next"]+1e-4)))
        f["d_simpson"]=abs(f["simpson"]-vt["simpson"])
        f["d_cellshare1"]=abs(f["within_lemma_share"]-VN1["target_within_share"])
        f["d_cellshare2"]=abs(f["within_lemma_share"]-VN2["target_within_share"])
        f["d_profile1"]=jsd_vec(f["profile"],VN1["profile"])
        f["d_profile2"]=jsd_vec(f["profile"],VN2["profile"])
        f["d_sep1"]=abs(f["paradigm_sep"]-VN1["context_sep"])
        f["d_sep2"]=abs(f["paradigm_sep"]-VN2["context_sep"])
        if len(arr)<2:
            # no paradigm: explicit penalty on paradigm dimensions
            f["d_cellshare1"]+=1;f["d_cellshare2"]+=1;f["d_profile1"]+=1;f["d_profile2"]+=1;f["d_sep1"]+=1;f["d_sep2"]+=1
        cand.append(f)
    if not cand:return {"status":"no_candidates","N":N,"docs":ndocs}

    # PRIMARY ranking uses only the ED1 Voynich neighbourhood.
    # ED2 is deliberately excluded from the score and retained solely as a
    # post-ranking robustness diagnostic because ED2 is much broader.
    metrics=["d_logfreq","d_start","d_end","d_self","d_simpson","d_cellshare1","d_profile1","d_sep1"]
    weights={"d_logfreq":2.0,"d_start":0.5,"d_end":0.5,"d_self":1.0,"d_simpson":1.0,
             "d_cellshare1":1.0,"d_profile1":2.0,"d_sep1":2.0}
    for m in metrics:
        pr=percentile_ranks(np.array([x[m] for x in cand],float))
        for x,p in zip(cand,pr):x["pct_"+m]=float(p)
    wsum=sum(weights.values())
    for x in cand:x["score"]=sum(weights[m]*x["pct_"+m] for m in metrics)/wsum
    cand.sort(key=lambda x:(x["score"],x["d_logfreq"],-x["count"]))

    # frequency-only nearest
    fq=sorted(cand,key=lambda x:(x["d_logfreq"],abs(x["rank"]-1)))[:25]
    top=cand[:40]
    named={}
    for lemma in NAMED:
        hits=[x for x in cand if x["cell"][0]==lemma]
        if hits:
            best=min(hits,key=lambda x:x["score"]);named[lemma]={"rank":cand.index(best)+1,"candidate":best}

    def pack(x):
        keep=["cell","count","share","rank","start","end","self_next","entropy","simpson","lemma_share","within_lemma_share",
              "lemma_cells","paradigm_sep","forms","docs","score","d_logfreq","d_profile1","d_profile2","d_sep1","d_sep2"]
        return {k:x[k] for k in keep}
    return {"status":"ok","N":N,"docs":ndocs,"min_candidate_count":mincount,
            "top_ranked":[pack(x) for x in top],"frequency_only":[pack(x) for x in fq],
            "named_probe_ranks":{k:{"rank":v["rank"],"candidate":pack(v["candidate"])} for k,v in named.items()}}

OUT={}
for label,pred in PANELS.items():
    print("DAIIN1_PANEL_BEGIN",label,flush=True)
    OUT[label]=panel_analysis(label,pred)
    if OUT[label].get("status")=="ok":

# stability: which lemma+POS families recur in top20 across manuscript dialect panels?
stable=collections.Counter()
stable_detail=collections.defaultdict(list)
for label in ["ALL_MANUSCRIPT","EAST_UPPER_MS","BAVARIAN_STRICT_MS","WEST_UPPER_MS","ALEMANNIC_STRICT_MS","NORTH_UPPER_MS"]:
    p=OUT.get(label,{})
    if p.get("status")!="ok":continue
    for rank,x in enumerate(p["top_ranked"][:20],1):
        key=(x["cell"][0],x["cell"][1])
        stable[key]+=1;stable_detail[key].append((label,rank,x["cell"][2],x["share"],x["score"]))
stab=[{"lemma":k[0],"pos":k[1],"panels_top20":n,"detail":stable_detail[k]} for k,n in stable.most_common()]
res={"phase":"DAIIN_1_REM","status":"complete","design":"blind grammatical-cell ranking",
     "voynich":vprint,"panels":OUT,"stability":stab[:100],
     "notes":["Candidate cell=(lemma_gen,pos_hits,inflection).","No German lemma is privileged in scoring.",
              "ED1 is the sole paradigm neighbourhood used in primary ranking; ED2 is reported only as secondary robustness.",
              "ReM is discovery; ReF 1350-1650 is intended as temporal replication, not for retuning."]}

for label,p in OUT.items():
    if p.get("status")!="ok":
        print("DAIIN1_COMPACT",label,json.dumps(p,ensure_ascii=False,separators=(",",":")),flush=True);continue
    compact={
      "N":p["N"],"docs":p["docs"],
      "top":[{"rank":i+1,"cell":x["cell"],"score":x["score"],"share":x["share"],"within":x["within_lemma_share"],"sep":x["paradigm_sep"],"forms":x["forms"]} for i,x in enumerate(p["top_ranked"][:15])],
      "freq":[{"freq_rank":i+1,"cell":x["cell"],"share":x["share"],"score":x["score"],"start":x["start"],"end":x["end"],"within":x["within_lemma_share"],"sep":x["paradigm_sep"],"forms":x["forms"]} for i,x in enumerate(p["frequency_only"][:20])],
      "named":{k:{"rank":v["rank"],"cell":v["candidate"]["cell"],"share":v["candidate"]["share"],"score":v["candidate"]["score"]} for k,v in p["named_probe_ranks"].items()}
    }
    print("DAIIN1_COMPACT",label,json.dumps(compact,ensure_ascii=False,separators=(",",":")),flush=True)
print("DAIIN1_STABILITY="+json.dumps(stab[:50],ensure_ascii=False,separators=(",",":")),flush=True)

