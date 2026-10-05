#!/usr/bin/env python3
"""DAIIN-1C hostile control: is the 'wësen/was' hit specific to daiin?
Frozen DAIIN-1 scoring, applied to top strict +P0 ZLZI tokens.
Panels: strict Bavarian manuscripts and strict Alemannic manuscripts.
"""
import collections, hashlib, io, json, math, re, urllib.request, zipfile
from itertools import combinations
import numpy as np

SEED=20261005
V_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/92ec41cb26d233a388b6f65fa1a4b7c45d7ad8c5/voynich_transcriptions_slim.json"
V_SHA="26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f"
REM_URL="https://zenodo.org/api/records/13982324/files/ReM-v2.1_json.zip/content"
WAS_CELL=("wësen","VAFIN","Ind.Past.Sg.3")
IST_CELL=("sîn","VAFIN","Ind.Pres.Sg.3")

def lev(a,b):
    if a==b:return 0
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]

def simpson(c):
    n=sum(c.values())
    if n<=0:return 1.
    p=np.array(list(c.values()),float)/n
    return float((p*p).sum())

def jsd_counter(a,b):
    keys=set(a)|set(b)
    if not keys:return 0.
    pa=np.array([a.get(k,0.) for k in keys],float);pb=np.array([b.get(k,0.) for k in keys],float)
    if pa.sum()==0 or pb.sum()==0:return 1.
    pa/=pa.sum();pb/=pb.sum();m=(pa+pb)/2
    def kl(p,q):
        z=p>0;return float(np.sum(p[z]*np.log2(p[z]/q[z])))
    return .5*kl(pa,m)+.5*kl(pb,m)

def jsd_vec(a,b):
    n=max(len(a),len(b));x=np.zeros(n);y=np.zeros(n);x[:len(a)]=a;y[:len(b)]=b
    if x.sum()==0 or y.sum()==0:return 1.
    x/=x.sum();y/=y.sum();m=(x+y)/2
    def kl(p,q):
        z=p>0;return float(np.sum(p[z]*np.log2(p[z]/q[z])))
    return .5*kl(x,m)+.5*kl(y,m)

def pct(vals):
    o=np.argsort(vals,kind="mergesort");r=np.empty(len(vals),float)
    for k,i in enumerate(o):r[i]=k/max(1,len(vals)-1)
    return r

# Voynich
raw=urllib.request.urlopen(V_URL,timeout=120).read()
if hashlib.sha256(raw).hexdigest()!=V_SHA:raise RuntimeError("sha")
V=json.loads(raw);vlines=[]
for fol,ld in V["pages"].items():
    for lid,rec in ld.items():
        if str(rec.get("u",""))!="+P0":continue
        xs=[x.lower() for x in rec.get("t",{}).get("ZLZI","").split() if re.fullmatch(r"[a-z]+",x.lower())]
        if xs:vlines.append(xs)
vf=collections.Counter(t for xs in vlines for t in xs);VN=sum(vf.values())
TARGETS=[t for t,n in vf.most_common(20)]

def target_fp(t):
    left=collections.Counter();right=collections.Counter();cnt=st=en=selfn=0
    contexts={}
    def ctx(form):
        if form in contexts:return contexts[form]
        c=collections.Counter();n=0
        for xs in vlines:
            for i,z in enumerate(xs):
                if z!=form:continue
                n+=1
                if i:c[xs[i-1]]+=1
                if i+1<len(xs):c[xs[i+1]]+=1
        contexts[form]=c;return c
    for xs in vlines:
        for i,z in enumerate(xs):
            if z!=t:continue
            cnt+=1;st+=i==0;en+=i==len(xs)-1
            if i:left[xs[i-1]]+=1
            if i+1<len(xs):
                right[xs[i+1]]+=1
                selfn+=xs[i+1]==t
    def ng(d):
        fs=[x for x,n in vf.items() if n>=3 and lev(x,t)<=d]
        fs.sort(key=lambda x:(-vf[x],x));cs=np.array([vf[x] for x in fs],float);prof=(cs/cs.sum()).tolist()
        top=fs[:min(8,len(fs))]
        pp=[jsd_counter(ctx(a),ctx(b)) for a,b in combinations(top,2)]
        return {"profile":prof,"within":vf[t]/cs.sum(),"sep":float(np.mean(pp)) if pp else 0.}
    return {"share":cnt/VN,"start":st/cnt,"end":en/cnt,"self":selfn/cnt,"simpson":simpson(left+right),"n1":ng(1),"n2":ng(2)}

VFP={t:target_fp(t) for t in TARGETS}
print("CONTROL_TARGETS",json.dumps([(t,vf[t],VFP[t]["share"]) for t in TARGETS]),flush=True)

# ReM parse once
rb=urllib.request.urlopen(REM_URL,timeout=300).read();z=zipfile.ZipFile(io.BytesIO(rb))
docs=[]
for name in [n for n in z.namelist() if n.endswith(".json")]:
    d=json.loads(z.read(name));md=d.get("metadata",{})
    if str(md.get("language","")).lower()!="mhd" or "handschrift" not in str(md.get("medium","")).lower():continue
    reg=str(md.get("language-region","")).lower();area=str(md.get("language-area","")).lower()
    bav=(("bair" in area or "bayr" in area or "österreich" in area or "oesterreich" in area) and "alemann" not in area)
    alem=(("alemann" in area or "schwäb" in area or "elsäss" in area) and "bair" not in area and "bayr" not in area)
    if not (bav or alem):continue
    rawt=[]
    for order,x in enumerate(d.get("token",[])):
        m=re.match(r"t(\d+)",str(x.get("virttok","")));vi=int(m.group(1)) if m else None
        le=str(x.get("lemma_gen","--"));po=str(x.get("pos_hits","--"));inf=str(x.get("infl","--"));norm=str(x.get("norm","--")).lower()
        valid=not (le in ("--","[!]","") or po in ("--","$_","FM") or norm in ("--","[!]","") or x.get("pos_upos")=="PUNCT")
        rawt.append((order,vi,{"cell":(le,po,inf),"norm":norm} if valid else None))
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
    if not lines:
        line=[x for _,_,x in rawt if x is not None]
        if line:lines=[line]
    docs.append((bav,alem,str(md.get("id",name)),lines))

def german_features(which):
    cellcnt=collections.Counter();lemmacnt=collections.Counter();forms=collections.defaultdict(collections.Counter)
    starts=collections.Counter();ends=collections.Counter();selfn=collections.Counter();neigh=collections.defaultdict(collections.Counter);N=0
    for bav,alem,docid,lines in docs:
        if (which=="BAV" and not bav) or (which=="ALEM" and not alem):continue
        for line in lines:
            for i,x in enumerate(line):
                c=x["cell"];N+=1;cellcnt[c]+=1;lemmacnt[c[0]]+=1;forms[c][x["norm"]]+=1
                if i==0:starts[c]+=1
                if i==len(line)-1:ends[c]+=1
                if i:neigh[c][line[i-1]["norm"]]+=1
                if i+1<len(line):
                    neigh[c][line[i+1]["norm"]]+=1
                    if line[i+1]["cell"]==c:selfn[c]+=1
    by=collections.defaultdict(list)
    for c,n in cellcnt.items():by[c[0]].append((c,n))
    prof={};sep={}
    for le,a in by.items():
        a.sort(key=lambda z:(-z[1],z[0]));tot=sum(n for _,n in a);prof[le]=[n/tot for _,n in a]
        top=[c for c,n in a[:8] if n>=3];pp=[jsd_counter(neigh[x],neigh[y]) for i,x in enumerate(top) for y in top[i+1:]]
        sep[le]=float(np.mean(pp)) if pp else 0.
    minc=max(20,int(N*.0001));rows=[]
    for c,n in cellcnt.items():
        if n<minc:continue
        le=c[0];a=by[le]
        rows.append({"cell":c,"share":n/N,"start":starts[c]/n,"end":ends[c]/n,"self":selfn[c]/n,"simpson":simpson(neigh[c]),
                     "within":n/lemmacnt[le],"profile":prof[le],"sep":sep[le],"cells":len(a),"forms":forms[c].most_common(4)})
    return N,rows

MET=["df","ds","de","dself","dsp","dc1","dp1","dx1"]
W={"df":2.,"ds":.5,"de":.5,"dself":1.,"dsp":1.,"dc1":1.,"dp1":2.,"dx1":2.}
def score(rows,v):
    rr=[]
    for g in rows:
        x=dict(g)
        x["df"]=abs(math.log((g["share"]+1e-12)/(v["share"]+1e-12)));x["ds"]=abs(g["start"]-v["start"]);x["de"]=abs(g["end"]-v["end"])
        x["dself"]=abs(math.log((g["self"]+1e-4)/(v["self"]+1e-4)));x["dsp"]=abs(g["simpson"]-v["simpson"])
        for k,n in [(1,v["n1"]),(2,v["n2"])]:
            x[f"dc{k}"]=abs(g["within"]-n["within"]);x[f"dp{k}"]=jsd_vec(g["profile"],n["profile"]);x[f"dx{k}"]=abs(g["sep"]-n["sep"])
            if g["cells"]<2:x[f"dc{k}"]+=1;x[f"dp{k}"]+=1;x[f"dx{k}"]+=1
        rr.append(x)
    for m in MET:
        q=pct(np.array([x[m] for x in rr]))
        for x,p in zip(rr,q):x["p"+m]=p
    ws=sum(W.values())
    for x in rr:x["score"]=sum(W[m]*x["p"+m] for m in MET)/ws
    rr.sort(key=lambda x:x["score"])
    return rr

OUT={}
for panel in ("BAV","ALEM"):
    N,grows=german_features(panel);po={}
    print("CONTROL_PANEL",panel,"N",N,"CAND",len(grows),flush=True)
    for t in TARGETS:
        rr=score(grows,VFP[t])
        rankwas=next((i+1 for i,x in enumerate(rr) if tuple(x["cell"])==WAS_CELL),None)
        rankist=next((i+1 for i,x in enumerate(rr) if tuple(x["cell"])==IST_CELL),None)
        po[t]={"rank_was":rankwas,"rank_ist":rankist,"top":[{"cell":x["cell"],"score":x["score"]} for x in rr[:5]]}
        print("CONTROL_RESULT",panel,t,json.dumps(po[t],ensure_ascii=False,separators=(",",":")),flush=True)
    OUT[panel]={"N":N,"targets":po}

# Cross-panel specificity of WAS.
spec=[]
for t in TARGETS:
    rb=OUT["BAV"]["targets"][t]["rank_was"];ra=OUT["ALEM"]["targets"][t]["rank_was"]
    spec.append((max(rb or 999999,ra or 999999), (rb or 999999)+(ra or 999999), t, rb, ra))
spec.sort()
res={"phase":"DAIIN_1C","status":"complete","targets":TARGETS,"was_specificity_order":[{"target":t,"bav_rank":rb,"alem_rank":ra} for _,__,t,rb,ra in spec],"panels":OUT}
print("DAIIN1C_SUMMARY="+json.dumps(res["was_specificity_order"],ensure_ascii=False,separators=(",",":")),flush=True)
print("DAIIN1C_RESULT_JSON="+json.dumps(res,ensure_ascii=False,separators=(",",":")),flush=True)
