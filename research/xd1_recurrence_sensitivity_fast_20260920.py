#!/usr/bin/env python3
"""Independent vectorized XD1 P5 permutation implementation for sensitivity checks."""
import json, re, hashlib, urllib.request, zipfile, unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

SEED=20260920; NPERM=200; BANDS=((1,1),(2,5),(6,16),(17,64))
NUR_RECORD="https://zenodo.org/api/records/13881575"
EXPECTED_NUR_SHA="59e5264acb4546477567e78c8b3d444c472f1a0a5256ee0ee7d0407a70904652"

def band_rate_one(ids,lo,hi):
    n=len(ids)
    if n<=lo:return None
    hit=np.zeros(n-lo,dtype=bool)
    for d in range(lo,hi+1):
        if d>=n:break
        hit[d-lo:] |= (ids[d:]==ids[:-d])
    return float(hit.mean())

def perm_rates_batch(ids,lo,hi,rng,nperm=NPERM):
    n=len(ids)
    if n<=lo:return None
    q=np.empty((nperm,n),dtype=ids.dtype)
    for k in range(nperm): q[k]=rng.permutation(ids)
    hit=np.zeros((nperm,n-lo),dtype=bool)
    for d in range(lo,hi+1):
        if d>=n:break
        hit[:,d-lo:] |= (q[:,d:]==q[:,:-d])
    return hit.mean(axis=1)

def p5_fast(seqs,seed=SEED,nperm=NPERM):
    encoded={}
    for b,s in seqs.items():
        if not s:continue
        mp={t:i for i,t in enumerate(sorted(set(s)))}
        encoded[b]=np.asarray([mp[t] for t in s],dtype=np.int32)
    rng=np.random.default_rng(seed); out=[]
    for lo,hi in BANDS:
        eligible={b:a for b,a in encoded.items() if len(a)>lo}
        actual={b:band_rate_one(a,lo,hi) for b,a in eligible.items()}
        sums=np.zeros(nperm,float); pos=neg=0
        for b,a in eligible.items():
            rr=perm_rates_batch(a,lo,hi,rng,nperm)
            sums+=rr
            nm=float(rr.mean())
            pos += actual[b]>nm; neg += actual[b]<nm
        reps=sums/len(eligible)
        am=float(np.mean(list(actual.values()))); nm=float(reps.mean()); ns=float(reps.std(ddof=1)); eff=am-nm
        out.append(dict(contrast=f"LAG_{lo}_{hi}",n_blocks=len(eligible),nperm=nperm,observed=am,
                        null_mean=nm,null_sd=ns,effect=eff,effect_over_null_sd=abs(eff)/ns,
                        positive_blocks=pos,negative_blocks=neg))
    return out

def local(t):return t.split("}")[-1]
def tokenize(text):
    text=unicodedata.normalize("NFC",text).replace("\n"," ");out=[];buf=[]
    for ch in text:
        cat=unicodedata.category(ch)
        if cat[0] in ("L","M") or cat=="Nd":buf.append(ch.lower())
        else:
            if buf:out.append("".join(buf));buf=[]
    if buf:out.append("".join(buf))
    return [t for t in out if any(unicodedata.category(ch).startswith("L") for ch in t)]
def render(el,drop_ex=False):
    def rec(x):
        if drop_ex and local(x.tag)=="ex":return x.tail or ""
        s=x.text or ""
        for ch in x:s+=rec(ch)
        return s+(x.tail or "")
    s=el.text or ""
    for ch in el:s+=rec(ch)
    return s
def baseline_y(line):
    bl=next((x for x in line if local(x.tag)=="Baseline"),None)
    if bl is None:return None
    ys=[]
    for p in bl.attrib.get("points","").split():
        try:ys.append(float(p.split(",")[1]))
        except:pass
    return sum(ys)/len(ys) if ys else None

def vms_paragraphs(root):
    obj=json.load(open(Path(root)/"research"/"xd1_vms_paragraphs_20260920.json",encoding="utf-8"))
    return {p["paragraph_id"]:[t for t in p["text"].lower().split() if re.fullmatch(r"[a-z]+",t)]
            for p in obj["paragraphs"] if p["text"].strip()}

def nuremberg(tmp="/tmp/nuremberg_labels.zip"):
    meta=json.load(urllib.request.urlopen(NUR_RECORD));f=next(x for x in meta["files"] if x["key"]=="labels.zip")
    urllib.request.urlretrieve(f["links"]["self"],tmp)
    sha=hashlib.sha256(Path(tmp).read_bytes()).hexdigest()
    if sha!=EXPECTED_NUR_SHA:raise RuntimeError(sha)
    un,ex={},{}
    with zipfile.ZipFile(tmp) as z:
        for fn in sorted(n for n in z.namelist() if "/diplomatic-regularised/" in n and n.endswith(".xml")):
            root=ET.fromstring(z.read(fn));lines=[]
            for idx,tl in enumerate(x for x in root.iter() if local(x.tag)=="TextLine"):
                ue=next((x for x in tl.iter() if local(x.tag)=="Unicode"),None)
                if ue is None:continue
                y=baseline_y(tl)
                if y is None:y=float(idx)
                lines.append((y,idx,tokenize(render(ue,True)),tokenize(render(ue,False))))
            lines.sort(key=lambda x:(x[0],x[1]))
            su=[];se=[]
            for _,__,a,b in lines:su.extend(a);se.extend(b)
            if su:un[fn]=su
            if se:ex[fn]=se
    return sha,un,ex

def main():
    root=Path(__file__).resolve().parents[1]
    v=vms_paragraphs(root)
    vr=p5_fast(v)
    print("FAST_VMS_PARAGRAPH="+json.dumps(vr,sort_keys=True),flush=True)
    sha,un,ex=nuremberg()
    print("NUR_LENGTHS="+json.dumps({"sha":sha,"n_un":len(un),"tokens_un":sum(map(len,un.values())),
                                      "n_ex":len(ex),"tokens_ex":sum(map(len,ex.values()))},sort_keys=True),flush=True)
    ur=p5_fast(un); er=p5_fast(ex)
    out={"analysis":"XD1_P5_SEGMENT_SENSITIVITY_FAST_INDEPENDENT","source_sha256":sha,
         "unexpanded":ur,"expanded":er}
    payload=json.dumps(out,sort_keys=True,separators=(",",":"))
    out["result_sha256"]=hashlib.sha256(payload.encode()).hexdigest()
    print("FAST_NUREMBERG_SEGMENT="+json.dumps(out,sort_keys=True),flush=True)
if __name__=="__main__":main()
