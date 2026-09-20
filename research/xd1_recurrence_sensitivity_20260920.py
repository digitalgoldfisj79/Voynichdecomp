#!/usr/bin/env python3
"""
XD1 recurrence segmentation sensitivity.
Scientific definitions unchanged from XD1 P5:
  lag bands 1, 2-5, 6-16, 17-64
  exact-token recurrence
  within-block token-multiset permutation null
  200 deterministic permutations, NumPy default_rng(20260920)
Sensitivity blocks:
  - VMS: natural paragraph_id units from frozen Supabase export
  - Nuremberg: individual diplomatic correspondence XML records
This is a sensitivity analysis, not a replacement for registered page-level P5.
"""
import collections, hashlib, json, os, re, unicodedata, urllib.request, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

SEED=20260920
NPERM=200
BANDS=((1,1),(2,5),(6,16),(17,64))
NUR_RECORD="https://zenodo.org/api/records/13881575"
EXPECTED_NUR_SHA="59e5264acb4546477567e78c8b3d444c472f1a0a5256ee0ee7d0407a70904652"

def local(tag): return tag.split("}")[-1]

def tokenize(text):
    text=unicodedata.normalize("NFC",text).replace("\n"," ")
    out=[];buf=[]
    for ch in text:
        cat=unicodedata.category(ch)
        if cat[0] in ("L","M") or cat=="Nd":
            buf.append(ch.lower())
        else:
            if buf: out.append("".join(buf));buf=[]
    if buf: out.append("".join(buf))
    return [t for t in out if any(unicodedata.category(ch).startswith("L") for ch in t)]

def render_unicode(el,drop_ex=False):
    def rec(x):
        if drop_ex and local(x.tag)=="ex":
            return x.tail or ""
        s=x.text or ""
        for ch in x: s+=rec(ch)
        return s+(x.tail or "")
    s=el.text or ""
    for ch in el: s+=rec(ch)
    return s

def baseline_y(line):
    bl=next((x for x in line if local(x.tag)=="Baseline"),None)
    if bl is None: return None
    ys=[]
    for p in bl.attrib.get("points","").split():
        try: ys.append(float(p.split(",")[1]))
        except Exception: pass
    return sum(ys)/len(ys) if ys else None

def band_rate(ids,lo,hi):
    n=len(ids);eligible=max(0,n-lo)
    if eligible==0:return None
    w=hi-lo+1
    padded=np.concatenate((np.full(hi,-1,dtype=ids.dtype),ids))
    windows=np.lib.stride_tricks.sliding_window_view(padded,w)
    prev=windows[lo:n]
    target=ids[lo:n]
    return float(np.any(prev==target[:,None],axis=1).mean())

def p5_sequences(seqs,nperm=NPERM,seed=SEED):
    rng=np.random.default_rng(seed)
    encoded={}
    for b,s in seqs.items():
        if not s: continue
        mp={t:i for i,t in enumerate(sorted(set(s)))}
        encoded[b]=np.asarray([mp[t] for t in s],dtype=np.int32)
    out=[]
    for lo,hi in BANDS:
        eligible={b:a for b,a in encoded.items() if len(a)>lo}
        actual={b:band_rate(a,lo,hi) for b,a in eligible.items()}
        reps=np.zeros(nperm,float)
        page_null={b:[] for b in eligible}
        for k in range(nperm):
            vals=[]
            for b,a in eligible.items():
                q=rng.permutation(a); rr=band_rate(q,lo,hi)
                vals.append(rr); page_null[b].append(rr)
            reps[k]=float(np.mean(vals)) if vals else np.nan
        actual_mean=float(np.mean(list(actual.values()))) if actual else None
        null_mean=float(np.nanmean(reps)) if len(reps) else None
        null_sd=float(np.nanstd(reps,ddof=1)) if len(reps)>1 else None
        effect=actual_mean-null_mean
        out.append(dict(
            contrast=f"LAG_{lo}_{hi}",n_blocks=len(actual),nperm=nperm,
            observed=actual_mean,null_mean=null_mean,null_sd=null_sd,effect=effect,
            effect_over_null_sd=(abs(effect)/null_sd if null_sd else None),
            positive_blocks=sum(actual[b]>float(np.mean(page_null[b])) for b in actual),
            negative_blocks=sum(actual[b]<float(np.mean(page_null[b])) for b in actual)
        ))
    return out

def vms_paragraphs(repo_root):
    path=Path(repo_root)/"research"/"xd1_vms_paragraphs_20260920.json"
    obj=json.load(open(path,encoding="utf-8"))
    seqs={}
    bad=0
    for p in obj["paragraphs"]:
        toks=[]
        for t in p["text"].split():
            t=t.lower()
            if re.fullmatch(r"[a-z]+",t): toks.append(t)
            else: bad+=1
        if toks: seqs[p["paragraph_id"]]=toks
    return obj,seqs,bad

def nuremberg_segments(tmp="/tmp/nuremberg_labels.zip"):
    meta=json.load(urllib.request.urlopen(NUR_RECORD))
    f=next(x for x in meta["files"] if x["key"]=="labels.zip")
    urllib.request.urlretrieve(f["links"]["self"],tmp)
    sha=hashlib.sha256(Path(tmp).read_bytes()).hexdigest()
    if sha!=EXPECTED_NUR_SHA:
        raise RuntimeError(f"Nuremberg SHA mismatch: {sha}")
    unexp={}; exp={}
    with zipfile.ZipFile(tmp) as z:
        names=sorted(n for n in z.namelist() if "/diplomatic-regularised/" in n and n.endswith(".xml"))
        for fn in names:
            root=ET.fromstring(z.read(fn))
            lines=[]
            for idx,tl in enumerate(x for x in root.iter() if local(x.tag)=="TextLine"):
                ue=next((x for x in tl.iter() if local(x.tag)=="Unicode"),None)
                if ue is None: continue
                y=baseline_y(tl)
                if y is None:
                    custom=tl.attrib.get("custom","");m=re.search(r"index:(\\d+)",custom)
                    y=float(m.group(1)) if m else float(idx)
                tu=tokenize(render_unicode(ue,True))
                te=tokenize(render_unicode(ue,False))
                if tu or te: lines.append((y,idx,tu,te))
            lines.sort(key=lambda x:(x[0],x[1]))
            su=[];se=[]
            for _,__,tu,te in lines:
                su.extend(tu);se.extend(te)
            if su: unexp[fn]=su
            if se: exp[fn]=se
    return sha,unexp,exp

def summarize_lengths(seqs):
    a=np.array([len(x) for x in seqs.values()],dtype=float)
    return dict(n_blocks=len(a),n_tokens=int(a.sum()),min=int(a.min()),median=float(np.median(a)),
                mean=float(a.mean()),p90=float(np.quantile(a,.9)),max=int(a.max()))

def main():
    root=Path(__file__).resolve().parents[1]
    vmeta,vseq,bad=vms_paragraphs(root)
    nsha,nun,nex=nuremberg_segments()
    result={
      "analysis":"XD1_P5_SEGMENTATION_SENSITIVITY_20260920",
      "definition":"registered P5 recurrence/null; sensitivity block only",
      "seed":SEED,"nperm":NPERM,"bands":[list(x) for x in BANDS],
      "vms_paragraph":{
        "source_sha256":vmeta["source_sha256"],"frozen_n_words":vmeta["n_words"],
        "parsed_bad_tokens":bad,"lengths":summarize_lengths(vseq),"P5":p5_sequences(vseq)
      },
      "nuremberg_correspondence_unexpanded":{
        "source_sha256":nsha,"lengths":summarize_lengths(nun),"P5":p5_sequences(nun)
      },
      "nuremberg_correspondence_expanded":{
        "source_sha256":nsha,"lengths":summarize_lengths(nex),"P5":p5_sequences(nex)
      }
    }
    payload=json.dumps(result,sort_keys=True,separators=(",",":"),ensure_ascii=False)
    result["result_sha256"]=hashlib.sha256(payload.encode()).hexdigest()
    out=root/"research"/"RESULTS_xd1_recurrence_sensitivity_20260920.json"
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print("XD1_SENSITIVITY="+json.dumps(result,ensure_ascii=False,sort_keys=True))

if __name__=="__main__": main()
