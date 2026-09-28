#!/usr/bin/env python3
"""XD1-REM-20260928: frozen ReM physical-line lag1/lag2 extension."""
from __future__ import annotations
import argparse, collections, hashlib, io, json, lzma, os, pickle, re, urllib.request, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import xd1_recipe_repetition_closeout_20260928 as xd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"research"/"xd1_rem_results_20260928"; OUT.mkdir(exist_ok=True)
A=OUT/"checkpoint_A_rem_sources.pkl"; B=OUT/"checkpoint_B_rem_metrics.pkl"
URL="https://zenodo.org/records/13982324/files/ReM-v2.1_tei.zip?download=1"
EXPECTED_MD5="c57828a32f2634f3c5e72c3009e958a2"
EXPECTED=(406,2236137,9967570,118)
NS="{http://www.tei-c.org/ns/1.0}"
XMLID="{http://www.w3.org/XML/1998/namespace}id"
CLEAN=re.compile(r"[\[\]<>|\\/*()=+#%$\"'{}0-9\-.,;:!?]")
BASE_SEED=20260928

def md5(b): return hashlib.md5(b).hexdigest()
def sha(b): return hashlib.sha256(b).hexdigest()
def atomic(obj,path):
    tmp=path.with_suffix(path.suffix+".tmp")
    with open(tmp,"wb") as f: pickle.dump(obj,f,protocol=5)
    os.replace(tmp,path)

def clean(s):
    return CLEAN.sub("",s.lower())

def base_id(w):
    wid=w.get(XMLID) or ""
    return wid.split("_m")[0] if "_m" in wid else wid

def parse_doc(data:bytes):
    root=ET.fromstring(data)

    # Canonical layer exactly matching the previously verified ReM builder.
    groups=collections.defaultdict(str); order=[]
    for w in root.iter(NS+"w"):
        b=base_id(w); txt=re.sub(r"\s+","","".join(w.itertext()))
        if b not in groups: order.append(b)
        groups[b]+=txt
    flat=[clean(groups[b]) for b in order if groups[b]]
    flat=[w for w in flat if w]

    # Physical-line assignment without breaking an original token.
    # ReM may split one original token across an <lb>. All base-id segments
    # are reconstructed first; the token is assigned to the line on which
    # its FIRST segment begins.
    line_idx=0
    first_line={}
    for el in root.iter():
        if el.tag==NS+"lb":
            line_idx+=1
        elif el.tag==NS+"w":
            b=base_id(el)
            if b not in first_line:
                first_line[b]=line_idx
    line_map=collections.defaultdict(list)
    for b in order:
        if not groups[b]: continue
        z=clean(groups[b])
        if z:
            line_map[first_line.get(b,0)].append(z)
    lines=[line_map[k] for k in sorted(line_map) if line_map[k]]
    lined=[t for line in lines for t in line]
    return flat,lines,lined

def preflight():
    req=urllib.request.Request(URL,headers={"User-Agent":"XD1-REM-20260928/1.0"})
    data=urllib.request.urlopen(req,timeout=600).read()
    if md5(data)!=EXPECTED_MD5:
        raise SystemExit(f"archive MD5 mismatch {md5(data)}")
    z=zipfile.ZipFile(io.BytesIO(data))
    names=sorted(n for n in z.namelist() if n.lower().endswith(".xml"))
    docs={}; line_mismatch=[]
    for name in names:
        did=Path(name).stem
        flat,lines,lined=parse_doc(z.read(name))
        if not flat: continue
        docs[did]={"flat":flat,"lines":lines}
        if flat!=lined:
            # Equality is stricter than count: catches regrouping/order errors around lb.
            line_mismatch.append({"doc":did,"flat_n":len(flat),"lined_n":len(lined),
                                  "first_diff":next((i for i,(a,b) in enumerate(zip(flat,lined)) if a!=b),None)})
    nd=len(docs); nt=sum(len(v["flat"]) for v in docs.values())
    nc=sum(len(t) for v in docs.values() for t in v["flat"])
    alpha=len({c for v in docs.values() for t in v["flat"] for c in t})
    got=(nd,nt,nc,alpha)
    if got!=EXPECTED:
        raise SystemExit(f"canonical ReM totals mismatch got={got} expected={EXPECTED}")
    if line_mismatch:
        raise SystemExit(f"physical-line flatten mismatch in {len(line_mismatch)} docs; first={line_mismatch[:3]}")
    eligible=sorted(k for k,v in docs.items() if len(v["flat"])>=3000)
    panel=eligible[:12]
    if len(panel)!=12:raise SystemExit(f"only {len(panel)} eligible panel docs")
    # A transparent manifest independent of outcomes.
    manifest=[{"doc_id":k,"n_tokens":len(docs[k]["flat"]),"n_lines":len(docs[k]["lines"])} for k in panel]
    msha=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    bundle={"protocol_id":"XD1-REM-20260928","archive_md5":md5(data),"archive_sha256":sha(data),
            "canonical_totals":{"documents":nd,"tokens":nt,"chars":nc,"charset":alpha},
            "panel_manifest":manifest,"panel_manifest_sha256":msha,
            "panel":{k:docs[k]["lines"] for k in panel}}
    atomic(bundle,A)
    public={k:v for k,v in bundle.items() if k!="panel"}
    public["checkpoint_sha256"]=sha(A.read_bytes())
    (OUT/"source_qc.json").write_text(json.dumps(public,indent=2,sort_keys=True),encoding="utf-8")
    print("REM_PREFLIGHT="+json.dumps(public,sort_keys=True))

def rows(lines,doc):
    return [{"doc":doc,"line_order":i,"tokens":t} for i,t in enumerate(lines) if t]

def run():
    if not A.exists():raise SystemExit("missing source checkpoint")
    src=pickle.load(open(A,"rb"))
    pooled=[];perdoc={}
    for doc,ls in src["panel"].items():
        rr=rows(ls,doc); pooled.extend(rr)
        perdoc[doc]={}
        for lag in (1,2):
            perdoc[doc][str(lag)]=xd.metric(rr,lag,2000,BASE_SEED+10000*lag+sum(map(ord,doc)))
    res={"protocol_id":"XD1-REM-20260928","archive_sha256":src["archive_sha256"],
         "panel_manifest":src["panel_manifest"],"panel_manifest_sha256":src["panel_manifest_sha256"],
         "n_lines":len(pooled),"n_tokens":sum(len(r["tokens"]) for r in pooled),
         "lags":{},"per_document":perdoc}
    for lag in (1,2):
        res["lags"][str(lag)]={
          "primary_200":xd.all_bins(pooled,lag,200,BASE_SEED+lag*1000),
          "sensitivity_2000_ALL":xd.metric(pooled,lag,2000,BASE_SEED+lag*1000+333),
          "bootstrap":xd.bootstrap(pooled,lag,BASE_SEED+lag*1000+77)
        }
    l1=res["lags"]["1"]["sensitivity_2000_ALL"]; l2=res["lags"]["2"]["sensitivity_2000_ALL"]
    res["frozen_falsifier_fires"]=bool(l1["observed_over_null"]>=0.90 and l2["observed_over_null"]>=1.10)
    res["document_sign_summary"]={
      str(lag):{
        "positive_effect_docs":sum(perdoc[d][str(lag)] and perdoc[d][str(lag)]["effect"]>0 for d in perdoc),
        "negative_effect_docs":sum(perdoc[d][str(lag)] and perdoc[d][str(lag)]["effect"]<0 for d in perdoc),
        "resolved_docs":sum(perdoc[d][str(lag)] and perdoc[d][str(lag)]["effect_over_null_sd"]>=2 for d in perdoc)
      } for lag in (1,2)
    }
    atomic(res,B); res["checkpoint_sha256"]=sha(B.read_bytes())
    (OUT/"metrics.json").write_text(json.dumps(res,indent=2,sort_keys=True),encoding="utf-8")
    print("REM_RESULT="+json.dumps(res,sort_keys=True))

def main():
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight",action="store_true");g.add_argument("--run",action="store_true")
    a=ap.parse_args()
    preflight() if a.preflight else run()
if __name__=="__main__":main()
