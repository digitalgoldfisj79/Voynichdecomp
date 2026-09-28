#!/usr/bin/env python3
"""XD1-STA-RF-20260928 frozen STA/RF representation sensitivity."""
from __future__ import annotations
import argparse, hashlib, json, os, pickle, re, urllib.request
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import xd1_adapters_20260920 as adapters
import xd1_recipe_repetition_closeout_20260928 as xd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"research"/"xd1_sta_rf_results_20260928"; OUT.mkdir(exist_ok=True)
A=OUT/"checkpoint_A_sta_source.pkl"; B=OUT/"checkpoint_B_sta_metrics.pkl"
URL="https://voynich.nu/data/sta/RF1b.txt"
CODE_RE=re.compile(r"[A-Z][0-9a-z]")
LOCUS_RE=re.compile(r"^<(?P<loc>f[^,>]+),(?P<mode>[^>]+)>\s*(?P<body>.*)$")
FOLIO_RE=re.compile(r"^(f\d+[rv]\d*)")
EXPECTED_LOCI=5385
EXPECTED_LONG_WORDS=37087
EXPECTED_SHORT_WORDS=37848
EXPECTED_UNCERTAIN_MARKERS=761
EXPECTED_STA_CODES=157254
BASE_SEED=20260928

def sha(b): return hashlib.sha256(b).hexdigest()
def atomic(obj,path):
    tmp=path.with_suffix(path.suffix+".tmp")
    with open(tmp,"wb") as f: pickle.dump(obj,f,protocol=5)
    os.replace(tmp,path)

def words_from_body(body,exclude_uncertain=False):
    # Long-word convention: uncertain spaces do not divide words.
    body=body.replace("<->","")
    out=[]
    for chunk in body.split("."):
        if exclude_uncertain and ("[" in chunk or "]" in chunk):
            continue
        codes=CODE_RE.findall(chunk)
        if codes:
            out.append("".join(codes))
    return out

def parse(text,exclude_uncertain=False,only_p=False):
    rows=[]; n_loci=0; n_codes=0; n_words=0
    for raw in text.splitlines():
        m=LOCUS_RE.match(raw.strip())
        if not m: continue
        n_loci+=1
        loc=m.group("loc"); mode=m.group("mode"); body=m.group("body")
        ws=words_from_body(body,exclude_uncertain)
        n_words+=len(ws)
        n_codes+=sum(len(CODE_RE.findall(x)) for x in body.replace("<->","").split("."))
        if only_p and "P" not in mode:
            continue
        fm=FOLIO_RE.match(loc)
        if not fm: continue
        fol=fm.group(1)
        num=re.match(r"f(\d+)",fol)
        if not num or int(num.group(1)) not in adapters.CANON_FOLIO_NUMS:
            continue
        if ws:
            rows.append({"folio":fol,"locus":loc,"mode":mode,"tokens":ws})
    return rows,dict(n_loci=n_loci,n_long_words=n_words,n_sta_codes=n_codes)

def preflight():
    req=urllib.request.Request(URL,headers={"User-Agent":"XD1-STA-RF-20260928/1.0"})
    data=urllib.request.urlopen(req,timeout=120).read()
    text=data.decode("utf-8")
    if not text.startswith("#=IVTFF STA1 2.0"):
        raise SystemExit("unexpected RF/STA header")
    allrows,stats=parse(text,False,False)
    short_words=0; uncertain_markers=0
    for raw in text.splitlines():
        m=LOCUS_RE.match(raw.strip())
        if not m: continue
        body=m.group("body"); uncertain_markers += body.count("<->")
        for chunk in re.split(r"\.|<->",body):
            if CODE_RE.findall(chunk): short_words += 1
    expected=(EXPECTED_LOCI,EXPECTED_LONG_WORDS,EXPECTED_SHORT_WORDS,EXPECTED_UNCERTAIN_MARKERS,EXPECTED_STA_CODES)
    got=(stats["n_loci"],stats["n_long_words"],short_words,uncertain_markers,stats["n_sta_codes"])
    if got!=expected:
        raise SystemExit(f"RF source-count gate mismatch got={got} expected={expected}")
    p,_=parse(text,False,True)
    q,_=parse(text,True,True)
    bundle={"protocol_id":"XD1-STA-RF-20260928","url":URL,"source_sha256":sha(data),
            "published_count_gate":dict(stats,n_short_words=short_words,n_uncertain_markers=uncertain_markers),
            "primary":{"n_lines":len(p),"n_tokens":sum(len(r["tokens"]) for r in p),"lines":p},
            "uncertainty_excluded":{"n_lines":len(q),"n_tokens":sum(len(r["tokens"]) for r in q),"lines":q}}
    atomic(bundle,A)
    public={k:v for k,v in bundle.items() if k not in ("primary","uncertainty_excluded")}
    public["primary_counts"]={k:v for k,v in bundle["primary"].items() if k!="lines"}
    public["uncertainty_excluded_counts"]={k:v for k,v in bundle["uncertainty_excluded"].items() if k!="lines"}
    public["checkpoint_sha256"]=sha(A.read_bytes())
    (OUT/"source_qc.json").write_text(json.dumps(public,indent=2,sort_keys=True),encoding="utf-8")
    print("STA_PREFLIGHT="+json.dumps(public,sort_keys=True))

def run():
    if not A.exists(): raise SystemExit("missing STA source checkpoint")
    src=pickle.load(open(A,"rb"))
    result={"protocol_id":"XD1-STA-RF-20260928","source_sha256":src["source_sha256"],"arms":{}}
    passes=[]
    for j,arm in enumerate(("primary","uncertainty_excluded")):
        rows=src[arm]["lines"]
        rr={"n_lines":len(rows),"n_tokens":sum(len(r["tokens"]) for r in rows),"lags":{}}
        for lag in (1,2):
            rr["lags"][str(lag)]={
              "primary_200":xd.all_bins(rows,lag,200,BASE_SEED+j*100000+lag*1000),
              "sensitivity_2000_ALL":xd.metric(rows,lag,2000,BASE_SEED+j*100000+lag*1000+333),
              "bootstrap":xd.bootstrap(rows,lag,BASE_SEED+j*100000+lag*1000+77)
            }
        l1=rr["lags"]["1"]["sensitivity_2000_ALL"]; l2=rr["lags"]["2"]["sensitivity_2000_ALL"]
        signed1=l1["effect"]/l1["null_sd"] if l1 and l1["null_sd"] else None
        signed2=l2["effect"]/l2["null_sd"] if l2 and l2["null_sd"] else None
        rr["representation_gate"]=bool(signed2 is not None and signed2>=2 and signed1 is not None and signed1>-2)
        rr["signed_effect_over_null_sd"]={"lag1":signed1,"lag2":signed2}
        passes.append(rr["representation_gate"])
        result["arms"][arm]=rr
    result["criterion5_sta_representation_gate"]=all(passes)
    result["scope_note"]="STA/RF is a representation sensitivity test; RF derives from ZL+GC and is not an independent palaeographic transcription."
    atomic(result,B); result["checkpoint_sha256"]=sha(B.read_bytes())
    (OUT/"metrics.json").write_text(json.dumps(result,indent=2,sort_keys=True),encoding="utf-8")
    print("STA_RESULT="+json.dumps(result,sort_keys=True))

def main():
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight",action="store_true");g.add_argument("--run",action="store_true")
    a=ap.parse_args()
    preflight() if a.preflight else run()
if __name__=="__main__": main()
