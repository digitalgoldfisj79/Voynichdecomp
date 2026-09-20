#!/usr/bin/env python3
"""Analytic expectation cross-check for XD1 within-line lag1/lag2 primary result."""
import json,hashlib,re,sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parent))
import xd1_adapters_20260920 as adapters

def vms_local(root):
    p=Path(root)/"voynich_transcriptions_slim.json"
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    if sha!=adapters.VMS_SHA:raise RuntimeError(sha)
    obj=json.load(open(p,encoding="utf-8"));out=[]
    for fol,ld in obj["pages"].items():
        mf=re.match(r"f(\d+)",fol)
        if not mf or int(mf.group(1)) not in adapters.CANON_FOLIO_NUMS:continue
        for rec in ld.values():
            if "P" not in str(rec.get("u","")):continue
            txt=rec.get("t",{}).get("ZLZI","")
            toks=[t.lower() for t in txt.split() if re.fullmatch(r"[a-z]+",t.lower())]
            if toks:out.append(toks)
    return sha,out

def analytic(lines,d):
    obs=opp=0;exp_hits=0.0
    for toks in lines:
        n=len(toks)
        if n<=d:continue
        o=n-d;opp+=o
        obs+=sum(toks[i]==toks[i-d] for i in range(d,n))
        c=Counter(toks)
        p=sum(v*(v-1) for v in c.values())/(n*(n-1))
        exp_hits += o*p
    observed=obs/opp
    expected=exp_hits/opp
    return {"lag":d,"observed_hits":obs,"opportunities":opp,"observed":observed,
            "analytic_null_mean":expected,"effect":observed-expected,
            "observed_over_analytic_null":observed/expected}

def main():
    root=Path(__file__).resolve().parents[1]
    vsha,vl=vms_local(root)
    nu,_=adapters.nuremberg(); nl=[r["tokens"] for r in nu["lines"] if r.get("tokens")]
    out={"vms_source_sha256":vsha,"nuremberg_source_sha256":nu["source_sha256"],
         "vms":[analytic(vl,d) for d in (1,2)],
         "nuremberg":[analytic(nl,d) for d in (1,2)]}
    print("XD1_ANALYTIC_CROSSCHECK="+json.dumps(out,sort_keys=True))
if __name__=="__main__":main()
