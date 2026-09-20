#!/usr/bin/env python3
"""Fast primary-only implementation of frozen XD1 within-line lag1/lag2 bridge."""
import json, hashlib, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import xd1_adapters_20260920 as adapters

NPERM=200; SEED=20260920; LAGS=(1,2)
BINS=(("ALL",2,10**9),("2-5",2,5),("6-10",6,10),("11-20",11,20),("21+",21,10**9))
MIN_LINES=50

def local_vms(root):
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

def encode(toks):
    mp={t:i for i,t in enumerate(sorted(set(toks)))}
    return np.asarray([mp[t] for t in toks],dtype=np.int32)

def binkeys(n):
    return [name for name,lo,hi in BINS if lo<=n<=hi]

def run(lines,seed):
    arr=[encode(x) for x in lines]
    out={}
    for lag in LAGS:
        rng=np.random.default_rng(seed+lag)
        agg={name:{"hits":0,"opp":0,"n_lines":0,"null_hits":np.zeros(NPERM,dtype=np.int64)}
             for name,_,__ in BINS}
        for a in arr:
            if len(a)<=lag:continue
            keys=binkeys(len(a));opp=len(a)-lag;hits=int(np.sum(a[lag:]==a[:-lag]))
            q=np.empty((NPERM,len(a)),dtype=a.dtype)
            for k in range(NPERM):q[k]=rng.permutation(a)
            nh=np.sum(q[:,lag:]==q[:,:-lag],axis=1)
            for key in keys:
                z=agg[key];z["hits"]+=hits;z["opp"]+=opp;z["n_lines"]+=1;z["null_hits"]+=nh
        rows=[]
        for name,_,__ in BINS:
            z=agg[name]
            if not z["opp"]:continue
            obs=z["hits"]/z["opp"];nr=z["null_hits"]/z["opp"]
            nm=float(nr.mean());sd=float(nr.std(ddof=1));eff=obs-nm
            rows.append(dict(length_bin=name,lag=lag,n_lines=z["n_lines"],opportunities=z["opp"],
                             observed_hits=z["hits"],observed=obs,null_mean=nm,null_sd=sd,effect=eff,
                             effect_over_null_sd=abs(eff)/sd,observed_over_null=obs/nm if nm else None,
                             formal=z["n_lines"]>=MIN_LINES))
        out[str(lag)]=rows
    return out

def main():
    root=Path(__file__).resolve().parents[1]
    vsha,vlines=local_vms(root)
    nu,_=adapters.nuremberg(); nlines=[r["tokens"] for r in nu["lines"] if r.get("tokens")]
    res={"analysis":"XD1_WITHIN_LINE_PRIMARY_LAG12_FAST","definition":"frozen pooled-opportunity within-line multiset permutation null",
         "nperm":NPERM,"seed":SEED,"vms_source_sha256":vsha,"nuremberg_source_sha256":nu["source_sha256"],
         "vms_counts":{"lines":len(vlines),"tokens":sum(map(len,vlines))},
         "nuremberg_counts":{"lines":len(nlines),"tokens":sum(map(len,nlines))},
         "vms":run(vlines,0),"nuremberg":run(nlines,10000)}
    contrasts=[]
    for lag in LAGS:
        vm={x["length_bin"]:x for x in res["vms"][str(lag)]};nm={x["length_bin"]:x for x in res["nuremberg"][str(lag)]}
        for name,_,__ in BINS:
            if name not in vm or name not in nm:continue
            a,b=vm[name],nm[name];csd=(a["null_sd"]**2+b["null_sd"]**2)**.5
            contrasts.append(dict(lag=lag,length_bin=name,formal=a["formal"] and b["formal"],
                                  vms_effect=a["effect"],nuremberg_effect=b["effect"],
                                  difference=a["effect"]-b["effect"],combined_null_sd=csd,
                                  abs_difference_over_combined_null_sd=abs(a["effect"]-b["effect"])/csd))
    res["contrasts"]=contrasts
    payload=json.dumps(res,sort_keys=True,separators=(",",":"))
    res["result_sha256"]=hashlib.sha256(payload.encode()).hexdigest()
    print("XD1_WITHIN_LINE_PRIMARY="+json.dumps(res,sort_keys=True))
if __name__=="__main__":main()
