#!/usr/bin/env python3
"""
XD1 exact-lag bridge: prospectively connects the August lag-1/lag-2 programme
to the current cross-domain framework.

Primary replication targets:
  D1 = exact lag 1
  D2 = exact lag 2
D3-D5 are descriptive shape controls, not independent discovery claims.

Units:
  VMS natural paragraphs
  Nuremberg individual diplomatic correspondence XML records
Length strata fixed from the already-frozen >=50-block gate:
  11-20, 21-40, 41-80 tokens
Null:
  within-block token-multiset permutation, 200 deterministic permutations.
"""
import json, hashlib, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import xd1_recurrence_sensitivity_fast_20260920 as src

NPERM=200
BASE_SEED=20260920
LAGS=(1,2,3,4,5)
BINS=((11,20),(21,40),(41,80))
MIN_BLOCKS=50

def encode(seq):
    mp={t:i for i,t in enumerate(sorted(set(seq)))}
    return np.asarray([mp[t] for t in seq],dtype=np.int32)

def exact_rate(a,d):
    if len(a)<=d:return None
    return float(np.mean(a[d:]==a[:-d]))

def perm_rates(a,d,rng,nperm=NPERM):
    if len(a)<=d:return None
    q=np.empty((nperm,len(a)),dtype=a.dtype)
    for k in range(nperm): q[k]=rng.permutation(a)
    return np.mean(q[:,d:]==q[:,:-d],axis=1)

def run(seqs,seed):
    enc={k:encode(v) for k,v in seqs.items() if v}
    rng=np.random.default_rng(seed); out=[]
    for d in LAGS:
        eligible={k:a for k,a in enc.items() if len(a)>d}
        actual={k:exact_rate(a,d) for k,a in eligible.items()}
        sums=np.zeros(NPERM,float)
        pos=neg=0
        for k,a in eligible.items():
            rr=perm_rates(a,d,rng)
            sums+=rr
            nm=float(rr.mean())
            pos+=actual[k]>nm;neg+=actual[k]<nm
        reps=sums/len(eligible)
        obs=float(np.mean(list(actual.values())))
        null=float(reps.mean());sd=float(reps.std(ddof=1));eff=obs-null
        out.append(dict(lag=d,n_blocks=len(eligible),observed=obs,null_mean=null,null_sd=sd,
                        effect=eff,effect_over_null_sd=abs(eff)/sd,
                        positive_blocks=pos,negative_blocks=neg,
                        role=("PRIMARY" if d in (1,2) else "DESCRIPTIVE")))
    return out

def subset(seqs,lo,hi):return {k:v for k,v in seqs.items() if lo<=len(v)<=hi}

def main():
    root=Path(__file__).resolve().parents[1]
    v=src.vms_paragraphs(root)
    sha,nur,_=src.nuremberg()
    result={"analysis":"XD1_EXACT_LAG_BRIDGE_20260920","lags":list(LAGS),
            "primary_lags":[1,2],"bins":[list(x) for x in BINS],
            "min_blocks":MIN_BLOCKS,"nperm":NPERM,"nuremberg_source_sha256":sha,"rows":[]}
    for bi,(lo,hi) in enumerate(BINS):
        vs=subset(v,lo,hi);ns=subset(nur,lo,hi)
        vr=run(vs,BASE_SEED+bi*10)
        nr=run(ns,BASE_SEED+1000+bi*10)
        vm={x["lag"]:x for x in vr};nm={x["lag"]:x for x in nr}
        for d in LAGS:
            a=vm[d];b=nm[d]
            csd=(a["null_sd"]**2+b["null_sd"]**2)**0.5
            result["rows"].append({
              "length_bin":f"{lo}-{hi}","lag":d,"role":a["role"],
              "vms":a,"nuremberg":b,
              "formal":a["n_blocks"]>=MIN_BLOCKS and b["n_blocks"]>=MIN_BLOCKS,
              "difference":a["effect"]-b["effect"],
              "combined_null_sd":csd,
              "abs_difference_over_combined_null_sd":abs(a["effect"]-b["effect"])/csd
            })
    payload=json.dumps(result,sort_keys=True,separators=(",",":"))
    result["result_sha256"]=hashlib.sha256(payload.encode()).hexdigest()
    print("XD1_EXACT_LAG_BRIDGE="+json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
