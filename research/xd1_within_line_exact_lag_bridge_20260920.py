#!/usr/bin/env python3
"""
XD1 within-line exact-lag bridge to the June/July/August 2026 tight-null programme.

Primary targets (predeclared):
  lag 1 exact repetition under within-line multiset-preserving shuffle
  lag 2 exact repetition under the same null

Primary estimator reproduces the old frame:
  pooled observed exact matches / pooled eligible opportunities across physical lines;
  each null replicate independently permutes tokens WITHIN EACH physical line,
  preserving every line's exact token multiset and length.

Secondary:
  exact lags 3-5 are descriptive shape only.
  fixed line-length bins test whether any primary contrast is length-composition driven.

No page, paragraph, section, hand, Currier or document conditioning.
"""
import json, hashlib, sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import xd1_adapters_20260920 as adapters

NPERM=200
SEED=20260920
LAGS=(1,2,3,4,5)
PRIMARY=(1,2)
BINS=(("ALL",2,10**9),("2-5",2,5),("6-10",6,10),("11-20",11,20),("21+",21,10**9))
MIN_LINES=50

def encode_line(tokens):
    mp={t:i for i,t in enumerate(sorted(set(tokens)))}
    return np.asarray([mp[t] for t in tokens],dtype=np.int32)

def pooled_test(lines,lag,seed,nperm=NPERM):
    arr=[encode_line(t) for t in lines if len(t)>lag]
    opp=sum(len(a)-lag for a in arr)
    actual=sum(int(np.sum(a[lag:]==a[:-lag])) for a in arr)
    rng=np.random.default_rng(seed)
    null_hits=np.zeros(nperm,dtype=np.int64)
    # Preserve each line's exact multiset in every replicate.
    for a in arr:
        n=len(a)
        q=np.empty((nperm,n),dtype=a.dtype)
        for k in range(nperm):
            q[k]=rng.permutation(a)
        null_hits += np.sum(q[:,lag:]==q[:,:-lag],axis=1)
    observed=actual/opp
    null_rates=null_hits/opp
    nm=float(null_rates.mean()); ns=float(null_rates.std(ddof=1)); eff=observed-nm
    return dict(
        lag=lag,n_lines=len(arr),opportunities=opp,observed_hits=actual,
        observed=observed,null_mean=nm,null_sd=ns,effect=eff,
        effect_over_null_sd=(abs(eff)/ns if ns else None),
        observed_over_null=(observed/nm if nm else None),
        role=("PRIMARY" if lag in PRIMARY else "DESCRIPTIVE")
    )

def token_lines(obj):
    return [r["tokens"] for r in obj["lines"] if r.get("tokens")]

def subset(lines,lo,hi):
    return [t for t in lines if lo<=len(t)<=hi]

def run_corpus(label,lines,seed_offset=0):
    out={"label":label,"n_lines":len(lines),"n_tokens":sum(map(len,lines)),"bins":[]}
    for bi,(name,lo,hi) in enumerate(BINS):
        ls=subset(lines,lo,hi)
        row={"length_bin":name,"n_source_lines":len(ls),"n_source_tokens":sum(map(len,ls)),"lags":[]}
        for lag in LAGS:
            r=pooled_test(ls,lag,SEED+seed_offset+bi*100+lag)
            r["formal"]=r["n_lines"]>=MIN_LINES
            row["lags"].append(r)
        out["bins"].append(row)
    return out

def main():
    v=adapters.vms()
    nu,ne=adapters.nuremberg()
    vl=token_lines(v); nul=token_lines(nu); nel=token_lines(ne)
    result={
      "analysis":"XD1_WITHIN_LINE_EXACT_LAG_BRIDGE_20260920",
      "definition":"pooled opportunity exact repeats; within-line multiset-preserving permutation null",
      "nperm":NPERM,"seed":SEED,"primary_lags":[1,2],"descriptive_lags":[3,4,5],
      "bins":[x[0] for x in BINS],"min_lines":MIN_LINES,
      "vms_source_sha256":v["source_sha256"],"nuremberg_source_sha256":nu["source_sha256"],
      "vms":run_corpus("VMS_CANONICAL_ZLZI",vl,0),
      "nuremberg_unexpanded":run_corpus("NUREMBERG_UNEXPANDED",nul,10000),
      "nuremberg_expanded":run_corpus("NUREMBERG_EXPANDED",nel,20000)
    }
    # Add direct VMS-vs-Nuremberg primary contrasts.
    contrasts=[]
    for vb,nb in zip(result["vms"]["bins"],result["nuremberg_unexpanded"]["bins"]):
        for lag in PRIMARY:
            a=next(x for x in vb["lags"] if x["lag"]==lag)
            b=next(x for x in nb["lags"] if x["lag"]==lag)
            csd=(a["null_sd"]**2+b["null_sd"]**2)**0.5
            contrasts.append(dict(length_bin=vb["length_bin"],lag=lag,
                                  vms_effect=a["effect"],nuremberg_effect=b["effect"],
                                  difference=a["effect"]-b["effect"],combined_null_sd=csd,
                                  abs_difference_over_combined_null_sd=(abs(a["effect"]-b["effect"])/csd if csd else None),
                                  formal=a["formal"] and b["formal"]))
    result["primary_contrasts"]=contrasts
    payload=json.dumps(result,sort_keys=True,separators=(",",":"))
    result["result_sha256"]=hashlib.sha256(payload.encode()).hexdigest()
    print("XD1_WITHIN_LINE_BRIDGE="+json.dumps(result,sort_keys=True))

if __name__=="__main__":main()
