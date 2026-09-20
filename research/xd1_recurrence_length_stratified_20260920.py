#!/usr/bin/env python3
"""
XD1 length-stratified recurrence sensitivity.
Frozen before outcomes. Uses the same P5 band statistic and within-block token-multiset permutation null.
Length bins are fixed from observed block-length ranges, not recurrence outcomes.
Formal bin comparison requires >=50 eligible blocks in BOTH corpora for the given lag band.
"""
import json, hashlib, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import xd1_recurrence_sensitivity_fast_20260920 as m

BINS=((2,5),(6,10),(11,20),(21,40),(41,80),(81,160),(161,10**9))
MIN_BLOCKS=50

def subset(seqs,lo,hi):
    return {k:v for k,v in seqs.items() if lo<=len(v)<=hi}

def main():
    root=Path(__file__).resolve().parents[1]
    v=m.vms_paragraphs(root)
    sha,nur,_=m.nuremberg()
    out={"analysis":"XD1_LENGTH_STRATIFIED_P5_20260920","bins":[list(x) for x in BINS],
         "min_blocks_per_corpus":MIN_BLOCKS,"nuremberg_source_sha256":sha,"rows":[]}
    for lo,hi in BINS:
        vs=subset(v,lo,hi); ns=subset(nur,lo,hi)
        vr=m.p5_fast(vs) if vs else []
        nr=m.p5_fast(ns) if ns else []
        vm={x["contrast"]:x for x in vr}; nm={x["contrast"]:x for x in nr}
        for contrast in sorted(set(vm)|set(nm)):
            a=vm.get(contrast); b=nm.get(contrast)
            formal=bool(a and b and a["n_blocks"]>=MIN_BLOCKS and b["n_blocks"]>=MIN_BLOCKS)
            out["rows"].append({
              "length_bin":f"{lo}-{hi if hi<10**9 else 'inf'}","contrast":contrast,
              "vms":a,"nuremberg":b,"formal":formal
            })
    payload=json.dumps(out,sort_keys=True,separators=(",",":"))
    out["result_sha256"]=hashlib.sha256(payload.encode()).hexdigest()
    print("XD1_LENGTH_STRATIFIED="+json.dumps(out,sort_keys=True))
if __name__=="__main__": main()
