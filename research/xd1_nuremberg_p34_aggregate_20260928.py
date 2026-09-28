#!/usr/bin/env python3
"""Aggregate five independently executed corrected XD1 Nuremberg P3/P4 outer folds."""
import argparse, hashlib, json, pickle
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import xd1_core_20260920 as m

def main():
    ap=argparse.ArgumentParser();ap.add_argument("directory");ap.add_argument("--output",required=True)
    a=ap.parse_args();d=Path(a.directory)
    xs=[]
    for k in range(5):
        hits=list(d.rglob(f"fold_{k}.json"))
        if len(hits)!=1:raise SystemExit(f"expected one fold_{k}.json, got {hits}")
        xs.append(json.loads(hits[0].read_text()))
    shas={x["source_sha256"] for x in xs}
    if len(shas)!=1:raise SystemExit(f"source hash mismatch {shas}")
    mb=[z for x in xs for z in x["morph_block_effects"]]
    eb=[z for x in xs for z in x["exact_block_effects"]]
    p3=m.signflip(mb);p3.update(metric_id="P3",contrast="PREV_MORPH_GAIN")
    p4=m.signflip(eb);p4.update(metric_id="P4",contrast="EXACT_AFTER_MORPH_GAIN",
                               exact_off_folds=sum(x["lambda_exact"]>=16384 for x in xs))
    out=dict(protocol_id="XD1-CLOSEOUT-20260928",corpus="NUREMBERG_2_5_DIPLOMATIC_UNEXPANDED_DROP_EX",
             source_sha256=next(iter(shas)),folds=[{k:v for k,v in x.items() if k not in ("morph_block_effects","exact_block_effects")} for x in xs],
             P3=p3,P4=p4)
    payload=json.dumps(out,sort_keys=True,separators=(",",":"));out["result_sha256"]=hashlib.sha256(payload.encode()).hexdigest()
    op=Path(a.output);op.write_text(json.dumps(out,indent=2,sort_keys=True),encoding="utf-8")
    pp=op.with_suffix(".pkl");pickle.dump(out,open(pp,"wb"),protocol=5)
    print("NUREMBERG_P34="+json.dumps({k:v for k,v in out.items() if k!="folds"},sort_keys=True))
    print("PICKLE_SHA256="+hashlib.sha256(pp.read_bytes()).hexdigest())
if __name__=="__main__":main()
