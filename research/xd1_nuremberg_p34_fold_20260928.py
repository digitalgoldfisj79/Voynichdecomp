#!/usr/bin/env python3
"""One outer fold of the corrected XD1 Nuremberg P3/P4 completion."""
import argparse, collections, hashlib, json, math, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import xd1_adapters_20260920 as adapters
import xd1_core_20260920 as m

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--outer",type=int,required=True);ap.add_argument("--output",required=True)
    a=ap.parse_args()
    if a.outer not in range(5):raise SystemExit("outer must be 0..4")
    nu,_=adapters.nuremberg()
    lines=m.normalize_input(nu); folds=m.assign_folds(lines)
    rows=m.transitions(lines)
    tr=[r for r in rows if folds[r["block"]]!=a.outer]
    te=[r for r in rows if folds[r["block"]]==a.outer]
    lm=m.tune_morph(tr,a.outer); le=m.tune_exact(tr,a.outer,lm)
    p=m.TokenParent(tr); mm=m.TokenChild(tr,m.morph,p,lm); ee=m.TokenChild(tr,m.exact,mm,le)
    by=collections.defaultdict(lambda:[0.0,0.0,0])
    for r in te:
        pp=max(p.prob(r,r["target"]),1e-300); pm=max(mm.prob(r,r["target"]),1e-300); pe=max(ee.prob(r,r["target"]),1e-300)
        z=by[r["block"]];z[0]+=math.log2(pm/pp);z[1]+=math.log2(pe/pm);z[2]+=1
    mb=[v[0]/v[2] for v in by.values() if v[2]]
    eb=[v[1]/v[2] for v in by.values() if v[2]]
    out=dict(protocol_id="XD1-CLOSEOUT-20260928",corpus="NUREMBERG_2_5_DIPLOMATIC_UNEXPANDED_DROP_EX",
             outer=a.outer,source_sha256=nu["source_sha256"],n_lines=len(lines),
             n_tokens=sum(len(r["tokens"]) for r in lines),n_test=len(te),
             lambda_morph=lm,lambda_exact=le,morph_block_effects=mb,exact_block_effects=eb,
             morph_effect=sum(mb)/len(mb),exact_effect=sum(eb)/len(eb))
    payload=json.dumps(out,sort_keys=True,separators=(",",":"));out["result_sha256"]=hashlib.sha256(payload.encode()).hexdigest()
    Path(a.output).write_text(json.dumps(out,sort_keys=True),encoding="utf-8")
    print(json.dumps({k:v for k,v in out.items() if k not in ("morph_block_effects","exact_block_effects")},sort_keys=True))
if __name__=="__main__":main()
