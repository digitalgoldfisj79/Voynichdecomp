#!/usr/bin/env python3
import argparse, json, pickle, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import taccola_calibration_v01b as core

ap=argparse.ArgumentParser()
ap.add_argument("--shard",type=int,required=True)
ap.add_argument("--nshards",type=int,required=True)
args=ap.parse_args()
if not (0 <= args.shard < args.nshards): raise SystemExit("invalid shard")
outdir=Path("taccola_shard_output"); outdir.mkdir(exist_ok=True)
panel=json.loads(json.dumps(core.LOCKED_PANEL))
panel_hash=core.sha256_json(panel)
if panel_hash != core.EXPECTED_PANEL_SHA256: raise RuntimeError("panel mismatch")
ids=[mid for i,mid in enumerate(core.MANUSCRIPTS) if i % args.nshards == args.shard]
print(json.dumps({"event":"shard_locked","shard":args.shard,"nshards":args.nshards,"ids":ids,"panel_sha256":panel_hash}),flush=True)
manifest_rows={}; feats={}; page_diag={}; errors=[]
sess=core.requests.Session()
for mid in ids:
    meta=core.MANUSCRIPTS[mid]
    try:
        rows=core.manifest_canvases(sess,meta["manifest"]); sampled=core.even_sample(rows,core.PAGE_SAMPLE)
        manifest_rows[mid]=sampled
        print(json.dumps({"event":"manifest","id":mid,"pages":len(rows),"sample":len(sampled)}),flush=True)
    except Exception as e:
        manifest_rows[mid]=[]; errors.append({"stage":"manifest","id":mid,"error":repr(e)})
        print(json.dumps({"event":"manifest_error","id":mid,"error":repr(e)}),flush=True)
    got=[]; diag=[]
    def work(r):
        try:
            g=core.fetch_gray(r["url"]); f=core.page_features(g)
            if f is None: raise ValueError("feature extraction returned None")
            return r,f,None
        except Exception as e: return r,None,repr(e)
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs=[ex.submit(work,r) for r in manifest_rows[mid]]
        for fu in as_completed(futs):
            r,f,e=fu.result()
            if f is not None:
                f["index"]=r["index"]; f["label"]=r["label"]; got.append(f)
                diag.append({"index":r["index"],"label":r["label"],"illustration_score":f["illustration_score"],"inkfrac":f["inkfrac"],"bboxfrac":f["bboxfrac"],"components":f["component_count"]})
            else:
                errors.append({"stage":"image","id":mid,"index":r.get("index"),"error":e})
    got.sort(key=lambda x:(-x["illustration_score"],x["index"])); got=got[:core.ILLUSTRATION_TOP_K]
    feats[mid]=got; page_diag[mid]=sorted(diag,key=lambda x:x["index"])
    safe=re.sub(r"[^A-Za-z0-9_.-]+","_",mid)
    with open(outdir/f"checkpoint_features_{safe}.pkl","wb") as f:
        pickle.dump({"panel_sha256":panel_hash,"id":mid,"features":got,"page_diag":page_diag[mid],"manifest_rows":manifest_rows[mid],"errors":[x for x in errors if x.get("id")==mid]},f,protocol=5)
    print(json.dumps({"event":"features","id":mid,"download_ok":len(diag),"selected":len(got),"errors_total":len(errors)}),flush=True)
bundle={"panel_sha256":panel_hash,"shard":args.shard,"nshards":args.nshards,"ids":ids,"features":feats,"page_diag":page_diag,"manifest_rows":manifest_rows,"errors":errors}
with open(outdir/f"shard_{args.shard}.pkl","wb") as f: pickle.dump(bundle,f,protocol=5)
(outdir/f"shard_{args.shard}.json").write_text(json.dumps({"shard":args.shard,"ids":ids,"feature_counts":{k:len(v) for k,v in feats.items()},"errors":errors[:100]},indent=2))
