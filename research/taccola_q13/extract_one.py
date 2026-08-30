#!/usr/bin/env python3
import argparse, json, pickle, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import taccola_calibration_v01b as core

ap=argparse.ArgumentParser(); ap.add_argument('--id',required=True); args=ap.parse_args()
mid=args.id
if mid not in core.MANUSCRIPTS: raise SystemExit(f'unknown locked panel id: {mid}')
outdir=Path('taccola_one_output'); outdir.mkdir(exist_ok=True)
panel_hash=core.sha256_json(core.LOCKED_PANEL)
if panel_hash != core.EXPECTED_PANEL_SHA256: raise RuntimeError('panel mismatch')
meta=core.MANUSCRIPTS[mid]; errors=[]; sess=core.requests.Session()
try:
    rows=core.manifest_canvases(sess,meta['manifest']); sampled=core.even_sample(rows,core.PAGE_SAMPLE)
    print(json.dumps({'event':'manifest','id':mid,'pages':len(rows),'sample':len(sampled)}),flush=True)
except Exception as e:
    rows=[]; sampled=[]; errors.append({'stage':'manifest','id':mid,'error':repr(e)})
    print(json.dumps({'event':'manifest_error','id':mid,'error':repr(e)}),flush=True)
got=[]; diag=[]
def work(r):
    try:
        g=core.fetch_gray(r['url']); f=core.page_features(g)
        if f is None: raise ValueError('feature extraction returned None')
        return r,f,None
    except Exception as e:return r,None,repr(e)
with ThreadPoolExecutor(max_workers=10) as ex:
    futs=[ex.submit(work,r) for r in sampled]
    for fu in as_completed(futs):
        r,f,e=fu.result()
        if f is not None:
            f['index']=r['index']; f['label']=r['label']; got.append(f)
            diag.append({'index':r['index'],'label':r['label'],'illustration_score':f['illustration_score'],'inkfrac':f['inkfrac'],'bboxfrac':f['bboxfrac'],'components':f['component_count']})
        else: errors.append({'stage':'image','id':mid,'index':r.get('index'),'error':e})
got.sort(key=lambda x:(-x['illustration_score'],x['index'])); got=got[:core.ILLUSTRATION_TOP_K]
bundle={'panel_sha256':panel_hash,'id':mid,'features':got,'page_diag':sorted(diag,key=lambda x:x['index']),'manifest_rows':sampled,'errors':errors}
with open(outdir/f'one_{re.sub(r"[^A-Za-z0-9_.-]+","_",mid)}.pkl','wb') as f: pickle.dump(bundle,f,protocol=5)
(outdir/f'one_{re.sub(r"[^A-Za-z0-9_.-]+","_",mid)}.json').write_text(json.dumps({'id':mid,'download_ok':len(diag),'selected':len(got),'errors':errors},indent=2))
print(json.dumps({'event':'features','id':mid,'download_ok':len(diag),'selected':len(got),'errors_total':len(errors)}),flush=True)
if len(got)<12: raise SystemExit(f'{mid}: fewer than 12 selected features')
