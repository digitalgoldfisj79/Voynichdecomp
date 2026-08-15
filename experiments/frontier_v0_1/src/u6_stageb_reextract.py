from __future__ import annotations
import argparse, base64, concurrent.futures as cf, hashlib, json, os, shutil, subprocess, sys, traceback, zlib
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

CALIB_FOLIOS = ['f10r','f10v','f11r','f11v','f13r','f13v','f14r','f14v','f15r','f15v','f16r','f16v','f17r','f17v','f18r','f18v','f19r','f19v','f1v','f20r','f20v','f21r','f21v','f22r','f22v','f23r','f23v','f24r','f24v','f25r','f25v','f26r','f26v','f27r','f27v','f28r','f28v','f29r','f29v','f2r','f2v','f30r','f30v','f31r','f31v','f32r','f32v','f33r','f33v','f34r','f34v','f35r','f35v','f36r','f36v','f37r','f37v','f38r','f38v','f39r','f39v','f3r','f3v','f40r','f40v','f41r','f42r','f42v','f43r','f43v','f44r','f44v','f45r','f45v','f46r','f46v','f47r','f47v','f48r','f48v','f49r','f49v','f4r','f4v','f50r','f50v','f51r','f51v','f52r','f52v','f53r','f53v','f54r','f54v','f55r','f55v','f56r','f56v','f5r','f6r','f6v','f7r','f7v','f8r','f8v','f9r','f9v']
EXPECTED_WORD_KEYSET_SHA256='c494eb695691e899d6e1dc648f9f7d7ec4afe49141a8890f9c1c40638b6a3f84'
EXPECTED_PAIR_SHA256='7f29bb7fe782130ddffe3d7809ce024e04a7eb01fa5c4194440d3be18cea3ed4'
EXPECTED_ENCODER_SHA256='54ef0612e623fa1755a488cdb975263c93f77c034085b3fa11eff21b62ba52b0'
EXPECTED_WORDS=9620
EXPECTED_FOLIOS=107


def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def post_json(url,key,obj):
    if not url or not key: return None
    r=requests.post(url,headers={'apikey':key,'Authorization':'Bearer '+key,'Content-Type':'application/json','x-upsert':'true'},data=json.dumps(obj,sort_keys=True).encode(),timeout=120)
    return r.status_code


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest-root',type=Path,required=True)
    ap.add_argument('--pipeline-root',type=Path,required=True)
    ap.add_argument('--stageb-script',type=Path,required=True)
    ap.add_argument('--encoder',type=Path,required=True)
    ap.add_argument('--pair-b64',type=Path,required=True)
    ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--status-url',default=os.getenv('STATUS_URL'))
    ap.add_argument('--audit-url',default=os.getenv('AUDIT_URL'))
    ap.add_argument('--result-url',default=os.getenv('RESULT_URL'))
    ap.add_argument('--result-key',default=os.getenv('RESULT_KEY'))
    a=ap.parse_args()
    a.work.mkdir(parents=True,exist_ok=True)
    status={'schema':'u6-stageb-reextract-v0.2','status':'initialising','target_opened':False,'true_retention_read':False}
    post_json(a.status_url,a.result_key,status)
    sys.path.insert(0,str(a.pipeline_root))
    from vdino3 import cfg,sources,register,crop
    # Only suppress downstream component proposals. Word extraction itself remains original.
    crop.connected_components=lambda *args,**kwargs: []
    crop.eva_soft_partition=lambda *args,**kwargs: []
    cfg.CACHE_DIR=str(a.work/'cache'); Path(cfg.CACHE_DIR).mkdir(parents=True,exist_ok=True)

    manifest=a.manifest_root/'results/corpus_crop_manifest.jsonl'
    expected=[]
    folset=set(CALIB_FOLIOS)
    with manifest.open('r',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r.get('kind')=='word' and r.get('view')=='norm' and str(r.get('folio')) in folset:
                expected.append({'id':str(r['id']),'folio':str(r['folio']),'word_index':int(r['word_index']),'word':str(r.get('word',''))})
    E=pd.DataFrame(expected).sort_values(['folio','word_index','id'],kind='stable').drop_duplicates(['folio','word_index'],keep='first').reset_index(drop=True)
    keytext=''.join(f'{f}|{int(i)}\n' for f,i in sorted(zip(E.folio,E.word_index)))
    keyhash=hashlib.sha256(keytext.encode()).hexdigest()
    if len(E)!=EXPECTED_WORDS or E.folio.nunique()!=EXPECTED_FOLIOS or keyhash!=EXPECTED_WORD_KEYSET_SHA256:
        raise RuntimeError(f'full-manifest calibration population gate failed rows={len(E)} folios={E.folio.nunique()} hash={keyhash}')
    byfolio={fol:g.copy() for fol,g in E.groupby('folio')}
    status.update(status='population_verified',word_rows=len(E),folios=E.folio.nunique(),word_keyset_sha256=keyhash)
    post_json(a.status_url,a.result_key,status)

    man=sources.yale_manifest(); canvases=sources.yale_canvases(man)
    outroot=a.work/'stageb_data'; (outroot/'results').mkdir(parents=True,exist_ok=True)
    shutil.copy2(manifest,outroot/'results/corpus_crop_manifest.jsonl')
    regaudit={}; failures=[]

    def one_folio(folio):
        g=byfolio[folio]
        reg,allc=register.register_folio(folio,canvases,6)
        rec={'folio':folio,'passed':bool(reg.passed),'matches':int(reg.matches),'inliers':int(reg.inliers),'inlier_ratio':float(reg.inlier_ratio),'median_reproj_px':float(reg.median_reproj_px),'p95_reproj_px':float(reg.p95_reproj_px),'service_id':reg.service_id,'candidate_n':len(allc),'required_words':len(g)}
        if not reg.passed: return rec,{'error':'registration_failed'}
        boxes=sources.parse_runtime_boxes(folio); mapped=register.transform_boxes(reg,boxes); md={int(x['index']):x for x in mapped}
        miss=[int(x) for x in g.word_index if int(x) not in md]
        if miss: return rec,{'error':'mapped_index_missing','indices':miss[:20],'n':len(miss)}
        info=json.loads(sources.fetch(reg.service_id+'/info.json','.json')); full_wh=(int(info['width']),int(info['height']))
        fd=outroot/'reextract'/folio; fd.mkdir(parents=True,exist_ok=True); w=crop.ProposalWriter(str(fd))
        for wi in g.word_index:
            w.add_word(folio,reg.service_id,full_wh,md[int(wi)])
        w.flush()
        mf=fd/'crop_manifest.jsonl'; got=[]
        with mf.open('r',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                if r.get('kind')=='word' and r.get('view')=='norm': got.append(r)
        G=pd.DataFrame(got)
        if len(G)!=len(g): return rec,{'error':'generated_word_count','expected':len(g),'got':len(G)}
        gm={(str(r.folio),int(r.word_index)):str(r.id) for r in G.itertuples()}; mism=[]
        for r in g.itertuples():
            k=(str(r.folio),int(r.word_index)); x=gm.get(k)
            if x!=str(r.id): mism.append({'key':k,'expected':str(r.id),'got':x})
        if mism: return rec,{'error':'crop_id_mismatch','n':len(mism),'examples':mism[:10]}
        return rec,None

    completed=0
    with cf.ThreadPoolExecutor(max_workers=max(1,a.workers)) as ex:
        fut={ex.submit(one_folio,fol):fol for fol in sorted(byfolio)}
        for f in cf.as_completed(fut):
            fol=fut[f]
            try: rec,err=f.result()
            except Exception as e:
                rec={'folio':fol}; err={'error':'exception','repr':repr(e),'trace':traceback.format_exc()[-4000:]}
            regaudit[fol]=rec
            if err: failures.append({'folio':fol,**err})
            completed+=1
            if completed%5==0 or err:
                status.update(status='reextracting',folios_completed=completed,folios_total=len(byfolio),failures=len(failures))
                post_json(a.status_url,a.result_key,status)
    audit={'schema':'u6-stageb-reextract-audit-v0.2','target_opened':False,'true_retention_read':False,'population':{'words':len(E),'folios':E.folio.nunique(),'word_keyset_sha256':keyhash},'registrations':regaudit,'failures':failures,'pipeline_contract':{'max_candidates':6,'word_pad_frac':float(cfg.WORD_PAD_FRAC),'iiif_word_maxdim':int(cfg.IIIF_WORD_MAXDIM),'component_proposals_suppressed_after_word_save':True}}
    post_json(a.audit_url,a.result_key,audit)
    if failures:
        status.update(status='failed_reextract',failures=len(failures),failure_examples=failures[:10]); post_json(a.status_url,a.result_key,status); raise RuntimeError(f're-extraction failed for {len(failures)} folios')

    # Verify exact number of recovered normalized word images.
    norm=list((outroot/'reextract').rglob('*_norm.png'))
    if len(norm)!=EXPECTED_WORDS:
        raise RuntimeError(f'norm crop count gate failed {len(norm)} != {EXPECTED_WORDS}')
    pair=a.work/'U6_STAGEB_EVENT_SKELETON.csv'
    raw=base64.b64decode(a.pair_b64.read_text().strip()); pair.write_bytes(zlib.decompress(raw))
    if sha256(pair)!=EXPECTED_PAIR_SHA256: raise RuntimeError(f'pair hash gate failed {sha256(pair)}')
    if sha256(a.encoder)!=EXPECTED_ENCODER_SHA256: raise RuntimeError(f'encoder hash gate failed {sha256(a.encoder)}')
    status.update(status='reextract_verified',folios_completed=len(byfolio),norm_word_crops=len(norm),failures=0)
    post_json(a.status_url,a.result_key,status)

    cmd=[sys.executable,str(a.stageb_script),'--data',str(outroot),'--encoder',str(a.encoder),'--pair-skeleton',str(pair),'--out',str(a.work/'stageb_out'),'--result-put-url',str(a.result_url),'--result-key',str(a.result_key)]
    status.update(status='stageb_calibrating'); post_json(a.status_url,a.result_key,status)
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    status.update(status='completed' if p.returncode==0 else 'failed_stageb',returncode=p.returncode,tail=p.stdout[-8000:])
    post_json(a.status_url,a.result_key,status)
    print(p.stdout[-16000:],flush=True)
    raise SystemExit(p.returncode)

if __name__=='__main__': main()
