#!/usr/bin/env python3
"""Fetch full-page images for the already-preregistered 12-locus La Sfera collation.

This changes no witness or locus selection. It only replaces the failed vertical-crop
heuristic with full pages so the frozen readings can be inspected without layout bias.
"""
from __future__ import annotations
import csv, io, json, re, sys
from pathlib import Path
from collections import Counter
import requests
from PIL import Image

OUT=Path(sys.argv[1] if len(sys.argv)>1 else 'la_sfera_frozen12_fullpages'); OUT.mkdir(parents=True,exist_ok=True)
LOCI=[
 ('T00031','01.01.02','secol'),('T00251','01.04.07','O creator'),
 ('T00416','01.06.07','poter'),('T00722','01.11.01','Son l’Ariete'),
 ('T00842','01.12.05','loro'),('T01033','01.14.07','temperato'),
 ('T01384','01.19.03','singolare'),('T01541','01.21.08','Che ’l vero'),
 ('T01729','01.23.08','ragion'),('T02062','01.28.05','tal eclissi'),
 ('T02232','01.31.02','ciel'),('T02554','01.36.08','Ringrazierà')]
MANIFESTS={
 'Laur2':'https://tecabml.contentdm.oclc.org/iiif/info/plutei/716054/manifest.json',
 'Yale4':'https://collections.library.yale.edu/manifests/2026466?manifest=https://collections.library.yale.edu/manifests/2026466',
 'Barb4':'https://digi.vatlib.it/iiif/MSS_Barb.lat.4048/manifest.json',
 'Cap1':'https://digi.vatlib.it/iiif/MSS_Cappon.56/manifest.json',
 'Urb2':'https://digi.vatlib.it/iiif/MSS_Urb.lat.1754/manifest.json'}
PAGES={
 'Laur2':{'T00031':'81r','T00251':'81r','T00416':'81v','T00722':'82r','T00842':'82r','T01033':'82v','T01384':'83r','T01541':'83v','T01729':'83v','T02062':'84r','T02232':'84v','T02554':'85r'},
 'Yale4':{'T00031':'25r','T00251':'25v','T00416':'25v','T00722':'26v','T00842':'26v','T01033':'27r','T01384':'28r','T01541':'28r','T01729':'28v','T02062':'29v','T02232':'30r','T02554':'30v'},
 'Barb4':{'T00031':'2r','T00251':'2v','T00416':'2v','T00722':'3v','T00842':'3v','T01033':'4r','T01384':'5r','T01541':'5r','T01729':'5v','T02062':'6v','T02232':'7r','T02554':'7v'},
 'Cap1':{'T00031':'1r','T00251':'1v','T00416':'1v','T00722':'2v','T00842':'2v','T01033':'3r','T01384':'4r','T01541':'4r','T01729':'4v','T02062':'5v','T02232':'6r','T02554':'6v'},
 'Urb2':{'T00031':'99r','T00251':'99v','T00416':'99v','T00722':'100v','T00842':'100v','T01033':'101r','T01384':'102r','T01541':'102r','T01729':'102v','T02062':'103v','T02232':'104r','T02554':'104v'}}

def normlabel(x):
    if isinstance(x,dict):
        vals=[]
        for v in x.values(): vals += v if isinstance(v,list) else [v]
        x=' '.join(map(str,vals))
    return re.sub(r'[^a-z0-9]+','',str(x).lower())

def canvases(man):
    out=[]
    if 'sequences' in man:
        for c in man['sequences'][0].get('canvases',[]):
            im=(c.get('images') or [{}])[0].get('resource',{}); svc=im.get('service') or {}
            if isinstance(svc,list): svc=svc[0] if svc else {}
            out.append((c.get('label',''),svc.get('@id') or svc.get('id'),im.get('@id') or im.get('id')))
    else:
        for c in man.get('items',[]):
            try: body=c['items'][0]['items'][0]['body']
            except Exception: body={}
            svc=body.get('service') or {}
            if isinstance(svc,list): svc=svc[0] if svc else {}
            out.append((c.get('label',''),svc.get('id') or svc.get('@id'),body.get('id') or body.get('@id')))
    return out

def choose(cs,folio):
    f=normlabel(folio); cand=[]
    for i,(lab,sid,iid) in enumerate(cs):
        nl=normlabel(lab); score=0
        if nl==f: score=5
        elif nl.endswith(f): score=4
        elif f in nl: score=3
        if score: cand.append((score,-len(nl),-i,lab,sid,iid))
    return max(cand) if cand else None

def imgurl(sid,iid): return sid.rstrip('/')+'/full/2200,/0/default.jpg' if sid else iid

def main():
    prereg={'parent':'la_sfera_manual_loci_prepare.py','selection':'unchanged frozen 12 loci x 5 witnesses','purpose':'full-page recovery after crop heuristic failure','loci':[{'VariantID':v,'line_code':c,'base_segment':t} for v,c,t in LOCI],'witnesses':list(MANIFESTS)}
    (OUT/'PREREG_FULLPAGES12.json').write_text(json.dumps(prereg,indent=2,ensure_ascii=False),encoding='utf8')
    idx=[]
    for sig,murl in MANIFESTS.items():
        print('manifest',sig,flush=True)
        try:
            mr=requests.get(murl,timeout=90); mr.raise_for_status(); cs=canvases(mr.json())
        except Exception as e:
            idx.append({'siglum':sig,'status':'MANIFEST_ERROR','error':repr(e)}); continue
        sd=OUT/sig; sd.mkdir(exist_ok=True)
        cache={}
        for vid,code,seg in LOCI:
            fol=PAGES[sig][vid]; ch=choose(cs,fol)
            if not ch:
                idx.append({'siglum':sig,'VariantID':vid,'line_code':code,'folio':fol,'status':'CANVAS_NOT_FOUND'}); continue
            _,_,_,lab,sid,iid=ch; fp=sd/f'{fol}.jpg'
            if fol not in cache:
                try:
                    rr=requests.get(imgurl(sid,iid),timeout=120); rr.raise_for_status(); Image.open(io.BytesIO(rr.content)).convert('RGB').save(fp,quality=94); cache[fol]=str(fp)
                except Exception as e:
                    cache[fol]=None; idx.append({'siglum':sig,'VariantID':vid,'line_code':code,'folio':fol,'status':'IMAGE_ERROR','canvas_label':str(lab),'error':repr(e)}); continue
            if cache[fol]: idx.append({'siglum':sig,'VariantID':vid,'line_code':code,'folio':fol,'status':'OK','canvas_label':str(lab),'file':cache[fol]})
    fields=sorted({k for r in idx for k in r})
    with (OUT/'index_fullpages12.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(idx)
    print(Counter(r['status'] for r in idx),flush=True)
if __name__=='__main__': main()
