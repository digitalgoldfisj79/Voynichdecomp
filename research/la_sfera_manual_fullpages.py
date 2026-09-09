#!/usr/bin/env python3
"""Preregistered La Sfera branch-marker expansion.

Expands ONLY the three candidate branch markers discovered in the frozen
Laur2/Yale4 pilot, across witnesses selected by public IIIF availability and
secure La Sfera Project folio maps BEFORE their readings are inspected.
No witness is selected by textual state.
"""
from __future__ import annotations
import csv, io, json, re, sys
from pathlib import Path
from collections import Counter
import requests
from PIL import Image

OUT=Path(sys.argv[1] if len(sys.argv)>1 else 'la_sfera_branch_markers'); OUT.mkdir(parents=True,exist_ok=True)
LOCI=[
 ('T00416','01.06.07','poter'),
 ('T01384','01.19.03','singolare'),
 ('T01541','01.21.08','Che ’l vero'),
]
# Selected solely because all three loci have secure Project folio maps and a public IIIF manifest.
# Par4 excluded prospectively because Gallica returned 403 to the runner in the prior frozen packet.
MANIFESTS={
 'Laur1':'https://tecabml.contentdm.oclc.org/iiif/info/plutei/1437043/manifest.json',
 'Bos':'https://iiif.archive.org/iiif/lasfera00dati/manifest.json',
 'Barb2':'https://digi.vatlib.it/iiif/MSS_Barb.lat.4005/manifest.json',
 'Barb3':'https://digi.vatlib.it/iiif/MSS_Barb.lat.4016/manifest.json',
 'Chig2':'https://digi.vatlib.it/iiif/MSS_Chig.M.VII.148/manifest.json',
 'Vat2':'https://digi.vatlib.it/iiif/MSS_Vat.lat.7612/manifest.json',
 'Borg':'https://digi.vatlib.it/iiif/MSS_Borg.lat.539/manifest.json',
 'Barb4':'https://digi.vatlib.it/iiif/MSS_Barb.lat.4048/manifest.json',
 'Laur3':'https://tecabml.contentdm.oclc.org/iiif/info/plutei/742245/manifest.json',
 'Urb1':'https://digi.vatlib.it/iiif/MSS_Urb.lat.752/manifest.json',
 'Spe':'https://purl.stanford.edu/jq331pk9591/iiif/manifest',
 'Laur2':'https://tecabml.contentdm.oclc.org/iiif/info/plutei/716054/manifest.json',
 'Laur5':'https://tecabml.contentdm.oclc.org/iiif/info/plutei/1155933/manifest.json',
 'Urb2':'https://digi.vatlib.it/iiif/MSS_Urb.lat.1754/manifest.json',
 'Yale4':'https://collections.library.yale.edu/manifests/2026466?manifest=https://collections.library.yale.edu/manifests/2026466',
}
PAGES={
 'Laur1':{'T00416':'177r','T01384':'177v','T01541':'177v'},
 'Bos':{'T00416':'1v','T01384':'3v','T01541':'4r'},
 'Barb2':{'T00416':'1v','T01384':'4r','T01541':'4r'},
 'Barb3':{'T00416':'1v','T01384':'4r','T01541':'4r'},
 'Chig2':{'T00416':'1v','T01384':'4r','T01541':'4r'},
 'Vat2':{'T00416':'1v','T01384':'4r','T01541':'4r'},
 'Borg':{'T00416':'2v','T01384':'5r','T01541':'5r'},
 'Barb4':{'T00416':'2v','T01384':'5r','T01541':'5r'},
 'Laur3':{'T00416':'39v','T01384':'42r','T01541':'42r'},
 'Urb1':{'T00416':'3v','T01384':'6r','T01541':'6r'},
 'Spe':{'T00416':'4','T01384':'9','T01541':'9'},
 'Laur2':{'T00416':'81v','T01384':'83r','T01541':'83v'},
 'Laur5':{'T00416':'93v','T01384':'96r','T01541':'96r'},
 'Urb2':{'T00416':'99v','T01384':'102r','T01541':'102r'},
 'Yale4':{'T00416':'25v','T01384':'28r','T01541':'28r'},
}

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

def imgurl(sid,iid): return sid.rstrip('/')+'/full/2000,/0/default.jpg' if sid else iid

def main():
    prereg={
      'loci':[{'VariantID':v,'line_code':c,'base_segment':t} for v,c,t in LOCI],
      'selection_rule':'all witnesses in the public export with secure folio maps at all three target loci and working public IIIF known before readings; Par4 excluded due prior runner 403',
      'target_hypothesis':'do omission states at 01.06.07 (di), 01.19.03 (et/e), 01.21.08 (ci) co-segregate beyond Laur2+Yale4?',
      'witnesses':list(MANIFESTS),
      'frozen_pair':['Laur2','Yale4'],
      'prior_controls':['Barb4','Urb2'],
      'no_florence_graph_until_gate':'require co-segregating branch or independent hard object/person/source edge'
    }
    (OUT/'PREREG_BRANCH_MARKERS.json').write_text(json.dumps(prereg,indent=2,ensure_ascii=False),encoding='utf8')
    idx=[]
    for sig,murl in MANIFESTS.items():
        print('manifest',sig,flush=True)
        try:
            mr=requests.get(murl,timeout=90); mr.raise_for_status(); cs=canvases(mr.json())
        except Exception as e:
            for vid,code,seg in LOCI: idx.append({'siglum':sig,'VariantID':vid,'line_code':code,'status':'MANIFEST_ERROR','error':repr(e)})
            continue
        sd=OUT/sig; sd.mkdir(exist_ok=True)
        cache={}
        for vid,code,seg in LOCI:
            fol=PAGES[sig][vid]; ch=choose(cs,fol)
            if not ch:
                idx.append({'siglum':sig,'VariantID':vid,'line_code':code,'folio':fol,'status':'CANVAS_NOT_FOUND'}); continue
            _,_,_,lab,sid,iid=ch; key=(fol,str(lab)); fn=f'{re.sub(r"[^A-Za-z0-9_-]+","_",fol)}.jpg'; fp=sd/fn
            if key not in cache:
                try:
                    rr=requests.get(imgurl(sid,iid),timeout=120); rr.raise_for_status(); Image.open(io.BytesIO(rr.content)).convert('RGB').save(fp,quality=92); cache[key]=str(fp)
                except Exception as e: cache[key]=None; idx.append({'siglum':sig,'VariantID':vid,'line_code':code,'folio':fol,'status':'IMAGE_ERROR','canvas_label':str(lab),'error':repr(e)}); continue
            if cache[key]: idx.append({'siglum':sig,'VariantID':vid,'line_code':code,'folio':fol,'status':'OK','canvas_label':str(lab),'file':cache[key]})
    fields=sorted({k for r in idx for k in r})
    with (OUT/'index_branch_markers.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(idx)
    print(Counter(r['status'] for r in idx),flush=True)
if __name__=='__main__': main()
