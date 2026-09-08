#!/usr/bin/env python3
"""Full-page packet for preregistered La Sfera textual collation.

This preserves the 12 loci frozen before target readings were inspected.  It
saves full IIIF pages (rather than approximate line crops) so line location is
checked visually and no false precision is introduced by page geometry.

No missing visual witness is substituted.  Par4 is omitted because Gallica
returns 403 to the runner; NYPL2 has no IIIF URL; Fn12/He1/He2 are retained as
uncollated until a safe Book-I page map is established; Cap1 is retained as
uncollated because the Vatican canvas labelled 1r did not visually match the
Project's text start in the first packet.
"""
from __future__ import annotations
import csv, io, json, re, sys
from pathlib import Path
from collections import Counter
import requests
from PIL import Image

OUT=Path(sys.argv[1] if len(sys.argv)>1 else 'la_sfera_fullpages'); OUT.mkdir(parents=True,exist_ok=True)
LOCI=[
 ('T00031','01.01.02','secol'),('T00251','01.04.07','O creator'),
 ('T00416','01.06.07','poter'),('T00722','01.11.01','Son l’Ariete'),
 ('T00842','01.12.05','loro'),('T01033','01.14.07','temperato'),
 ('T01384','01.19.03','singolare'),('T01541','01.21.08','Che ’l vero'),
 ('T01729','01.23.08','ragion'),('T02062','01.28.05','tal eclissi'),
 ('T02232','01.31.02','ciel'),('T02554','01.36.08','Ringrazierà')]
FROZEN_RIGHT=['Fn12','He1','He2','NYPL2','Spe','Par4','Barb4','Cap1','Urb2','Vat3']
MANIFESTS={
 'Laur2':'https://tecabml.contentdm.oclc.org/iiif/info/plutei/716054/manifest.json',
 'Yale4':'https://collections.library.yale.edu/manifests/2026466?manifest=https://collections.library.yale.edu/manifests/2026466',
 'Barb4':'https://digi.vatlib.it/iiif/MSS_Barb.lat.4048/manifest.json',
 'Urb2':'https://digi.vatlib.it/iiif/MSS_Urb.lat.1754/manifest.json',
 'Vat3':'https://digi.vatlib.it/iiif/MSS_Vat.lat.6802/manifest.json',
 'Spe':'https://purl.stanford.edu/jq331pk9591/iiif/manifest'}
# standard layout: first folio, ottave/side, evidence label
STD={
 'Laur2':(81,4,'project_line_map_start_plus_32line_continuity'),
 'Yale4':(25,3,'project_metadata_start_plus_24line_continuity'),
 'Barb4':(2,3,'project_line_map_start_plus_24line_continuity'),
 'Urb2':(99,3,'project_line_map_start_plus_24line_continuity'),
 'Vat3':(1,3,'project_line_map_1r_start_plus_24line_continuity')}
# Only project-explicit Spe pages; no inference for earlier loci.
SPE={'T01729':'10','T02062':'12','T02232':'13','T02554':'14'}

def normlabel(x):
    if isinstance(x,dict):
        vals=[]
        for v in x.values(): vals += v if isinstance(v,list) else [v]
        x=' '.join(map(str,vals))
    return re.sub(r'[^a-z0-9]+','',str(x).lower())

def page_for(sig,vid,code):
    if sig=='Spe':
        f=SPE.get(vid); return (f,'project_explicit_line_map') if f else (None,'NO_SAFE_PAGE_MAP')
    b,s,l=map(int,code.split('.')); first,k,method=STD[sig]
    side=(s-1)//k; fol=first+side//2; face='r' if side%2==0 else 'v'
    return f'{fol}{face}',method

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
    prereg={'loci':[{'VariantID':v,'line_code':c,'base_segment':t} for v,c,t in LOCI],
            'selection_rule':'12 approximately equally spaced loci from the ordered 78-item public Ba VariantID list, frozen before target readings',
            'visual_pair':['Laur2','Yale4'],'visual_right':FROZEN_RIGHT,
            'packet_witnesses':list(MANIFESTS),
            'omitted_not_substituted':{'Par4':'Gallica 403 from runner','NYPL2':'no IIIF URL','Fn12':'no safe Book-I line map yet','He1':'no safe Book-I line map yet','He2':'no safe Book-I line map yet','Cap1':'canvas/page-map mismatch found in packet 1'}}
    (OUT/'PREREG_FULLPAGES.json').write_text(json.dumps(prereg,indent=2,ensure_ascii=False),encoding='utf8')
    idx=[]
    for sig,murl in MANIFESTS.items():
        print('manifest',sig,flush=True)
        mr=requests.get(murl,timeout=90); mr.raise_for_status(); cs=canvases(mr.json())
        sd=OUT/sig; sd.mkdir(exist_ok=True)
        cache={}
        for vid,code,seg in LOCI:
            fol,method=page_for(sig,vid,code)
            if not fol:
                idx.append({'siglum':sig,'VariantID':vid,'line_code':code,'base_segment':seg,'status':'NO_SAFE_PAGE_MAP','map_method':method}); continue
            ch=choose(cs,fol)
            if not ch:
                idx.append({'siglum':sig,'VariantID':vid,'line_code':code,'base_segment':seg,'folio':fol,'status':'CANVAS_NOT_FOUND','map_method':method}); continue
            _,_,_,lab,sid,iid=ch; key=(fol,str(lab)); fn=f'{re.sub(r"[^A-Za-z0-9_-]+","_",fol)}.jpg'
            fp=sd/fn
            if key not in cache:
                try:
                    rr=requests.get(imgurl(sid,iid),timeout=120); rr.raise_for_status()
                    im=Image.open(io.BytesIO(rr.content)).convert('RGB'); im.save(fp,quality=91)
                    cache[key]=str(fp)
                except Exception as e:
                    cache[key]=None
                    idx.append({'siglum':sig,'VariantID':vid,'line_code':code,'base_segment':seg,'folio':fol,'status':'IMAGE_ERROR','map_method':method,'canvas_label':str(lab),'error':repr(e)}); continue
            if cache[key]:
                idx.append({'siglum':sig,'VariantID':vid,'line_code':code,'base_segment':seg,'folio':fol,'status':'OK','map_method':method,'canvas_label':str(lab),'file':cache[key]})
    fields=sorted({k for r in idx for k in r})
    with (OUT/'index_fullpages.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(idx)
    print(Counter(r['status'] for r in idx),flush=True)
if __name__=='__main__': main()
