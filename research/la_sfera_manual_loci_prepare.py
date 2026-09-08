#!/usr/bin/env python3
"""Prepare frozen La Sfera manual-collation crops.

The public La Sfera textual-variant export currently contains readings only for
Ba. To avoid post-hoc cherry-picking, this packet freezes 12 loci sampled at
roughly equal intervals from those 78 pre-existing VariantIDs BEFORE reading
any target witness.

Frozen visual selections remain unchanged:
 pair = Laur2, Yale4
 right = Fn12, He1, He2, NYPL2, Spe, Par4, Barb4, Cap1, Urb2, Vat3

Packet 1 uses only witnesses whose Book-I pagination is safely determined from
project metadata: Laur2 (32 lines / 4 ottave per side from 81r), Yale4,
Par4, Barb4, Cap1 and Urb2 (24 lines / 3 ottave per side from their stated
starts). No omitted witness is replaced by another manuscript.
"""
from __future__ import annotations
import csv, io, json, re, sys
from pathlib import Path
from collections import Counter
import requests
from PIL import Image, ImageDraw

OUT=Path(sys.argv[1] if len(sys.argv)>1 else 'manual_loci_out'); OUT.mkdir(parents=True,exist_ok=True)
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
 'Par4':'https://gallica.bnf.fr/iiif/ark:/12148/btv1b55013450m/manifest.json',
 'Barb4':'https://digi.vatlib.it/iiif/MSS_Barb.lat.4048/manifest.json',
 'Cap1':'https://digi.vatlib.it/iiif/MSS_Cappon.56/manifest.json',
 'Urb2':'https://digi.vatlib.it/iiif/MSS_Urb.lat.1754/manifest.json'}
# (first folio number, ottave per side, mapping status)
LAYOUT={'Laur2':(81,4,'project_metadata_32line_standard'),
        'Yale4':(25,3,'project_metadata_24line_standard'),
        'Par4':(55,3,'project_line_map_24line_standard'),
        'Barb4':(2,3,'project_line_map_24line_standard'),
        'Cap1':(1,3,'project_metadata_24line_standard'),
        'Urb2':(99,3,'project_line_map_24line_standard')}
FROZEN_RIGHT=['Fn12','He1','He2','NYPL2','Spe','Par4','Barb4','Cap1','Urb2','Vat3']

def normlabel(x):
    if isinstance(x,dict):
        vals=[]
        for v in x.values(): vals += v if isinstance(v,list) else [v]
        x=' '.join(map(str,vals))
    return re.sub(r'[^a-z0-9]+','',str(x).lower())

def target_page(sig,code):
    b,s,l=map(int,code.split('.')); assert b==1
    first,k,method=LAYOUT[sig]
    side=(s-1)//k; fol=first+side//2; face='r' if side%2==0 else 'v'
    line_in=((s-1)%k)*8+(l-1); span=k*8
    return f'{fol}{face}',line_in,span,method

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

def choose_canvas(cs,folio):
    f=normlabel(folio); cand=[]
    for i,(lab,sid,iid) in enumerate(cs):
        nl=normlabel(lab); score=0
        if nl==f: score=5
        elif nl.endswith(f): score=4
        elif f in nl: score=3
        if score: cand.append((score,-len(nl),-i,lab,sid,iid))
    return max(cand) if cand else None

def image_url(sid,iid): return sid.rstrip('/')+'/full/1800,/0/default.jpg' if sid else iid

def crop_band(im,offset,span):
    W,H=im.size; frac=(offset+.5)/span
    yc=int(H*(.13+.74*frac)); hh=max(220,int(H*.14)); y0=max(0,yc-hh//2); y1=min(H,yc+hh//2)
    return im.crop((0,y0,W,y1))

def main():
    prereg={'selection_rule':'12 approximately equally spaced loci from ordered 78-item public Ba VariantID list; frozen before target readings',
            'loci':[{'VariantID':v,'line_code':c,'base_segment':t} for v,c,t in LOCI],
            'visual_pair':['Laur2','Yale4'],'visual_right':FROZEN_RIGHT,
            'packet_witnesses':list(MANIFESTS),'omitted_not_substituted':[x for x in FROZEN_RIGHT if x not in MANIFESTS]}
    (OUT/'PREREG.json').write_text(json.dumps(prereg,indent=2,ensure_ascii=False),encoding='utf8')
    idx=[]
    for sig,url in MANIFESTS.items():
        print('MANIFEST',sig,flush=True)
        r=requests.get(url,timeout=90); r.raise_for_status(); man=r.json(); cs=canvases(man)
        (OUT/f'{sig}_canvas_labels.json').write_text(json.dumps([str(x[0]) for x in cs],indent=2,ensure_ascii=False),encoding='utf8')
        sd=OUT/sig; sd.mkdir(exist_ok=True)
        for vid,code,seg in LOCI:
            fol,off,span,method=target_page(sig,code); ch=choose_canvas(cs,fol)
            if not ch:
                idx.append({'siglum':sig,'VariantID':vid,'line_code':code,'folio':fol,'status':'CANVAS_NOT_FOUND','map_method':method}); continue
            _,_,_,lab,sid,iid=ch; u=image_url(sid,iid)
            try:
                rr=requests.get(u,timeout=90); rr.raise_for_status(); im=Image.open(io.BytesIO(rr.content)).convert('RGB')
                cr=crop_band(im,off,span); head=Image.new('RGB',(cr.width,76),'white'); d=ImageDraw.Draw(head)
                d.text((10,8),f'{sig} | {vid} | {code} | fol {fol} | {method} | base: {seg}',fill='black')
                out=Image.new('RGB',(cr.width,cr.height+76),'white'); out.paste(head,(0,0)); out.paste(cr,(0,76))
                fn=f'{vid}_{code.replace(".","-")}_{fol}.jpg'; out.save(sd/fn,quality=92)
                idx.append({'siglum':sig,'VariantID':vid,'line_code':code,'folio':fol,'status':'OK','map_method':method,'canvas_label':str(lab),'file':str(sd/fn)})
            except Exception as e:
                idx.append({'siglum':sig,'VariantID':vid,'line_code':code,'folio':fol,'status':'IMAGE_ERROR','map_method':method,'error':repr(e),'image_url':u})
    fields=sorted({k for r in idx for k in r})
    with (OUT/'index.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(idx)
    print('STATUS',Counter(r['status'] for r in idx),flush=True)
if __name__=='__main__': main()
