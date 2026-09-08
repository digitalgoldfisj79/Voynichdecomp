#!/usr/bin/env python3
"""Prepare a preregistered manual-collation packet for La Sfera.

Purpose: the public textual-variant export currently contains only Ba, so we
cannot infer textual families computationally. This script freezes 12 loci
WITHOUT inspecting the readings of the visual-selected witnesses, then pulls
IIIF page images and crops broad bands around each locus for manual diplomatic
transcription.

Frozen visual sets from the f68r experiment are not changed:
  pair: Laur2, Yale4
  right: Fn12, He1, He2, NYPL2, Spe, Par4, Barb4, Cap1, Urb2, Vat3

This first manual packet uses witnesses for which Book I foliation is either
explicitly line-mapped by the project (Laur2, Par4, Barb4, Urb2, parts of Spe)
or is a complete 24-folio/24-line witness with a directly checkable standard
3-stanza-per-side Book-I layout (Yale4, Cap1). No missing witness is replaced.
The rest remain explicitly uncollated in this packet.
"""
from __future__ import annotations
import csv, io, json, math, os, re, sys
from pathlib import Path
import requests
from PIL import Image, ImageOps, ImageDraw

OUT=Path(sys.argv[1] if len(sys.argv)>1 else 'manual_loci_out')
OUT.mkdir(parents=True, exist_ok=True)

# 12 evenly spaced loci from the *pre-existing* 78 Ba variant IDs, ordered by
# their position in the project export. Selection was made before inspecting
# any target-witness reading.
LOCI=[
 ('T00031','01.01.02','secol'),
 ('T00251','01.04.07','O creator'),
 ('T00416','01.06.07','poter'),
 ('T00722','01.11.01','Son l’Ariete'),
 ('T00842','01.12.05','loro'),
 ('T01033','01.14.07','temperato'),
 ('T01384','01.19.03','singolare'),
 ('T01541','01.21.08','Che ’l vero'),
 ('T01729','01.23.08','ragion'),
 ('T02062','01.28.05','tal eclissi'),
 ('T02232','01.31.02','ciel'),
 ('T02554','01.36.08','Ringrazierà'),
]

MANIFESTS={
 'Laur2':'https://tecabml.contentdm.oclc.org/iiif/info/plutei/716054/manifest.json',
 'Yale4':'https://collections.library.yale.edu/manifests/2026466?manifest=https://collections.library.yale.edu/manifests/2026466',
 'Spe':'https://purl.stanford.edu/jq331pk9591/iiif/manifest',
 'Par4':'https://gallica.bnf.fr/iiif/ark:/12148/btv1b55013450m/manifest.json',
 'Barb4':'https://digi.vatlib.it/iiif/MSS_Barb.lat.4048/manifest.json',
 'Cap1':'https://digi.vatlib.it/iiif/MSS_Cappon.56/manifest.json',
 'Urb2':'https://digi.vatlib.it/iiif/MSS_Urb.lat.1754/manifest.json',
}
# Explicitly omitted here, not substituted: Fn12, He1, He2, NYPL2, Vat3.

FOLIOS_URL='https://sferaproject.org/data/folioscsv'


def normlabel(x):
    if isinstance(x,dict):
        vals=[]
        for v in x.values():
            vals += v if isinstance(v,list) else [v]
        x=' '.join(map(str,vals))
    return re.sub(r'[^a-z0-9]+','',str(x).lower())

def ordinal(code):
    b,s,l=map(int,code.split('.'))
    return ((b-1)*36+(s-1))*8+l

def inferred_folio_24(start_folio, code):
    # Book I standard layout: 3 ottave (=24 verse lines) per side.
    _,s,l=map(int,code.split('.'))
    side=(s-1)//3
    fol=start_folio+side//2
    face='r' if side%2==0 else 'v'
    line_in=((s-1)%3)*8+(l-1)
    return f'{fol}{face}', line_in, 24, 'inferred_standard_24line'

def fetch_csv(url):
    r=requests.get(url,timeout=60); r.raise_for_status()
    return list(csv.DictReader(io.StringIO(r.text)))

def explicit_page(rows,sig,code):
    o=ordinal(code)
    rr=[r for r in rows if r.get('manuscript')==sig and r.get('line_code_starts')]
    best=None
    for r in rr:
        try: a=ordinal(r['line_code_starts'])
        except: continue
        n=r.get('next_start_line')
        try: z=ordinal(n) if n and n!='-' else a+32
        except: z=a+32
        if a<=o<z:
            return r['folio'], o-a, max(1,z-a), 'project_line_map'
    return None

def canvases(man):
    out=[]
    if 'sequences' in man:
        for c in man['sequences'][0].get('canvases',[]):
            label=c.get('label','')
            im=(c.get('images') or [{}])[0].get('resource',{})
            svc=im.get('service')
            if isinstance(svc,list): svc=svc[0] if svc else None
            sid=(svc or {}).get('@id') or (svc or {}).get('id')
            iid=im.get('@id') or im.get('id')
            out.append((label,sid,iid))
    else:
        for c in man.get('items',[]):
            label=c.get('label','')
            try: body=c['items'][0]['items'][0]['body']
            except Exception: body={}
            svcs=body.get('service') or []
            svc=svcs[0] if isinstance(svcs,list) and svcs else (svcs if isinstance(svcs,dict) else {})
            sid=svc.get('id') or svc.get('@id')
            iid=body.get('id') or body.get('@id')
            out.append((label,sid,iid))
    return out

def choose_canvas(cs,folio):
    f=normlabel(folio)
    # exact-ish folio token match, prefer labels ending in token.
    scored=[]
    for i,(lab,sid,iid) in enumerate(cs):
        nl=normlabel(lab)
        score=0
        if nl==f: score=5
        if nl.endswith(f): score=max(score,4)
        if f in nl: score=max(score,3)
        if score: scored.append((score,-len(nl),i,lab,sid,iid))
    if not scored: return None
    return max(scored)

def image_url(sid,iid):
    if sid:
        return sid.rstrip('/')+'/full/1800,/0/default.jpg'
    return iid

def crop_band(img, offset, span):
    # Broad band; avoid pretending line geometry is exact. Keep enough context
    # to identify neighboring verse lines manually.
    W,H=img.size
    frac=(offset+0.5)/max(span,1)
    yc=int(H*(0.13+0.74*frac))
    hh=max(170,int(H*0.11))
    y0=max(0,yc-hh//2); y1=min(H,yc+hh//2)
    return img.crop((0,y0,W,y1))

def main():
    rows=fetch_csv(FOLIOS_URL)
    prereg={'loci':[{'VariantID':v,'line_code':c,'base_segment':t} for v,c,t in LOCI],
            'selection_rule':'12 evenly spaced positions from ordered 78-item public Ba variant list; frozen before target-witness reading inspection',
            'visual_pair':['Laur2','Yale4'],
            'visual_right':['Fn12','He1','He2','NYPL2','Spe','Par4','Barb4','Cap1','Urb2','Vat3'],
            'packet_witnesses':list(MANIFESTS),
            'omitted_not_substituted':['Fn12','He1','He2','NYPL2','Vat3']}
    (OUT/'PREREG.json').write_text(json.dumps(prereg,indent=2,ensure_ascii=False),encoding='utf8')
    index=[]
    for sig,url in MANIFESTS.items():
        print('MANIFEST',sig,flush=True)
        man=requests.get(url,timeout=90).json(); cs=canvases(man)
        (OUT/f'{sig}_canvas_labels.json').write_text(json.dumps([str(x[0]) for x in cs],indent=2,ensure_ascii=False),encoding='utf8')
        sdir=OUT/sig; sdir.mkdir(exist_ok=True)
        for vid,code,seg in LOCI:
            page=explicit_page(rows,sig,code)
            if page is None and sig=='Yale4': page=inferred_folio_24(25,code)
            if page is None and sig=='Cap1': page=inferred_folio_24(1,code)
            # For Spe only collate loci covered by explicit project line map.
            if page is None:
                index.append({'siglum':sig,'VariantID':vid,'line_code':code,'status':'NO_SAFE_PAGE_MAP'})
                continue
            fol,off,span,method=page
            ch=choose_canvas(cs,fol)
            if not ch:
                index.append({'siglum':sig,'VariantID':vid,'line_code':code,'folio':fol,'status':'CANVAS_NOT_FOUND','map_method':method})
                continue
            _,_,ci,lab,sid,iid=ch
            u=image_url(sid,iid)
            try:
                r=requests.get(u,timeout=90); r.raise_for_status()
                im=Image.open(io.BytesIO(r.content)).convert('RGB')
                crop=crop_band(im,off,span)
                # Header in blank border, not over manuscript pixels.
                header=Image.new('RGB',(crop.width,70),'white')
                d=ImageDraw.Draw(header); d.text((10,8),f'{sig} | {vid} | {code} | fol {fol} | {method} | base: {seg}',fill='black')
                out=Image.new('RGB',(crop.width,crop.height+70),'white'); out.paste(header,(0,0)); out.paste(crop,(0,70))
                fn=f'{vid}_{code.replace(".","-")}_{fol}.jpg'
                out.save(sdir/fn,quality=90)
                index.append({'siglum':sig,'VariantID':vid,'line_code':code,'folio':fol,'status':'OK','map_method':method,'canvas_label':str(lab),'image_url':u,'file':str(sdir/fn)})
            except Exception as e:
                index.append({'siglum':sig,'VariantID':vid,'line_code':code,'folio':fol,'status':'IMAGE_ERROR','map_method':method,'error':repr(e),'image_url':u})
    with (OUT/'index.csv').open('w',newline='',encoding='utf8') as f:
        fields=sorted({k for x in index for k in x})
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(index)
    print('STATUS COUNTS')
    from collections import Counter
    print(Counter(x['status'] for x in index))

if __name__=='__main__': main()
