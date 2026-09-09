#!/usr/bin/env python3
"""Recover the later duplicate Cap1 folio labels used by La Sfera Book I.

Vatlib's Cappon.56 IIIF manifest repeats 1r-3v. The frozen collation's generic
canvas selector chose the first (blank/preliminary) occurrences. This script
selects the later duplicate labels only; loci and witness selection are unchanged.
"""
import io, json, re, requests
from pathlib import Path
from PIL import Image
OUT=Path('la_sfera_cap1_fix'); OUT.mkdir(exist_ok=True)
MAN='https://digi.vatlib.it/iiif/MSS_Cappon.56/manifest.json'
FOLIOS=['1r','1v','2v','3r']

def norm(x): return re.sub(r'[^a-z0-9]+','',str(x).lower())
def canvases(man):
    out=[]
    for c in man['sequences'][0].get('canvases',[]):
        im=(c.get('images') or [{}])[0].get('resource',{}); svc=im.get('service') or {}
        if isinstance(svc,list): svc=svc[0] if svc else {}
        out.append((c.get('label',''),svc.get('@id') or svc.get('id'),im.get('@id') or im.get('id')))
    return out
m=requests.get(MAN,timeout=90); m.raise_for_status(); cs=canvases(m.json())
index=[]
for fol in FOLIOS:
    matches=[(i,lab,sid,iid) for i,(lab,sid,iid) in enumerate(cs) if norm(lab)==norm(fol)]
    if not matches: index.append({'folio':fol,'status':'NOT_FOUND'}); continue
    i,lab,sid,iid=matches[-1]
    u=sid.rstrip('/')+'/full/2200,/0/default.jpg' if sid else iid
    r=requests.get(u,timeout=120); r.raise_for_status(); im=Image.open(io.BytesIO(r.content)).convert('RGB')
    fp=OUT/f'{fol}.jpg'; im.save(fp,quality=94)
    index.append({'folio':fol,'status':'OK','canvas_index':i,'canvas_label':str(lab),'n_duplicate_matches':len(matches),'file':str(fp)})
(OUT/'index.json').write_text(json.dumps(index,indent=2),encoding='utf8')
print(index)
