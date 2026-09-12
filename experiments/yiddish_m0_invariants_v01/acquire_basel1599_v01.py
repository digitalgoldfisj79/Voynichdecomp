from __future__ import annotations
import hashlib, json
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO

MANIFEST='https://www.e-rara.ch/i3f/v20/2611258/manifest'
OUT=Path('experiments/yiddish_m0_invariants_v01/basel1599_acquisition')
OUT.mkdir(parents=True,exist_ok=True)

def sha256(b:bytes)->str:return hashlib.sha256(b).hexdigest()

m_bytes=requests.get(MANIFEST,timeout=60).content
m=json.loads(m_bytes)
(OUT/'manifest.json').write_bytes(m_bytes)
rows=[]
canvases=m['sequences'][0]['canvases']
for idx,c in enumerate(canvases,1):
    label=str(c.get('label',idx))
    svc=c['images'][0]['resource']['service']['@id'].rstrip('/')
    url=svc+'/full/full/0/default.jpg'
    r=requests.get(url,timeout=120); r.raise_for_status(); raw=r.content
    im=Image.open(BytesIO(raw)).convert('L')
    ow,oh=im.size
    nw=3600; nh=round(oh*nw/ow)
    im=im.resize((nw,nh),Image.Resampling.LANCZOS)
    if nh>5400:
        top=(nh-5400)//2; im=im.crop((0,top,3600,top+5400))
    elif nh<5400:
        canvas=Image.new('L',(3600,5400),255); top=(5400-nh)//2; canvas.paste(im,(0,top)); im=canvas
    canon=OUT/f'canvas_{idx:02d}.jpg'
    im.save(canon,quality=92,optimize=True)
    cb=canon.read_bytes()
    rows.append({'index':idx,'label':label,'canvas_id':c.get('@id'),'image_url':url,'source_bytes':len(raw),'source_sha256':sha256(raw),'original_size':[ow,oh],'canonical_file':canon.name,'canonical_bytes':len(cb),'canonical_sha256':sha256(cb)})
    print(idx,label,ow,oh,len(raw),rows[-1]['source_sha256'][:12])
(OUT/'acquisition.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print('COUNT',len(rows))
