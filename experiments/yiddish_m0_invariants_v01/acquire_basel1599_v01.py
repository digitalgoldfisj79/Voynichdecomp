from __future__ import annotations
import hashlib, json
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO

# Exact ordered image IDs from frozen e-rara IIIF manifest 2611258.
IMAGE_IDS=[2611260,2611261,2611262,2611263,2611264,2611265,2611266,2611267,2611269,2611270,2611271,2611272,2611273,2611274,2611275,2611276,2611277,2611278,2611279,2611280,2611281,2611282,2611283,2611284,2611285,2611286,2611287,2611288,2611289,2611290,2611291,2611292,2611293,2611294,2611295,2611296,2611297,2611298,2611299,2611300]
OUT=Path('experiments/yiddish_m0_invariants_v01/basel1599_acquisition')
OUT.mkdir(parents=True,exist_ok=True)
HEADERS={'User-Agent':'Mozilla/5.0 (compatible; Voynichdecomp research acquisition/1.0)'}

def sha256(b:bytes)->str:return hashlib.sha256(b).hexdigest()

rows=[]
for idx,image_id in enumerate(IMAGE_IDS,1):
    # e-rara documented direct IIIF image endpoint (v21 transport only).
    url=f'https://www.e-rara.ch/i3f/v21/{image_id}/full/0/0/default.jpg'
    r=requests.get(url,headers=HEADERS,timeout=120); r.raise_for_status(); raw=r.content
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
    rows.append({'index':idx,'label':f'[{idx}]','image_id':image_id,'image_url':url,'source_bytes':len(raw),'source_sha256':sha256(raw),'original_size':[ow,oh],'canonical_file':canon.name,'canonical_bytes':len(cb),'canonical_sha256':sha256(cb)})
    print(idx,image_id,ow,oh,len(raw),rows[-1]['source_sha256'][:12])
(OUT/'acquisition.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
(OUT/'frozen_image_ids.json').write_text(json.dumps(IMAGE_IDS,indent=2),encoding='utf-8')
print('COUNT',len(rows))
