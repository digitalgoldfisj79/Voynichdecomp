#!/usr/bin/env python3
"""Transport-only extraction for the two Archive.org-blocked locked witnesses.

Scientific feature extraction is imported unchanged from taccola_calibration_v01b.
- pal766: same 102 r15 page order, same frozen midpoint positions; pages delivered by
  Internet Archive's 6.47 MB PDF derivative and rendered to 900 px width.
- ljs419: same 205 r15 page order, same frozen midpoint positions; pages delivered by
  OPenn direct web JPEGs already stored in ManuComp r15.
No similarity score is available or consulted here.
"""
import argparse, io, json, pickle
from pathlib import Path
import numpy as np
from PIL import Image
import requests
import fitz
import taccola_calibration_v01b as core

ap=argparse.ArgumentParser(); ap.add_argument('--id',required=True,choices=['pal766','ljs419']); args=ap.parse_args()
mid=args.id
outdir=Path('taccola_one_output'); outdir.mkdir(exist_ok=True)
panel_hash=core.sha256_json(core.LOCKED_PANEL)
if panel_hash != core.EXPECTED_PANEL_SHA256: raise RuntimeError('panel mismatch')

N={'pal766':102,'ljs419':205}[mid]
positions=[int((i+0.5)*N/core.PAGE_SAMPLE) for i in range(core.PAGE_SAMPLE)]
positions=[min(N-1,max(0,p)) for p in positions]
# exact midpoint rule must yield 60 unique rows for these N.
if len(set(positions)) != 60: raise RuntimeError('unexpected midpoint duplicate')
errors=[]; got=[]; diag=[]; manifest_rows=[]

def add_feature(idx,label,gray):
    f=core.page_features(gray)
    if f is None: raise ValueError('feature extraction returned None')
    f['index']=idx; f['label']=label; got.append(f)
    diag.append({'index':idx,'label':label,'illustration_score':f['illustration_score'],'inkfrac':f['inkfrac'],'bboxfrac':f['bboxfrac'],'components':f['component_count']})

if mid=='pal766':
    url='https://archive.org/download/palatino-766-images/Palatino%20766.pdf'
    r=requests.get(url,headers={'User-Agent':core.USER_AGENT},timeout=90); r.raise_for_status()
    doc=fitz.open(stream=r.content,filetype='pdf')
    if doc.page_count != 102: raise RuntimeError(f'Palatino PDF page count {doc.page_count}, expected 102')
    for idx in positions:
        label=str(idx); manifest_rows.append({'index':idx,'label':label,'transport':'ia_pdf_derivative'})
        try:
            page=doc.load_page(idx)
            rect=page.rect; zoom=core.IMAGE_WIDTH/max(1.0,rect.width)
            pix=page.get_pixmap(matrix=fitz.Matrix(zoom,zoom),colorspace=fitz.csGRAY,alpha=False)
            gray=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width)
            add_feature(idx,label,gray)
        except Exception as e:
            errors.append({'stage':'image','id':mid,'index':idx,'error':repr(e)})
else:
    s=requests.Session(); h={'User-Agent':core.USER_AGENT}
    for idx in positions:
        label=f'0265_{idx:04d}_web.jpg'
        url=f'https://openn.library.upenn.edu/Data/0001/ljs419/data/web/{label}'
        manifest_rows.append({'index':idx,'label':label,'url':url,'transport':'openn_r15_direct'})
        try:
            r=s.get(url,headers=h,timeout=20); r.raise_for_status()
            im=Image.open(io.BytesIO(r.content)).convert('L')
            gray=np.asarray(im,dtype=np.uint8)
            if gray.shape[1] > core.IMAGE_WIDTH:
                hh=max(1,int(round(gray.shape[0]*core.IMAGE_WIDTH/gray.shape[1])))
                gray=core.cv2.resize(gray,(core.IMAGE_WIDTH,hh),interpolation=core.cv2.INTER_AREA)
            add_feature(idx,label,gray)
        except Exception as e:
            errors.append({'stage':'image','id':mid,'index':idx,'error':repr(e)})

got.sort(key=lambda x:(-x['illustration_score'],x['index'])); got=got[:core.ILLUSTRATION_TOP_K]
bundle={'panel_sha256':panel_hash,'id':mid,'features':got,'page_diag':sorted(diag,key=lambda x:x['index']),'manifest_rows':manifest_rows,'errors':errors,'transport_repair':{'pal766':'Internet Archive PDF derivative, same 102 r15 page order and frozen midpoint indices','ljs419':'OPenn direct r15 JPEGs, same 205 r15 page order and frozen midpoint indices'}[mid]}
with open(outdir/f'one_{mid}.pkl','wb') as f: pickle.dump(bundle,f,protocol=5)
(outdir/f'one_{mid}.json').write_text(json.dumps({'id':mid,'download_ok':len(diag),'selected':len(got),'positions':positions,'errors':errors,'transport_repair':bundle['transport_repair']},indent=2))
print(json.dumps({'event':'features','id':mid,'download_ok':len(diag),'selected':len(got),'errors_total':len(errors),'positions':positions}),flush=True)
if len(got)<12: raise SystemExit(f'{mid}: fewer than 12 selected features')
