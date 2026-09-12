from __future__ import annotations
import hashlib, json, math, shutil
from pathlib import Path
from io import BytesIO
import requests
import cv2
import numpy as np
from PIL import Image
import anon_word_v02 as aw

OBJECT='bsb10170884'
MANIFEST=f'https://api.digitale-sammlungen.de/iiif/presentation/v2/{OBJECT}/manifest'
OUT=Path('experiments/yiddish_m0_anon_word_v02/shmuel1544_census')
TMP=OUT/'_tmp'
OUT.mkdir(parents=True,exist_ok=True); TMP.mkdir(parents=True,exist_ok=True)
SESSION=requests.Session(); SESSION.headers.update({'User-Agent':'Voynichdecomp-Yiddish-M0-v02/1.0'})

def sha256(b:bytes)->str:return hashlib.sha256(b).hexdigest()

def get(url,timeout=120):
    r=SESSION.get(url,timeout=timeout); r.raise_for_status(); return r

def service_id(canvas):
    s=canvas['images'][0]['resource']['service']
    if isinstance(s,list): s=s[0]
    return (s.get('@id') or s.get('id')).rstrip('/')

def canonicalize(raw:bytes):
    im=Image.open(BytesIO(raw)).convert('L')
    ow,oh=im.size
    nw=3600; nh=round(oh*nw/ow)
    im=im.resize((nw,nh),Image.Resampling.LANCZOS)
    if nh>5400:
        top=(nh-5400)//2; im=im.crop((0,top,3600,top+5400))
    elif nh<5400:
        canvas=Image.new('L',(3600,5400),255); top=(5400-nh)//2; canvas.paste(im,(0,top)); im=canvas
    return np.array(im),[ow,oh]

def save_jpg(arr,path):
    Image.fromarray(arr).save(path,quality=92,optimize=True)

m_bytes=get(MANIFEST,60).content
m=json.loads(m_bytes)
manifest_sha=sha256(m_bytes)
(OUT/'manifest.json').write_bytes(m_bytes)
canvases=m['sequences'][0]['canvases']
N=len(canvases)
lo=math.ceil(0.15*N); hi=math.floor(0.85*N)  # one-based inclusive
rows=[]; cumulative=0; long_pages=[]
print('MANIFEST_SHA256',manifest_sha,'N',N,'ELIGIBLE',lo,hi,flush=True)
for idx in range(lo,hi+1):
    c=canvases[idx-1]
    cid=c.get('@id') or c.get('id') or f'canvas-{idx}'
    svc=service_id(c)
    # Width 3600 is the largest useful practical resolution because the frozen
    # canonical representation is exactly 3600 px wide.
    url=svc+'/full/3600,/0/default.jpg'
    r=get(url,180); raw=r.content
    arr,orig=canonicalize(raw)
    H,S,words=aw.page_raw(arr)
    nwords=len(words); nlines=len(set(li for li,_,_ in words))
    text_bearing=(nlines>=10 and nwords>=80)
    row={'index':idx,'canvas_id':cid,'label':str(c.get('label',idx)),'service_id':svc,
         'image_url':url,'source_sha256':sha256(raw),'source_bytes':len(raw),
         'original_size':orig,'retained_lines':nlines,'retained_words':nwords,
         'text_bearing':text_bearing}
    rows.append(row)
    if text_bearing:
        f=TMP/f'canvas_{idx:04d}.jpg'; save_jpg(arr,f)
        if cumulative<4128:
            take=min(nwords,4128-cumulative)
            long_pages.append({'index':idx,'canvas_id':cid,'file':f.name,'page_words':nwords,
                               'take_words':take,'cell_start':cumulative,'cell_end_exclusive':cumulative+take})
            cumulative+=take
    print(idx,nlines,nwords,'T' if text_bearing else '-',cumulative,flush=True)

eligible_text=[r for r in rows if r['text_bearing']]
for r in eligible_text:
    seed=f'shmuel1544-anon-v02-audit|{OBJECT}|{r["canvas_id"]}'.encode()
    r['audit_digest']=hashlib.sha256(seed).hexdigest()
audit=sorted(eligible_text,key=lambda r:r['audit_digest'])[:6]

# Freeze sample metadata before copying/opening audit images downstream.
summary={'object':OBJECT,'manifest_url':MANIFEST,'manifest_sha256':manifest_sha,'canvas_count':N,
         'eligible_one_based':[lo,hi],'text_bearing_count':len(eligible_text),
         'retained_words_all_text_bearing':sum(r['retained_words'] for r in eligible_text),
         'long_cell_available':cumulative>=4128,'long_cell_words':min(cumulative,4128),
         'long_cell_pages':long_pages,'audit_sample':[{'index':r['index'],'canvas_id':r['canvas_id'],
             'label':r['label'],'audit_digest':r['audit_digest']} for r in audit]}
(OUT/'census.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
(OUT/'AUDIT_SAMPLE.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

# Preserve only registered long-cell and audit canonical images in artifact.
keep={p['index'] for p in long_pages}|{r['index'] for r in audit}
sel=OUT/'selected_pages'; sel.mkdir(exist_ok=True)
for idx in sorted(keep):
    src=TMP/f'canvas_{idx:04d}.jpg'
    if not src.exists():
        # audit page is text-bearing so it should have been cached.
        raise RuntimeError(f'missing cached selected page {idx}')
    shutil.copy2(src,sel/src.name)
shutil.rmtree(TMP)
print('SUMMARY',json.dumps(summary,sort_keys=True),flush=True)
if cumulative<4128:
    raise SystemExit('QUANTITY_FAIL: fewer than 4128 retained words in frozen eligible body interval')
