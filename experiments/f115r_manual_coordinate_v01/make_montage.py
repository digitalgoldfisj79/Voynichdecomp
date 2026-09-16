#!/usr/bin/env python3
from pathlib import Path
import sys, json, cv2, numpy as np, requests

ROOT=Path(__file__).resolve().parents[2]
V01=ROOT/'experiments'/'f115r_local_hand_boundary_v01'
sys.path.insert(0,str(V01))
import run_assay as v1

OUT=Path(__file__).resolve().parent/'results'
OUT.mkdir(parents=True,exist_ok=True)
imgp=OUT/'source.jpg'
MANIFEST='https://collections.library.yale.edu/manifests/oid/2002046'


def label_text(obj):
    lab=obj.get('label',{})
    if isinstance(lab,str): return lab
    if isinstance(lab,dict):
        vals=[]
        for v in lab.values():
            if isinstance(v,list): vals += [str(x) for x in v]
            else: vals.append(str(v))
        return ' '.join(vals)
    return str(lab)


def body_from_canvas(c):
    try:
        body=c['items'][0]['items'][0]['body']
        if isinstance(body,dict) and body.get('type')=='Choice':
            body=body.get('default') or body.get('items',[{}])[0]
        return body
    except Exception:
        return None


def yale_image_url():
    r=requests.get(MANIFEST,headers={'User-Agent':'VoynichResearch/1.0'},timeout=45)
    r.raise_for_status(); m=r.json()
    canvases=m.get('items',[]) or m.get('sequences',[{}])[0].get('canvases',[])
    chosen=None
    for c in canvases:
        txt=label_text(c)+' '+json.dumps(c.get('metadata',{}),ensure_ascii=False)+' '+c.get('id','')
        if '115r' in txt.lower() or '1006274' in json.dumps(c):
            chosen=c; break
    if chosen is None:
        raise RuntimeError('Could not locate f115r in Yale IIIF manifest')
    body=body_from_canvas(chosen)
    if not body:
        # IIIF v2 fallback
        try: body=chosen['images'][0]['resource']
        except Exception: raise RuntimeError('Could not resolve image body for f115r')
    url=body.get('id') or body.get('@id')
    service=body.get('service')
    if isinstance(service,list): service=service[0] if service else None
    if service:
        sid=service.get('id') or service.get('@id')
        if sid: url=sid.rstrip('/')+'/full/full/0/default.jpg'
    if not url: raise RuntimeError('No image URL for f115r')
    (OUT/'yale_source.txt').write_text('canvas='+str(chosen.get('id') or chosen.get('@id'))+'\nlabel='+label_text(chosen)+'\nimage='+url+'\n')
    return url

url=yale_image_url()
r=requests.get(url,headers={'User-Agent':'VoynichResearch/1.0'},timeout=90)
r.raise_for_status(); imgp.write_bytes(r.content)
img=cv2.imread(str(imgp));
if img is None: raise RuntimeError('Downloaded Yale image could not be decoded')
H,W=img.shape[:2]
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); mask=v1.adaptive_ink(gray)
x0,x1=int(.105*W),int(.925*W); y0,y1=int(.038*H),int(.865*H)
centers,edges,_=v1.detect_lines(mask,x0,x1,y0,y1)
font=cv2.FONT_HERSHEY_SIMPLEX
rows=[]
for ln in range(7,25):
    ya=max(0,int(edges[ln-1])-8); yb=min(H,int(edges[ln])+8)
    crop=img[ya:yb,x0:x1].copy()
    label=np.full((crop.shape[0],100,3),255,np.uint8)
    cv2.putText(label,f'L{ln:02d}',(8,max(28,crop.shape[0]//2)),font,0.8,(0,0,0),2,cv2.LINE_AA)
    rows.append(np.hstack([label,crop]))
width=max(r.shape[1] for r in rows)
padded=[]
for rr in rows:
    if rr.shape[1]<width:
        rr=np.hstack([rr,np.full((rr.shape[0],width-rr.shape[1],3),255,np.uint8)])
    padded.append(rr)
sep=np.full((3,width,3),220,np.uint8)
mont=padded[0]
for rr in padded[1:]: mont=np.vstack([mont,sep,rr])
cv2.imwrite(str(OUT/'f115r_lines_07_24.png'),mont,[cv2.IMWRITE_PNG_COMPRESSION,6])
(OUT/'line_geometry.txt').write_text('\n'.join(f'{i+1}\tcenter={centers[i]}\tedge0={edges[i]}\tedge1={edges[i+1]}' for i in range(45)))
imgp.unlink(missing_ok=True)
print('WROTE',OUT/'f115r_lines_07_24.png',mont.shape,'FROM',url)
