#!/usr/bin/env python3
import argparse, base64, io, json
from pathlib import Path
import cv2, numpy as np, requests
from PIL import Image

URL='https://collections.library.yale.edu/iiif/2/1006076/full/2400,/0/default.jpg'
# Coordinates measured on the authentic Yale f1r scan after reference QA.
# &253/STA Yy: lower red V/U form plus vertical squiggle.
TARGET=(400,1060,540,1260)
# &252/STA Yx: adjacent upper red bird/V form without the squiggle.
CONTROL=(385,770,590,990)

def get_page():
    r=requests.get(URL,timeout=(10,60),headers={'User-Agent':'ManuComp-SeagullQuery/0.1 (+research)'})
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert('RGB')

def pigment_mask(crop):
    a=np.array(crop)
    hsv=cv2.cvtColor(a,cv2.COLOR_RGB2HSV)
    h,s,v=cv2.split(hsv)
    # Empirically calibrated on the f1r red pigment itself; avoids parchment and green wash.
    m=((h<22)&(s>105)&(v<155)).astype(np.uint8)*255
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,lab,stats,_=cv2.connectedComponentsWithStats((m>0).astype(np.uint8),8)
    clean=np.zeros_like(m)
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]>=6:
            clean[lab==i]=255
    ys,xs=np.where(clean>0)
    if not len(xs): raise RuntimeError('empty pigment mask')
    # Preserve relative geometry of disconnected squiggle/base; only trim outer whitespace.
    pad=4
    x0=max(0,xs.min()-pad); x1=min(clean.shape[1],xs.max()+pad+1)
    y0=max(0,ys.min()-pad); y1=min(clean.shape[0],ys.max()+pad+1)
    return clean[y0:y1,x0:x1]

def save_query(out,name,page,box):
    crop=page.crop(box)
    crop.save(out/f'{name}_crop.jpg',quality=95)
    m=pigment_mask(crop)
    Image.fromarray(m).save(out/f'{name}_mask.png')
    b=io.BytesIO(); Image.fromarray(m).save(b,format='PNG')
    (out/f'{name}.b64').write_text(base64.b64encode(b.getvalue()).decode('ascii'))
    return {'name':name,'box':box,'mask_width':m.shape[1],'mask_height':m.shape[0],'foreground_px':int((m>0).sum())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--outdir',default='seagull_queries'); args=ap.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    p=get_page()
    meta=[save_query(out,'seagull_target',p,TARGET),save_query(out,'seagull_control',p,CONTROL)]
    payload={'source':URL,'source_size':p.size,'target':'EVA &253 / STA Yy','control':'EVA &252 / STA Yx','queries':meta}
    (out/'metadata.json').write_text(json.dumps(payload,indent=2)); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
