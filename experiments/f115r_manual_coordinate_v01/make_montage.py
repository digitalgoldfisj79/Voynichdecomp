#!/usr/bin/env python3
from pathlib import Path
import sys, cv2, numpy as np

ROOT=Path(__file__).resolve().parents[2]
V01=ROOT/'experiments'/'f115r_local_hand_boundary_v01'
sys.path.insert(0,str(V01))
import run_assay as v1

OUT=Path(__file__).resolve().parent/'results'
OUT.mkdir(parents=True,exist_ok=True)
imgp=OUT/'source.jpg'
v1.download(v1.IMAGE_URL,imgp)
img=cv2.imread(str(imgp)); H,W=img.shape[:2]
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
for r in rows:
    if r.shape[1]<width:
        r=np.hstack([r,np.full((r.shape[0],width-r.shape[1],3),255,np.uint8)])
    padded.append(r)
sep=np.full((3,width,3),220,np.uint8)
mont=padded[0]
for r in padded[1:]: mont=np.vstack([mont,sep,r])
cv2.imwrite(str(OUT/'f115r_lines_07_24.png'),mont,[cv2.IMWRITE_PNG_COMPRESSION,6])
(OUT/'line_geometry.txt').write_text('\n'.join(f'{i+1}\tcenter={centers[i]}\tedge0={edges[i]}\tedge1={edges[i+1]}' for i in range(45)))
imgp.unlink(missing_ok=True)
print('WROTE',OUT/'f115r_lines_07_24.png',mont.shape)
