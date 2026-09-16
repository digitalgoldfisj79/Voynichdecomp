#!/usr/bin/env python3
from pathlib import Path
import base64, cv2

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
mont=cv2.imread(str(OUT/'f115r_lines_07_24.png'))
if mont is None: raise SystemExit('montage missing')
# Recover line-strip boundaries from known concatenation: use horizontal separator rows (near-uniform grey).
g=cv2.cvtColor(mont,cv2.COLOR_BGR2GRAY)
row_sd=g.std(axis=1); row_mean=g.mean(axis=1)
sep=(row_sd<2.0)&(row_mean>180)&(row_mean<245)
# contiguous separator runs
runs=[]; i=0
while i<len(sep):
    if not sep[i]: i+=1; continue
    j=i
    while j+1<len(sep) and sep[j+1]: j+=1
    if j-i+1>=2: runs.append((i,j))
    i=j+1
# line row bands are between separator runs; there should be 18 bands.
bounds=[]; start=0
for a,b in runs:
    if a>start: bounds.append((start,a))
    start=b+1
if start<mont.shape[0]: bounds.append((start,mont.shape[0]))
# retain substantial bands only
bounds=[(a,b) for a,b in bounds if b-a>20]
if len(bounds)!=18:
    raise SystemExit(f'expected 18 line bands, found {len(bounds)}: {bounds}')
for name,lo,hi in [('L07_12',0,6),('L13_18',6,12),('L19_24',12,18)]:
    a=bounds[lo][0]; b=bounds[hi-1][1]
    crop=mont[a:b]
    ok,buf=cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,92])
    if not ok: raise SystemExit('jpeg encode failed')
    (OUT/f'{name}.jpg.b64').write_text(base64.b64encode(buf.tobytes()).decode('ascii'))
    print(name,crop.shape,len(buf))
