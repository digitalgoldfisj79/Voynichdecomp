#!/usr/bin/env python3
"""Human-readable QC for v0.1 extraction. Does not use hand labels."""
from pathlib import Path
import json, sys
import cv2, numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_assay as ra

OUT=HERE/'results'
img=cv2.imread(str(OUT/'1006274_f115r.jpg'))
if img is None: raise SystemExit('source image missing; run assay first')
H,W=img.shape[:2]
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); mask=ra.adaptive_ink(gray)
x0,x1=int(.105*W),int(.925*W); y0,y1=int(.038*H),int(.865*H)
centers,edges,_=ra.detect_lines(mask,x0,x1,y0,y1)
trans=ra.parse_f_transcription(ra.fetch_text(ra.TR_URL))


def art(cm,w=24,h=32):
    ys,xs=np.nonzero(cm)
    if not len(xs): return ['(empty)']
    g=cm[ys.min():ys.max()+1,xs.min():xs.max()+1].astype(np.uint8)*255
    # preserve aspect ratio within fixed canvas
    gh,gw=g.shape; s=min((w-2)/max(1,gw),(h-2)/max(1,gh)); nw=max(1,int(round(gw*s))); nh=max(1,int(round(gh*s)))
    r=cv2.resize(g,(nw,nh),interpolation=cv2.INTER_NEAREST)>0
    can=np.zeros((h,w),bool); oy=(h-nh)//2; ox=(w-nw)//2; can[oy:oy+nh,ox:ox+nw]=r
    return [''.join('██' if v else '  ' for v in row) for row in can]

lines=[]; word_dump=[]
for rec in trans:
    ln=rec['line']; ya,yb=int(edges[ln-1]),int(edges[ln]); lm=mask[ya:yb,x0:x1]
    boxes,qc=ra.segment_words(lm,len(rec['tokens']))
    widths=[b-a for a,b in boxes]
    if 1<=ln<=25:
        lines.append(f"LINE {ln:02d} center={int(centers[ln-1])} gap_before={(int(centers[ln-1]-centers[ln-2]) if ln>1 else 'NA')} tokens={len(rec['tokens'])} gap_sep={qc.get('gap_sep')}")
        lines.append('  TOKENS: '+' | '.join(rec['tokens']))
        lines.append('  WIDTHS: '+' | '.join(map(str,widths)))
    word_dump.append({'line':ln,'tokens':rec['tokens'],'widths':widths,'boxes':boxes,'qc':qc})
    if not boxes or not (7<=ln<=24): continue
    for wi,(tok,(xa,xb)) in enumerate(zip(rec['tokens'],boxes)):
        if tok.count('k')!=1 or any(c in tok for c in 'tpf'): continue
        cm,meta=ra.choose_k_component(lm[:,xa:xb],tok)
        if cm is None: continue
        feat=ra.k_features(cm)
        lines.append(f"  K-CAND line={ln:02d} word={wi+1:02d} token={tok} pred_err={meta['pred_err']:.3f} bbox={meta['bbox']} accepted={feat is not None and meta['pred_err']<=.38}")
        lines.extend('    '+s for s in art(cm))
        lines.append('')

(OUT/'k_ascii_qc.txt').write_text('\n'.join(lines),encoding='utf-8')
(OUT/'word_segmentation_qc.json').write_text(json.dumps(word_dump,indent=2),encoding='utf-8')
print('ASCII_QC_WRITTEN',len(lines),'text rows')
