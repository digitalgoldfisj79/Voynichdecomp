#!/usr/bin/env python3
"""Cross-folio replication test for the f70r1 DOT/LOOP micro-register.

Scans predefined circular structures on f57v, f67r/v, f68r/v, f69r,
f70r and f70v across all plausible annular radii. Uses the same polar
component morphology rules as the persisted f70r1 analysis, but normalizes
angular sampling so one polar x-bin is ~0.454 source pixels at each radius.

Primary recurrence statistic:
- ordered DOT/LOOP objects around each annulus;
- longest circular run of LOOP-class objects;
- exact conditional null P(max LOOP run >= observed | n objects, k LOOPs)
  under random placement of k LOOPs among n positions.

Each circle's minimum p across scanned radii is Bonferroni-corrected for
the annulus search. A liberal morphology-replication gate is also reported:
>=8 LOOP objects and a contiguous LOOP run >=5 (f70r1 baseline has ~10
LOOPs in one island).
"""
from __future__ import annotations
import argparse, math, subprocess
from pathlib import Path
import cv2, numpy as np, pandas as pd

CANDIDATES=[
('f57v',115,900,700,520),
('f67r_L',122,430,760,390),('f67r_R',122,1460,650,440),
('f67v',123,1500,590,420),
('f68r',124,1490,450,330),
('f68v_L',125,520,450,340),('f68v_M',125,1110,440,270),('f68v_R',125,1625,440,270),
('f69r',126,650,900,440),
('f70r_L',127,400,390,260),('f70r1',127,903,386,260),('f70r_R',127,1425,390,260),
('f70v_partA',128,1040,925,650),('f70v_partB',129,850,900,575)
]

def polar_annulus(gray,cx,cy,r0,r1,nang):
    a=np.linspace(0,2*np.pi,nang,endpoint=False); rs=np.arange(r0,r1+1)
    xs=np.rint(cx+np.outer(rs,np.cos(a))).astype(np.int32)
    ys=np.rint(cy+np.outer(rs,np.sin(a))).astype(np.int32)
    out=np.full(xs.shape,255,np.uint8)
    v=(xs>=0)&(xs<gray.shape[1])&(ys>=0)&(ys<gray.shape[0])
    out[v]=gray[ys[v],xs[v]]
    return out

def extract(gray,cx,cy,R,thr=165,depth=10):
    r0=max(2,int(round(R-25))); r1=int(round(R+25))
    nang=max(800,int(round(2*np.pi*R/0.454)))
    pol=polar_annulus(gray,cx,cy,r0,r1,nang)
    dark=pol<thr; mask=np.zeros_like(dark,np.uint8)
    for j in range(nang):
        yy=np.where(dark[:,j])[0]
        if len(yy):
            ymax=int(yy.max()); lo=max(0,ymax-depth)
            mask[lo:ymax+1,j]=dark[lo:ymax+1,j]
    n,lab,stats,cent=cv2.connectedComponentsWithStats(mask,8)
    rows=[]
    for i in range(1,n):
        x,y,w,h,area=stats[i]; r=r0+cent[i][1]
        if not (R-12<=r<=R+12): continue
        ar=w/max(h,1)
        if 12<=w<=20 and 5<=h<=8 and 35<=area<=80 and ar>=1.8: cl='LOOP'
        elif w<=10 and h<=5 and area<=25: cl='DOT'
        elif area>=80 or w>=22 or h>=9: cl='GLYPH'
        else: cl='AMBIG'
        rows.append((cent[i][0]/nang*360,cl))
    return rows

def maxrun(seq):
    seq=np.asarray(seq,dtype=bool); n=len(seq)
    if not n: return 0
    if seq.all(): return n
    best=cur=0
    for v in np.r_[seq,seq]:
        cur=cur+1 if v else 0; best=max(best,cur)
    return min(best,n)

def p_circular_run_ge(n,k,r):
    """Exact conditional tail probability for longest circular 1-run."""
    if r<=0: return 1.0
    if k<r: return 0.0
    if k==n: return 1.0
    z=n-k; m=r-1
    # coefficient of (1+x+...+x^m)^z at x^k
    dp=[0]*(k+1); dp[0]=1
    for _ in range(z):
        nd=[0]*(k+1); s=0
        for j in range(k+1):
            s+=dp[j]
            if j-m-1>=0: s-=dp[j-m-1]
            nd[j]=s
        dp=nd
    good=(n*dp[k])/z
    return max(0.0,1-good/math.comb(n,k))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pdf',required=True)
    ap.add_argument('--outdir',required=True)
    args=ap.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,page,cx,cy,R0 in CANDIDATES:
        pp=out/f'p{page}'; pp.mkdir(exist_ok=True)
        pref=pp/'img'
        subprocess.run(['pdfimages','-f',str(page),'-l',str(page),'-png',args.pdf,str(pref)],check=True)
        img=sorted(pp.glob('img-*.png'))[0]
        g=cv2.imread(str(img),cv2.IMREAD_GRAYSCALE)
        radii=np.arange(max(60,.35*R0),1.06*R0,8)
        for R in radii:
            a=extract(g,cx,cy,float(R),165)
            b=sorted((ang,cl) for ang,cl in a if cl in ('DOT','LOOP'))
            n=len(b); k=sum(cl=='LOOP' for _,cl in b)
            run=maxrun([cl=='LOOP' for _,cl in b]) if n else 0
            p=p_circular_run_ge(n,k,run) if n>=10 and k>=2 else 1.0
            rows.append((name,page,R,n,k,run,p))
    D=pd.DataFrame(rows,columns=['name','pdf_page','radius','n_binary','loops','max_loop_run','p_raw'])
    D.to_csv(out/'f70r1_crossfolio_all_annuli_20261006.csv',index=False)
    compact=[]
    ncirc=len(CANDIDATES)
    for name,d in D.groupby('name',sort=False):
        q=d.loc[d.p_raw.idxmin()]
        pann=min(1.0,float(q.p_raw)*len(d))
        sig=d[(d.loops>=8)&(d.max_loop_run>=5)]
        compact.append({
          'name':name,'n_radii':len(d),'best_radius':float(q.radius),
          'n_binary':int(q.n_binary),'loops':int(q.loops),'max_loop_run':int(q.max_loop_run),
          'p_raw':float(q.p_raw),'p_annulus_bonf':pann,
          'p_global_circles_bonf':min(1.0,pann*ncirc),
          'f70like_annuli':int(len(sig)),
          'best_f70like_p':None if len(sig)==0 else float(sig.p_raw.min())
        })
    C=pd.DataFrame(compact)
    C.to_csv(out/'f70r1_crossfolio_summary_20261006.csv',index=False)
    print(C.to_string(index=False))

if __name__=='__main__': main()
