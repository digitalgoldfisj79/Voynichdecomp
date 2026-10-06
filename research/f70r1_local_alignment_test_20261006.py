#!/usr/bin/env python3
"""Test whether f70r1 micro-register changes align locally with adjacent circular text.

Uses Yale Beinecke 408 PDF page 127, original 2000x872 embedded image.
The robust micro-register transition angles come from the prior frozen
81-setting extraction. Word/text boundaries are derived independently from
inner radial bands using only grayscale threshold + 1D morphological joining.

Null: common angular rotation of the frozen transition set within the active
228..339 degree arc, preserving its representation spread.
"""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import cv2, numpy as np, pandas as pd
from scipy.ndimage import binary_closing,binary_opening

SEED=20261006
CUT2=np.array([278.5,278.5,278.5,277.5,277.5,275.5,277.5,277.5,277.5,
               277.5,277.5,277.5,279.5,277.5,277.5,277.5,277.5,277.5,
               276.5,276.5,276.5,276.5,276.5,276.5,276.5,276.5,276.5])
MICRO=np.array([243.07659884,262.92851959,277.27777778])
LO,HI=228.0,339.0
L=HI-LO

def polar(gray,cx,cy,r0=150,r1=240,nang=5001,a0=220,a1=345):
    ad=np.linspace(a0,a1,nang); ar=np.deg2rad(ad); rr=np.arange(r0,r1+1)
    xs=np.rint(cx+np.outer(rr,np.cos(ar))).astype(int)
    ys=np.rint(cy+np.outer(rr,np.sin(ar))).astype(int)
    p=np.full(xs.shape,255,np.uint8); v=(xs>=0)&(xs<gray.shape[1])&(ys>=0)&(ys<gray.shape[0])
    p[v]=gray[ys[v],xs[v]]
    return ad,rr,p

def boundaries(ad,rr,p,rlo,rhi,thr,minpix,close_deg,minrun_deg=.2):
    step=ad[1]-ad[0]; sel=(rr>=rlo)&(rr<=rhi)
    occ=(p[sel,:]<thr).sum(axis=0)>=minpix
    nc=max(1,int(round(close_deg/step)))
    x=binary_closing(occ,structure=np.ones(nc,bool))
    no=max(1,int(round(minrun_deg/step)))
    x=binary_opening(x,structure=np.ones(no,bool))
    dom=(ad>=LO)&(ad<=HI); xx=x[dom]; aa=ad[dom]
    return np.array([(aa[i-1]+aa[i])/2 for i in range(1,len(xx)) if xx[i]!=xx[i-1]])

def cdist(vals,edges):
    v=(np.asarray(vals)-LO)%L; e=(np.asarray(edges)-LO)%L
    d=np.abs(v[:,None]-e[None,:]); d=np.minimum(d,L-d)
    return d.min(axis=1)

def rot_null(bounds,edges,stat='median',reps=5000,rng=None):
    obs=float(np.median(cdist(bounds,edges)) if stat=='median' else np.mean(cdist(bounds,edges)))
    shifts=rng.uniform(0,L,reps); null=np.empty(reps)
    for i,s in enumerate(shifts):
        b=LO+((bounds-LO+s)%L)
        d=cdist(b,edges)
        null[i]=np.median(d) if stat=='median' else np.mean(d)
    sd=float(null.std(ddof=1))
    return obs,float(null.mean()),sd,float((obs-null.mean())/sd),float((np.sum(null<=obs)+1)/(reps+1))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pdf',required=True); ap.add_argument('--outdir',required=True)
    a=ap.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    pref=out/'page'; subprocess.run(['pdfimages','-f','127','-l','127','-png',a.pdf,str(pref)],check=True)
    gray=cv2.imread(str(sorted(out.glob('page-*.png'))[0]),cv2.IMREAD_GRAYSCALE)
    ad,rr,p=polar(gray,903,386); rng=np.random.default_rng(SEED)
    outer=[]; multi=[]
    for rlo,rhi in [(196,218),(198,220),(200,222),(202,224),(204,226)]:
      for thr in [140,150,160,170,180]:
       for mp in [1,2,3]:
        for cl in [.5,.75,1.0,1.25]:
          e=boundaries(ad,rr,p,rlo,rhi,thr,mp,cl)
          if len(e)>=2:
            o,nm,ns,z,pv=rot_null(CUT2,e,'median',5000,rng)
            outer.append([rlo,rhi,thr,mp,cl,len(e),o,nm,ns,z,pv])
    for sh in [-4,-2,0,2,4]:
      for thr in [140,150,160,170,180]:
       for mp in [1,2,3]:
        for cl in [.5,.75,1.0,1.25]:
          eo=boundaries(ad,rr,p,200+sh,222+sh,thr,mp,cl)
          ei=boundaries(ad,rr,p,178+sh,200+sh,thr,mp,cl)
          e=np.sort(np.r_[eo,ei])
          if len(e)>=4:
            o,nm,ns,z,pv=rot_null(MICRO,e,'mean',5000,rng)
            multi.append([sh,thr,mp,cl,len(eo),len(ei),o,nm,ns,z,pv])
    O=pd.DataFrame(outer,columns=['rlo','rhi','thr','minpix','close','nedges','obs','null_mean','null_sd','z','p'])
    M=pd.DataFrame(multi,columns=['shift','thr','minpix','close','nout','nin','obs','null_mean','null_sd','z','p'])
    O.to_csv(out/'f70r1_local_text_outer_sensitivity_20261006.csv',index=False)
    M.to_csv(out/'f70r1_local_text_multiboundary_sensitivity_20261006.csv',index=False)
    rep=O[(O.rlo==200)&(O.rhi==222)&(O.thr==160)&(O.minpix==1)&(O.close==.75)].iloc[0]
    summary={
      'primary_outer_ring':{
        'representative':rep.to_dict(),
        'n_settings':len(O),
        'median_nearest_boundary_deg':float(O.obs.median()),
        'median_null_mean_deg':float(O.null_mean.median()),
        'median_null_sd_deg':float(O.null_sd.median()),
        'median_z':float(O.z.median()),'median_p':float(O.p.median()),
        'fraction_settings_p_lt_0_05':float((O.p<.05).mean()),
        'fraction_settings_distance_lt_0_5deg':float((O.obs<.5).mean())},
      'three_micro_boundaries_to_union_two_text_rings':{
        'micro_boundaries_deg':MICRO.tolist(),'n_settings':len(M),
        'median_mean_distance_deg':float(M.obs.median()),
        'median_null_mean_deg':float(M.null_mean.median()),
        'median_null_sd_deg':float(M.null_sd.median()),
        'median_z':float(M.z.median()),'median_p':float(M.p.median()),
        'fraction_settings_p_lt_0_05':float((M.p<.05).mean())},
      'decision':{
        'local_text_alignment':'not resolved; attractive in selected representations but fails source-resolution sensitivity',
        'robust_277_transition':'can lie very near a detected outer-ring text edge in some settings, but not representation-stable',
        'multi_boundary_alignment':'not resolved',
        'structural_spoke_alignment':'not robust under Hough detector settings',
        'next':'stop inferring function from f70r1 rim alone; require an independent repeated instance elsewhere in the Voynich or a historical exact structural match'}
    }
    with open(out/'f70r1_local_alignment_summary_20261006.json','w') as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
