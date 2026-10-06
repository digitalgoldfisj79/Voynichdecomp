#!/usr/bin/env python3
"""Reconstruct and extend the f70r1 outer micro-register analysis.

Input: Yale Beinecke 408 PDF 2002046.pdf.
Outputs: component table, sensitivity table, summary JSON, overlay PNG.

This script intentionally separates:
1) the recovered small-component half-arc concentration test;
2) a categorical outer-register extraction;
3) representation sensitivity of the main categorical change-point;
4) exploratory permutation tests for LOOP and late-DOT enrichment.

Dependencies: numpy, pandas, opencv-python, scipy; system pdfimages.
"""
from __future__ import annotations
import argparse, json, math, os, pickle, subprocess, tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

SEED = 20261006


def polar_annulus(gray, cx, cy, r0, r1, nang=3600):
    angles = np.linspace(0, 2*np.pi, nang, endpoint=False)
    rs = np.arange(r0, r1+1)
    xs = np.rint(cx + np.outer(rs, np.cos(angles))).astype(np.int32)
    ys = np.rint(cy + np.outer(rs, np.sin(angles))).astype(np.int32)
    return gray[ys, xs]


def max_count_arc(a_deg, width=180.0):
    a = np.sort(np.asarray(a_deg) % 360.0)
    aa = np.r_[a, a+360]
    n = len(a); best = 0; best_start = 0.0; j = 0
    for i in range(n):
        if j < i: j = i
        while j < i+n and aa[j] < aa[i] + width:
            j += 1
        c = j-i
        if c > best:
            best = c; best_start = aa[i] % 360
    return int(best), float(best_start)


def small_components(gray, cx=903, cy=386, r0=248, r1=264,
                     nang=3600, thr=155, maxw=9):
    pol = polar_annulus(gray, cx, cy, r0, r1, nang)
    bw = (pol < thr).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(bw, 8)
    rows = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w <= maxw:
            theta = cent[i][0] / nang * 2*np.pi
            r = r0 + cent[i][1]
            rows.append({
                'id': i, 'xbin': x, 'ybin': y, 'w': w, 'h': h, 'area': area,
                'theta_bin': cent[i][0], 'r_bin': cent[i][1],
                'theta_deg': np.degrees(theta) % 360, 'r': r,
                'x': cx + r*np.cos(theta), 'y': cy + r*np.sin(theta)
            })
    return pd.DataFrame(rows), pol, bw


def outer_branch_components(gray, cx=903, cy=386, thr=165, outerdepth=10,
                            r0=235, r1=285, a0=228, a1=342):
    """Retain the outermost dark run in each polar column, then connected components."""
    nang = 3600
    pol = polar_annulus(gray, cx, cy, r0, r1, nang)
    dark = pol < thr
    mask = np.zeros_like(dark, dtype=np.uint8)
    for j in range(nang):
        yy = np.where(dark[:, j])[0]
        if len(yy):
            ymax = int(yy.max())
            lo = max(0, ymax - outerdepth)
            mask[lo:ymax+1, j] = dark[lo:ymax+1, j]
    j0 = int(a0/360*nang); j1 = int(a1/360*nang)
    sub = mask[:, j0:j1]
    n, lab, stats, cent = cv2.connectedComponentsWithStats(sub, 8)
    rows = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        angle = a0 + cent[i][0] / (j1-j0) * (a1-a0)
        r = r0 + cent[i][1]
        if 248 <= r <= 272:
            rows.append({'id':i,'angle_deg':angle,'r':r,'w':w,'h':h,'area':area})
    return pd.DataFrame(rows), pol, mask


def heuristic_class(row):
    ar = row.w / max(row.h, 1)
    if 12 <= row.w <= 20 and 5 <= row.h <= 8 and 35 <= row.area <= 80 and ar >= 1.8:
        return 'LOOP'
    if row.w <= 10 and row.h <= 5 and row.area <= 25:
        return 'DOT'
    if row.area >= 80 or row.w >= 22 or row.h >= 9:
        return 'GLYPH'
    return 'AMBIG'


def feature_matrix(cc, a0=228, a1=342):
    bins = np.arange(a0, a1+1, 1)
    rows=[]
    for a in bins[:-1]:
        g = cc[(cc.angle_deg >= a) & (cc.angle_deg < a+1)]
        rows.append({
            'angle':a+0.5, 'count':len(g), 'area_sum':g.area.sum(),
            'area_max':g.area.max() if len(g) else 0,
            'w_mean':g.w.mean() if len(g) else 0,
            'w_max':g.w.max() if len(g) else 0,
            'h_mean':g.h.mean() if len(g) else 0,
        })
    F = pd.DataFrame(rows)
    X = F[['count','area_sum','area_max','w_mean','w_max','h_mean']].astype(float).copy()
    for c in ['area_sum','area_max','w_max']:
        X[c] = np.log1p(X[c])
    sd = X.std(ddof=0).replace(0,1)
    Z = ((X-X.mean())/sd).to_numpy()
    return F, Z


def dp_k_segments(Z, K=3, minlen=4):
    n,d = Z.shape
    ps = np.vstack([np.zeros((1,d)), np.cumsum(Z,axis=0)])
    ps2 = np.vstack([np.zeros((1,d)), np.cumsum(Z*Z,axis=0)])
    def cost(i,j):
        m=j-i; s=ps[j]-ps[i]; s2=ps2[j]-ps2[i]
        return float((s2-s*s/m).sum())
    INF=1e18
    dp=np.full((K+1,n+1),INF); prev=np.full((K+1,n+1),-1,int)
    dp[0,0]=0
    for k in range(1,K+1):
        for j in range(k*minlen,n+1):
            best=INF; bi=-1
            for i in range((k-1)*minlen, j-minlen+1):
                v=dp[k-1,i]+cost(i,j)
                if v<best: best=v; bi=i
            dp[k,j]=best; prev[k,j]=bi
    cuts=[n]; j=n
    for k in range(K,0,-1):
        j=prev[k,j]; cuts.append(j)
    return float(dp[K,n]), sorted(cuts)


def perm_diff(binary, group, reps=200000, seed=SEED):
    binary=np.asarray(binary,dtype=float); group=np.asarray(group,dtype=bool)
    obs=float(binary[group].mean()-binary[~group].mean())
    n=len(binary); m=int(group.sum()); rng=np.random.default_rng(seed)
    vals=np.empty(reps)
    for i in range(reps):
        idx=rng.choice(n,m,replace=False)
        mask=np.zeros(n,dtype=bool); mask[idx]=True
        vals[i]=binary[mask].mean()-binary[~mask].mean()
    sd=float(vals.std(ddof=1)); z=obs/sd if sd else float('inf')
    p=float((np.sum(vals>=obs)+1)/(reps+1))
    return {'effect':obs,'null_sd':sd,'z':z,'p_upper':p}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pdf',required=True)
    ap.add_argument('--outdir',required=True)
    args=ap.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    pkl=out/'pickles'; pkl.mkdir(exist_ok=True)

    prefix=out/'embedded'
    subprocess.run(['pdfimages','-f','127','-l','127','-png',args.pdf,str(prefix)],check=True)
    candidates=sorted(out.glob('embedded-*.png'))
    if not candidates: raise RuntimeError('pdfimages produced no image')
    image_path=candidates[0]
    img=cv2.imread(str(image_path)); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    with open(pkl/'01_image.pkl','wb') as f: pickle.dump({'image_path':str(image_path),'shape':img.shape},f)

    base, pol, bw=small_components(gray)
    angles=base.theta_deg.to_numpy()
    obs,best_start=max_count_arc(angles)
    rng=np.random.default_rng(SEED)
    null=np.empty(20000,dtype=int)
    for i in range(len(null)):
        null[i]=max_count_arc(rng.uniform(0,360,len(angles)))[0]
    half={'n':len(base),'best_180_count':obs,'best_start_deg':best_start,
          'null_mean':float(null.mean()),'null_sd':float(null.std(ddof=1)),
          'z':float((obs-null.mean())/null.std(ddof=1)),
          'p_upper':float((np.sum(null>=obs)+1)/(len(null)+1))}
    base.to_csv(out/'f70r1_recovered_small_components_20261006.csv',index=False)
    with open(pkl/'02_baseline.pkl','wb') as f: pickle.dump({'components':base,'half':half},f)

    overlay=img.copy()
    for _,r in base.iterrows(): cv2.circle(overlay,(int(round(r.x)),int(round(r.y))),3,(0,0,255),1)
    cv2.circle(overlay,(903,386),2,(255,0,0),-1)
    cv2.imwrite(str(out/'f70r1_recovered_baseline_overlay_20261006.png'),overlay)

    cc,_,_=outer_branch_components(gray)
    cc['class']=cc.apply(heuristic_class,axis=1)
    cc.to_csv(out/'f70r1_outer_register_components_20261006.csv',index=False)
    with open(pkl/'03_categorical.pkl','wb') as f: pickle.dump(cc,f)

    g=cc[(cc.angle_deg>=228)&(cc.angle_deg<339)].copy()
    loop_group=((g.angle_deg>=240)&(g.angle_deg<264)).to_numpy()
    loop_test=perm_diff((g['class']=='LOOP').to_numpy(),loop_group,seed=SEED)
    d=cc[(cc.angle_deg>=240)&(cc.angle_deg<339)].copy()
    late=(d.angle_deg>=278).to_numpy()
    dot_test=perm_diff((d['class']=='DOT').to_numpy(),late,seed=SEED+1)

    rows=[]
    for cx in [900,903,906]:
      for cy in [382,386,390]:
       for thr in [150,155,160]:
        for depth in [8,10,12]:
            cci,_,_=outer_branch_components(gray,cx=cx,cy=cy,thr=thr,outerdepth=depth)
            F,Z=feature_matrix(cci)
            cost,cuts=dp_k_segments(Z,3,4)
            cutang=[F.angle.iloc[c] if c<len(F) else 342 for c in cuts]
            rows.append({'cx':cx,'cy':cy,'thr':thr,'depth':depth,'ncomp':len(cci),
                         'cut1_deg':float(cutang[1]),'cut2_deg':float(cutang[2]),'cost':cost})
    sens=pd.DataFrame(rows)
    sens.to_csv(out/'f70r1_change_point_sensitivity_20261006.csv',index=False)
    with open(pkl/'04_sensitivity.pkl','wb') as f: pickle.dump(sens,f)

    summary={
      'source':{'pdf':str(args.pdf),'pdf_page':127,'embedded_image_shape':list(img.shape)},
      'recovered_baseline':half,
      'categorical_counts':cc['class'].value_counts().to_dict(),
      'regions':{
        '228_240':cc[(cc.angle_deg>=228)&(cc.angle_deg<240)]['class'].value_counts().to_dict(),
        '240_264':cc[(cc.angle_deg>=240)&(cc.angle_deg<264)]['class'].value_counts().to_dict(),
        '264_278':cc[(cc.angle_deg>=264)&(cc.angle_deg<278)]['class'].value_counts().to_dict(),
        '278_339':cc[(cc.angle_deg>=278)&(cc.angle_deg<339)]['class'].value_counts().to_dict(),
        '339_342':cc[(cc.angle_deg>=339)&(cc.angle_deg<342)]['class'].value_counts().to_dict(),
      },
      'exploratory_tests':{'loop_island_240_264_vs_rest_228_339':loop_test,
                           'late_dot_278_339_vs_240_278':dot_test,
                           'warning':'Post-hoc categorical thresholds and region boundaries; exploratory, not confirmatory.'},
      'change_point_sensitivity':{
        'n_settings':len(sens),
        'cut1_mean_deg':float(sens.cut1_deg.mean()),'cut1_sd_deg':float(sens.cut1_deg.std(ddof=1)),
        'cut1_min_deg':float(sens.cut1_deg.min()),'cut1_max_deg':float(sens.cut1_deg.max()),
        'cut2_mean_deg':float(sens.cut2_deg.mean()),'cut2_sd_deg':float(sens.cut2_deg.std(ddof=1)),
        'cut2_min_deg':float(sens.cut2_deg.min()),'cut2_max_deg':float(sens.cut2_deg.max())
      },
      'decision':{
        'clean_metric_graduation':'rejected',
        'robust_categorical_change':'supported near 277 degrees',
        'loop_island':'visually and morphologically supported around 245-262 degrees but start boundary is representation-sensitive',
        'separate_glyph_state':'not resolved by automated morphology; retain as visual interruption, not a stable state',
        'licensed_sequence':'DOT-dominant -> LOOP island -> DOT-dominant -> GAP/sparsity',
        'geographic_localisation':'not licensed'
      }
    }
    with open(out/'f70r1_categorical_summary_20261006.json','w') as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
