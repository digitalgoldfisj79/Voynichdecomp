#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import cv2, numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture

SEED=20261006

def polar_annulus(gray,cx,cy,r0,r1,nang=3600):
    ang=np.linspace(0,2*np.pi,nang,endpoint=False); rs=np.arange(r0,r1+1)
    xs=np.rint(cx+np.outer(rs,np.cos(ang))).astype(np.int32)
    ys=np.rint(cy+np.outer(rs,np.sin(ang))).astype(np.int32)
    return gray[ys,xs]

def outer_branch_components(gray,cx=903,cy=386,thr=165,outerdepth=10,r0=235,r1=285,a0=228,a1=342):
    nang=3600; pol=polar_annulus(gray,cx,cy,r0,r1,nang); dark=pol<thr
    mask=np.zeros_like(dark,dtype=np.uint8)
    for j in range(nang):
        yy=np.where(dark[:,j])[0]
        if len(yy):
            ymax=int(yy.max()); lo=max(0,ymax-outerdepth); mask[lo:ymax+1,j]=dark[lo:ymax+1,j]
    j0=int(a0/360*nang); j1=int(a1/360*nang); sub=mask[:,j0:j1]
    n,lab,stats,cent=cv2.connectedComponentsWithStats(sub,8)
    rows=[]
    for i in range(1,n):
        x,y,w,h,area=stats[i]; angle=a0+cent[i][0]/(j1-j0)*(a1-a0); r=r0+cent[i][1]
        if 248<=r<=272: rows.append({'id':i,'angle_deg':angle,'r':r,'w':w,'h':h,'area':area})
    return pd.DataFrame(rows)

def heuristic_class(row):
    ar=row.w/max(row.h,1)
    if 12<=row.w<=20 and 5<=row.h<=8 and 35<=row.area<=80 and ar>=1.8: return 'LOOP'
    if row.w<=10 and row.h<=5 and row.area<=25: return 'DOT'
    if row.area>=80 or row.w>=22 or row.h>=9: return 'GLYPH'
    return 'AMBIG'

def derive_gap_units(cc,a0=228,a1=339):
    c=cc[(cc.angle_deg>=a0)&(cc.angle_deg<a1)].sort_values('angle_deg').reset_index(drop=True).copy()
    c['halfwidth']=c.w*0.1/2
    raw=np.diff(c.angle_deg.to_numpy())
    clear=np.maximum(raw-(c.halfwidth.iloc[:-1].to_numpy()+c.halfwidth.iloc[1:].to_numpy()),0)
    gm=GaussianMixture(n_components=2,random_state=0,n_init=10).fit(clear.reshape(-1,1))
    order=np.argsort(gm.means_.ravel()); grid=np.linspace(clear.min(),clear.max(),20001).reshape(-1,1)
    p=gm.predict_proba(grid)[:,order]; cutoff=float(grid[np.argmin(np.abs(p[:,0]-p[:,1])),0])
    sizes=[]; start=0
    for i,g in enumerate(clear):
        if g>cutoff: sizes.append(i-start+1); start=i+1
    sizes.append(len(c)-start); sizes=np.asarray(sizes)
    return {
      'ncomp':len(c),'gap_threshold_deg':cutoff,'nunits':len(sizes),'max_unit':int(sizes.max()),
      'frac_components_in_units_size_1_or_2':float(sizes[sizes<=2].sum()/sizes.sum()),
      'frac_units_size_1_or_2':float((sizes<=2).mean()),'n_units_gt2':int((sizes>2).sum()),
      'unit_sizes':';'.join(map(str,sizes.tolist()))
    }

def cast60(seed16):
    M=np.asarray(seed16,dtype=np.uint8).reshape(4,4); D=M.T.copy()
    N=np.array([M[0]^M[1],M[2]^M[3],D[0]^D[1],D[2]^D[3]],dtype=np.uint8)
    W=np.array([N[0]^N[1],N[2]^N[3]],dtype=np.uint8); J=(W[0]^W[1])[None,:]
    return np.concatenate([M.ravel(),D.ravel(),N.ravel(),W.ravel(),J.ravel()])

def valid_cast_masks():
    valid=np.empty((65536,60),dtype=np.uint8)
    for x in range(65536):
        b=np.array([(x>>i)&1 for i in range(15,-1,-1)],dtype=np.uint8); valid[x]=cast60(b)
    powers=(1<<np.arange(60,dtype=np.uint64)); masks=(valid.astype(np.uint64)*powers).sum(axis=1,dtype=np.uint64)
    return masks,powers

POP8=np.array([bin(i).count('1') for i in range(256)],dtype=np.uint8)
def min_dist_window(w,vmasks,powers):
    m=np.sum(w.astype(np.uint64)*powers,dtype=np.uint64); x=(vmasks^m)
    return int(POP8[x.view(np.uint8).reshape(-1,8)].sum(axis=1).min())

def score_cast(arr,vmasks,powers,allow_mapping_flip=True):
    best=999
    flips=[0,1] if allow_mapping_flip else [0]
    for flip in flips:
      a=arr^flip
      for rev in [False,True]:
        b=a[::-1] if rev else a
        for st in range(len(b)-59): best=min(best,min_dist_window(b[st:st+60],vmasks,powers))
    return int(best)

def period4_test(g,mc=100000):
    ang=g.angle_deg.to_numpy(); gaps=np.diff(ang); mid=(ang[:-1]+ang[1:])/2; seg=(mid>=277).astype(int)
    def best(x):
      z=-1e9
      for off in range(4):
        idx=np.arange(len(x)); b=((idx-(off+3))%4)==0; v=idx>=off; b=b&v; nb=(~b)&v
        if b.sum() and nb.sum(): z=max(z,float(x[b].mean()-x[nb].mean()))
      return z
    obs=best(gaps); rng=np.random.default_rng(SEED); null=np.empty(mc)
    for i in range(mc):
      x=gaps.copy()
      for sg in [0,1]:
        ix=np.where(seg==sg)[0]; x[ix]=rng.permutation(x[ix])
      null[i]=best(x)
    return {'effect_deg':obs,'null_mean':float(null.mean()),'null_sd':float(null.std(ddof=1)),
            'z_vs_optimized_null':float((obs-null.mean())/null.std(ddof=1)),
            'p_upper':float((np.sum(null>=obs)+1)/(mc+1))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pdf',required=True); ap.add_argument('--outdir',required=True); ap.add_argument('--mc',type=int,default=300)
    a=ap.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    pref=out/'embedded'; subprocess.run(['pdfimages','-f','127','-l','127','-png',a.pdf,str(pref)],check=True)
    im=cv2.imread(str(sorted(out.glob('embedded-*.png'))[0])); gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    vmasks,powers=valid_cast_masks()
    rows=[]
    for cx in [900,903,906]:
      for cy in [382,386,390]:
       for th in [150,155,160]:
        for depth in [8,10,12]:
          cc=outer_branch_components(gray,cx,cy,th,depth); cc['class']=cc.apply(heuristic_class,axis=1)
          rec=derive_gap_units(cc); g=cc[(cc.angle_deg>=228)&(cc.angle_deg<339)&cc['class'].isin(['DOT','LOOP'])].sort_values('angle_deg')
          arr=np.array([0 if x=='DOT' else 1 for x in g['class']],dtype=np.uint8)
          rec.update({'cx':cx,'cy':cy,'thr':th,'depth':depth,'binary_n':len(arr),'dot_n':int((arr==0).sum()),'loop_n':int((arr==1).sum())})
          if len(arr)>=60:
            rec['cast_score_best_mapping']=score_cast(arr,vmasks,powers,True)
            rec['cast_score_natural_dot_single']=score_cast(arr^1,vmasks,powers,False)
          else:
            rec['cast_score_best_mapping']=np.nan; rec['cast_score_natural_dot_single']=np.nan
          rows.append(rec)
    S=pd.DataFrame(rows); S.to_csv(out/'f70r1_geomancy_sensitivity_20261006.csv',index=False)
    cc=outer_branch_components(gray); cc['class']=cc.apply(heuristic_class,axis=1)
    g=cc[(cc.angle_deg>=228)&(cc.angle_deg<339)&cc['class'].isin(['DOT','LOOP'])].sort_values('angle_deg')
    arr=np.array([0 if x=='DOT' else 1 for x in g['class']],dtype=np.uint8)
    best_any=score_cast(arr,vmasks,powers,True); natural=score_cast(arr^1,vmasks,powers,False)
    rng=np.random.default_rng(SEED+1); nul=[]
    for i in range(a.mc):
      x=np.ones(len(arr),dtype=np.uint8); x[rng.choice(len(arr),int((arr==1).sum()),replace=False)]=0
      nul.append(score_cast(x,vmasks,powers,False))
    nul=np.asarray(nul); p4=period4_test(g,100000)
    summ={
      'baseline':{'binary_n':len(arr),'dot_n':int((arr==0).sum()),'loop_n':int((arr==1).sum()),
                  'runs':['DOTx21','LOOPx10','DOTx39'],'best_cast_hamming_any_mapping':best_any,
                  'natural_mapping_dot_single_loop_double_hamming':natural},
      'natural_gap_unit_sensitivity':{
          'n_settings':len(S),'median_component_fraction_in_units_size_1_or_2':float(S.frac_components_in_units_size_1_or_2.median()),
          'range_component_fraction_in_units_size_1_or_2':[float(S.frac_components_in_units_size_1_or_2.min()),float(S.frac_components_in_units_size_1_or_2.max())],
          'settings_ge_0_9_fraction':int((S.frac_components_in_units_size_1_or_2>=.9).sum()),
          'median_max_unit_size':float(S.max_unit.median()),'range_max_unit_size':[int(S.max_unit.min()),int(S.max_unit.max())]},
      'four_row_spacing_periodicity':p4,
      'shield_algebra_sensitivity':{'settings_with_at_least_60_binary_objects':int(S.cast_score_best_mapping.notna().sum()),
          'best_mapping_scores':sorted(S.cast_score_best_mapping.dropna().astype(int).unique().tolist()),
          'natural_mapping_scores':sorted(S.cast_score_natural_dot_single.dropna().astype(int).unique().tolist())},
      'natural_mapping_count_preserving_null':{'mc':a.mc,'null_mean_hamming':float(nul.mean()),'null_sd_hamming':float(nul.std(ddof=1)),
          'observed_hamming':natural,'z_observed_minus_null':float((natural-nul.mean())/nul.std(ddof=1)),
          'fraction_null_as_good_or_better':float((np.sum(nul<=natural)+1)/(len(nul)+1))},
      'decision':{
          'literal_one_or_two_mark_rows':'rejected: objective gap units do not reduce to singles/pairs',
          'four_row_geomantic_spacing':'not resolved; optimized period-4 spacing is no better than local-shuffle null',
          'shield_chart_algebra':'rejected under tested object-as-row mapping; best score is trivial/majority-state and natural mapping is worse than random count-matched sequences',
          'broader_operational_register':'remains open',
          'geomancy_as_specific_explanation':'downgraded / rejected for f70r1 micro-register under this encoding'
      }}
    with open(out/'f70r1_geomancy_falsification_summary_20261006.json','w') as f: json.dump(summ,f,indent=2)
    print(json.dumps(summ,indent=2))
if __name__=='__main__': main()
