#!/usr/bin/env python3
"""Taccola × Q13 v0.2 development, using ONLY exposed v0.1 calibration artifacts.

Holdouts Clm 28800 and BnF Lat.7239 are intentionally absent from this file.
Q13 is intentionally absent.

Purpose: test a small pre-declared set of local motif representations to diagnose the
v0.1 whole-page HOG/chamfer failure. The output is DEVELOPMENT ONLY and cannot itself
unseal Q13. A separate frozen v0.2 protocol must be committed before holdout access.
"""
from __future__ import annotations
import argparse, json, math, pickle
from pathlib import Path
import cv2
import numpy as np
import taccola_calibration_v01b as core

VERSION='taccola-q13-v0.2-motif-development-20260830'
MAX_LOCAL_COMPONENTS=5
TOP_MOTIF_MATCHES=20
MIN_COMPONENT_AREA=18
LOCAL_REPS=['fourier','hu','polar','topology','local_hog']
ALL_REPS=['page_geometry']+LOCAL_REPS

# Manuscripts delivered by digitale-sammlungen.de in the locked v0.1b panel.
BSB_NULL_IDS=[k for k,v in core.MANUSCRIPTS.items() if v['role']=='control' and 'digitale-sammlungen.de' in v['manifest']]


def norm_mask(mask,size=96):
    m=np.asarray(mask,dtype=np.uint8)>0
    ys,xs=np.where(m)
    if not len(xs): return np.zeros((size,size),np.uint8)
    crop=m[ys.min():ys.max()+1,xs.min():xs.max()+1].astype(np.uint8)
    side=max(crop.shape); pad=np.zeros((side,side),np.uint8)
    oy=(side-crop.shape[0])//2; ox=(side-crop.shape[1])//2
    pad[oy:oy+crop.shape[0],ox:ox+crop.shape[1]]=crop
    return (cv2.resize(pad,(size,size),interpolation=cv2.INTER_NEAREST)>0).astype(np.uint8)


def motif_masks(page_shape):
    s=(np.asarray(page_shape)>0).astype(np.uint8)
    out=[norm_mask(s)]  # full illustration crop, but not page position/layout
    d=cv2.dilate(s,np.ones((3,3),np.uint8),iterations=2)
    n,lab,stats,_=cv2.connectedComponentsWithStats(d,8)
    candidates=[]
    H,W=s.shape
    for k in range(1,n):
        x,y,w,h,area=map(int,stats[k])
        if area<MIN_COMPONENT_AREA or max(w,h)<8: continue
        if h<4 and w>20: continue
        region=(lab==k)
        orig=(s>0)&region
        if int(orig.sum())<10: continue
        candidates.append((area,x,y,w,h,orig))
    candidates.sort(key=lambda z:(-z[0],z[1],z[2]))
    for _,x,y,w,h,orig in candidates[:MAX_LOCAL_COMPONENTS]:
        crop=orig[y:y+h,x:x+w]
        out.append(norm_mask(crop))
    # deterministic dedupe by packed bytes
    uniq=[]; seen=set()
    for m in out:
        key=np.packbits(m).tobytes()
        if key not in seen:
            uniq.append(m); seen.add(key)
    return uniq


def largest_contour(mask):
    cs,_=cv2.findContours((mask*255).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    return max(cs,key=cv2.contourArea)[:,0,:].astype(np.float64) if cs else np.empty((0,2),np.float64)


def resample_contour(pts,n=128):
    if len(pts)<3:return np.zeros((n,2),np.float64)
    q=np.vstack([pts,pts[:1]])
    seg=np.sqrt(((q[1:]-q[:-1])**2).sum(1)); cum=np.r_[0,np.cumsum(seg)]
    if cum[-1]<=1e-9:return np.zeros((n,2),np.float64)
    t=np.linspace(0,cum[-1],n,endpoint=False)
    x=np.interp(t,cum,q[:,0]); y=np.interp(t,cum,q[:,1])
    return np.c_[x,y]


def fourier_desc(mask):
    pts=resample_contour(largest_contour(mask),128)
    z=(pts[:,0]-pts[:,0].mean())+1j*(pts[:,1]-pts[:,1].mean())
    scale=np.sqrt(np.mean(np.abs(z)**2))+1e-9; z=z/scale
    f=np.fft.fft(z)
    v=np.abs(f[1:21]).astype(np.float32)
    return v/(np.linalg.norm(v)+1e-9)


def hu_desc(mask):
    h=cv2.HuMoments(cv2.moments(mask.astype(np.uint8))).flatten()
    return (-np.sign(h)*np.log10(np.abs(h)+1e-30)).astype(np.float32)


def polar_desc(mask):
    ys,xs=np.where(mask>0)
    if not len(xs):return np.zeros(14,np.float32)
    x=xs-xs.mean(); y=ys-ys.mean(); r=np.sqrt(x*x+y*y); rmax=r.max()+1e-9
    rh,_=np.histogram(r/rmax,bins=np.linspace(0,1,9)); rh=rh.astype(np.float64); rh/=rh.sum()+1e-9
    a=(np.arctan2(y,x)+2*np.pi)%(2*np.pi)
    ah,_=np.histogram(a,bins=np.linspace(0,2*np.pi,13)); ah=ah.astype(np.float64); ah/=ah.sum()+1e-9
    af=np.abs(np.fft.rfft(ah))[1:7]  # rotation-invariant angular spectrum
    v=np.r_[rh,af].astype(np.float32)
    return v/(np.linalg.norm(v)+1e-9)


def skeleton(mask):
    img=(mask>0).astype(np.uint8)*255; sk=np.zeros_like(img)
    elem=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    for _ in range(96):
        er=cv2.erode(img,elem); op=cv2.dilate(er,elem); temp=cv2.subtract(img,op); sk=cv2.bitwise_or(sk,temp); img=er
        if cv2.countNonZero(img)==0:break
    return sk>0


def topology_desc(mask):
    m=(mask>0).astype(np.uint8); ys,xs=np.where(m)
    if not len(xs):return np.zeros(12,np.float32)
    sk=skeleton(m); neigh=cv2.filter2D(sk.astype(np.uint8),-1,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT)-sk.astype(np.uint8)
    endpoints=int(np.sum(sk&(neigh==1))); junctions=int(np.sum(sk&(neigh>=3)))
    contours,hier=cv2.findContours((m*255).astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    holes=0
    if hier is not None: holes=int(np.sum(hier[0,:,3]>=0))
    cnt=largest_contour(m); per=float(cv2.arcLength(cnt.astype(np.float32).reshape(-1,1,2),True)) if len(cnt)>=3 else 0.0
    area=float(m.sum()); x0,x1=xs.min(),xs.max()+1; y0,y1=ys.min(),ys.max()+1; w=x1-x0; h=y1-y0
    hull=cv2.convexHull(cnt.astype(np.float32).reshape(-1,1,2)) if len(cnt)>=3 else None
    hull_area=float(cv2.contourArea(hull)) if hull is not None else area
    vals=[area/(96*96),w/96,h/96,w/(h+1e-9),float(sk.mean()),endpoints/50.0,junctions/50.0,holes/10.0,per/(2*(w+h)+1e-9),area/(w*h+1e-9),area/(hull_area+1e-9),float(np.std(xs)/96+np.std(ys)/96)]
    return np.asarray(vals,np.float32)


def descs(mask):
    return {'fourier':fourier_desc(mask),'hu':hu_desc(mask),'polar':polar_desc(mask),'topology':topology_desc(mask),'local_hog':core.hog_vec(mask>0)}


def cosmat(A,B,key):
    X=np.vstack([x[key] for x in A]).astype(np.float32); Y=np.vstack([x[key] for x in B]).astype(np.float32)
    X/=np.linalg.norm(X,axis=1,keepdims=True)+1e-9; Y/=np.linalg.norm(Y,axis=1,keepdims=True)+1e-9
    return np.clip(X@Y.T,-1,1)


def eucmat(A,B,key,mu,sd):
    X=(np.vstack([x[key] for x in A])-mu)/sd; Y=(np.vstack([x[key] for x in B])-mu)/sd
    # bounded RBF-like similarity on RMS standardized distance
    d2=((X[:,None,:]-Y[None,:,:])**2).mean(axis=2)
    return np.exp(-np.sqrt(d2)).astype(np.float32)


def set_score(mat,k=TOP_MOTIF_MATCHES):
    if not mat.size:return float('nan')
    a=mat.max(axis=0); b=mat.max(axis=1); ka=min(k,len(a)); kb=min(k,len(b))
    return .5*(float(np.mean(np.sort(a)[-ka:]))+float(np.mean(np.sort(b)[-kb:])))


def stat(pos,scores,nullids):
    vals=np.array([scores[k] for k in nullids],float); mu=vals.mean(); sd=vals.std(ddof=1)
    return {'positive':float(pos),'null_mean':float(mu),'null_sd':float(sd),'effect':float(pos-mu),'z':float((pos-mu)/(sd+1e-9)),'p':float((1+np.sum(vals>=pos))/(1+len(vals))),'rank':int(1+np.sum(vals>pos))}


def host_uplift(scores,nullids):
    b=[scores[k] for k in nullids if k in BSB_NULL_IDS]; o=[scores[k] for k in nullids if k not in BSB_NULL_IDS]
    allv=np.array([scores[k] for k in nullids]); sd=allv.std(ddof=1)
    return {'n_bsb':len(b),'n_other':len(o),'bsb_mean':float(np.mean(b)),'other_mean':float(np.mean(o)),'uplift_sd':float((np.mean(b)-np.mean(o))/(sd+1e-9))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default='dev_inputs'); args=ap.parse_args()
    bundles={}
    for p in Path(args.input).rglob('one_*.pkl'):
        with open(p,'rb') as f:b=pickle.load(f)
        if b['panel_sha256']!=core.EXPECTED_PANEL_SHA256:raise RuntimeError('panel hash mismatch')
        bundles[b['id']]=b
    if sorted(bundles)!=sorted(core.MANUSCRIPTS):raise RuntimeError(f'development coverage mismatch {len(bundles)}')
    pagefeats={k:v['features'] for k,v in bundles.items()}
    motifs={}; counts={}
    for mid,pages in pagefeats.items():
        mm=[]
        for pg in pages:
            for mask in motif_masks(pg['shape']):
                d=descs(mask); d['mask']=mask; mm.append(d)
        motifs[mid]=mm; counts[mid]=len(mm)
    # Standardization is unsupervised over the full DEVELOPMENT panel only.
    scales={}
    for key in ['hu','topology']:
        X=np.vstack([m[key] for mid in motifs for m in motifs[mid]])
        mu=X.mean(0); sd=X.std(0); sd=np.where(sd<1e-6,1.0,sd); scales[key]=(mu,sd)
    nullids=list(core.NULL_IDS)
    directions=[('clm197','pal766'),('pal766','clm197')]
    out={'version':VERSION,'data':'v0.1b exposed artifacts only','q13_accessed':False,'holdout_visual_access':False,'constants':{'max_local_components':MAX_LOCAL_COMPONENTS,'top_motif_matches':TOP_MOTIF_MATCHES,'local_reps':LOCAL_REPS},'motif_counts':counts,'bsb_null_ids':BSB_NULL_IDS,'directions':{}}
    score_store={}
    for template,positive in directions:
        dk=f'{template}_to_{positive}'; out['directions'][dk]={}; score_store[dk]={}
        ids=[positive]+nullids+['dresden_mixed']
        for rep in ALL_REPS:
            scores={}
            for cid in ids:
                if rep=='page_geometry':
                    geoms=np.vstack([x['geom'] for mid in pagefeats for x in pagefeats[mid]])
                    mu=geoms.mean(0); sd=geoms.std(0); sd=np.where(sd<1e-6,1,sd)
                    mat=core.pair_matrix(pagefeats[template],pagefeats[cid],'geometry',mu,sd); scores[cid]=core.manuscript_score(mat)
                else:
                    if rep in ('hu','topology'):
                        mu,sd=scales[rep]; mat=eucmat(motifs[template],motifs[cid],rep,mu,sd)
                    else: mat=cosmat(motifs[template],motifs[cid],rep)
                    scores[cid]=set_score(mat)
            st=stat(scores[positive],scores,nullids)
            tstat=stat(scores[positive],scores,core.TECH_IDS)
            out['directions'][dk][rep]={'stats':st,'technical':tstat,'host_uplift':host_uplift(scores,nullids),'derivative_score':float(scores['dresden_mixed'])}
            score_store[dk][rep]=scores
    # Fixed selection rule declared in code before results: local rep must transfer both directions,
    # rank in top 5 overall and top 3 technical, and avoid >0.5 SD BSB same-host uplift.
    qualified=[]
    for rep in LOCAL_REPS:
        ds=[out['directions'][f'{a}_to_{b}'][rep] for a,b in directions]
        if (min(d['stats']['z'] for d in ds)>=1.5 and max(d['stats']['rank'] for d in ds)<=5 and max(d['technical']['rank'] for d in ds)<=3 and max(d['host_uplift']['uplift_sd'] for d in ds)<=0.5):
            qualified.append(rep)
    qualified.sort(key=lambda r:min(out['directions'][f'{a}_to_{b}'][r]['stats']['z'] for a,b in directions),reverse=True)
    selected=qualified[:2]
    composite_reps=['page_geometry']+selected
    out['selection']={'qualified_local_reps':qualified,'selected_local_reps':selected,'composite_reps':composite_reps,'rule':'local min z>=1.5 both directions; overall rank<=5; technical rank<=3; BSB uplift<=0.5 SD; take top 2 by worst-direction z'}
    out['composite']={}
    for template,positive in directions:
        dk=f'{template}_to_{positive}'
        # standardize each rep over null manuscript blocks, then equal-weight.
        comp={cid:[] for cid in [positive]+nullids+['dresden_mixed']}
        for rep in composite_reps:
            s=score_store[dk][rep]; nv=np.array([s[k] for k in nullids]); mu=nv.mean(); sd=nv.std(ddof=1)
            for cid in comp: comp[cid].append((s[cid]-mu)/(sd+1e-9))
        comp={k:float(np.mean(v)) for k,v in comp.items()}
        cs=stat(comp[positive],comp,nullids); ct=stat(comp[positive],comp,core.TECH_IDS)
        # representation correlations across null manuscript blocks
        corr={}
        for i,r1 in enumerate(composite_reps):
            for r2 in composite_reps[i+1:]:
                x=np.array([score_store[dk][r1][k] for k in nullids]); y=np.array([score_store[dk][r2][k] for k in nullids])
                corr[f'{r1}__{r2}']=float(np.corrcoef(x,y)[0,1])
        out['composite'][dk]={'stats':cs,'technical':ct,'scores':comp,'null_rep_correlations':corr}
    dev_gate=(len(selected)>=1 and all(out['composite'][f'{a}_to_{b}']['stats']['z']>=2.0 for a,b in directions) and all(out['composite'][f'{a}_to_{b}']['stats']['p']<=0.05 for a,b in directions) and all(out['composite'][f'{a}_to_{b}']['technical']['rank']<=2 for a,b in directions))
    out['development_gate']={'passed':bool(dev_gate),'criteria':'at least one qualified local rep; geometry+selected local composite z>=2 and p<=.05 both directions; technical rank<=2 both directions','note':'Development pass permits protocol freezing only. It does NOT permit holdout interpretation or Q13 access.'}
    od=Path('taccola_v02_development'); od.mkdir(exist_ok=True)
    (od/'development.json').write_text(json.dumps(out,indent=2))
    lines=['# Taccola × Q13 v0.2 motif development','','## RETRACTED FINDINGS','','- **RETRACTED:** the raw Siena/Taccola concentration is not independent evidence; family-collapsed Fisher p=0.318.','','Q13 was not accessed. Clm 28800 and BnF Lat.7239 visual material was not accessed. This development run uses only the exposed v0.1b feature artifacts.','',f"Qualified local reps: `{qualified}`",f"Selected local reps: `{selected}`",f"Development gate passed: `{dev_gate}`",'']
    for a,b in directions:
        dk=f'{a}_to_{b}'; c=out['composite'][dk]
        lines.append(f"- {dk}: composite z={c['stats']['z']:.3f}, p={c['stats']['p']:.4f}, rank={c['stats']['rank']}/35, technical rank={c['technical']['rank']}/7")
        for r in ALL_REPS:
            d=out['directions'][dk][r]
            lines.append(f"  - {r}: z={d['stats']['z']:.3f}, p={d['stats']['p']:.4f}, rank={d['stats']['rank']}/35, technical rank={d['technical']['rank']}/7, BSB uplift={d['host_uplift']['uplift_sd']:.2f} SD")
    (od/'DEVELOPMENT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'event':'v02_development_done','qualified':qualified,'selected':selected,'development_gate':dev_gate,'composite':{k:v['stats']['z'] for k,v in out['composite'].items()}},sort_keys=True),flush=True)

if __name__=='__main__':main()
