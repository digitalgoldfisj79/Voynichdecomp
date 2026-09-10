#!/usr/bin/env python3
import itertools, json, math, os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter

PROTOCOL = "vms_oversized_parchment_class_20260910_v01"
OUT = Path("artifacts/vms_oversized_parchment_class_20260910")
OUT.mkdir(parents=True, exist_ok=True)

# Frozen before scoring. Candidates are the six multi-panel / oversized Yale scans in Q14,Q15,Q17,Q19.
# Controls are adjacent simple leaf scans, excluding labels explicitly marked part/foldout.
SCANS = [
    (158,"Q14","f85v_f86r","1006231",1),
    (162,"Q15","f88v_f89r","1006233",1),
    (164,"Q15","f89v_f90r","1006235",1),
    (170,"Q17","f94v_f95r","1006241",1),
    (178,"Q19","f100v_f101r","1006249",1),
    (180,"Q19","f101v_f102r","1006251",1),
    (159,"Q15","f87r","1006232",0),
    (160,"Q15","f87v","1043429",0),
    (161,"Q15","f88r","1037112",0),
    (165,"Q15","f90r","1006236",0),
    (167,"Q17","f93r","1006238",0),
    (168,"Q17","f93v","1006239",0),
    (169,"Q17","f94r","1006240",0),
    (172,"Q17","f95v","1006243",0),
    (173,"Q17","f96r","1006244",0),
    (174,"Q17","f96v","1006245",0),
    (175,"Q19","f99r","1006246",0),
    (176,"Q19","f99v","1006247",0),
    (177,"Q19","f100r","1006248",0),
]

HEADERS={"User-Agent":"Voynich-research/1.0 contact: research"}

def download(service_id, width=2400):
    p=OUT/f"{service_id}_{width}.jpg"
    if p.exists() and p.stat().st_size>10000:
        return p
    url=f"https://collections.library.yale.edu/iiif/2/{service_id}/full/{width},/0/default.jpg"
    r=requests.get(url,headers=HEADERS,timeout=60)
    r.raise_for_status()
    p.write_bytes(r.content)
    return p

def largest_parchment_mask(img):
    # Yale page scans use a dark board. Otsu on lightness gives a reproducible first-pass page mask.
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    L=lab[:,:,0]
    blur=cv2.GaussianBlur(L,(0,0),2.0)
    _,m=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,k,iterations=2)
    n,labels,stats,_=cv2.connectedComponentsWithStats(m,8)
    if n<2: raise RuntimeError("no foreground component")
    # Prefer largest non-border-spanning light component.
    idx=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
    mask=(labels==idx).astype(np.uint8)
    # Fill internal holes so ink/paint cannot create artificial shape deficit.
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    mask2=np.zeros_like(mask)
    cv2.drawContours(mask2,contours,-1,1,thickness=cv2.FILLED)
    return mask2

def interp_profile(v):
    v=np.asarray(v,float)
    ok=np.isfinite(v)
    if ok.sum()<3: return v
    x=np.arange(len(v))
    return np.interp(x,x[ok],v[ok])

def boundary_profiles(mask):
    H,W=mask.shape
    ys=np.where(mask.any(axis=1))[0]
    xs=np.where(mask.any(axis=0))[0]
    y0,y1=ys[0],ys[-1]; x0,x1=xs[0],xs[-1]
    left=[];right=[]
    for y in range(y0,y1+1):
        xx=np.where(mask[y]>0)[0]
        left.append(xx[0] if len(xx) else np.nan)
        right.append(xx[-1] if len(xx) else np.nan)
    top=[];bottom=[]
    for x in range(x0,x1+1):
        yy=np.where(mask[:,x]>0)[0]
        top.append(yy[0] if len(yy) else np.nan)
        bottom.append(yy[-1] if len(yy) else np.nan)
    return (np.array(left,float),np.array(right,float),np.array(top,float),np.array(bottom,float),(x0,y0,x1,y1))

def detrended_rms(v, denom):
    v=interp_profile(v)
    n=len(v); x=np.linspace(-1,1,n)
    trim=max(2,int(.03*n)); sl=slice(trim,n-trim)
    co=np.polyfit(x[sl],v[sl],2)
    res=v[sl]-np.polyval(co,x[sl])
    # Smooth enough to suppress 1-pixel segmentation chatter but preserve mm-scale edge shape.
    sigma=max(1,n/700)
    res=gaussian_filter1d(res,sigma)
    return float(np.sqrt(np.mean(res**2))/denom)

def firstdiff_mad(v,denom):
    v=interp_profile(v)
    n=len(v); trim=max(2,int(.03*n)); v=v[trim:n-trim]
    sigma=max(1,n/700); v=gaussian_filter1d(v,sigma)
    d=np.diff(v)
    med=np.median(d); mad=np.median(np.abs(d-med))*1.4826
    return float(mad/denom)

def crease_features(img,mask,bbox):
    # Generic, label-free vertical crease candidate assay. Exploratory only: no old/current historical role assigned.
    x0,y0,x1,y1=bbox
    crop=img[y0:y1+1,x0:x1+1]
    mm=mask[y0:y1+1,x0:x1+1].astype(bool)
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).astype(float)
    sat=hsv[:,:,1]
    val=hsv[:,:,2]
    h,w=gray.shape
    ya,yb=int(.08*h),int(.92*h)
    sig=np.full(w,np.nan)
    for x in range(w):
        ok=mm[ya:yb,x] & (sat[ya:yb,x] < 85) & (val[ya:yb,x] > 45)
        if ok.sum() >= max(20,int(.08*(yb-ya))):
            sig[x]=np.median(gray[ya:yb,x][ok])
    sig=interp_profile(sig)
    if not np.isfinite(sig).all():
        return {"crease_n_peaks":0,"crease_max_z":np.nan,"crease_close_pair_z":np.nan,"crease_close_pair_sep":np.nan}
    small=gaussian_filter1d(sig,max(1,w/1200))
    broad=gaussian_filter1d(sig,max(8,w/45))
    hp=np.abs(small-broad)
    med=np.median(hp); mad=np.median(np.abs(hp-med))*1.4826 + 1e-9
    z=(hp-med)/mad
    peaks,props=find_peaks(z,distance=max(4,int(.012*w)),prominence=1.5)
    # Exclude near outer edges where segmentation/background dominates.
    peaks=np.array([p for p in peaks if .06*w<p<.94*w],dtype=int)
    top=sorted(peaks,key=lambda p:z[p],reverse=True)[:12]
    best=None
    for a,b in itertools.combinations(top,2):
        sep=abs(a-b)/w
        if .012 <= sep <= .080:
            sc=min(z[a],z[b])
            if best is None or sc>best[0]: best=(float(sc),float(sep))
    return {"crease_n_peaks":int(len(peaks)),"crease_max_z":float(max([z[p] for p in peaks],default=np.nan)),
            "crease_close_pair_z": best[0] if best else np.nan,
            "crease_close_pair_sep": best[1] if best else np.nan}

def features(seq,quire,name,sid,is_candidate):
    p=download(sid)
    img=cv2.imread(str(p),cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError(f"bad image {p}")
    mask=largest_parchment_mask(img)
    left,right,top,bottom,bbox=boundary_profiles(mask)
    x0,y0,x1,y1=bbox; bw=x1-x0+1; bh=y1-y0+1
    area=float(mask.sum())
    cnt=max(cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0],key=cv2.contourArea)
    hull=cv2.convexHull(cnt)
    hull_area=float(cv2.contourArea(hull))
    perimeter=float(cv2.arcLength(cnt,True)); hull_per=float(cv2.arcLength(hull,True))
    denom=float(bh)
    widths=right-left+1
    heights=bottom-top+1
    wr=np.std(widths)/np.mean(widths)
    hr=np.std(heights)/np.mean(heights)
    vals={
        "seq":seq,"quire":quire,"name":name,"service_id":sid,"candidate":is_candidate,
        "img_w":img.shape[1],"img_h":img.shape[0],"bbox_w":bw,"bbox_h":bh,
        "bbox_aspect":bw/bh,"area":area,
        "bbox_deficit":1-area/(bw*bh),
        "convexity_deficit":1-area/max(hull_area,1),
        "perimeter_excess":perimeter/max(hull_per,1)-1,
        "width_cv":float(wr),"height_cv":float(hr),
        "lr_rms":.5*(detrended_rms(left,denom)+detrended_rms(right,denom)),
        "tb_rms":.5*(detrended_rms(top,denom)+detrended_rms(bottom,denom)),
        "lr_diffmad":.5*(firstdiff_mad(left,denom)+firstdiff_mad(right,denom)),
        "tb_diffmad":.5*(firstdiff_mad(top,denom)+firstdiff_mad(bottom,denom)),
    }
    vals.update(crease_features(img,mask,bbox))
    return vals

def exact_label_test(df,metric,greater=True):
    vals=df[metric].to_numpy(float); lab=df.candidate.to_numpy(int); n1=lab.sum(); n=len(vals)
    obs=vals[lab==1].mean()-vals[lab==0].mean()
    null=[]
    idx=np.arange(n)
    for comb in itertools.combinations(idx,n1):
        g=np.zeros(n,dtype=bool); g[list(comb)]=True
        null.append(vals[g].mean()-vals[~g].mean())
    null=np.array(null)
    mu=float(null.mean()); sd=float(null.std(ddof=1)); z=(obs-mu)/sd if sd else np.nan
    if greater: p=(np.sum(null>=obs)+1)/(len(null)+1)
    else: p=(np.sum(null<=obs)+1)/(len(null)+1)
    return {"metric":metric,"observed_effect":float(obs),"null_mean":mu,"null_sd":sd,"effect_over_null_sd":float(z),"p_exact":float(p),"n_null":len(null)}

def loo_candidates(df,metric):
    out=[]
    cand=df[df.candidate==1]
    for seq in cand.seq:
        d=df[df.seq!=seq]
        r=exact_label_test(d,metric,True)
        r["left_out_seq"]=int(seq); out.append(r)
    return out

rows=[]
for rec in SCANS:
    print("MEASURE",rec,flush=True)
    rows.append(features(*rec))
df=pd.DataFrame(rows)
df.to_csv(OUT/"features.csv",index=False)

# Frozen primary: boundary RMS on lateral edges, size-normalised; detects large-scale non-rectilinear/natural edge shape.
# Secondary checks are different representations of the same physical proposition.
metrics=["lr_rms","tb_rms","bbox_deficit","convexity_deficit","perimeter_excess","width_cv","height_cv","lr_diffmad","tb_diffmad"]
tests=[exact_label_test(df,m,True) for m in metrics]
primary=next(x for x in tests if x["metric"]=="lr_rms")

# Representation dependence: candidate-v-control direction count across predeclared secondary metrics.
dir_count=sum(t["observed_effect"]>0 for t in tests)
# Leave-one-candidate-out bound on primary.
loo=loo_candidates(df,"lr_rms")

# Quire-balanced sensitivity: one candidate scan per quire cannot be selected without post-hoc choice for Q15/Q19.
# Therefore report quire means (Q14,Q15,Q17,Q19) vs control quire means only descriptively, not as p-value.
q=df.groupby(["candidate","quire"])[metrics].mean(numeric_only=True).reset_index()
q.to_csv(OUT/"quire_means.csv",index=False)

# Crease assay is exploratory: compare candidate close-pair score distribution only when finite. No confirmatory p-value.
crease=df[["seq","quire","name","candidate","crease_n_peaks","crease_max_z","crease_close_pair_z","crease_close_pair_sep"]].copy()
crease.to_csv(OUT/"crease_exploratory.csv",index=False)

summary={
 "protocol":PROTOCOL,
 "frozen_candidate_seqs":[158,162,164,170,178,180],
 "frozen_control_seqs":[159,160,161,165,167,168,169,172,173,174,175,176,177],
 "primary":primary,
 "secondary":tests,
 "primary_leave_one_candidate_out":loo,
 "secondary_direction_count":{"candidate_more_irregular":dir_count,"n_metrics":len(tests)},
 "crease_exploratory":crease.replace({np.nan:None}).to_dict(orient="records"),
 "limits":[
   "Candidate set is structurally selected as oversized/multi-panel, so bbox_aspect itself is not tested as evidence.",
   "RGB edge morphology cannot identify animal/skin batch.",
   "No public folio-level hair/flesh map exists; surface-side test remains unavailable.",
   "Generic crease detector does not assign historical old/current roles; only the pre-existing registered f89/f102 physical retest does that.",
   "Damage/trimming, especially f102, can alter edge morphology."
 ]
}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
