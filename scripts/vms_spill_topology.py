#!/usr/bin/env python3
"""VMS upper-margin spill topology measurement v02.

The v01 detector is RETRACTED: diagnostics showed scan-border/text/paint leakage.
v02 aligns every page to the physical parchment top edge and measures only a
narrow band *inside* the parchment. Dark ink and high-saturation pigment are
excluded before aggregation. No Voynich textual metadata enters the score.

Outputs remain physical diagnostics only. Linear/original order inference is
not licensed by this script.
"""
from __future__ import annotations

import csv, hashlib, io, json, math, os, random, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, median_filter

OUT = Path(os.environ.get("SPILL_OUT", "artifacts/vms_spill_v01"))
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ.get("SPILL_CACHE", ".cache/vms_spill_v02"))
CACHE.mkdir(parents=True, exist_ok=True)

PAGES: List[Tuple[str,int]]=[]
i=1006076
for f in range(1,12):
    for s in "rv": PAGES.append((f"f{f}{s}",i)); i+=1
for f in range(13,57):
    for s in "rv": PAGES.append((f"f{f}{s}",i)); i+=1
assert len(PAGES)==110 and i==1006186

BIFOLIA={
 "q01":[(1,8),(2,7),(3,6),(4,5)],
 "q02":[(9,16),(10,15),(11,14),(12,13)],
 "q03":[(17,24),(18,23),(19,22),(20,21)],
 "q04":[(25,32),(26,31),(27,30),(28,29)],
 "q05":[(33,40),(34,39),(35,38),(36,37)],
 "q06":[(41,48),(42,47),(43,46),(44,45)],
 "q07":[(49,56),(50,55),(51,54),(52,53)],
}

S=requests.Session(); S.headers.update({"User-Agent":"VoynichTopologyResearch/0.2"})

def fetch(label,iid,width=900):
    p=CACHE/f"{label}_{width}.jpg"
    if not p.exists():
        url=f"https://collections.library.yale.edu/iiif/2/{iid}/full/{width},/0/default.jpg"
        err=None
        for a in range(5):
            try:
                r=S.get(url,timeout=60); r.raise_for_status(); p.write_bytes(r.content); break
            except Exception as e:
                err=e; time.sleep(2**a)
        else: raise RuntimeError(f"fetch failed {label}: {err}")
    b=p.read_bytes(); return Image.open(io.BytesIO(b)).convert("RGB"), hashlib.sha256(b).hexdigest()

def top_edge(arr:np.ndarray)->np.ndarray:
    """Estimate top parchment boundary y(x), excluding outer scan margins."""
    g=.2126*arr[:,:,0]+.7152*arr[:,:,1]+.0722*arr[:,:,2]
    h,w=g.shape
    # smooth only enough to suppress hair/edge noise
    gs=gaussian_filter(g,sigma=(2.0,1.5),mode="nearest")
    sample=gs[:int(.32*h),int(.05*w):int(.95*w)]
    lo=float(np.quantile(sample,.08)); hi=float(np.quantile(sample,.82))
    thr=lo+.43*(hi-lo)
    e=np.full(w,np.nan,dtype=np.float32)
    maxy=int(.22*h)
    for x in range(w):
        col=gs[:maxy,x]
        # sustained brightness crossing: at least 4/5 pixels above threshold
        above=(col>thr).astype(np.int8)
        hit=np.convolve(above,np.ones(5,dtype=np.int8),mode="same")>=4
        ys=np.where(hit)[0]
        if len(ys): e[x]=ys[0]
    # fill from nearest valid central values; then robustly smooth physical edge
    valid=np.where(np.isfinite(e))[0]
    if len(valid)<.7*w: raise RuntimeError("parchment edge detection failed")
    e=np.interp(np.arange(w),valid,e[valid])
    e=median_filter(e,size=max(7,int(w/35))|1,mode="nearest")
    return e

def aligned_strip(arr:np.ndarray,e:np.ndarray,max_frac=.17):
    h,w,_=arr.shape; dmax=max(40,int(max_frac*h)); out=np.full((dmax,w,3),np.nan,np.float32)
    for x in range(w):
        y0=int(round(float(e[x])))
        n=min(dmax,h-y0)
        if n>0: out[:n,x]=arr[y0:y0+n,x]
    return out

def features(im:Image.Image)->Dict[str,object]:
    a=np.asarray(im).astype(np.float32)/255.
    h,w,_=a.shape
    # tiny horizontal trim only; top is deliberately not cropped
    x0,x1=int(.02*w),int(.98*w); a=a[:,x0:x1]; w=a.shape[1]
    e=top_edge(a)
    st=aligned_strip(a,e,.17)
    lum=.2126*st[:,:,0]+.7152*st[:,:,1]+.0722*st[:,:,2]
    mx=np.nanmax(st,axis=2); mn=np.nanmin(st,axis=2)
    sat=(mx-mn)/(mx+1e-6)
    yellow=(st[:,:,0]+st[:,:,1])/2-st[:,:,2]

    # Physical depths below parchment edge, expressed in page-height fractions.
    d0,d1=int(.008*h),int(.105*h)
    r0,r1=int(.115*h),min(st.shape[0],int(.165*h))
    if r1-r0<8: r0=max(d1,int(.11*h)); r1=st.shape[0]

    # Background is estimated columnwise from the deeper upper margin using only
    # parchment-like pixels.  High quantile luminance suppresses text/illustration.
    bgl=np.zeros(w,np.float32); bgy=np.zeros(w,np.float32)
    for x in range(w):
        lv=lum[r0:r1,x]; sv=sat[r0:r1,x]; yv=yellow[r0:r1,x]
        ok=np.isfinite(lv)&(lv>.42)&(sv<.28)
        if ok.sum()>=4:
            bgl[x]=np.quantile(lv[ok],.80); bgy[x]=np.median(yv[ok])
        else:
            allok=np.isfinite(lv)
            bgl[x]=np.quantile(lv[allok],.80) if allok.any() else .75
            bgy[x]=np.median(yv[allok]) if allok.any() else .08
    bgl=gaussian_filter(bgl,sigma=max(2,w/100),mode="nearest")
    bgy=gaussian_filter(bgy,sigma=max(2,w/100),mode="nearest")

    L=lum[d0:d1]; Y=yellow[d0:d1]; SAT=sat[d0:d1]
    dark=np.clip(bgl[None,:]-L,0,.18)
    warm=np.clip(Y-bgy[None,:],0,.12)
    score=.78*dark+.22*warm

    # Exclude obvious ink/pigment and edge failures *before* low-dimensional aggregation.
    valid=np.isfinite(score)&np.isfinite(L)&(L>.38)&(SAT<.30)
    # Local gradients catch residual glyph/pigment edges without removing broad stain.
    lf=np.nan_to_num(L,nan=.8)
    gy,gx=np.gradient(gaussian_filter(lf,sigma=1.0))
    grad=np.hypot(gx,gy)
    valid &= grad<.055
    score=np.where(valid,score,np.nan)

    # Broad low-frequency stain field; normalized convolution avoids NaN bleed.
    num=gaussian_filter(np.nan_to_num(score,nan=0.),sigma=(2.2,4.0),mode="nearest")
    den=gaussian_filter(np.isfinite(score).astype(np.float32),sigma=(2.2,4.0),mode="nearest")
    smooth=np.where(den>.25,num/np.maximum(den,1e-6),np.nan)

    ny,nx=6,32; grid=np.zeros((ny,nx),np.float32); coverage=np.zeros((ny,nx),np.float32)
    for iy in range(ny):
        ya,yb=int(iy*smooth.shape[0]/ny),int((iy+1)*smooth.shape[0]/ny)
        for ix in range(nx):
            xa,xb=int(ix*w/nx),int((ix+1)*w/nx)
            v=smooth[ya:yb,xa:xb]; good=np.isfinite(v)
            coverage[iy,ix]=good.mean()
            grid[iy,ix]=np.nanmedian(v) if good.any() else 0.
    profile=np.nanmedian(np.where(np.isfinite(smooth),smooth,np.nan),axis=0)
    pb=np.zeros(nx,np.float32)
    for ix in range(nx):
        xa,xb=int(ix*w/nx),int((ix+1)*w/nx)
        v=profile[xa:xb]; pb[ix]=np.nanmedian(v) if np.isfinite(v).any() else 0.

    vals=smooth[np.isfinite(smooth)]
    return {
      "stain_mean":float(np.mean(vals)),"stain_p90":float(np.quantile(vals,.9)),
      "stain_area_015":float(np.mean(vals>.015)),"stain_area_025":float(np.mean(vals>.025)),
      "valid_fraction":float(np.mean(valid)),"edge_y_mean":float(np.mean(e)/h),
      "edge_y_sd":float(np.std(e)/h),"grid":grid,"profile":pb,
      "aligned":st,"score":smooth,"valid":valid,"height":h,"width":w,
    }

def standardize(X):
    med=np.median(X,axis=0); mad=np.median(np.abs(X-med),axis=0)
    sc=np.where(mad>1e-6,1.4826*mad,np.std(X,axis=0)+1e-6)
    return (X-med)/np.where(sc>1e-6,sc,1.)

def dmat(X):
    Z=standardize(X); d=Z[:,None,:]-Z[None,:,:]; return np.sqrt(np.mean(d*d,axis=2))

def energy(order,D): return float(np.mean([D[order[i],order[i+1]] for i in range(len(order)-1)]))
def ptest(order,D,rng,n):
    obs=energy(order,D); vals=[]
    for _ in range(n):
        p=order[:]; rng.shuffle(p); vals.append(energy(p,D))
    a=np.asarray(vals); m=float(a.mean()); sd=float(a.std(ddof=1))
    return {"observed":obs,"null_mean":m,"null_sd":sd,"z":float((obs-m)/sd) if sd else None,
            "p_lower":float((1+(a<=obs).sum())/(n+1)),"n_perm":n}

def side_vec(d): return np.concatenate([d["grid"].ravel(),d["profile"]])

def contact(rows,labels,path):
    W=420; H=145; block=3*H+24; sh=Image.new("RGB",(2*W,math.ceil(len(labels)/2)*block),"white"); dr=ImageDraw.Draw(sh)
    for k,l in enumerate(labels):
        rr,cc=divmod(k,2); x=cc*W; y=rr*block; d=rows[l]
        im=Image.fromarray(np.uint8(np.clip(d["aligned"][:int(.11*d["height"])],0,1)*255)).resize((W,H))
        sh.paste(im,(x,y+20))
        sc=d["score"]; q=np.nan_to_num(np.clip(sc/.045,0,1),nan=0.)
        valid=np.isfinite(sc); heat=np.zeros((sc.shape[0],sc.shape[1],3),np.uint8)+220
        heat[:,:,0]=255; heat[:,:,1]=np.uint8(255*(1-q)); heat[:,:,2]=np.uint8(255*(1-q)); heat[~valid]=180
        sh.paste(Image.fromarray(heat).resize((W,H)),(x,y+20+H))
        # binary-ish threshold view for contour plausibility
        th=np.zeros_like(heat)+255; on=np.isfinite(sc)&(sc>.015); th[on]=[180,40,40]; th[~np.isfinite(sc)]=[190,190,190]
        sh.paste(Image.fromarray(th).resize((W,H)),(x,y+20+2*H))
        dr.text((x+4,y+2),f"{l} mean={d['stain_mean']:.4f} area={d['stain_area_015']:.3f} valid={d['valid_fraction']:.2f}",fill="black")
    sh.save(path,quality=91)

def main():
    rows={}; hashes={}
    for label,iid in PAGES:
        im,hh=fetch(label,iid); hashes[label]=hh; rows[label]=features(im)
        print(label,rows[label]["stain_mean"],rows[label]["valid_fraction"],flush=True)

    scal=["stain_mean","stain_p90","stain_area_015","stain_area_025","valid_fraction","edge_y_mean","edge_y_sd"]
    with (OUT/"spill_page_features.csv").open("w",newline="") as f:
        wr=csv.writer(f); wr.writerow(["page_id","iiif_id","sha256",*scal,*[f"profile_{i:02d}" for i in range(32)]])
        for l,iid in PAGES: wr.writerow([l,iid,hashes[l],*[rows[l][c] for c in scal],*rows[l]["profile"].tolist()])

    folios=list(range(1,12))+list(range(13,57)); F=[]; R=[]; V=[]; fm={}
    for fol in folios:
        r,v=rows[f"f{fol}r"],rows[f"f{fol}v"]
        rv,vv=side_vec(r),side_vec(v); R.append(rv); V.append(vv); F.append((rv+vv)/2)
        fm[fol]={c:(r[c]+v[c])/2 for c in scal}
    F=np.stack(F); R=np.stack(R); V=np.stack(V); D=dmat(F); Dr=dmat(R); Dv=dmat(V); fi={f:i for i,f in enumerate(folios)}
    rng=random.Random(20260907); cur=list(range(len(folios)))
    tests={"folio_mean":ptest(cur,D,rng,20000),"recto_only":ptest(cur,Dr,rng,10000),"verso_only":ptest(cur,Dv,rng,10000)}
    qt={}
    for q,pairs in BIFOLIA.items():
        leaves=sorted({z for p in pairs for z in p if z in fi}); idx=[fi[z] for z in leaves]
        qt[q]=ptest(idx,D,rng,10000); qt[q]["folios"]=leaves

    # Reliability: scalar and full-profile agreement between two sides of the same folio.
    ra=np.array([rows[f"f{x}r"]["stain_mean"] for x in folios]); va=np.array([rows[f"f{x}v"]["stain_mean"] for x in folios])
    rm=np.array([rows[f"f{x}r"]["stain_area_015"] for x in folios]); vm=np.array([rows[f"f{x}v"]["stain_area_015"] for x in folios])
    profcorr=[]
    for fol in folios:
        a=rows[f"f{fol}r"]["profile"]; b=rows[f"f{fol}v"]["profile"]
        profcorr.append(float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-8 and np.std(b)>1e-8 else 0.)

    # Specific f32v/f33r discontinuity relative to consecutive-facing v->r pairs in ff1-56.
    side_labels=[l for l,_ in PAGES]; SX=np.stack([side_vec(rows[l]) for l in side_labels]); SD=dmat(SX); si={l:i for i,l in enumerate(side_labels)}
    face=[]
    for a,b in zip(folios[:-1],folios[1:]):
        if b==a+1: face.append(float(SD[si[f"f{a}v"],si[f"f{b}r"]]))
    target=float(SD[si["f32v"],si["f33r"]]); fa=np.asarray(face); fz=float((target-fa.mean())/fa.std(ddof=1))

    conjoint=[]
    for q,pairs in BIFOLIA.items():
        leaves=sorted({z for p in pairs for z in p if z in fi}); allp=[D[fi[leaves[a]],fi[leaves[b]]] for a in range(len(leaves)) for b in range(a+1,len(leaves))]
        obs=[D[fi[x],fi[y]] for x,y in pairs if x in fi and y in fi]; sd=float(np.std(allp,ddof=1))
        conjoint.append({"quire":q,"obs_mean":float(np.mean(obs)),"allpair_mean":float(np.mean(allp)),"allpair_sd":sd,
                         "effect":float(np.mean(obs)-np.mean(allp)),"z":float((np.mean(obs)-np.mean(allp))/sd) if sd else None,"n_obs":len(obs)})

    summary={
      "protocol":"vms_spill_topology_s2_20260907","retracts":"vms_spill_topology_s1_20260907",
      "source":"Beinecke IIIF","n_page_sides":110,"n_folios":55,"missing_folios":[12],
      "reliability":{"recto_verso_stain_mean_r":float(np.corrcoef(ra,va)[0,1]),"recto_verso_area_r":float(np.corrcoef(rm,vm)[0,1]),
                     "median_recto_verso_profile_r":float(np.median(profcorr)),"mean_recto_verso_profile_r":float(np.mean(profcorr)),
                     "median_valid_fraction":float(np.median([rows[l]["valid_fraction"] for l in side_labels]))},
      "current_order_tests":tests,"quire_current_order_tests":qt,"conjoint_similarity":conjoint,
      "f32v_f33r":{"distance":target,"facing_pair_mean":float(fa.mean()),"facing_pair_sd":float(fa.std(ddof=1)),"z_vs_facing_pairs":fz,"n_reference_pairs":len(face)},
      "license":"PHYSICAL_DIAGNOSTIC_ONLY__NO_LINEAR_ORDER_INFERENCE",
      "decision_rule":"Reject detector if diagnostic sheet retains material scan-border/text/paint leakage OR recto/verso profile agreement is poor; do not optimize historical order unless detector passes.",
    }
    (OUT/"spill_summary.json").write_text(json.dumps(summary,indent=2))
    with (OUT/"spill_folio_distance.csv").open("w",newline="") as f:
        wr=csv.writer(f); wr.writerow(["folio",*folios]); [wr.writerow([fol,*D[j].tolist()]) for j,fol in enumerate(folios)]
    diag=["f1r","f8v","f9r","f16v","f17r","f24v","f25r","f31v","f32v","f33r","f34v","f40v","f41r","f48v","f49r","f56v"]
    contact(rows,diag,OUT/"spill_diagnostic_contact.jpg")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
