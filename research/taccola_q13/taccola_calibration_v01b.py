#!/usr/bin/env python3
"""Taccola x Q13 programme: calibration only. Q13 is intentionally absent.

Frozen design:
- release metadata locked 2026-08-26-r15 before visual scoring
- two image-rich Taccola witnesses: Clm 197 II and Palatino 766
- six technical controls, four Italian subject controls, 24 deterministic European/date controls
- equal page sampling and deterministic illustration gate
- three classical representations: HOG, symmetric chamfer, geometry
- manuscript is the inferential unit; no page-level pseudo-replication
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import pickle
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

VERSION = "taccola-q13-calibration-v0.1b-transport-repair-20260830"
RELEASE_ID = "2026-08-26-r15"
PAGE_SAMPLE = 60
ILLUSTRATION_TOP_K = 24
TOP_MATCHES = 8
BOOTSTRAPS = 200
RNG_SEED = 20260830
IMAGE_WIDTH = 900
USER_AGENT = "ManuComp-TaccolaCalibration/0.1b (+https://github.com/digitalgoldfisj79/Voynichdecomp)"

# Positive labels and controls are locked in a separate pre-score panel commit.
# v0.1b changes transport-accessibility only; no similarity score existed when repaired.
PANEL_PATH = Path(__file__).with_name("panel_v01b.json")
EXPECTED_PANEL_SHA256 = "8226f0435ee07d8af1e4b0d8cb4a8f09af8f7a82b32bf04441f8dab7ae49c905"
LOCKED_PANEL = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
assert LOCKED_PANEL["version"] == VERSION
assert LOCKED_PANEL["release_id"] == RELEASE_ID
assert LOCKED_PANEL["page_sample"] == PAGE_SAMPLE
assert LOCKED_PANEL["illustration_top_k"] == ILLUSTRATION_TOP_K
assert LOCKED_PANEL["top_matches"] == TOP_MATCHES
assert LOCKED_PANEL["bootstraps"] == BOOTSTRAPS
assert LOCKED_PANEL["repair_provenance"]["q13_sealed"] is True
MANUSCRIPTS = LOCKED_PANEL["manuscripts"]

TECH_IDS = [k for k,v in MANUSCRIPTS.items() if v["stratum"] == "technical"]
ITALY_IDS = [k for k,v in MANUSCRIPTS.items() if v["stratum"] == "italian"]
NULL_IDS = [k for k,v in MANUSCRIPTS.items() if v["role"] == "control"]


def sha256_json(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def robust_get(session, url, *, timeout=40, attempts=5):
    last = None
    for i in range(attempts):
        try:
            r = session.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            time.sleep(min(12.0, 0.8 * (2 ** i)))
    raise last


def label_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        if "en" in x and isinstance(x["en"], list):
            return " ".join(map(str, x["en"]))
        vals=[]
        for v in x.values():
            if isinstance(v, list): vals.extend(map(str,v))
            else: vals.append(str(v))
        return " ".join(vals)
    return str(x)


def iiif_image_from_canvas(canvas, width=IMAGE_WIDTH):
    """Return a width-bounded IIIF image URL from v2 or v3 canvas JSON."""
    label = label_text(canvas.get("label"))
    body = None
    # IIIF v3: canvas.items -> AnnotationPage.items -> Annotation.body
    try:
        body = canvas["items"][0]["items"][0]["body"]
        if isinstance(body, list): body = body[0]
    except Exception:
        pass
    # IIIF v2: canvas.images[0].resource
    if body is None:
        try:
            body = canvas["images"][0]["resource"]
        except Exception:
            body = None
    if not isinstance(body, dict):
        return None, label
    services = body.get("service") or body.get("services") or []
    if isinstance(services, dict): services = [services]
    service_id = None
    for s in services:
        if isinstance(s, dict):
            service_id = s.get("id") or s.get("@id")
            if service_id: break
    if service_id:
        return service_id.rstrip("/") + f"/full/{width},/0/default.jpg", label
    bid = body.get("id") or body.get("@id")
    return (bid, label) if bid else (None, label)


def manifest_canvases(session, url):
    m = robust_get(session, url, timeout=60).json()
    if isinstance(m.get("items"), list):
        canvases = m["items"]
    else:
        seqs = m.get("sequences") or []
        canvases = seqs[0].get("canvases", []) if seqs else []
    rows=[]
    for idx,c in enumerate(canvases):
        u,label = iiif_image_from_canvas(c)
        if u:
            rows.append({"index":idx,"label":label,"url":u})
    return rows


def even_sample(rows, n=PAGE_SAMPLE):
    if len(rows) <= n: return list(rows)
    # Fixed midpoint bins avoid cover-heavy endpoints and random variation.
    positions = [int((i + 0.5) * len(rows) / n) for i in range(n)]
    positions = [min(len(rows)-1, max(0,p)) for p in positions]
    out=[]; seen=set()
    for p in positions:
        if p not in seen:
            out.append(rows[p]); seen.add(p)
    return out


def fetch_gray(url):
    s=requests.Session()
    raw=robust_get(s,url,timeout=50).content
    im=Image.open(io.BytesIO(raw)).convert("L")
    a=np.asarray(im,dtype=np.uint8)
    if a.shape[1] > IMAGE_WIDTH:
        h=max(1,int(round(a.shape[0]*IMAGE_WIDTH/a.shape[1])))
        a=cv2.resize(a,(IMAGE_WIDTH,h),interpolation=cv2.INTER_AREA)
    return a


def entropy1d(v):
    v=np.asarray(v,dtype=np.float64)
    s=v.sum()
    if s<=0:return 0.0
    p=v/s; p=p[p>0]
    return float(-(p*np.log(p)).sum()/max(1e-9,math.log(len(v))))


def hog_vec(mask, cell=16, bins=9):
    img=cv2.resize(mask.astype(np.uint8)*255,(128,128),interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
    gx=cv2.Sobel(img,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(img,cv2.CV_32F,0,1,ksize=3)
    mag,ang=cv2.cartToPolar(gx,gy,angleInDegrees=True); ang=np.mod(ang,180.0)
    ny=128//cell; nx=128//cell; hist=np.zeros((ny,nx,bins),np.float32); bw=180.0/bins
    for cy in range(ny):
        for cx in range(nx):
            ys=slice(cy*cell,(cy+1)*cell); xs=slice(cx*cell,(cx+1)*cell)
            m=mag[ys,xs].ravel(); aa=ang[ys,xs].ravel()/bw
            b0=np.floor(aa).astype(np.int32)%bins; fr=aa-np.floor(aa); b1=(b0+1)%bins
            np.add.at(hist[cy,cx],b0,m*(1-fr)); np.add.at(hist[cy,cx],b1,m*fr)
    blocks=[]
    for cy in range(ny-1):
        for cx in range(nx-1):
            b=hist[cy:cy+2,cx:cx+2].ravel(); b=b/(np.linalg.norm(b)+1e-6); blocks.append(b)
    return np.concatenate(blocks).astype(np.float32)


def page_features(gray):
    # Border removal + illumination normalization removes page/photography background.
    H0,W0=gray.shape
    y0=int(.04*H0); y1=max(y0+10,int(.96*H0)); x0=int(.04*W0); x1=max(x0+10,int(.96*W0))
    g=gray[y0:y1,x0:x1]
    if g.size<100:return None
    bg=cv2.GaussianBlur(g,(0,0),25)
    norm=cv2.divide(g,bg,scale=240)
    norm=np.clip(norm,0,255).astype(np.uint8)
    # Otsu threshold, but clamp to avoid background staining becoming "ink".
    t,_=cv2.threshold(norm,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    t=min(225,max(155,int(t)))
    ink=(norm<t).astype(np.uint8)
    H,W=ink.shape
    # Connect strokes locally; reject text-line-like and tiny components.
    d=cv2.dilate(ink,np.ones((3,3),np.uint8),iterations=2)
    n,lab,stats,cent=cv2.connectedComponentsWithStats(d,8)
    keep=np.zeros_like(ink,dtype=np.uint8)
    kept=[]
    for k in range(1,n):
        x,y,w,h,area=stats[k]
        af=area/(H*W)
        if af < 0.00055: continue
        if h/H < 0.028 and w/W > 0.18: continue  # text-line-like
        if h/H < 0.045 and w/W < 0.075: continue # word/small marginalia
        if max(h/H,w/W) < 0.075: continue
        region=(lab==k)
        # Bring back only original ink within the retained dilated component.
        keep[region & (ink>0)] = 1
        kept.append((x,y,w,h,area))
    had_large_components = bool(kept)
    if not kept:
        # Preserve deterministic low-information representation; gate score remains zero.
        keep=ink.copy()
    ys,xs=np.where(keep>0)
    if len(xs):
        bx0,bx1,by0,by1=xs.min(),xs.max()+1,ys.min(),ys.max()+1
        crop=keep[by0:by1,bx0:bx1]
        # normalize translation/scale for shape reps
        side=max(crop.shape); pad=np.zeros((side,side),np.uint8)
        oy=(side-crop.shape[0])//2; ox=(side-crop.shape[1])//2
        pad[oy:oy+crop.shape[0],ox:ox+crop.shape[1]]=crop
    else:
        pad=np.zeros((32,32),np.uint8); bx0=by0=0; bx1=by1=1
    shape=cv2.resize(pad,(96,96),interpolation=cv2.INTER_NEAREST)>0
    # Distance transform precomputed for symmetric chamfer.
    dt=cv2.distanceTransform((~shape).astype(np.uint8),cv2.DIST_L2,3).astype(np.float32)/math.sqrt(96**2+96**2)
    hvec=hog_vec(shape)
    # Geometry representation.
    inkfrac=float(keep.mean())
    bboxfrac=float(((bx1-bx0)*(by1-by0))/(H*W)) if len(xs) else 0.0
    cx=float(xs.mean()/W) if len(xs) else .5; cy=float(ys.mean()/H) if len(xs) else .5
    sx=float(xs.std()/W) if len(xs) else 0.; sy=float(ys.std()/H) if len(xs) else 0.
    q=[]
    for yy in (slice(0,H//2),slice(H//2,H)):
        for xx in (slice(0,W//2),slice(W//2,W)):
            q.append(float(keep[yy,xx].mean()))
    rows=keep.sum(axis=1); cols=keep.sum(axis=0)
    # Orientation histogram on illustration mask.
    sm=(keep*255).astype(np.float32); gx=cv2.Sobel(sm,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(sm,cv2.CV_32F,0,1,ksize=3)
    mag,ang=cv2.cartToPolar(gx,gy,angleInDegrees=True); ang=np.mod(ang,180.0)
    oh=[]
    for lo in np.linspace(0,180,13)[:-1]:
        hi=lo+15; oh.append(float(mag[(ang>=lo)&(ang<hi)].sum()))
    oh=np.asarray(oh,np.float64); oh=(oh/(oh.sum()+1e-9)).tolist()
    compareas=sorted([a/(H*W) for *_,a in kept]) if kept else [0.0]
    qs=np.quantile(compareas,[0,.25,.5,.75,1]).tolist()
    geom=np.array([inkfrac,bboxfrac,cx,cy,sx,sy,entropy1d(rows),entropy1d(cols),len(kept)/100.0,*q,*oh,*qs],dtype=np.float32)
    # Illustration gate score is label-free and independent of similarity target.
    illustration_score=(bboxfrac*math.sqrt(max(inkfrac,1e-9))) if had_large_components else 0.0
    return {"hog":hvec,"shape":shape,"dt":dt,"geom":geom,"illustration_score":illustration_score,"inkfrac":inkfrac,"bboxfrac":bboxfrac,"component_count":len(kept)}


def cosine(a,b):
    a=np.asarray(a,np.float32); b=np.asarray(b,np.float32)
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    return float(np.dot(a,b)/(na*nb+1e-9))


def chamfer(a,b):
    ma=a["shape"]; mb=b["shape"]
    if not ma.any() or not mb.any():return 0.0
    d1=float(b["dt"][ma].mean()); d2=float(a["dt"][mb].mean())
    d=.5*(d1+d2)
    return float(math.exp(-18.0*d))


def pair_matrix(A,B,rep,geom_mu=None,geom_sd=None):
    out=np.zeros((len(A),len(B)),np.float32)
    if rep=="hog":
        for i,a in enumerate(A):
            for j,b in enumerate(B): out[i,j]=cosine(a["hog"],b["hog"])
    elif rep=="chamfer":
        for i,a in enumerate(A):
            for j,b in enumerate(B): out[i,j]=chamfer(a,b)
    elif rep=="geometry":
        for i,a in enumerate(A):
            za=(a["geom"]-geom_mu)/geom_sd
            for j,b in enumerate(B):
                zb=(b["geom"]-geom_mu)/geom_sd
                # distance -> bounded similarity; scale by dimension
                d=float(np.sqrt(np.mean((za-zb)**2)))
                out[i,j]=math.exp(-d)
    else: raise ValueError(rep)
    return out


def manuscript_score(mat,top_matches=TOP_MATCHES):
    if mat.size==0:return float("nan")
    # Symmetric best-page coverage; top-k captures a shared motif subset without page-count advantage.
    a=np.max(mat,axis=0); b=np.max(mat,axis=1)
    k1=min(top_matches,len(a)); k2=min(top_matches,len(b))
    sa=float(np.mean(np.sort(a)[-k1:])); sb=float(np.mean(np.sort(b)[-k2:]))
    return .5*(sa+sb)


def z_and_p(pos, null):
    null=np.asarray([x for x in null if np.isfinite(x)],np.float64)
    mu=float(null.mean()); sd=float(null.std(ddof=1)) if len(null)>1 else float("nan")
    z=float((pos-mu)/sd) if sd>0 else float("nan")
    p=float((1+np.sum(null>=pos))/(1+len(null)))
    return {"positive":float(pos),"null_mean":mu,"null_sd":sd,"effect":float(pos-mu),"z":z,"empirical_block_p":p,"n_null":int(len(null))}


def boot_stability(mats, null_ids, reps, n=BOOTSTRAPS, seed=RNG_SEED):
    rng=np.random.default_rng(seed); passes=0; zs=[]
    for _ in range(n):
        rep_pos=[]; rep_null={k:[] for k in null_ids}
        for rep in reps:
            pm=mats[rep]["positive"]
            ia=rng.integers(0,pm.shape[0],pm.shape[0]); ib=rng.integers(0,pm.shape[1],pm.shape[1])
            ps=manuscript_score(pm[np.ix_(ia,ib)])
            ns=[]
            for k in null_ids:
                m=mats[rep][k]
                i1=rng.integers(0,m.shape[0],m.shape[0]); i2=rng.integers(0,m.shape[1],m.shape[1])
                s=manuscript_score(m[np.ix_(i1,i2)]); ns.append(s); rep_null[k].append(s)
            st=z_and_p(ps,ns); rep_pos.append(st["z"])
        # Equal-weight standardized representation composite.
        pcomp=float(np.mean(rep_pos))
        # approximate null composite using within-bootstrap rep z values
        null_comp=[]
        for k in null_ids:
            zks=[]
            for ri,rep in enumerate(reps):
                vals=np.array([rep_null[j][ri] for j in null_ids])
                mu=vals.mean(); sd=vals.std(ddof=1)
                zks.append((rep_null[k][ri]-mu)/(sd+1e-9))
            null_comp.append(float(np.mean(zks)))
        sdnc=float(np.std(null_comp,ddof=1)); zc=pcomp/(sdnc+1e-9)
        zs.append(zc); passes += int(zc>=2.0)
    return {"n":n,"fraction_composite_z_ge_2":passes/n,"median_z":float(np.median(zs)),"q05_z":float(np.quantile(zs,.05)),"q95_z":float(np.quantile(zs,.95))}


def build_direction(template_id, positive_id, feats, geom_mu, geom_sd):
    reps=["hog","chamfer","geometry"]
    ids=[positive_id]+NULL_IDS+["dresden_mixed"]
    mats={r:{} for r in reps}; scores={r:{} for r in reps}; stats={}
    for rep in reps:
        for cid in ids:
            m=pair_matrix(feats[template_id],feats[cid],rep,geom_mu,geom_sd)
            key="positive" if cid==positive_id else cid
            mats[rep][key]=m; scores[rep][cid]=manuscript_score(m)
        null=[scores[rep][k] for k in NULL_IDS]
        stats[rep]=z_and_p(scores[rep][positive_id],null)
        stats[rep]["technical"] = z_and_p(scores[rep][positive_id],[scores[rep][k] for k in TECH_IDS])
        stats[rep]["italian"] = z_and_p(scores[rep][positive_id],[scores[rep][k] for k in ITALY_IDS])
        stats[rep]["positive_rank_all"] = 1+sum(scores[rep][k]>scores[rep][positive_id] for k in NULL_IDS)
        stats[rep]["derivative_score"] = scores[rep]["dresden_mixed"]
    # Equal-weight composite of manuscript-level z scores per representation.
    comp={}
    rep_null_mu={r:np.mean([scores[r][k] for k in NULL_IDS]) for r in reps}
    rep_null_sd={r:np.std([scores[r][k] for k in NULL_IDS],ddof=1) for r in reps}
    for cid in [positive_id]+NULL_IDS+["dresden_mixed"]:
        comp[cid]=float(np.mean([(scores[r][cid]-rep_null_mu[r])/(rep_null_sd[r]+1e-9) for r in reps]))
    cnull=[comp[k] for k in NULL_IDS]
    cst=z_and_p(comp[positive_id],cnull)
    cst["technical_rank"] = 1+sum(comp[k]>comp[positive_id] for k in TECH_IDS)
    cst["italian_rank"] = 1+sum(comp[k]>comp[positive_id] for k in ITALY_IDS)
    cst["derivative_composite"] = comp["dresden_mixed"]
    # Representation dependence on null manuscripts.
    corr={}
    for i,r1 in enumerate(reps):
        for r2 in reps[i+1:]:
            x=np.array([scores[r1][k] for k in NULL_IDS]); y=np.array([scores[r2][k] for k in NULL_IDS])
            corr[f"{r1}__{r2}"]=float(np.corrcoef(x,y)[0,1])
    # Decision-rule fragility: top matches 5/8/12.
    sens={}
    for topk in (5,8,12):
        ss={r:{} for r in reps}
        for r in reps:
            for cid in [positive_id]+NULL_IDS:
                key="positive" if cid==positive_id else cid
                ss[r][cid]=manuscript_score(mats[r][key],topk)
        pc=[]; nc={k:[] for k in NULL_IDS}
        for r in reps:
            null=np.array([ss[r][k] for k in NULL_IDS]); mu=null.mean(); sd=null.std(ddof=1)
            pc.append((ss[r][positive_id]-mu)/(sd+1e-9))
            for k in NULL_IDS:nc[k].append((ss[r][k]-mu)/(sd+1e-9))
        posc=float(np.mean(pc)); ncs=np.array([np.mean(nc[k]) for k in NULL_IDS]);
        sens[str(topk)]={"positive_standardized_composite":posc,"null_sd":float(ncs.std(ddof=1)),"z":float(posc/(ncs.std(ddof=1)+1e-9)),"p":float((1+np.sum(ncs>=posc))/(1+len(ncs)))}
    boot=boot_stability(mats,NULL_IDS,reps)
    return {"template":template_id,"positive":positive_id,"scores":scores,"representation_stats":stats,"composite_scores":comp,"composite_stats":cst,"null_rep_correlations":corr,"sensitivity_top_matches":sens,"bootstrap":boot}


def main():
    global NULL_IDS
    outdir=Path("taccola_calibration_output"); outdir.mkdir(exist_ok=True)
    panel=json.loads(json.dumps(LOCKED_PANEL))
    panel_hash=sha256_json(panel)
    if panel_hash != EXPECTED_PANEL_SHA256:
        raise RuntimeError(f"locked panel checksum mismatch: {panel_hash}")
    panel["panel_sha256"]=panel_hash
    (outdir/"panel.json").write_text(json.dumps(panel,indent=2,ensure_ascii=False))
    print(json.dumps({"event":"panel_locked","sha256":panel_hash,"n_manuscripts":len(MANUSCRIPTS),"n_null":len(NULL_IDS)}),flush=True)

    sess=requests.Session(); manifest_rows={}; errors=[]
    for mid,meta in MANUSCRIPTS.items():
        try:
            rows=manifest_canvases(sess,meta["manifest"]); sampled=even_sample(rows,PAGE_SAMPLE)
            manifest_rows[mid]=sampled
            print(json.dumps({"event":"manifest","id":mid,"pages":len(rows),"sample":len(sampled)}),flush=True)
        except Exception as e:
            errors.append({"stage":"manifest","id":mid,"error":repr(e)}); manifest_rows[mid]=[]
            print(json.dumps({"event":"manifest_error","id":mid,"error":repr(e)}),flush=True)
        safe_mid=re.sub(r"[^A-Za-z0-9_.-]+","_",mid)
        with open(outdir/f"checkpoint_01_manifest_{safe_mid}.pkl","wb") as f:
            pickle.dump({"panel_sha256":panel_hash,"id":mid,"rows":manifest_rows[mid],"errors":[x for x in errors if x.get("id")==mid]},f,protocol=5)
    with open(outdir/"checkpoint_01_manifests.pkl","wb") as f:pickle.dump({"panel":panel,"manifest_rows":manifest_rows,"errors":errors},f,protocol=5)

    feats={}; page_diag={}
    for mid,rows in manifest_rows.items():
        got=[]; diag=[]
        def work(r):
            try:
                g=fetch_gray(r["url"]); f=page_features(g)
                if f is None: raise ValueError("feature extraction returned None")
                return r,f,None
            except Exception as e:return r,None,repr(e)
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs=[ex.submit(work,r) for r in rows]
            for n,fu in enumerate(as_completed(futs),1):
                r,f,e=fu.result()
                if f is not None:
                    f["index"]=r["index"]; f["label"]=r["label"]; got.append(f)
                    diag.append({"index":r["index"],"label":r["label"],"illustration_score":f["illustration_score"],"inkfrac":f["inkfrac"],"bboxfrac":f["bboxfrac"],"components":f["component_count"]})
                else: errors.append({"stage":"image","id":mid,"index":r.get("index"),"error":e})
        got.sort(key=lambda x:(-x["illustration_score"],x["index"])); got=got[:ILLUSTRATION_TOP_K]
        feats[mid]=got; page_diag[mid]=sorted(diag,key=lambda x:x["index"])
        print(json.dumps({"event":"features","id":mid,"download_ok":len(diag),"selected":len(got),"errors_total":len(errors)}),flush=True)
        safe_mid=re.sub(r"[^A-Za-z0-9_.-]+","_",mid)
        with open(outdir/f"checkpoint_02_features_{safe_mid}.pkl","wb") as f:
            pickle.dump({"panel_sha256":panel_hash,"id":mid,"features":got,"page_diag":page_diag[mid],"errors":[x for x in errors if x.get("id")==mid]},f,protocol=5)
    with open(outdir/"checkpoint_02_features.pkl","wb") as f:pickle.dump({"panel":panel,"features":feats,"page_diag":page_diag,"errors":errors},f,protocol=5)
    (outdir/"page_diagnostics.json").write_text(json.dumps(page_diag,indent=2))

    # Audit completeness before calibration.
    required=["clm197","pal766"]+TECH_IDS+ITALY_IDS
    missing=[k for k in required if len(feats.get(k,[]))<12]
    processed_null=[k for k in NULL_IDS if len(feats.get(k,[]))>=12]
    # Freeze active nulls to only manuscripts that passed a label-blind minimum-data criterion.
    original_null=list(NULL_IDS); NULL_IDS=processed_null
    audit={"required_missing_or_lt12":missing,"original_n_null":len(original_null),"active_n_null":len(NULL_IDS),"failed_nulls":[k for k in original_null if k not in NULL_IDS],"image_errors":len([e for e in errors if e["stage"]=="image"]),"manifest_errors":len([e for e in errors if e["stage"]=="manifest"])}
    if missing or len(NULL_IDS)<30 or any(k not in NULL_IDS for k in (TECH_IDS+ITALY_IDS)):
        audit["calibration_allowed"]=False
        result={"version":VERSION,"panel_sha256":panel_hash,"audit":audit,"status":"CALIBRATION_BLOCKED_INCOMPLETE","errors":errors[:200]}
        (outdir/"calibration.json").write_text(json.dumps(result,indent=2))
        raise SystemExit("Calibration blocked by completeness gate")
    audit["calibration_allowed"]=True

    allgeom=np.vstack([f["geom"] for mid in feats for f in feats[mid] if mid in (["clm197","pal766","dresden_mixed"]+NULL_IDS)])
    geom_mu=allgeom.mean(axis=0); geom_sd=allgeom.std(axis=0); geom_sd=np.where(geom_sd<1e-6,1.0,geom_sd)
    d1=build_direction("clm197","pal766",feats,geom_mu,geom_sd)
    d2=build_direction("pal766","clm197",feats,geom_mu,geom_sd)

    reps=["hog","chamfer","geometry"]
    rep_pass=[]
    for r in reps:
        rep_pass.append(r) if min(d1["representation_stats"][r]["z"],d2["representation_stats"][r]["z"])>=2.0 else None
    # At least one pair of passing reps must not be degenerate (r<0.90 in both directions).
    nondeg_pair=False
    for i,r1 in enumerate(rep_pass):
        for r2 in rep_pass[i+1:]:
            key=f"{r1}__{r2}" if f"{r1}__{r2}" in d1["null_rep_correlations"] else f"{r2}__{r1}"
            if abs(d1["null_rep_correlations"].get(key,1.0))<.90 and abs(d2["null_rep_correlations"].get(key,1.0))<.90:
                nondeg_pair=True
    # Composite gate in both leave-one-witness-out directions.
    composite_gate=all([
        d1["composite_stats"]["z"]>=2.0,d2["composite_stats"]["z"]>=2.0,
        d1["composite_stats"]["empirical_block_p"]<=.05,d2["composite_stats"]["empirical_block_p"]<=.05,
        d1["bootstrap"]["fraction_composite_z_ge_2"]>=.80,d2["bootstrap"]["fraction_composite_z_ge_2"]>=.80,
    ])
    # Hard specificity diagnostic: positive must outrank all technical controls in both directions.
    technical_rank_gate=(d1["composite_stats"]["technical_rank"]==1 and d2["composite_stats"]["technical_rank"]==1)
    # Decision rule must not collapse under top-k 5/8/12: all sensitivity z >= 1.5 and same positive direction.
    fragility_gate=all(v["z"]>=1.5 for d in (d1,d2) for v in d["sensitivity_top_matches"].values())
    passed=(len(rep_pass)>=2 and nondeg_pair and composite_gate and technical_rank_gate and fragility_gate)

    result={
        "version":VERSION,"release_id":RELEASE_ID,"panel_sha256":panel_hash,"audit":audit,
        "direction_clm197_to_pal766":d1,"direction_pal766_to_clm197":d2,
        "gate":{
            "passing_representations_both_directions":rep_pass,
            "nondegenerate_passing_pair":nondeg_pair,
            "composite_gate":composite_gate,
            "technical_rank_gate":technical_rank_gate,
            "decision_rule_fragility_gate":fragility_gate,
            "calibration_passed":passed,
            "q13_unseal_allowed":passed,
            "criteria":{
                "rep_z_min":2.0,"min_passing_reps":2,"max_abs_null_rep_corr":0.90,
                "composite_z_min":2.0,"block_p_max":0.05,"bootstrap_stability_min":0.80,
                "technical_rank_required":1,"sensitivity_z_floor":1.5,
            },
        },
        "errors":errors[:200],
    }
    (outdir/"calibration.json").write_text(json.dumps(result,indent=2))
    with open(outdir/"checkpoint_03_calibration.pkl","wb") as f:pickle.dump(result,f,protocol=5)

    # Running results keeps retractions first and Q13 explicitly sealed/unsealed state.
    lines=["# Taccola × Q13 — Running Results","","## RETRACTED FINDINGS","",
           "- **RETRACTED:** the raw Siena/Taccola concentration (4/12 vs 5/185) is not independent evidence. After family collapse it is 1/12 vs 5/185; effect +5.63 percentage points, exact conditional null SD 5.13 points, 1.10 null SD, Fisher p=0.318. The metric does not resolve this.","",
           "## Protocol repair audit","",
           "- v0.1b is a pre-score transport repair only. The parent run timed out before checkpoint_02 or any similarity score existed. Heidelberg controls were inaccessible (HTTP 403) and Bodleian image delivery timed out from GitHub runners. Control counts and all scientific scoring/gate functions are unchanged; replacements were selected from r15 metadata before scoring.","",
           "## Calibration (Q13 sealed)","",
           f"Panel SHA-256: `{panel_hash}`",f"Active manuscript-block nulls: {len(NULL_IDS)}.",""]
    for title,d in [("Clm197 → Pal766",d1),("Pal766 → Clm197",d2)]:
        c=d["composite_stats"]
        prefix="the metric does not resolve this — " if abs(c["z"])<2 else ""
        lines.append(f"- **{title}:** {prefix}composite effect {c['effect']:.4f}, null SD {c['null_sd']:.4f}, {c['z']:.2f} null SD, block p={c['empirical_block_p']:.4f}; technical rank {c['technical_rank']}/{len(TECH_IDS)+1}; bootstrap stability={d['bootstrap']['fraction_composite_z_ge_2']:.3f}.")
        for r in reps:
            ss=d["representation_stats"][r]
            pfx="the metric does not resolve this — " if abs(ss["z"])<2 else ""
            lines.append(f"  - {r}: {pfx}effect {ss['effect']:.4f}, null SD {ss['null_sd']:.4f}, {ss['z']:.2f} null SD, p={ss['empirical_block_p']:.4f}.")
    lines += ["",f"**Calibration passed:** `{passed}`",f"**Q13 unseal allowed:** `{passed}`","","No Q13 image, URL, feature or score was accessed by this script."]
    (outdir/"RUNNING_RESULTS.md").write_text("\n".join(lines)+"\n")
    (outdir/"calibration.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"event":"done","calibration_passed":passed,"q13_unseal_allowed":passed,"panel_sha256":panel_hash,"active_nulls":len(NULL_IDS)}),flush=True)

if __name__=="__main__":
    main()
