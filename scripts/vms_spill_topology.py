#!/usr/bin/env python3
"""Quantify upper-margin stain morphology across VMS ff.1-56.

Stage S1 is deliberately image-only. It does not use Voynich transcription,
Currier language, scribe labels, or downstream statistical outcomes.

Outputs compact CSV/JSON measurements plus diagnostic contact sheets.  Linear
ordering conclusions are NOT licensed by this script; it supplies physical
surface evidence and current-order/permutation diagnostics only.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing, binary_fill_holes

OUT = Path(os.environ.get("SPILL_OUT", "artifacts/vms_spill_v01"))
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ.get("SPILL_CACHE", ".cache/vms_spill_v01"))
CACHE.mkdir(parents=True, exist_ok=True)

# Beinecke IIIF ids are consecutive across these surviving sides.
PAGES: List[Tuple[str, int]] = []
_i = 1006076
for f in range(1, 12):
    for side in "rv":
        PAGES.append((f"f{f}{side}", _i)); _i += 1
for f in range(13, 57):
    for side in "rv":
        PAGES.append((f"f{f}{side}", _i)); _i += 1
assert len(PAGES) == 110 and _i == 1006186

# Explicit conjoint mapping from Stolfi/Beinecke collation, not inferred by solver.
BIFOLIA = {
    "q01": [(1,8),(2,7),(3,6),(4,5)],
    "q02": [(9,16),(10,15),(11,14),(12,13)],
    "q03": [(17,24),(18,23),(19,22),(20,21)],
    "q04": [(25,32),(26,31),(27,30),(28,29)],
    "q05": [(33,40),(34,39),(35,38),(36,37)],
    "q06": [(41,48),(42,47),(43,46),(44,45)],
    "q07": [(49,56),(50,55),(51,54),(52,53)],
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "VoynichTopologyResearch/0.1 (+research; GitHub Actions)"})

def fetch_page(label: str, iid: int, width: int = 1200) -> Image.Image:
    p = CACHE / f"{label}_{width}.jpg"
    if not p.exists():
        url = f"https://collections.library.yale.edu/iiif/2/{iid}/full/{width},/0/default.jpg"
        last = None
        for attempt in range(5):
            try:
                r = SESSION.get(url, timeout=60)
                r.raise_for_status()
                p.write_bytes(r.content)
                break
            except Exception as exc:
                last = exc
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"fetch failed {label} {url}: {last}")
    data = p.read_bytes()
    im = Image.open(io.BytesIO(data)).convert("RGB")
    return im


def robust_page_crop(im: Image.Image) -> Image.Image:
    """Trim only tiny scan borders; avoid content-dependent manuscript cropping."""
    w, h = im.size
    x0, x1 = int(0.015*w), int(0.985*w)
    y0, y1 = int(0.005*h), int(0.995*h)
    return im.crop((x0,y0,x1,y1))


def feature_page(im: Image.Image) -> Dict[str, object]:
    im = robust_page_crop(im)
    a = np.asarray(im).astype(np.float32) / 255.0
    h, w, _ = a.shape
    # Upper 22%; diagnostics also retain a conservative upper-margin core (3-16%).
    roi = a[: max(40, int(0.22*h)), :, :]
    gray = 0.2126*roi[:,:,0] + 0.7152*roi[:,:,1] + 0.0722*roi[:,:,2]
    # Broad stain survives; glyph strokes and most pigment edges are suppressed.
    sigma = max(3.0, w/180.0)
    low = gaussian_filter(gray, sigma=sigma, mode="nearest")

    # Estimate parchment illumination as a smooth high quantile in x-bins, then a global
    # robust ceiling. This removes page-to-page scanner exposure without using content labels.
    nxb = 48
    bgx = np.zeros(nxb, dtype=np.float32)
    edges = np.linspace(0,w,nxb+1).astype(int)
    for j in range(nxb):
        sl = low[:, edges[j]:edges[j+1]]
        bgx[j] = np.quantile(sl, 0.82)
    bgx = gaussian_filter(bgx, sigma=2.0)
    bgmap = np.zeros_like(low)
    for j in range(nxb): bgmap[:, edges[j]:edges[j+1]] = bgx[j]
    # Positive darkness residual. Cap to reduce ink/paint leverage.
    resid = np.clip(bgmap - low, 0, 0.18)

    y0, y1 = int(0.025*roi.shape[0]), int(0.78*roi.shape[0])
    core = resid[y0:y1]
    # Fixed physical-scale thresholds after exposure normalization.
    t_lo, t_hi = 0.022, 0.038
    mask = core > t_lo
    # Morphology operates on low-frequency image, so it removes isolated drawing remnants.
    rad = max(1, int(w/500))
    st = np.ones((2*rad+1, 2*rad+1), dtype=bool)
    mask = binary_opening(mask, structure=st)
    mask = binary_closing(mask, structure=np.ones((3,3), dtype=bool), iterations=2)
    mask = binary_fill_holes(mask)

    # Profiles are intentionally low-dimensional: 32 x bins by 10 y bins.
    ny, nx = 10, 32
    grid = np.zeros((ny,nx), dtype=np.float32)
    for iy in range(ny):
        ya, yb = int(iy*core.shape[0]/ny), int((iy+1)*core.shape[0]/ny)
        for ix in range(nx):
            xa, xb = int(ix*core.shape[1]/nx), int((ix+1)*core.shape[1]/nx)
            grid[iy,ix] = float(np.mean(core[ya:yb,xa:xb]))

    # Deepest mask penetration per x bin (normalized y within upper ROI).
    contour = np.zeros(nx, dtype=np.float32)
    for ix in range(nx):
        xa, xb = int(ix*mask.shape[1]/nx), int((ix+1)*mask.shape[1]/nx)
        ys = np.where(mask[:,xa:xb].any(axis=1))[0]
        contour[ix] = 0.0 if len(ys)==0 else (ys.max()+1)/mask.shape[0]

    vals = core.flatten()
    metrics = {
        "stain_area_lo": float(np.mean(core > t_lo)),
        "stain_area_hi": float(np.mean(core > t_hi)),
        "stain_mean": float(np.mean(core)),
        "stain_p90": float(np.quantile(vals, .90)),
        "stain_p95": float(np.quantile(vals, .95)),
        "contour_mean": float(np.mean(contour)),
        "contour_max": float(np.max(contour)),
        "background_mean": float(np.mean(bgx)),
        "grid": grid,
        "contour": contour,
        "resid": resid,
        "mask": mask,
        "crop": im,
        "roi_shape": tuple(roi.shape[:2]),
    }
    return metrics


def standardize(X: np.ndarray) -> np.ndarray:
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X-med), axis=0)
    scale = np.where(mad > 1e-6, 1.4826*mad, np.std(X,axis=0)+1e-6)
    return (X-med)/np.where(scale>1e-6, scale, 1.0)


def dist_matrix(X: np.ndarray) -> np.ndarray:
    Z = standardize(X)
    # RMS robust-z distance: comparable across descriptor families.
    d = Z[:,None,:] - Z[None,:,:]
    return np.sqrt(np.mean(d*d, axis=2))


def path_energy(order: List[int], D: np.ndarray) -> float:
    if len(order)<2: return 0.0
    return float(np.mean([D[order[i],order[i+1]] for i in range(len(order)-1)]))


def permutation_test_current(order: List[int], D: np.ndarray, rng: random.Random, n=20000):
    obs = path_energy(order,D)
    vals=[]
    base=order[:]
    for _ in range(n):
        p=base[:]; rng.shuffle(p); vals.append(path_energy(p,D))
    ar=np.asarray(vals)
    mean=float(ar.mean()); sd=float(ar.std(ddof=1))
    z=(obs-mean)/sd if sd else None
    # lower energy = smoother
    p=(1+int(np.sum(ar<=obs)))/(n+1)
    return {"observed":obs,"null_mean":mean,"null_sd":sd,"z":z,"p_lower":p,"n_perm":n}


def make_contact(rows: Dict[str,Dict[str,object]], labels: List[str], path: Path):
    thumb_w, thumb_h = 420, 220
    panel_h = 2*thumb_h + 38
    sheet=Image.new("RGB",(thumb_w*3,panel_h*math.ceil(len(labels)/3)),"white")
    draw=ImageDraw.Draw(sheet)
    for k,label in enumerate(labels):
        r,c=divmod(k,3); x=c*thumb_w; y=r*panel_h
        dat=rows[label]; crop=dat["crop"]
        w,h=crop.size
        roi=crop.crop((0,0,w,int(.22*h))).resize((thumb_w,thumb_h))
        sheet.paste(roi,(x,y+20))
        resid=dat["resid"]
        rr=np.clip(resid/0.08,0,1)
        heat=np.uint8(255*(1-rr))
        heat_rgb=np.stack([np.full_like(heat,255),heat,np.full_like(heat,255)],axis=2)
        him=Image.fromarray(heat_rgb).resize((thumb_w,thumb_h))
        sheet.paste(him,(x,y+20+thumb_h))
        txt=f"{label} area={dat['stain_area_lo']:.3f} mean={dat['stain_mean']:.4f}"
        draw.text((x+4,y+2),txt,fill="black")
    sheet.save(path,quality=90)


def main():
    rows: Dict[str,Dict[str,object]]={}
    image_hashes={}
    for label,iid in PAGES:
        im=fetch_page(label,iid)
        p=CACHE/f"{label}_1200.jpg"
        image_hashes[label]=hashlib.sha256(p.read_bytes()).hexdigest()
        rows[label]=feature_page(im)
        print(label, rows[label]["stain_area_lo"], rows[label]["stain_mean"], flush=True)

    # page-side output
    scalar_cols=["stain_area_lo","stain_area_hi","stain_mean","stain_p90","stain_p95","contour_mean","contour_max","background_mean"]
    with (OUT/"spill_page_features.csv").open("w",newline="") as f:
        wr=csv.writer(f); wr.writerow(["page_id","iiif_id","sha256",*scalar_cols,*[f"contour_{i:02d}" for i in range(32)]])
        for label,iid in PAGES:
            d=rows[label]
            wr.writerow([label,iid,image_hashes[label],*[d[c] for c in scalar_cols],*d["contour"].tolist()])

    # Folio descriptor = mean recto/verso when both survive. This reduces scan-side noise.
    folios=[]; F=[]; folio_metrics={}
    for fol in list(range(1,12))+list(range(13,57)):
        a=rows[f"f{fol}r"]; b=rows[f"f{fol}v"]
        vec=np.concatenate([(a["grid"]+b["grid"]).ravel()/2, (a["contour"]+b["contour"])/2])
        folios.append(fol); F.append(vec)
        folio_metrics[fol]={c:(a[c]+b[c])/2 for c in scalar_cols}
    F=np.stack(F)
    Df=dist_matrix(F)
    fi={f:i for i,f in enumerate(folios)}

    # Current numeric folio order, with missing f12 omitted.
    current=list(range(len(folios)))
    rng=random.Random(20260907)
    current_test=permutation_test_current(current,Df,rng)

    # Within each current quire: current folio sequence smoothness vs all permutations.
    quire_tests={}
    for q,pairs in BIFOLIA.items():
        leaves=sorted({x for p in pairs for x in p if x in fi})
        idx=[fi[x] for x in leaves]
        quire_tests[q]=permutation_test_current(idx,Df,rng,n=10000)
        quire_tests[q]["folios"]=leaves

    # Conjoint similarity: compare observed conjoint distances with all within-quire nonself pairs.
    conjoint=[]
    for q,pairs in BIFOLIA.items():
        leaves=sorted({x for p in pairs for x in p if x in fi})
        allpairs=[]
        for ia in range(len(leaves)):
            for ib in range(ia+1,len(leaves)):
                allpairs.append(Df[fi[leaves[ia]],fi[leaves[ib]]])
        obs=[]
        for x,y in pairs:
            if x in fi and y in fi: obs.append(Df[fi[x],fi[y]])
        conjoint.append({"quire":q,"obs_mean":float(np.mean(obs)) if obs else None,
                         "allpair_mean":float(np.mean(allpairs)),"allpair_sd":float(np.std(allpairs,ddof=1)),
                         "effect":float(np.mean(obs)-np.mean(allpairs)) if obs else None,
                         "z":float((np.mean(obs)-np.mean(allpairs))/np.std(allpairs,ddof=1)) if obs and np.std(allpairs,ddof=1)>0 else None,
                         "n_obs":len(obs),"n_allpairs":len(allpairs)})

    # Page-side recto-vs-verso agreement: a necessary reliability check.
    side_scalar=[]
    for fol in folios:
        side_scalar.append([rows[f"f{fol}r"]["stain_area_lo"],rows[f"f{fol}v"]["stain_area_lo"]])
    side_scalar=np.asarray(side_scalar)
    side_corr=float(np.corrcoef(side_scalar[:,0],side_scalar[:,1])[0,1])

    # Candidate scalar orderings only as diagnostics (not licensed reconstruction): ranks by stain area/contour.
    ranks={}
    for metric in ["stain_area_lo","stain_mean","contour_mean"]:
        ranks[metric]=sorted(folios,key=lambda f:folio_metrics[f][metric],reverse=True)

    summary={
        "protocol":"vms_spill_topology_s1_20260907",
        "source":"Beinecke IIIF via collections.library.yale.edu",
        "n_page_sides":len(PAGES),"n_folios":len(folios),"missing_folios":[12],
        "recto_verso_area_corr":side_corr,
        "current_order_test":current_test,
        "quire_current_order_tests":quire_tests,
        "conjoint_similarity":conjoint,
        "diagnostic_scalar_rankings":ranks,
        "license":"PHYSICAL_DIAGNOSTIC_ONLY__NO_LINEAR_ORDER_INFERENCE",
        "notes":[
            "Upper-margin descriptor uses only pixels; no VMS text metadata enters scoring.",
            "Current-order permutation tests diagnose smoothness, not historical correctness.",
            "Alternative topology inference requires mask validation and hard codicological constraints.",
        ],
    }
    (OUT/"spill_summary.json").write_text(json.dumps(summary,indent=2))

    # Distance matrix for downstream constrained solver.
    with (OUT/"spill_folio_distance.csv").open("w",newline="") as f:
        wr=csv.writer(f); wr.writerow(["folio",*folios])
        for i,fol in enumerate(folios): wr.writerow([fol,*Df[i].tolist()])

    # Diagnostics deliberately include f32v/f33r and broad depth samples.
    diag=["f1r","f8v","f9r","f16v","f17r","f24v","f25r","f32v","f33r","f40v","f41r","f48v","f49r","f56v"]
    make_contact(rows,diag,OUT/"spill_diagnostic_contact.jpg")

    print(json.dumps(summary,indent=2))

if __name__ == "__main__":
    main()
