#!/usr/bin/env python3
"""Preregistered physical-unit test of the Stolfi oversized/off-cut parchment class.

RETRACTION AT TOP
-----------------
The first implementation incorrectly treated Yale frames 1006235 (f89v+f90r)
and 1006251 (f101v+f102r) as single physical sheets. They are current openings
spanning different bifolia. That implementation is VOID for inference.

This version measures six codicologically established physical units. For four
units a full-sheet view is available. For q15_b87_90 and q19_b99_102, only the
terminal free edges are taken from component/current-opening views; no geometry
across the current opening is used.
"""
import itertools, json, os, pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
from scipy.ndimage import gaussian_filter1d

PROTOCOL = "vms_oversized_parchment_class_20260910_v03"
OUT = Path("artifacts/vms_oversized_parchment_class_20260910_v03")
OUT.mkdir(parents=True, exist_ok=True)
HEADERS={"User-Agent":"Voynich-research/1.0"}
# All are true downsampled widths below every source image's native width.
RESOLUTIONS=(1200,1800,2400)

RETRACTED_FINDINGS=[{
 "finding":"Initial candidate-frame definition treated 1006235 and 1006251 as single oversized physical sheets.",
 "status":"RETRACTED",
 "reason":"Both are current openings spanning different physical bifolia; full-frame geometry would leak present binding topology.",
 "replacement":"Physical bifolium units and terminal free edges only."
}]

CANDIDATES=[
 {"unit":"q14_b85_86","quire":"Q14","kind":"spread","sid":"1006231"},
 {"unit":"q15_b87_90","quire":"Q15","kind":"pair","a_sid":"1006232","a_side":"right","b_sid":"1006235","b_side":"right"},
 {"unit":"q15_b88_89","quire":"Q15","kind":"spread","sid":"1006233"},
 {"unit":"q17_b94_95","quire":"Q17","kind":"spread","sid":"1006241"},
 {"unit":"q19_b99_102","quire":"Q19","kind":"pair","a_sid":"1006246","a_side":"right","b_sid":"1006251","b_side":"right"},
 {"unit":"q19_b100_101","quire":"Q19","kind":"spread","sid":"1006249"},
]
CONTROLS=[
 {"unit":"q13_b75_84","quire":"Q13","kind":"pair","a_sid":"1006208","a_side":"right","b_sid":"1006227","b_side":"left"},
 {"unit":"q13_b76_83","quire":"Q13","kind":"pair","a_sid":"1006210","a_side":"right","b_sid":"1006225","b_side":"left"},
 {"unit":"q13_b77_82","quire":"Q13","kind":"pair","a_sid":"1006212","a_side":"right","b_sid":"1006223","b_side":"left"},
 {"unit":"q13_b78_81","quire":"Q13","kind":"pair","a_sid":"1006214","a_side":"right","b_sid":"1006221","b_side":"left"},
 {"unit":"q13_b79_80","quire":"Q13","kind":"pair","a_sid":"1006216","a_side":"right","b_sid":"1006219","b_side":"left"},
 {"unit":"q17_b93_96","quire":"Q17","kind":"pair","a_sid":"1006238","a_side":"right","b_sid":"1006245","b_side":"left"},
 {"unit":"q20_b103_116","quire":"Q20","kind":"pair","a_sid":"1006254","a_side":"right","b_sid":"1006277","b_side":"left"},
 {"unit":"q20_b104_115","quire":"Q20","kind":"pair","a_sid":"1006256","a_side":"right","b_sid":"1006275","b_side":"left"},
 {"unit":"q20_b105_114","quire":"Q20","kind":"pair","a_sid":"1006258","a_side":"right","b_sid":"1006273","b_side":"left"},
 {"unit":"q20_b106_113","quire":"Q20","kind":"pair","a_sid":"1006260","a_side":"right","b_sid":"1006271","b_side":"left"},
 {"unit":"q20_b107_112","quire":"Q20","kind":"pair","a_sid":"1006262","a_side":"right","b_sid":"1006269","b_side":"left"},
 {"unit":"q20_b108_111","quire":"Q20","kind":"pair","a_sid":"1006264","a_side":"right","b_sid":"1006267","b_side":"left"},
]


def checkpoint(stage, payload=None):
    obj={"protocol":PROTOCOL,"retracted_findings":RETRACTED_FINDINGS,"stage":stage,"payload":payload}
    tmp=OUT/"checkpoint.pkl.tmp"; final=OUT/"checkpoint.pkl"
    with open(tmp,"wb") as f: pickle.dump(obj,f)
    os.replace(tmp,final)


def download(sid,width):
    p=OUT/f"scan_{sid}_{width}.jpg"
    if p.exists() and p.stat().st_size>10000: return p
    url=f"https://collections.library.yale.edu/iiif/2/{sid}/full/{width},/0/default.jpg"
    r=requests.get(url,headers=HEADERS,timeout=90); r.raise_for_status(); p.write_bytes(r.content)
    return p


def page_mask(img):
    L=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)[:,:,0]
    blur=cv2.GaussianBlur(L,(0,0),2.0)
    _,m=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,k,iterations=2)
    n,lab,stats,_=cv2.connectedComponentsWithStats(m,8)
    if n<2: raise RuntimeError("no parchment component")
    idx=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
    z=(lab==idx).astype(np.uint8)
    cs,_=cv2.findContours(z,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not cs: raise RuntimeError("no parchment contour")
    out=np.zeros_like(z); cv2.drawContours(out,[max(cs,key=cv2.contourArea)],-1,1,cv2.FILLED)
    return out


def side_profile(mask,side):
    ys=np.where(mask.any(axis=1))[0]
    y0,y1=int(ys[0]),int(ys[-1]); vals=[]
    for y in range(y0,y1+1):
        xx=np.where(mask[y]>0)[0]
        vals.append(float(xx[0] if side=="left" else xx[-1]) if len(xx) else np.nan)
    return np.asarray(vals,float), float(y1-y0+1)


def interp(v):
    ok=np.isfinite(v); x=np.arange(len(v))
    if ok.sum()<10: raise RuntimeError("insufficient edge points")
    return np.interp(x,x[ok],v[ok])


def edge_features(v,height):
    v=interp(v); n=len(v); trim=max(4,int(.04*n)); v=v[trim:n-trim]
    v=gaussian_filter1d(v,max(1.0,n/900.0))
    x=np.linspace(-1,1,len(v)); fit=np.polyval(np.polyfit(x,v,2),x); r=v-fit
    rms=float(np.sqrt(np.mean(r*r))/height)
    qrange=float((np.quantile(r,.95)-np.quantile(r,.05))/height)
    d=np.diff(v); med=np.median(d); mad=float(np.median(np.abs(d-med))*1.4826/height)
    return {"rms":rms,"qrange":qrange,"diffmad":mad}


def get_edge(sid,side,width):
    img=cv2.imread(str(download(sid,width)),cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError(f"failed scan {sid}")
    m=page_mask(img); v,h=side_profile(m,side); return edge_features(v,h)


def measure_unit(spec,width):
    if spec["kind"]=="spread":
        e1=get_edge(spec["sid"],"left",width); e2=get_edge(spec["sid"],"right",width)
    else:
        e1=get_edge(spec["a_sid"],spec["a_side"],width); e2=get_edge(spec["b_sid"],spec["b_side"],width)
    row={"unit":spec["unit"],"quire":spec["quire"]}
    for m in ("rms","qrange","diffmad"):
        row[f"edge1_{m}"]=e1[m]; row[f"edge2_{m}"]=e2[m]
        row[f"max_{m}"]=max(e1[m],e2[m]); row[f"mean_{m}"]=.5*(e1[m]+e2[m])
    return row


def exact_test(df,metric):
    vals=df[metric].to_numpy(float); lab=df.candidate.to_numpy(int); n1=int(lab.sum()); idx=np.arange(len(df))
    obs=float(vals[lab==1].mean()-vals[lab==0].mean()); null=[]
    for comb in itertools.combinations(idx,n1):
        g=np.zeros(len(df),bool); g[list(comb)]=True
        null.append(float(vals[g].mean()-vals[~g].mean()))
    null=np.asarray(null); mu=float(null.mean()); sd=float(null.std(ddof=1))
    z=float((obs-mu)/sd) if sd>0 else None
    p=float((np.sum(null>=obs)+1)/(len(null)+1))
    return {"metric":metric,"candidate_mean":float(vals[lab==1].mean()),"control_mean":float(vals[lab==0].mean()),
            "effect":obs,"null_mean":mu,"null_sd":sd,"effect_over_null_sd":z,"p_upper_exact":p,"n_null":int(len(null))}


def loo(df,metric):
    out=[]
    for u in df[df.candidate==1].unit:
        r=exact_test(df[df.unit!=u].reset_index(drop=True),metric); r["left_out_candidate"]=u; out.append(r)
    return out


def run_resolution(width):
    rows=[]
    for c,specs in ((1,CANDIDATES),(0,CONTROLS)):
        for spec in specs:
            print("MEASURE",width,c,spec["unit"],flush=True)
            r=measure_unit(spec,width); r["candidate"]=c; r["resolution"]=width; rows.append(r)
    df=pd.DataFrame(rows); df.to_csv(OUT/f"features_{width}.csv",index=False)
    metrics=["max_rms","mean_rms","max_qrange","mean_qrange","max_diffmad","mean_diffmad"]
    return df,[exact_test(df,m) for m in metrics],loo(df,"max_rms")

checkpoint("frozen_protocol",{"candidates":[x["unit"] for x in CANDIDATES],"controls":[x["unit"] for x in CONTROLS],
                               "primary":"max_rms","resolutions":RESOLUTIONS})
all_tests={}; all_loo={}
for w in RESOLUTIONS:
    df,tests,los=run_resolution(w); all_tests[str(w)]=tests; all_loo[str(w)]=los
    checkpoint(f"completed_{w}",{"tests":tests,"loo":los})

primary=[]
for w in RESOLUTIONS:
    r=next(x for x in all_tests[str(w)] if x["metric"]=="max_rms"); primary.append({"resolution":w,**r})
sec_dir={}
for metric in ["mean_rms","max_qrange","mean_qrange","max_diffmad","mean_diffmad"]:
    sec_dir[metric]=[next(x for x in all_tests[str(w)] if x["metric"]==metric)["effect"] for w in RESOLUTIONS]
f102_loo=[x for x in all_loo["2400"] if x["left_out_candidate"]=="q19_b99_102"][0]
summary={
 "protocol":PROTOCOL,
 "retracted_findings":RETRACTED_FINDINGS,
 "hypothesis":"Oversized/foldout physical units Q14/Q15/Q17/Q19 preserve more irregular terminal free parchment edges than ordinary bifolia, as predicted if they preferentially used irregular/off-cut material.",
 "primary_metric":"max_rms: maximum over the two terminal free edges of quadratic-detrended contour RMS / edge height",
 "primary_by_resolution":primary,
 "secondary_by_resolution":all_tests,
 "leave_one_candidate_out":all_loo,
 "f102_damage_sensitivity_2400":f102_loo,
 "secondary_effects":sec_dir,
 "decision_rule":"Support only if primary effect is positive and >=2 null SD at all three resolutions, remains positive in every leave-one-candidate-out run, and at least 3/5 secondary representations are positive at all three resolutions. Otherwise unresolved or negative.",
 "excluded_from_inference":[
   "Overall sheet width/aspect ratio: circular because oversized/foldout status defines the candidate set.",
   "Current-opening full-frame geometry: present-binding leakage.",
   "Text/Currier/illustration labels: not used.",
   "Hair/flesh side: no public folio-level map; unavailable rather than imputed.",
   "Generic stain similarity: previously retracted as physical identification evidence."
 ],
 "frozen_prior_fold_evidence":"Separate prior registered test: f89/f102 show two physical fold states conditional on selected cases, pooled current-minus-old crease strength effect 11.8933 vs null SD 4.0168 = 2.961 SD; f89 alone 1.861 SD (unresolved), f102 alone 2.322 SD. This is corroboration only and not included in the edge-class statistic."
}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2))
with open(OUT/"final.pkl","wb") as f: pickle.dump(summary,f)
checkpoint("complete",summary)
print(json.dumps(summary,indent=2),flush=True)
