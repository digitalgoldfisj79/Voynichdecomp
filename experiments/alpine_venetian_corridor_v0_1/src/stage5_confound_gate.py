#!/usr/bin/env python3
"""Stage 5 confound gate. Contains no VMS inputs or corridor/control labels."""
from __future__ import annotations
import ast, hashlib, json, os, sys, time
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageFilter
import torch
from transformers import AutoImageProcessor, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

SEED = 20260808
MODEL = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
VARIANTS = ["rgb_norm_v1", "gray_bgdiv_v1", "inkmask_v1"]
PASS_MAX = 0.65
CAUTION_MAX = 0.70
MIN_ACQUIRED_FRAC = 0.80
MIN_MANUSCRIPTS = 6
HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / "stage5_confound_manifest.tsv"
OUT = Path(os.environ.get("OUT_DIR", "/tmp/stage5_confound"))
OUT.mkdir(parents=True, exist_ok=True)
HEADERS = {"User-Agent": "VoynichCorridorResearch/0.1 (+bounded scientific image QA)"}


def fetch(url: str, tries: int = 3) -> Image.Image:
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=40)
            r.raise_for_status()
            return Image.open(BytesIO(r.content)).convert("RGB")
        except Exception as e:
            last = e
            time.sleep(1.0 + k * 1.5)
    raise RuntimeError(f"fetch failed after {tries}: {url}: {last}")


def crop_norm(im: Image.Image, box):
    x0,y0,x1,y1 = box
    w,h = im.size
    px = (max(0, round(x0*w/1000)), max(0, round(y0*h/1000)),
          min(w, round(x1*w/1000)), min(h, round(y1*h/1000)))
    if px[2] <= px[0] or px[3] <= px[1]:
        raise ValueError(f"empty crop {px} from {im.size}")
    return im.crop(px)


def square_pad(im: Image.Image, fill=(255,255,255)) -> Image.Image:
    w,h = im.size
    s = max(w,h)
    out = Image.new("RGB", (s,s), fill)
    out.paste(im, ((s-w)//2,(s-h)//2))
    return out


def bgdiv_gray(im: Image.Image):
    g = im.convert("L")
    a = np.asarray(g, dtype=np.float32)
    radius = max(3.0, min(im.size)*0.08)
    bg = np.asarray(g.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)
    flat = np.clip(a / np.maximum(bg, 1.0) * 240.0, 0, 255).astype(np.uint8)
    return flat


def variant(im: Image.Image, name: str) -> Image.Image:
    if name == "rgb_norm_v1":
        return square_pad(im.convert("RGB"))
    flat = bgdiv_gray(im)
    if name == "gray_bgdiv_v1":
        return square_pad(Image.fromarray(flat, mode="L").convert("RGB"))
    if name == "inkmask_v1":
        mask = np.where(flat < 215, 0, 255).astype(np.uint8)
        return square_pad(Image.fromarray(mask, mode="L").convert("RGB"))
    raise KeyError(name)


def load_manifest():
    df = pd.read_csv(MANIFEST, sep="\t")
    df["bbox"] = df["bbox_1000"].map(ast.literal_eval)
    return df


def acquire(df):
    cache = {}
    rows, errors = [], []
    for i,r in df.iterrows():
        url = r.image_url
        try:
            if url not in cache:
                cache[url] = fetch(url)
            c = crop_norm(cache[url], r.bbox)
            rec = {"row": int(i), "candidate_key": r.candidate_key, "image_url": url, "crop": c}
            rows.append(rec)
            print("ACQUIRED=" + json.dumps({"row":int(i),"candidate_key":r.candidate_key,"page":hashlib.sha1(url.encode()).hexdigest()[:12],"size":c.size}), flush=True)
        except Exception as e:
            errors.append({"row":int(i),"candidate_key":r.candidate_key,"image_url":url,"error":repr(e)})
            print("ACQUIRE_ERROR=" + json.dumps(errors[-1]), flush=True)
    return rows, errors


def embed(rows, processor, model, device, name, batch_size=2):
    embs=[]
    with torch.inference_mode():
        for st in range(0,len(rows),batch_size):
            ims=[variant(x["crop"],name) for x in rows[st:st+batch_size]]
            inp=processor(images=ims, return_tensors="pt")
            inp={k:v.to(device) for k,v in inp.items()}
            out=model(**inp)
            z=out.last_hidden_state[:,0,:].float()
            z=torch.nn.functional.normalize(z,dim=1)
            embs.append(z.cpu().numpy())
    X=np.concatenate(embs,axis=0)
    np.save(OUT/f"embeddings_{name}.npy",X)
    return X


def page_heldout_oof(X, y, groups):
    le=LabelEncoder().fit(y)
    yi=le.transform(y)
    K=len(le.classes_)
    oof=np.full((len(y),K),np.nan,dtype=float)
    uniq=sorted(set(groups))
    fold_meta=[]
    for g in uniq:
        te=np.array([x==g for x in groups])
        tr=~te
        if not te.any(): continue
        # Every manuscript has >=2 acquired pages after post-acquisition filtering.
        pipe=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight="balanced",max_iter=5000,random_state=SEED,solver="lbfgs"))
        pipe.fit(X[tr],yi[tr])
        p=pipe.predict_proba(X[te])
        cls=pipe[-1].classes_.astype(int)
        for j,c in enumerate(cls): oof[te,c]=p[:,j]
        fold_meta.append({"heldout_page":hashlib.sha1(g.encode()).hexdigest()[:12],"n_test":int(te.sum())})
    if np.isnan(oof).any(): raise RuntimeError("OOF probability matrix incomplete")
    auc=float(roc_auc_score(yi,oof,multi_class="ovr",average="macro",labels=np.arange(K)))
    acc=float(accuracy_score(yi,np.argmax(oof,axis=1)))
    return auc,acc,le.classes_.tolist(),fold_meta


def random_diag_oof(X,y):
    le=LabelEncoder().fit(y); yi=le.transform(y); K=len(le.classes_)
    min_n=min(Counter(yi).values()); n_splits=min(5,min_n)
    if n_splits < 2: return None
    cv=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=SEED)
    oof=np.full((len(y),K),np.nan)
    for tr,te in cv.split(X,yi):
        pipe=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight="balanced",max_iter=5000,random_state=SEED,solver="lbfgs"))
        pipe.fit(X[tr],yi[tr]); p=pipe.predict_proba(X[te]); cls=pipe[-1].classes_.astype(int)
        for j,c in enumerate(cls): oof[te,c]=p[:,j]
    if np.isnan(oof).any(): return {"n_splits":n_splits,"auc":None,"reason":"class absent in a training fold"}
    return {"n_splits":n_splits,"auc":float(roc_auc_score(yi,oof,multi_class="ovr",average="macro",labels=np.arange(K)))}


def decision(auc):
    if auc <= PASS_MAX: return "PASS"
    if auc <= CAUTION_MAX: return "CAUTION"
    return "FAIL"


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    df=load_manifest()
    assert len(df)==59, len(df)
    print("CONFIG="+json.dumps({"model":MODEL,"rows":len(df),"manuscripts":df.candidate_key.nunique(),"variants":VARIANTS,"seed":SEED}),flush=True)
    rows,errors=acquire(df)
    acquired_frac=len(rows)/len(df)
    # After acquisition, keep only manuscripts retaining >=2 DISTINCT source pages.
    pages=defaultdict(set)
    for r in rows: pages[r["candidate_key"]].add(r["image_url"])
    eligible={k for k,v in pages.items() if len(v)>=2}
    rows=[r for r in rows if r["candidate_key"] in eligible]
    acquisition={"requested":len(df),"acquired":len(rows),"raw_acquired":len(rows)+sum(1 for r in []),"errors":len(errors),"fraction_before_eligibility":acquired_frac,"eligible_manuscripts":len(eligible),"eligible_rows":len(rows),"pages_per_manuscript":{k:len(pages[k]) for k in sorted(eligible)}}
    print("ACQUISITION_SUMMARY="+json.dumps(acquisition),flush=True)
    if acquired_frac < MIN_ACQUIRED_FRAC or len(eligible) < MIN_MANUSCRIPTS:
        res={"verdict":"NONRESOLVING","reason":"acquisition gate","acquisition":acquisition,"errors":errors}
        (OUT/"confound_result.json").write_text(json.dumps(res,indent=2))
        print("RESULT="+json.dumps(res),flush=True); return 3

    y=np.array([r["candidate_key"] for r in rows]); groups=np.array([r["image_url"] for r in rows])
    token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    processor=AutoImageProcessor.from_pretrained(MODEL, token=token)
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model=AutoModel.from_pretrained(MODEL, token=token, torch_dtype=dtype, low_cpu_mem_usage=True)
    device="cuda" if torch.cuda.is_available() else "cpu"; model.to(device); model.eval()
    print("MODEL_LOADED="+json.dumps({"device":device,"dtype":str(dtype),"hidden":getattr(model.config,"hidden_size",None)}),flush=True)

    results={}
    for name in VARIANTS:
        X=embed(rows,processor,model,device,name)
        auc,acc,classes,folds=page_heldout_oof(X,y,groups)
        diag=random_diag_oof(X,y)
        results[name]={"grouped_macro_ovr_auc":auc,"grouped_top1_accuracy":acc,"decision":decision(auc),"n":len(y),"manuscripts":len(classes),"random_crop_diagnostic":diag}
        print("CONFOUND="+json.dumps({"variant":name,**results[name]}),flush=True)
    overall="ALL_FAIL" if all(v["decision"]=="FAIL" for v in results.values()) else "SURVIVING_REPRESENTATION"
    res={"verdict":overall,"model":MODEL,"acquisition":acquisition,"results":results,"errors":errors}
    (OUT/"confound_result.json").write_text(json.dumps(res,indent=2))
    print("RESULT="+json.dumps(res),flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
