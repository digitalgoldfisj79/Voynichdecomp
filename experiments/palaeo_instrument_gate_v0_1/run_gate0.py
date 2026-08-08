#!/usr/bin/env python3
from __future__ import annotations
import ast, io, json, math, os, random, time
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageFilter
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoImageProcessor, AutoModel

SEED = 20260808
RNG = np.random.default_rng(SEED)
MODEL = "facebook/dinov2-small"
REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
MANIFEST_URL = "https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/experiment/alpine-venetian-corridor-v0.1-20260808/experiments/alpine_venetian_corridor_v0_1/stage5_confound_manifest.tsv"
HEADERS = {"User-Agent":"PalaeoInstrumentGate/0.1"}


def morphology8(im: Image.Image) -> np.ndarray:
    g = np.asarray(im.convert("L"), dtype=np.float64) / 255.0
    w = np.clip(1.0 - g, 0.0, 1.0)
    h, wid = w.shape
    yy, xx = np.mgrid[0:h, 0:wid]
    x = -1.0 + 2.0 * xx / max(wid - 1, 1)
    y = -1.0 + 2.0 * yy / max(h - 1, 1)
    mass = float(w.sum())
    denom = max(mass, 1e-12)
    cx = float((w*x).sum()/denom); cy = float((w*y).sum()/denom)
    vx = float((w*(x-cx)**2).sum()/denom); vy = float((w*(y-cy)**2).sum()/denom)
    cov = float((w*(x-cx)*(y-cy)).sum()/denom)
    hs = float(np.mean(np.abs(w - np.fliplr(w))))
    vs = float(np.mean(np.abs(w - np.flipud(w))))
    return np.array([mass/(h*wid), cx, cy, vx, vy, cov, hs, vs], dtype=np.float32)


def dino_embed(images, processor, model, device, batch=64):
    out=[]
    with torch.inference_mode():
        for s in range(0,len(images),batch):
            inp=processor(images=images[s:s+batch], return_tensors="pt")
            inp={k:v.to(device) for k,v in inp.items()}
            z=model(**inp).last_hidden_state[:,0,:].float()
            z=torch.nn.functional.normalize(z,dim=1)
            out.append(z.cpu().numpy())
    return np.concatenate(out,axis=0)


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0,max_iter=5000,random_state=SEED,solver="lbfgs"))


def g0a(processor,model,device):
    ds=load_digits()
    ims=[]
    for a in ds.images:
        # sklearn digits: 0=background, 16=ink. Preserve requested 72x64 preprocessor input.
        u=np.clip(a/16.0*255,0,255).astype(np.uint8)
        im=Image.fromarray(255-u,mode="L").resize((64,72),Image.Resampling.BICUBIC).convert("RGB")
        ims.append(im)
    y=ds.target.astype(int)
    idx=np.arange(len(y))
    tr,te=train_test_split(idx,test_size=0.30,random_state=SEED,stratify=y)
    Xn=np.stack([morphology8(x) for x in ims])
    Xd=dino_embed(ims,processor,model,device,batch=64)
    pn=clf().fit(Xn[tr],y[tr]).predict(Xn[te])
    pdn=clf().fit(Xd[tr],y[tr]).predict(Xd[te])
    an=balanced_accuracy_score(y[te],pn); ad=balanced_accuracy_score(y[te],pdn)
    # paired bootstrap on ordinary accuracy difference; stratification is already in the held-out test.
    correct_n=(pn==y[te]).astype(float); correct_d=(pdn==y[te]).astype(float)
    diffs=[]
    n=len(te)
    for _ in range(2000):
        b=RNG.integers(0,n,size=n)
        diffs.append(float((correct_d[b]-correct_n[b]).mean()))
    lo,hi=np.percentile(diffs,[2.5,97.5])
    verdict="PASS" if lo>=0 else ("FAIL" if hi<0 else "INDETERMINATE")
    return {"n":len(y),"test_n":len(te),"naive_balanced_accuracy":float(an),"dino_balanced_accuracy":float(ad),"difference":float(ad-an),"paired_bootstrap_accuracy_diff_ci95":[float(lo),float(hi)],"verdict":verdict}


def fetch(url,tries=4):
    err=None
    for k in range(tries):
        try:
            r=requests.get(url,headers=HEADERS,timeout=60); r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception as e:
            err=e; time.sleep(1.5*(k+1))
    raise RuntimeError(f"fetch failed {url}: {err}")


def crop_norm(im,box):
    x0,y0,x1,y1=box; w,h=im.size
    p=(max(0,round(x0*w/1000)),max(0,round(y0*h/1000)),min(w,round(x1*w/1000)),min(h,round(y1*h/1000)))
    return im.crop(p)


def bgdiv_gray(im):
    g=im.convert("L"); a=np.asarray(g,dtype=np.float32)
    radius=max(3.0,min(im.size)*0.08)
    bg=np.asarray(g.filter(ImageFilter.GaussianBlur(radius=radius)),dtype=np.float32)
    return np.clip(a/np.maximum(bg,1.0)*240.0,0,255).astype(np.uint8)


def inkmask(im):
    flat=bgdiv_gray(im); m=np.where(flat<215,0,255).astype(np.uint8)
    ii=Image.fromarray(m,mode="L").convert("RGB")
    w,h=ii.size; s=max(w,h); out=Image.new("RGB",(s,s),(255,255,255)); out.paste(ii,((s-w)//2,(s-h)//2)); return out


def source_of(key):
    if key in {"external:bsb_cod_icon_242","external:bsb_clm_14684","registry:mn_munchen_bayerische_staatsbibliothek_clm_14622"}: return "BSB"
    if key in {"external:vat_lat_4082","external:walsperger_pal_lat_1362b"}: return "DIGIVAT"
    return None


def domain_of(key):
    if key in {"external:bsb_cod_icon_242","external:vat_lat_4082"}: return "corridor"
    if key in {"external:bsb_clm_14684","registry:mn_munchen_bayerische_staatsbibliothek_clm_14622","external:walsperger_pal_lat_1362b"}: return "bavaria"
    return None


def fit_source_predict(X,y,tr,te):
    pipe=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight="balanced",max_iter=5000,random_state=SEED,solver="lbfgs"))
    pipe.fit(X[tr],y[tr])
    p=pipe.predict_proba(X[te])[:,list(pipe[-1].classes_).index(1)]
    pred=(p>=0.5).astype(int)
    return float(roc_auc_score(y[te],p)),float(balanced_accuracy_score(y[te],pred)),p


def g0c(processor,model,device):
    txt=requests.get(MANIFEST_URL,headers=HEADERS,timeout=60); txt.raise_for_status()
    df=pd.read_csv(io.StringIO(txt.text),sep="\t"); df["bbox"]=df.bbox_1000.map(ast.literal_eval)
    df["source"]=df.candidate_key.map(source_of); df["domain"]=df.candidate_key.map(domain_of)
    df=df[df.source.notna() & df.domain.notna()].copy()
    cache={}; rows=[]; errors=[]
    for _,r in df.iterrows():
        try:
            if r.image_url not in cache: cache[r.image_url]=fetch(r.image_url)
            c=crop_norm(cache[r.image_url],r.bbox); m=inkmask(c)
            rows.append((r.candidate_key,r.source,r.domain,r.image_url,m))
        except Exception as e: errors.append({"key":r.candidate_key,"url":r.image_url,"error":repr(e)})
    sources=np.array([x[1] for x in rows]); domains=np.array([x[2] for x in rows]); y=(sources=="DIGIVAT").astype(int)
    ims=[x[4] for x in rows]
    Xnaive=np.stack([morphology8(x) for x in ims])
    Xpix=np.stack([np.asarray(x.convert("L").resize((32,32),Image.Resampling.BILINEAR),dtype=np.float32).reshape(-1)/255.0 for x in ims])
    Xdino=dino_embed(ims,processor,model,device,batch=32)
    reps={"naive8":Xnaive,"pixels32":Xpix,"dinov2_cls":Xdino}
    result={"n":len(rows),"source_counts":dict(zip(*np.unique(sources,return_counts=True))),"domain_counts":dict(zip(*np.unique(domains,return_counts=True))),"errors":errors,"representations":{}}
    for name,X in reps.items():
        dirs=[]; pooled_y=[]; pooled_p=[]
        for train_dom,test_dom in [("corridor","bavaria"),("bavaria","corridor")]:
            tr=np.where(domains==train_dom)[0]; te=np.where(domains==test_dom)[0]
            auc,bacc,p=fit_source_predict(X,y,tr,te)
            dirs.append({"train":train_dom,"test":test_dom,"n_train":len(tr),"n_test":len(te),"auc":auc,"balanced_accuracy":bacc})
            pooled_y.extend(y[te].tolist()); pooled_p.extend(p.tolist())
        mean_auc=float(np.mean([d["auc"] for d in dirs])); pooled_auc=float(roc_auc_score(pooled_y,pooled_p))
        # Permute training labels independently within each domain; test labels remain factual.
        perm=[]
        for b in range(999):
            aucs=[]
            for train_dom,test_dom in [("corridor","bavaria"),("bavaria","corridor")]:
                tr=np.where(domains==train_dom)[0]; te=np.where(domains==test_dom)[0]
                yp=y.copy(); yp[tr]=RNG.permutation(yp[tr])
                try:
                    auc,_,_=fit_source_predict(X,yp,tr,te); aucs.append(auc)
                except Exception:
                    aucs.append(0.5)
            perm.append(float(np.mean(aucs)))
        pval=float((1+sum(v>=mean_auc for v in perm))/(1+len(perm)))
        fail=(mean_auc>=0.70 and min(d["auc"] for d in dirs)>=0.65 and pval<=0.05)
        result["representations"][name]={"directions":dirs,"mean_directional_auc":mean_auc,"pooled_auc":pooled_auc,"permutation_p":pval,"verdict":"FAIL" if fail else "INDETERMINATE"}
    result["verdict"]="FAIL" if any(v["verdict"]=="FAIL" for v in result["representations"].values()) else "INDETERMINATE"
    return result


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    processor=AutoImageProcessor.from_pretrained(MODEL,revision=REVISION,token=token)
    model=AutoModel.from_pretrained(MODEL,revision=REVISION,token=token)
    device="cuda" if torch.cuda.is_available() else "cpu"; model.to(device); model.eval()
    print("MODEL="+json.dumps({"model":MODEL,"revision":REVISION,"device":device}),flush=True)
    a=g0a(processor,model,device); print("G0A="+json.dumps(a,sort_keys=True),flush=True)
    c=g0c(processor,model,device); print("G0C="+json.dumps(c,sort_keys=True),flush=True)
    print("FINAL="+json.dumps({"g0a":a,"g0b":{"verdict":"FAIL","binding_auc":0.7905697030,"threshold":0.70,"source":"Stage5 sealed pre-target confound gate"},"g0c":c},sort_keys=True),flush=True)

if __name__=="__main__": main()
