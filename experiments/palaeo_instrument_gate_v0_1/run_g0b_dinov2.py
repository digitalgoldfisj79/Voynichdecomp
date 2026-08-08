#!/usr/bin/env python3
from __future__ import annotations
import ast, hashlib, io, json, os, time
from collections import defaultdict
import numpy as np, pandas as pd, requests
from PIL import Image, ImageFilter
import torch
from transformers import AutoImageProcessor, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

SEED=20260808
MODEL='facebook/dinov2-small'
REVISION='ed25f3a31f01632728cabb09d1542f84ab7b0056'
MANIFEST_URL='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/experiment/alpine-venetian-corridor-v0.1-20260808/experiments/alpine_venetian_corridor_v0_1/stage5_confound_manifest.tsv'
VARIANTS=['rgb_norm_v1','gray_bgdiv_v1','inkmask_v1']
HEADERS={'User-Agent':'PalaeoInstrumentGate/0.1'}

def fetch(url):
    e=None
    for k in range(3):
        try:
            r=requests.get(url,headers=HEADERS,timeout=30); r.raise_for_status(); return Image.open(io.BytesIO(r.content)).convert('RGB')
        except Exception as ex: e=ex; time.sleep(1+k)
    raise RuntimeError(f'{url}: {e}')

def crop_norm(im,b):
    x0,y0,x1,y1=b; w,h=im.size
    p=(max(0,round(x0*w/1000)),max(0,round(y0*h/1000)),min(w,round(x1*w/1000)),min(h,round(y1*h/1000)))
    if p[2]<=p[0] or p[3]<=p[1]: raise ValueError(p)
    return im.crop(p)

def square(im):
    w,h=im.size; s=max(w,h); z=Image.new('RGB',(s,s),(255,255,255)); z.paste(im,((s-w)//2,(s-h)//2)); return z

def bgdiv(im):
    g=im.convert('L'); a=np.asarray(g,dtype=np.float32); radius=max(3.,min(im.size)*.08)
    bg=np.asarray(g.filter(ImageFilter.GaussianBlur(radius=radius)),dtype=np.float32)
    return np.clip(a/np.maximum(bg,1.)*240.,0,255).astype(np.uint8)

def variant(im,name):
    if name=='rgb_norm_v1': return square(im.convert('RGB'))
    f=bgdiv(im)
    if name=='gray_bgdiv_v1': return square(Image.fromarray(f,'L').convert('RGB'))
    if name=='inkmask_v1': return square(Image.fromarray(np.where(f<215,0,255).astype(np.uint8),'L').convert('RGB'))
    raise KeyError(name)

def embed(ims,proc,model,dev,batch=32):
    zs=[]
    with torch.inference_mode():
        for s in range(0,len(ims),batch):
            q=proc(images=ims[s:s+batch],return_tensors='pt'); q={k:v.to(dev) for k,v in q.items()}
            z=model(**q).last_hidden_state[:,0,:].float(); z=torch.nn.functional.normalize(z,dim=1); zs.append(z.cpu().numpy())
    return np.concatenate(zs)

def oof(X,y,groups):
    le=LabelEncoder().fit(y); yi=le.transform(y); K=len(le.classes_); P=np.full((len(y),K),np.nan)
    for g in sorted(set(groups)):
        te=(groups==g); tr=~te
        pipe=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=5000,random_state=SEED,solver='lbfgs'))
        pipe.fit(X[tr],yi[tr]); p=pipe.predict_proba(X[te]); cls=pipe[-1].classes_.astype(int)
        for j,c in enumerate(cls): P[te,c]=p[:,j]
    if np.isnan(P).any(): raise RuntimeError('incomplete OOF')
    return float(roc_auc_score(yi,P,multi_class='ovr',average='macro',labels=np.arange(K))),float(accuracy_score(yi,np.argmax(P,axis=1)))
def decision(a): return 'PASS' if a<=.65 else ('CAUTION' if a<=.70 else 'FAIL')

def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    r=requests.get(MANIFEST_URL,headers=HEADERS,timeout=30); r.raise_for_status(); df=pd.read_csv(io.StringIO(r.text),sep='\t'); assert len(df)==59; df['bbox']=df.bbox_1000.map(ast.literal_eval)
    cache={}; rows=[]; errors=[]
    for i,x in df.iterrows():
        try:
            if x.image_url not in cache: cache[x.image_url]=fetch(x.image_url)
            rows.append({'candidate_key':str(x.candidate_key),'image_url':str(x.image_url),'crop':crop_norm(cache[x.image_url],x.bbox)})
        except Exception as e: errors.append({'row':int(i),'key':str(x.candidate_key),'error':repr(e)})
    pages=defaultdict(set)
    for x in rows: pages[x['candidate_key']].add(x['image_url'])
    eligible={k for k,v in pages.items() if len(v)>=2}; rows=[x for x in rows if x['candidate_key'] in eligible]
    y=np.array([x['candidate_key'] for x in rows]); groups=np.array([x['image_url'] for x in rows])
    token=os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN'); proc=AutoImageProcessor.from_pretrained(MODEL,revision=REVISION,token=token); model=AutoModel.from_pretrained(MODEL,revision=REVISION,token=token); dev='cuda' if torch.cuda.is_available() else 'cpu'; model.to(dev).eval()
    out={'model':MODEL,'revision':REVISION,'requested':59,'eligible_rows':int(len(rows)),'eligible_manuscripts':int(len(eligible)),'errors':errors,'variants':{}}
    for name in VARIANTS:
        X=embed([variant(x['crop'],name) for x in rows],proc,model,dev); auc,acc=oof(X,y,groups); out['variants'][name]={'page_heldout_macro_ovr_auc':auc,'top1':acc,'decision':decision(auc)}
    out['g0b_verdict']=out['variants']['inkmask_v1']['decision']
    print('G0B_DINOV2='+json.dumps(out,sort_keys=True),flush=True)
if __name__=='__main__': main()
