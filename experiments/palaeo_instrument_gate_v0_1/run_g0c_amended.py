#!/usr/bin/env python3
from __future__ import annotations
import ast, io, json, os, time
import numpy as np, pandas as pd, requests
from PIL import Image, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoImageProcessor, AutoModel

SEED=20260808
MODEL='facebook/dinov2-small'
REVISION='ed25f3a31f01632728cabb09d1542f84ab7b0056'
MANIFEST_URL='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/experiment/alpine-venetian-corridor-v0.1-20260808/experiments/alpine_venetian_corridor_v0_1/stage5_confound_manifest.tsv'
HEADERS={'User-Agent':'PalaeoInstrumentGate/0.1'}

def morphology8(im):
    g=np.asarray(im.convert('L'),dtype=np.float64)/255.; w=np.clip(1-g,0,1); h,ww=w.shape
    yy,xx=np.mgrid[0:h,0:ww]; x=-1+2*xx/max(ww-1,1); y=-1+2*yy/max(h-1,1)
    mass=float(w.sum()); d=max(mass,1e-12); cx=float((w*x).sum()/d); cy=float((w*y).sum()/d)
    vx=float((w*(x-cx)**2).sum()/d); vy=float((w*(y-cy)**2).sum()/d); cov=float((w*(x-cx)*(y-cy)).sum()/d)
    hs=float(np.mean(np.abs(w-np.fliplr(w)))); vs=float(np.mean(np.abs(w-np.flipud(w))))
    return np.array([mass/(h*ww),cx,cy,vx,vy,cov,hs,vs],dtype=np.float32)

def fetch(url):
    e=None
    for k in range(3):
        try:
            r=requests.get(url,headers=HEADERS,timeout=30); r.raise_for_status(); return Image.open(io.BytesIO(r.content)).convert('RGB')
        except Exception as ex: e=ex; time.sleep(1+k)
    raise RuntimeError(f'{url}: {e}')

def crop_norm(im,b):
    x0,y0,x1,y1=b; w,h=im.size
    return im.crop((max(0,round(x0*w/1000)),max(0,round(y0*h/1000)),min(w,round(x1*w/1000)),min(h,round(y1*h/1000))))

def mask(im):
    g=im.convert('L'); a=np.asarray(g,dtype=np.float32); rad=max(3.,min(im.size)*.08)
    bg=np.asarray(g.filter(ImageFilter.GaussianBlur(radius=rad)),dtype=np.float32)
    flat=np.clip(a/np.maximum(bg,1.)*240.,0,255); m=np.where(flat<215,0,255).astype(np.uint8)
    q=Image.fromarray(m,'L').convert('RGB'); w,h=q.size; s=max(w,h); z=Image.new('RGB',(s,s),(255,255,255)); z.paste(q,((s-w)//2,(s-h)//2)); return z

def src(k):
    if k in {'external:bsb_cod_icon_242','external:bsb_clm_14684','registry:mn_munchen_bayerische_staatsbibliothek_clm_14622'}: return 'BSB'
    if k in {'external:vat_lat_4082','external:walsperger_pal_lat_1362b'}: return 'DIGIVAT'
def dom(k):
    if k in {'external:bsb_cod_icon_242','external:vat_lat_4082'}: return 'corridor'
    if k in {'external:bsb_clm_14684','registry:mn_munchen_bayerische_staatsbibliothek_clm_14622','external:walsperger_pal_lat_1362b'}: return 'bavaria'

def emb(ims,proc,model,dev):
    zs=[]
    with torch.inference_mode():
        for s in range(0,len(ims),32):
            q=proc(images=ims[s:s+32],return_tensors='pt'); q={k:v.to(dev) for k,v in q.items()}
            z=model(**q).last_hidden_state[:,0,:].float(); z=torch.nn.functional.normalize(z,dim=1); zs.append(z.cpu().numpy())
    return np.concatenate(zs)

def pred(X,y,tr,te):
    p=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=5000,random_state=SEED,solver='lbfgs'))
    p.fit(X[tr],y[tr]); probs=p.predict_proba(X[te])[:,list(p[-1].classes_).index(1)]; lab=(probs>=.5).astype(int)
    return float(roc_auc_score(y[te],probs)),float(balanced_accuracy_score(y[te],lab)),probs

def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    token=os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
    proc=AutoImageProcessor.from_pretrained(MODEL,revision=REVISION,token=token)
    model=AutoModel.from_pretrained(MODEL,revision=REVISION,token=token); dev='cuda' if torch.cuda.is_available() else 'cpu'; model.to(dev).eval()
    r=requests.get(MANIFEST_URL,headers=HEADERS,timeout=30); r.raise_for_status(); df=pd.read_csv(io.StringIO(r.text),sep='\t'); df['bbox']=df.bbox_1000.map(ast.literal_eval); df['source']=df.candidate_key.map(src); df['domain']=df.candidate_key.map(dom); df=df[df.source.notna()&df.domain.notna()]
    cache={}; rows=[]; errors=[]
    for _,x in df.iterrows():
        try:
            if x.image_url not in cache: cache[x.image_url]=fetch(x.image_url)
            rows.append((str(x.candidate_key),str(x.source),str(x.domain),str(x.image_url),mask(crop_norm(cache[x.image_url],x.bbox))))
        except Exception as e: errors.append({'key':str(x.candidate_key),'url':str(x.image_url),'error':repr(e)})
    sources=np.array([x[1] for x in rows]); domains=np.array([x[2] for x in rows]); manuscripts=np.array([x[0] for x in rows]); y=(sources=='DIGIVAT').astype(int); ims=[x[4] for x in rows]
    reps={'naive8':np.stack([morphology8(i) for i in ims]),'pixels32':np.stack([np.asarray(i.convert('L').resize((32,32),Image.Resampling.BILINEAR),dtype=np.float32).ravel()/255 for i in ims]),'dinov2_cls':emb(ims,proc,model,dev)}
    out={'n_crops':int(len(rows)),'n_manuscripts':int(len(set(manuscripts))),'manuscripts':sorted(set(manuscripts.tolist())),'source_counts':{str(k):int(v) for k,v in zip(*np.unique(sources,return_counts=True))},'domain_counts':{str(k):int(v) for k,v in zip(*np.unique(domains,return_counts=True))},'errors':errors,'representations':{}}
    for name,X in reps.items():
        ds=[]; py=[]; pp=[]
        for a,b in [('corridor','bavaria'),('bavaria','corridor')]:
            tr=np.where(domains==a)[0]; te=np.where(domains==b)[0]; auc,ba,probs=pred(X,y,tr,te)
            ds.append({'train':a,'test':b,'n_train':int(len(tr)),'n_test':int(len(te)),'auc':auc,'balanced_accuracy':ba})
            py += y[te].tolist(); pp += probs.tolist()
        mean=float(np.mean([d['auc'] for d in ds])); pooled=float(roc_auc_score(py,pp)); diag=(mean>=.70 and min(d['auc'] for d in ds)>=.65)
        out['representations'][name]={'directions':ds,'mean_directional_auc':mean,'pooled_auc':pooled,'amendment_001_status':'DIAGNOSTIC_LEAKAGE' if diag else 'INDETERMINATE'}
    out['verdict']='DIAGNOSTIC_LEAKAGE' if any(v['amendment_001_status']=='DIAGNOSTIC_LEAKAGE' for v in out['representations'].values()) else 'INDETERMINATE'
    print('G0C_AMENDED='+json.dumps(out,sort_keys=True),flush=True)
if __name__=='__main__': main()
