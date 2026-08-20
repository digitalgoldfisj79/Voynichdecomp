#!/usr/bin/env python3
import argparse, base64, io, json, math, os, time, zipfile
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

REPO = "digitalgoldfisj79/Voynichdecomp"
PILOT_RUN = 32339977419
MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"


def cosine(a, b):
    a=np.asarray(a,dtype=np.float32).ravel(); b=np.asarray(b,dtype=np.float32).ravel()
    na=float(np.linalg.norm(a)); nb=float(np.linalg.norm(b))
    if na==0 or nb==0: return 0.0
    return float(np.dot(a,b)/(na*nb))


def http_get(sess, url, **kwargs):
    last=None
    for i in range(5):
        try:
            r=sess.get(url, timeout=kwargs.pop('timeout',45), **kwargs)
            r.raise_for_status(); return r
        except Exception as e:
            last=e; time.sleep(1.5*(2**i))
    raise last


def download_pilot(sess, token):
    h={'Authorization':f'Bearer {token}','Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'}
    r=http_get(sess,f'https://api.github.com/repos/{REPO}/actions/runs/{PILOT_RUN}/artifacts',headers=h)
    arts=r.json().get('artifacts',[])
    a=next((x for x in arts if x.get('name')=='sobel-pilot-combined' and not x.get('expired')),None)
    if not a: raise RuntimeError('pilot combined artifact not found')
    z=http_get(sess,a['archive_download_url'],headers=h,timeout=90).content
    with zipfile.ZipFile(io.BytesIO(z)) as zz:
        name=next(n for n in zz.namelist() if n.endswith('sobel_pilot_combined.json'))
        return json.loads(zz.read(name))


def load_mask(path):
    raw=base64.b64decode(Path(path).read_text().strip())
    a=np.array(Image.open(io.BytesIO(raw)).convert('L'))>127
    ys,xs=np.where(a)
    return a[ys.min():ys.max()+1,xs.min():xs.max()+1]


def transformed_query(mask, base_w, ang):
    ar=mask.shape[0]/mask.shape[1]
    h=max(10,int(round(base_w*ar)))
    m=cv2.resize((mask*255).astype(np.uint8),(int(base_w),h),interpolation=cv2.INTER_NEAREST)
    H,W=m.shape; pad=max(H,W)//3+4
    mp=cv2.copyMakeBorder(m,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=0)
    c=(mp.shape[1]/2,mp.shape[0]/2)
    M=cv2.getRotationMatrix2D(c,float(ang),1.0)
    mr=cv2.warpAffine(mp,M,(mp.shape[1],mp.shape[0]),flags=cv2.INTER_NEAREST,borderValue=0)
    ys,xs=np.where(mr>0)
    if len(xs)==0: return mr
    return mr[ys.min():ys.max()+1,xs.min():xs.max()+1]


def edge128(gray):
    g=cv2.resize(gray,(128,128),interpolation=cv2.INTER_AREA)
    g=cv2.GaussianBlur(g,(3,3),0)
    e=cv2.Canny(g,40,120)
    return e


def hog_vec(edge):
    hog=cv2.HOGDescriptor((128,128),(32,32),(16,16),(16,16),9)
    return hog.compute(edge).ravel()


def fetch_candidate_crop(sess, rec):
    r=http_get(sess,rec['search_url'],headers={'User-Agent':'ManuComp-CosineCalibration/0.1'},timeout=45)
    gray=np.array(Image.open(io.BytesIO(r.content)).convert('L'))
    if gray.shape[1] != int(rec['page_w']):
        s=float(rec['page_w'])/gray.shape[1]
        gray=cv2.resize(gray,(int(rec['page_w']),max(1,int(round(gray.shape[0]*s)))),interpolation=cv2.INTER_AREA)
    x,y,w,h=[int(rec[k]) for k in ('x','y','w','h')]
    x=max(0,min(x,gray.shape[1]-1)); y=max(0,min(y,gray.shape[0]-1))
    crop=gray[y:min(gray.shape[0],y+h),x:min(gray.shape[1],x+w)]
    if crop.size==0: raise RuntimeError('empty localized crop')
    return crop


def square_rgb_from_gray(gray, size=256):
    im=Image.fromarray(gray.astype(np.uint8)).convert('RGB')
    canvas=Image.new('RGB',(size,size),'white')
    im.thumbnail((int(size*.86),int(size*.86)),Image.Resampling.LANCZOS)
    canvas.paste(im,((size-im.width)//2,(size-im.height)//2))
    return canvas


def query_rgb(mask, size=256):
    ys,xs=np.where(mask)
    m=(~mask*255).astype(np.uint8)  # black ink on white
    return square_rgb_from_gray(m,size)


def run_dino(rows, mask, limit):
    status={'attempted':True,'model':MODEL,'revision':REVISION,'ok':False}
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        proc=AutoImageProcessor.from_pretrained(MODEL,revision=REVISION)
        model=AutoModel.from_pretrained(MODEL,revision=REVISION)
        model.eval()
        qimg=query_rgb(mask)
        with torch.no_grad():
            qi=proc(images=[qimg],return_tensors='pt')
            qo=model(**qi)
            qv=qo.last_hidden_state[:,0,:]
            qv=torch.nn.functional.normalize(qv,dim=1)
        sess=requests.Session(); selected=rows[:limit]
        done=[]
        for i in range(0,len(selected),4):
            batch=selected[i:i+4]; imgs=[]; keep=[]
            for r in batch:
                try:
                    imgs.append(square_rgb_from_gray(fetch_candidate_crop(sess,r)))
                    keep.append(r)
                except Exception as e:
                    r['dino_error']=str(e)
            if not imgs: continue
            with torch.no_grad():
                inp=proc(images=imgs,return_tensors='pt')
                out=model(**inp)
                v=torch.nn.functional.normalize(out.last_hidden_state[:,0,:],dim=1)
                sims=(v@qv.T).squeeze(1).cpu().numpy().tolist()
            for r,s in zip(keep,sims):
                r['dino_cosine']=float(s); done.append(r)
        status.update(ok=True,n=len(done))
        return status
    except Exception as e:
        status['error']=repr(e)
        return status


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--query',default='experiments/sobel_shape_search/query_mask.b64')
    ap.add_argument('--classical-top',type=int,default=200)
    ap.add_argument('--dino-top',type=int,default=48)
    ap.add_argument('--out',default='cosine_pilot.json')
    args=ap.parse_args()
    token=os.environ.get('GITHUB_TOKEN')
    if not token: raise RuntimeError('GITHUB_TOKEN missing')
    sess=requests.Session(); pilot=download_pilot(sess,token)
    rows=pilot.get('results',[])[:args.classical_top]
    mask=load_mask(args.query)
    completed=[]; errors=[]
    for i,r0 in enumerate(rows,1):
        r=dict(r0)
        try:
            crop=fetch_candidate_crop(sess,r)
            ce=edge128(crop)
            tq=transformed_query(mask,r.get('base_width',64),r.get('rotation_deg',0))
            qe=edge128(tq)
            r['hog_cosine']=cosine(hog_vec(qe),hog_vec(ce))
            r['edge_cosine']=cosine((qe>0).astype(np.float32),(ce>0).astype(np.float32))
            r['classical_cosine']=0.75*r['hog_cosine']+0.25*r['edge_cosine']
            completed.append(r)
        except Exception as e:
            errors.append({'rank':i,'work_id':r.get('work_id'),'error':str(e)})
        if i%25==0: print(json.dumps({'event':'classical_progress','seen':i,'ok':len(completed),'errors':len(errors)}),flush=True)
    # preserve breadth: half from Sobel rank, half from cosine rank
    bycos=sorted(completed,key=lambda x:x['classical_cosine'],reverse=True)
    n=max(1,args.dino_top//2); chosen=[]; ids=set()
    for r in completed[:n]+bycos[:n]:
        k=r.get('work_id')
        if k not in ids: ids.add(k); chosen.append(r)
    dino_status=run_dino(chosen,mask,args.dino_top) if args.dino_top>0 else {'attempted':False}
    # propagate DINO scores from chosen back to completed by work_id
    dm={r.get('work_id'):r.get('dino_cosine') for r in chosen if r.get('dino_cosine') is not None}
    for r in completed:
        if r.get('work_id') in dm: r['dino_cosine']=dm[r['work_id']]
    out={
      'version':'cosine-calibration-v0.1','source_pilot_run':PILOT_RUN,'source_pages':3000,
      'classical_attempted':len(rows),'classical_ok':len(completed),'errors':errors,
      'dino':dino_status,
      'by_sobel':sorted(completed,key=lambda x:x['score'])[:50],
      'by_classical_cosine':bycos[:50],
      'by_dino_cosine':sorted([r for r in completed if r.get('dino_cosine') is not None],key=lambda x:x['dino_cosine'],reverse=True)[:50]
    }
    Path(args.out).write_text(json.dumps(out,indent=2))
    md=Path(args.out).with_suffix('.md')
    with md.open('w') as f:
        f.write('# Cosine calibration on completed Sobel pilot\n\n')
        f.write(f"Classical cosine: {len(completed)}/{len(rows)} candidates processed; errors: {len(errors)}.  \\n")
        f.write(f"DINOv3: {'OK ('+str(dino_status.get('n',0))+' crops)' if dino_status.get('ok') else 'not available: '+str(dino_status.get('error','not attempted'))}.\n\n")
        f.write('| rank | classical cosine | HOG | edge | Sobel | manuscript | folio |\n|---:|---:|---:|---:|---:|---|---|\n')
        for i,r in enumerate(bycos[:25],1):
            f.write(f"| {i} | {r['classical_cosine']:.4f} | {r['hog_cosine']:.4f} | {r['edge_cosine']:.4f} | {r['score']:.4f} | `{r['manuscript_id']}` | {r.get('folio_label') or r.get('canvas_index')} |\n")
        if out['by_dino_cosine']:
            f.write('\n## DINOv3 cosine\n\n| rank | DINO cosine | classical cosine | Sobel | manuscript | folio |\n|---:|---:|---:|---:|---|---|\n')
            for i,r in enumerate(out['by_dino_cosine'][:25],1):
                f.write(f"| {i} | {r['dino_cosine']:.4f} | {r['classical_cosine']:.4f} | {r['score']:.4f} | `{r['manuscript_id']}` | {r.get('folio_label') or r.get('canvas_index')} |\n")
    print(json.dumps({'event':'done','classical_ok':len(completed),'dino':dino_status}),flush=True)

if __name__=='__main__':
    main()
