#!/usr/bin/env python3
import argparse, base64, io, json, math, os, time, heapq
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image


def load_query_mask(path: str) -> np.ndarray:
    raw = base64.b64decode(Path(path).read_text().strip())
    arr = np.array(Image.open(io.BytesIO(raw)).convert('L'))
    mask = arr > 127
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError('query mask has no foreground')
    return mask[ys.min():ys.max()+1, xs.min():xs.max()+1]


def make_templates(mask: np.ndarray, widths, angles):
    out=[]
    ar = mask.shape[0]/mask.shape[1]
    for w in widths:
        h=max(10, int(round(w*ar)))
        m=cv2.resize((mask*255).astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)
        for ang in angles:
            H,W=m.shape
            pad=max(H,W)//3+4
            mp=cv2.copyMakeBorder(m,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=0)
            c=(mp.shape[1]/2,mp.shape[0]/2)
            M=cv2.getRotationMatrix2D(c,ang,1.0)
            mr=cv2.warpAffine(mp,M,(mp.shape[1],mp.shape[0]),flags=cv2.INTER_NEAREST,borderValue=0)
            ys,xs=np.where(mr>0)
            if len(xs)==0:
                continue
            mr=mr[ys.min():ys.max()+1,xs.min():xs.max()+1]
            qe=cv2.Canny(mr,40,120)>0
            if qe.sum()<12:
                continue
            ys,xs=np.where(qe)
            qe=qe[ys.min():ys.max()+1,xs.min():xs.max()+1]
            out.append((w,float(ang),qe))
    return out


def page_edges(gray: np.ndarray):
    if gray.shape[1] > 768:
        s=768/gray.shape[1]
        gray=cv2.resize(gray,(768,max(1,int(round(gray.shape[0]*s)))),interpolation=cv2.INTER_AREA)
    g=cv2.GaussianBlur(gray,(3,3),0)
    med=float(np.median(g))
    lo=max(20,int(0.45*med)); hi=min(245,max(lo+30,int(0.85*med)))
    e=cv2.Canny(g,lo,hi)>0
    return gray,e


def score_page(gray: np.ndarray, templates):
    gray, pe = page_edges(gray)
    dist=cv2.distanceTransform((~pe).astype(np.uint8),cv2.DIST_L2,3).astype(np.float32)
    best=None
    for base_w, ang, qe in templates:
        h,w=qe.shape
        if h>=dist.shape[0] or w>=dist.shape[1]:
            continue
        k=qe.astype(np.float32)
        n=float(k.sum())
        surf=cv2.matchTemplate(dist,k,cv2.TM_CCORR)/n
        mn,_,mnloc,_=cv2.minMaxLoc(surf)
        x,y=mnloc
        patch=pe[y:y+h,x:x+w]
        qn=max(1,int(qe.sum())); pn=max(1,int(patch.sum()))
        edge_ratio=pn/qn
        forward=float(mn)*64.0/max(1.0,float(base_w))
        density_pen=max(0.0, math.log(max(1.0,edge_ratio/3.0))) * 0.25
        final=forward+density_pen
        cand=dict(score=final,forward=forward,raw_chamfer=float(mn),edge_ratio=edge_ratio,
                  base_width=int(base_w),rotation_deg=float(ang),x=int(x),y=int(y),w=int(w),h=int(h),
                  page_w=int(gray.shape[1]),page_h=int(gray.shape[0]))
        if best is None or cand['score']<best['score']:
            best=cand
    return best


def get_with_retry(session, url, *, params=None, headers=None, timeout=(10, 30), attempts=6, label='request'):
    last=None
    for attempt in range(1, attempts+1):
        try:
            r=session.get(url,params=params,headers=headers,timeout=timeout)
            r.raise_for_status()
            return r
        except (requests.RequestException, TimeoutError) as e:
            last=e
            if attempt >= attempts:
                break
            delay=min(30.0, 1.5*(2**(attempt-1)))
            print(json.dumps({'event':'http_retry','label':label,'attempt':attempt,'delay_sec':delay,'error':str(e)[:300]}),flush=True)
            time.sleep(delay)
    raise last


def rest_rows(session, base, key, release, shard, offset, limit):
    params={
        'select':'work_id,manuscript_id,canvas_index,folio_label,canvas_id,search_url,thumbnail_url,source_image_url,image_service_id,width,height',
        'release_id':f'eq.{release}', 'shard_32':f'eq.{shard}',
        'order':'work_id.asc', 'offset':str(offset), 'limit':str(limit)
    }
    r=get_with_retry(session,base.rstrip('/')+'/rest/v1/manucomp_sobel_worklist_v01',params=params,
                     headers={'apikey':key},timeout=(10,30),attempts=7,label=f'worklist shard={shard} offset={offset}')
    return r.json()


def fetch_gray(session,url):
    r=get_with_retry(session,url,timeout=(10,25),attempts=4,
                     headers={'User-Agent':'ManuComp-SobelSearch/0.2 (+research)'},label='image')
    return np.array(Image.open(io.BytesIO(r.content)).convert('L'))


def region_url(row,best):
    svc=row.get('image_service_id'); ow=row.get('width')
    if not svc or not ow or not best or not best.get('page_w'):
        return None
    try:
        scale=float(ow)/float(best['page_w'])
        x=max(0,int(round(best['x']*scale))); y=max(0,int(round(best['y']*scale)))
        w=max(1,int(round(best['w']*scale))); h=max(1,int(round(best['h']*scale)))
        px=int(w*0.35); py=int(h*0.35)
        x=max(0,x-px); y=max(0,y-py); w=w+2*px; h=h+2*py
        return svc.rstrip('/')+f'/{x},{y},{w},{h}/full/600,/0/default.jpg'
    except Exception:
        return None


def write_snapshot(out, args, heap, seen, ok, errors, err_examples, t0, templates, offset, complete=False):
    top=sorted((x[2] for x in heap),key=lambda r:r['score'])
    summary={'version':'sobel-shape-search-v0.2','release_id':args.release,'shard':args.shard,
             'seen':seen,'ok':ok,'errors':errors,'elapsed_sec':time.time()-t0,
             'templates':len(templates),'top_k':len(top),'error_examples':err_examples,
             'best_score':top[0]['score'] if top else None,
             'p10_top_score':top[min(len(top)-1,max(0,len(top)//10))]['score'] if top else None,
             'checkpoint_offset':offset,'complete':complete}
    Path(out).write_text(json.dumps({'summary':summary,'results':top},indent=2))
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--shard',type=int,required=True)
    ap.add_argument('--release',default=os.environ.get('RELEASE_ID','2026-08-19-r10'))
    ap.add_argument('--max-items',type=int,default=int(os.environ.get('MAX_ITEMS','750')))
    ap.add_argument('--top-k',type=int,default=int(os.environ.get('TOP_K','100')))
    ap.add_argument('--page-size',type=int,default=250)
    ap.add_argument('--query',default='experiments/sobel_shape_search/query_mask.b64')
    ap.add_argument('--out',default=None)
    args=ap.parse_args()

    base=os.environ['SUPABASE_URL']; key=os.environ['SUPABASE_PUBLISHABLE_KEY']
    q=load_query_mask(args.query)
    templates=make_templates(q,widths=[24,32,42,54,70,90,115,145],angles=[-10,-5,0,5,10])
    if not templates:
        raise RuntimeError('no query templates')
    sess=requests.Session()
    out=args.out or f'sobel_results_shard_{args.shard:02d}.json'
    offset=0; seen=0; ok=0; errors=0; t0=time.time(); heap=[]; err_examples=[]

    while seen<args.max_items:
        want=min(args.page_size,args.max_items-seen)
        try:
            rows=rest_rows(sess,base,key,args.release,args.shard,offset,want)
        except Exception as e:
            write_snapshot(out,args,heap,seen,ok,errors,err_examples,t0,templates,offset,complete=False)
            print(json.dumps({'event':'fatal_worklist_error','shard':args.shard,'offset':offset,'seen':seen,'error':str(e)}),flush=True)
            raise
        if not rows:
            break
        for row in rows:
            seen+=1
            try:
                gray=fetch_gray(sess,row['search_url'])
                best=score_page(gray,templates)
                if best is None:
                    raise RuntimeError('no valid template')
                ok+=1
                rec={**best,
                    'work_id':row['work_id'],'manuscript_id':row['manuscript_id'],'canvas_index':row.get('canvas_index'),
                    'folio_label':row.get('folio_label'),'canvas_id':row.get('canvas_id'),'search_url':row.get('search_url'),
                    'thumbnail_url':row.get('thumbnail_url'),'source_image_url':row.get('source_image_url'),
                    'image_service_id':row.get('image_service_id'),'region_url':region_url(row,best),
                    'source_width':row.get('width'),'source_height':row.get('height')}
                item=(-rec['score'], rec['work_id'], rec)
                if len(heap)<args.top_k:
                    heapq.heappush(heap,item)
                elif item>heap[0]:
                    heapq.heapreplace(heap,item)
            except Exception as e:
                errors+=1
                if len(err_examples)<12:
                    err_examples.append({'work_id':row.get('work_id'),'url':row.get('search_url'),'error':str(e)})
            if seen%100==0:
                elapsed=time.time()-t0
                print(json.dumps({'shard':args.shard,'seen':seen,'ok':ok,'errors':errors,'sec':round(elapsed,1),'rate_per_s':round(seen/max(elapsed,1e-9),2)}),flush=True)
        offset+=len(rows)
        write_snapshot(out,args,heap,seen,ok,errors,err_examples,t0,templates,offset,complete=False)
        if len(rows)<want:
            break

    summary=write_snapshot(out,args,heap,seen,ok,errors,err_examples,t0,templates,offset,complete=True)
    print('SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':
    main()
