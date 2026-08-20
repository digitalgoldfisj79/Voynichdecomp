#!/usr/bin/env python3
import argparse, heapq, json, os, time
from pathlib import Path

import requests

from sobel_scan import load_query_mask, make_templates, rest_rows, fetch_gray, score_page, region_url


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--shard',type=int,required=True)
    ap.add_argument('--start-offset',type=int,required=True)
    ap.add_argument('--release',default=os.environ.get('RELEASE_ID','2026-08-19-r10'))
    ap.add_argument('--max-items',type=int,default=3000)
    ap.add_argument('--top-k',type=int,default=100)
    ap.add_argument('--page-size',type=int,default=250)
    ap.add_argument('--query',default='experiments/sobel_shape_search/query_mask.b64')
    ap.add_argument('--out',required=True)
    args=ap.parse_args()

    base=os.environ['SUPABASE_URL']; key=os.environ['SUPABASE_PUBLISHABLE_KEY']
    q=load_query_mask(args.query)
    templates=make_templates(q,widths=[24,32,42,54,70,90,115,145],angles=[-10,-5,0,5,10])
    if not templates:
        raise RuntimeError('no query templates')

    sess=requests.Session()
    offset=args.start_offset
    seen=0; ok=0; errors=0; t0=time.time(); heap=[]; err_examples=[]

    while seen < args.max_items:
        want=min(args.page_size,args.max_items-seen)
        rows=rest_rows(sess,base,key,args.release,args.shard,offset,want)
        if not rows:
            break
        for row in rows:
            seen += 1
            try:
                gray=fetch_gray(sess,row['search_url'])
                best=score_page(gray,templates)
                if best is None:
                    raise RuntimeError('no valid template')
                ok += 1
                rec={**best,
                    'work_id':row['work_id'],'manuscript_id':row['manuscript_id'],'canvas_index':row.get('canvas_index'),
                    'folio_label':row.get('folio_label'),'canvas_id':row.get('canvas_id'),'search_url':row.get('search_url'),
                    'thumbnail_url':row.get('thumbnail_url'),'source_image_url':row.get('source_image_url'),
                    'image_service_id':row.get('image_service_id'),'region_url':region_url(row,best),
                    'source_width':row.get('width'),'source_height':row.get('height'),
                    'shard':args.shard,'start_offset':args.start_offset}
                item=(-rec['score'], rec['work_id'], rec)
                if len(heap) < args.top_k:
                    heapq.heappush(heap,item)
                elif item > heap[0]:
                    heapq.heapreplace(heap,item)
            except Exception as e:
                errors += 1
                if len(err_examples) < 20:
                    err_examples.append({'work_id':row.get('work_id'),'url':row.get('search_url'),'error':str(e)})
            if seen % 250 == 0:
                elapsed=time.time()-t0
                print(json.dumps({'shard':args.shard,'start_offset':args.start_offset,'seen':seen,'ok':ok,'errors':errors,'sec':round(elapsed,1),'rate_per_s':round(seen/max(elapsed,1e-9),2)}),flush=True)
        offset += len(rows)
        if len(rows) < want:
            break

    top=sorted((x[2] for x in heap),key=lambda r:r['score'])
    summary={
        'version':'sobel-shape-search-v0.3-full','release_id':args.release,'shard':args.shard,
        'start_offset':args.start_offset,'seen':seen,'ok':ok,'errors':errors,'elapsed_sec':time.time()-t0,
        'templates':len(templates),'top_k':len(top),'error_examples':err_examples,
        'best_score':top[0]['score'] if top else None,
        'p10_top_score':top[min(len(top)-1,max(0,len(top)//10))]['score'] if top else None,
        'checkpoint_offset':offset,'complete':seen < args.max_items or seen == args.max_items
    }
    Path(args.out).write_text(json.dumps({'summary':summary,'results':top},indent=2))
    print('SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':
    main()
