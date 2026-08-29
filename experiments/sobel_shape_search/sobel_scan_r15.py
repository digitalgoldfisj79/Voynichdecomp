#!/usr/bin/env python3
import argparse, heapq, json, math, os, time
from pathlib import Path

import requests

from sobel_scan import (
    fetch_gray,
    get_with_retry,
    load_query_mask,
    make_templates,
    region_url,
    score_page,
)


def rest_rows(session, base, key, release, shard, offset, limit, min_id, max_id, shard_count):
    span = max_id - min_id + 1
    lo = min_id + (span * shard) // shard_count
    hi = min_id + (span * (shard + 1)) // shard_count
    params = {
        'select': 'id,manuscript_id,canvas_index,folio_label,canvas_id,image_url,thumbnail_url,image_service_id,width,height',
        'release_id': f'eq.{release}',
        'id': f'gte.{lo}',
        'order': 'id.asc',
        'offset': str(offset),
        'limit': str(limit),
        'image_url': 'not.is.null',
    }
    headers = {'apikey': key}
    r = get_with_retry(
        session,
        base.rstrip('/') + '/rest/v1/manucomp_release_images',
        params=params,
        headers=headers,
        timeout=(10, 30),
        attempts=7,
        label=f'release_images shard={shard} offset={offset}',
    )
    rows = r.json()
    # PostgREST cannot express both gte and lt on the same key through a plain dict.
    # Trim the final page locally to this deterministic shard range.
    return [x for x in rows if int(x['id']) < hi], lo, hi


def write_snapshot(out, args, heap, seen, ok, errors, err_examples, t0, templates, offset, lo, hi, complete=False):
    top = sorted((x[2] for x in heap), key=lambda r: r['score'])
    summary = {
        'version': 'sobel-shape-search-r15-v0.1',
        'release_id': args.release,
        'shard': args.shard,
        'shard_count': args.shard_count,
        'id_lo': lo,
        'id_hi_exclusive': hi,
        'seen': seen,
        'ok': ok,
        'errors': errors,
        'elapsed_sec': time.time() - t0,
        'templates': len(templates),
        'top_k': len(top),
        'error_examples': err_examples,
        'best_score': top[0]['score'] if top else None,
        'checkpoint_offset': offset,
        'complete': complete,
    }
    Path(out).write_text(json.dumps({'summary': summary, 'results': top}, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--shard-count', type=int, default=32)
    ap.add_argument('--release', default=os.environ.get('RELEASE_ID', '2026-08-26-r15'))
    ap.add_argument('--min-id', type=int, default=int(os.environ.get('MIN_IMAGE_ID', '8575495')))
    ap.add_argument('--max-id', type=int, default=int(os.environ.get('MAX_IMAGE_ID', '9159973')))
    ap.add_argument('--max-items', type=int, default=int(os.environ.get('MAX_ITEMS', '750')))
    ap.add_argument('--top-k', type=int, default=int(os.environ.get('TOP_K', '150')))
    ap.add_argument('--page-size', type=int, default=250)
    ap.add_argument('--query', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    if not (0 <= args.shard < args.shard_count):
        raise SystemExit('invalid shard')

    base = os.environ['SUPABASE_URL']
    key = os.environ['SUPABASE_PUBLISHABLE_KEY']
    q = load_query_mask(args.query)
    templates = make_templates(q, widths=[24, 32, 42, 54, 70, 90, 115, 145], angles=[-10, -5, 0, 5, 10])
    if not templates:
        raise RuntimeError('no query templates')

    sess = requests.Session()
    offset = seen = ok = errors = 0
    heap = []
    err_examples = []
    t0 = time.time()
    lo = hi = None

    while seen < args.max_items:
        want = min(args.page_size, args.max_items - seen)
        rows, lo, hi = rest_rows(sess, base, key, args.release, args.shard, offset, want, args.min_id, args.max_id, args.shard_count)
        if not rows:
            break
        for row in rows:
            seen += 1
            url = row.get('image_url')
            try:
                gray = fetch_gray(sess, url)
                best = score_page(gray, templates)
                if best is None:
                    raise RuntimeError('no valid template')
                ok += 1
                mapped = {
                    'work_id': int(row['id']),
                    'manuscript_id': row['manuscript_id'],
                    'canvas_index': row.get('canvas_index'),
                    'folio_label': row.get('folio_label'),
                    'canvas_id': row.get('canvas_id'),
                    'search_url': url,
                    'thumbnail_url': row.get('thumbnail_url'),
                    'source_image_url': url,
                    'image_service_id': row.get('image_service_id'),
                    'width': row.get('width'),
                    'height': row.get('height'),
                }
                rec = {
                    **best,
                    **{k: mapped[k] for k in ('work_id','manuscript_id','canvas_index','folio_label','canvas_id','search_url','thumbnail_url','source_image_url','image_service_id')},
                    'region_url': region_url(mapped, best),
                    'source_width': row.get('width'),
                    'source_height': row.get('height'),
                }
                item = (-rec['score'], rec['work_id'], rec)
                if len(heap) < args.top_k:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
            except Exception as e:
                errors += 1
                if len(err_examples) < 12:
                    err_examples.append({'work_id': row.get('id'), 'url': url, 'error': str(e)[:500]})
            if seen % 100 == 0:
                elapsed = time.time() - t0
                print(json.dumps({'shard': args.shard, 'seen': seen, 'ok': ok, 'errors': errors, 'rate_per_s': round(seen/max(elapsed,1e-9),2)}), flush=True)
        offset += len(rows)
        write_snapshot(args.out, args, heap, seen, ok, errors, err_examples, t0, templates, offset, lo, hi, complete=False)
        if len(rows) < want:
            break

    summary = write_snapshot(args.out, args, heap, seen, ok, errors, err_examples, t0, templates, offset, lo, hi, complete=True)
    if seen == 0:
        raise RuntimeError(f'zero visible rows for current release {args.release} shard {args.shard}')
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
