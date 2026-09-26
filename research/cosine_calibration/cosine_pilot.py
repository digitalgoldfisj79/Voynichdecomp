#!/usr/bin/env python3
import argparse, base64, io, json, os, time, zipfile
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

REPO = "digitalgoldfisj79/Voynichdecomp"
PILOT_RUN = 32339977419


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def http_get(sess, url, **kwargs):
    last = None
    timeout = kwargs.pop('timeout', 45)
    for i in range(5):
        try:
            r = sess.get(url, timeout=timeout, **kwargs)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            time.sleep(1.5 * (2 ** i))
    raise last


def download_pilot(sess, token):
    h = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28'
    }
    r = http_get(sess, f'https://api.github.com/repos/{REPO}/actions/runs/{PILOT_RUN}/artifacts', headers=h)
    arts = r.json().get('artifacts', [])
    a = next((x for x in arts if x.get('name') == 'sobel-pilot-combined' and not x.get('expired')), None)
    if not a:
        raise RuntimeError('pilot combined artifact not found')
    z = http_get(sess, a['archive_download_url'], headers=h, timeout=90).content
    with zipfile.ZipFile(io.BytesIO(z)) as zz:
        name = next(n for n in zz.namelist() if n.endswith('sobel_pilot_combined.json'))
        return json.loads(zz.read(name))


def load_mask(path):
    raw = base64.b64decode(Path(path).read_text().strip())
    a = np.array(Image.open(io.BytesIO(raw)).convert('L')) > 127
    ys, xs = np.where(a)
    if not len(xs):
        raise RuntimeError('query mask has no foreground')
    return a[ys.min():ys.max()+1, xs.min():xs.max()+1]


def transformed_query(mask, base_w, ang):
    ar = mask.shape[0] / mask.shape[1]
    h = max(10, int(round(base_w * ar)))
    m = cv2.resize((mask * 255).astype(np.uint8), (int(base_w), h), interpolation=cv2.INTER_NEAREST)
    H, W = m.shape
    pad = max(H, W) // 3 + 4
    mp = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    c = (mp.shape[1] / 2, mp.shape[0] / 2)
    M = cv2.getRotationMatrix2D(c, float(ang), 1.0)
    mr = cv2.warpAffine(mp, M, (mp.shape[1], mp.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)
    ys, xs = np.where(mr > 0)
    if not len(xs):
        return mr
    return mr[ys.min():ys.max()+1, xs.min():xs.max()+1]


def edge128(gray):
    g = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    return cv2.Canny(g, 40, 120)


def hog_vec(edge, cell=16, bins=9):
    """Small explicit unsigned-HOG implementation, independent of cv2.HOGDescriptor."""
    img = edge.astype(np.float32) / 255.0
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = np.mod(ang, 180.0)
    ny = img.shape[0] // cell; nx = img.shape[1] // cell
    hist = np.zeros((ny, nx, bins), dtype=np.float32)
    bw = 180.0 / bins
    for cy in range(ny):
        for cx in range(nx):
            ys = slice(cy * cell, (cy + 1) * cell)
            xs = slice(cx * cell, (cx + 1) * cell)
            m = mag[ys, xs].ravel()
            a = ang[ys, xs].ravel() / bw
            b0 = np.floor(a).astype(np.int32) % bins
            frac = a - np.floor(a)
            b1 = (b0 + 1) % bins
            np.add.at(hist[cy, cx], b0, m * (1.0 - frac))
            np.add.at(hist[cy, cx], b1, m * frac)
    blocks = []
    eps = 1e-6
    for cy in range(max(1, ny - 1)):
        for cx in range(max(1, nx - 1)):
            block = hist[cy:min(cy+2, ny), cx:min(cx+2, nx)].ravel()
            block = block / np.sqrt(float(np.dot(block, block)) + eps)
            blocks.append(block)
    return np.concatenate(blocks) if blocks else hist.ravel()


def fetch_candidate_crop(sess, rec):
    r = http_get(sess, rec['search_url'], headers={'User-Agent':'ManuComp-CosineCalibration/0.3'}, timeout=45)
    gray = np.array(Image.open(io.BytesIO(r.content)).convert('L'))
    if gray.shape[1] != int(rec['page_w']):
        s = float(rec['page_w']) / gray.shape[1]
        gray = cv2.resize(gray, (int(rec['page_w']), max(1, int(round(gray.shape[0] * s)))), interpolation=cv2.INTER_AREA)
    x, y, w, h = [int(rec[k]) for k in ('x','y','w','h')]
    x = max(0, min(x, gray.shape[1]-1)); y = max(0, min(y, gray.shape[0]-1))
    crop = gray[y:min(gray.shape[0], y+h), x:min(gray.shape[1], x+w)]
    if crop.size == 0:
        raise RuntimeError('empty localized crop')
    return crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--query', default='experiments/sobel_shape_search/query_mask.b64')
    ap.add_argument('--top', type=int, default=200)
    ap.add_argument('--out', default='cosine_pilot.json')
    args = ap.parse_args()

    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        raise RuntimeError('GITHUB_TOKEN missing')
    sess = requests.Session()
    pilot = download_pilot(sess, token)
    rows = pilot.get('results', [])[:args.top]
    mask = load_mask(args.query)
    completed = []; errors = []

    for i, r0 in enumerate(rows, 1):
        r = dict(r0)
        try:
            crop = fetch_candidate_crop(sess, r)
            ce = edge128(crop)
            tq = transformed_query(mask, r.get('base_width', 64), r.get('rotation_deg', 0))
            qe = edge128(tq)
            r['hog_cosine'] = cosine(hog_vec(qe), hog_vec(ce))
            r['edge_cosine'] = cosine((qe > 0).astype(np.float32), (ce > 0).astype(np.float32))
            r['classical_cosine'] = 0.75 * r['hog_cosine'] + 0.25 * r['edge_cosine']
            completed.append(r)
        except Exception as e:
            errors.append({'rank': i, 'work_id': r.get('work_id'), 'error': str(e)})
        if i % 25 == 0:
            print(json.dumps({'event':'progress','seen':i,'ok':len(completed),'errors':len(errors)}), flush=True)

    bycos = sorted(completed, key=lambda x: x['classical_cosine'], reverse=True)
    bysobel = sorted(completed, key=lambda x: x['score'])
    sobel_ranks = {r.get('work_id'): i+1 for i, r in enumerate(bysobel)}
    cosine_ranks = {r.get('work_id'): i+1 for i, r in enumerate(bycos)}
    if len(completed) > 1:
        a = np.array([sobel_ranks[r['work_id']] for r in completed], dtype=np.float64)
        b = np.array([cosine_ranks[r['work_id']] for r in completed], dtype=np.float64)
        rank_corr = float(np.corrcoef(a, b)[0, 1])
    else:
        rank_corr = None

    out = {
        'version': 'cosine-calibration-v0.3-classical',
        'source_pilot_run': PILOT_RUN,
        'source_pages': 3000,
        'attempted': len(rows),
        'ok': len(completed),
        'error_count': len(errors),
        'errors': errors[:40],
        'sobel_vs_cosine_rank_pearson': rank_corr,
        'by_sobel': bysobel[:50],
        'by_classical_cosine': bycos[:50]
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    md = Path(args.out).with_suffix('.md')
    with md.open('w') as f:
        f.write('# Classical cosine calibration on completed Sobel pilot\n\n')
        f.write(f"Processed {len(completed)}/{len(rows)} candidates; errors: {len(errors)}.\n\n")
        f.write(f"Sobel-rank vs classical-cosine-rank correlation: {rank_corr if rank_corr is not None else 'n/a'}.\n\n")
        f.write('| rank | classical cosine | HOG | edge | Sobel | manuscript | folio |\n|---:|---:|---:|---:|---:|---|---|\n')
        for i, r in enumerate(bycos[:25], 1):
            f.write(f"| {i} | {r['classical_cosine']:.4f} | {r['hog_cosine']:.4f} | {r['edge_cosine']:.4f} | {r['score']:.4f} | `{r['manuscript_id']}` | {r.get('folio_label') or r.get('canvas_index')} |\n")
    print(json.dumps({'event':'done','ok':len(completed),'errors':len(errors),'rank_corr':rank_corr}), flush=True)
    if not completed:
        raise RuntimeError('cosine calibration produced zero valid candidates')


if __name__ == '__main__':
    main()
