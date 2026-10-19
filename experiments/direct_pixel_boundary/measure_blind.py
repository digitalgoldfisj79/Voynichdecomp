#!/usr/bin/env python3
"""Label-blind direct ink-gap measurement at 700px and 2500px.

The 700px arm is algorithmically identical to Rozanova & Temerev's
measure_direct_pixels.py (blob SHA 28e8fd46...), with page caching only.
The high-resolution arm scales pixel-distance and component-area constants
by image-width ratio before labels are revealed.
"""
from __future__ import annotations
import argparse, hashlib, json, math, time, urllib.request
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from PIL import Image

UA = "Voynich-boundary-scale-replication/0.1 (+https://github.com/digitalgoldfisj79/Voynichdecomp)"


def get(url: str, retries=4, timeout=90):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def local_ink_mask(arr, threshold_offset=0, area_min=2):
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    base = min(170, max(105, int(otsu)))
    thr = max(80, min(200, base + threshold_offset))
    mask = ((gray < thr) & (hsv[:, :, 1] < 150)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= area_min:
            out[lab == i] = 1
    return out, thr


def scaled_row(r, sx, sy):
    d = r._asdict().copy()
    for k in ("lx0", "lx1", "rx0", "rx1"):
        d[k] = float(d[k]) * sx
    for k in ("ly0", "ly1", "ry0", "ry1"):
        d[k] = float(d[k]) * sy
    return SimpleNamespace(**d)


def measure_array(r, arr_full, threshold_offset=0, geom_scale=1.0):
    H, W = arr_full.shape[:2]
    pad_x = max(1, int(round(5 * geom_scale)))
    pad_y = max(1, int(round(3 * geom_scale)))
    search = max(1, int(round(10 * geom_scale)))
    far = max(1, int(round(15 * geom_scale)))
    area = max(2, int(round(2 * geom_scale * geom_scale)))
    x0 = max(0, int(math.floor(min(r.lx0, r.rx0) - pad_x)))
    x1 = min(W, int(math.ceil(max(r.lx1, r.rx1) + pad_x)))
    y0 = max(0, int(math.floor(min(r.ly0, r.ry0) - pad_y)))
    y1 = min(H, int(math.ceil(max(r.ly1, r.ry1) + pad_y)))
    if x1 <= x0 or y1 <= y0:
        return dict(gap_px=np.nan, threshold=np.nan, qc="bad_locator")
    arr = arr_full[y0:y1, x0:x1, :]
    mask, thr = local_ink_mask(arr, threshold_offset, area)
    hh = mask.shape[0]
    yy0, yy1 = int(round(.08 * hh)), int(round(.92 * hh))
    if yy1 <= yy0:
        return dict(gap_px=np.nan, threshold=thr, qc="bad_locator")
    col = mask[yy0:yy1, :].sum(axis=0)
    mid_global = (r.lx1 + r.rx0) / 2
    mid = mid_global - x0
    search_lo = max(1, int(math.floor(mid - search)))
    search_hi = min(mask.shape[1] - 2, int(math.ceil(mid + search)))
    tol = max(0, int(round(.02 * (yy1 - yy0))))
    ink = col > tol
    if search_lo > search_hi:
        return dict(gap_px=np.nan, threshold=thr, qc="bad_locator", locator_mid=mid_global)
    candidates = []
    for c in range(search_lo, search_hi + 1):
        lo, hi = max(0, c - 1), min(len(col), c + 2)
        candidates.append((col[lo:hi].sum(), abs(c - mid), c))
    split = min(candidates)[2]
    li = np.where(ink[:split + 1])[0]
    ri = np.where(ink[split + 1:])[0]
    if not len(li) or not len(ri):
        return dict(gap_px=np.nan, threshold=thr, qc="fail_no_ink", locator_mid=mid_global)
    le = int(li.max())
    re = int(split + 1 + ri.min())
    gap = max(0, re - le - 1)
    qc = "ok" if split - le <= far and re - split <= far else "review_far_edge"
    return dict(
        gap_px=float(gap), threshold=int(thr), qc=qc, locator_mid=float(mid_global),
        left_edge_global=float(x0 + le), right_edge_global=float(x0 + re), split_global=float(x0 + split),
    )


def service_cache_name(service):
    return hashlib.sha1(service.encode()).hexdigest()[:16]


def load_scan(service, width, cache):
    key = service_cache_name(service)
    p = cache / f"{key}_{width}.jpg"
    if not p.exists():
        p.write_bytes(get(f"{service.rstrip('/')}/full/{width},/0/default.jpg"))
    return np.array(Image.open(p).convert("RGB"))


def run_arm(df, width, cache, geom_scaled, offsets):
    rows = []
    for fol, g in df.groupby("folio", sort=False):
        service = str(g.iloc[0].iiif_service)
        arr = load_scan(service, width, cache)
        ah, aw = arr.shape[:2]
        rw = float(g.iloc[0].reg_width)
        rh = float(g.iloc[0].reg_height)
        sx, sy = aw / rw, ah / rh
        geom_scale = (aw / 700.0) if geom_scaled else 1.0
        for r0 in g.itertuples(index=False):
            r = scaled_row(r0, sx, sy)
            for off in offsets:
                m = measure_array(r, arr, threshold_offset=off, geom_scale=geom_scale)
                gap = m.get("gap_px", np.nan)
                m.update(
                    blind_id=r.blind_id, folio=r.folio, line=r.line,
                    width_px=aw, height_px=ah, threshold_offset=off, geom_scale=geom_scale,
                    gap_700eq=(gap * 700.0 / aw if np.isfinite(gap) else np.nan),
                )
                rows.append(m)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--cache", default="scan_cache")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest)

    forbidden = {"label", "certain", "uncertain", "left_token", "right_token", "group"}
    bad = forbidden.intersection({c.lower() for c in df.columns})
    if bad:
        raise SystemExit(f"BLINDING FAILURE: measurement manifest contains {sorted(bad)}")

    d700 = run_arm(df, 700, cache, geom_scaled=False, offsets=list(range(-30, 31, 5)))
    p700 = out / f"measure_700_s{args.shard}.csv"
    d700.to_csv(p700, index=False)

    d2500 = run_arm(df, 2500, cache, geom_scaled=True, offsets=[0])
    p2500 = out / f"measure_2500_s{args.shard}.csv"
    d2500.to_csv(p2500, index=False)

    freeze = {
        "schema": "direct-pixel-measurement-freeze-v0.1",
        "shard": args.shard,
        "input_manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "rows_input": int(len(df)),
        "rows_700": int(len(d700)),
        "rows_2500": int(len(d2500)),
        "primary_700": {
            "algorithm": "Rozanova-Temerev measure_direct_pixels.py blob 28e8fd46, page-cached equivalent",
            "offset": 0,
            "geom_scale": 1.0,
        },
        "highres": {
            "width": 2500,
            "distance_scale": "actual_width/700",
            "component_area_scale": "(actual_width/700)^2",
        },
        "threshold_offsets": list(range(-30, 31, 5)),
        "measure_700_sha256": hashlib.sha256(p700.read_bytes()).hexdigest(),
        "measure_2500_sha256": hashlib.sha256(p2500.read_bytes()).hexdigest(),
    }
    (out / f"measurement_freeze_s{args.shard}.json").write_text(json.dumps(freeze, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
