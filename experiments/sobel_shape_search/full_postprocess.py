#!/usr/bin/env python3
"""Full-r10 Sobel post-processing using the frozen quarter-checkpoint reranker.

Requires all 32x7 = 224 scan cells from the authoritative full-r10 workflow run.
It then takes the global raw top 500, applies the unchanged classical reranker
(0.2 raw rank + 0.4 symmetric chamfer rank + 0.4 HOG/edge cosine rank), and
writes the global top 100 plus a top-20 visual contact sheet and individual crops.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw

W, H = 192, 144
REPO = "digitalgoldfisj79/Voynichdecomp"
ART_RE = re.compile(r"^sobel-full-s(\d+)-c(\d+)$")
EXPECTED = {(s, c) for s in range(32) for c in range(7)}
UA = "ManuComp-full-r10-rerank/0.1"


def gh_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def get(session, url, headers=None, timeout=30, tries=5):
    last = None
    for i in range(tries):
        try:
            r = session.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            time.sleep(1.5 * (2**i))
    raise last


def list_artifacts(session, run_id: int, headers: dict[str, str]):
    arts = []
    page = 1
    while True:
        r = get(
            session,
            f"https://api.github.com/repos/{REPO}/actions/runs/{run_id}/artifacts?per_page=100&page={page}",
            headers=headers,
            timeout=60,
        )
        batch = r.json().get("artifacts", [])
        arts.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return arts


def choose_latest_cells(arts):
    latest = {}
    for a in arts:
        m = ART_RE.match(a.get("name", ""))
        if not m or a.get("expired"):
            continue
        cell = tuple(map(int, m.groups()))
        old = latest.get(cell)
        if old is None or (a.get("created_at", ""), int(a.get("id", 0))) > (
            old.get("created_at", ""),
            int(old.get("id", 0)),
        ):
            latest[cell] = a
    return latest


def load_cells(session, cells, headers):
    rows, sums, failures = [], [], []
    for n, (cell, a) in enumerate(sorted(cells.items()), 1):
        try:
            z = get(session, a["archive_download_url"], headers=headers, timeout=90)
            with zipfile.ZipFile(io.BytesIO(z.content)) as zz:
                names = [x for x in zz.namelist() if x.endswith(".json")]
                if not names:
                    raise RuntimeError("artifact contains no JSON")
                d = json.loads(zz.read(names[0]))
            sm = dict(d.get("summary", {}))
            sm["cell_shard"], sm["cell_chunk"] = cell
            sm["artifact_id"] = a["id"]
            sums.append(sm)
            rows.extend(d.get("results", []))
        except Exception as e:
            failures.append({"cell": cell, "artifact_id": a.get("id"), "error": repr(e)})
        if n % 25 == 0:
            print(json.dumps({"stage": "load", "cells": n, "rows": len(rows), "failures": len(failures)}), flush=True)
    return rows, sums, failures


def norm_query(q0, angle):
    m = (q0 > 127).astype(np.uint8) * 255
    pad = max(m.shape) // 3 + 4
    m = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    ctr = (m.shape[1] / 2, m.shape[0] / 2)
    M = cv2.getRotationMatrix2D(ctr, float(angle or 0), 1.0)
    m = cv2.warpAffine(m, M, (m.shape[1], m.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)
    yy, xx = np.where(m > 0)
    if len(xx):
        m = m[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]
    return cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)


def edge(gray, binary=False):
    g = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
    if not binary:
        g = cv2.GaussianBlur(g, (3, 3), 0)
    return cv2.Canny(g, 20 if binary else 45, 80 if binary else 140)


def hog(gray):
    g = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = cv2.phase(gx, gy, angleInDegrees=True) % 180.0
    feats = []
    cy, cx = H // 4, W // 4
    for iy in range(4):
        for ix in range(4):
            mm = mag[iy * cy : (iy + 1) * cy, ix * cx : (ix + 1) * cx].ravel()
            aa = ang[iy * cy : (iy + 1) * cy, ix * cx : (ix + 1) * cx].ravel()
            hist = np.zeros(9, np.float32)
            bins = np.minimum((aa / 20).astype(int), 8)
            for b, v in zip(bins, mm):
                hist[b] += v
            feats.extend((hist / (np.linalg.norm(hist) + 1e-6)).tolist())
    v = np.asarray(feats, np.float32)
    return v / (np.linalg.norm(v) + 1e-6)


def cosine(a, b):
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))


def sym_chamfer(qe, ce):
    q = (qe > 0).astype(np.uint8)
    c = (ce > 0).astype(np.uint8)
    if q.sum() == 0 or c.sum() == 0:
        return 99.0
    dc = cv2.distanceTransform((1 - c).astype(np.uint8), cv2.DIST_L2, 3)
    dq = cv2.distanceTransform((1 - q).astype(np.uint8), cv2.DIST_L2, 3)
    return float((dc[q > 0].mean() + dq[c > 0].mean()) / 2.0 / max(W, H))


def fetch_crop_array(session, r):
    for u, is_region in ((r.get("region_url"), True), (r.get("search_url"), False)):
        if not u:
            continue
        try:
            rr = get(session, u, headers={"User-Agent": UA}, timeout=30, tries=4)
            im = Image.open(io.BytesIO(rr.content)).convert("L")
            a = np.array(im)
            if not is_region:
                pw = float(r.get("page_w") or a.shape[1])
                sc = a.shape[1] / pw if pw else 1.0
                x = int(float(r.get("x", 0)) * sc)
                y = int(float(r.get("y", 0)) * sc)
                ww = max(1, int(float(r.get("w", a.shape[1])) * sc))
                hh = max(1, int(float(r.get("h", a.shape[0])) * sc))
                a = a[max(0, y) : min(a.shape[0], y + hh), max(0, x) : min(a.shape[1], x + ww)]
            if a.size:
                return a
        except Exception:
            pass
    raise RuntimeError("candidate crop unavailable")


def candidate_uid(r, i):
    return "|".join(
        str(x)
        for x in (
            r.get("manuscript_id"),
            r.get("work_id"),
            r.get("folio_label"),
            r.get("canvas_index"),
            r.get("x"),
            r.get("y"),
            r.get("w"),
            r.get("h"),
            r.get("rotation_deg"),
            i,
        )
    )


def rerank(session, candidates, q0):
    def one(i, r):
        a = fetch_crop_array(session, r)
        qm = norm_query(q0, r.get("rotation_deg", 0))
        qe = edge(qm, True)
        ce = edge(a, False)
        qh = hog(255 - qm)
        ch = hog(a)
        hcos = cosine(qh, ch)
        ecos = cosine((qe > 0).astype(np.float32).ravel(), (ce > 0).astype(np.float32).ravel())
        sch = sym_chamfer(qe, ce)
        dens = float((ce > 0).sum() / max((qe > 0).sum(), 1))
        z = dict(r)
        z.update(
            _uid=candidate_uid(r, i),
            raw_rank=i + 1,
            hog_cosine=hcos,
            edge_cosine=ecos,
            classical_cosine=0.8 * hcos + 0.2 * ecos,
            symmetric_chamfer=sch,
            edge_density_ratio=dens,
        )
        return z

    done, failures = [], []
    with ThreadPoolExecutor(max_workers=12) as ex:
        fs = {ex.submit(one, i, r): (i, r) for i, r in enumerate(candidates)}
        for n, f in enumerate(as_completed(fs), 1):
            i, r = fs[f]
            try:
                done.append(f.result())
            except Exception as e:
                failures.append({"raw_rank": i + 1, "work_id": r.get("work_id"), "error": repr(e)})
            if n % 50 == 0:
                print(json.dumps({"stage": "rerank", "processed": n, "ok": len(done), "errors": len(failures)}), flush=True)

    N = max(len(done), 1)
    byraw = sorted(done, key=lambda r: r["raw_rank"])
    bych = sorted(done, key=lambda r: r["symmetric_chamfer"])
    bycos = sorted(done, key=lambda r: r["classical_cosine"], reverse=True)
    rr = {r["_uid"]: i + 1 for i, r in enumerate(byraw)}
    cr = {r["_uid"]: i + 1 for i, r in enumerate(bych)}
    kr = {r["_uid"]: i + 1 for i, r in enumerate(bycos)}
    for r in done:
        r["symmetric_chamfer_rank"] = cr[r["_uid"]]
        r["classical_cosine_rank"] = kr[r["_uid"]]
        r["fusion_score"] = 0.2 * rr[r["_uid"]] / N + 0.4 * cr[r["_uid"]] / N + 0.4 * kr[r["_uid"]] / N
        r.pop("_uid", None)
    return sorted(done, key=lambda r: r["fusion_score"]), failures


def fetch_display(session, r):
    errs = []
    u = r.get("region_url")
    if u:
        try:
            rr = get(session, u, headers={"User-Agent": UA}, timeout=30, tries=4)
            return Image.open(io.BytesIO(rr.content)).convert("RGB"), "region_url"
        except Exception as e:
            errs.append(repr(e))
    u = r.get("search_url")
    if u:
        try:
            rr = get(session, u, headers={"User-Agent": UA}, timeout=30, tries=4)
            im = Image.open(io.BytesIO(rr.content)).convert("RGB")
            pw = float(r.get("page_w") or im.width)
            sc = im.width / pw if pw else 1.0
            x = int(float(r.get("x", 0)) * sc)
            y = int(float(r.get("y", 0)) * sc)
            ww = max(1, int(float(r.get("w", im.width)) * sc))
            hh = max(1, int(float(r.get("h", im.height)) * sc))
            pad = max(12, int(0.25 * max(ww, hh)))
            return im.crop((max(0, x - pad), max(0, y - pad), min(im.width, x + ww + pad), min(im.height, y + hh + pad))), "search_url_crop"
        except Exception as e:
            errs.append(repr(e))
    raise RuntimeError("; ".join(errs))


def write_outputs(session, outdir: Path, fused, meta):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "crops").mkdir(exist_ok=True)
    payload = dict(meta)
    payload["results"] = fused
    (outdir / "sobel_full_reranked.json").write_text(json.dumps(payload, indent=2))

    with (outdir / "sobel_full_reranked_top100.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "fusion_score", "raw_rank", "raw_score", "symmetric_chamfer", "classical_cosine", "hog_cosine", "edge_cosine", "edge_density_ratio", "manuscript_id", "folio_label", "canvas_index", "work_id", "region_url", "search_url"])
        for i, r in enumerate(fused[:100], 1):
            w.writerow([i, r.get("fusion_score"), r.get("raw_rank"), r.get("score"), r.get("symmetric_chamfer"), r.get("classical_cosine"), r.get("hog_cosine"), r.get("edge_cosine"), r.get("edge_density_ratio"), r.get("manuscript_id"), r.get("folio_label"), r.get("canvas_index"), r.get("work_id"), r.get("region_url"), r.get("search_url")])

    records = []
    for i, r in enumerate(fused[:20], 1):
        rec = {k: r.get(k) for k in ["manuscript_id", "folio_label", "canvas_index", "fusion_score", "raw_rank", "symmetric_chamfer", "classical_cosine", "edge_density_ratio", "work_id"]}
        try:
            im, source = fetch_display(session, r)
            fn = outdir / "crops" / f"rank_{i:02d}.jpg"
            im.save(fn, quality=94)
            rec.update(rank=i, status="ok", source=source, file=str(fn), width=im.width, height=im.height)
        except Exception as e:
            rec.update(rank=i, status="error", error=repr(e))
        records.append(rec)

    tw, th, cols = 360, 320, 4
    sheet = Image.new("RGB", (cols * tw, 5 * th), "white")
    dr = ImageDraw.Draw(sheet)
    for rec in records:
        i = rec["rank"] - 1
        ox, oy = (i % cols) * tw, (i // cols) * th
        if rec["status"] == "ok":
            im = Image.open(rec["file"]).convert("RGB")
            im.thumbnail((tw - 20, th - 100), Image.Resampling.LANCZOS)
            sheet.paste(im, (ox + (tw - im.width) // 2, oy + 5))
        else:
            dr.rectangle((ox + 10, oy + 10, ox + tw - 10, oy + th - 105), outline="gray")
            dr.text((ox + 20, oy + 50), "image unavailable", fill="black")
        lab = f"#{rec['rank']} fused {float(rec.get('fusion_score') or 0):.3f}\n{rec.get('manuscript_id','')}\n{rec.get('folio_label') or rec.get('canvas_index','')}\ncos {float(rec.get('classical_cosine') or 0):.3f} ch {float(rec.get('symmetric_chamfer') or 0):.3f}"
        dr.text((ox + 8, oy + th - 92), lab, fill="black")
    sheet.save(outdir / "sobel_full_top20.jpg", quality=94)
    (outdir / "sobel_full_top20.json").write_text(json.dumps(records, indent=2))

    ok = sum(r["status"] == "ok" for r in records)
    with (outdir / "SOBEL_FULL_POSTPROCESS_REPORT.md").open("w") as f:
        f.write("# Full r10 Sobel postprocess\n\n")
        f.write(f"Source run: `{meta['source_run']}`. Cells loaded: **{meta['cells_loaded']}/224**. Pages seen: **{meta['pages_seen']:,}**; successful: **{meta['pages_ok']:,}**; errors: **{meta['page_errors']:,}**.\n\n")
        f.write(f"Frozen classical rerank attempted **{meta['rerank_attempted']}** global raw candidates; successful **{meta['rerank_ok']}**; rerank errors **{meta['rerank_errors']}**. Visual crops saved **{ok}/20**.\n\n")
        f.write("| rank | fusion | raw rank | manuscript | folio | chamfer | cosine |\n|---:|---:|---:|---|---|---:|---:|\n")
        for i, r in enumerate(fused[:20], 1):
            f.write(f"| {i} | {r['fusion_score']:.4f} | {r['raw_rank']} | `{r.get('manuscript_id','')}` | {r.get('folio_label') or r.get('canvas_index','')} | {r['symmetric_chamfer']:.4f} | {r['classical_cosine']:.4f} |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=int, required=True)
    ap.add_argument("--outdir", default="full_postprocess")
    ap.add_argument("--require-complete", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN required")
    session = requests.Session()
    headers = gh_headers(token)

    arts = list_artifacts(session, args.run_id, headers)
    cells = choose_latest_cells(arts)
    missing = sorted(EXPECTED - set(cells))
    print(json.dumps({"source_run": args.run_id, "unique_cells": len(cells), "missing": missing}), flush=True)
    if args.require_complete and missing:
        raise SystemExit(f"FULL_CORPUS_GATE_FAIL: missing {len(missing)} cells: {missing}")

    rows, sums, load_failures = load_cells(session, cells, headers)
    if load_failures:
        raise SystemExit(f"FULL_CORPUS_GATE_FAIL: {len(load_failures)} artifacts unreadable: {load_failures[:5]}")
    if args.require_complete and len(sums) != 224:
        raise SystemExit(f"FULL_CORPUS_GATE_FAIL: loaded {len(sums)}/224 cell summaries")

    rows.sort(key=lambda r: r.get("score", 1e99))
    candidates = rows[:500]
    qraw = base64.b64decode(Path("experiments/sobel_shape_search/query_mask.b64").read_text().strip())
    q0 = np.array(Image.open(io.BytesIO(qraw)).convert("L"))
    ys, xs = np.where(q0 > 127)
    q0 = q0[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    fused, rerank_failures = rerank(session, candidates, q0)
    meta = {
        "version": "sobel-full-r10-postprocess-v0.1",
        "source_run": args.run_id,
        "cells_loaded": len(sums),
        "missing_cells": missing,
        "pages_seen": sum(int(x.get("seen", 0) or 0) for x in sums),
        "pages_ok": sum(int(x.get("ok", 0) or 0) for x in sums),
        "page_errors": sum(int(x.get("errors", 0) or 0) for x in sums),
        "raw_candidate_rows_merged": len(rows),
        "rerank_attempted": len(candidates),
        "rerank_ok": len(fused),
        "rerank_errors": len(rerank_failures),
        "rerank_failure_details": rerank_failures,
        "method": {
            "prefilter": "global raw Sobel/chamfer top 500 across 224 r10 cells",
            "rerank": "0.2 raw-rank + 0.4 symmetric-chamfer-rank + 0.4 HOG/edge-cosine-rank",
            "models": "none",
            "status": "frozen from quarter checkpoint; input scope only changed",
        },
    }
    write_outputs(session, Path(args.outdir), fused, meta)
    print(json.dumps({k: meta[k] for k in ["cells_loaded", "pages_seen", "pages_ok", "page_errors", "rerank_attempted", "rerank_ok", "rerank_errors"]}), flush=True)


if __name__ == "__main__":
    main()
