#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

import requests
from lxml import etree

MANUSCRIPT_IDS = [
    "a1","b1","b2","b3","b4","b5","b6","br1","bs1","bs2","db1","ds1",
    "er1","er2","gr1","h1","h2","h3","h4","ha1","hi1","k1","ka1","ka2",
    "ka3","ko1","m1","m2","m3","m4","m5","m6","m7","m8","m9","m10",
    "m12","m13","m11","mi1","n1","n2","pa1","pr1","sb1","sb2","sb3",
    "so1","st1","ste1","w1","w2","w3","w4","wo1","wo2","wo3","wo4",
    "wo5","wo7","wo8","wo9","wo10","wo11","wol1","zu1",
]


def is_xml(content: bytes) -> bool:
    head = content.lstrip()[:300].lower()
    if not (head.startswith(b"<?xml") or head.startswith(b"<tei") or b"<tei" in head):
        return False
    try:
        etree.fromstring(content)
        return True
    except Exception:
        return False


def fetch_one(mid: str, out: Path) -> dict:
    dest = out / f"{mid}.recipes.xml"
    if dest.exists() and dest.stat().st_size > 100:
        return {"id": mid, "status": "cached", "bytes": dest.stat().st_size}
    urls = [
        f"https://gams.uni-graz.at/o%3Acorema.{mid}.recipes/TEI_SOURCE",
        f"https://gams.uni-graz.at/o:corema.{mid}.recipes/TEI_SOURCE",
    ]
    errors = []
    headers = {"User-Agent": "VoynichRecoverabilityResearch/0.6"}
    for url in urls:
        try:
            r = requests.get(url, timeout=(10, 45), allow_redirects=True, headers=headers)
            if r.status_code == 200 and is_xml(r.content):
                tmp = dest.with_suffix(".tmp")
                tmp.write_bytes(r.content)
                tmp.replace(dest)
                return {"id": mid, "status": "downloaded", "bytes": len(r.content), "url": url}
            errors.append(f"{r.status_code}:{r.headers.get('content-type')}:{len(r.content)}")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}:{str(exc)[:160]}")
    return {"id": mid, "status": "failed", "errors": errors}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(lambda mid: fetch_one(mid, args.out), MANUSCRIPT_IDS))
    audit = {
        "attempted": len(MANUSCRIPT_IDS),
        "downloaded_or_cached": sum(x["status"] != "failed" for x in results),
        "failed": sum(x["status"] == "failed" for x in results),
        "results": results,
    }
    (args.out / "prefetch_audit_v0_6.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("attempted", "downloaded_or_cached", "failed")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
