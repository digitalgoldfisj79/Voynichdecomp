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


def parse_tei(content: bytes) -> tuple[bool, list[str]]:
    head = content.lstrip()[:300].lower()
    if not (head.startswith(b"<?xml") or head.startswith(b"<tei") or b"<tei" in head):
        return False, ["payload does not begin with TEI/XML"]
    try:
        parser = etree.XMLParser(recover=True, huge_tree=True)
        root = etree.fromstring(content, parser)
        if root is None:
            return False, ["lxml recovery returned no root"]
        local = etree.QName(root).localname.lower()
        if local not in {"tei", "teicorpus"}:
            return False, [f"unexpected root element {local}"]
        return True, [str(item) for item in parser.error_log]
    except Exception as exc:
        return False, [f"{type(exc).__name__}:{str(exc)[:300]}"]


def fetch_one(mid: str, out: Path) -> dict:
    dest = out / f"{mid}.recipes.xml"
    if dest.exists() and dest.stat().st_size > 100:
        valid, issues = parse_tei(dest.read_bytes())
        if valid:
            row = {"id": mid, "status": "cached", "bytes": dest.stat().st_size}
            if issues:
                row["xml_recovery_issues"] = issues
            return row
        dest.unlink()
    urls = [
        f"https://gams.uni-graz.at/o%3Acorema.{mid}.recipes/TEI_SOURCE",
        f"https://gams.uni-graz.at/o:corema.{mid}.recipes/TEI_SOURCE",
    ]
    errors = []
    headers = {"User-Agent": "VoynichRecoverabilityResearch/0.6"}
    for url in urls:
        try:
            r = requests.get(url, timeout=(10, 45), allow_redirects=True, headers=headers)
            valid, issues = parse_tei(r.content) if r.status_code == 200 else (False, [])
            if r.status_code == 200 and valid:
                tmp = dest.with_suffix(".tmp")
                tmp.write_bytes(r.content)
                tmp.replace(dest)
                row = {"id": mid, "status": "downloaded", "bytes": len(r.content), "url": url}
                if issues:
                    row["xml_recovery_issues"] = issues
                return row
            errors.append(f"{r.status_code}:{r.headers.get('content-type')}:{len(r.content)}" + (f":{' | '.join(issues)}" if issues else ""))
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
        "recovered_with_xml_issues": sum(bool(x.get("xml_recovery_issues")) for x in results),
        "results": results,
    }
    (args.out / "prefetch_audit_v0_6.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("attempted", "downloaded_or_cached", "failed", "recovered_with_xml_issues")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
