#!/usr/bin/env python3
"""Rewrite manifest paths relative to the manifest and verify hashes."""
from __future__ import annotations
import argparse,csv,hashlib
from pathlib import Path

def digest(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument('manifest');a=ap.parse_args()
    manifest=Path(a.manifest).resolve();base=manifest.parent;cwd=Path.cwd().resolve()
    with manifest.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames or [];rows=list(reader)
    if not rows:return 0
    for row in rows:
        original=Path(row['path']);candidates=[]
        if original.is_absolute():candidates.append(original)
        else:candidates.extend([(base/original).resolve(),(cwd/original).resolve()])
        actual=next((p for p in candidates if p.is_file()),None)
        if actual is None:raise FileNotFoundError(f"cannot resolve {row['path']} from {manifest}")
        row['path']=actual.relative_to(base).as_posix();expected=row.get('sha256','').lower()
        if expected and digest(actual)!=expected:raise ValueError(f"hash mismatch: {actual}")
    with manifest.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(f"RELATIVIZED {len(rows)} rows in {manifest}");return 0
if __name__=='__main__':raise SystemExit(main())
