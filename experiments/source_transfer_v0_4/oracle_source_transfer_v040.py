#!/usr/bin/env python3
"""Hash-verified launcher for oracle source-transfer v0.4.0."""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "SOURCE_BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
encoded_parts: list[str] = []
for row in MANIFEST["parts"]:
    path = HERE / row["path"]
    value = path.read_text(encoding="ascii").strip()
    actual = hashlib.sha256(value.encode("ascii")).hexdigest()
    if actual != row["sha256"]:
        raise RuntimeError(f"source part hash mismatch: {path.name}: {actual}")
    encoded_parts.append(value)
encoded = "".join(encoded_parts)
if len(encoded) != MANIFEST["encoded_chars"]:
    raise RuntimeError("encoded source length mismatch")
if hashlib.sha256(encoded.encode("ascii")).hexdigest() != MANIFEST["gzip_base64_sha256"]:
    raise RuntimeError("encoded source hash mismatch")
raw = gzip.decompress(base64.b64decode(encoded))
if len(raw) != MANIFEST["raw_bytes"]:
    raise RuntimeError("source byte length mismatch")
if hashlib.sha256(raw).hexdigest() != MANIFEST["raw_sha256"]:
    raise RuntimeError("source hash mismatch")
target = HERE / MANIFEST["source_file"]
target.write_bytes(raw)
sys.argv[0] = str(target)
runpy.run_path(str(target), run_name="__main__")
