#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "run_external_recoverability_v05.py"
EXPECTED = "3f15c8e7b344a0ca0069b8cb4f0717aa2fc26ae39376f04c0c8c8ec920ec8648"

chunks = []
for i in range(1, 5):
    encoded = (ROOT / f"external_runner_part{i}.b64z").read_text(encoding="utf-8").strip()
    chunks.append(zlib.decompress(base64.b64decode(encoded)))
raw = b"".join(chunks)
actual = hashlib.sha256(raw).hexdigest()
if actual != EXPECTED:
    raise SystemExit(f"runner SHA-256 mismatch: {actual} != {EXPECTED}")
OUT.write_bytes(raw)
print(f"wrote {OUT} ({len(raw)} bytes; sha256={actual})")
