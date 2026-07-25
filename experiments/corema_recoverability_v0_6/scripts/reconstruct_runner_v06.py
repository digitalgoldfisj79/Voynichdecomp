#!/usr/bin/env python3
from __future__ import annotations
import base64
import hashlib
import zlib
from pathlib import Path

here = Path(__file__).resolve().parent
payload = ''.join((here / f'runner_part{i}.b64z').read_text().strip() for i in (1, 2))
raw = zlib.decompress(base64.b64decode(payload))
out = here / 'run_corema_recoverability_v06.py'
out.write_bytes(raw)
sha = hashlib.sha256(raw).hexdigest()
expected = 'a9b99b6a9e1255883e63423bb743083805aeb62c242656692a787c63362018d8'
if sha != expected:
    raise SystemExit(f'SHA mismatch: {sha} != {expected}')
print(f'reconstructed {out} sha256={sha} bytes={len(raw)}')
