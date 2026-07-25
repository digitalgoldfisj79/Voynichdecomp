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
expected = '213603fd38d99725cc99cb640bccb428eea0480de48298630dd3834426a49616'
if sha != expected:
    raise SystemExit(f'SHA mismatch: {sha} != {expected}')
print(f'reconstructed {out} sha256={sha} bytes={len(raw)}')
