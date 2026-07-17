#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import pathlib
import runpy
import urllib.request

PARTS_COMMIT = "7541f99629eb68c4e5663478b828054a07459039"
ROOT = (
    "https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/"
    + PARTS_COMMIT
    + "/experiments/blind_stroke_palaeography_v1/code/v1_5_parts/"
)
EXPECTED_BYTES = 23391
EXPECTED_SHA256 = "e064648d07e28eac56a2f46012012d5e472aacc4e44dfa81c7018235b220b934"


def main() -> int:
    chunks = []
    for i in range(7):
        url = ROOT + f"part{i:02d}.pyfrag"
        with urllib.request.urlopen(url, timeout=120) as response:
            chunks.append(response.read())
    raw = b"".join(chunks)
    actual = hashlib.sha256(raw).hexdigest()
    if len(raw) != EXPECTED_BYTES or actual != EXPECTED_SHA256:
        raise RuntimeError(
            f"assembled v1.5 source mismatch: bytes={len(raw)}, sha256={actual}"
        )
    destination = pathlib.Path("/tmp/saghog_v1_5_full.py")
    destination.write_bytes(raw)
    print(
        f"V15_ASSEMBLED bytes={len(raw)} sha256={actual} parts_commit={PARTS_COMMIT}",
        flush=True,
    )
    runpy.run_path(str(destination), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
