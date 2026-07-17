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
PARENT_BYTES = 23391
PARENT_SHA256 = "e064648d07e28eac56a2f46012012d5e472aacc4e44dfa81c7018235b220b934"
DERIVED_BYTES = 23391
DERIVED_SHA256 = "fd8f93893a488b59d41eba4395de82e5690ebb491bc8bbe6c1de581a2884cdd8"
OLD = "MAX_WRITERS = 48 if PREFLIGHT else None"
NEW = "MAX_WRITERS = 80 if PREFLIGHT else None"


def main() -> int:
    parent = b"".join(
        urllib.request.urlopen(ROOT + f"part{i:02d}.pyfrag", timeout=120).read()
        for i in range(7)
    )
    parent_sha = hashlib.sha256(parent).hexdigest()
    if len(parent) != PARENT_BYTES or parent_sha != PARENT_SHA256:
        raise RuntimeError(
            f"parent v1.5 mismatch: bytes={len(parent)}, sha256={parent_sha}"
        )
    source = parent.decode("utf-8")
    if source.count(OLD) != 1:
        raise RuntimeError("expected exactly one v1.5.1 substitution target")
    derived = source.replace(OLD, NEW, 1).encode("utf-8")
    derived_sha = hashlib.sha256(derived).hexdigest()
    if len(derived) != DERIVED_BYTES or derived_sha != DERIVED_SHA256:
        raise RuntimeError(
            f"derived v1.5.1 mismatch: bytes={len(derived)}, sha256={derived_sha}"
        )
    destination = pathlib.Path("/tmp/saghog_v1_5_1_full.py")
    destination.write_bytes(derived)
    print(
        f"V15_1_ASSEMBLED bytes={len(derived)} sha256={derived_sha} parent_sha256={parent_sha}",
        flush=True,
    )
    runpy.run_path(str(destination), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
