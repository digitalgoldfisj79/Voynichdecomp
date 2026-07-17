#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
from pathlib import Path

BUNDLES = {
    "external_calibration.py": {
        "pattern": "external_calibration.py.v4.gz.b64.part*",
        "size": 37401,
        "sha256": "f93fc90c0527266d71d876962050923b9f7e4020c77dc8c7fad83019b80ac883",
    },
    "blind_model_selection.py": {
        "pattern": "blind_model_selection.py.v2.gz.b64.part*",
        "size": 21739,
        "sha256": "c1597213530eb54cf3cd1093ab209dcecdce6fd8abf06362039f92480c523b9a",
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()
    root = Path(args.root)
    for target_name, spec in BUNDLES.items():
        parts = sorted((root / "source_bundle").glob(spec["pattern"]))
        if not parts:
            raise SystemExit(f"source parts not found: {spec['pattern']}")
        encoded = "".join(p.read_text(encoding="ascii").strip() for p in parts)
        raw = gzip.decompress(base64.b64decode(encoded, validate=True))
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) != spec["size"] or digest != spec["sha256"]:
            raise SystemExit(
                f"source integrity failure for {target_name}: size={len(raw)} sha256={digest}"
            )
        target = root / "code" / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        target.chmod(0o755)
        print(f"wrote {target} ({len(raw)} bytes; sha256={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
