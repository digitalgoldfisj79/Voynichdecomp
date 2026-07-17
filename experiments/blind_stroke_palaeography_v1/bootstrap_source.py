#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import gzip
from pathlib import Path

BUNDLES = {
    "external_calibration.py": "external_calibration.py.gz.b64.part*",
    "blind_model_selection.py": "blind_model_selection.py.gz.b64.part*",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()
    root = Path(args.root)
    for target_name, pattern in BUNDLES.items():
        parts = sorted((root / "source_bundle").glob(pattern))
        if not parts:
            raise SystemExit(f"source parts not found: {pattern}")
        encoded = "".join(p.read_text(encoding="ascii").strip() for p in parts)
        raw = gzip.decompress(base64.b64decode(encoded))
        target = root / "code" / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        target.chmod(0o755)
        print(f"wrote {target} ({len(raw)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
