#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FROZEN_PATHS = [
    "FROZEN_PROTOCOL.md",
    "AMENDMENT_001.md",
    "config/protocol_v1.json",
    "LITERATURE_AND_DATA_AUDIT.md",
    "bootstrap_source.py",
    "code/preflight.py",
    "code/freeze_manifest.py",
    "source_bundle/external_calibration.py.v4.gz.b64.part00",
    "source_bundle/external_calibration.py.v4.gz.b64.part01",
    "source_bundle/external_calibration.py.v4.gz.b64.part02",
    "source_bundle/external_calibration.py.v4.gz.b64.part03",
    "source_bundle/external_calibration.py.v4.gz.b64.part04",
    "source_bundle/blind_model_selection.py.v2.gz.b64.part00",
    "source_bundle/blind_model_selection.py.v2.gz.b64.part01",
    "source_bundle/blind_model_selection.py.v2.gz.b64.part02",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--output", default="FREEZE_RECORD_V1_1.json")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    rows = {}
    for rel in FROZEN_PATHS:
        p = root / rel
        if not p.is_file():
            raise SystemExit(f"missing frozen file: {p}")
        rows[rel] = {"size_bytes": p.stat().st_size, "sha256": sha256(p)}
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    record = {
        "schema": "blind-palaeography-freeze-v1.1",
        "parent_freeze_sha256": "78f57d1d1ea52c6a8a4f6de9438b094edc56b670ab22863767cfae659aaddeaa",
        "bounded_repairs_consumed": 1,
        "phase1_voynich_opened": False,
        "davis_labels_loaded": False,
        "frozen_files": rows,
        "aggregate_sha256": hashlib.sha256(canonical).hexdigest(),
    }
    out = root / args.output
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
