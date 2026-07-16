#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = (
    "p183_interlinear.for_mining.COMPAT.tsv",
    "p140_tokens.COMPAT.tsv",
    "daiin_vms.pkl",
)

rows = []
for name in NAMES:
    matches = sorted(ROOT.rglob(name))
    rows.append({"name": name, "matches": [str(p.relative_to(ROOT)) for p in matches]})

# Also locate likely canonical transcription assets without reading their content.
likely = []
for path in ROOT.rglob("*"):
    if not path.is_file():
        continue
    low = path.name.lower()
    if any(term in low for term in ("interlinear", "ivtff", "eva", "transcription", "voynich")):
        likely.append(str(path.relative_to(ROOT)))

print("V060_VOYNICH_ASSET_PREFLIGHT", json.dumps({
    "exact": rows,
    "likely": sorted(likely)[:200],
}, sort_keys=True), flush=True)
