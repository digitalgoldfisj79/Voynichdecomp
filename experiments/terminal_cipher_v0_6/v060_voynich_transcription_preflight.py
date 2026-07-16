#!/usr/bin/env python3
from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
data = json.loads((ROOT / "voynich_transcriptions_slim.json").read_text(encoding="utf-8"))

summary = {
    "top_keys": sorted(data.keys()),
    "sources": data.get("sources", []),
    "transcribers": data.get("transcribers", []),
    "folios": len(data.get("pages", {})),
    "streams": {},
}

ids = [t["id"] for t in data.get("transcribers", [])]
for tid in ids:
    lines = []
    tokens = []
    chars = collections.Counter()
    folios = set()
    samples = []
    for fid, page in sorted(data.get("pages", {}).items()):
        for lnum, row in sorted(page.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 999999):
            text = row.get("t", {}).get(tid, "")
            if not text:
                continue
            folios.add(fid)
            lines.append((fid, str(lnum), text))
            toks = text.split()
            tokens.extend(toks)
            chars.update("".join(toks))
            if len(samples) < 3:
                samples.append({"folio": fid, "line": str(lnum), "text": text})
    if lines:
        summary["streams"][tid] = {
            "folios": len(folios),
            "lines": len(lines),
            "tokens": len(tokens),
            "types": len(set(tokens)),
            "character_types": len(chars),
            "characters": "".join(sorted(chars)),
            "top_characters": chars.most_common(20),
            "samples": samples,
        }

print("V060_VOYNICH_TRANSCRIPTION_PREFLIGHT", json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
