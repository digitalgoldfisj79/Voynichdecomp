#!/usr/bin/env python3
"""Execution-only launcher correcting ColumnarTrial length references."""
from __future__ import annotations

import hashlib
from pathlib import Path

path = Path(__file__).with_name("v060_family_t_stage_a.py")
source = path.read_text(encoding="utf-8")
patched = source.replace("trial.length", "len(trial.plain)")
if patched == source:
    raise RuntimeError("length patch site not found")
print(
    "V060_TA_LAUNCHER_PATCH",
    {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "scientific_search_changed": False,
        "reason": "derive stored trial length from plaintext sequence",
    },
    flush=True,
)
exec(compile(patched, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
