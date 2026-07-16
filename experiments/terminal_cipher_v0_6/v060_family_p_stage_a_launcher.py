#!/usr/bin/env python3
"""Launch Family P with an execution-only Numba integer-typing patch."""
from __future__ import annotations

import hashlib
from pathlib import Path

path = Path(__file__).with_name("v060_family_p_stage_a.py")
source = path.read_text(encoding="utf-8")
replacements = {
    "                first = second = -1\n": (
        "                first = np.int64(-1)\n"
        "                second = np.int64(-1)\n"
    ),
    "                old_shift = -1\n                slot = -1\n": (
        "                old_shift = np.int64(-1)\n"
        "                slot = np.int64(-1)\n"
    ),
}
patched = source
for old, new in replacements.items():
    if patched.count(old) != 1:
        raise RuntimeError(f"patch site mismatch: {old!r}")
    patched = patched.replace(old, new)
print(
    "V060_P_LAUNCHER_PATCH",
    {
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "patched_sha256": hashlib.sha256(patched.encode("utf-8")).hexdigest(),
        "scientific_search_changed": False,
        "reason": "Numba branch-index type stabilization",
    },
    flush=True,
)
exec(compile(patched, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
