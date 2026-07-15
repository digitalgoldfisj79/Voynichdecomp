#!/usr/bin/env python3
"""Launch the frozen flexible v0.5.2 solver with explicit Numba index casts."""
from __future__ import annotations

import hashlib
from pathlib import Path

source_path = Path(__file__).with_name("homophonic_solver_v052_flexible.py")
source = source_path.read_text(encoding="utf-8")
needle = "                old_label = int(key[first])\n                if new_label != old_label"
replacement = (
    "                old_label = int(key[first])\n"
    "                new_label = int(new_label)\n"
    "                if new_label != old_label"
)
occurrences = source.count(needle)
if occurrences != 3:
    raise RuntimeError(f"expected 3 reassignment sites, found {occurrences}")
patched = source.replace(needle, replacement)
print("V052F_PATCH", {
    "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    "patched_sha256": hashlib.sha256(patched.encode("utf-8")).hexdigest(),
    "sites": occurrences,
}, flush=True)
namespace = {
    "__name__": "__main__",
    "__file__": str(source_path),
}
exec(compile(patched, str(source_path), "exec"), namespace)
