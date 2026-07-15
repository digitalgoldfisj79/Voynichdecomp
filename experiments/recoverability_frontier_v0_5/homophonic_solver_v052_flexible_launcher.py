#!/usr/bin/env python3
"""Launch the frozen flexible v0.5.2 solver with explicit Numba index casts."""
from __future__ import annotations

import hashlib
from pathlib import Path

source_path = Path(__file__).with_name("homophonic_solver_v052_flexible.py")
source = source_path.read_text(encoding="utf-8")

reassignment_needle = "                old_label = int(key[first])\n                if new_label != old_label"
reassignment_replacement = (
    "                first = int(first)\n"
    "                old_label = int(key[first])\n"
    "                new_label = int(new_label)\n"
    "                if new_label != old_label"
)
reassignment_sites = source.count(reassignment_needle)
if reassignment_sites != 3:
    raise RuntimeError(f"expected 3 reassignment sites, found {reassignment_sites}")
patched = source.replace(reassignment_needle, reassignment_replacement)

anneal_needle = (
    "            if not changed:\n"
    "                temperature *= cooling\n"
    "                continue\n\n"
    "            candidate_score"
)
anneal_replacement = (
    "            if not changed:\n"
    "                temperature *= cooling\n"
    "                continue\n\n"
    "            first = int(first)\n"
    "            second = int(second)\n"
    "            old_label = int(old_label)\n"
    "            new_label = int(new_label)\n"
    "            candidate_score"
)
if patched.count(anneal_needle) != 1:
    raise RuntimeError("annealing scoring site mismatch")
patched = patched.replace(anneal_needle, anneal_replacement)

polish_needle = (
    "            if not changed:\n"
    "                continue\n"
    "            candidate_score"
)
polish_replacement = (
    "            if not changed:\n"
    "                continue\n"
    "            first = int(first)\n"
    "            second = int(second)\n"
    "            old_label = int(old_label)\n"
    "            new_label = int(new_label)\n"
    "            candidate_score"
)
if patched.count(polish_needle) != 1:
    raise RuntimeError("polishing scoring site mismatch")
patched = patched.replace(polish_needle, polish_replacement)

print("V052F_PATCH", {
    "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    "patched_sha256": hashlib.sha256(patched.encode("utf-8")).hexdigest(),
    "reassignment_sites": reassignment_sites,
    "scoring_sites": 2,
}, flush=True)
namespace = {
    "__name__": "__main__",
    "__file__": str(source_path),
}
exec(compile(patched, str(source_path), "exec"), namespace)
