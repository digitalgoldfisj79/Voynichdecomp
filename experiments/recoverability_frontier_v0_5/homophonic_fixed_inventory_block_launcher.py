#!/usr/bin/env python3
"""Launch fixed-inventory block search after removing unsupported Numba del."""
from pathlib import Path
import hashlib

path = Path(__file__).with_name("homophonic_fixed_inventory_block_v052.py")
source = path.read_text(encoding="utf-8")
needle = "    del slot_pool, max_counts\n"
if source.count(needle) != 1:
    raise RuntimeError("unused-argument site mismatch")
patched = source.replace(needle, "")
print("V052_FIXED_BLOCK_PATCH", {
    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    "patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
}, flush=True)
exec(compile(patched, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
