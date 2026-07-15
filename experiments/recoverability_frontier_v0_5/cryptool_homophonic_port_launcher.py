#!/usr/bin/env python3
"""Launch the attributed CrypTool-style port with explicit Numba label casts."""
from __future__ import annotations

import hashlib
from pathlib import Path

path = Path(__file__).with_name("cryptool_homophonic_port_v052.py")
source = path.read_text(encoding="utf-8")
needle = (
    "        if selected < 0:\n"
    "            state, selected = rng_int(state, alphabet_size)\n"
    "        key[key_index] = selected\n"
    "        distribution[selected] += 1\n"
)
replacement = (
    "        if selected < 0:\n"
    "            state, selected = rng_int(state, alphabet_size)\n"
    "        selected = int(selected)\n"
    "        key[key_index] = selected\n"
    "        distribution[selected] += 1\n"
)
if source.count(needle) != 1:
    raise RuntimeError("distributor cast site mismatch")
patched = source.replace(needle, replacement)
print("V052_CRYPTTOOL_PATCH", {
    "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    "patched_sha256": hashlib.sha256(patched.encode("utf-8")).hexdigest(),
    "cast_sites": 1,
}, flush=True)
exec(compile(patched, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
