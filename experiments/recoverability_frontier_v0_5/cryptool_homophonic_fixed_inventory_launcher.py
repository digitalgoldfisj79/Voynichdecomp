#!/usr/bin/env python3
"""Launch the attributed CrypTool-style port with inventory mutation disabled."""
from __future__ import annotations

import hashlib
from pathlib import Path

path = Path(__file__).with_name("cryptool_homophonic_port_v052.py")
source = path.read_text(encoding="utf-8")

cast_needle = (
    "        if selected < 0:\n"
    "            state, selected = rng_int(state, alphabet_size)\n"
    "        key[key_index] = selected\n"
    "        distribution[selected] += 1\n"
)
cast_replacement = (
    "        if selected < 0:\n"
    "            state, selected = rng_int(state, alphabet_size)\n"
    "        selected = int(selected)\n"
    "        key[key_index] = selected\n"
    "        distribution[selected] += 1\n"
)
if source.count(cast_needle) != 1:
    raise RuntimeError("distributor cast site mismatch")
patched = source.replace(cast_needle, cast_replacement)

mutation_needle = (
    "        50,\n"
    "        3,\n"
    "        int(trial[\"seed\"] & 0x7FFFFFFFFFFFFFFF),\n"
)
mutation_replacement = (
    "        50,\n"
    "        0,\n"
    "        int(trial[\"seed\"] & 0x7FFFFFFFFFFFFFFF),\n"
)
if patched.count(mutation_needle) != 1:
    raise RuntimeError("mutation-count site mismatch")
patched = patched.replace(mutation_needle, mutation_replacement)

print("V052_CRYPTTOOL_FIXED_PATCH", {
    "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    "patched_sha256": hashlib.sha256(patched.encode("utf-8")).hexdigest(),
    "inventory_mutation_count": 0,
}, flush=True)
exec(compile(patched, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
