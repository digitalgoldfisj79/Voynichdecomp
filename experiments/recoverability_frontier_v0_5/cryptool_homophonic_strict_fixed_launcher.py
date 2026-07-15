#!/usr/bin/env python3
"""Launch the CrypTool-style port with the inferred inventory fixed in every restart."""
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

restart_needle = (
    "        if restart == 0:\n"
    "            start_key = initial_key.copy()\n"
    "        else:\n"
    "            state, start_key = distributor_key(\n"
    "                state, initial_key.shape[0], min_counts, max_counts\n"
    "            )\n"
)
restart_replacement = (
    "        start_key = initial_key.copy()\n"
    "        if restart > 0:\n"
    "            for shuffle_index in range(start_key.shape[0] - 1, 0, -1):\n"
    "                state, shuffle_other = rng_int(state, shuffle_index + 1)\n"
    "                shuffle_temporary = start_key[shuffle_index]\n"
    "                start_key[shuffle_index] = start_key[shuffle_other]\n"
    "                start_key[shuffle_other] = shuffle_temporary\n"
)
if patched.count(restart_needle) != 1:
    raise RuntimeError("restart inventory site mismatch")
patched = patched.replace(restart_needle, restart_replacement)

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

print("V052_CRYPTTOOL_STRICT_FIXED_PATCH", {
    "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    "patched_sha256": hashlib.sha256(patched.encode("utf-8")).hexdigest(),
    "restart_inventory": "permutation_of_inferred_multiset",
    "inventory_mutation_count": 0,
}, flush=True)
exec(compile(patched, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
