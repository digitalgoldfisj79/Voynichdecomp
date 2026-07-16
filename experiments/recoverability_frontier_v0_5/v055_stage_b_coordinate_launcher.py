#!/usr/bin/env python3
"""Launch v0.5.5 Stage B after removing a quadratic diagnostic calculation.

The scientific search is unchanged. The patch computes the true frequency-screen
score once instead of searching for it again for every candidate while reporting
its rank.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

path = Path(__file__).with_name("v055_stage_b_coordinate.py")
source = path.read_text(encoding="utf-8")

insert_needle = "    prediction = current_key[detransposed].tolist()\n    true_equivalent = (\n"
insert_replacement = (
    "    prediction = current_key[detransposed].tolist()\n"
    "    true_frequency_score = next(\n"
    "        value\n"
    "        for value, candidate in ranked_frequency\n"
    "        if candidate == (trial.block_size, trial.permutation)\n"
    "    )\n"
    "    true_frequency_rank = 1 + sum(\n"
    "        value > true_frequency_score + 1e-9\n"
    "        for value, _candidate in ranked_frequency\n"
    "    )\n"
    "    true_equivalent = (\n"
)
if source.count(insert_needle) != 1:
    raise RuntimeError("rank precomputation insertion site mismatch")
patched = source.replace(insert_needle, insert_replacement)

rank_needle = '''        "top_frequency_true_rank": 1
        + sum(
            score > next(
                value
                for value, candidate in ranked_frequency
                if candidate == (trial.block_size, trial.permutation)
            )
            + 1e-9
            for score, _candidate in ranked_frequency
        ),
'''
rank_replacement = '        "top_frequency_true_rank": true_frequency_rank,\n'
if patched.count(rank_needle) != 1:
    raise RuntimeError("quadratic rank calculation site mismatch")
patched = patched.replace(rank_needle, rank_replacement)

print(
    "V055_COORDINATE_PATCH",
    {
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "patched_sha256": hashlib.sha256(patched.encode("utf-8")).hexdigest(),
        "scientific_search_changed": False,
        "diagnostic_complexity": "O(N) instead of O(N^2)",
    },
    flush=True,
)
exec(compile(patched, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
