#!/usr/bin/env python3
"""Run the frozen stratified coordinate search with deterministic seed parallelism.

Only execution scheduling changes: independent, deterministically seeded starting
states are evaluated concurrently. Scientific candidates, objectives and results
are unchanged.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

path = Path(__file__).with_name("v055_stage_b_coordinate_stratified.py")
source = path.read_text(encoding="utf-8")
source = source.replace(
    "import random\nimport statistics\nimport time\n",
    "import concurrent.futures\nimport random\nimport statistics\nimport time\n",
)
needle = '''    seed_results = [
        base.solve_seed(
            trial,
            initial_key,
            candidate,
            block_sizes,
            banks,
            model,
            cycles,
            short_iterations,
            short_restarts,
            seed_index,
        )
        for seed_index, candidate in enumerate(seeds)
    ]
'''
replacement = '''    def evaluate_seed(item):
        seed_index, candidate = item
        return base.solve_seed(
            trial,
            initial_key,
            candidate,
            block_sizes,
            banks,
            model,
            cycles,
            short_iterations,
            short_restarts,
            seed_index,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as seed_executor:
        seed_results = list(seed_executor.map(evaluate_seed, enumerate(seeds)))
'''
if source.count(needle) != 1:
    raise RuntimeError("seed-evaluation site mismatch")
patched = source.replace(needle, replacement)
print(
    "V055_STRATIFIED_PARALLEL_PATCH",
    {
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "patched_sha256": hashlib.sha256(patched.encode("utf-8")).hexdigest(),
        "scientific_search_changed": False,
        "seed_workers_per_trial": 4,
    },
    flush=True,
)
exec(compile(patched, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
