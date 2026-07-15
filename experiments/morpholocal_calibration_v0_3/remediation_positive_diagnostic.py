#!/usr/bin/env python3
"""Balanced positive diagnostic runner for v0.3.1 remediation.

Runs an explicit preregistered set of positive scenario indices. Each task uses
exactly the original v0.3 seed rule and length assignment; only task selection
is restricted to the declared diagnostic panel.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import remediation_runtime as remediation

base = remediation.base

DEFAULT_INDICES = (
    3, 4, 8, 14, 19, 21,
    27, 28, 32, 38, 43, 45,
    51, 52, 56, 62, 67, 69,
    75, 76, 80, 86, 91, 93,
)


def parse_indices(value: str) -> tuple[int, ...]:
    indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not indices:
        raise argparse.ArgumentTypeError("at least one index is required")
    if len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError("indices must be unique")
    if any(index < 0 or index >= 96 for index in indices):
        raise argparse.ArgumentTypeError("indices must be in [0, 95]")
    return indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--solver", choices=sorted(base.SOLVERS), required=True)
    parser.add_argument(
        "--indices",
        type=parse_indices,
        default=DEFAULT_INDICES,
        help="comma-separated positive scenario indices",
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=3030303)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = {
        "steps": 2500,
        "restarts": 8,
        "alternations": 2,
        "refine_iterations": 1000,
        "refine_temperature": 1.0,
        "beam_width": 256,
        "policy_rerank": 64,
        "mcmc_steps": 5000,
        "temperatures": [1.0, 2.0, 4.0, 8.0],
        "policy_update_every": 100,
    }
    config.update(json.loads(args.config.read_text()))

    remediation.install()
    lengths = tuple(base.LENGTHS)
    tasks = [
        {
            "kind": "positive",
            "index": index,
            "seed": args.seed + 100000 + index * 7919,
            "length_profile": lengths[index % len(lengths)],
        }
        for index in args.indices
    ]

    started = time.time()
    results = []
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {
            pool.submit(base.run_task, str(args.repo), args.solver, config, task): task
            for task in tasks
        }
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            task = futures[future]
            print(
                f"V031_POSITIVE_PROGRESS solver={args.solver} "
                f"completed={completed}/{len(tasks)} index={task['index']} "
                f"length={task['length_profile']} elapsed={time.time()-started:.1f}s",
                flush=True,
            )

    results.sort(key=lambda row: int(row["trial_index"]))
    payload = {
        "programme": "morpholocal-calibration-v0.3.1-remediation-positive-diagnostic",
        "formal": False,
        "solver": args.solver,
        "seed": args.seed,
        "config": config,
        "positive_indices": list(args.indices),
        "workers": args.workers,
        "selection_design": {
            "offsets_per_policy": [3, 4, 8, 14, 19, 21],
            "policy_blocks": [0, 24, 48, 72],
            "marginal_balance_per_policy": {
                "length_profile": {"short": 2, "medium": 2, "long": 2},
                "key_scheme": {"global": 3, "currier": 3},
                "null_count": {"0": 3, "2": 3},
                "size_profile": {"balanced": 3, "unequal": 3},
                "external_profile": {"word_heavy": 2, "balanced": 2, "letter_heavy": 2},
                "selector": {"none": 3, "adjacent_length": 3},
            },
        },
        "summary": base.aggregate(results),
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("V031_POSITIVE_SUMMARY", json.dumps(payload["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
