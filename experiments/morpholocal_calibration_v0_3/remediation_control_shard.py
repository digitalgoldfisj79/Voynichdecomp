#!/usr/bin/env python3
"""Deterministic control-index shard runner for v0.3.1 remediation.

The task construction is identical to tournament_runner.py; only a declared
half-open control-index interval is executed. Linux fork is used so the
identity-safe remediation patches are inherited by workers.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--solver", choices=sorted(base.SOLVERS), required=True)
    parser.add_argument("--control-start", type=int, required=True)
    parser.add_argument("--control-end", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3030303)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not (0 <= args.control_start < args.control_end):
        raise SystemExit("require 0 <= control-start < control-end")

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
            "kind": "control",
            "index": index,
            "seed": args.seed + 900000 + index * 104729,
            "length_profile": lengths[index % len(lengths)],
        }
        for index in range(args.control_start, args.control_end)
    ]

    started = time.time()
    results = []
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {
            pool.submit(
                base.run_task,
                str(args.repo),
                args.solver,
                config,
                task,
            ): task
            for task in tasks
        }
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            task = futures[future]
            print(
                f"V031_SHARD_PROGRESS solver={args.solver} "
                f"range={args.control_start}:{args.control_end} "
                f"completed={completed}/{len(tasks)} index={task['index']} "
                f"length={task['length_profile']} elapsed={time.time()-started:.1f}s",
                flush=True,
            )

    results.sort(key=lambda row: int(row["trial_index"]))
    payload = {
        "programme": "morpholocal-calibration-v0.3.1-remediation-control-shard",
        "formal": False,
        "solver": args.solver,
        "seed": args.seed,
        "config": config,
        "control_start": args.control_start,
        "control_end": args.control_end,
        "workers": args.workers,
        "summary": base.aggregate(results),
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("V031_SHARD_SUMMARY", json.dumps(payload["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
