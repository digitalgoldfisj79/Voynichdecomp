#!/usr/bin/env python3
"""Restart-safe positive-sensitivity shard runner for v0.3.2 selector parity."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import remediation_positive_shard as v031_runner
import remediation_runtime_v032 as remediation

base = remediation.base


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--solver", choices=sorted(base.SOLVERS), required=True)
    parser.add_argument("--positive-start", type=int, required=True)
    parser.add_argument("--positive-end", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3030303)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (0 <= args.positive_start < args.positive_end):
        raise SystemExit("require 0 <= positive-start < positive-end")

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
    base.load_v02(args.repo)
    lengths = tuple(base.LENGTHS)
    tasks = [
        {
            "kind": "positive",
            "index": index,
            "seed": args.seed + 100000 + index * 7919,
            "length_profile": lengths[index % len(lengths)],
        }
        for index in range(args.positive_start, args.positive_end)
    ]

    started = time.time()
    results: list[dict[str, Any]] = []
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {
            pool.submit(base.run_task, str(args.repo), args.solver, config, task): task
            for task in tasks
        }
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            result["legacy_positive_success"] = bool(result["positive_success"])
            result["strict_positive_success"] = v031_runner.strict_success(result)
            results.append(result)
            task = futures[future]
            print(
                f"V032_POSITIVE_PROGRESS solver={args.solver} "
                f"range={args.positive_start}:{args.positive_end} "
                f"completed={completed}/{len(tasks)} index={task['index']} "
                f"length={task['length_profile']} "
                f"legacy={int(result['legacy_positive_success'])} "
                f"strict={int(result['strict_positive_success'])} "
                f"elapsed={time.time()-started:.1f}s",
                flush=True,
            )

    results.sort(key=lambda row: int(row["trial_index"]))
    expected = list(range(args.positive_start, args.positive_end))
    observed = [int(row["trial_index"]) for row in results]
    if observed != expected:
        raise RuntimeError(f"positive index mismatch: expected={expected} observed={observed}")

    root = args.repo
    source_paths = [
        root / "experiments/morpholocal_calibration_v0_3/remediation_positive_shard_v032.py",
        root / "experiments/morpholocal_calibration_v0_3/remediation_runtime_v032.py",
        root / "experiments/morpholocal_calibration_v0_3/remediation_runtime.py",
        root / "experiments/morpholocal_calibration_v0_3/tournament_runner.py",
        root / "experiments/morpholocal_calibration_v0_3/production_null_registry.py",
        root / "experiments/morpholocal_calibration_v0_3/tournament_kt.py",
        root / "experiments/morpholocal_calibration_v0_3/tournament_fast.py",
        root / "experiments/morpholocal_calibration_v0_2/morpholocal_gate_impl.py",
    ]
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    payload = {
        "programme": "morpholocal-calibration-v0.3.2-selector-parity-positive-shard",
        "formal": False,
        "primary_metric": "strict_positive_success",
        "comparison_metric": "legacy_positive_success",
        "solver": args.solver,
        "seed": args.seed,
        "config": config,
        "positive_start": args.positive_start,
        "positive_end": args.positive_end,
        "workers": args.workers,
        "git_commit": commit,
        "selector_parity": True,
        "scientific_source_sha256": {
            str(path.relative_to(root)): sha256_file(path) for path in source_paths
        },
        "summary": v031_runner.aggregate(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print("V032_POSITIVE_SUMMARY", json.dumps(payload["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
