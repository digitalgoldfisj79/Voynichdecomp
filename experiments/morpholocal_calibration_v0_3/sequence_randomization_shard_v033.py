#!/usr/bin/env python3
"""Restart-safe v0.3.3 latent-order randomization shard runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import remediation_runtime_v033 as remediation

base = remediation.base


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate(results: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    positives = [row for row in results if row["trial_type"] == "positive"]
    controls = [row for row in results if row["trial_type"] == "control"]

    def passed(row: dict[str, Any]) -> bool:
        audit = row["sequence_randomization"]
        return bool(float(audit["p_value"]) <= alpha and float(audit["advantage_bits"]) > 0.0)

    summary: dict[str, Any] = {
        "alpha": alpha,
        "positive": {
            "passes": sum(passed(row) for row in positives),
            "trials": len(positives),
            "median_p": base.statistics.median(
                [float(row["sequence_randomization"]["p_value"]) for row in positives]
            ) if positives else None,
            "median_advantage_bits_per_transition": base.statistics.median(
                [
                    float(row["sequence_randomization"]["advantage_bits_per_transition"])
                    for row in positives
                ]
            ) if positives else None,
        },
        "control": {
            "false_positives": sum(passed(row) for row in controls),
            "trials": len(controls),
            "median_p": base.statistics.median(
                [float(row["sequence_randomization"]["p_value"]) for row in controls]
            ) if controls else None,
            "median_advantage_bits_per_transition": base.statistics.median(
                [
                    float(row["sequence_randomization"]["advantage_bits_per_transition"])
                    for row in controls
                ]
            ) if controls else None,
        },
        "positive_policy": {},
        "control_family": {},
        "length_profile": {},
    }
    for key, target, rows in (
        ("truth.selection_policy", "positive_policy", positives),
        ("control_family", "control_family", controls),
        ("length_profile", "length_profile", results),
    ):
        buckets: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if key == "truth.selection_policy":
                value = row["truth"]["selection_policy"]
            else:
                value = row.get(key)
            buckets[str(value)].append(row)
        summary[target] = {
            value: {
                "passes": sum(passed(row) for row in bucket),
                "trials": len(bucket),
                "median_p": base.statistics.median(
                    [float(row["sequence_randomization"]["p_value"]) for row in bucket]
                ),
                "median_advantage_bits_per_transition": base.statistics.median(
                    [
                        float(row["sequence_randomization"]["advantage_bits_per_transition"])
                        for row in bucket
                    ]
                ),
            }
            for value, bucket in sorted(buckets.items())
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--solver", choices=sorted(base.SOLVERS), required=True)
    parser.add_argument("--positive-start", type=int, default=0)
    parser.add_argument("--positive-end", type=int, default=0)
    parser.add_argument("--control-start", type=int, default=0)
    parser.add_argument("--control-end", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3030303)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.positive_end < args.positive_start or args.control_end < args.control_start:
        raise SystemExit("end indices must be at least start indices")
    if args.positive_end == args.positive_start and args.control_end == args.control_start:
        raise SystemExit("at least one non-empty range is required")

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
        "sequence_randomizations": 199,
    }
    config.update(json.loads(args.config.read_text()))

    remediation.install()
    base.load_v02(args.repo)
    lengths = tuple(base.LENGTHS)
    tasks: list[dict[str, Any]] = []
    tasks.extend(
        {
            "kind": "positive",
            "index": index,
            "seed": args.seed + 100000 + index * 7919,
            "length_profile": lengths[index % len(lengths)],
        }
        for index in range(args.positive_start, args.positive_end)
    )
    tasks.extend(
        {
            "kind": "control",
            "index": index,
            "seed": args.seed + 900000 + index * 104729,
            "length_profile": lengths[index % len(lengths)],
        }
        for index in range(args.control_start, args.control_end)
    )

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
            results.append(result)
            task = futures[future]
            seq = result["sequence_randomization"]
            print(
                f"V033_SEQUENCE_PROGRESS solver={args.solver} "
                f"completed={completed}/{len(tasks)} kind={task['kind']} "
                f"index={task['index']} length={task['length_profile']} "
                f"p={float(seq['p_value']):.6g} "
                f"effect={float(seq['advantage_bits_per_transition']):.6g} "
                f"elapsed={time.time()-started:.1f}s",
                flush=True,
            )

    results.sort(key=lambda row: (row["trial_type"], int(row["trial_index"])))
    root = args.repo
    source_paths = [
        root / "experiments/morpholocal_calibration_v0_3/sequence_randomization_shard_v033.py",
        root / "experiments/morpholocal_calibration_v0_3/remediation_runtime_v033.py",
        root / "experiments/morpholocal_calibration_v0_3/remediation_runtime_v032.py",
        root / "experiments/morpholocal_calibration_v0_3/remediation_runtime.py",
        root / "experiments/morpholocal_calibration_v0_3/tournament_runner.py",
        root / "experiments/morpholocal_calibration_v0_3/production_null_registry.py",
        root / "experiments/morpholocal_calibration_v0_2/morpholocal_gate_impl.py",
    ]
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    payload = {
        "programme": "morpholocal-calibration-v0.3.3-latent-order-randomization",
        "formal": False,
        "primary_metric": "sequence_randomization_pass",
        "solver": args.solver,
        "seed": args.seed,
        "config": config,
        "alpha": args.alpha,
        "positive_start": args.positive_start,
        "positive_end": args.positive_end,
        "control_start": args.control_start,
        "control_end": args.control_end,
        "workers": args.workers,
        "git_commit": commit,
        "scientific_source_sha256": {
            str(path.relative_to(root)): sha256_file(path) for path in source_paths
        },
        "summary": aggregate(results, args.alpha),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print("V033_SEQUENCE_SUMMARY", json.dumps(payload["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
