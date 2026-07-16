#!/usr/bin/env python3
"""Valid v0.5.5 A2 rerun using the passing v0.5.1 mono solver."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import recoverability_v050 as core
import mono_solver_v051 as mono
import mono_solver_v051_search2 as mono_search
import v055_transposition_stage_a as stage


def canonical_sha(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def solve(
    trial: stage.TranspositionTrial,
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    iterations: int,
    restarts: int,
) -> dict[str, Any]:
    return stage.solve_a2(trial, language, model, iterations, restarts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iso", default="en")
    parser.add_argument("--split", default="dev", choices=("dev", "test"))
    parser.add_argument("--length", type=int, default=384)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--block-sizes", default="4,6,8")
    parser.add_argument("--iterations", type=int, default=700000)
    parser.add_argument("--restarts", type=int, default=50)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    block_sizes = tuple(int(value) for value in args.block_sizes.split(",") if value)
    experiment = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        experiment / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v050_corpora",
    )
    language = languages[args.iso]
    model = mono.build_language_model(language)
    trials = [
        stage.make_trial(
            language,
            args.split,
            args.length,
            block_size,
            args.offset + replicate,
        )
        for block_size in block_sizes
        for replicate in range(args.replicates)
    ]

    compile_trial = trials[0]
    detransposed = stage.invert_blocks(
        compile_trial.cipher, compile_trial.permutation
    )
    initial = stage.frequency_initial_key(
        detransposed, compile_trial.observed_plain_inventory, language
    )
    mono_search.anneal_mono_search2(
        np.asarray(detransposed, dtype=np.int32),
        initial,
        model[0],
        model[1],
        2,
        1,
        1,
    )

    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                solve,
                trial,
                language,
                model,
                args.iterations,
                args.restarts,
            )
            for trial in trials
        ]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            print("V055_VALID_A2_TRIAL", json.dumps(row, sort_keys=True), flush=True)
    rows.sort(key=lambda row: (row["block_size"], row["replicate"]))
    summary = stage.summarize(rows, "a2")
    gate_by_size: dict[str, dict[str, bool]] = {}
    for block_size in block_sizes:
        item = summary["by_block_size"][str(block_size)]
        gate_by_size[str(block_size)] = {
            "mean_90_pass": item["mean_accuracy"] >= 0.90,
            "median_99_pass": item["median_accuracy"] >= 0.99,
            "seven_of_eight_90_pass": item["at_least_90_rate"] >= 0.875,
        }
        gate_by_size[str(block_size)]["pass"] = all(
            gate_by_size[str(block_size)].values()
        )
    gate = {
        "by_block_size": gate_by_size,
        "pass": all(item["pass"] for item in gate_by_size.values()),
    }
    payload: dict[str, Any] = {
        "programme": "recoverability-frontier-v0.5.5-valid-a2-mono",
        "iso": args.iso,
        "split": args.split,
        "length": args.length,
        "offset": args.offset,
        "replicates_per_block_size": args.replicates,
        "block_sizes": list(block_sizes),
        "solver": "unmodified v0.5.1 trigram-plus-unigram mono search",
        "schedule": {"iterations": args.iterations, "restarts": args.restarts},
        "summary": summary,
        "gate": gate,
        "rows": rows,
    }
    payload["scientific_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V055_VALID_A2_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V055_VALID_A2_GATE", json.dumps(gate, sort_keys=True), flush=True)
    print("V055_VALID_A2_SHA256", payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
