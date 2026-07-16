#!/usr/bin/env python3
"""Run one frozen v0.5.4 A1 codebook-identifiability frontier cell."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import recoverability_v050 as core
import v054_nomenclator_stage_a as stage


def canonical_sha(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iso", default="en")
    parser.add_argument("--split", default="dev", choices=("dev", "test"))
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--candidate-pool", type=int, required=True)
    parser.add_argument("--codebook-size", type=int, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--restarts", type=int, default=16)
    parser.add_argument("--sweeps", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    experiment = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        experiment / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v050_corpora",
    )
    language = languages[args.iso]
    word_model = stage.build_word_model(
        language, candidate_pool_size=args.candidate_pool
    )
    trials = [
        stage.make_trial(
            language,
            word_model,
            args.split,
            args.length,
            args.offset + replicate,
            args.codebook_size,
        )
        for replicate in range(args.replicates)
    ]
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                stage.solve_a1,
                trial,
                language,
                word_model,
                args.restarts,
                args.sweeps,
            )
            for trial in trials
        ]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            print("V054_A1_FRONTIER_TRIAL", json.dumps(row, sort_keys=True), flush=True)
    rows.sort(key=lambda row: row["replicate"])
    summary = stage.summarize(rows, "a1")
    observed_occurrences = []
    for trial in trials:
        observed_occurrences.append(
            sum(symbol in trial.code_to_word for symbol in trial.surface)
        )
    summary["mean_observed_code_occurrences"] = statistics.fmean(observed_occurrences)
    summary["median_observed_code_occurrences"] = statistics.median(observed_occurrences)
    summary["minimum_observed_code_symbols"] = min(
        row["observed_code_symbols"] for row in rows
    )
    payload: dict[str, Any] = {
        "programme": "recoverability-frontier-v0.5.4-a1-identifiability-cell",
        "iso": args.iso,
        "split": args.split,
        "target_length": args.length,
        "candidate_pool": args.candidate_pool,
        "codebook_size": args.codebook_size,
        "offset": args.offset,
        "replicates": args.replicates,
        "schedule": {"restarts": args.restarts, "sweeps": args.sweeps},
        "summary": summary,
        "rows": rows,
    }
    payload["scientific_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V054_A1_FRONTIER_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V054_A1_FRONTIER_SHA256", payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
