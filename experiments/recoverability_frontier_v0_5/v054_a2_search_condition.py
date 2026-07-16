#!/usr/bin/env python3
"""Run one frozen v0.5.4 A2 residual-character-key search budget."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import recoverability_v050 as core
from homophonic_confirm_v052_quadgram import build_quadgram_model
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
    parser.add_argument("--length", type=int, default=384)
    parser.add_argument("--candidate-pool", type=int, default=96)
    parser.add_argument("--codebook-size", type=int, default=24)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--restarts", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16)
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
    quadgram = build_quadgram_model(language)
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

    compile_trial = trials[0]
    mixed, symbols = stage.build_mixed_cipher(
        compile_trial, compile_trial.code_to_word, len(language.alphabet)
    )
    _symbol_list, initial = stage.frequency_initial_key(compile_trial, language)
    full_key = np.empty(len(symbols) + len(language.alphabet), dtype=np.int32)
    full_key[: len(symbols)] = initial
    full_key[len(symbols) :] = np.arange(len(language.alphabet), dtype=np.int32)
    stage.anneal_locked_key(
        mixed, full_key, len(symbols), quadgram[0], quadgram[1], 2, 1, 1
    )

    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                stage.solve_a2,
                trial,
                language,
                quadgram,
                args.iterations,
                args.restarts,
            )
            for trial in trials
        ]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            print("V054_A2_SEARCH_TRIAL", json.dumps(row, sort_keys=True), flush=True)
    rows.sort(key=lambda row: row["replicate"])
    summary = stage.summarize(rows, "a2")
    summary["at_least_70_rate"] = sum(
        row["expanded_accuracy"] >= 0.70 for row in rows
    ) / len(rows)
    summary["at_least_90_rate"] = sum(
        row["expanded_accuracy"] >= 0.90 for row in rows
    ) / len(rows)
    payload: dict[str, Any] = {
        "programme": "recoverability-frontier-v0.5.4-a2-search-cell",
        "iso": args.iso,
        "split": args.split,
        "target_length": args.length,
        "candidate_pool": args.candidate_pool,
        "codebook_size": args.codebook_size,
        "offset": args.offset,
        "replicates": args.replicates,
        "schedule": {"iterations": args.iterations, "restarts": args.restarts},
        "summary": summary,
        "rows": rows,
    }
    payload["scientific_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V054_A2_SEARCH_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V054_A2_SEARCH_SHA256", payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
