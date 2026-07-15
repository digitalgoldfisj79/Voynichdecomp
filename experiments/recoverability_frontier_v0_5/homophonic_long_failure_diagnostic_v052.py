#!/usr/bin/env python3
"""Diagnose English/Hebrew 384-character homophonic failures."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

import recoverability_v050 as core
import homophonic_solver_v052 as fixed
import mono_solver_v051 as mono
import mono_solver_v051_search2 as mono_search
from homophonic_confirm_v052_quadgram import (
    build_quadgram_model,
    load_flexible_namespace,
    quadgram_score_key,
)


def oracle_frequency_key(trial: dict, language: core.LanguageData) -> np.ndarray:
    labels = np.asarray(trial["true_labels"], dtype=np.int32)
    counts = Counter(map(int, labels))
    expected = np.asarray(
        [float(language.probabilities[int(label)]) / counts[int(label)] for label in labels],
        dtype=np.float64,
    )
    return fixed.frequency_slot_key(trial["cipher"], labels, expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--iso", required=True, choices=("en", "he"))
    parser.add_argument("--offset", type=int, default=64)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=700000)
    parser.add_argument("--restarts", type=int, default=50)
    args = parser.parse_args()

    experiment = args.repo / "experiments" / "recoverability_frontier_v0_5"
    namespace, patched_sha = load_flexible_namespace(
        experiment / "homophonic_solver_v052_flexible.py"
    )
    flexible_search = namespace["flexible_homophonic_search"]
    family_arrays = namespace["family_arrays"]

    mono.score_key = quadgram_score_key
    mono.build_language_model = build_quadgram_model
    languages = core.load_languages(
        experiment / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v050_corpora",
    )
    language = languages[args.iso]
    quadgram, unigram = build_quadgram_model(language)
    pool, caps, cdf = family_arrays(language)

    # Compile both search kernels before parallel execution.
    flexible_search(
        np.asarray([0, 1, 2, 3, 0, 2], dtype=np.int32),
        np.asarray([0, 0, 1, 1], dtype=np.int32),
        quadgram,
        unigram,
        pool,
        caps,
        cdf,
        2,
        1,
        1,
    )
    mono_search.anneal_mono_search2(
        np.asarray([0, 1, 2, 3, 0, 2], dtype=np.int32),
        np.asarray([0, 0, 1, 1], dtype=np.int32),
        quadgram,
        unigram,
        2,
        1,
        1,
    )

    trials = [
        fixed.make_trial(language, "test", 384, args.offset + replicate)
        for replicate in range(args.replicates)
    ]

    def solve(trial: dict) -> dict:
        cipher = np.asarray(trial["cipher"], dtype=np.int32)
        truth = list(map(int, trial["plain"]))
        true_key = np.asarray(trial["true_labels"], dtype=np.int32)
        inferred_initial = fixed.frequency_slot_key(
            trial["cipher"],
            trial["inferred_labels"],
            trial["expected_slot_probabilities"],
        )
        recovered_key, recovered_score = flexible_search(
            cipher,
            inferred_initial,
            quadgram,
            unigram,
            pool,
            caps,
            cdf,
            args.iterations,
            args.restarts,
            int(trial["seed"] & 0x7FFFFFFFFFFFFFFF),
        )
        oracle_initial = oracle_frequency_key(trial, language)
        oracle_key, oracle_score = mono_search.anneal_mono_search2(
            cipher,
            oracle_initial,
            quadgram,
            unigram,
            args.iterations,
            args.restarts,
            int((trial["seed"] + 1) & 0x7FFFFFFFFFFFFFFF),
        )
        true_score = float(quadgram_score_key(cipher, true_key, quadgram, unigram))
        recovered_prediction = recovered_key[cipher].tolist()
        oracle_prediction = oracle_key[cipher].tolist()
        return {
            "replicate": int(trial["replicate"]),
            "seed": int(trial["seed"]),
            "true_score": true_score,
            "recovered_score": float(recovered_score),
            "oracle_inventory_score": float(oracle_score),
            "true_minus_recovered_score_per_char": (true_score - float(recovered_score)) / len(cipher),
            "true_minus_oracle_score_per_char": (true_score - float(oracle_score)) / len(cipher),
            "true_beats_recovered": true_score >= float(recovered_score),
            "true_beats_oracle": true_score >= float(oracle_score),
            "recovered_accuracy": mono.fast_accuracy(truth, recovered_prediction),
            "oracle_inventory_accuracy": mono.fast_accuracy(truth, oracle_prediction),
            "initial_inventory_overlap": float(trial["inventory_overlap"]),
            "recovered_inventory_overlap": fixed.multiset_overlap(recovered_key, trial["true_labels"]),
            "oracle_inventory_overlap": fixed.multiset_overlap(oracle_key, trial["true_labels"]),
        }

    rows: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(solve, trial) for trial in trials]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 5 == 0 or completed == len(futures):
                print(f"V052_LONG_DIAGNOSTIC_PROGRESS {args.iso} {completed}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: row["replicate"])

    summary = {
        "programme": "v0.5.2-long-homophonic-failure-diagnostic",
        "iso": args.iso,
        "length": 384,
        "offset": args.offset,
        "replicates": args.replicates,
        "schedule": {"iterations": args.iterations, "restarts": args.restarts},
        "patched_solver_sha256": patched_sha,
        "mean_recovered_accuracy": statistics.fmean(row["recovered_accuracy"] for row in rows),
        "median_recovered_accuracy": statistics.median(row["recovered_accuracy"] for row in rows),
        "mean_oracle_inventory_accuracy": statistics.fmean(row["oracle_inventory_accuracy"] for row in rows),
        "median_oracle_inventory_accuracy": statistics.median(row["oracle_inventory_accuracy"] for row in rows),
        "true_beats_recovered_rate": statistics.fmean(float(row["true_beats_recovered"]) for row in rows),
        "true_beats_oracle_rate": statistics.fmean(float(row["true_beats_oracle"]) for row in rows),
        "mean_true_minus_recovered_score_per_char": statistics.fmean(
            row["true_minus_recovered_score_per_char"] for row in rows
        ),
        "mean_true_minus_oracle_score_per_char": statistics.fmean(
            row["true_minus_oracle_score_per_char"] for row in rows
        ),
        "mean_initial_inventory_overlap": statistics.fmean(row["initial_inventory_overlap"] for row in rows),
        "mean_recovered_inventory_overlap": statistics.fmean(row["recovered_inventory_overlap"] for row in rows),
        "rows": rows,
    }
    scientific_blob = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()
    summary["scientific_sha256"] = hashlib.sha256(scientific_blob).hexdigest()
    print("V052_LONG_DIAGNOSTIC", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
