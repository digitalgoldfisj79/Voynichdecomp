#!/usr/bin/env python3
"""Compare inferred versus oracle observed homophone inventories for v0.5.2."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

import recoverability_v050 as core
import homophonic_solver_v052 as homo
import mono_solver_v051 as mono
import mono_solver_v051_search2 as search2


def oracle_slot_key(trial: dict, language: core.LanguageData) -> np.ndarray:
    labels = np.asarray(trial["true_labels"], dtype=np.int32)
    label_counts = Counter(map(int, labels))
    expected = np.asarray(
        [float(language.probabilities[int(label)]) / label_counts[int(label)] for label in labels],
        dtype=np.float64,
    )
    return homo.frequency_slot_key(trial["cipher"], labels, expected)


def solve_arm(trial: dict, language: core.LanguageData, models, oracle: bool, iterations: int, restarts: int):
    trigram, unigram = models[language.iso]
    cipher = np.asarray(trial["cipher"], dtype=np.int32)
    if oracle:
        initial = oracle_slot_key(trial, language)
    else:
        initial = homo.frequency_slot_key(
            trial["cipher"],
            trial["inferred_labels"],
            trial["expected_slot_probabilities"],
        )
    key, score = search2.anneal_mono_search2(
        cipher,
        initial,
        trigram,
        unigram,
        iterations,
        restarts,
        int((trial["seed"] + (1 if oracle else 0)) & 0x7FFFFFFFFFFFFFFF),
    )
    prediction = key[cipher].tolist()
    return mono.fast_accuracy(trial["plain"], prediction), float(score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--replicates", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=300000)
    parser.add_argument("--restarts", type=int, default=35)
    args = parser.parse_args()

    experiment = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        experiment / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v050_corpora",
    )
    languages = {iso: languages[iso] for iso in ("en", "tr")}
    models = {iso: mono.build_language_model(language) for iso, language in languages.items()}
    first_iso = sorted(languages)[0]
    search2.anneal_mono_search2(
        np.asarray([0, 1, 2, 3, 0, 2], dtype=np.int32),
        np.asarray([0, 0, 1, 1], dtype=np.int32),
        models[first_iso][0],
        models[first_iso][1],
        2,
        1,
        1,
    )

    trials = []
    for iso, language in languages.items():
        for replicate in range(args.replicates):
            trials.append((homo.make_trial(language, "dev", 96, replicate), language))

    def execute(item):
        trial, language = item
        inferred_accuracy, inferred_score = solve_arm(
            trial, language, models, False, args.iterations, args.restarts
        )
        oracle_accuracy, oracle_score = solve_arm(
            trial, language, models, True, args.iterations, args.restarts
        )
        return {
            "iso": language.iso,
            "replicate": trial["replicate"],
            "inventory_overlap": float(trial["inventory_overlap"]),
            "inferred_accuracy": inferred_accuracy,
            "oracle_accuracy": oracle_accuracy,
            "oracle_minus_inferred_accuracy": oracle_accuracy - inferred_accuracy,
            "oracle_minus_inferred_score": oracle_score - inferred_score,
        }

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for completed, row in enumerate(executor.map(execute, trials), start=1):
            rows.append(row)
            if completed % 8 == 0 or completed == len(trials):
                print(f"V052_INVENTORY_PROGRESS {completed}/{len(trials)}", flush=True)

    summary = {
        "trials": len(rows),
        "iterations": args.iterations,
        "restarts": args.restarts,
        "mean_inventory_overlap": statistics.fmean(row["inventory_overlap"] for row in rows),
        "mean_inferred_accuracy": statistics.fmean(row["inferred_accuracy"] for row in rows),
        "mean_oracle_accuracy": statistics.fmean(row["oracle_accuracy"] for row in rows),
        "mean_oracle_gain": statistics.fmean(row["oracle_minus_inferred_accuracy"] for row in rows),
        "by_language": {
            iso: {
                "mean_inventory_overlap": statistics.fmean(row["inventory_overlap"] for row in rows if row["iso"] == iso),
                "mean_inferred_accuracy": statistics.fmean(row["inferred_accuracy"] for row in rows if row["iso"] == iso),
                "mean_oracle_accuracy": statistics.fmean(row["oracle_accuracy"] for row in rows if row["iso"] == iso),
            }
            for iso in sorted(languages)
        },
        "rows": rows,
    }
    print("V052_INVENTORY_DIAGNOSTIC", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
