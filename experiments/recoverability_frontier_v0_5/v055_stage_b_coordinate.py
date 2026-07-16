#!/usr/bin/env python3
"""v0.5.5 joint substitution/transposition coordinate solver."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

import recoverability_v050 as core
import mono_solver_v051 as mono
import mono_solver_v051_search2 as mono_search
import v055_stage_b_screening as screening
import v055_transposition_stage_a as stage


Candidate = tuple[int, tuple[int, ...]]


def all_candidate_scores(
    decoded_transposed: np.ndarray,
    block_sizes: tuple[int, ...],
    banks: dict[int, np.ndarray],
    model: tuple[np.ndarray, np.ndarray],
) -> list[tuple[float, Candidate]]:
    output: list[tuple[float, Candidate]] = []
    for block_size in block_sizes:
        bank = banks[block_size]
        scores = screening.score_bank(
            decoded_transposed, bank, model[0], model[1]
        )
        for index, score in enumerate(scores):
            output.append(
                (
                    float(score),
                    (
                        block_size,
                        tuple(int(value) for value in bank[index]),
                    ),
                )
            )
    output.sort(key=lambda item: item[0], reverse=True)
    return output


def candidate_catalog(
    block_sizes: tuple[int, ...], banks: dict[int, np.ndarray]
) -> list[Candidate]:
    return [
        (block_size, tuple(int(value) for value in permutation))
        for block_size in block_sizes
        for permutation in banks[block_size]
    ]


def solve_seed(
    trial: stage.TranspositionTrial,
    initial_key: np.ndarray,
    seed_candidate: Candidate,
    block_sizes: tuple[int, ...],
    banks: dict[int, np.ndarray],
    model: tuple[np.ndarray, np.ndarray],
    cycles: int,
    short_iterations: int,
    short_restarts: int,
    seed_index: int,
) -> dict[str, Any]:
    cipher_array = np.asarray(trial.cipher, dtype=np.int32)
    current_candidate = seed_candidate
    current_key = initial_key.copy()
    trajectory: list[dict[str, Any]] = []
    for cycle in range(cycles):
        detransposed = np.asarray(
            stage.invert_blocks(trial.cipher, current_candidate[1]),
            dtype=np.int32,
        )
        current_key, current_score = mono_search.anneal_mono_search2(
            detransposed,
            current_key,
            model[0],
            model[1],
            short_iterations,
            short_restarts,
            int(
                core.stable_seed(
                    "v055-coordinate-short",
                    trial.seed,
                    seed_index,
                    cycle,
                )
                & 0x7FFFFFFFFFFFFFFF
            ),
        )
        decoded_transposed = current_key[cipher_array]
        ranked = all_candidate_scores(
            decoded_transposed, block_sizes, banks, model
        )
        selected_score, selected_candidate = ranked[0]
        trajectory.append(
            {
                "cycle": cycle + 1,
                "input_block_size": current_candidate[0],
                "selected_block_size": selected_candidate[0],
                "selected_permutation": list(selected_candidate[1]),
                "substitution_score_before_reenumeration": float(current_score),
                "reenumerated_score": float(selected_score),
            }
        )
        current_candidate = selected_candidate
    final_detransposed = np.asarray(
        stage.invert_blocks(trial.cipher, current_candidate[1]), dtype=np.int32
    )
    final_score = float(
        mono.score_key(final_detransposed, current_key, model[0], model[1])
    )
    return {
        "seed_index": seed_index,
        "candidate": current_candidate,
        "key": current_key,
        "score": final_score,
        "trajectory": trajectory,
    }


def solve_trial(
    trial: stage.TranspositionTrial,
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    block_sizes: tuple[int, ...],
    banks: dict[int, np.ndarray],
    catalog: list[Candidate],
    top_seeds: int,
    random_seeds: int,
    cycles: int,
    short_iterations: int,
    short_restarts: int,
    full_iterations: int,
    full_restarts: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    cipher_array = np.asarray(trial.cipher, dtype=np.int32)
    initial_key = mono.frequency_key(trial.cipher, language)
    decoded_frequency = initial_key[cipher_array]
    ranked_frequency = all_candidate_scores(
        decoded_frequency, block_sizes, banks, model
    )
    seeds: list[Candidate] = [
        candidate for _score, candidate in ranked_frequency[:top_seeds]
    ]
    rng = random.Random(core.stable_seed("v055-coordinate-random-seeds", trial.seed))
    for candidate in rng.sample(catalog, min(random_seeds, len(catalog))):
        if candidate not in seeds:
            seeds.append(candidate)
    seed_results = [
        solve_seed(
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
    deduplicated: dict[Candidate, dict[str, Any]] = {}
    for item in seed_results:
        candidate = item["candidate"]
        incumbent = deduplicated.get(candidate)
        if incumbent is None or item["score"] > incumbent["score"]:
            deduplicated[candidate] = item
    best_seed = max(deduplicated.values(), key=lambda item: item["score"])
    current_candidate: Candidate = best_seed["candidate"]
    current_key: np.ndarray = best_seed["key"]

    detransposed = np.asarray(
        stage.invert_blocks(trial.cipher, current_candidate[1]), dtype=np.int32
    )
    current_key, current_score = mono_search.anneal_mono_search2(
        detransposed,
        current_key,
        model[0],
        model[1],
        full_iterations,
        full_restarts,
        int(core.stable_seed("v055-coordinate-full-1", trial.seed) & 0x7FFFFFFFFFFFFFFF),
    )
    decoded_transposed = current_key[cipher_array]
    ranked_final = all_candidate_scores(
        decoded_transposed, block_sizes, banks, model
    )
    _reenumerated_score, reenumerated_candidate = ranked_final[0]
    changed_after_full = reenumerated_candidate != current_candidate
    current_candidate = reenumerated_candidate
    if changed_after_full:
        detransposed = np.asarray(
            stage.invert_blocks(trial.cipher, current_candidate[1]),
            dtype=np.int32,
        )
        current_key, current_score = mono_search.anneal_mono_search2(
            detransposed,
            current_key,
            model[0],
            model[1],
            full_iterations,
            full_restarts,
            int(core.stable_seed("v055-coordinate-full-2", trial.seed) & 0x7FFFFFFFFFFFFFFF),
        )
    else:
        detransposed = np.asarray(
            stage.invert_blocks(trial.cipher, current_candidate[1]),
            dtype=np.int32,
        )

    prediction = current_key[detransposed].tolist()
    true_equivalent = (
        current_candidate[0] == trial.block_size
        and current_candidate[1] == trial.permutation
    )
    return {
        "block_size": trial.block_size,
        "replicate": trial.replicate,
        "accuracy": mono.fast_accuracy(trial.plain, prediction),
        "exact": prediction == trial.plain,
        "selected_block_size": current_candidate[0],
        "selected_permutation": list(current_candidate[1]),
        "true_permutation": list(trial.permutation),
        "block_size_correct": current_candidate[0] == trial.block_size,
        "permutation_correct": true_equivalent,
        "top_frequency_true_rank": 1
        + sum(
            score > next(
                value
                for value, candidate in ranked_frequency
                if candidate == (trial.block_size, trial.permutation)
            )
            + 1e-9
            for score, _candidate in ranked_frequency
        ),
        "seed_count": len(seeds),
        "converged_candidate_count": len(deduplicated),
        "best_seed_score": float(best_seed["score"]),
        "final_score": float(current_score),
        "changed_after_full_refinement": changed_after_full,
        "best_seed_trajectory": best_seed["trajectory"],
        "elapsed_seconds": time.perf_counter() - started,
    }


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accuracies = [float(row["accuracy"]) for row in rows]
    result = {
        "trials": len(rows),
        "mean_accuracy": statistics.fmean(accuracies),
        "median_accuracy": statistics.median(accuracies),
        "minimum_accuracy": min(accuracies),
        "exact_rate": statistics.fmean(float(row["exact"]) for row in rows),
        "at_least_70_rate": statistics.fmean(value >= 0.70 for value in accuracies),
        "at_least_90_rate": statistics.fmean(value >= 0.90 for value in accuracies),
        "block_size_accuracy": statistics.fmean(
            float(row["block_size_correct"]) for row in rows
        ),
        "permutation_accuracy": statistics.fmean(
            float(row["permutation_correct"]) for row in rows
        ),
        "mean_seconds": statistics.fmean(float(row["elapsed_seconds"]) for row in rows),
    }
    result["by_block_size"] = {
        str(block_size): basic_summary(
            [row for row in rows if row["block_size"] == block_size]
        )
        for block_size in sorted({row["block_size"] for row in rows})
    }
    return result


def basic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accuracies = [float(row["accuracy"]) for row in rows]
    return {
        "trials": len(rows),
        "mean_accuracy": statistics.fmean(accuracies),
        "median_accuracy": statistics.median(accuracies),
        "at_least_70_rate": statistics.fmean(value >= 0.70 for value in accuracies),
        "block_size_accuracy": statistics.fmean(
            float(row["block_size_correct"]) for row in rows
        ),
        "permutation_accuracy": statistics.fmean(
            float(row["permutation_correct"]) for row in rows
        ),
    }


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
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--block-sizes", default="4,6,8")
    parser.add_argument("--top-seeds", type=int, default=16)
    parser.add_argument("--random-seeds", type=int, default=16)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--short-iterations", type=int, default=50000)
    parser.add_argument("--short-restarts", type=int, default=5)
    parser.add_argument("--full-iterations", type=int, default=700000)
    parser.add_argument("--full-restarts", type=int, default=50)
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
    banks = {size: stage.permutation_bank(size) for size in block_sizes}
    catalog = candidate_catalog(block_sizes, banks)
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
    compile_key = mono.frequency_key(compile_trial.cipher, language)
    compile_cipher = np.asarray(compile_trial.cipher, dtype=np.int32)
    screening.score_bank(
        compile_key[compile_cipher],
        banks[compile_trial.block_size][:1],
        model[0],
        model[1],
    )
    mono_search.anneal_mono_search2(
        np.asarray(
            stage.invert_blocks(compile_trial.cipher, compile_trial.permutation),
            dtype=np.int32,
        ),
        compile_key,
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
                solve_trial,
                trial,
                language,
                model,
                block_sizes,
                banks,
                catalog,
                args.top_seeds,
                args.random_seeds,
                args.cycles,
                args.short_iterations,
                args.short_restarts,
                args.full_iterations,
                args.full_restarts,
            )
            for trial in trials
        ]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            print("V055_COORDINATE_TRIAL", json.dumps(row, sort_keys=True), flush=True)
    rows.sort(key=lambda row: (row["block_size"], row["replicate"]))
    result_summary = summary(rows)
    gate = {
        "mean_70_pass": result_summary["mean_accuracy"] >= 0.70,
        "median_90_pass": result_summary["median_accuracy"] >= 0.90,
        "twenty_of_twenty_four_70_pass": result_summary["at_least_70_rate"] >= (20 / 24),
        "block_size_90_pass": result_summary["block_size_accuracy"] >= 0.90,
        "permutation_80_pass": result_summary["permutation_accuracy"] >= 0.80,
    }
    gate["pass"] = all(gate.values())
    payload: dict[str, Any] = {
        "programme": "recoverability-frontier-v0.5.5-stage-b-coordinate",
        "iso": args.iso,
        "split": args.split,
        "length": args.length,
        "offset": args.offset,
        "replicates_per_block_size": args.replicates,
        "block_sizes": list(block_sizes),
        "schedule": {
            "top_frequency_seeds": args.top_seeds,
            "random_full_space_seeds": args.random_seeds,
            "coordinate_cycles": args.cycles,
            "short_iterations": args.short_iterations,
            "short_restarts": args.short_restarts,
            "full_iterations": args.full_iterations,
            "full_restarts": args.full_restarts,
        },
        "summary": result_summary,
        "gate": gate,
        "rows": rows,
    }
    payload["scientific_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V055_COORDINATE_SUMMARY", json.dumps(result_summary, sort_keys=True), flush=True)
    print("V055_COORDINATE_GATE", json.dumps(gate, sort_keys=True), flush=True)
    print("V055_COORDINATE_SHA256", payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
