#!/usr/bin/env python3
"""Fixed-inventory block/anneal hybrid for English homophonic development."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

import recoverability_v050 as core
import homophonic_block_search_v052 as block
import homophonic_fixed_inventory_block_v052 as fixed_block
import homophonic_solver_v052 as fixed
import mono_solver_v051 as mono
import mono_solver_v051_search2 as mono_search
from homophonic_confirm_v052_quadgram import build_quadgram_model, quadgram_score_key


def fixed_counts_from_key(key: np.ndarray, alphabet_size: int) -> np.ndarray:
    counts = np.zeros(alphabet_size, dtype=np.int32)
    for value in key:
        counts[int(value)] += 1
    return counts


def shuffled_numpy(key: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = key.copy()
    rng.shuffle(output)
    return output


def hybrid_search(
    cipher: np.ndarray,
    initial_key: np.ndarray,
    model: tuple[np.ndarray, np.ndarray],
    outer_restarts: int,
    block_sweeps: int,
    anneal_iterations: int,
    anneal_restarts: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    fixed_counts = fixed_counts_from_key(initial_key, model[0].shape[0])
    best_key = initial_key.copy()
    best_score = float(quadgram_score_key(cipher, best_key, model[0], model[1]))

    for outer in range(outer_restarts):
        if outer == 0:
            candidate = initial_key.copy()
        elif outer % 3 == 1:
            candidate = best_key.copy()
            rng = np.random.default_rng(seed + 7919 * outer)
            for _ in range(2 + outer % 9):
                first = int(rng.integers(0, len(candidate)))
                second = int(rng.integers(0, len(candidate)))
                if first != second:
                    candidate[first], candidate[second] = candidate[second], candidate[first]
        else:
            candidate = shuffled_numpy(initial_key, seed + 104729 * outer)

        candidate, _ = fixed_block.pair_block_polish_fixed(
            cipher,
            candidate,
            model[0],
            model[1],
            fixed_counts,
            block_sweeps,
            seed + 17 * outer,
        )
        candidate, _ = mono_search.anneal_mono_search2(
            cipher,
            candidate,
            model[0],
            model[1],
            anneal_iterations,
            anneal_restarts,
            int((seed + 65537 * outer) & 0x7FFFFFFFFFFFFFFF),
        )
        candidate, candidate_score = fixed_block.pair_block_polish_fixed(
            cipher,
            candidate,
            model[0],
            model[1],
            fixed_counts,
            block_sweeps,
            seed + 31 * outer,
        )
        if float(candidate_score) > best_score:
            best_score = float(candidate_score)
            best_key = candidate.copy()

    return best_key, best_score


def solve_trial(trial: dict, language: core.LanguageData, model, schedule: tuple[int, int, int, int]) -> dict:
    outer, sweeps, iterations, inner = schedule
    cipher = np.asarray(trial["cipher"], dtype=np.int32)
    initial = fixed.frequency_slot_key(
        trial["cipher"],
        trial["inferred_labels"],
        trial["expected_slot_probabilities"],
    )
    key, score = hybrid_search(
        cipher,
        initial,
        model,
        outer,
        sweeps,
        iterations,
        inner,
        int(trial["seed"] & 0x7FFFFFFFFFFFFFFF),
    )
    prediction = key[cipher].tolist()
    baseline = initial[cipher].tolist()
    return {
        "replicate": trial["replicate"],
        "accuracy": mono.fast_accuracy(trial["plain"], prediction),
        "baseline_accuracy": mono.fast_accuracy(trial["plain"], baseline),
        "exact": prediction == trial["plain"],
        "initial_inventory_overlap": float(trial["inventory_overlap"]),
        "final_inventory_overlap": fixed.multiset_overlap(key, trial["true_labels"]),
        "score": float(score),
    }


def summarize(rows: list[dict]) -> dict:
    return {
        "trials": len(rows),
        "mean_accuracy": statistics.fmean(row["accuracy"] for row in rows),
        "median_accuracy": statistics.median(row["accuracy"] for row in rows),
        "baseline_mean_accuracy": statistics.fmean(row["baseline_accuracy"] for row in rows),
        "exact_rate": statistics.fmean(float(row["exact"]) for row in rows),
        "mean_initial_inventory_overlap": statistics.fmean(row["initial_inventory_overlap"] for row in rows),
        "mean_final_inventory_overlap": statistics.fmean(row["final_inventory_overlap"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    mono.score_key = quadgram_score_key
    experiment = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        experiment / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v050_corpora",
    )
    language = languages["en"]
    model = build_quadgram_model(language)

    # Compile kernels before parallel execution.
    dummy_cipher = np.asarray([0, 1, 2, 3, 0, 2], dtype=np.int32)
    dummy_key = np.asarray([0, 0, 1, 1], dtype=np.int32)
    dummy_counts = fixed_counts_from_key(dummy_key, len(language.alphabet))
    fixed_block.pair_block_polish_fixed(
        dummy_cipher, dummy_key, model[0], model[1], dummy_counts, 1, 1
    )
    mono_search.anneal_mono_search2(
        dummy_cipher, dummy_key, model[0], model[1], 2, 1, 1
    )

    trials = [fixed.make_trial(language, "dev", 384, replicate) for replicate in range(args.replicates)]
    schedules = (
        (3, 5, 50000, 2),
        (6, 8, 100000, 3),
        (12, 12, 200000, 4),
    )
    candidates = []
    selected = None
    selected_score = -1.0

    for schedule in schedules:
        rows = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(solve_trial, trial, language, model, schedule)
                for trial in trials
            ]
            for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                rows.append(future.result())
                if completed % 4 == 0 or completed == len(futures):
                    print(f"V052_HYBRID_PROGRESS {completed}/{len(futures)} schedule={schedule}", flush=True)
        rows.sort(key=lambda row: row["replicate"])
        summary = summarize(rows)
        candidate = {
            "outer_restarts": schedule[0],
            "block_sweeps": schedule[1],
            "anneal_iterations": schedule[2],
            "anneal_restarts": schedule[3],
            "summary": summary,
        }
        candidates.append(candidate)
        print("V052_HYBRID_CANDIDATE", json.dumps(candidate, sort_keys=True), flush=True)
        if summary["mean_accuracy"] > selected_score:
            selected_score = summary["mean_accuracy"]
            selected = candidate

    payload = {
        "programme": "v0.5.2-fixed-inventory-block-anneal-hybrid",
        "iso": "en",
        "split": "dev",
        "length": 384,
        "replicates": args.replicates,
        "candidates": candidates,
        "selected": selected,
        "gate": {
            "english_70_percent_pass": selected["summary"]["mean_accuracy"] >= 0.70,
        },
    }
    payload["gate"]["pass"] = all(payload["gate"].values())
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["scientific_sha256"] = hashlib.sha256(blob).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V052_HYBRID_SELECTED", json.dumps(selected, sort_keys=True), flush=True)
    print("V052_HYBRID_GATE", json.dumps(payload["gate"], sort_keys=True), flush=True)
    print("V052_HYBRID_SHA256", payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
