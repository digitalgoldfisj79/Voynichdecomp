#!/usr/bin/env python3
"""Rank all v0.5.5 transposition candidates under a frequency key."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit

import recoverability_v050 as core
import mono_solver_v051 as mono
import v055_transposition_stage_a as stage

PREFIXES = (1, 4, 16, 64, 256, 1024)


@njit(cache=True, nogil=True)
def trigram_score(values, trigram_logp, unigram_logp):
    if values.shape[0] == 0:
        return -1e300
    score = 0.15 * unigram_logp[values[0]]
    if values.shape[0] >= 2:
        score += 0.15 * unigram_logp[values[1]]
    for index in range(2, values.shape[0]):
        score += trigram_logp[
            values[index - 2], values[index - 1], values[index]
        ]
        score += 0.15 * unigram_logp[values[index]]
    return score


@njit(cache=True, nogil=True)
def score_bank(decoded_transposed, permutations, trigram_logp, unigram_logp):
    count = permutations.shape[0]
    block_size = permutations.shape[1]
    candidate = np.empty(decoded_transposed.shape[0], dtype=np.int32)
    scores = np.empty(count, dtype=np.float64)
    for permutation_index in range(count):
        permutation = permutations[permutation_index]
        for offset in range(0, decoded_transposed.shape[0], block_size):
            for cipher_position in range(block_size):
                candidate[offset + int(permutation[cipher_position])] = (
                    decoded_transposed[offset + cipher_position]
                )
        scores[permutation_index] = trigram_score(
            candidate, trigram_logp, unigram_logp
        )
    return scores


def solve_trial(
    trial: stage.TranspositionTrial,
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    block_sizes: tuple[int, ...],
    banks: dict[int, np.ndarray],
) -> dict[str, Any]:
    initial_key = mono.frequency_key(trial.cipher, language)
    decoded = initial_key[np.asarray(trial.cipher, dtype=np.int32)]
    candidates: list[tuple[float, int, tuple[int, ...]]] = []
    true_score = None
    for block_size in block_sizes:
        bank = banks[block_size]
        scores = score_bank(decoded, bank, model[0], model[1])
        for index, score in enumerate(scores):
            permutation = tuple(int(value) for value in bank[index])
            value = float(score)
            candidates.append((value, block_size, permutation))
            if block_size == trial.block_size and permutation == trial.permutation:
                true_score = value
    if true_score is None:
        raise RuntimeError("true candidate missing")
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0]
    best_decoded = stage.invert_blocks(decoded, best[2])
    true_rank = 1 + sum(item[0] > true_score + 1e-9 for item in candidates)
    return {
        "block_size": trial.block_size,
        "replicate": trial.replicate,
        "true_rank": true_rank,
        "true_score": true_score,
        "best_score": best[0],
        "best_block_size": best[1],
        "best_permutation": list(best[2]),
        "best_frequency_accuracy": mono.fast_accuracy(trial.plain, best_decoded),
        "true_in_prefix": {
            str(prefix): true_rank <= prefix for prefix in PREFIXES
        },
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "trials": len(rows),
        "median_true_rank": statistics.median(row["true_rank"] for row in rows),
        "mean_true_rank": statistics.fmean(row["true_rank"] for row in rows),
        "maximum_true_rank": max(row["true_rank"] for row in rows),
        "mean_best_frequency_accuracy": statistics.fmean(
            row["best_frequency_accuracy"] for row in rows
        ),
        "prefix_recall": {
            str(prefix): statistics.fmean(
                float(row["true_in_prefix"][str(prefix)]) for row in rows
            )
            for prefix in PREFIXES
        },
    }
    result["by_block_size"] = {
        str(block_size): summarize_basic(
            [row for row in rows if row["block_size"] == block_size]
        )
        for block_size in sorted({row["block_size"] for row in rows})
    }
    return result


def summarize_basic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "trials": len(rows),
        "median_true_rank": statistics.median(row["true_rank"] for row in rows),
        "mean_true_rank": statistics.fmean(row["true_rank"] for row in rows),
        "maximum_true_rank": max(row["true_rank"] for row in rows),
        "prefix_recall": {
            str(prefix): statistics.fmean(
                float(row["true_in_prefix"][str(prefix)]) for row in rows
            )
            for prefix in PREFIXES
        },
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
    compile_decoded = compile_key[np.asarray(compile_trial.cipher, dtype=np.int32)]
    score_bank(
        compile_decoded,
        banks[compile_trial.block_size][:1],
        model[0],
        model[1],
    )

    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                solve_trial, trial, language, model, block_sizes, banks
            )
            for trial in trials
        ]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            print("V055_SCREEN_TRIAL", json.dumps(row, sort_keys=True), flush=True)
    rows.sort(key=lambda row: (row["block_size"], row["replicate"]))
    summary = summarize(rows)
    payload: dict[str, Any] = {
        "programme": "recoverability-frontier-v0.5.5-stage-b-screening",
        "iso": args.iso,
        "split": args.split,
        "length": args.length,
        "offset": args.offset,
        "replicates_per_block_size": args.replicates,
        "block_sizes": list(block_sizes),
        "screening_key": "train-frequency rank key",
        "summary": summary,
        "rows": rows,
    }
    payload["scientific_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V055_SCREEN_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V055_SCREEN_SHA256", payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
