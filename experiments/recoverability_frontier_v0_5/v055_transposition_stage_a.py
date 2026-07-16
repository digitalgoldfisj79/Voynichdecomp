#!/usr/bin/env python3
"""v0.5.5 component-oracle gates for substitution plus block transposition."""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import itertools
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit

import recoverability_v050 as core
import homophonic_solver_v052 as homophonic
import mono_solver_v051 as mono
import mono_solver_v051_search2 as mono_search
from homophonic_confirm_v052_quadgram import build_quadgram_model

BLOCK_SIZES = (4, 6, 8)


@dataclasses.dataclass
class TranspositionTrial:
    iso: str
    split: str
    length: int
    block_size: int
    replicate: int
    seed: int
    plain: list[int]
    cipher: list[int]
    canonical_to_plain: dict[int, int]
    observed_plain_inventory: tuple[int, ...]
    permutation: tuple[int, ...]


def make_trial(
    language: core.LanguageData,
    split: str,
    length: int,
    block_size: int,
    replicate: int,
) -> TranspositionTrial:
    if length % block_size:
        raise RuntimeError("scored length must be exactly divisible by block size")
    chunks = core.source_chunks(language, split, length)
    if not chunks:
        raise RuntimeError(f"no chunks for {language.iso}/{split}/{length}")
    plain = list(chunks[replicate % len(chunks)])
    seed = core.stable_seed(
        "v055-transposition", language.iso, split, length, block_size, replicate
    )
    rng = random.Random(seed)
    alphabet_size = len(language.alphabet)
    substitution = list(range(alphabet_size))
    rng.shuffle(substitution)
    joint_relabel = list(range(alphabet_size))
    rng.shuffle(joint_relabel)
    permutation = list(range(block_size))
    while True:
        rng.shuffle(permutation)
        if permutation != list(range(block_size)):
            break

    substituted = [joint_relabel[substitution[value]] for value in plain]
    raw_cipher: list[int] = []
    for offset in range(0, len(substituted), block_size):
        block = substituted[offset : offset + block_size]
        raw_cipher.extend(block[index] for index in permutation)
    cipher, canonical_to_raw = homophonic.canonicalize_with_inverse(raw_cipher)
    raw_to_canonical = {raw: canonical for canonical, raw in enumerate(canonical_to_raw)}
    canonical_to_plain: dict[int, int] = {}
    for plain_value, substituted_value in enumerate(substitution):
        raw = joint_relabel[substituted_value]
        if raw in raw_to_canonical:
            canonical_to_plain[raw_to_canonical[raw]] = plain_value

    return TranspositionTrial(
        iso=language.iso,
        split=split,
        length=length,
        block_size=block_size,
        replicate=replicate,
        seed=seed,
        plain=plain,
        cipher=cipher,
        canonical_to_plain=canonical_to_plain,
        observed_plain_inventory=tuple(sorted(canonical_to_plain.values())),
        permutation=tuple(permutation),
    )


def invert_blocks(values: list[int] | np.ndarray, permutation: tuple[int, ...] | np.ndarray) -> list[int]:
    block_size = len(permutation)
    output = [0] * len(values)
    for offset in range(0, len(values), block_size):
        for cipher_position, original_position in enumerate(permutation):
            output[offset + int(original_position)] = int(values[offset + cipher_position])
    return output


@njit(cache=True, nogil=True)
def score_plain(values: np.ndarray, quadgram_logp: np.ndarray, unigram_logp: np.ndarray) -> float:
    length = values.shape[0]
    if length == 0:
        return -1e300
    score = 0.0
    prefix = 3 if length >= 3 else length
    for index in range(prefix):
        score += 0.12 * unigram_logp[values[index]]
    for index in range(3, length):
        score += quadgram_logp[
            values[index - 3], values[index - 2], values[index - 1], values[index]
        ]
        score += 0.12 * unigram_logp[values[index]]
    return score


@njit(cache=True, nogil=True)
def score_permutations(
    transposed_plain: np.ndarray,
    permutations: np.ndarray,
    quadgram_logp: np.ndarray,
    unigram_logp: np.ndarray,
) -> np.ndarray:
    count = permutations.shape[0]
    block_size = permutations.shape[1]
    candidate = np.empty(transposed_plain.shape[0], dtype=np.int32)
    scores = np.empty(count, dtype=np.float64)
    for permutation_index in range(count):
        permutation = permutations[permutation_index]
        for offset in range(0, transposed_plain.shape[0], block_size):
            for cipher_position in range(block_size):
                original_position = int(permutation[cipher_position])
                candidate[offset + original_position] = transposed_plain[
                    offset + cipher_position
                ]
        scores[permutation_index] = score_plain(
            candidate, quadgram_logp, unigram_logp
        )
    return scores


def permutation_bank(block_size: int) -> np.ndarray:
    return np.asarray(list(itertools.permutations(range(block_size))), dtype=np.int16)


def solve_a1(
    trial: TranspositionTrial,
    candidate_sizes: tuple[int, ...],
    banks: dict[int, np.ndarray],
    quadgram: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    started = time.perf_counter()
    decoded_transposed = np.asarray(
        [trial.canonical_to_plain[symbol] for symbol in trial.cipher], dtype=np.int32
    )
    all_candidates: list[tuple[float, int, tuple[int, ...]]] = []
    true_score: float | None = None
    for block_size in candidate_sizes:
        if len(decoded_transposed) % block_size:
            continue
        bank = banks[block_size]
        scores = score_permutations(
            decoded_transposed, bank, quadgram[0], quadgram[1]
        )
        for index, score in enumerate(scores):
            permutation = tuple(int(value) for value in bank[index])
            value = float(score)
            all_candidates.append((value, block_size, permutation))
            if block_size == trial.block_size and permutation == trial.permutation:
                true_score = value
    if true_score is None:
        raise RuntimeError("true transposition candidate absent from enumeration")
    all_candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, selected_size, selected_permutation = all_candidates[0]
    selected_plain = invert_blocks(decoded_transposed, selected_permutation)
    true_rank = 1 + sum(candidate[0] > true_score for candidate in all_candidates)
    second_score = all_candidates[1][0] if len(all_candidates) > 1 else best_score
    return {
        "block_size": trial.block_size,
        "replicate": trial.replicate,
        "selected_block_size": selected_size,
        "selected_permutation": list(selected_permutation),
        "true_permutation": list(trial.permutation),
        "block_size_correct": selected_size == trial.block_size,
        "permutation_correct": (
            selected_size == trial.block_size
            and selected_permutation == trial.permutation
        ),
        "accuracy": mono.fast_accuracy(trial.plain, selected_plain),
        "true_candidate_rank": true_rank,
        "true_score": true_score,
        "best_score": best_score,
        "best_minus_second_score": best_score - second_score,
        "best_minus_true_score": best_score - true_score,
        "candidate_count": len(all_candidates),
        "elapsed_seconds": time.perf_counter() - started,
    }


def frequency_initial_key(
    detransposed_cipher: list[int],
    inventory: tuple[int, ...],
    language: core.LanguageData,
) -> np.ndarray:
    symbol_count = max(detransposed_cipher, default=-1) + 1
    counts = np.bincount(
        np.asarray(detransposed_cipher, dtype=np.int32), minlength=symbol_count
    )
    cipher_rank = np.argsort(-counts, kind="stable")
    inventory_rank = sorted(
        inventory, key=lambda value: (-language.probabilities[value], value)
    )
    if len(inventory_rank) != symbol_count:
        raise RuntimeError("observed inventory and symbol count differ")
    key = np.empty(symbol_count, dtype=np.int32)
    for symbol, label in zip(cipher_rank, inventory_rank):
        key[int(symbol)] = int(label)
    return key


def solve_a2(
    trial: TranspositionTrial,
    language: core.LanguageData,
    quadgram: tuple[np.ndarray, np.ndarray],
    iterations: int,
    restarts: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    detransposed = invert_blocks(trial.cipher, trial.permutation)
    cipher_array = np.asarray(detransposed, dtype=np.int32)
    initial_key = frequency_initial_key(
        detransposed, trial.observed_plain_inventory, language
    )
    baseline = initial_key[cipher_array].tolist()
    solved_key, score = mono_search.anneal_mono_search2(
        cipher_array,
        initial_key,
        quadgram[0],
        quadgram[1],
        iterations,
        restarts,
        int(core.stable_seed("v055-a2", trial.seed) & 0x7FFFFFFFFFFFFFFF),
    )
    prediction = solved_key[cipher_array].tolist()
    true_key = np.empty(len(trial.canonical_to_plain), dtype=np.int32)
    for symbol, value in trial.canonical_to_plain.items():
        true_key[int(symbol)] = int(value)
    return {
        "block_size": trial.block_size,
        "replicate": trial.replicate,
        "baseline_accuracy": mono.fast_accuracy(trial.plain, baseline),
        "accuracy": mono.fast_accuracy(trial.plain, prediction),
        "exact": prediction == trial.plain,
        "key_accuracy": statistics.fmean(
            int(solved_key[index]) == int(true_key[index])
            for index in range(len(true_key))
        ),
        "score": float(score),
        "elapsed_seconds": time.perf_counter() - started,
    }


def subset_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    accuracies = [float(row["accuracy"]) for row in rows]
    result: dict[str, Any] = {
        "trials": len(rows),
        "mean_accuracy": statistics.fmean(accuracies),
        "median_accuracy": statistics.median(accuracies),
        "minimum_accuracy": min(accuracies),
        "at_least_90_rate": statistics.fmean(value >= 0.90 for value in accuracies),
        "mean_seconds": statistics.fmean(float(row["elapsed_seconds"]) for row in rows),
    }
    if arm == "a1":
        result.update(
            {
                "block_size_accuracy": statistics.fmean(
                    float(row["block_size_correct"]) for row in rows
                ),
                "permutation_accuracy": statistics.fmean(
                    float(row["permutation_correct"]) for row in rows
                ),
                "mean_true_candidate_rank": statistics.fmean(
                    float(row["true_candidate_rank"]) for row in rows
                ),
                "maximum_true_candidate_rank": max(
                    int(row["true_candidate_rank"]) for row in rows
                ),
            }
        )
    else:
        result.update(
            {
                "mean_baseline_accuracy": statistics.fmean(
                    float(row["baseline_accuracy"]) for row in rows
                ),
                "mean_key_accuracy": statistics.fmean(
                    float(row["key_accuracy"]) for row in rows
                ),
                "exact_rate": statistics.fmean(float(row["exact"]) for row in rows),
            }
        )
    return result


def summarize(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    result = subset_summary(rows, arm)
    result["by_block_size"] = {
        str(block_size): subset_summary(
            [row for row in rows if int(row["block_size"]) == block_size], arm
        )
        for block_size in sorted({int(row["block_size"]) for row in rows})
    }
    return result


def canonical_sha(payload: dict[str, Any]) -> str:
    blob = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
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
    parser.add_argument("--a2-iterations", type=int, default=700000)
    parser.add_argument("--a2-restarts", type=int, default=50)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    candidate_sizes = tuple(
        int(value) for value in args.block_sizes.split(",") if value
    )
    if any(args.length % value for value in candidate_sizes):
        raise RuntimeError("all candidate block sizes must divide scored length")
    experiment = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        experiment / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v050_corpora",
    )
    language = languages[args.iso]
    quadgram = build_quadgram_model(language)
    banks = {block_size: permutation_bank(block_size) for block_size in candidate_sizes}
    trials = [
        make_trial(
            language,
            args.split,
            args.length,
            block_size,
            args.offset + replicate,
        )
        for block_size in candidate_sizes
        for replicate in range(args.replicates)
    ]

    # Compile both Numba kernels before concurrency.
    compile_trial = trials[0]
    decoded = np.asarray(
        [compile_trial.canonical_to_plain[symbol] for symbol in compile_trial.cipher],
        dtype=np.int32,
    )
    score_permutations(decoded, banks[compile_trial.block_size][:1], quadgram[0], quadgram[1])
    detransposed = invert_blocks(compile_trial.cipher, compile_trial.permutation)
    compile_initial = frequency_initial_key(
        detransposed, compile_trial.observed_plain_inventory, language
    )
    mono_search.anneal_mono_search2(
        np.asarray(detransposed, dtype=np.int32),
        compile_initial,
        quadgram[0],
        quadgram[1],
        2,
        1,
        1,
    )

    a1_rows: list[dict[str, Any]] = []
    a2_rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        a1_futures = [
            executor.submit(
                solve_a1, trial, candidate_sizes, banks, quadgram
            )
            for trial in trials
        ]
        a2_futures = [
            executor.submit(
                solve_a2,
                trial,
                language,
                quadgram,
                args.a2_iterations,
                args.a2_restarts,
            )
            for trial in trials
        ]
        for future in concurrent.futures.as_completed(a1_futures):
            row = future.result()
            a1_rows.append(row)
            print("V055_A1_TRIAL", json.dumps(row, sort_keys=True), flush=True)
        for future in concurrent.futures.as_completed(a2_futures):
            row = future.result()
            a2_rows.append(row)
            print("V055_A2_TRIAL", json.dumps(row, sort_keys=True), flush=True)
    a1_rows.sort(key=lambda row: (row["block_size"], row["replicate"]))
    a2_rows.sort(key=lambda row: (row["block_size"], row["replicate"]))
    a1_summary = summarize(a1_rows, "a1")
    a2_summary = summarize(a2_rows, "a2")

    gate_by_size: dict[str, dict[str, bool]] = {}
    for block_size in candidate_sizes:
        a1 = a1_summary["by_block_size"][str(block_size)]
        a2 = a2_summary["by_block_size"][str(block_size)]
        gate_by_size[str(block_size)] = {
            "a1_mean_99_pass": a1["mean_accuracy"] >= 0.99,
            "a1_block_size_8_of_8_pass": a1["block_size_accuracy"] == 1.0,
            "a1_permutation_7_of_8_pass": a1["permutation_accuracy"] >= 0.875,
            "a2_mean_90_pass": a2["mean_accuracy"] >= 0.90,
            "a2_median_99_pass": a2["median_accuracy"] >= 0.99,
            "a2_7_of_8_90_pass": a2["at_least_90_rate"] >= 0.875,
        }
        gate_by_size[str(block_size)]["pass"] = all(
            gate_by_size[str(block_size)].values()
        )
    gate = {
        "by_block_size": gate_by_size,
        "pass": all(item["pass"] for item in gate_by_size.values()),
    }
    payload: dict[str, Any] = {
        "programme": "recoverability-frontier-v0.5.5-transposition-stage-a",
        "iso": args.iso,
        "split": args.split,
        "length": args.length,
        "offset": args.offset,
        "replicates_per_block_size": args.replicates,
        "candidate_block_sizes": list(candidate_sizes),
        "generator": {
            "fresh_substitution_key": True,
            "fresh_non_identity_block_permutation": True,
            "joint_surface_relabelling": True,
            "first_occurrence_canonicalisation": True,
            "padding": False,
        },
        "a2_schedule": {
            "iterations": args.a2_iterations,
            "restarts": args.a2_restarts,
        },
        "a1_summary": a1_summary,
        "a2_summary": a2_summary,
        "gate": gate,
        "a1_rows": a1_rows,
        "a2_rows": a2_rows,
    }
    payload["scientific_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V055_A1_SUMMARY", json.dumps(a1_summary, sort_keys=True), flush=True)
    print("V055_A2_SUMMARY", json.dumps(a2_summary, sort_keys=True), flush=True)
    print("V055_GATE", json.dumps(gate, sort_keys=True), flush=True)
    print("V055_SHA256", payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
