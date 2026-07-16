#!/usr/bin/env python3
"""Final development-only stratified coordinate search for v0.5.5."""
from __future__ import annotations

import random
import statistics
import time
from typing import Any

import numpy as np

import recoverability_v050 as core
import mono_solver_v051 as mono
import mono_solver_v051_search2 as mono_search
import v055_stage_b_coordinate as base
import v055_transposition_stage_a as stage


def solve_trial_stratified(
    trial: stage.TranspositionTrial,
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    block_sizes: tuple[int, ...],
    banks: dict[int, np.ndarray],
    catalog: list[base.Candidate],
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
    ranked_frequency = base.all_candidate_scores(
        decoded_frequency, block_sizes, banks, model
    )

    seeds: list[base.Candidate] = []
    rng = random.Random(
        core.stable_seed("v055-coordinate-stratified-random-seeds", trial.seed)
    )
    for block_size in block_sizes:
        ranked_for_size = [
            candidate
            for _score, candidate in ranked_frequency
            if candidate[0] == block_size
        ]
        seeds.extend(ranked_for_size[:top_seeds])
        catalog_for_size = [candidate for candidate in catalog if candidate[0] == block_size]
        for candidate in rng.sample(
            catalog_for_size, min(random_seeds, len(catalog_for_size))
        ):
            if candidate not in seeds:
                seeds.append(candidate)

    seed_results = [
        base.solve_seed(
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
    deduplicated: dict[base.Candidate, dict[str, Any]] = {}
    for item in seed_results:
        candidate = item["candidate"]
        incumbent = deduplicated.get(candidate)
        if incumbent is None or item["score"] > incumbent["score"]:
            deduplicated[candidate] = item
    best_seed = max(deduplicated.values(), key=lambda item: item["score"])
    current_candidate: base.Candidate = best_seed["candidate"]
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
        int(
            core.stable_seed("v055-coordinate-stratified-full-1", trial.seed)
            & 0x7FFFFFFFFFFFFFFF
        ),
    )
    decoded_transposed = current_key[cipher_array]
    ranked_final = base.all_candidate_scores(
        decoded_transposed, block_sizes, banks, model
    )
    _reenumerated_score, reenumerated_candidate = ranked_final[0]
    changed_after_full = reenumerated_candidate != current_candidate
    current_candidate = reenumerated_candidate
    if changed_after_full:
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
            int(
                core.stable_seed("v055-coordinate-stratified-full-2", trial.seed)
                & 0x7FFFFFFFFFFFFFFF
            ),
        )
    else:
        detransposed = np.asarray(
            stage.invert_blocks(trial.cipher, current_candidate[1]), dtype=np.int32
        )

    prediction = current_key[detransposed].tolist()
    true_frequency_score = next(
        value
        for value, candidate in ranked_frequency
        if candidate == (trial.block_size, trial.permutation)
    )
    true_frequency_rank = 1 + sum(
        value > true_frequency_score + 1e-9
        for value, _candidate in ranked_frequency
    )
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
        "top_frequency_true_rank": true_frequency_rank,
        "seed_count": len(seeds),
        "converged_candidate_count": len(deduplicated),
        "best_seed_score": float(best_seed["score"]),
        "final_score": float(current_score),
        "changed_after_full_refinement": changed_after_full,
        "best_seed_trajectory": best_seed["trajectory"],
        "elapsed_seconds": time.perf_counter() - started,
    }


base.solve_trial = solve_trial_stratified
base.main()
