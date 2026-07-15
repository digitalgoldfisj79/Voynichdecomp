#!/usr/bin/env python3
"""CrypTool-style homophonic hill climber for the v0.5.2 benchmark.

Search architecture ported from CrypTool 2 HomophonicSubstitutionAnalyzer,
commit d7d754af55c167941bec7fb56e965f309d050a12.

Original relevant sources are Apache-2.0 licensed:
- HillClimber.cs, copyright 2020 Nils Kopal
- SimulatedAnnealing.cs, copyright 2020 Nils Kopal; algorithms by George Lasry

This Python implementation is a benchmark-specific reimplementation. It adds
incremental quadgram scoring, deterministic seeds, recurrence-canonical input,
and train/dev/test provenance controls.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit

import recoverability_v050 as core
import homophonic_solver_v052 as homophonic
import mono_solver_v051 as mono
from homophonic_confirm_v052_quadgram import build_quadgram_model, quadgram_score_key

CRYPTOOL_COMMIT = "d7d754af55c167941bec7fb56e965f309d050a12"
ACCEPTANCE_FLOOR = 0.0085
UNIGRAM_WEIGHT = 0.12


@njit(cache=True, nogil=True)
def rng_step(state: np.uint64) -> np.uint64:
    state ^= state >> np.uint64(12)
    state ^= state << np.uint64(25)
    state ^= state >> np.uint64(27)
    return state * np.uint64(2685821657736338717)


@njit(cache=True, nogil=True)
def rng_int(state: np.uint64, upper: int) -> tuple[np.uint64, int]:
    state = rng_step(state)
    return state, int(state % np.uint64(upper))


@njit(cache=True, nogil=True)
def rng_float(state: np.uint64) -> tuple[np.uint64, float]:
    state = rng_step(state)
    return state, float(state >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True, nogil=True)
def sample_cdf(state: np.uint64, cdf: np.ndarray) -> tuple[np.uint64, int]:
    state, value = rng_float(state)
    for index in range(cdf.shape[0]):
        if value <= cdf[index]:
            return state, index
    return state, cdf.shape[0] - 1


@njit(cache=True, nogil=True)
def full_score(cipher, key, quadgram_logp, unigram_logp):
    return quadgram_score_key(cipher, key, quadgram_logp, unigram_logp)


@njit(cache=True, nogil=True)
def swap_delta_apply(
    cipher: np.ndarray,
    key: np.ndarray,
    first_symbol: int,
    second_symbol: int,
    quadgram_logp: np.ndarray,
    unigram_logp: np.ndarray,
    positions: np.ndarray,
    offsets: np.ndarray,
    endpoint_marks: np.ndarray,
    mark_id: int,
    endpoint_buffer: np.ndarray,
) -> tuple[float, int]:
    first_label = int(key[first_symbol])
    second_label = int(key[second_symbol])
    if first_label == second_label:
        return 0.0, mark_id

    old_score = 0.0
    new_score = 0.0
    endpoint_count = 0
    mark_id += 1
    if mark_id == 2_000_000_000:
        for index in range(endpoint_marks.shape[0]):
            endpoint_marks[index] = 0
        mark_id = 1

    for symbol in (first_symbol, second_symbol):
        replacement_label = second_label if symbol == first_symbol else first_label
        for flat_index in range(int(offsets[symbol]), int(offsets[symbol + 1])):
            position = int(positions[flat_index])
            old_score += UNIGRAM_WEIGHT * unigram_logp[int(key[symbol])]
            new_score += UNIGRAM_WEIGHT * unigram_logp[replacement_label]
            start = 3 if position < 3 else position
            end = position + 3
            if end >= cipher.shape[0]:
                end = cipher.shape[0] - 1
            for endpoint in range(start, end + 1):
                if endpoint_marks[endpoint] != mark_id:
                    endpoint_marks[endpoint] = mark_id
                    endpoint_buffer[endpoint_count] = endpoint
                    endpoint_count += 1

    for buffer_index in range(endpoint_count):
        endpoint = int(endpoint_buffer[buffer_index])
        a_symbol = int(cipher[endpoint - 3])
        b_symbol = int(cipher[endpoint - 2])
        c_symbol = int(cipher[endpoint - 1])
        d_symbol = int(cipher[endpoint])
        old_score += quadgram_logp[
            int(key[a_symbol]), int(key[b_symbol]), int(key[c_symbol]), int(key[d_symbol])
        ]

    temporary = key[first_symbol]
    key[first_symbol] = key[second_symbol]
    key[second_symbol] = temporary

    for buffer_index in range(endpoint_count):
        endpoint = int(endpoint_buffer[buffer_index])
        new_score += quadgram_logp[
            int(key[int(cipher[endpoint - 3])]),
            int(key[int(cipher[endpoint - 2])]),
            int(key[int(cipher[endpoint - 1])]),
            int(key[int(cipher[endpoint])]),
        ]

    return new_score - old_score, mark_id


@njit(cache=True, nogil=True)
def estimate_start_temperature(
    cipher,
    key,
    quadgram_logp,
    unigram_logp,
    positions,
    offsets,
    target_acceptance: float,
) -> float:
    key_length = key.shape[0]
    negative = np.empty(512, dtype=np.float64)
    negative_count = 0
    endpoint_marks = np.zeros(cipher.shape[0], dtype=np.int32)
    endpoint_buffer = np.empty(cipher.shape[0], dtype=np.int32)
    mark_id = 0
    examined = 0
    for first in range(key_length - 1):
        for second in range(first + 1, key_length):
            if key[first] == key[second]:
                continue
            delta, mark_id = swap_delta_apply(
                cipher, key, first, second, quadgram_logp, unigram_logp,
                positions, offsets, endpoint_marks, mark_id, endpoint_buffer,
            )
            temporary = key[first]
            key[first] = key[second]
            key[second] = temporary
            if delta < 0.0 and negative_count < negative.shape[0]:
                negative[negative_count] = -delta
                negative_count += 1
            examined += 1
            if examined >= 512:
                break
        if examined >= 512:
            break
    if negative_count == 0:
        return 1.0
    values = np.sort(negative[:negative_count])
    median = values[negative_count // 2]
    target = target_acceptance
    if target < 1e-6:
        target = 1e-6
    if target > 0.95:
        target = 0.95
    temperature = median / (-math.log(target))
    return max(temperature, 1e-6)


@njit(cache=True, nogil=True)
def distributor_key(
    state: np.uint64,
    key_length: int,
    min_counts: np.ndarray,
    max_counts: np.ndarray,
) -> tuple[np.uint64, np.ndarray]:
    alphabet_size = min_counts.shape[0]
    distribution = np.zeros(alphabet_size, dtype=np.int32)
    key = np.empty(key_length, dtype=np.int32)
    for key_index in range(key_length):
        state, start = rng_int(state, alphabet_size)
        selected = -1
        for offset in range(alphabet_size):
            label = (start + offset) % alphabet_size
            if distribution[label] < min_counts[label]:
                selected = label
                break
        if selected < 0:
            state, start = rng_int(state, alphabet_size)
            for offset in range(alphabet_size):
                label = (start + offset) % alphabet_size
                if distribution[label] < max_counts[label]:
                    selected = label
                    break
        if selected < 0:
            state, selected = rng_int(state, alphabet_size)
        key[key_index] = selected
        distribution[selected] += 1

    for index in range(key_length - 1, 0, -1):
        state, other = rng_int(state, index + 1)
        temporary = key[index]
        key[index] = key[other]
        key[other] = temporary
    return state, key


@njit(cache=True, nogil=True)
def mutate_rare_symbols(
    state: np.uint64,
    key: np.ndarray,
    rare_order: np.ndarray,
    mutation_event: int,
    mutation_count: int,
    max_counts: np.ndarray,
    proposal_cdf: np.ndarray,
) -> np.uint64:
    alphabet_size = max_counts.shape[0]
    counts = np.zeros(alphabet_size, dtype=np.int32)
    for value in key:
        counts[int(value)] += 1

    for mutation_offset in range(mutation_count):
        symbol = int(rare_order[(mutation_event * mutation_count + mutation_offset) % rare_order.shape[0]])
        old_label = int(key[symbol])
        counts[old_label] -= 1
        selected = old_label
        for _ in range(100):
            state, proposal = sample_cdf(state, proposal_cdf)
            if proposal != old_label and counts[proposal] < max_counts[proposal]:
                selected = proposal
                break
        key[symbol] = selected
        counts[selected] += 1
    return state


@njit(cache=True, nogil=True)
def cryptool_style_single_run(
    cipher: np.ndarray,
    start_key: np.ndarray,
    quadgram_logp: np.ndarray,
    unigram_logp: np.ndarray,
    positions: np.ndarray,
    offsets: np.ndarray,
    rare_order: np.ndarray,
    max_counts: np.ndarray,
    proposal_cdf: np.ndarray,
    steps: int,
    target_acceptance: float,
    stagnation_sweeps: int,
    mutation_count: int,
    seed: int,
) -> tuple[np.ndarray, float, float, int]:
    key = start_key.copy()
    current_score = full_score(cipher, key, quadgram_logp, unigram_logp)
    best_key = key.copy()
    best_score = current_score
    start_temperature = estimate_start_temperature(
        cipher, key, quadgram_logp, unigram_logp, positions, offsets, target_acceptance
    )
    state = np.uint64(seed if seed > 0 else 1)
    endpoint_marks = np.zeros(cipher.shape[0], dtype=np.int32)
    endpoint_buffer = np.empty(cipher.shape[0], dtype=np.int32)
    mark_id = 0
    proposals = 0
    no_global_best_sweeps = 0
    mutation_event = 0
    key_length = key.shape[0]

    while proposals < steps:
        global_improved = False
        for first in range(key_length - 1):
            for second in range(first + 1, key_length):
                if key[first] == key[second]:
                    continue
                delta, mark_id = swap_delta_apply(
                    cipher, key, first, second, quadgram_logp, unigram_logp,
                    positions, offsets, endpoint_marks, mark_id, endpoint_buffer,
                )
                proposals += 1
                remaining = 1.0 - proposals / max(1.0, float(steps))
                temperature = start_temperature * max(remaining, 1e-9)
                accept = delta >= 0.0
                if not accept:
                    probability = math.exp(delta / temperature)
                    if probability > ACCEPTANCE_FLOOR:
                        state, draw = rng_float(state)
                        accept = draw < probability
                if accept:
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_key = key.copy()
                        global_improved = True
                else:
                    temporary = key[first]
                    key[first] = key[second]
                    key[second] = temporary
                if proposals >= steps:
                    break
            if proposals >= steps:
                break

        if global_improved:
            no_global_best_sweeps = 0
        else:
            no_global_best_sweeps += 1
            if no_global_best_sweeps >= stagnation_sweeps:
                state = mutate_rare_symbols(
                    state, key, rare_order, mutation_event, mutation_count,
                    max_counts, proposal_cdf,
                )
                mutation_event += 1
                current_score = full_score(cipher, key, quadgram_logp, unigram_logp)
                if current_score > best_score:
                    best_score = current_score
                    best_key = key.copy()
                no_global_best_sweeps = 0

    return best_key, best_score, start_temperature, mutation_event


@njit(cache=True, nogil=True)
def cryptool_style_restarts(
    cipher,
    initial_key,
    quadgram_logp,
    unigram_logp,
    positions,
    offsets,
    rare_order,
    min_counts,
    max_counts,
    proposal_cdf,
    steps,
    restarts,
    target_acceptance,
    stagnation_sweeps,
    mutation_count,
    seed,
):
    best_key = initial_key.copy()
    best_score = full_score(cipher, best_key, quadgram_logp, unigram_logp)
    temperature_sum = 0.0
    mutation_sum = 0
    state = np.uint64(seed if seed > 0 else 1)

    for restart in range(restarts):
        if restart == 0:
            start_key = initial_key.copy()
        else:
            state, start_key = distributor_key(
                state, initial_key.shape[0], min_counts, max_counts
            )
        run_key, run_score, run_temperature, run_mutations = cryptool_style_single_run(
            cipher,
            start_key,
            quadgram_logp,
            unigram_logp,
            positions,
            offsets,
            rare_order,
            max_counts,
            proposal_cdf,
            steps,
            target_acceptance,
            stagnation_sweeps,
            mutation_count,
            int((state + np.uint64(restart + 1) * np.uint64(104729)) & np.uint64(0x7FFFFFFFFFFFFFFF)),
        )
        temperature_sum += run_temperature
        mutation_sum += run_mutations
        if run_score > best_score:
            best_score = run_score
            best_key = run_key.copy()
        state = rng_step(state)

    return best_key, best_score, temperature_sum / max(1, restarts), mutation_sum


def build_positions(cipher: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(cipher, dtype=np.int32)
    key_length = int(values.max()) + 1
    counts = np.bincount(values, minlength=key_length).astype(np.int32)
    offsets = np.zeros(key_length + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts)
    positions = np.empty(len(values), dtype=np.int32)
    cursor = offsets[:-1].copy()
    for position, symbol_value in enumerate(values):
        symbol = int(symbol_value)
        positions[int(cursor[symbol])] = position
        cursor[symbol] += 1
    rare_order = np.argsort(counts, kind="stable").astype(np.int32)
    return positions, offsets, rare_order


def distribution_arrays(language: core.LanguageData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.asarray(language.probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    max_counts = np.asarray(
        [homophonic.multiplicity(float(probability)) for probability in probabilities],
        dtype=np.int32,
    )
    alphabet_size = len(language.alphabet)
    min_counts = np.asarray(
        [min(int(max_counts[index]), max(1, int(math.ceil(probabilities[index] * alphabet_size))))
         for index in range(alphabet_size)],
        dtype=np.int32,
    )
    proposal_cdf = np.cumsum(probabilities)
    proposal_cdf[-1] = 1.0
    return min_counts, max_counts, proposal_cdf


def solve_trial(
    trial: dict[str, Any],
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    steps: int,
    restarts: int,
    target_acceptance: float,
) -> dict[str, Any]:
    cipher_list = list(map(int, trial["cipher"]))
    cipher = np.asarray(cipher_list, dtype=np.int32)
    initial_key = homophonic.frequency_slot_key(
        cipher_list, trial["inferred_labels"], trial["expected_slot_probabilities"]
    )
    positions, offsets, rare_order = build_positions(cipher_list)
    min_counts, max_counts, proposal_cdf = distribution_arrays(language)
    key, score, mean_temperature, mutations = cryptool_style_restarts(
        cipher,
        initial_key,
        model[0],
        model[1],
        positions,
        offsets,
        rare_order,
        min_counts,
        max_counts,
        proposal_cdf,
        steps,
        restarts,
        target_acceptance,
        50,
        3,
        int(trial["seed"] & 0x7FFFFFFFFFFFFFFF),
    )
    prediction = key[cipher].tolist()
    baseline = initial_key[cipher].tolist()
    return {
        "replicate": int(trial["replicate"]),
        "accuracy": mono.fast_accuracy(trial["plain"], prediction),
        "baseline_accuracy": mono.fast_accuracy(trial["plain"], baseline),
        "exact": prediction == trial["plain"],
        "initial_inventory_overlap": float(trial["inventory_overlap"]),
        "final_inventory_overlap": homophonic.multiset_overlap(key, trial["true_labels"]),
        "score": float(score),
        "mean_start_temperature": float(mean_temperature),
        "total_mutation_events": int(mutations),
        "steps_per_restart": steps,
        "restarts": restarts,
        "target_acceptance": target_acceptance,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "trials": len(rows),
        "mean_accuracy": statistics.fmean(row["accuracy"] for row in rows),
        "median_accuracy": statistics.median(row["accuracy"] for row in rows),
        "baseline_mean_accuracy": statistics.fmean(row["baseline_accuracy"] for row in rows),
        "exact_rate": statistics.fmean(float(row["exact"]) for row in rows),
        "mean_initial_inventory_overlap": statistics.fmean(row["initial_inventory_overlap"] for row in rows),
        "mean_final_inventory_overlap": statistics.fmean(row["final_inventory_overlap"] for row in rows),
        "mean_start_temperature": statistics.fmean(row["mean_start_temperature"] for row in rows),
        "mean_mutation_events": statistics.fmean(row["total_mutation_events"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iso", default="en")
    parser.add_argument("--split", default="dev", choices=("dev", "test"))
    parser.add_argument("--length", type=int, default=384)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--restarts", type=int, required=True)
    parser.add_argument("--target-acceptance", type=float, required=True)
    args = parser.parse_args()

    experiment = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        experiment / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v050_corpora",
    )
    language = languages[args.iso]
    model = build_quadgram_model(language)
    trials = [
        homophonic.make_trial(language, args.split, args.length, args.offset + replicate)
        for replicate in range(args.replicates)
    ]

    # Compile all kernels on a small real-shaped input before parallel execution.
    compile_trial = trials[0]
    compile_cipher = list(map(int, compile_trial["cipher"]))
    compile_initial = homophonic.frequency_slot_key(
        compile_cipher,
        compile_trial["inferred_labels"],
        compile_trial["expected_slot_probabilities"],
    )
    compile_positions, compile_offsets, compile_rare = build_positions(compile_cipher)
    compile_min, compile_max, compile_cdf = distribution_arrays(language)
    cryptool_style_restarts(
        np.asarray(compile_cipher, dtype=np.int32),
        compile_initial,
        model[0], model[1],
        compile_positions, compile_offsets, compile_rare,
        compile_min, compile_max, compile_cdf,
        10, 1, args.target_acceptance, 2, 1, 1,
    )

    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                solve_trial,
                trial,
                language,
                model,
                args.steps,
                args.restarts,
                args.target_acceptance,
            )
            for trial in trials
        ]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 2 == 0 or completed == len(futures):
                print(
                    f"V052_CRYPTTOOL_PROGRESS {args.iso} {completed}/{len(futures)} "
                    f"steps={args.steps} restarts={args.restarts} acceptance={args.target_acceptance}",
                    flush=True,
                )
    rows.sort(key=lambda row: row["replicate"])
    summary = summarize(rows)
    gate = {"english_70_percent_pass": summary["mean_accuracy"] >= 0.70}
    gate["pass"] = all(gate.values())
    payload = {
        "programme": "v0.5.2-cryptool-style-homophonic-port",
        "source_project": "CrypToolProject/CrypTool-2",
        "source_commit": CRYPTOOL_COMMIT,
        "source_license": "Apache-2.0",
        "iso": args.iso,
        "split": args.split,
        "length": args.length,
        "offset": args.offset,
        "replicates": args.replicates,
        "schedule": {
            "steps_per_restart": args.steps,
            "restarts": args.restarts,
            "target_initial_acceptance": args.target_acceptance,
            "stagnation_sweeps": 50,
            "rare_symbol_mutations": 3,
            "acceptance_probability_floor": ACCEPTANCE_FLOOR,
        },
        "summary": summary,
        "gate": gate,
        "rows": rows,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["scientific_sha256"] = hashlib.sha256(blob).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V052_CRYPTTOOL_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V052_CRYPTTOOL_GATE", json.dumps(gate, sort_keys=True), flush=True)
    print("V052_CRYPTTOOL_SHA256", payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
