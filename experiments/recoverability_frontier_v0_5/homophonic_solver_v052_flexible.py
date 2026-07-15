#!/usr/bin/env python3
"""Flexible bounded-inventory homophonic solver for v0.5.2."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from numba import njit

import recoverability_v050 as core
import homophonic_solver_v052 as fixed
import mono_solver_v051 as mono


@njit(cache=True, nogil=True)
def _rng_step(state: np.uint64) -> np.uint64:
    state ^= state >> np.uint64(12)
    state ^= state << np.uint64(25)
    state ^= state >> np.uint64(27)
    return state * np.uint64(2685821657736338717)


@njit(cache=True, nogil=True)
def _rng_int(state: np.uint64, upper: int) -> tuple[np.uint64, int]:
    state = _rng_step(state)
    return state, int(state % np.uint64(upper))


@njit(cache=True, nogil=True)
def _rng_float(state: np.uint64) -> tuple[np.uint64, float]:
    state = _rng_step(state)
    return state, float(state >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True, nogil=True)
def _sample_cdf(state: np.uint64, cdf: np.ndarray) -> tuple[np.uint64, int]:
    state, value = _rng_float(state)
    for index in range(cdf.shape[0]):
        if value <= cdf[index]:
            return state, index
    return state, cdf.shape[0] - 1


@njit(cache=True, nogil=True)
def _counts_for_key(key: np.ndarray, alphabet_size: int) -> np.ndarray:
    counts = np.zeros(alphabet_size, dtype=np.int32)
    for value in key:
        counts[int(value)] += 1
    return counts


@njit(cache=True, nogil=True)
def _random_bounded_key(
    state: np.uint64,
    slot_pool: np.ndarray,
    key_length: int,
) -> tuple[np.uint64, np.ndarray]:
    pool = slot_pool.copy()
    pool_length = pool.shape[0]
    key = np.empty(key_length, dtype=np.int32)
    for index in range(key_length):
        state, offset = _rng_int(state, pool_length - index)
        selected = index + offset
        temporary = pool[index]
        pool[index] = pool[selected]
        pool[selected] = temporary
        key[index] = pool[index]
    for index in range(key_length - 1, 0, -1):
        state, other = _rng_int(state, index + 1)
        temporary = key[index]
        key[index] = key[other]
        key[other] = temporary
    return state, key


@njit(cache=True, nogil=True)
def flexible_homophonic_search(
    cipher: np.ndarray,
    initial_key: np.ndarray,
    trigram_logp: np.ndarray,
    unigram_logp: np.ndarray,
    slot_pool: np.ndarray,
    max_counts: np.ndarray,
    proposal_cdf: np.ndarray,
    iterations: int,
    restarts: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    key_length = initial_key.shape[0]
    alphabet_size = max_counts.shape[0]
    best_key = initial_key.copy()
    best_score = mono.score_key(cipher, best_key, trigram_logp, unigram_logp)
    state = np.uint64(seed if seed > 0 else 1)

    for restart in range(restarts):
        if restart == 0:
            key = initial_key.copy()
        elif restart % 6 == 0:
            state, key = _random_bounded_key(state, slot_pool, key_length)
        elif restart % 2 == 0:
            key = best_key.copy()
        else:
            key = initial_key.copy()

        counts = _counts_for_key(key, alphabet_size)
        perturbations = 5 + (restart % 17) * 3
        for _ in range(perturbations):
            state, move_draw = _rng_float(state)
            if move_draw < 0.60:
                state, first = _rng_int(state, key_length)
                state, second = _rng_int(state, key_length)
                if first != second:
                    temporary = key[first]
                    key[first] = key[second]
                    key[second] = temporary
            else:
                state, first = _rng_int(state, key_length)
                state, proposal_draw = _rng_float(state)
                if proposal_draw < 0.85:
                    state, new_label = _sample_cdf(state, proposal_cdf)
                else:
                    state, new_label = _rng_int(state, alphabet_size)
                old_label = int(key[first])
                if new_label != old_label and counts[new_label] < max_counts[new_label]:
                    key[first] = new_label
                    counts[old_label] -= 1
                    counts[new_label] += 1

        counts = _counts_for_key(key, alphabet_size)
        current_score = mono.score_key(cipher, key, trigram_logp, unigram_logp)
        if current_score > best_score:
            best_score = current_score
            best_key = key.copy()

        temperature = 35.0
        cooling = math.exp(math.log(0.025 / 35.0) / max(1, iterations))
        stagnant = 0

        for _ in range(iterations):
            state, move_draw = _rng_float(state)
            changed = False
            move_type = 0
            first = 0
            second = 0
            old_label = 0
            new_label = 0

            if move_draw < 0.72:
                move_type = 1
                state, first = _rng_int(state, key_length)
                state, second = _rng_int(state, key_length)
                if first != second and key[first] != key[second]:
                    temporary = key[first]
                    key[first] = key[second]
                    key[second] = temporary
                    changed = True
            else:
                move_type = 2
                state, first = _rng_int(state, key_length)
                state, proposal_draw = _rng_float(state)
                if proposal_draw < 0.85:
                    state, new_label = _sample_cdf(state, proposal_cdf)
                else:
                    state, new_label = _rng_int(state, alphabet_size)
                old_label = int(key[first])
                if new_label != old_label and counts[new_label] < max_counts[new_label]:
                    key[first] = new_label
                    counts[old_label] -= 1
                    counts[new_label] += 1
                    changed = True

            if not changed:
                temperature *= cooling
                continue

            candidate_score = mono.score_key(cipher, key, trigram_logp, unigram_logp)
            delta = candidate_score - current_score
            accept = delta >= 0.0
            if not accept:
                state, uniform = _rng_float(state)
                accept = uniform < math.exp(delta / max(temperature, 1e-12))

            if accept:
                current_score = candidate_score
                if current_score > best_score:
                    best_score = current_score
                    best_key = key.copy()
                    stagnant = 0
                else:
                    stagnant += 1
            else:
                if move_type == 1:
                    temporary = key[first]
                    key[first] = key[second]
                    key[second] = temporary
                else:
                    key[first] = old_label
                    counts[new_label] -= 1
                    counts[old_label] += 1
                stagnant += 1

            temperature *= cooling
            if stagnant >= 20000:
                temperature = max(temperature, 3.0)
                stagnant = 0

        polish_iterations = max(5000, iterations // 5)
        for _ in range(polish_iterations):
            state, move_draw = _rng_float(state)
            changed = False
            move_type = 0
            first = 0
            second = 0
            old_label = 0
            new_label = 0
            if move_draw < 0.65:
                move_type = 1
                state, first = _rng_int(state, key_length)
                state, second = _rng_int(state, key_length)
                if first != second and key[first] != key[second]:
                    temporary = key[first]
                    key[first] = key[second]
                    key[second] = temporary
                    changed = True
            else:
                move_type = 2
                state, first = _rng_int(state, key_length)
                state, new_label = _sample_cdf(state, proposal_cdf)
                old_label = int(key[first])
                if new_label != old_label and counts[new_label] < max_counts[new_label]:
                    key[first] = new_label
                    counts[old_label] -= 1
                    counts[new_label] += 1
                    changed = True
            if not changed:
                continue
            candidate_score = mono.score_key(cipher, key, trigram_logp, unigram_logp)
            if candidate_score >= current_score:
                current_score = candidate_score
                if current_score > best_score:
                    best_score = current_score
                    best_key = key.copy()
            else:
                if move_type == 1:
                    temporary = key[first]
                    key[first] = key[second]
                    key[second] = temporary
                else:
                    key[first] = old_label
                    counts[new_label] -= 1
                    counts[old_label] += 1

    return best_key, best_score


def family_arrays(language: core.LanguageData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_counts = np.asarray(
        [fixed.multiplicity(float(probability)) for probability in language.probabilities],
        dtype=np.int32,
    )
    slot_pool = np.asarray(
        [label for label, count in enumerate(max_counts) for _ in range(int(count))],
        dtype=np.int32,
    )
    probabilities = np.asarray(language.probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    proposal_cdf = np.cumsum(probabilities)
    proposal_cdf[-1] = 1.0
    return slot_pool, max_counts, proposal_cdf


def flexible_solve_trial(
    trial: dict,
    language: core.LanguageData,
    trigram_logp: np.ndarray,
    unigram_logp: np.ndarray,
    iterations: int,
    restarts: int,
) -> dict:
    started = __import__("time").perf_counter()
    cipher = list(map(int, trial["cipher"]))
    cipher_array = np.asarray(cipher, dtype=np.int32)
    initial_key = fixed.frequency_slot_key(
        cipher,
        trial["inferred_labels"],
        trial["expected_slot_probabilities"],
    )
    baseline = initial_key[cipher_array].tolist()
    slot_pool, max_counts, proposal_cdf = family_arrays(language)
    solved_key, score = flexible_homophonic_search(
        cipher_array,
        initial_key,
        trigram_logp,
        unigram_logp,
        slot_pool,
        max_counts,
        proposal_cdf,
        iterations,
        restarts,
        int(trial["seed"] & 0x7FFFFFFFFFFFFFFF),
    )
    prediction = solved_key[cipher_array].tolist()
    return {
        "iso": trial["iso"],
        "split": trial["split"],
        "length": trial["length"],
        "replicate": trial["replicate"],
        "seed": trial["seed"],
        "distinct_cipher_symbols": len(trial["inferred_labels"]),
        "initial_inventory_overlap": float(trial["inventory_overlap"]),
        "final_inventory_overlap": fixed.multiset_overlap(solved_key, trial["true_labels"]),
        "iterations": iterations,
        "restarts": restarts,
        "baseline_accuracy": mono.fast_accuracy(trial["plain"], baseline),
        "accuracy": mono.fast_accuracy(trial["plain"], prediction),
        "exact": prediction == trial["plain"],
        "score": float(score),
        "elapsed_seconds": __import__("time").perf_counter() - started,
    }


def summarize(rows: list[dict]) -> dict:
    def subset_summary(subset: list[dict]) -> dict:
        import statistics
        return {
            "trials": len(subset),
            "mean_accuracy": statistics.fmean(row["accuracy"] for row in subset),
            "median_accuracy": statistics.median(row["accuracy"] for row in subset),
            "baseline_mean_accuracy": statistics.fmean(row["baseline_accuracy"] for row in subset),
            "exact_rate": statistics.fmean(float(row["exact"]) for row in subset),
            "mean_initial_inventory_overlap": statistics.fmean(row["initial_inventory_overlap"] for row in subset),
            "mean_final_inventory_overlap": statistics.fmean(row["final_inventory_overlap"] for row in subset),
            "mean_seconds": statistics.fmean(row["elapsed_seconds"] for row in subset),
        }
    result = subset_summary(rows)
    result["by_language"] = {
        iso: subset_summary([row for row in rows if row["iso"] == iso])
        for iso in sorted({row["iso"] for row in rows})
    }
    result["by_length"] = {
        str(length): subset_summary([row for row in rows if row["length"] == length])
        for length in sorted({row["length"] for row in rows})
    }
    return result


def canonical_sha(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dev-reps", type=int, default=8)
    parser.add_argument("--test-reps", type=int, default=20)
    args = parser.parse_args()

    fixed.solve_trial = flexible_solve_trial
    fixed.summarize = summarize
    experiment = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        experiment / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v050_corpora",
    )
    if args.smoke:
        languages = {iso: languages[iso] for iso in ("en", "tr")}
        lengths = (96,)
        args.dev_reps = min(args.dev_reps, 4)
        args.test_reps = min(args.test_reps, 6)
        schedules = ((100000, 20), (300000, 35), (700000, 50))
    else:
        lengths = (96, 192, 384)
        schedules = ((300000, 35), (700000, 50), (1200000, 70))

    models = {iso: mono.build_language_model(language) for iso, language in languages.items()}
    first_iso = sorted(languages)[0]
    pool, caps, cdf = family_arrays(languages[first_iso])
    flexible_homophonic_search(
        np.asarray([0, 1, 2, 3, 0, 2], dtype=np.int32),
        np.asarray([0, 0, 1, 1], dtype=np.int32),
        models[first_iso][0],
        models[first_iso][1],
        pool,
        caps,
        cdf,
        2,
        1,
        1,
    )

    candidates = []
    selected = None
    selected_score = -1.0
    for iterations, restarts in schedules:
        rows = fixed.run_grid(
            languages, models, "dev", args.dev_reps, lengths,
            iterations, restarts, args.workers,
        )
        summary = summarize(rows)
        candidate = {"iterations": iterations, "restarts": restarts, "summary": summary}
        candidates.append(candidate)
        print("V052F_DEV", json.dumps(candidate, sort_keys=True), flush=True)
        if summary["mean_accuracy"] > selected_score:
            selected_score = summary["mean_accuracy"]
            selected = (iterations, restarts)
    assert selected is not None

    test_rows = fixed.run_grid(
        languages, models, "test", args.test_reps, lengths,
        selected[0], selected[1], args.workers,
    )
    test_summary = summarize(test_rows)
    language_floor = min(item["mean_accuracy"] for item in test_summary["by_language"].values())
    short_accuracy = test_summary["by_length"][str(min(lengths))]["mean_accuracy"]
    gate = {
        "overall_pass": test_summary["mean_accuracy"] >= 0.70,
        "language_floor_pass": language_floor >= 0.60 if args.smoke else language_floor >= 0.50,
        "short_text_pass": short_accuracy >= 0.70 if args.smoke else short_accuracy >= 0.60,
        "improves_fixed_smoke": test_summary["mean_accuracy"] > 0.572048611111111,
    }
    gate["pass"] = all(gate.values())
    payload = {
        "programme": "recoverability-frontier-v0.5.2-flexible-homophonic",
        "development_candidates": candidates,
        "selected_schedule": {"iterations": selected[0], "restarts": selected[1]},
        "test_summary": test_summary,
        "test_rows": test_rows,
        "gate": gate,
    }
    payload["scientific_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V052F_SELECTED", json.dumps(payload["selected_schedule"], sort_keys=True), flush=True)
    print("V052F_TEST", json.dumps(test_summary, sort_keys=True), flush=True)
    print("V052F_GATE", json.dumps(gate, sort_keys=True), flush=True)
    print("V052F_SHA256", payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
