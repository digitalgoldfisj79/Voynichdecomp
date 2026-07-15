#!/usr/bin/env python3
"""v0.5.1 key-invariant monoalphabetic substitution solver benchmark."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numba import njit
from rapidfuzz.distance import Levenshtein

import recoverability_v050 as core


def canonicalize(values: Iterable[int]) -> list[int]:
    mapping: dict[int, int] = {}
    out: list[int] = []
    for value in values:
        value = int(value)
        if value not in mapping:
            mapping[value] = len(mapping)
        out.append(mapping[value])
    return out


def fast_accuracy(truth: list[int], predicted: list[int]) -> float:
    if not truth and not predicted:
        return 1.0
    denominator = max(1, len(truth), len(predicted))
    return max(0.0, 1.0 - Levenshtein.distance(truth, predicted) / denominator)


def build_language_model(language: core.LanguageData, alpha: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    alphabet_size = len(language.alphabet)
    trigram_counts = np.full(
        (alphabet_size, alphabet_size, alphabet_size),
        alpha,
        dtype=np.float64,
    )
    context_counts = np.full(
        (alphabet_size, alphabet_size),
        alpha * alphabet_size,
        dtype=np.float64,
    )
    stream = language.train_stream
    for first, second, third in zip(stream, stream[1:], stream[2:]):
        trigram_counts[first, second, third] += 1.0
        context_counts[first, second] += 1.0
    trigram_logp = np.log(trigram_counts / context_counts[:, :, None]).astype(np.float64)
    unigram_logp = np.log(np.asarray(language.probabilities, dtype=np.float64))
    return trigram_logp, unigram_logp


def frequency_key(cipher: list[int], language: core.LanguageData) -> np.ndarray:
    alphabet_size = len(language.alphabet)
    counts = np.bincount(np.asarray(cipher, dtype=np.int64), minlength=alphabet_size)
    cipher_rank = np.argsort(-counts, kind="stable")
    plain_rank = np.argsort(-np.asarray(language.probabilities), kind="stable")
    key = np.empty(alphabet_size, dtype=np.int32)
    for cipher_symbol, plain_symbol in zip(cipher_rank, plain_rank):
        key[int(cipher_symbol)] = int(plain_symbol)
    return key


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
    value = float(state >> np.uint64(11)) * (1.0 / 9007199254740992.0)
    return state, value


@njit(cache=True, nogil=True)
def score_key(
    cipher: np.ndarray,
    key: np.ndarray,
    trigram_logp: np.ndarray,
    unigram_logp: np.ndarray,
) -> float:
    length = cipher.shape[0]
    if length == 0:
        return -1e300
    score = 0.15 * unigram_logp[key[cipher[0]]]
    if length >= 2:
        score += 0.15 * unigram_logp[key[cipher[1]]]
    for index in range(2, length):
        first = key[cipher[index - 2]]
        second = key[cipher[index - 1]]
        third = key[cipher[index]]
        score += trigram_logp[first, second, third]
        score += 0.15 * unigram_logp[third]
    return score


@njit(cache=True, nogil=True)
def anneal_mono(
    cipher: np.ndarray,
    initial_key: np.ndarray,
    trigram_logp: np.ndarray,
    unigram_logp: np.ndarray,
    iterations: int,
    restarts: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    alphabet_size = initial_key.shape[0]
    best_key = initial_key.copy()
    best_score = score_key(cipher, best_key, trigram_logp, unigram_logp)
    state = np.uint64(seed if seed > 0 else 1)

    for restart in range(restarts):
        key = initial_key.copy()
        perturbations = 2 + restart * 3
        for _ in range(perturbations):
            state, first = _rng_int(state, alphabet_size)
            state, second = _rng_int(state, alphabet_size)
            if first != second:
                temporary = key[first]
                key[first] = key[second]
                key[second] = temporary

        current_score = score_key(cipher, key, trigram_logp, unigram_logp)
        if current_score > best_score:
            best_score = current_score
            best_key = key.copy()

        temperature = 12.0
        cooling = math.exp(math.log(0.08 / 12.0) / max(1, iterations))
        stagnant = 0

        for _ in range(iterations):
            state, first = _rng_int(state, alphabet_size)
            state, second = _rng_int(state, alphabet_size)
            if first == second:
                continue

            temporary = key[first]
            key[first] = key[second]
            key[second] = temporary
            candidate_score = score_key(cipher, key, trigram_logp, unigram_logp)
            delta = candidate_score - current_score

            accept = delta >= 0.0
            if not accept:
                state, uniform = _rng_float(state)
                accept = uniform < math.exp(delta / max(temperature, 1e-9))

            if accept:
                current_score = candidate_score
                if current_score > best_score:
                    best_score = current_score
                    best_key = key.copy()
                    stagnant = 0
                else:
                    stagnant += 1
            else:
                temporary = key[first]
                key[first] = key[second]
                key[second] = temporary
                stagnant += 1

            temperature *= cooling
            if stagnant > 3000:
                temperature = max(temperature, 1.5)
                stagnant = 0

    return best_key, best_score


def make_trial(
    language: core.LanguageData,
    split: str,
    length: int,
    replicate: int,
) -> dict[str, Any]:
    chunks = core.source_chunks(language, split, length)
    if not chunks:
        raise RuntimeError(f"no chunks for {language.iso}/{split}/{length}")
    plain = list(chunks[replicate % len(chunks)])
    seed = core.stable_seed("v051-mono", split, language.iso, length, replicate)
    import random

    rng = random.Random(seed)
    packet = core.encrypt_sequence(plain, "mono", language, rng, parameter_mode=split)
    cipher = canonicalize(packet.cipher)
    return {
        "iso": language.iso,
        "split": split,
        "length": length,
        "replicate": replicate,
        "seed": seed,
        "plain": plain,
        "cipher": cipher,
    }


def solve_trial(
    trial: dict[str, Any],
    language: core.LanguageData,
    trigram_logp: np.ndarray,
    unigram_logp: np.ndarray,
    iterations: int,
    restarts: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    cipher = trial["cipher"]
    initial_key = frequency_key(cipher, language)
    cipher_array = np.asarray(cipher, dtype=np.int32)
    baseline = initial_key[cipher_array].astype(np.int32).tolist()
    baseline_accuracy = fast_accuracy(trial["plain"], baseline)

    solved_key, score = anneal_mono(
        cipher_array,
        initial_key,
        trigram_logp,
        unigram_logp,
        iterations,
        restarts,
        int(trial["seed"] & 0x7FFFFFFFFFFFFFFF),
    )
    prediction = solved_key[cipher_array].astype(np.int32).tolist()
    accuracy = fast_accuracy(trial["plain"], prediction)
    return {
        "iso": trial["iso"],
        "split": trial["split"],
        "length": trial["length"],
        "replicate": trial["replicate"],
        "seed": trial["seed"],
        "iterations": iterations,
        "restarts": restarts,
        "baseline_accuracy": baseline_accuracy,
        "accuracy": accuracy,
        "exact": prediction == trial["plain"],
        "score": float(score),
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_grid(
    languages: dict[str, core.LanguageData],
    models: dict[str, tuple[np.ndarray, np.ndarray]],
    split: str,
    replicates: int,
    lengths: tuple[int, ...],
    iterations: int,
    restarts: int,
    workers: int,
) -> list[dict[str, Any]]:
    jobs: list[tuple[dict[str, Any], core.LanguageData, np.ndarray, np.ndarray]] = []
    for iso in sorted(languages):
        language = languages[iso]
        trigram_logp, unigram_logp = models[iso]
        for length in lengths:
            for replicate in range(replicates):
                jobs.append(
                    (
                        make_trial(language, split, length, replicate),
                        language,
                        trigram_logp,
                        unigram_logp,
                    )
                )

    def execute(job: tuple[dict[str, Any], core.LanguageData, np.ndarray, np.ndarray]) -> dict[str, Any]:
        trial, language, trigram_logp, unigram_logp = job
        return solve_trial(
            trial,
            language,
            trigram_logp,
            unigram_logp,
            iterations,
            restarts,
        )

    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(execute, job) for job in jobs]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                print(
                    f"V051_PROGRESS split={split} completed={completed}/{len(futures)} "
                    f"iterations={iterations} restarts={restarts}",
                    flush=True,
                )
    rows.sort(key=lambda row: (row["iso"], row["length"], row["replicate"]))
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_language: dict[str, Any] = {}
    by_length: dict[str, Any] = {}
    for iso in sorted({row["iso"] for row in rows}):
        subset = [row for row in rows if row["iso"] == iso]
        by_language[iso] = {
            "trials": len(subset),
            "mean_accuracy": statistics.fmean(row["accuracy"] for row in subset),
            "median_accuracy": statistics.median(row["accuracy"] for row in subset),
            "baseline_mean_accuracy": statistics.fmean(row["baseline_accuracy"] for row in subset),
            "exact_rate": statistics.fmean(float(row["exact"]) for row in subset),
            "mean_seconds": statistics.fmean(row["elapsed_seconds"] for row in subset),
        }
    for length in sorted({row["length"] for row in rows}):
        subset = [row for row in rows if row["length"] == length]
        by_length[str(length)] = {
            "trials": len(subset),
            "mean_accuracy": statistics.fmean(row["accuracy"] for row in subset),
            "median_accuracy": statistics.median(row["accuracy"] for row in subset),
            "baseline_mean_accuracy": statistics.fmean(row["baseline_accuracy"] for row in subset),
            "exact_rate": statistics.fmean(float(row["exact"]) for row in subset),
        }
    return {
        "trials": len(rows),
        "mean_accuracy": statistics.fmean(row["accuracy"] for row in rows),
        "median_accuracy": statistics.median(row["accuracy"] for row in rows),
        "baseline_mean_accuracy": statistics.fmean(row["baseline_accuracy"] for row in rows),
        "exact_rate": statistics.fmean(float(row["exact"]) for row in rows),
        "mean_seconds": statistics.fmean(row["elapsed_seconds"] for row in rows),
        "by_language": by_language,
        "by_length": by_length,
    }


def canonical_json_sha(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dev-reps", type=int, default=8)
    parser.add_argument("--test-reps", type=int, default=20)
    args = parser.parse_args()

    experiment_dir = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        experiment_dir / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v050_corpora",
    )
    if args.smoke:
        languages = {iso: languages[iso] for iso in ("en", "tr")}
        lengths = (96,)
        args.dev_reps = min(args.dev_reps, 2)
        args.test_reps = min(args.test_reps, 3)
        schedule_grid = ((1200, 2), (3000, 3))
    else:
        lengths = (96, 192, 384)
        schedule_grid = ((8000, 4), (18000, 6), (35000, 8))

    models = {iso: build_language_model(language) for iso, language in languages.items()}
    # Trigger Numba compilation once before parallel timing.
    first_language = languages[sorted(languages)[0]]
    first_model = models[sorted(languages)[0]]
    dummy_cipher = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int32)
    dummy_key = np.arange(len(first_language.alphabet), dtype=np.int32)
    anneal_mono(dummy_cipher, dummy_key, first_model[0], first_model[1], 2, 1, 1)

    development_candidates: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] | None = None
    selected_schedule: tuple[int, int] | None = None
    selected_score = -1.0

    for iterations, restarts in schedule_grid:
        rows = run_grid(
            languages,
            models,
            "dev",
            args.dev_reps,
            lengths,
            iterations,
            restarts,
            args.workers,
        )
        summary = summarize(rows)
        candidate = {
            "iterations": iterations,
            "restarts": restarts,
            "summary": summary,
        }
        development_candidates.append(candidate)
        print("V051_DEV", json.dumps(candidate, sort_keys=True), flush=True)
        if summary["mean_accuracy"] > selected_score:
            selected_score = summary["mean_accuracy"]
            selected_schedule = (iterations, restarts)
            selected_rows = rows

    assert selected_schedule is not None and selected_rows is not None
    test_rows = run_grid(
        languages,
        models,
        "test",
        args.test_reps,
        lengths,
        selected_schedule[0],
        selected_schedule[1],
        args.workers,
    )
    test_summary = summarize(test_rows)
    language_floor = min(value["mean_accuracy"] for value in test_summary["by_language"].values())
    gate = {
        "mean_accuracy_pass": test_summary["mean_accuracy"] >= 0.70,
        "language_floor_pass": language_floor >= 0.50,
    }
    gate["pass"] = all(gate.values())

    scientific_payload = {
        "programme": "recoverability-frontier-v0.5.1-mono-solver",
        "input_representation": "first-occurrence recurrence canonicalisation",
        "language_model": "smoothed character trigram plus unigram term",
        "development_candidates": development_candidates,
        "selected_schedule": {
            "iterations": selected_schedule[0],
            "restarts": selected_schedule[1],
        },
        "test_summary": test_summary,
        "gate": gate,
        "test_rows": test_rows,
    }
    scientific_payload["scientific_sha256"] = canonical_json_sha(scientific_payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scientific_payload, indent=2, sort_keys=True), encoding="utf-8")

    print("V051_SELECTED", json.dumps(scientific_payload["selected_schedule"], sort_keys=True), flush=True)
    print("V051_TEST", json.dumps(test_summary, sort_keys=True), flush=True)
    print("V051_GATE", json.dumps(gate, sort_keys=True), flush=True)
    print("V051_SCIENTIFIC_SHA256", scientific_payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
