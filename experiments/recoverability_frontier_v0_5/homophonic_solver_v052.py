#!/usr/bin/env python3
"""v0.5.2 key-invariant bounded homophonic substitution solver."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import recoverability_v050 as core
import mono_solver_v051 as mono
import mono_solver_v051_search2 as search2


def canonicalize_with_inverse(values: list[int]) -> tuple[list[int], list[int]]:
    raw_to_canonical: dict[int, int] = {}
    canonical_to_raw: list[int] = []
    output: list[int] = []
    for raw in values:
        raw = int(raw)
        if raw not in raw_to_canonical:
            raw_to_canonical[raw] = len(raw_to_canonical)
            canonical_to_raw.append(raw)
        output.append(raw_to_canonical[raw])
    return output, canonical_to_raw


def multiplicity(probability: float) -> int:
    return 1 + min(3, int(round(3.5 * math.sqrt(max(probability, 0.0)))))


def inferred_slot_inventory(language: core.LanguageData, observed_symbols: int) -> tuple[np.ndarray, np.ndarray]:
    slots: list[tuple[float, int]] = []
    for plain_symbol, probability in enumerate(language.probabilities):
        count = multiplicity(float(probability))
        per_symbol_probability = float(probability) / count
        for _ in range(count):
            slots.append((per_symbol_probability, plain_symbol))
    slots.sort(key=lambda item: (-item[0], item[1]))
    if observed_symbols > len(slots):
        raise RuntimeError(f"observed {observed_symbols} symbols exceeds bounded inventory {len(slots)}")
    selected = slots[:observed_symbols]
    labels = np.asarray([label for _, label in selected], dtype=np.int32)
    expected = np.asarray([probability for probability, _ in selected], dtype=np.float64)
    return labels, expected


def frequency_slot_key(
    cipher: list[int],
    labels: np.ndarray,
    expected_symbol_probabilities: np.ndarray,
) -> np.ndarray:
    observed_symbols = len(labels)
    counts = np.bincount(np.asarray(cipher, dtype=np.int32), minlength=observed_symbols)
    cipher_rank = np.argsort(-counts, kind="stable")
    slot_rank = np.argsort(-expected_symbol_probabilities, kind="stable")
    key = np.empty(observed_symbols, dtype=np.int32)
    for cipher_symbol, slot_index in zip(cipher_rank, slot_rank):
        key[int(cipher_symbol)] = int(labels[int(slot_index)])
    return key


def multiset_overlap(first: list[int] | np.ndarray, second: list[int] | np.ndarray) -> float:
    first_counts = Counter(map(int, first))
    second_counts = Counter(map(int, second))
    denominator = max(1, sum(first_counts.values()), sum(second_counts.values()))
    overlap = sum(min(first_counts[key], second_counts[key]) for key in first_counts.keys() | second_counts.keys())
    return overlap / denominator


def make_trial(language: core.LanguageData, split: str, length: int, replicate: int) -> dict[str, Any]:
    chunks = core.source_chunks(language, split, length)
    if not chunks:
        raise RuntimeError(f"no chunks for {language.iso}/{split}/{length}")
    plain = list(chunks[replicate % len(chunks)])
    seed = core.stable_seed("v052-homophonic", split, language.iso, length, replicate)
    packet = core.encrypt_sequence(
        plain,
        "homophonic",
        language,
        random.Random(seed),
        parameter_mode=split,
    )
    cipher, canonical_to_raw = canonicalize_with_inverse(packet.cipher)
    true_inverse = packet.metadata["inverse"]
    true_labels = [int(true_inverse[int(raw)]) for raw in canonical_to_raw]
    inferred_labels, expected = inferred_slot_inventory(language, len(canonical_to_raw))
    return {
        "iso": language.iso,
        "split": split,
        "length": length,
        "replicate": replicate,
        "seed": seed,
        "plain": plain,
        "cipher": cipher,
        "true_labels": true_labels,
        "inferred_labels": inferred_labels,
        "expected_slot_probabilities": expected,
        "inventory_overlap": multiset_overlap(inferred_labels, true_labels),
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
    cipher = list(map(int, trial["cipher"]))
    cipher_array = np.asarray(cipher, dtype=np.int32)
    initial_key = frequency_slot_key(
        cipher,
        trial["inferred_labels"],
        trial["expected_slot_probabilities"],
    )
    baseline = initial_key[cipher_array].astype(np.int32).tolist()
    baseline_accuracy = mono.fast_accuracy(trial["plain"], baseline)
    solved_key, score = search2.anneal_mono_search2(
        cipher_array,
        initial_key,
        trigram_logp,
        unigram_logp,
        iterations,
        restarts,
        int(trial["seed"] & 0x7FFFFFFFFFFFFFFF),
    )
    prediction = solved_key[cipher_array].astype(np.int32).tolist()
    return {
        "iso": trial["iso"],
        "split": trial["split"],
        "length": trial["length"],
        "replicate": trial["replicate"],
        "seed": trial["seed"],
        "distinct_cipher_symbols": len(trial["inferred_labels"]),
        "inventory_overlap": float(trial["inventory_overlap"]),
        "iterations": iterations,
        "restarts": restarts,
        "baseline_accuracy": baseline_accuracy,
        "accuracy": mono.fast_accuracy(trial["plain"], prediction),
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
                jobs.append((make_trial(language, split, length, replicate), language, trigram_logp, unigram_logp))

    def execute(job: tuple[dict[str, Any], core.LanguageData, np.ndarray, np.ndarray]) -> dict[str, Any]:
        trial, language, trigram_logp, unigram_logp = job
        return solve_trial(trial, language, trigram_logp, unigram_logp, iterations, restarts)

    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(execute, job) for job in jobs]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                print(
                    f"V052_PROGRESS split={split} completed={completed}/{len(futures)} "
                    f"iterations={iterations} restarts={restarts}",
                    flush=True,
                )
    rows.sort(key=lambda row: (row["iso"], row["length"], row["replicate"]))
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summary_for(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "trials": len(subset),
            "mean_accuracy": statistics.fmean(row["accuracy"] for row in subset),
            "median_accuracy": statistics.median(row["accuracy"] for row in subset),
            "baseline_mean_accuracy": statistics.fmean(row["baseline_accuracy"] for row in subset),
            "exact_rate": statistics.fmean(float(row["exact"]) for row in subset),
            "mean_inventory_overlap": statistics.fmean(row["inventory_overlap"] for row in subset),
            "mean_distinct_cipher_symbols": statistics.fmean(row["distinct_cipher_symbols"] for row in subset),
            "mean_seconds": statistics.fmean(row["elapsed_seconds"] for row in subset),
        }

    result = summary_for(rows)
    result["by_language"] = {
        iso: summary_for([row for row in rows if row["iso"] == iso])
        for iso in sorted({row["iso"] for row in rows})
    }
    result["by_length"] = {
        str(length): summary_for([row for row in rows if row["length"] == length])
        for length in sorted({row["length"] for row in rows})
    }
    return result


def canonical_sha(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dev-reps", type=int, default=8)
    parser.add_argument("--test-reps", type=int, default=20)
    args = parser.parse_args()

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
        schedule_grid = ((100000, 20), (300000, 35), (700000, 50))
    else:
        lengths = (96, 192, 384)
        schedule_grid = ((300000, 35), (700000, 50), (1200000, 70))

    models = {iso: mono.build_language_model(language) for iso, language in languages.items()}
    first_iso = sorted(languages)[0]
    dummy_key = np.asarray([0, 0, 1, 1], dtype=np.int32)
    search2.anneal_mono_search2(
        np.asarray([0, 1, 2, 3, 0, 2], dtype=np.int32),
        dummy_key,
        models[first_iso][0],
        models[first_iso][1],
        2,
        1,
        1,
    )

    candidates: list[dict[str, Any]] = []
    selected: tuple[int, int] | None = None
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
        candidate = {"iterations": iterations, "restarts": restarts, "summary": summary}
        candidates.append(candidate)
        print("V052_DEV", json.dumps(candidate, sort_keys=True), flush=True)
        if summary["mean_accuracy"] > selected_score:
            selected_score = summary["mean_accuracy"]
            selected = (iterations, restarts)

    assert selected is not None
    test_rows = run_grid(
        languages,
        models,
        "test",
        args.test_reps,
        lengths,
        selected[0],
        selected[1],
        args.workers,
    )
    test_summary = summarize(test_rows)
    language_floor = min(item["mean_accuracy"] for item in test_summary["by_language"].values())
    short_accuracy = test_summary["by_length"][str(min(lengths))]["mean_accuracy"]
    gate = {
        "overall_pass": test_summary["mean_accuracy"] >= 0.70,
        "language_floor_pass": language_floor >= 0.50,
        "short_text_pass": short_accuracy >= 0.60,
    }
    gate["pass"] = all(gate.values())
    payload = {
        "programme": "recoverability-frontier-v0.5.2-homophonic",
        "development_candidates": candidates,
        "selected_schedule": {"iterations": selected[0], "restarts": selected[1]},
        "test_summary": test_summary,
        "test_rows": test_rows,
        "gate": gate,
    }
    payload["scientific_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V052_SELECTED", json.dumps(payload["selected_schedule"], sort_keys=True), flush=True)
    print("V052_TEST", json.dumps(test_summary, sort_keys=True), flush=True)
    print("V052_GATE", json.dumps(gate, sort_keys=True), flush=True)
    print("V052_SHA256", payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
