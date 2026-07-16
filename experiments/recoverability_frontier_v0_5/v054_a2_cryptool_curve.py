#!/usr/bin/env python3
"""v0.5.4 A2 CrypTool-style exhaustive residual-key restart curve."""
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

import cryptool_homophonic_port_v052 as cryptool
import recoverability_v050 as core
from homophonic_confirm_v052_quadgram import build_quadgram_model
import mono_solver_v051 as mono
import v054_nomenclator_stage_a as stage

PREFIXES = (12, 24, 48, 96)


@njit(cache=True, nogil=True)
def estimate_locked_temperature(
    cipher,
    key,
    swappable,
    quadgram_logp,
    unigram_logp,
    positions,
    offsets,
    target_acceptance,
):
    negative = np.empty(512, dtype=np.float64)
    negative_count = 0
    endpoint_marks = np.zeros(cipher.shape[0], dtype=np.int32)
    endpoint_buffer = np.empty(cipher.shape[0], dtype=np.int32)
    mark_id = 0
    examined = 0
    for first in range(swappable - 1):
        for second in range(first + 1, swappable):
            if key[first] == key[second]:
                continue
            delta, mark_id = cryptool.swap_delta_apply(
                cipher,
                key,
                first,
                second,
                quadgram_logp,
                unigram_logp,
                positions,
                offsets,
                endpoint_marks,
                mark_id,
                endpoint_buffer,
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
    target = min(0.95, max(1e-6, target_acceptance))
    return max(median / (-math.log(target)), 1e-6)


@njit(cache=True, nogil=True)
def locked_single_run(
    cipher,
    start_key,
    swappable,
    quadgram_logp,
    unigram_logp,
    positions,
    offsets,
    steps,
    target_acceptance,
    seed,
):
    key = start_key.copy()
    current_score = cryptool.full_score(cipher, key, quadgram_logp, unigram_logp)
    best_key = key.copy()
    best_score = current_score
    start_temperature = estimate_locked_temperature(
        cipher,
        key,
        swappable,
        quadgram_logp,
        unigram_logp,
        positions,
        offsets,
        target_acceptance,
    )
    state = np.uint64(seed if seed > 0 else 1)
    endpoint_marks = np.zeros(cipher.shape[0], dtype=np.int32)
    endpoint_buffer = np.empty(cipher.shape[0], dtype=np.int32)
    mark_id = 0
    proposals = 0
    while proposals < steps:
        for first in range(swappable - 1):
            for second in range(first + 1, swappable):
                if key[first] == key[second]:
                    continue
                delta, mark_id = cryptool.swap_delta_apply(
                    cipher,
                    key,
                    first,
                    second,
                    quadgram_logp,
                    unigram_logp,
                    positions,
                    offsets,
                    endpoint_marks,
                    mark_id,
                    endpoint_buffer,
                )
                proposals += 1
                remaining = 1.0 - proposals / max(1.0, float(steps))
                temperature = start_temperature * max(remaining, 1e-9)
                accept = delta >= 0.0
                if not accept:
                    probability = math.exp(delta / temperature)
                    if probability > cryptool.ACCEPTANCE_FLOOR:
                        state, draw = cryptool.rng_float(state)
                        accept = draw < probability
                if accept:
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_key = key.copy()
                else:
                    temporary = key[first]
                    key[first] = key[second]
                    key[second] = temporary
                if proposals >= steps:
                    break
            if proposals >= steps:
                break
    return best_key, best_score, start_temperature


def shuffled_variable_key(initial_key: np.ndarray, swappable: int, seed: int) -> np.ndarray:
    output = initial_key.copy()
    rng = np.random.default_rng(seed)
    variable = output[:swappable].copy()
    rng.shuffle(variable)
    output[:swappable] = variable
    return output


def prepare_trial(
    trial: stage.NomenclatorTrial,
    language: core.LanguageData,
) -> tuple[np.ndarray, np.ndarray, int]:
    mixed, char_symbols = stage.build_mixed_cipher(
        trial, trial.code_to_word, len(language.alphabet)
    )
    _symbols, initial_variable = stage.frequency_initial_key(trial, language)
    swappable = len(char_symbols)
    full_key = np.empty(swappable + len(language.alphabet), dtype=np.int32)
    full_key[:swappable] = initial_variable
    full_key[swappable:] = np.arange(len(language.alphabet), dtype=np.int32)
    return mixed, full_key, swappable


def run_restart(
    trial: stage.NomenclatorTrial,
    language: core.LanguageData,
    quadgram: tuple[np.ndarray, np.ndarray],
    restart_index: int,
    steps: int,
    target_acceptance: float,
) -> dict[str, Any]:
    mixed, initial_key, swappable = prepare_trial(trial, language)
    if restart_index == 0:
        start_key = initial_key
    else:
        start_key = shuffled_variable_key(
            initial_key,
            swappable,
            core.stable_seed("v054-a2-ct-shuffle", trial.seed, restart_index),
        )
    positions, offsets, _rare = cryptool.build_positions(mixed.tolist())
    solved, score, temperature = locked_single_run(
        mixed,
        start_key,
        swappable,
        quadgram[0],
        quadgram[1],
        positions,
        offsets,
        steps,
        target_acceptance,
        int(
            core.stable_seed("v054-a2-ct-run", trial.seed, restart_index)
            & 0x7FFFFFFFFFFFFFFF
        ),
    )
    prediction = solved[mixed].tolist()
    return {
        "replicate": trial.replicate,
        "restart": restart_index + 1,
        "score": float(score),
        "accuracy": mono.fast_accuracy(trial.plain, prediction),
        "start_temperature": float(temperature),
    }


def prefix_summary(rows: list[dict[str, Any]], prefix: int) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for replicate in sorted({row["replicate"] for row in rows}):
        candidates = [
            row
            for row in rows
            if row["replicate"] == replicate and row["restart"] <= prefix
        ]
        selected.append(max(candidates, key=lambda row: row["score"]))
    accuracies = [row["accuracy"] for row in selected]
    return {
        "prefix_restarts": prefix,
        "mean_accuracy": statistics.fmean(accuracies),
        "median_accuracy": statistics.median(accuracies),
        "minimum_accuracy": min(accuracies),
        "at_least_90_rate": statistics.fmean(value >= 0.90 for value in accuracies),
        "selected_rows": selected,
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
    parser.add_argument("--candidate-pool", type=int, default=96)
    parser.add_argument("--codebook-size", type=int, default=24)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3000000)
    parser.add_argument("--max-restarts", type=int, default=96)
    parser.add_argument("--target-acceptance", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    experiment = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        experiment / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v050_corpora",
    )
    language = languages[args.iso]
    word_model = stage.build_word_model(
        language, candidate_pool_size=args.candidate_pool
    )
    quadgram = build_quadgram_model(language)
    trials = [
        stage.make_trial(
            language,
            word_model,
            args.split,
            args.length,
            args.offset + replicate,
            args.codebook_size,
        )
        for replicate in range(args.replicates)
    ]

    compile_mixed, compile_key, compile_swappable = prepare_trial(trials[0], language)
    compile_positions, compile_offsets, _rare = cryptool.build_positions(
        compile_mixed.tolist()
    )
    locked_single_run(
        compile_mixed,
        compile_key,
        compile_swappable,
        quadgram[0],
        quadgram[1],
        compile_positions,
        compile_offsets,
        10,
        args.target_acceptance,
        1,
    )

    jobs = [
        (trial, restart)
        for trial in trials
        for restart in range(args.max_restarts)
    ]
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                run_restart,
                trial,
                language,
                quadgram,
                restart,
                args.steps,
                args.target_acceptance,
            )
            for trial, restart in jobs
        ]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            rows.append(future.result())
            if completed % 96 == 0 or completed == len(futures):
                print(
                    f"V054_A2_CRYTOOL_PROGRESS {completed}/{len(futures)}",
                    flush=True,
                )
    rows.sort(key=lambda row: (row["replicate"], row["restart"]))
    summaries = [prefix_summary(rows, prefix) for prefix in PREFIXES]
    passing = [
        item
        for item in summaries
        if item["mean_accuracy"] >= 0.90
        and item["median_accuracy"] >= 0.99
        and item["minimum_accuracy"] >= 0.90
    ]
    selected = min(passing, key=lambda item: item["prefix_restarts"]) if passing else None
    payload: dict[str, Any] = {
        "programme": "recoverability-frontier-v0.5.4-a2-cryptool-prefix-curve",
        "iso": args.iso,
        "split": args.split,
        "length": args.length,
        "candidate_pool": args.candidate_pool,
        "codebook_size": args.codebook_size,
        "offset": args.offset,
        "replicates": args.replicates,
        "steps_per_restart": args.steps,
        "maximum_restarts": args.max_restarts,
        "target_initial_acceptance": args.target_acceptance,
        "prefix_summaries": summaries,
        "selected_prefix": None if selected is None else selected["prefix_restarts"],
        "development_gate_pass": selected is not None,
        "all_rows": rows,
    }
    payload["scientific_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    for item in summaries:
        compact = {key: value for key, value in item.items() if key != "selected_rows"}
        print("V054_A2_CRYTOOL_PREFIX", json.dumps(compact, sort_keys=True), flush=True)
    print("V054_A2_CRYTOOL_SELECTED", payload["selected_prefix"], flush=True)
    print("V054_A2_CRYTOOL_SHA256", payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
