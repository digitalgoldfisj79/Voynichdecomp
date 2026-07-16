#!/usr/bin/env python3
"""v0.6 Family S2: segmented fresh-key polygraphic substitution oracle."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from rapidfuzz.distance import Levenshtein

HERE = Path(__file__).resolve().parent
V05 = HERE.parent / "recoverability_frontier_v0_5"
if str(V05) not in sys.path:
    sys.path.insert(0, str(V05))

import recoverability_v050 as core
import mono_solver_v051 as mono
import v060_family_s_stage_s1 as s1


def build_unit_model(
    language: core.LanguageData,
    inventory: list[tuple[int, ...]],
    alpha: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unit_to_id = {unit: index for index, unit in enumerate(inventory)}
    stream: list[int] = []
    for sentence in language.encoded_sentences["train"]:
        units = s1.unitise(list(sentence), inventory)
        stream.extend(unit_to_id[unit] for unit in units)
    size = len(inventory)
    trigram_counts = np.full((size, size, size), alpha, dtype=np.float64)
    context_counts = np.full((size, size), alpha * size, dtype=np.float64)
    for first, second, third in zip(stream, stream[1:], stream[2:]):
        trigram_counts[first, second, third] += 1.0
        context_counts[first, second] += 1.0
    trigram = np.log(trigram_counts / context_counts[:, :, None]).astype(np.float64)
    counts = np.bincount(np.asarray(stream, dtype=np.int32), minlength=size).astype(np.float64)
    probabilities = (counts + 0.5) / (counts.sum() + 0.5 * size)
    unigram = np.log(probabilities).astype(np.float64)
    return trigram, unigram, probabilities


def segmented_symbols(
    trial: s1.SegmentationTrial,
    inventory: list[tuple[int, ...]],
) -> tuple[list[int], np.ndarray, list[tuple[int, ...]]]:
    unit_to_id = {unit: index for index, unit in enumerate(inventory)}
    codes: list[tuple[int, ...]] = []
    left = 0
    for right in trial.boundaries:
        codes.append(tuple(trial.cipher[left:right]))
        left = right
    code_to_symbol: dict[tuple[int, ...], int] = {}
    symbols: list[int] = []
    for code in codes:
        if code not in code_to_symbol:
            code_to_symbol[code] = len(code_to_symbol)
        symbols.append(code_to_symbol[code])
    true_key = np.full(len(inventory), -1, dtype=np.int32)
    for code, symbol in code_to_symbol.items():
        true_key[symbol] = unit_to_id[trial.visible_codebook[code]]
    observed_codes = [None] * len(code_to_symbol)
    for code, symbol in code_to_symbol.items():
        observed_codes[symbol] = code
    return symbols, true_key, observed_codes  # type: ignore[return-value]


def frequency_key(
    cipher: list[int], probabilities: np.ndarray, inventory_size: int
) -> np.ndarray:
    counts = np.bincount(np.asarray(cipher, dtype=np.int32), minlength=inventory_size)
    cipher_rank = np.argsort(-counts, kind="stable")
    plain_rank = np.argsort(-probabilities, kind="stable")
    key = np.empty(inventory_size, dtype=np.int32)
    for cipher_symbol, unit_id in zip(cipher_rank, plain_rank):
        key[int(cipher_symbol)] = int(unit_id)
    return key


def expand_units(
    unit_ids: list[int], inventory: list[tuple[int, ...]]
) -> list[int]:
    out: list[int] = []
    for unit_id in unit_ids:
        out.extend(inventory[int(unit_id)])
    return out


def char_accuracy(truth: list[int], predicted: list[int]) -> float:
    return max(
        0.0,
        1.0 - Levenshtein.distance(truth, predicted) / max(1, len(truth), len(predicted)),
    )


def solve_trial(
    trial: s1.SegmentationTrial,
    inventory: list[tuple[int, ...]],
    model: tuple[np.ndarray, np.ndarray, np.ndarray],
    iterations: int,
    restarts: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    trigram, unigram, probabilities = model
    cipher, true_key, _observed_codes = segmented_symbols(trial, inventory)
    initial = frequency_key(cipher, probabilities, len(inventory))
    cipher_array = np.asarray(cipher, dtype=np.int32)
    baseline_units = initial[cipher_array].tolist()
    baseline_plain = expand_units(baseline_units, inventory)
    solved_key, score = mono.anneal_mono(
        cipher_array,
        initial,
        trigram,
        unigram,
        iterations,
        restarts,
        int(core.stable_seed("v060-family-s2-search", trial.seed) & 0x7FFFFFFFFFFFFFFF),
    )
    predicted_units = solved_key[cipher_array].tolist()
    predicted_plain = expand_units(predicted_units, inventory)
    observed_symbols = sorted(set(cipher))
    mapping_accuracy = statistics.fmean(
        int(solved_key[symbol]) == int(true_key[symbol])
        for symbol in observed_symbols
    )
    return {
        "iso": trial.iso,
        "split": trial.split,
        "replicate": trial.replicate,
        "plaintext_length": len(trial.plain),
        "unit_tokens": len(cipher),
        "observed_code_groups": len(observed_symbols),
        "inventory_size": len(inventory),
        "baseline_accuracy": char_accuracy(trial.plain, baseline_plain),
        "plaintext_accuracy": char_accuracy(trial.plain, predicted_plain),
        "plaintext_exact": predicted_plain == trial.plain,
        "observed_mapping_accuracy": mapping_accuracy,
        "score": float(score),
        "elapsed_seconds": time.perf_counter() - started,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recovery = [float(row["plaintext_accuracy"]) for row in rows]
    mapping = [float(row["observed_mapping_accuracy"]) for row in rows]
    return {
        "trials": len(rows),
        "plaintext": {
            "mean": statistics.fmean(recovery),
            "median": statistics.median(recovery),
            "minimum": min(recovery),
            "at_least_80_rate": statistics.fmean(value >= 0.80 for value in recovery),
            "at_least_90_rate": statistics.fmean(value >= 0.90 for value in recovery),
            "exact_rate": statistics.fmean(row["plaintext_exact"] for row in rows),
        },
        "mapping": {
            "mean": statistics.fmean(mapping),
            "median": statistics.median(mapping),
            "minimum": min(mapping),
        },
        "mean_baseline_accuracy": statistics.fmean(
            float(row["baseline_accuracy"]) for row in rows
        ),
        "gate": {
            "pass": (
                statistics.fmean(recovery) >= 0.80
                and statistics.median(recovery) >= 0.90
                and sum(value >= 0.80 for value in recovery) >= 14
                and statistics.fmean(mapping) >= 0.75
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iso", default="en")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--length", type=int, default=384)
    parser.add_argument("--replicates", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=700000)
    parser.add_argument("--restarts", type=int, default=50)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        root / "corpus_manifest_v050.json", args.repo / ".cache" / "v060-family-s2"
    )
    language = languages[args.iso]
    inventory = s1.candidate_inventory(language)
    model = build_unit_model(language, inventory)
    trials = [
        s1.make_trial(language, args.split, args.length, replicate)
        for replicate in range(args.replicates)
    ]

    def run_one(trial: s1.SegmentationTrial) -> dict[str, Any]:
        row = solve_trial(trial, inventory, model, args.iterations, args.restarts)
        print("V060_S2_TRIAL", json.dumps(row, sort_keys=True), flush=True)
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(run_one, trials))
    summary = summarize(rows)
    payload = {
        "config": vars(args) | {"repo": str(args.repo), "output": str(args.output)},
        "rows": rows,
        "summary": summary,
    }
    scientific = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(scientific).hexdigest()
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V060_S2_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V060_S2_SHA256", payload["sha256"], flush=True)


if __name__ == "__main__":
    main()
