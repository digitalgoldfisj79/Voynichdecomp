#!/usr/bin/env python3
"""v0.6 Family S1: exact segmentation oracle for unspaced polygraphic codes."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import dataclasses
import hashlib
import json
import random
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


@dataclasses.dataclass
class SegmentationTrial:
    iso: str
    split: str
    replicate: int
    seed: int
    plain: list[int]
    units: list[tuple[int, ...]]
    cipher: list[int]
    visible_codebook: dict[tuple[int, ...], tuple[int, ...]]
    boundaries: list[int]
    inventory_size: int


def candidate_inventory(language: core.LanguageData) -> list[tuple[int, ...]]:
    singles = [(i,) for i in range(len(language.alphabet))]
    bigrams: collections.Counter[tuple[int, ...]] = collections.Counter()
    trigrams: collections.Counter[tuple[int, ...]] = collections.Counter()
    space = language.char_to_id.get(" ")
    for word in language.train_words:
        for i in range(len(word) - 1):
            unit = tuple(word[i : i + 2])
            if space not in unit:
                bigrams[unit] += 1
        for i in range(len(word) - 2):
            unit = tuple(word[i : i + 3])
            if space not in unit:
                trigrams[unit] += 1
    selected = singles
    selected += [unit for unit, _ in bigrams.most_common(16)]
    selected += [unit for unit, _ in trigrams.most_common(8)]
    out: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for unit in selected:
        if unit not in seen:
            seen.add(unit)
            out.append(unit)
    return out


def unitise(plain: list[int], inventory: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    lookup = set(inventory)
    units: list[tuple[int, ...]] = []
    i = 0
    while i < len(plain):
        selected: tuple[int, ...] | None = None
        for width in (3, 2):
            if i + width <= len(plain):
                candidate = tuple(plain[i : i + width])
                if candidate in lookup:
                    selected = candidate
                    break
        if selected is None:
            selected = (plain[i],)
        units.append(selected)
        i += len(selected)
    return units


def all_codes() -> dict[int, list[tuple[int, ...]]]:
    return {
        1: [(a,) for a in range(10)],
        2: [(a, b) for a in range(10) for b in range(10)],
        3: [(a, b, c) for a in range(10) for b in range(10) for c in range(10)],
    }


def sample_codebook(
    rng: random.Random, inventory: list[tuple[int, ...]]
) -> dict[tuple[int, ...], tuple[int, ...]]:
    pools = all_codes()
    for pool in pools.values():
        rng.shuffle(pool)
    weights = ((1, 0.20), (2, 0.45), (3, 0.35))
    codebook: dict[tuple[int, ...], tuple[int, ...]] = {}
    for unit in inventory:
        value = rng.random()
        cumulative = 0.0
        desired = 3
        for width, weight in weights:
            cumulative += weight
            if value <= cumulative:
                desired = width
                break
        choices = [desired, 2, 3, 1]
        width = next(candidate for candidate in choices if pools[candidate])
        codebook[unit] = pools[width].pop()
    return codebook


def make_trial(
    language: core.LanguageData,
    split: str,
    length: int,
    replicate: int,
) -> SegmentationTrial:
    chunks = core.source_chunks(language, split, length)
    if not chunks:
        raise RuntimeError(f"no chunks for {language.iso}/{split}/{length}")
    plain = list(chunks[replicate % len(chunks)])
    seed = core.stable_seed("v060-family-s1", language.iso, split, length, replicate)
    rng = random.Random(seed)
    inventory = candidate_inventory(language)
    units = unitise(plain, inventory)
    codebook = sample_codebook(rng, inventory)
    surface = list(range(10))
    rng.shuffle(surface)
    visible_codebook = {
        tuple(surface[digit] for digit in code): unit
        for unit, code in codebook.items()
    }
    cipher: list[int] = []
    boundaries: list[int] = []
    for unit in units:
        visible_code = next(code for code, mapped in visible_codebook.items() if mapped == unit)
        cipher.extend(visible_code)
        boundaries.append(len(cipher))
    return SegmentationTrial(
        iso=language.iso,
        split=split,
        replicate=replicate,
        seed=seed,
        plain=plain,
        units=units,
        cipher=cipher,
        visible_codebook=visible_codebook,
        boundaries=boundaries,
        inventory_size=len(inventory),
    )


def extend_score(
    score: float,
    last_two: tuple[int, int],
    unit: tuple[int, ...],
    trigram: np.ndarray,
    unigram: np.ndarray,
) -> tuple[float, tuple[int, int]]:
    first, second = last_two
    for value in unit:
        if first < 0:
            score += 0.15 * float(unigram[value])
            first, second = second, value
        elif second < 0:
            score += 0.15 * float(unigram[value])
            first, second = second, value
        else:
            score += float(trigram[first, second, value])
            score += 0.15 * float(unigram[value])
            first, second = second, value
    return score, (first, second)


def decode_exact(
    cipher: list[int],
    codebook: dict[tuple[int, ...], tuple[int, ...]],
    trigram: np.ndarray,
    unigram: np.ndarray,
) -> tuple[list[int], list[int], float]:
    n = len(cipher)
    # dp[position][last-two] = (score, previous-position, previous-state, code)
    dp: list[dict[tuple[int, int], tuple[float, int, tuple[int, int], tuple[int, ...]]]] = [
        {} for _ in range(n + 1)
    ]
    start = (-1, -1)
    dp[0][start] = (0.0, -1, start, ())
    by_length: dict[int, dict[tuple[int, ...], tuple[int, ...]]] = {
        width: {code: unit for code, unit in codebook.items() if len(code) == width}
        for width in (1, 2, 3)
    }
    for position in range(n):
        if not dp[position]:
            continue
        matches: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for width in (1, 2, 3):
            if position + width <= n:
                code = tuple(cipher[position : position + width])
                unit = by_length[width].get(code)
                if unit is not None:
                    matches.append((code, unit))
        for state, record in list(dp[position].items()):
            score = record[0]
            for code, unit in matches:
                candidate_score, candidate_state = extend_score(
                    score, state, unit, trigram, unigram
                )
                target = position + len(code)
                incumbent = dp[target].get(candidate_state)
                if incumbent is None or candidate_score > incumbent[0]:
                    dp[target][candidate_state] = (
                        candidate_score,
                        position,
                        state,
                        code,
                    )
    if not dp[n]:
        raise RuntimeError("no complete segmentation path")
    final_state, final_record = max(dp[n].items(), key=lambda item: item[1][0])
    score = final_record[0]
    codes_reversed: list[tuple[int, ...]] = []
    position = n
    state = final_state
    while position > 0:
        record = dp[position][state]
        code = record[3]
        codes_reversed.append(code)
        position, state = record[1], record[2]
    codes = list(reversed(codes_reversed))
    plaintext: list[int] = []
    boundaries: list[int] = []
    cursor = 0
    for code in codes:
        plaintext.extend(codebook[code])
        cursor += len(code)
        boundaries.append(cursor)
    return plaintext, boundaries, score


def boundary_metrics(truth: list[int], predicted: list[int]) -> dict[str, float]:
    truth_set = set(truth[:-1])
    predicted_set = set(predicted[:-1])
    true_positive = len(truth_set & predicted_set)
    precision = true_positive / max(1, len(predicted_set))
    recall = true_positive / max(1, len(truth_set))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def accuracy(truth: list[int], predicted: list[int]) -> float:
    return max(
        0.0,
        1.0 - Levenshtein.distance(truth, predicted) / max(1, len(truth), len(predicted)),
    )


def solve_trial(
    trial: SegmentationTrial,
    model: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    started = time.perf_counter()
    predicted, boundaries, score = decode_exact(
        trial.cipher, trial.visible_codebook, model[0], model[1]
    )
    metrics = boundary_metrics(trial.boundaries, boundaries)
    return {
        "iso": trial.iso,
        "split": trial.split,
        "replicate": trial.replicate,
        "plaintext_length": len(trial.plain),
        "cipher_length": len(trial.cipher),
        "true_units": len(trial.units),
        "predicted_units": len(boundaries),
        "inventory_size": trial.inventory_size,
        "boundary_precision": metrics["precision"],
        "boundary_recall": metrics["recall"],
        "boundary_f1": metrics["f1"],
        "plaintext_accuracy": accuracy(trial.plain, predicted),
        "plaintext_exact": predicted == trial.plain,
        "score": score,
        "elapsed_seconds": time.perf_counter() - started,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    f1 = [float(row["boundary_f1"]) for row in rows]
    recovery = [float(row["plaintext_accuracy"]) for row in rows]
    return {
        "trials": len(rows),
        "boundary_f1": {
            "mean": statistics.fmean(f1),
            "median": statistics.median(f1),
            "minimum": min(f1),
        },
        "plaintext": {
            "mean": statistics.fmean(recovery),
            "median": statistics.median(recovery),
            "minimum": min(recovery),
            "at_least_90_rate": statistics.fmean(value >= 0.90 for value in recovery),
            "exact_rate": statistics.fmean(row["plaintext_exact"] for row in rows),
        },
        "gate": {
            "pass": (
                statistics.fmean(f1) >= 0.95
                and min(f1) >= 0.85
                and statistics.fmean(recovery) >= 0.95
                and sum(value >= 0.90 for value in recovery) >= 14
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
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        root / "corpus_manifest_v050.json", args.repo / ".cache" / "v060-family-s1"
    )
    language = languages[args.iso]
    model = mono.build_language_model(language)
    trials = [
        make_trial(language, args.split, args.length, replicate)
        for replicate in range(args.replicates)
    ]

    def run_one(trial: SegmentationTrial) -> dict[str, Any]:
        row = solve_trial(trial, model)
        print("V060_S1_TRIAL", json.dumps(row, sort_keys=True), flush=True)
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
    print("V060_S1_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V060_S1_SHA256", payload["sha256"], flush=True)


if __name__ == "__main__":
    main()
