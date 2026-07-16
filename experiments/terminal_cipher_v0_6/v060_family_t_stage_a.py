#!/usr/bin/env python3
"""v0.6 Family T: bounded ragged columnar component oracles."""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
V05 = HERE.parent / "recoverability_frontier_v0_5"
if str(V05) not in sys.path:
    sys.path.insert(0, str(V05))

import recoverability_v050 as core
import mono_solver_v051 as mono

MODES = ("global", "line_reset")
DEV_WIDTHS = tuple(range(4, 11))
TEST_WIDTHS = tuple(range(11, 15))


@dataclasses.dataclass
class ColumnarTrial:
    iso: str
    split: str
    replicate: int
    seed: int
    plain: list[int]
    cipher: list[int]
    substitution_inverse: list[int]
    mode: str
    width: int
    permutation: list[int]
    line_starts: list[int]


def make_line_starts(rng: random.Random, length: int) -> list[int]:
    starts = [0]
    cursor = 0
    while cursor < length:
        cursor += rng.randint(40, 72)
        if cursor < length:
            starts.append(cursor)
    return starts


def encrypt_segment(values: list[int], width: int, permutation: list[int]) -> list[int]:
    out: list[int] = []
    for column in permutation:
        out.extend(values[index] for index in range(column, len(values), width))
    return out


def encrypt_columnar(
    values: list[int], width: int, permutation: list[int], mode: str,
    line_starts: list[int],
) -> list[int]:
    if mode == "global":
        return encrypt_segment(values, width, permutation)
    starts = line_starts + [len(values)]
    out: list[int] = []
    for left, right in zip(starts, starts[1:]):
        out.extend(encrypt_segment(values[left:right], width, permutation))
    return out


def make_trial(
    language: core.LanguageData, split: str, length: int, mode: str, replicate: int
) -> ColumnarTrial:
    chunks = core.source_chunks(language, split, length)
    if not chunks:
        raise RuntimeError(f"no chunks for {language.iso}/{split}/{length}")
    plain = list(chunks[replicate % len(chunks)])
    seed = core.stable_seed("v060-family-t", language.iso, split, length, mode, replicate)
    rng = random.Random(seed)
    widths = TEST_WIDTHS if split == "test" else DEV_WIDTHS
    width = rng.choice(widths)
    permutation = list(range(width))
    while permutation == list(range(width)):
        rng.shuffle(permutation)
    mapping = list(range(len(language.alphabet)))
    rng.shuffle(mapping)
    inverse = [0] * len(mapping)
    for plain_symbol, cipher_symbol in enumerate(mapping):
        inverse[cipher_symbol] = plain_symbol
    substituted = [mapping[value] for value in plain]
    line_rng = random.Random(core.stable_seed("v060-family-t-lines", seed))
    line_starts = make_line_starts(line_rng, length)
    cipher = encrypt_columnar(substituted, width, permutation, mode, line_starts)
    return ColumnarTrial(
        iso=language.iso,
        split=split,
        replicate=replicate,
        seed=seed,
        plain=plain,
        cipher=cipher,
        substitution_inverse=inverse,
        mode=mode,
        width=width,
        permutation=permutation,
        line_starts=line_starts,
    )


@njit(cache=True, nogil=True)
def decrypt_segment_into(
    cipher: np.ndarray, cipher_offset: int, length: int, width: int,
    permutation: np.ndarray, output: np.ndarray, output_offset: int,
) -> None:
    base_length = length // width
    remainder = length % width
    cursor = cipher_offset
    for order_index in range(width):
        column = permutation[order_index]
        column_length = base_length + (1 if column < remainder else 0)
        for row in range(column_length):
            target = output_offset + row * width + column
            if target < output_offset + length:
                output[target] = cipher[cursor]
            cursor += 1


@njit(cache=True, nogil=True)
def decrypt_columnar_array(
    cipher: np.ndarray, width: int, permutation: np.ndarray,
    mode_flag: int, line_starts: np.ndarray,
) -> np.ndarray:
    output = np.empty(cipher.shape[0], dtype=np.int32)
    if mode_flag == 0:
        decrypt_segment_into(
            cipher, 0, cipher.shape[0], width, permutation, output, 0
        )
        return output
    cursor = 0
    for line_index in range(line_starts.shape[0] - 1):
        left = line_starts[line_index]
        right = line_starts[line_index + 1]
        length = right - left
        decrypt_segment_into(
            cipher, cursor, length, width, permutation, output, left
        )
        cursor += length
    return output


@njit(cache=True, nogil=True)
def score_plain(
    values: np.ndarray, trigram: np.ndarray, unigram: np.ndarray
) -> float:
    if values.shape[0] == 0:
        return -1e300
    score = 0.15 * unigram[values[0]]
    if values.shape[0] >= 2:
        score += 0.15 * unigram[values[1]]
    for index in range(2, values.shape[0]):
        score += trigram[values[index - 2], values[index - 1], values[index]]
        score += 0.15 * unigram[values[index]]
    return score


@njit(cache=True, nogil=True)
def score_permutation(
    cipher: np.ndarray, width: int, permutation: np.ndarray, mode_flag: int,
    line_starts: np.ndarray, trigram: np.ndarray, unigram: np.ndarray,
) -> float:
    plain = decrypt_columnar_array(cipher, width, permutation, mode_flag, line_starts)
    return score_plain(plain, trigram, unigram)


@njit(cache=True, nogil=True)
def anneal_permutation(
    cipher: np.ndarray, width: int, mode_flag: int, line_starts: np.ndarray,
    trigram: np.ndarray, unigram: np.ndarray, iterations: int, restarts: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    state = np.uint64(seed if seed > 0 else 1)
    best = np.arange(width, dtype=np.int32)
    best_score = score_permutation(
        cipher, width, best, mode_flag, line_starts, trigram, unigram
    )
    for restart in range(restarts):
        permutation = np.arange(width, dtype=np.int32)
        for _ in range(width + restart * 2):
            state, first = mono._rng_int(state, width)
            state, second = mono._rng_int(state, width)
            if first != second:
                temporary = permutation[first]
                permutation[first] = permutation[second]
                permutation[second] = temporary
        current = score_permutation(
            cipher, width, permutation, mode_flag, line_starts, trigram, unigram
        )
        if current > best_score:
            best_score = current
            best = permutation.copy()
        temperature = 12.0
        cooling = math.exp(math.log(0.05 / 12.0) / max(1, iterations))
        for _ in range(iterations):
            state, first = mono._rng_int(state, width)
            state, second = mono._rng_int(state, width)
            if first == second:
                continue
            temporary = permutation[first]
            permutation[first] = permutation[second]
            permutation[second] = temporary
            candidate = score_permutation(
                cipher, width, permutation, mode_flag, line_starts,
                trigram, unigram
            )
            delta = candidate - current
            accept = delta >= 0.0
            if not accept:
                state, uniform = mono._rng_float(state)
                accept = uniform < math.exp(delta / max(temperature, 1e-9))
            if accept:
                current = candidate
                if current > best_score:
                    best_score = current
                    best = permutation.copy()
            else:
                temporary = permutation[first]
                permutation[first] = permutation[second]
                permutation[second] = temporary
            temperature *= cooling
    return best, best_score


def canonical_permutation(permutation: list[int]) -> list[int]:
    # Column order is directly identifiable for ragged columnar transposition.
    return list(permutation)


def mdl_score(raw: float, width: int, length: int) -> float:
    return raw - math.lgamma(width + 1) - 0.5 * math.log(max(2, length))


def solve_trial(
    trial: ColumnarTrial,
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    perm_iterations: int,
    perm_restarts: int,
    mono_iterations: int,
    mono_restarts: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    trigram, unigram = model
    line_starts = np.asarray(trial.line_starts + [trial.length], dtype=np.int32)
    cipher = np.asarray(trial.cipher, dtype=np.int32)
    mapped = np.asarray(
        [trial.substitution_inverse[value] for value in trial.cipher], dtype=np.int32
    )
    candidates = []
    widths = TEST_WIDTHS if trial.split == "test" else DEV_WIDTHS
    for mode in MODES:
        mode_flag = 0 if mode == "global" else 1
        for width in widths:
            permutation, raw = anneal_permutation(
                mapped,
                width,
                mode_flag,
                line_starts,
                trigram,
                unigram,
                perm_iterations,
                perm_restarts,
                int(core.stable_seed(
                    "v060-t1", trial.seed, mode, width
                ) & 0x7FFFFFFFFFFFFFFF),
            )
            prediction = decrypt_columnar_array(
                mapped, width, permutation, mode_flag, line_starts
            ).tolist()
            candidates.append({
                "mode": mode,
                "width": width,
                "permutation": permutation.tolist(),
                "score": mdl_score(float(raw), width, trial.length),
                "accuracy": mono.fast_accuracy(trial.plain, prediction),
            })
    selected = max(candidates, key=lambda row: row["score"])

    true_mode_flag = 0 if trial.mode == "global" else 1
    true_permutation = np.asarray(trial.permutation, dtype=np.int32)
    detransposed = decrypt_columnar_array(
        cipher,
        trial.width,
        true_permutation,
        true_mode_flag,
        line_starts,
    )
    initial = mono.frequency_key(detransposed.tolist(), language)
    solved_key, mono_raw = mono.anneal_mono(
        detransposed,
        initial,
        trigram,
        unigram,
        mono_iterations,
        mono_restarts,
        int(core.stable_seed("v060-t2", trial.seed) & 0x7FFFFFFFFFFFFFFF),
    )
    mono_prediction = solved_key[detransposed].tolist()
    return {
        "iso": trial.iso,
        "split": trial.split,
        "replicate": trial.replicate,
        "true_mode": trial.mode,
        "true_width": trial.width,
        "selected_mode": selected["mode"],
        "selected_width": selected["width"],
        "mode_correct": selected["mode"] == trial.mode,
        "width_correct": selected["width"] == trial.width,
        "permutation_correct": (
            selected["mode"] == trial.mode
            and selected["width"] == trial.width
            and canonical_permutation(selected["permutation"])
            == canonical_permutation(trial.permutation)
        ),
        "t1_accuracy": selected["accuracy"],
        "t2_accuracy": mono.fast_accuracy(trial.plain, mono_prediction),
        "t2_exact": mono_prediction == trial.plain,
        "t2_score": float(mono_raw),
        "elapsed_seconds": time.perf_counter() - started,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    t1 = [float(row["t1_accuracy"]) for row in rows]
    t2 = [float(row["t2_accuracy"]) for row in rows]
    return {
        "trials": len(rows),
        "t1": {
            "mean": statistics.fmean(t1),
            "median": statistics.median(t1),
            "minimum": min(t1),
            "mode_accuracy": statistics.fmean(row["mode_correct"] for row in rows),
            "width_accuracy": statistics.fmean(row["width_correct"] for row in rows),
            "permutation_accuracy": statistics.fmean(
                row["permutation_correct"] for row in rows
            ),
        },
        "t2": {
            "mean": statistics.fmean(t2),
            "median": statistics.median(t2),
            "minimum": min(t2),
            "at_least_95_rate": statistics.fmean(value >= 0.95 for value in t2),
            "exact_rate": statistics.fmean(row["t2_exact"] for row in rows),
        },
        "gates": {
            "t1_pass": (
                statistics.fmean(t1) >= 0.95
                and min(t1) >= 0.85
                and sum(row["mode_correct"] for row in rows) >= 14
                and sum(row["width_correct"] for row in rows) >= 14
                and sum(row["permutation_correct"] for row in rows) >= 12
            ),
            "t2_pass": (
                statistics.fmean(t2) >= 0.95
                and min(t2) >= 0.90
                and sum(value >= 0.95 for value in t2) >= 14
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iso", default="en")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--length", type=int, default=384)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--perm-iterations", type=int, default=200000)
    parser.add_argument("--perm-restarts", type=int, default=32)
    parser.add_argument("--mono-iterations", type=int, default=700000)
    parser.add_argument("--mono-restarts", type=int, default=50)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        root / "corpus_manifest_v050.json", args.repo / ".cache" / "v060-family-t"
    )
    language = languages[args.iso]
    model = mono.build_language_model(language)
    trials = [
        make_trial(language, args.split, args.length, mode, replicate)
        for mode in MODES
        for replicate in range(args.replicates)
    ]

    def run_one(trial: ColumnarTrial) -> dict[str, Any]:
        row = solve_trial(
            trial, language, model,
            args.perm_iterations, args.perm_restarts,
            args.mono_iterations, args.mono_restarts,
        )
        print("V060_TA_TRIAL", json.dumps(row, sort_keys=True), flush=True)
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
    print("V060_TA_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V060_TA_SHA256", payload["sha256"], flush=True)


if __name__ == "__main__":
    main()
