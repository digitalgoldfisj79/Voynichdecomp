#!/usr/bin/env python3
"""v0.5.1 monoalphabetic specialist using simulated annealing.

The solver uses only character statistics from the pinned corpus training split.
All keys and source chunks are held out.  Development selects search effort;
test is evaluated once with the selected setting.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numba import njit, prange
from rapidfuzz.distance import Levenshtein

import recoverability_v050 as core


@njit(cache=True)
def score_mapping(cipher: np.ndarray, mapping: np.ndarray, logp3: np.ndarray) -> float:
    if cipher.size < 3:
        return 0.0
    total = 0.0
    a = mapping[cipher[0]]
    b = mapping[cipher[1]]
    for index in range(2, cipher.size):
        c = mapping[cipher[index]]
        total += logp3[a, b, c]
        a, b = b, c
    return total


@njit(cache=True)
def xorshift64(state: np.uint64) -> np.uint64:
    state ^= state << np.uint64(13)
    state ^= state >> np.uint64(7)
    state ^= state << np.uint64(17)
    return state


@njit(cache=True)
def rand_float(state: np.uint64) -> tuple[np.uint64, float]:
    state = xorshift64(state)
    return state, float(state & np.uint64((1 << 53) - 1)) / float(1 << 53)


@njit(cache=True)
def rand_int(state: np.uint64, n: int) -> tuple[np.uint64, int]:
    state, value = rand_float(state)
    return state, min(n - 1, int(value * n))


@njit(cache=True)
def anneal_once(
    cipher: np.ndarray,
    initial: np.ndarray,
    logp3: np.ndarray,
    iterations: int,
    start_temp: float,
    end_temp: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    mapping = initial.copy()
    best = mapping.copy()
    score = score_mapping(cipher, mapping, logp3)
    best_score = score
    state = np.uint64(seed + 1)
    alphabet = mapping.size
    ratio = (end_temp / start_temp) ** (1.0 / max(1, iterations - 1))
    temperature = start_temp

    for _ in range(iterations):
        state, i = rand_int(state, alphabet)
        state, j = rand_int(state, alphabet - 1)
        if j >= i:
            j += 1
        old_i = mapping[i]
        old_j = mapping[j]
        mapping[i] = old_j
        mapping[j] = old_i
        proposed = score_mapping(cipher, mapping, logp3)
        delta = proposed - score
        accept = delta >= 0.0
        if not accept:
            state, value = rand_float(state)
            accept = value < math.exp(delta / max(temperature, 1e-9))
        if accept:
            score = proposed
            if score > best_score:
                best_score = score
                best = mapping.copy()
        else:
            mapping[i] = old_i
            mapping[j] = old_j
        temperature *= ratio
    return best, best_score


@njit(cache=True, parallel=True)
def multi_restart(
    cipher: np.ndarray,
    initials: np.ndarray,
    logp3: np.ndarray,
    iterations: int,
    start_temp: float,
    end_temp: float,
    seeds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    restarts = initials.shape[0]
    mappings = np.empty_like(initials)
    scores = np.empty(restarts, dtype=np.float64)
    for index in prange(restarts):
        mapping, score = anneal_once(
            cipher,
            initials[index],
            logp3,
            iterations,
            start_temp,
            end_temp,
            int(seeds[index]),
        )
        mappings[index] = mapping
        scores[index] = score
    return mappings, scores


def build_trigram_model(language: core.LanguageData, alpha: float = 0.05) -> np.ndarray:
    alphabet = len(language.alphabet)
    counts = np.full((alphabet, alphabet, alphabet), alpha, dtype=np.float64)
    stream = language.train_stream
    for a, b, c in zip(stream, stream[1:], stream[2:]):
        counts[a, b, c] += 1.0
    denominators = counts.sum(axis=2, keepdims=True)
    return np.log(counts / denominators)


def frequency_initial(
    cipher: Sequence[int],
    language: core.LanguageData,
) -> np.ndarray:
    alphabet = len(language.alphabet)
    cipher_counts = Counter(int(value) for value in cipher)
    plain_counts = Counter(language.train_stream)
    cipher_order = [value for value, _ in cipher_counts.most_common()]
    cipher_order.extend(value for value in range(alphabet) if value not in cipher_counts)
    plain_order = [value for value, _ in plain_counts.most_common()]
    plain_order.extend(value for value in range(alphabet) if value not in plain_counts)
    mapping = np.zeros(alphabet, dtype=np.int64)
    for cipher_symbol, plain_symbol in zip(cipher_order, plain_order):
        mapping[cipher_symbol] = plain_symbol
    return mapping


def randomised_initials(
    base_mapping: np.ndarray,
    restarts: int,
    seed: int,
) -> np.ndarray:
    rng = random.Random(seed)
    rows = []
    for restart in range(restarts):
        mapping = base_mapping.copy()
        swaps = 0 if restart == 0 else rng.randint(2, max(3, len(mapping) // 2))
        for _ in range(swaps):
            i, j = rng.sample(range(len(mapping)), 2)
            mapping[i], mapping[j] = mapping[j], mapping[i]
        rows.append(mapping)
    return np.stack(rows)


def generated_plain(
    language: core.LanguageData,
    length: int,
    family: str,
    rng: random.Random,
) -> list[int]:
    if family == "motif":
        return core.motif_generate(language, length, rng)
    if family == "copy_mutate":
        return core.copy_mutate_generate(language, length, rng)
    # iid is an intentionally hostile non-language stress control for this solver.
    return [core.weighted_choice(rng, language.probabilities) for _ in range(length)]


def solve_trial(
    cipher: Sequence[int],
    truth: Sequence[int],
    language: core.LanguageData,
    logp3: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    cipher_array = np.asarray(cipher, dtype=np.int64)
    initial = frequency_initial(cipher, language)
    initials = randomised_initials(initial, int(config["restarts"]), seed)
    seeds = np.asarray([core.stable_seed(seed, index) & 0x7FFFFFFF for index in range(len(initials))], dtype=np.int64)
    mappings, scores = multi_restart(
        cipher_array,
        initials,
        logp3,
        int(config["iterations"]),
        float(config["start_temp"]),
        float(config["end_temp"]),
        seeds,
    )
    best_index = int(np.argmax(scores))
    mapping = mappings[best_index]
    decoded = [int(mapping[int(value)]) for value in cipher]
    distance = int(Levenshtein.distance(list(truth), decoded))
    accuracy = max(0.0, 1.0 - distance / max(1, len(truth), len(decoded)))
    return {
        "accuracy": accuracy,
        "exact": list(truth) == decoded,
        "score_per_trigram": float(scores[best_index] / max(1, len(cipher) - 2)),
    }


def make_trials(
    languages: dict[str, core.LanguageData],
    split: str,
    replicates: int,
    lengths: Sequence[int],
    source_types: Sequence[str],
    seed: int,
) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for iso, language in sorted(languages.items()):
        for length in lengths:
            chunks = core.source_chunks(language, split, length)
            for source_type in source_types:
                for replicate in range(replicates):
                    rng = random.Random(core.stable_seed("mono-v051", seed, split, iso, length, source_type, replicate))
                    if source_type == "natural":
                        plain = list(chunks[replicate % len(chunks)])
                        generator = None
                    else:
                        generator = ("motif", "copy_mutate", "iid")[replicate % 3]
                        plain = generated_plain(language, length, generator, rng)
                    packet = core.encrypt_sequence(plain, "mono", language, rng, parameter_mode=split)
                    trials.append({
                        "iso": iso,
                        "length": length,
                        "source_type": source_type,
                        "generator": generator,
                        "replicate": replicate,
                        "plain": plain,
                        "cipher": packet.cipher,
                    })
    return trials


def evaluate_config(
    trials: Sequence[dict[str, Any]],
    languages: dict[str, core.LanguageData],
    models: dict[str, np.ndarray],
    config: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for index, trial in enumerate(trials):
        result = solve_trial(
            trial["cipher"],
            trial["plain"],
            languages[trial["iso"]],
            models[trial["iso"]],
            config,
            core.stable_seed(seed, index),
        )
        rows.append({key: value for key, value in trial.items() if key not in ("plain", "cipher")} | result)
        if (index + 1) % 20 == 0:
            print("V051_MONO_PROGRESS", index + 1, len(trials), flush=True)
    return rows


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def grouped(field: str) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        return {
            key: {
                "trials": len(subset),
                "mean_accuracy": statistics.fmean(row["accuracy"] for row in subset),
                "median_accuracy": statistics.median(row["accuracy"] for row in subset),
                "exact_rate": statistics.fmean(float(row["exact"]) for row in subset),
                "mean_score_per_trigram": statistics.fmean(row["score_per_trigram"] for row in subset),
            }
            for key, subset in sorted(groups.items())
        }
    return {
        "trials": len(rows),
        "mean_accuracy": statistics.fmean(row["accuracy"] for row in rows),
        "median_accuracy": statistics.median(row["accuracy"] for row in rows),
        "exact_rate": statistics.fmean(float(row["exact"]) for row in rows),
        "by_language": grouped("iso"),
        "by_length": grouped("length"),
        "by_source_type": grouped("source_type"),
    }


def emit(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    encoded = base64.b64encode(compressed).decode("ascii")
    parts = [encoded[index:index + 60000] for index in range(0, len(encoded), 60000)]
    metadata = {
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "raw_bytes": len(raw),
        "compressed_bytes": len(compressed),
        "parts": len(parts),
    }
    print("V051_MONO_ARTIFACT_META " + json.dumps(metadata, sort_keys=True), flush=True)
    for index, part in enumerate(parts):
        print(f"V051_MONO_ARTIFACT_PART {index:04d}/{len(parts):04d} {part}", flush=True)
    print("V051_MONO_ARTIFACT_END " + metadata["raw_sha256"], flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=505111)
    parser.add_argument("--dev-replicates", type=int, default=4)
    parser.add_argument("--test-replicates", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    root = args.repo / "experiments/recoverability_frontier_v0_5"
    languages = core.load_languages(root / "corpus_manifest_v050.json", args.repo / ".cache/ud-v050")
    lengths = (96, 192, 384)
    if args.smoke:
        languages = {iso: languages[iso] for iso in ("en", "tr")}
        lengths = (96,)
        args.dev_replicates = 1
        args.test_replicates = 1

    models = {iso: build_trigram_model(language) for iso, language in languages.items()}
    configs = [
        {"name": "fast", "restarts": 8, "iterations": 4000, "start_temp": 8.0, "end_temp": 0.05},
        {"name": "deep", "restarts": 16, "iterations": 10000, "start_temp": 10.0, "end_temp": 0.02},
    ]
    if args.smoke:
        configs = [{"name": "smoke", "restarts": 2, "iterations": 200, "start_temp": 5.0, "end_temp": 0.1}]

    dev_trials = make_trials(languages, "dev", args.dev_replicates, lengths, ("natural",), args.seed)
    development = []
    best_config = None
    best_accuracy = -1.0
    for config in configs:
        rows = evaluate_config(dev_trials, languages, models, config, core.stable_seed(args.seed, config["name"]))
        summary = summarize(rows)
        development.append({"config": config, "summary": summary})
        print("V051_MONO_DEV", json.dumps(development[-1], sort_keys=True), flush=True)
        if summary["mean_accuracy"] > best_accuracy:
            best_accuracy = summary["mean_accuracy"]
            best_config = config

    assert best_config is not None
    test_trials = make_trials(
        languages,
        "test",
        args.test_replicates,
        lengths,
        ("natural", "generated"),
        args.seed,
    )
    test_rows = evaluate_config(test_trials, languages, models, best_config, core.stable_seed(args.seed, "test"))
    test = summarize(test_rows)
    natural_rows = [row for row in test_rows if row["source_type"] == "natural"]
    natural = summarize(natural_rows)
    gate = {
        "natural_mean_accuracy_over_70": natural["mean_accuracy"] >= 0.70,
        "every_length_noiseless_over_50": all(
            row["mean_accuracy"] >= 0.50 for row in natural["by_length"].values()
        ),
    }
    gate["pass"] = all(gate.values())
    payload = {
        "programme": "recoverability-frontier-v0.5.1-mono-specialist",
        "config": vars(args),
        "development": development,
        "selected_config": best_config,
        "test": test,
        "test_natural": natural,
        "gate": gate,
        "rows": test_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, args.output)
    print("V051_MONO_TEST", json.dumps({"selected_config": best_config, "test": test, "test_natural": natural, "gate": gate}, sort_keys=True), flush=True)
    emit(args.output)


if __name__ == "__main__":
    main()
