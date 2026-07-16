#!/usr/bin/env python3
"""v0.5.4 nomenclator component-oracle recoverability gates.

A1 supplies the character key and recovers observed nomenclator code words.
A2 supplies the code-word map and observed plaintext-character inventory and
recovers the residual monoalphabetic key around fixed plaintext spans.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numba import njit

import recoverability_v050 as core
import homophonic_solver_v052 as homophonic
import mono_solver_v051 as mono
from homophonic_confirm_v052_quadgram import build_quadgram_model, quadgram_score_key


@dataclasses.dataclass
class NomenclatorTrial:
    iso: str
    split: str
    target_length: int
    replicate: int
    seed: int
    plain: list[int]
    surface: list[int]
    char_to_plain: dict[int, int]
    code_to_word: dict[int, tuple[int, ...]]
    char_symbols: tuple[int, ...]
    code_symbols: tuple[int, ...]
    observed_plain_inventory: tuple[int, ...]
    selected_codebook_size: int


@dataclasses.dataclass
class WordModel:
    candidate_words: tuple[tuple[int, ...], ...]
    candidate_frequencies: np.ndarray
    word_to_id: dict[tuple[int, ...], int]
    unigram_logp: np.ndarray
    bigram_logp: dict[tuple[int, int], float]
    bigram_default: np.ndarray
    trigram_logp: dict[tuple[int, int, int], float]
    trigram_default: dict[tuple[int, int], float]
    unknown_id: int
    bos_id: int

    def score(self, words: list[tuple[int, ...]]) -> float:
        first = self.bos_id
        second = self.bos_id
        total = 0.0
        for word in words:
            current = self.word_to_id.get(word, self.unknown_id)
            total += 0.20 * float(self.unigram_logp[current])
            total += 0.55 * self.bigram_logp.get(
                (second, current), float(self.bigram_default[second])
            )
            total += self.trigram_logp.get(
                (first, second, current),
                self.trigram_default.get(
                    (first, second), -math.log(max(2, len(self.unigram_logp)))
                ),
            )
            first, second = second, current
        return total


def encode_word(language: core.LanguageData, value: str) -> tuple[int, ...]:
    return tuple(language.char_to_id[ch] for ch in value if ch in language.char_to_id)


def word_aligned_chunks(
    language: core.LanguageData, split: str, target_length: int
) -> list[list[int]]:
    space = language.char_to_id[" "]
    chunks: list[list[int]] = []
    current_words: list[tuple[int, ...]] = []
    current_length = 0
    for text in language.texts[split]:
        for raw_word in text.split():
            word = encode_word(language, raw_word)
            if not word:
                continue
            added = len(word) + (1 if current_words else 0)
            current_words.append(word)
            current_length += added
            if current_length >= target_length:
                values: list[int] = []
                for index, item in enumerate(current_words):
                    if index:
                        values.append(space)
                    values.extend(item)
                chunks.append(values)
                current_words = []
                current_length = 0
    return chunks


def split_words(values: Iterable[int], space: int) -> list[tuple[int, ...]]:
    words: list[tuple[int, ...]] = []
    current: list[int] = []
    for value in values:
        if int(value) == space:
            if current:
                words.append(tuple(current))
                current = []
        else:
            current.append(int(value))
    if current:
        words.append(tuple(current))
    return words


def weighted_sample_without_replacement(
    items: list[tuple[int, ...]], weights: list[float], count: int, rng: random.Random
) -> list[tuple[int, ...]]:
    available_items = list(items)
    available_weights = list(weights)
    selected: list[tuple[int, ...]] = []
    for _ in range(min(count, len(available_items))):
        total = sum(available_weights)
        draw = rng.random() * total
        cumulative = 0.0
        chosen = len(available_items) - 1
        for index, weight in enumerate(available_weights):
            cumulative += weight
            if draw <= cumulative:
                chosen = index
                break
        selected.append(available_items.pop(chosen))
        available_weights.pop(chosen)
    return selected


def build_word_model(
    language: core.LanguageData, candidate_pool_size: int = 96, vocabulary_size: int = 6000
) -> WordModel:
    train_counts = collections.Counter(language.train_words)
    candidates = [
        word for word, _count in train_counts.most_common() if len(word) >= 2
    ][:candidate_pool_size]
    candidate_frequencies = np.asarray(
        [float(train_counts[word]) for word in candidates], dtype=np.float64
    )
    candidate_frequencies /= candidate_frequencies.sum()

    vocabulary_words = [
        word for word, _count in train_counts.most_common(max(1, vocabulary_size - 2))
    ]
    unknown_id = 0
    bos_id = 1
    word_to_id = {word: index + 2 for index, word in enumerate(vocabulary_words)}
    vocab = len(word_to_id) + 2
    unigram = np.full(vocab, 0.10, dtype=np.float64)
    bigram_counts: collections.Counter[tuple[int, int]] = collections.Counter()
    bigram_context: collections.Counter[int] = collections.Counter()
    trigram_counts: collections.Counter[tuple[int, int, int]] = collections.Counter()
    trigram_context: collections.Counter[tuple[int, int]] = collections.Counter()

    for text in language.texts["train"]:
        ids = [
            word_to_id.get(encode_word(language, raw), unknown_id)
            for raw in text.split()
            if encode_word(language, raw)
        ]
        first = bos_id
        second = bos_id
        unigram[bos_id] += 2.0
        for current in ids:
            unigram[current] += 1.0
            bigram_counts[(second, current)] += 1
            bigram_context[second] += 1
            trigram_counts[(first, second, current)] += 1
            trigram_context[(first, second)] += 1
            first, second = second, current

    unigram_logp = np.log(unigram / unigram.sum())
    alpha = 0.10
    bigram_default = np.empty(vocab, dtype=np.float64)
    for previous in range(vocab):
        bigram_default[previous] = math.log(
            alpha / (bigram_context[previous] + alpha * vocab)
        )
    bigram_logp = {
        key: math.log((count + alpha) / (bigram_context[key[0]] + alpha * vocab))
        for key, count in bigram_counts.items()
    }
    trigram_default = {
        context: math.log(alpha / (count + alpha * vocab))
        for context, count in trigram_context.items()
    }
    trigram_logp = {
        key: math.log(
            (count + alpha) / (trigram_context[(key[0], key[1])] + alpha * vocab)
        )
        for key, count in trigram_counts.items()
    }
    return WordModel(
        candidate_words=tuple(candidates),
        candidate_frequencies=candidate_frequencies,
        word_to_id=word_to_id,
        unigram_logp=unigram_logp,
        bigram_logp=bigram_logp,
        bigram_default=bigram_default,
        trigram_logp=trigram_logp,
        trigram_default=trigram_default,
        unknown_id=unknown_id,
        bos_id=bos_id,
    )


def make_trial(
    language: core.LanguageData,
    word_model: WordModel,
    split: str,
    target_length: int,
    replicate: int,
    codebook_size: int,
) -> NomenclatorTrial:
    chunks = word_aligned_chunks(language, split, target_length)
    if not chunks:
        raise RuntimeError(f"no word-aligned chunks for {language.iso}/{split}")
    plain = list(chunks[replicate % len(chunks)])
    seed = core.stable_seed(
        "v054-nomenclator", language.iso, split, target_length, replicate, codebook_size
    )
    rng = random.Random(seed)
    selected = weighted_sample_without_replacement(
        list(word_model.candidate_words),
        [math.sqrt(max(value, 1e-12)) for value in word_model.candidate_frequencies],
        codebook_size,
        rng,
    )
    selected_set = set(selected)
    alphabet_size = len(language.alphabet)
    character_permutation = list(range(alphabet_size))
    rng.shuffle(character_permutation)
    internal_code = {word: alphabet_size + index for index, word in enumerate(selected)}
    total_symbols = alphabet_size + len(selected)
    joint_labels = list(range(total_symbols))
    rng.shuffle(joint_labels)

    space = language.char_to_id[" "]
    plaintext_words = split_words(plain, space)
    internal_stream: list[int] = []
    for index, word in enumerate(plaintext_words):
        if index:
            internal_stream.append(character_permutation[space])
        if word in selected_set:
            internal_stream.append(internal_code[word])
        else:
            internal_stream.extend(character_permutation[value] for value in word)
    raw_surface = [joint_labels[value] for value in internal_stream]
    surface, canonical_to_raw = homophonic.canonicalize_with_inverse(raw_surface)
    raw_to_canonical = {raw: canonical for canonical, raw in enumerate(canonical_to_raw)}

    char_to_plain: dict[int, int] = {}
    for plain_value, character_internal in enumerate(character_permutation):
        raw = joint_labels[character_internal]
        if raw in raw_to_canonical:
            char_to_plain[raw_to_canonical[raw]] = plain_value
    code_to_word: dict[int, tuple[int, ...]] = {}
    for word, code_internal in internal_code.items():
        raw = joint_labels[code_internal]
        if raw in raw_to_canonical:
            code_to_word[raw_to_canonical[raw]] = tuple(word)

    return NomenclatorTrial(
        iso=language.iso,
        split=split,
        target_length=target_length,
        replicate=replicate,
        seed=seed,
        plain=plain,
        surface=surface,
        char_to_plain=char_to_plain,
        code_to_word=code_to_word,
        char_symbols=tuple(sorted(char_to_plain)),
        code_symbols=tuple(sorted(code_to_word)),
        observed_plain_inventory=tuple(sorted(char_to_plain.values())),
        selected_codebook_size=len(selected),
    )


def parse_word_units(
    trial: NomenclatorTrial, char_mapping: dict[int, int], space: int
) -> list[tuple[str, Any]]:
    units: list[tuple[str, Any]] = []
    current: list[int] = []
    code_set = set(trial.code_symbols)
    for symbol in trial.surface:
        if symbol in code_set:
            if current:
                units.append(("word", tuple(current)))
                current = []
            units.append(("code", int(symbol)))
            continue
        value = int(char_mapping[int(symbol)])
        if value == space:
            if current:
                units.append(("word", tuple(current)))
                current = []
        else:
            current.append(value)
    if current:
        units.append(("word", tuple(current)))
    return units


def score_code_assignment(
    units: list[tuple[str, Any]], assignment: dict[int, tuple[int, ...]], model: WordModel
) -> float:
    words: list[tuple[int, ...]] = []
    for kind, value in units:
        if kind == "word":
            words.append(value)
        else:
            candidate = assignment.get(int(value))
            if candidate is None:
                return -1e300
            words.append(candidate)
    return model.score(words)


def optimize_code_assignment(
    units: list[tuple[str, Any]],
    code_symbols: tuple[int, ...],
    model: WordModel,
    seed: int,
    restarts: int,
    sweeps: int,
) -> tuple[dict[int, tuple[int, ...]], float]:
    if not code_symbols:
        return {}, model.score([value for kind, value in units if kind == "word"])
    occurrence_counts = collections.Counter(
        int(value) for kind, value in units if kind == "code"
    )
    ranked_codes = sorted(code_symbols, key=lambda value: (-occurrence_counts[value], value))
    ranked_words = list(model.candidate_words)
    rng = random.Random(seed)
    best_assignment: dict[int, tuple[int, ...]] = {}
    best_score = -1e300

    for restart in range(restarts):
        if restart == 0:
            chosen = ranked_words[: len(ranked_codes)]
        else:
            chosen = weighted_sample_without_replacement(
                ranked_words,
                [float(value) for value in model.candidate_frequencies],
                len(ranked_codes),
                rng,
            )
            rng.shuffle(chosen)
        assignment = {code: word for code, word in zip(ranked_codes, chosen)}
        current_score = score_code_assignment(units, assignment, model)
        for _ in range(sweeps):
            improved = False
            order = list(ranked_codes)
            rng.shuffle(order)
            for code in order:
                previous = assignment[code]
                used_elsewhere = {
                    word for other, word in assignment.items() if other != code
                }
                local_best = previous
                local_score = current_score
                for candidate in ranked_words:
                    if candidate in used_elsewhere:
                        continue
                    assignment[code] = candidate
                    candidate_score = score_code_assignment(units, assignment, model)
                    if candidate_score > local_score:
                        local_score = candidate_score
                        local_best = candidate
                assignment[code] = local_best
                if local_score > current_score + 1e-9:
                    current_score = local_score
                    improved = True
            if not improved:
                break
        if current_score > best_score:
            best_score = current_score
            best_assignment = dict(assignment)
    return best_assignment, best_score


def expand_surface(
    trial: NomenclatorTrial,
    char_mapping: dict[int, int],
    code_mapping: dict[int, tuple[int, ...]],
) -> list[int]:
    output: list[int] = []
    code_set = set(trial.code_symbols)
    for symbol in trial.surface:
        if symbol in code_set:
            output.extend(code_mapping[int(symbol)])
        else:
            output.append(int(char_mapping[int(symbol)]))
    return output


def code_metrics(
    trial: NomenclatorTrial, predicted: dict[int, tuple[int, ...]]
) -> tuple[float, float]:
    if not trial.code_symbols:
        return 1.0, 1.0
    mapping_accuracy = statistics.fmean(
        predicted.get(symbol) == trial.code_to_word[symbol]
        for symbol in trial.code_symbols
    )
    occurrences = [symbol for symbol in trial.surface if symbol in trial.code_to_word]
    occurrence_accuracy = statistics.fmean(
        predicted.get(symbol) == trial.code_to_word[symbol] for symbol in occurrences
    ) if occurrences else 1.0
    return float(mapping_accuracy), float(occurrence_accuracy)


@njit(cache=True, nogil=True)
def anneal_locked_key(
    cipher: np.ndarray,
    initial_key: np.ndarray,
    swappable: int,
    quadgram_logp: np.ndarray,
    unigram_logp: np.ndarray,
    iterations: int,
    restarts: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    best_key = initial_key.copy()
    best_score = quadgram_score_key(cipher, best_key, quadgram_logp, unigram_logp)
    state = np.uint64(seed if seed > 0 else 1)
    for restart in range(restarts):
        key = initial_key.copy()
        if restart > 0:
            for index in range(swappable - 1, 0, -1):
                state, other = mono._rng_int(state, index + 1)
                temporary = key[index]
                key[index] = key[other]
                key[other] = temporary
        current_score = quadgram_score_key(cipher, key, quadgram_logp, unigram_logp)
        if current_score > best_score:
            best_score = current_score
            best_key = key.copy()
        temperature = 35.0
        cooling = math.exp(math.log(0.025 / 35.0) / max(1, iterations))
        stagnant = 0
        for _ in range(iterations):
            state, first = mono._rng_int(state, swappable)
            state, second = mono._rng_int(state, swappable)
            if first == second:
                continue
            temporary = key[first]
            key[first] = key[second]
            key[second] = temporary
            candidate_score = quadgram_score_key(
                cipher, key, quadgram_logp, unigram_logp
            )
            delta = candidate_score - current_score
            accept = delta >= 0.0
            if not accept:
                state, draw = mono._rng_float(state)
                accept = draw < math.exp(delta / max(temperature, 1e-12))
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
            if stagnant >= 20000:
                temperature = max(temperature, 3.0)
                stagnant = 0
        for _ in range(max(5000, iterations // 5)):
            state, first = mono._rng_int(state, swappable)
            state, second = mono._rng_int(state, swappable)
            if first == second:
                continue
            temporary = key[first]
            key[first] = key[second]
            key[second] = temporary
            candidate_score = quadgram_score_key(
                cipher, key, quadgram_logp, unigram_logp
            )
            if candidate_score >= current_score:
                current_score = candidate_score
                if current_score > best_score:
                    best_score = current_score
                    best_key = key.copy()
            else:
                temporary = key[first]
                key[first] = key[second]
                key[second] = temporary
    return best_key, best_score


def frequency_initial_key(
    trial: NomenclatorTrial, language: core.LanguageData
) -> tuple[list[int], np.ndarray]:
    symbols = list(trial.char_symbols)
    counts = collections.Counter(
        symbol for symbol in trial.surface if symbol in trial.char_to_plain
    )
    cipher_rank = sorted(symbols, key=lambda value: (-counts[value], value))
    inventory_rank = sorted(
        trial.observed_plain_inventory,
        key=lambda value: (-language.probabilities[value], value),
    )
    mapping = {symbol: label for symbol, label in zip(cipher_rank, inventory_rank)}
    key = np.asarray([mapping[symbol] for symbol in symbols], dtype=np.int32)
    return symbols, key


def build_mixed_cipher(
    trial: NomenclatorTrial,
    code_mapping: dict[int, tuple[int, ...]],
    alphabet_size: int,
) -> tuple[np.ndarray, list[int]]:
    char_symbols = list(trial.char_symbols)
    char_index = {symbol: index for index, symbol in enumerate(char_symbols)}
    variable_count = len(char_symbols)
    mixed: list[int] = []
    for symbol in trial.surface:
        if symbol in trial.code_to_word:
            for plain_value in code_mapping[int(symbol)]:
                mixed.append(variable_count + int(plain_value))
        else:
            mixed.append(char_index[int(symbol)])
    return np.asarray(mixed, dtype=np.int32), char_symbols


def solve_a1(
    trial: NomenclatorTrial,
    language: core.LanguageData,
    word_model: WordModel,
    restarts: int,
    sweeps: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    units = parse_word_units(
        trial, trial.char_to_plain, language.char_to_id[" "]
    )
    predicted, score = optimize_code_assignment(
        units,
        trial.code_symbols,
        word_model,
        core.stable_seed("v054-a1", trial.seed),
        restarts,
        sweeps,
    )
    expanded = expand_surface(trial, trial.char_to_plain, predicted)
    mapping_accuracy, occurrence_accuracy = code_metrics(trial, predicted)
    return {
        "replicate": trial.replicate,
        "observed_code_symbols": len(trial.code_symbols),
        "code_mapping_accuracy": mapping_accuracy,
        "code_occurrence_accuracy": occurrence_accuracy,
        "expanded_accuracy": mono.fast_accuracy(trial.plain, expanded),
        "word_model_score": score,
        "elapsed_seconds": time.perf_counter() - started,
    }


def solve_a2(
    trial: NomenclatorTrial,
    language: core.LanguageData,
    quadgram: tuple[np.ndarray, np.ndarray],
    iterations: int,
    restarts: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    mixed, char_symbols = build_mixed_cipher(
        trial, trial.code_to_word, len(language.alphabet)
    )
    _symbols, initial_variable = frequency_initial_key(trial, language)
    variable_count = len(char_symbols)
    full_key = np.empty(variable_count + len(language.alphabet), dtype=np.int32)
    full_key[:variable_count] = initial_variable
    full_key[variable_count:] = np.arange(len(language.alphabet), dtype=np.int32)
    solved, score = anneal_locked_key(
        mixed,
        full_key,
        variable_count,
        quadgram[0],
        quadgram[1],
        iterations,
        restarts,
        int(core.stable_seed("v054-a2", trial.seed) & 0x7FFFFFFFFFFFFFFF),
    )
    predicted = solved[mixed].tolist()
    key_accuracy = statistics.fmean(
        int(solved[index]) == trial.char_to_plain[symbol]
        for index, symbol in enumerate(char_symbols)
    ) if char_symbols else 1.0
    baseline_predicted = full_key[mixed].tolist()
    return {
        "replicate": trial.replicate,
        "observed_char_symbols": variable_count,
        "char_key_accuracy": float(key_accuracy),
        "baseline_expanded_accuracy": mono.fast_accuracy(
            trial.plain, baseline_predicted
        ),
        "expanded_accuracy": mono.fast_accuracy(trial.plain, predicted),
        "quadgram_score": float(score),
        "elapsed_seconds": time.perf_counter() - started,
    }


def summarize(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    if arm == "a1":
        return {
            "trials": len(rows),
            "mean_expanded_accuracy": statistics.fmean(
                row["expanded_accuracy"] for row in rows
            ),
            "median_expanded_accuracy": statistics.median(
                row["expanded_accuracy"] for row in rows
            ),
            "mean_code_mapping_accuracy": statistics.fmean(
                row["code_mapping_accuracy"] for row in rows
            ),
            "mean_code_occurrence_accuracy": statistics.fmean(
                row["code_occurrence_accuracy"] for row in rows
            ),
            "mean_observed_code_symbols": statistics.fmean(
                row["observed_code_symbols"] for row in rows
            ),
            "mean_seconds": statistics.fmean(row["elapsed_seconds"] for row in rows),
        }
    return {
        "trials": len(rows),
        "mean_expanded_accuracy": statistics.fmean(
            row["expanded_accuracy"] for row in rows
        ),
        "median_expanded_accuracy": statistics.median(
            row["expanded_accuracy"] for row in rows
        ),
        "mean_baseline_expanded_accuracy": statistics.fmean(
            row["baseline_expanded_accuracy"] for row in rows
        ),
        "mean_char_key_accuracy": statistics.fmean(
            row["char_key_accuracy"] for row in rows
        ),
        "mean_observed_char_symbols": statistics.fmean(
            row["observed_char_symbols"] for row in rows
        ),
        "mean_seconds": statistics.fmean(row["elapsed_seconds"] for row in rows),
    }


def canonical_sha(payload: dict[str, Any]) -> str:
    blob = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iso", default="en")
    parser.add_argument("--split", default="dev", choices=("dev", "test"))
    parser.add_argument("--length", type=int, default=384)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--codebook-size", type=int, default=24)
    parser.add_argument("--a1-restarts", type=int, default=16)
    parser.add_argument("--a1-sweeps", type=int, default=10)
    parser.add_argument("--a2-iterations", type=int, default=300000)
    parser.add_argument("--a2-restarts", type=int, default=35)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    experiment = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        experiment / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v050_corpora",
    )
    language = languages[args.iso]
    word_model = build_word_model(language)
    quadgram = build_quadgram_model(language)
    trials = [
        make_trial(
            language,
            word_model,
            args.split,
            args.length,
            args.offset + replicate,
            args.codebook_size,
        )
        for replicate in range(args.replicates)
    ]

    # Compile locked-key search before concurrent execution.
    compile_trial = trials[0]
    compile_mixed, compile_symbols = build_mixed_cipher(
        compile_trial, compile_trial.code_to_word, len(language.alphabet)
    )
    _symbols, compile_initial = frequency_initial_key(compile_trial, language)
    compile_key = np.empty(len(compile_symbols) + len(language.alphabet), dtype=np.int32)
    compile_key[: len(compile_symbols)] = compile_initial
    compile_key[len(compile_symbols) :] = np.arange(
        len(language.alphabet), dtype=np.int32
    )
    anneal_locked_key(
        compile_mixed,
        compile_key,
        len(compile_symbols),
        quadgram[0],
        quadgram[1],
        2,
        1,
        1,
    )

    a1_rows: list[dict[str, Any]] = []
    a2_rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        a1_futures = [
            executor.submit(
                solve_a1,
                trial,
                language,
                word_model,
                args.a1_restarts,
                args.a1_sweeps,
            )
            for trial in trials
        ]
        a2_futures = [
            executor.submit(
                solve_a2,
                trial,
                language,
                quadgram,
                args.a2_iterations,
                args.a2_restarts,
            )
            for trial in trials
        ]
        for completed, future in enumerate(
            concurrent.futures.as_completed(a1_futures), start=1
        ):
            row = future.result()
            a1_rows.append(row)
            print("V054_A1_TRIAL", json.dumps(row, sort_keys=True), flush=True)
        for completed, future in enumerate(
            concurrent.futures.as_completed(a2_futures), start=1
        ):
            row = future.result()
            a2_rows.append(row)
            print("V054_A2_TRIAL", json.dumps(row, sort_keys=True), flush=True)
    a1_rows.sort(key=lambda row: row["replicate"])
    a2_rows.sort(key=lambda row: row["replicate"])
    a1_summary = summarize(a1_rows, "a1")
    a2_summary = summarize(a2_rows, "a2")
    gate = {
        "a1_expanded_90_pass": a1_summary["mean_expanded_accuracy"] >= 0.90,
        "a1_code_mapping_80_pass": a1_summary["mean_code_mapping_accuracy"] >= 0.80,
        "a2_expanded_90_pass": a2_summary["mean_expanded_accuracy"] >= 0.90,
    }
    gate["pass"] = all(gate.values())
    payload: dict[str, Any] = {
        "programme": "recoverability-frontier-v0.5.4-nomenclator-stage-a",
        "iso": args.iso,
        "split": args.split,
        "target_length": args.length,
        "offset": args.offset,
        "replicates": args.replicates,
        "codebook_size": args.codebook_size,
        "generator": {
            "candidate_pool": 96,
            "fresh_character_key": True,
            "fresh_codebook": True,
            "joint_surface_permutation": True,
            "first_occurrence_canonicalisation": True,
        },
        "a1_schedule": {
            "restarts": args.a1_restarts,
            "coordinate_sweeps": args.a1_sweeps,
        },
        "a2_schedule": {
            "iterations": args.a2_iterations,
            "restarts": args.a2_restarts,
        },
        "a1_summary": a1_summary,
        "a2_summary": a2_summary,
        "gate": gate,
        "a1_rows": a1_rows,
        "a2_rows": a2_rows,
    }
    payload["scientific_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V054_A1_SUMMARY", json.dumps(a1_summary, sort_keys=True), flush=True)
    print("V054_A2_SUMMARY", json.dumps(a2_summary, sort_keys=True), flush=True)
    print("V054_GATE", json.dumps(gate, sort_keys=True), flush=True)
    print("V054_SHA256", payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
