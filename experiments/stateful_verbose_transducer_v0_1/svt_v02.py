#!/usr/bin/env python3
"""SVT v0.2 deterministic factorised head solver.

This module reuses the frozen v0.1 FSVT generator but replaces the failed
stochastic key optimiser. It contains no Voynich loader.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import njit

import svt_v01 as v0

core = v0.core
mono = v0.mono
MODES = v0.MODES
CANDIDATE_PERIODS = v0.CANDIDATE_PERIODS
make_svt_trial = v0.make_svt_trial

LOCAL_BIC_WEIGHT = 0.50
SCHEDULE_BIC_WEIGHT = 0.50
COORDINATE_CYCLES = 3
GLOBAL_MOVES_PER_CYCLE = 4


@njit(cache=True, nogil=True)
def _raw_score(
    cipher: np.ndarray,
    phase: np.ndarray,
    inverses: np.ndarray,
    trigram: np.ndarray,
    unigram: np.ndarray,
) -> float:
    return v0.score_stateful(cipher, phase, inverses, trigram, unigram)


@njit(cache=True, nogil=True)
def _best_global_swap(
    cipher: np.ndarray,
    phase: np.ndarray,
    inverses: np.ndarray,
    trigram: np.ndarray,
    unigram: np.ndarray,
    current: float,
) -> tuple[int, int, float]:
    a = inverses.shape[1]
    states = inverses.shape[0]
    best_i = -1
    best_j = -1
    best = current
    for i in range(a - 1):
        for j in range(i + 1, a):
            for s in range(states):
                tmp = inverses[s, i]
                inverses[s, i] = inverses[s, j]
                inverses[s, j] = tmp
            candidate = _raw_score(cipher, phase, inverses, trigram, unigram)
            for s in range(states):
                tmp = inverses[s, i]
                inverses[s, i] = inverses[s, j]
                inverses[s, j] = tmp
            if candidate > best:
                best = candidate
                best_i = i
                best_j = j
    return best_i, best_j, best


@njit(cache=True, nogil=True)
def _best_local_swap(
    cipher: np.ndarray,
    phase: np.ndarray,
    inverses: np.ndarray,
    state: int,
    trigram: np.ndarray,
    unigram: np.ndarray,
    current: float,
) -> tuple[int, int, float]:
    a = inverses.shape[1]
    best_i = -1
    best_j = -1
    best = current
    for i in range(a - 1):
        for j in range(i + 1, a):
            tmp = inverses[state, i]
            inverses[state, i] = inverses[state, j]
            inverses[state, j] = tmp
            candidate = _raw_score(cipher, phase, inverses, trigram, unigram)
            tmp = inverses[state, i]
            inverses[state, i] = inverses[state, j]
            inverses[state, j] = tmp
            if candidate > best:
                best = candidate
                best_i = i
                best_j = j
    return best_i, best_j, best


@njit(cache=True, nogil=True)
def coordinate_refine(
    cipher: np.ndarray,
    phase: np.ndarray,
    initial_inverse: np.ndarray,
    trigram: np.ndarray,
    unigram: np.ndarray,
    local_cap: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    period = int(np.max(phase)) + 1
    a = initial_inverse.shape[0]
    inverses = np.empty((period, a), dtype=np.int32)
    for s in range(period):
        inverses[s] = initial_inverse
    local_moves = np.zeros(period, dtype=np.int32)
    current = _raw_score(cipher, phase, inverses, trigram, unigram)
    local_charge = LOCAL_BIC_WEIGHT * math.log(max(2, cipher.shape[0]))

    for _cycle in range(COORDINATE_CYCLES):
        # Shared-key refinement has no extra model dimension: every state moves
        # together and remains represented by one common base permutation.
        for _ in range(GLOBAL_MOVES_PER_CYCLE):
            i, j, candidate = _best_global_swap(
                cipher, phase, inverses, trigram, unigram, current
            )
            if i < 0 or candidate <= current + 1e-9:
                break
            for s in range(period):
                tmp = inverses[s, i]
                inverses[s, i] = inverses[s, j]
                inverses[s, j] = tmp
            current = candidate

        # Sparse state-local departures from the common key. A move is admitted
        # only if it pays its preregistered BIC charge on the full sequence.
        for state in range(period):
            while local_moves[state] < local_cap:
                i, j, candidate = _best_local_swap(
                    cipher, phase, inverses, state, trigram, unigram, current
                )
                if i < 0 or candidate - current <= local_charge:
                    break
                tmp = inverses[state, i]
                inverses[state, i] = inverses[state, j]
                inverses[state, j] = tmp
                current = candidate
                local_moves[state] += 1

    return inverses, current, local_moves


def initial_shared_inverse(
    heads: list[int],
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    seed: int,
) -> np.ndarray:
    cipher = np.asarray(heads, dtype=np.int32)
    trigram, unigram = model
    freq = mono.frequency_key(heads, language)
    inverse, _ = mono.anneal_mono(
        cipher,
        np.asarray(freq, dtype=np.int32),
        trigram,
        unigram,
        120000,
        16,
        int(core.stable_seed("svt-v02-shared-base", seed) & 0x7FFFFFFFFFFFFFFF),
    )
    return np.asarray(inverse, dtype=np.int32)


def local_cap_for_alphabet(a: int) -> int:
    return max(2, round(v0.STATE_SWAP_FRACTION * a)) + 1


def candidate_score(raw: float, local_moves: np.ndarray, period: int, n: int) -> float:
    return float(
        raw
        - LOCAL_BIC_WEIGHT * int(np.sum(local_moves)) * math.log(max(2, n))
        - SCHEDULE_BIC_WEIGHT * max(0, period - 1) * math.log(max(2, n))
    )


def solve_true_structure(
    trial: v0.SVTTrial,
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    heads = trial.head.cipher
    cipher = np.asarray(heads, dtype=np.int32)
    trigram, unigram = model
    initial = initial_shared_inverse(heads, language, model, trial.head.seed)
    phase = v0._phase(
        len(heads), trial.head.period, trial.head.mode, trial.head.line_starts or [0]
    )
    inv, raw, moves = coordinate_refine(
        cipher,
        phase,
        initial,
        trigram,
        unigram,
        local_cap_for_alphabet(len(language.alphabet)),
    )
    prediction = v0.decode_stateful(heads, phase, inv)
    return {
        "true_mode": trial.head.mode,
        "true_period": trial.head.period,
        "replicate": trial.head.replicate,
        "recovery": float(mono.fast_accuracy(trial.head.plain, prediction)),
        "raw_score": float(raw),
        "local_moves": [int(x) for x in moves],
        "score": candidate_score(raw, moves, trial.head.period, len(heads)),
    }


def solve_blind_structure(
    trial: v0.SVTTrial,
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    heads = trial.head.cipher
    cipher = np.asarray(heads, dtype=np.int32)
    trigram, unigram = model
    initial = initial_shared_inverse(heads, language, model, trial.head.seed)
    local_cap = local_cap_for_alphabet(len(language.alphabet))
    rows = []
    for mode in MODES:
        for period in CANDIDATE_PERIODS:
            phase = v0._phase(len(heads), period, mode, trial.head.line_starts or [0])
            inv, raw, moves = coordinate_refine(
                cipher, phase, initial, trigram, unigram, local_cap
            )
            prediction = v0.decode_stateful(heads, phase, inv)
            rows.append({
                "mode": mode,
                "period": period,
                "score": candidate_score(raw, moves, period, len(heads)),
                "raw_score": float(raw),
                "recovery": float(mono.fast_accuracy(trial.head.plain, prediction)),
                "local_moves": [int(x) for x in moves],
            })
    selected = max(rows, key=lambda row: row["score"])
    return {
        "true_mode": trial.head.mode,
        "true_period": trial.head.period,
        "replicate": trial.head.replicate,
        "selected_mode": selected["mode"],
        "selected_period": selected["period"],
        "mode_correct": selected["mode"] == trial.head.mode,
        "period_correct": selected["period"] == trial.head.period,
        "recovery": selected["recovery"],
        "score": selected["score"],
        "local_moves": selected["local_moves"],
    }
