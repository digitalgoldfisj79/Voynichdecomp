#!/usr/bin/env python3
"""SVT v0.1.1 factorised head decoder repair.

Reuses the frozen v0.1 surface generator and segmentation lattice. The only
change is the stateful head solver: one shared base inverse alphabet plus sparse
state-local swaps, matching the FSVT generator actually frozen in v0.1.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import njit

import svt_v01 as v0

# Re-export the frozen generator, controls and shared infrastructure.
core = v0.core
mono = v0.mono
MODES = v0.MODES
CANDIDATE_PERIODS = v0.CANDIDATE_PERIODS
HeadSolution = v0.HeadSolution
SurfaceSolution = v0.SurfaceSolution
make_svt_trial = v0.make_svt_trial
make_nonfactorable_control = v0.make_nonfactorable_control
make_shuffled_control = v0.make_shuffled_control
top_segmentations = v0.top_segmentations
boundary_f1 = v0.boundary_f1

DELTA_PENALTY = 0.50
SCHEDULE_PENALTY = 0.50
GLOBAL_MOVE_RATE_TENTHS = 4


@njit(cache=True, nogil=True)
def _delta_swaps(base: np.ndarray, inverses: np.ndarray) -> float:
    mismatches = 0
    for s in range(inverses.shape[0]):
        for i in range(inverses.shape[1]):
            if inverses[s, i] != base[i]:
                mismatches += 1
    return 0.5 * mismatches


@njit(cache=True, nogil=True)
def _factor_score(
    cipher: np.ndarray,
    phase: np.ndarray,
    base: np.ndarray,
    inverses: np.ndarray,
    trigram: np.ndarray,
    unigram: np.ndarray,
    period: int,
) -> tuple[float, float, float]:
    raw = v0.score_stateful(cipher, phase, inverses, trigram, unigram)
    n = max(2, cipher.shape[0])
    swaps = _delta_swaps(base, inverses)
    penalty = (
        DELTA_PENALTY * swaps * math.log(n)
        + SCHEDULE_PENALTY * max(0, period - 1) * math.log(n)
    )
    return raw - penalty, raw, swaps


@njit(cache=True, nogil=True)
def anneal_factorized(
    cipher: np.ndarray,
    phase: np.ndarray,
    initial_base: np.ndarray,
    period: int,
    trigram: np.ndarray,
    unigram: np.ndarray,
    iterations: int,
    restarts: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    a = initial_base.shape[0]
    rs = np.uint64(seed if seed > 0 else 1)
    best_base = initial_base.copy()
    best_inv = np.empty((period, a), dtype=np.int32)
    for s in range(period):
        best_inv[s] = initial_base
    best_score, best_raw, best_swaps = _factor_score(
        cipher, phase, best_base, best_inv, trigram, unigram, period
    )

    for restart in range(restarts):
        base = initial_base.copy()
        inv = np.empty((period, a), dtype=np.int32)
        for s in range(period):
            inv[s] = base

        # Small fresh perturbations keep restarts distinct without destroying
        # the shared-base initialization.
        for _ in range(restart):
            rs, s_raw = v0._rng_int(rs, period)
            rs, i_raw = v0._rng_int(rs, a)
            rs, j_raw = v0._rng_int(rs, a)
            s = np.int64(s_raw)
            i = np.int64(i_raw)
            j = np.int64(j_raw)
            if i != j:
                tmp = inv[s, i]
                inv[s, i] = inv[s, j]
                inv[s, j] = tmp

        current, _, _ = _factor_score(cipher, phase, base, inv, trigram, unigram, period)
        if current > best_score:
            best_score, best_raw, best_swaps = _factor_score(
                cipher, phase, base, inv, trigram, unigram, period
            )
            best_base = base.copy()
            best_inv = inv.copy()

        temp = 10.0
        cooling = math.exp(math.log(0.08 / 10.0) / max(1, iterations))
        for _ in range(iterations):
            rs, move_raw = v0._rng_int(rs, 10)
            rs, i_raw = v0._rng_int(rs, a)
            rs, j_raw = v0._rng_int(rs, a)
            move = np.int64(move_raw)
            i = np.int64(i_raw)
            j = np.int64(j_raw)
            if i == j:
                continue

            local_state = np.int64(-1)
            if move < GLOBAL_MOVE_RATE_TENTHS:
                # Shared swap: changes the common key and every state together.
                tmp = base[i]
                base[i] = base[j]
                base[j] = tmp
                for s in range(period):
                    tmp2 = inv[s, i]
                    inv[s, i] = inv[s, j]
                    inv[s, j] = tmp2
            else:
                rs, s_raw = v0._rng_int(rs, period)
                local_state = np.int64(s_raw)
                tmp = inv[local_state, i]
                inv[local_state, i] = inv[local_state, j]
                inv[local_state, j] = tmp

            candidate, raw, swaps = _factor_score(
                cipher, phase, base, inv, trigram, unigram, period
            )
            delta = candidate - current
            accept = delta >= 0.0
            if not accept:
                rs, u = v0._rng_float(rs)
                accept = u < math.exp(delta / max(temp, 1e-9))

            if accept:
                current = candidate
                if candidate > best_score:
                    best_score = candidate
                    best_raw = raw
                    best_swaps = swaps
                    best_base = base.copy()
                    best_inv = inv.copy()
            elif local_state < 0:
                tmp = base[i]
                base[i] = base[j]
                base[j] = tmp
                for s in range(period):
                    tmp2 = inv[s, i]
                    inv[s, i] = inv[s, j]
                    inv[s, j] = tmp2
            else:
                tmp = inv[local_state, i]
                inv[local_state, i] = inv[local_state, j]
                inv[local_state, j] = tmp

            temp *= cooling

    return best_base, best_inv, best_score, best_raw, best_swaps


def solve_head_stream(
    heads: list[int],
    head_line_starts: list[int],
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    iterations: int,
    restarts: int,
    seed: int,
) -> HeadSolution:
    if heads and max(heads) >= len(language.alphabet):
        raise ValueError("cipher head alphabet exceeds candidate plaintext alphabet")
    cipher = np.asarray(heads, dtype=np.int32)
    trigram, unigram = model
    candidates: list[HeadSolution] = []

    # One shared monoalphabetic initializer. This is deliberately estimated
    # without knowing the state schedule; state-specific swaps are then learned
    # jointly for each candidate structure.
    freq = mono.frequency_key(heads, language)
    base, _ = mono.anneal_mono(
        cipher,
        np.asarray(freq, dtype=np.int32),
        trigram,
        unigram,
        max(5000, iterations // 2),
        max(2, restarts // 2),
        int(core.stable_seed("svt-v011-base", seed) & 0x7FFFFFFFFFFFFFFF),
    )

    for mode in MODES:
        for period in CANDIDATE_PERIODS:
            phase = v0._phase(len(heads), period, mode, head_line_starts or [0])
            _, inv, penalised, raw, _swaps = anneal_factorized(
                cipher,
                phase,
                np.asarray(base, dtype=np.int32),
                period,
                trigram,
                unigram,
                iterations,
                restarts,
                int(core.stable_seed("svt-v011", seed, mode, period) & 0x7FFFFFFFFFFFFFFF),
            )
            prediction = v0.decode_stateful(heads, phase, inv)
            candidates.append(
                HeadSolution(
                    mode,
                    period,
                    float(penalised),
                    float(raw),
                    prediction,
                    [[int(x) for x in row] for row in inv],
                )
            )
    return max(candidates, key=lambda row: row.score)


def solve_surface(
    surface: list[int],
    line_starts: list[int],
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    iterations: int = 80000,
    restarts: int = 8,
    beam: int = v0.SEGMENTATION_BEAM,
    boundary_weight: float = v0.BOUNDARY_WEIGHT,
    seed: int = 1,
) -> SurfaceSolution:
    paths = v0.top_segmentations(surface, line_starts, len(language.alphabet), beam)
    if not paths:
        raise RuntimeError("boundary lattice produced no complete segmentation")
    candidates: list[SurfaceSolution] = []
    for rank, path in enumerate(paths):
        heads = [surface[i] for i in path.starts]
        sol = solve_head_stream(
            heads,
            path.head_line_starts,
            language,
            model,
            iterations,
            restarts,
            int(core.stable_seed("svt-v011-surface", seed, rank)),
        )
        hn = sol.score / max(1, len(heads))
        bn = path.score / max(1, len(surface))
        candidates.append(
            SurfaceSolution(path, sol, float(hn + boundary_weight * bn), float(hn), float(bn))
        )
    return max(candidates, key=lambda row: row.joint_score)


def solve_true_heads(
    trial: v0.SVTTrial,
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    iterations: int,
    restarts: int,
) -> dict[str, Any]:
    sol = solve_head_stream(
        trial.head.cipher,
        trial.head.line_starts,
        language,
        model,
        iterations,
        restarts,
        int(core.stable_seed("svt-v011-oracle", trial.head.seed)),
    )
    rec = float(mono.fast_accuracy(trial.head.plain, sol.prediction))
    return {
        "family": trial.family,
        "mode": trial.head.mode,
        "period": trial.head.period,
        "replicate": trial.head.replicate,
        "selected_mode": sol.mode,
        "selected_period": sol.period,
        "recovery": rec,
        "structure_correct": sol.mode == trial.head.mode and sol.period == trial.head.period,
        "score": sol.score,
    }


def solve_svt_trial(
    trial: v0.SVTTrial,
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    iterations: int,
    restarts: int,
    beam: int,
    boundary_weight: float,
) -> dict[str, Any]:
    sol = solve_surface(
        trial.surface,
        trial.surface_line_starts,
        language,
        model,
        iterations,
        restarts,
        beam,
        boundary_weight,
        int(core.stable_seed("svt-v011-trial", trial.head.seed, trial.family)),
    )
    rec = float(mono.fast_accuracy(trial.head.plain, sol.head_solution.prediction))
    bf = float(v0.boundary_f1(sol.path.starts, trial.head_positions))
    return {
        "family": trial.family,
        "mode": trial.head.mode,
        "period": trial.head.period,
        "replicate": trial.head.replicate,
        "selected_mode": sol.head_solution.mode,
        "selected_period": sol.head_solution.period,
        "recovery": rec,
        "boundary_f1": bf,
        "structure_correct": (
            sol.head_solution.mode == trial.head.mode
            and sol.head_solution.period == trial.head.period
        ),
        "joint_score": sol.joint_score,
    }


def summarize_joint(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return v0.summarize_joint(rows)
