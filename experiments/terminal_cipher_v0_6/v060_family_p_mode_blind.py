#!/usr/bin/env python3
"""Corrected mode-blind v0.6 Family P harness.

Every ciphertext carries observed line boundaries. Candidate inference compares
periodic and line-reset schedules jointly across periods 2--12.
"""
from __future__ import annotations

import argparse
import concurrent.futures
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
import v060_family_p_stage_a as base

MODES = ("periodic", "line_reset")
CANDIDATE_PERIODS = tuple(range(2, 13))


def make_trial(
    language: core.LanguageData,
    split: str,
    length: int,
    mode: str,
    replicate: int,
) -> base.WheelTrial:
    chunks = core.source_chunks(language, split, length)
    if not chunks:
        raise RuntimeError(f"no chunks for {language.iso}/{split}/{length}")
    plain = list(chunks[replicate % len(chunks)])
    seed = core.stable_seed("v060-family-p", language.iso, split, length, mode, replicate)
    rng = random.Random(seed)
    periods = base.TEST_PERIODS if split == "test" else base.DEV_PERIODS
    period = rng.choice(periods)
    shifts = base.make_shifts(rng, len(language.alphabet), period)
    wheel = list(range(len(language.alphabet)))
    rng.shuffle(wheel)
    inverse = [0] * len(wheel)
    for plain_symbol, ring_symbol in enumerate(wheel):
        inverse[ring_symbol] = plain_symbol

    line_rng = random.Random(core.stable_seed("v060-family-p-observed-lines", seed))
    line_starts = base.make_line_starts(line_rng, length)
    phase = base.phase_array(length, period, mode, line_starts)
    a = len(wheel)
    cipher = [
        (wheel[int(value)] + shifts[int(phase[i])]) % a
        for i, value in enumerate(plain)
    ]
    return base.WheelTrial(
        iso=language.iso,
        split=split,
        length=length,
        mode=mode,
        replicate=replicate,
        seed=seed,
        plain=plain,
        cipher=cipher,
        base_inverse=inverse,
        period=period,
        shifts=shifts,
        line_starts=line_starts,
    )


@njit(cache=True, nogil=True)
def anneal_joint(
    cipher: np.ndarray,
    phase: np.ndarray,
    initial_inverse: np.ndarray,
    period: int,
    trigram_logp: np.ndarray,
    unigram_logp: np.ndarray,
    iterations: int,
    restarts: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    a = initial_inverse.shape[0]
    state = np.uint64(seed if seed > 0 else 1)
    best_inverse = initial_inverse.copy()
    best_shifts = np.zeros(period, dtype=np.int32)
    best_score = base.score_wheel(
        cipher, phase, best_inverse, best_shifts, trigram_logp, unigram_logp
    )
    for restart in range(restarts):
        shifts = np.zeros(period, dtype=np.int32)
        for j in range(1, period):
            state, value = base._rng_int(state, a)
            shifts[j] = value
        inverse = initial_inverse.copy()
        for _ in range(2 + 2 * restart):
            state, first_raw = base._rng_int(state, a)
            state, second_raw = base._rng_int(state, a)
            first_idx = np.int64(first_raw)
            second_idx = np.int64(second_raw)
            if first_idx != second_idx:
                temporary = inverse[first_idx]
                inverse[first_idx] = inverse[second_idx]
                inverse[second_idx] = temporary
        current_score = base.score_wheel(
            cipher, phase, inverse, shifts, trigram_logp, unigram_logp
        )
        if current_score > best_score:
            best_score = current_score
            best_inverse = inverse.copy()
            best_shifts = shifts.copy()
        temperature = 12.0
        cooling = math.exp(math.log(0.08 / 12.0) / max(1, iterations))
        for _ in range(iterations):
            state, move = base._rng_int(state, 10)
            if move < 3 and period > 1:
                state, slot_raw = base._rng_int(state, period - 1)
                slot_idx = np.int64(slot_raw + 1)
                old_shift = shifts[slot_idx]
                state, proposal_raw = base._rng_int(state, a)
                proposal = np.int64(proposal_raw)
                if proposal == old_shift:
                    continue
                shifts[slot_idx] = proposal
                candidate = base.score_wheel(
                    cipher, phase, inverse, shifts, trigram_logp, unigram_logp
                )
                delta = candidate - current_score
                accept = delta >= 0.0
                if not accept:
                    state, uniform = base._rng_float(state)
                    accept = uniform < math.exp(delta / max(temperature, 1e-9))
                if accept:
                    current_score = candidate
                    if candidate > best_score:
                        best_score = candidate
                        best_inverse = inverse.copy()
                        best_shifts = shifts.copy()
                else:
                    shifts[slot_idx] = old_shift
            else:
                state, first_raw = base._rng_int(state, a)
                state, second_raw = base._rng_int(state, a)
                first_idx = np.int64(first_raw)
                second_idx = np.int64(second_raw)
                if first_idx == second_idx:
                    continue
                temporary = inverse[first_idx]
                inverse[first_idx] = inverse[second_idx]
                inverse[second_idx] = temporary
                candidate = base.score_wheel(
                    cipher, phase, inverse, shifts, trigram_logp, unigram_logp
                )
                delta = candidate - current_score
                accept = delta >= 0.0
                if not accept:
                    state, uniform = base._rng_float(state)
                    accept = uniform < math.exp(delta / max(temperature, 1e-9))
                if accept:
                    current_score = candidate
                    if candidate > best_score:
                        best_score = candidate
                        best_inverse = inverse.copy()
                        best_shifts = shifts.copy()
                else:
                    temporary = inverse[first_idx]
                    inverse[first_idx] = inverse[second_idx]
                    inverse[second_idx] = temporary
            temperature *= cooling
    return best_inverse, best_shifts, best_score


def structure_candidates(trial: base.WheelTrial):
    for mode in MODES:
        for period in CANDIDATE_PERIODS:
            yield mode, period, base.phase_array(
                trial.length, period, mode, trial.line_starts
            )


def score_penalized(raw: float, period: int, length: int, alphabet_size: int) -> float:
    return base.mdl_score(raw, period, length, alphabet_size)


def solve_oracle(
    trial: base.WheelTrial,
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    mono_iterations: int,
    mono_restarts: int,
    shift_iterations: int,
    shift_restarts: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    cipher = np.asarray(trial.cipher, dtype=np.int32)
    true_inverse = np.asarray(trial.base_inverse, dtype=np.int32)
    trigram, unigram = model

    true_phase = base.phase_array(
        trial.length, trial.period, trial.mode, trial.line_starts
    )
    detrended = np.asarray(
        [
            (trial.cipher[i] - trial.shifts[int(true_phase[i])])
            % len(language.alphabet)
            for i in range(trial.length)
        ],
        dtype=np.int32,
    )
    initial = mono.frequency_key(detrended.tolist(), language)
    solved_inverse, mono_raw = mono.anneal_mono(
        detrended,
        initial,
        trigram,
        unigram,
        mono_iterations,
        mono_restarts,
        int(core.stable_seed("v060-p1-oracle-schedule", trial.seed) & 0x7FFFFFFFFFFFFFFF),
    )
    schedule_prediction = solved_inverse[detrended].tolist()

    candidates = []
    for mode, period, phase in structure_candidates(trial):
        shifts, raw = base.anneal_shifts(
            cipher,
            phase,
            true_inverse,
            period,
            trigram,
            unigram,
            shift_iterations,
            shift_restarts,
            int(
                core.stable_seed("v060-p1-oracle-base", trial.seed, mode, period)
                & 0x7FFFFFFFFFFFFFFF
            ),
        )
        prediction = base.decode(trial.cipher, phase, true_inverse, shifts)
        candidates.append(
            {
                "mode": mode,
                "period": period,
                "score": score_penalized(
                    float(raw), period, trial.length, len(language.alphabet)
                ),
                "accuracy": mono.fast_accuracy(trial.plain, prediction),
            }
        )
    selected = max(candidates, key=lambda row: row["score"])
    return {
        "stage": "oracle",
        "iso": trial.iso,
        "split": trial.split,
        "length": trial.length,
        "true_mode": trial.mode,
        "true_period": trial.period,
        "replicate": trial.replicate,
        "oracle_schedule_accuracy": mono.fast_accuracy(
            trial.plain, schedule_prediction
        ),
        "oracle_schedule_exact": schedule_prediction == trial.plain,
        "oracle_base_accuracy": selected["accuracy"],
        "selected_mode": selected["mode"],
        "selected_period": selected["period"],
        "mode_correct": selected["mode"] == trial.mode,
        "period_correct": selected["period"] == trial.period,
        "structure_correct": (
            selected["mode"] == trial.mode and selected["period"] == trial.period
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def solve_joint(
    trial: base.WheelTrial,
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    iterations: int,
    restarts: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    cipher = np.asarray(trial.cipher, dtype=np.int32)
    trigram, unigram = model
    candidates = []
    for mode, period, phase in structure_candidates(trial):
        initial_inverse = mono.frequency_key(trial.cipher, language)
        inverse, shifts, raw = anneal_joint(
            cipher,
            phase,
            initial_inverse,
            period,
            trigram,
            unigram,
            iterations,
            restarts,
            int(
                core.stable_seed("v060-p1-joint", trial.seed, mode, period)
                & 0x7FFFFFFFFFFFFFFF
            ),
        )
        prediction = base.decode(trial.cipher, phase, inverse, shifts)
        candidates.append(
            {
                "mode": mode,
                "period": period,
                "score": score_penalized(
                    float(raw), period, trial.length, len(language.alphabet)
                ),
                "raw_score": float(raw),
                "accuracy": mono.fast_accuracy(trial.plain, prediction),
                "prediction": prediction,
            }
        )
    selected = max(candidates, key=lambda row: row["score"])
    return {
        "stage": "joint",
        "iso": trial.iso,
        "split": trial.split,
        "length": trial.length,
        "true_mode": trial.mode,
        "true_period": trial.period,
        "replicate": trial.replicate,
        "accuracy": selected["accuracy"],
        "exact": selected["prediction"] == trial.plain,
        "selected_mode": selected["mode"],
        "selected_period": selected["period"],
        "mode_correct": selected["mode"] == trial.mode,
        "period_correct": selected["period"] == trial.period,
        "structure_correct": (
            selected["mode"] == trial.mode and selected["period"] == trial.period
        ),
        "score": selected["score"],
        "elapsed_seconds": time.perf_counter() - started,
    }


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "at_least_80_rate": statistics.fmean(value >= 0.80 for value in values),
        "at_least_90_rate": statistics.fmean(value >= 0.90 for value in values),
    }


def summarize(stage: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if stage == "oracle":
        schedule = [float(row["oracle_schedule_accuracy"]) for row in rows]
        base_values = [float(row["oracle_base_accuracy"]) for row in rows]
        return {
            "trials": len(rows),
            "oracle_schedule": stats(schedule),
            "oracle_base": stats(base_values),
            "mode_accuracy": statistics.fmean(row["mode_correct"] for row in rows),
            "period_accuracy": statistics.fmean(row["period_correct"] for row in rows),
            "structure_accuracy": statistics.fmean(
                row["structure_correct"] for row in rows
            ),
            "gate": {
                "pass": (
                    statistics.fmean(schedule) >= 0.95
                    and min(schedule) >= 0.90
                    and statistics.fmean(base_values) >= 0.95
                    and min(base_values) >= 0.90
                    and sum(row["structure_correct"] for row in rows) >= 14
                )
            },
        }
    values = [float(row["accuracy"]) for row in rows]
    return {
        "trials": len(rows),
        "recovery": stats(values),
        "exact_rate": statistics.fmean(row["exact"] for row in rows),
        "mode_accuracy": statistics.fmean(row["mode_correct"] for row in rows),
        "period_accuracy": statistics.fmean(row["period_correct"] for row in rows),
        "structure_accuracy": statistics.fmean(
            row["structure_correct"] for row in rows
        ),
        "gate": {
            "pass": (
                statistics.fmean(values) >= 0.80
                and statistics.median(values) >= 0.90
                and sum(value >= 0.80 for value in values) >= 14
                and sum(row["mode_correct"] for row in rows) >= 14
                and sum(row["structure_correct"] for row in rows) >= 12
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("oracle", "joint"), required=True)
    parser.add_argument("--iso", default="en")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--length", type=int, default=384)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--mono-iterations", type=int, default=700000)
    parser.add_argument("--mono-restarts", type=int, default=50)
    parser.add_argument("--shift-iterations", type=int, default=50000)
    parser.add_argument("--shift-restarts", type=int, default=12)
    parser.add_argument("--joint-iterations", type=int, default=250000)
    parser.add_argument("--joint-restarts", type=int, default=24)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        root / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v060-family-p-mode-blind",
    )
    language = languages[args.iso]
    model = mono.build_language_model(language)
    trials = [
        make_trial(language, args.split, args.length, mode, replicate)
        for mode in MODES
        for replicate in range(args.replicates)
    ]

    def run_one(trial: base.WheelTrial) -> dict[str, Any]:
        if args.stage == "oracle":
            row = solve_oracle(
                trial,
                language,
                model,
                args.mono_iterations,
                args.mono_restarts,
                args.shift_iterations,
                args.shift_restarts,
            )
        else:
            row = solve_joint(
                trial,
                language,
                model,
                args.joint_iterations,
                args.joint_restarts,
            )
        print("V060_P1_TRIAL", json.dumps(row, sort_keys=True), flush=True)
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(run_one, trials))
    summary = summarize(args.stage, rows)
    payload = {
        "config": vars(args) | {"repo": str(args.repo), "output": str(args.output)},
        "rows": rows,
        "summary": summary,
    }
    scientific = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(scientific).hexdigest()
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V060_P1_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V060_P1_SHA256", payload["sha256"], flush=True)


if __name__ == "__main__":
    main()
