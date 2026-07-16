#!/usr/bin/env python3
"""Final permitted Family P development amendment: coordinate wheel/shift search."""
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
import v060_family_p_mode_blind as blind


@njit(cache=True, nogil=True)
def anneal_shifts_seeded(
    cipher: np.ndarray,
    phase: np.ndarray,
    inverse: np.ndarray,
    initial_shifts: np.ndarray,
    trigram: np.ndarray,
    unigram: np.ndarray,
    iterations: int,
    restarts: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    period = initial_shifts.shape[0]
    a = inverse.shape[0]
    state = np.uint64(seed if seed > 0 else 1)
    best_shifts = initial_shifts.copy()
    best_shifts[0] = 0
    best_score = base.score_wheel(
        cipher, phase, inverse, best_shifts, trigram, unigram
    )
    for restart in range(restarts):
        shifts = initial_shifts.copy()
        shifts[0] = 0
        perturbations = 1 + restart
        for _ in range(perturbations):
            if period <= 1:
                break
            state, slot_raw = base._rng_int(state, period - 1)
            slot = np.int64(slot_raw + 1)
            state, value = base._rng_int(state, a)
            shifts[slot] = value
        current_score = base.score_wheel(
            cipher, phase, inverse, shifts, trigram, unigram
        )
        if current_score > best_score:
            best_score = current_score
            best_shifts = shifts.copy()
        temperature = 10.0
        cooling = math.exp(math.log(0.08 / 10.0) / max(1, iterations))
        for _ in range(iterations):
            if period <= 1:
                break
            state, slot_raw = base._rng_int(state, period - 1)
            slot = np.int64(slot_raw + 1)
            old = shifts[slot]
            state, proposal_raw = base._rng_int(state, a)
            proposal = np.int64(proposal_raw)
            if proposal == old:
                continue
            shifts[slot] = proposal
            candidate = base.score_wheel(
                cipher, phase, inverse, shifts, trigram, unigram
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
                    best_shifts = shifts.copy()
            else:
                shifts[slot] = old
            temperature *= cooling
    return best_shifts, best_score


def phase_histogram_seeds(
    cipher: list[int], phase: np.ndarray, period: int, alphabet_size: int,
    count: int, seed: int,
) -> list[np.ndarray]:
    histograms = np.zeros((period, alphabet_size), dtype=np.float64)
    for index, symbol in enumerate(cipher):
        histograms[int(phase[index]), int(symbol)] += 1.0
    for row in histograms:
        total = row.sum()
        if total > 0:
            row /= total
    reference = histograms[0]
    ranked: list[list[int]] = [[0]]
    for slot in range(1, period):
        scores = []
        for shift in range(alphabet_size):
            aligned = np.roll(histograms[slot], -shift)
            scores.append((float(np.dot(reference, aligned)), shift))
        scores.sort(reverse=True)
        ranked.append([shift for _score, shift in scores[: min(5, alphabet_size)]])
    seeds: list[np.ndarray] = []
    best = np.zeros(period, dtype=np.int32)
    for slot in range(1, period):
        best[slot] = ranked[slot][0]
    seeds.append(best.copy())
    for slot in range(1, period):
        for alternative in ranked[slot][1:3]:
            candidate = best.copy()
            candidate[slot] = alternative
            seeds.append(candidate)
            if len(seeds) >= count:
                return seeds
    rng = random.Random(seed)
    while len(seeds) < count:
        candidate = np.zeros(period, dtype=np.int32)
        for slot in range(1, period):
            candidate[slot] = rng.choice(ranked[slot])
        if not any(np.array_equal(candidate, existing) for existing in seeds):
            seeds.append(candidate)
    return seeds


def detrend(cipher: list[int], phase: np.ndarray, shifts: np.ndarray, a: int) -> np.ndarray:
    return np.asarray(
        [(int(symbol) - int(shifts[int(phase[i])])) % a
         for i, symbol in enumerate(cipher)],
        dtype=np.int32,
    )


def coordinate_run(
    trial: base.WheelTrial,
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    mode: str,
    period: int,
    initial_shifts: np.ndarray,
    seed_index: int,
    cycles: int,
    mono_iterations: int,
    mono_restarts: int,
    shift_iterations: int,
    shift_restarts: int,
    label: str,
    initial_inverse: np.ndarray | None = None,
) -> dict[str, Any]:
    trigram, unigram = model
    phase = base.phase_array(trial.length, period, mode, trial.line_starts)
    a = len(language.alphabet)
    cipher_array = np.asarray(trial.cipher, dtype=np.int32)
    shifts = initial_shifts.copy()
    inverse = initial_inverse.copy() if initial_inverse is not None else None
    trajectory = []
    for cycle in range(cycles):
        detrended = detrend(trial.cipher, phase, shifts, a)
        if inverse is None:
            inverse = mono.frequency_key(detrended.tolist(), language)
        inverse, mono_score = mono.anneal_mono(
            detrended,
            inverse,
            trigram,
            unigram,
            mono_iterations,
            mono_restarts,
            int(core.stable_seed(
                "v060-p2-mono", trial.seed, mode, period, label, seed_index, cycle
            ) & 0x7FFFFFFFFFFFFFFF),
        )
        shifts, shift_score = anneal_shifts_seeded(
            cipher_array,
            phase,
            inverse,
            shifts,
            trigram,
            unigram,
            shift_iterations,
            shift_restarts,
            int(core.stable_seed(
                "v060-p2-shift", trial.seed, mode, period, label, seed_index, cycle
            ) & 0x7FFFFFFFFFFFFFFF),
        )
        trajectory.append({
            "cycle": cycle + 1,
            "mono_score": float(mono_score),
            "shift_score": float(shift_score),
        })
    raw_score = base.score_wheel(
        cipher_array, phase, inverse, shifts, trigram, unigram
    )
    prediction = base.decode(trial.cipher, phase, inverse, shifts)
    return {
        "mode": mode,
        "period": period,
        "inverse": inverse,
        "shifts": shifts,
        "raw_score": float(raw_score),
        "score": base.mdl_score(float(raw_score), period, trial.length, a),
        "accuracy": mono.fast_accuracy(trial.plain, prediction),
        "prediction": prediction,
        "trajectory": trajectory,
        "seed_index": seed_index,
    }


def solve_trial(
    trial: base.WheelTrial,
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    seed_count: int,
    screen_cycles: int,
    screen_mono_iterations: int,
    screen_mono_restarts: int,
    screen_shift_iterations: int,
    screen_shift_restarts: int,
    top_refine: int,
    refine_cycles: int,
    refine_mono_iterations: int,
    refine_mono_restarts: int,
    refine_shift_iterations: int,
    refine_shift_restarts: int,
    final_mono_iterations: int,
    final_mono_restarts: int,
    final_shift_iterations: int,
    final_shift_restarts: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    screen: list[dict[str, Any]] = []
    a = len(language.alphabet)
    for mode in blind.MODES:
        for period in blind.CANDIDATE_PERIODS:
            phase = base.phase_array(trial.length, period, mode, trial.line_starts)
            seeds = phase_histogram_seeds(
                trial.cipher,
                phase,
                period,
                a,
                seed_count,
                core.stable_seed("v060-p2-seeds", trial.seed, mode, period),
            )
            for seed_index, shifts in enumerate(seeds):
                screen.append(coordinate_run(
                    trial, language, model, mode, period, shifts, seed_index,
                    screen_cycles,
                    screen_mono_iterations,
                    screen_mono_restarts,
                    screen_shift_iterations,
                    screen_shift_restarts,
                    "screen",
                ))
    screen.sort(key=lambda row: row["score"], reverse=True)
    refined: list[dict[str, Any]] = []
    seen_structures: set[tuple[str, int, tuple[int, ...]]] = set()
    for candidate in screen:
        fingerprint = (
            candidate["mode"], candidate["period"],
            tuple(int(x) for x in candidate["shifts"]),
        )
        if fingerprint in seen_structures:
            continue
        seen_structures.add(fingerprint)
        refined.append(coordinate_run(
            trial,
            language,
            model,
            candidate["mode"],
            candidate["period"],
            candidate["shifts"],
            candidate["seed_index"],
            refine_cycles,
            refine_mono_iterations,
            refine_mono_restarts,
            refine_shift_iterations,
            refine_shift_restarts,
            "refine",
            initial_inverse=candidate["inverse"],
        ))
        if len(refined) >= top_refine:
            break
    best = max(refined, key=lambda row: row["score"])
    phase = base.phase_array(
        trial.length, best["period"], best["mode"], trial.line_starts
    )
    trigram, unigram = model
    cipher_array = np.asarray(trial.cipher, dtype=np.int32)
    shifts = best["shifts"].copy()
    inverse = best["inverse"].copy()
    # Final wheel -> shift -> wheel closure.
    for pass_index in range(2):
        detrended = detrend(trial.cipher, phase, shifts, a)
        inverse, _mono_score = mono.anneal_mono(
            detrended,
            inverse,
            trigram,
            unigram,
            final_mono_iterations,
            final_mono_restarts,
            int(core.stable_seed(
                "v060-p2-final-mono", trial.seed, best["mode"],
                best["period"], pass_index
            ) & 0x7FFFFFFFFFFFFFFF),
        )
        if pass_index == 0:
            shifts, _shift_score = anneal_shifts_seeded(
                cipher_array,
                phase,
                inverse,
                shifts,
                trigram,
                unigram,
                final_shift_iterations,
                final_shift_restarts,
                int(core.stable_seed(
                    "v060-p2-final-shift", trial.seed, best["mode"], best["period"]
                ) & 0x7FFFFFFFFFFFFFFF),
            )
    prediction = base.decode(trial.cipher, phase, inverse, shifts)
    accuracy = mono.fast_accuracy(trial.plain, prediction)
    return {
        "iso": trial.iso,
        "split": trial.split,
        "length": trial.length,
        "replicate": trial.replicate,
        "true_mode": trial.mode,
        "true_period": trial.period,
        "selected_mode": best["mode"],
        "selected_period": best["period"],
        "mode_correct": best["mode"] == trial.mode,
        "period_correct": best["period"] == trial.period,
        "structure_correct": (
            best["mode"] == trial.mode and best["period"] == trial.period
        ),
        "accuracy": accuracy,
        "exact": prediction == trial.plain,
        "screen_best_accuracy": screen[0]["accuracy"],
        "screen_best_mode": screen[0]["mode"],
        "screen_best_period": screen[0]["period"],
        "refined_candidates": len(refined),
        "elapsed_seconds": time.perf_counter() - started,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["accuracy"]) for row in rows]
    return {
        "trials": len(rows),
        "recovery": {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "minimum": min(values),
            "at_least_80_rate": statistics.fmean(value >= 0.80 for value in values),
            "at_least_90_rate": statistics.fmean(value >= 0.90 for value in values),
        },
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
    parser.add_argument("--iso", default="en")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--length", type=int, default=384)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--screen-cycles", type=int, default=2)
    parser.add_argument("--screen-mono-iterations", type=int, default=50000)
    parser.add_argument("--screen-mono-restarts", type=int, default=5)
    parser.add_argument("--screen-shift-iterations", type=int, default=25000)
    parser.add_argument("--screen-shift-restarts", type=int, default=6)
    parser.add_argument("--top-refine", type=int, default=4)
    parser.add_argument("--refine-cycles", type=int, default=2)
    parser.add_argument("--refine-mono-iterations", type=int, default=250000)
    parser.add_argument("--refine-mono-restarts", type=int, default=24)
    parser.add_argument("--refine-shift-iterations", type=int, default=50000)
    parser.add_argument("--refine-shift-restarts", type=int, default=12)
    parser.add_argument("--final-mono-iterations", type=int, default=700000)
    parser.add_argument("--final-mono-restarts", type=int, default=50)
    parser.add_argument("--final-shift-iterations", type=int, default=100000)
    parser.add_argument("--final-shift-restarts", type=int, default=24)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        root / "corpus_manifest_v050.json", args.repo / ".cache" / "v060-family-p2"
    )
    language = languages[args.iso]
    model = mono.build_language_model(language)
    trials = [
        blind.make_trial(language, args.split, args.length, mode, replicate)
        for mode in blind.MODES
        for replicate in range(args.replicates)
    ]

    def run_one(trial: base.WheelTrial) -> dict[str, Any]:
        row = solve_trial(
            trial, language, model,
            args.seed_count,
            args.screen_cycles,
            args.screen_mono_iterations,
            args.screen_mono_restarts,
            args.screen_shift_iterations,
            args.screen_shift_restarts,
            args.top_refine,
            args.refine_cycles,
            args.refine_mono_iterations,
            args.refine_mono_restarts,
            args.refine_shift_iterations,
            args.refine_shift_restarts,
            args.final_mono_iterations,
            args.final_mono_restarts,
            args.final_shift_iterations,
            args.final_shift_restarts,
        )
        print("V060_P2_TRIAL", json.dumps(row, sort_keys=True), flush=True)
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
    print("V060_P2_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V060_P2_SHA256", payload["sha256"], flush=True)


if __name__ == "__main__":
    main()
