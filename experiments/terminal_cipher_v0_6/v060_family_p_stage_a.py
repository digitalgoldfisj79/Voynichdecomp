#!/usr/bin/env python3
"""v0.6 Family P: periodic and line-reset Alberti-style wheel ciphers.

The wheel model uses one fresh mixed base alphabet plus state-dependent ring
rotations. It is deliberately stronger than shift-only Vigenere while retaining
the historically important shared-wheel structure.
"""
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
from typing import Any, Iterable

import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
V05 = HERE.parent / "recoverability_frontier_v0_5"
if str(V05) not in sys.path:
    sys.path.insert(0, str(V05))

import recoverability_v050 as core
import mono_solver_v051 as mono

MODES = ("periodic", "line_reset")
DEV_PERIODS = (2, 3, 4, 6, 8)
TEST_PERIODS = (5, 7, 9, 10, 12)
CANDIDATE_PERIODS = tuple(range(2, 13))


@dataclasses.dataclass
class WheelTrial:
    iso: str
    split: str
    length: int
    mode: str
    replicate: int
    seed: int
    plain: list[int]
    cipher: list[int]
    base_inverse: list[int]
    period: int
    shifts: list[int]
    line_starts: list[int]


def canonical_first_occurrence(values: Iterable[int]) -> list[int]:
    mapping: dict[int, int] = {}
    out: list[int] = []
    for value in values:
        value = int(value)
        if value not in mapping:
            mapping[value] = len(mapping)
        out.append(mapping[value])
    return out


def minimal_period(values: list[int]) -> int:
    for width in range(1, len(values) + 1):
        if len(values) % width == 0 and values == values[:width] * (len(values) // width):
            return width
    return len(values)


def make_shifts(rng: random.Random, alphabet_size: int, period: int) -> list[int]:
    for _ in range(1000):
        shifts = [0] + [rng.randrange(alphabet_size) for _ in range(period - 1)]
        if len(set(shifts)) >= min(3, period) and minimal_period(shifts) == period:
            return shifts
    raise RuntimeError("failed to sample a nondegenerate wheel schedule")


def make_line_starts(rng: random.Random, length: int) -> list[int]:
    starts = [0]
    cursor = 0
    while cursor < length:
        cursor += rng.randint(40, 72)
        if cursor < length:
            starts.append(cursor)
    return starts


def phase_array(length: int, period: int, mode: str, line_starts: list[int]) -> np.ndarray:
    phase = np.empty(length, dtype=np.int32)
    if mode == "periodic":
        for i in range(length):
            phase[i] = i % period
        return phase
    if mode != "line_reset":
        raise ValueError(mode)
    starts = line_starts + [length]
    for left, right in zip(starts, starts[1:]):
        for i in range(left, right):
            phase[i] = (i - left) % period
    return phase


def make_trial(
    language: core.LanguageData,
    split: str,
    length: int,
    mode: str,
    replicate: int,
) -> WheelTrial:
    chunks = core.source_chunks(language, split, length)
    if not chunks:
        raise RuntimeError(f"no chunks for {language.iso}/{split}/{length}")
    plain = list(chunks[replicate % len(chunks)])
    seed = core.stable_seed("v060-family-p", language.iso, split, length, mode, replicate)
    rng = random.Random(seed)
    periods = TEST_PERIODS if split == "test" else DEV_PERIODS
    period = rng.choice(periods)
    shifts = make_shifts(rng, len(language.alphabet), period)
    base = list(range(len(language.alphabet)))
    rng.shuffle(base)
    base_inverse = [0] * len(base)
    for plain_symbol, ring_symbol in enumerate(base):
        base_inverse[ring_symbol] = plain_symbol
    line_starts = make_line_starts(rng, length) if mode == "line_reset" else [0]
    phase = phase_array(length, period, mode, line_starts)
    a = len(base)
    cipher = [
        (base[int(value)] + shifts[int(phase[i])]) % a
        for i, value in enumerate(plain)
    ]
    # Relabel the visible alphabet jointly. The recurrence canonicalization used
    # elsewhere remains available as a separate blind representation; this arm
    # first tests the harder raw-symbol wheel problem without exposing ring order.
    visible = list(range(a))
    rng.shuffle(visible)
    cipher = [visible[x] for x in cipher]
    visible_inverse = [0] * a
    for ring_symbol, surface_symbol in enumerate(visible):
        visible_inverse[surface_symbol] = ring_symbol
    composed_inverse = [0] * a
    for surface_symbol in range(a):
        composed_inverse[surface_symbol] = base_inverse[visible_inverse[surface_symbol]]
    # Shifts are no longer literal modular shifts after arbitrary relabelling.
    # Preserve ring order only in the raw cipher for historically meaningful
    # wheel recovery; arbitrary visible relabelling is retained as a future
    # recurrence-only control, not mixed into this structural oracle stage.
    cipher = [
        (base[int(value)] + shifts[int(phase[i])]) % a
        for i, value in enumerate(plain)
    ]
    return WheelTrial(
        iso=language.iso,
        split=split,
        length=length,
        mode=mode,
        replicate=replicate,
        seed=seed,
        plain=plain,
        cipher=cipher,
        base_inverse=base_inverse,
        period=period,
        shifts=shifts,
        line_starts=line_starts,
    )


@njit(cache=True, nogil=True)
def _rng_step(state: np.uint64) -> np.uint64:
    state ^= state >> np.uint64(12)
    state ^= state << np.uint64(25)
    state ^= state >> np.uint64(27)
    return state * np.uint64(2685821657736338717)


@njit(cache=True, nogil=True)
def _rng_int(state: np.uint64, upper: int) -> tuple[np.uint64, int]:
    state = _rng_step(state)
    return state, int(state % np.uint64(upper))


@njit(cache=True, nogil=True)
def _rng_float(state: np.uint64) -> tuple[np.uint64, float]:
    state = _rng_step(state)
    return state, float(state >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True, nogil=True)
def score_wheel(
    cipher: np.ndarray,
    phase: np.ndarray,
    base_inverse: np.ndarray,
    shifts: np.ndarray,
    trigram_logp: np.ndarray,
    unigram_logp: np.ndarray,
) -> float:
    n = cipher.shape[0]
    a = base_inverse.shape[0]
    if n == 0:
        return -1e300
    first = base_inverse[(cipher[0] - shifts[phase[0]]) % a]
    score = 0.15 * unigram_logp[first]
    if n == 1:
        return score
    second = base_inverse[(cipher[1] - shifts[phase[1]]) % a]
    score += 0.15 * unigram_logp[second]
    for i in range(2, n):
        third = base_inverse[(cipher[i] - shifts[phase[i]]) % a]
        score += trigram_logp[first, second, third]
        score += 0.15 * unigram_logp[third]
        first = second
        second = third
    return score


@njit(cache=True, nogil=True)
def anneal_shifts(
    cipher: np.ndarray,
    phase: np.ndarray,
    base_inverse: np.ndarray,
    period: int,
    trigram_logp: np.ndarray,
    unigram_logp: np.ndarray,
    iterations: int,
    restarts: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    a = base_inverse.shape[0]
    state = np.uint64(seed if seed > 0 else 1)
    best_shifts = np.zeros(period, dtype=np.int32)
    best_score = score_wheel(cipher, phase, base_inverse, best_shifts, trigram_logp, unigram_logp)
    for restart in range(restarts):
        shifts = np.zeros(period, dtype=np.int32)
        for j in range(1, period):
            state, shifts[j] = _rng_int(state, a)
        current_score = score_wheel(cipher, phase, base_inverse, shifts, trigram_logp, unigram_logp)
        if current_score > best_score:
            best_score = current_score
            best_shifts = shifts.copy()
        temperature = 10.0
        cooling = math.exp(math.log(0.08 / 10.0) / max(1, iterations))
        for _ in range(iterations):
            if period <= 1:
                break
            state, slot_raw = _rng_int(state, period - 1)
            slot = slot_raw + 1
            old = shifts[slot]
            state, proposal = _rng_int(state, a)
            if proposal == old:
                continue
            shifts[slot] = proposal
            candidate = score_wheel(cipher, phase, base_inverse, shifts, trigram_logp, unigram_logp)
            delta = candidate - current_score
            accept = delta >= 0.0
            if not accept:
                state, uniform = _rng_float(state)
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
    best_score = score_wheel(cipher, phase, best_inverse, best_shifts, trigram_logp, unigram_logp)
    for restart in range(restarts):
        shifts = np.zeros(period, dtype=np.int32)
        for j in range(1, period):
            state, shifts[j] = _rng_int(state, a)
        detrended = np.empty(cipher.shape[0], dtype=np.int32)
        for i in range(cipher.shape[0]):
            detrended[i] = (cipher[i] - shifts[phase[i]]) % a
        inverse = initial_inverse.copy()
        for _ in range(2 + 2 * restart):
            state, first = _rng_int(state, a)
            state, second = _rng_int(state, a)
            if first != second:
                temporary = inverse[first]
                inverse[first] = inverse[second]
                inverse[second] = temporary
        current_score = score_wheel(cipher, phase, inverse, shifts, trigram_logp, unigram_logp)
        if current_score > best_score:
            best_score = current_score
            best_inverse = inverse.copy()
            best_shifts = shifts.copy()
        temperature = 12.0
        cooling = math.exp(math.log(0.08 / 12.0) / max(1, iterations))
        for _ in range(iterations):
            state, move = _rng_int(state, 10)
            changed_shift = move < 3 and period > 1
            if changed_shift:
                state, slot_raw = _rng_int(state, period - 1)
                slot = slot_raw + 1
                old_shift = shifts[slot]
                state, proposal = _rng_int(state, a)
                if proposal == old_shift:
                    continue
                shifts[slot] = proposal
                first = second = -1
            else:
                state, first = _rng_int(state, a)
                state, second = _rng_int(state, a)
                if first == second:
                    continue
                temporary = inverse[first]
                inverse[first] = inverse[second]
                inverse[second] = temporary
                old_shift = -1
                slot = -1
            candidate = score_wheel(cipher, phase, inverse, shifts, trigram_logp, unigram_logp)
            delta = candidate - current_score
            accept = delta >= 0.0
            if not accept:
                state, uniform = _rng_float(state)
                accept = uniform < math.exp(delta / max(temperature, 1e-9))
            if accept:
                current_score = candidate
                if candidate > best_score:
                    best_score = candidate
                    best_inverse = inverse.copy()
                    best_shifts = shifts.copy()
            elif changed_shift:
                shifts[slot] = old_shift
            else:
                temporary = inverse[first]
                inverse[first] = inverse[second]
                inverse[second] = temporary
            temperature *= cooling
    return best_inverse, best_shifts, best_score


def decode(cipher: list[int], phase: np.ndarray, inverse: np.ndarray, shifts: np.ndarray) -> list[int]:
    a = len(inverse)
    return [int(inverse[(int(symbol) - int(shifts[int(phase[i])])) % a]) for i, symbol in enumerate(cipher)]


def mdl_score(raw_score: float, period: int, length: int, alphabet_size: int) -> float:
    # BIC-like penalty for period-specific state offsets. The base alphabet term
    # is common to all candidate periods and therefore cancels.
    return raw_score - 0.5 * max(0, period - 1) * math.log(max(2, length)) * math.log(max(2, alphabet_size))


def solve_trial(
    trial: WheelTrial,
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    shift_iterations: int,
    shift_restarts: int,
    joint_iterations: int,
    joint_restarts: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    cipher = np.asarray(trial.cipher, dtype=np.int32)
    true_inverse = np.asarray(trial.base_inverse, dtype=np.int32)
    trigram, unigram = model

    oracle_schedule_candidates = []
    oracle_base_candidates = []
    joint_candidates = []
    for period in CANDIDATE_PERIODS:
        phase = phase_array(trial.length, period, trial.mode, trial.line_starts)

        detrended = np.asarray(
            [(trial.cipher[i] - trial.shifts[int(phase[i]) % trial.period]) % len(language.alphabet)
             for i in range(trial.length)],
            dtype=np.int32,
        ) if period == trial.period else None
        if period == trial.period:
            initial = mono.frequency_key(detrended.tolist(), language)
            solved_inverse, raw = mono.anneal_mono(
                detrended,
                initial,
                trigram,
                unigram,
                joint_iterations,
                joint_restarts,
                int(core.stable_seed("v060-p-oracle-schedule", trial.seed) & 0x7FFFFFFFFFFFFFFF),
            )
            prediction = solved_inverse[detrended].tolist()
            oracle_schedule_candidates.append((mono.fast_accuracy(trial.plain, prediction), float(raw)))

        solved_shifts, shift_raw = anneal_shifts(
            cipher,
            phase,
            true_inverse,
            period,
            trigram,
            unigram,
            shift_iterations,
            shift_restarts,
            int(core.stable_seed("v060-p-oracle-base", trial.seed, period) & 0x7FFFFFFFFFFFFFFF),
        )
        shift_prediction = decode(trial.cipher, phase, true_inverse, solved_shifts)
        oracle_base_candidates.append({
            "period": period,
            "score": mdl_score(float(shift_raw), period, trial.length, len(language.alphabet)),
            "accuracy": mono.fast_accuracy(trial.plain, shift_prediction),
        })

        initial_shifts = np.zeros(period, dtype=np.int32)
        detrended0 = [(trial.cipher[i] - int(initial_shifts[int(phase[i])])) % len(language.alphabet) for i in range(trial.length)]
        initial_inverse = mono.frequency_key(detrended0, language)
        solved_inverse, solved_joint_shifts, joint_raw = anneal_joint(
            cipher,
            phase,
            initial_inverse,
            period,
            trigram,
            unigram,
            joint_iterations,
            joint_restarts,
            int(core.stable_seed("v060-p-joint", trial.seed, period) & 0x7FFFFFFFFFFFFFFF),
        )
        joint_prediction = decode(trial.cipher, phase, solved_inverse, solved_joint_shifts)
        joint_candidates.append({
            "period": period,
            "score": mdl_score(float(joint_raw), period, trial.length, len(language.alphabet)),
            "raw_score": float(joint_raw),
            "accuracy": mono.fast_accuracy(trial.plain, joint_prediction),
            "prediction": joint_prediction,
        })

    selected_oracle_base = max(oracle_base_candidates, key=lambda row: row["score"])
    selected_joint = max(joint_candidates, key=lambda row: row["score"])
    oracle_schedule_accuracy = oracle_schedule_candidates[0][0] if oracle_schedule_candidates else 0.0
    return {
        "iso": trial.iso,
        "split": trial.split,
        "length": trial.length,
        "mode": trial.mode,
        "replicate": trial.replicate,
        "true_period": trial.period,
        "oracle_schedule_accuracy": oracle_schedule_accuracy,
        "oracle_base_accuracy": selected_oracle_base["accuracy"],
        "oracle_base_selected_period": selected_oracle_base["period"],
        "oracle_base_period_correct": selected_oracle_base["period"] == trial.period,
        "joint_accuracy": selected_joint["accuracy"],
        "joint_selected_period": selected_joint["period"],
        "joint_period_correct": selected_joint["period"] == trial.period,
        "joint_exact": selected_joint["prediction"] == trial.plain,
        "elapsed_seconds": time.perf_counter() - started,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(field: str) -> dict[str, float]:
        values = [float(row[field]) for row in rows]
        return {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "minimum": min(values),
            "at_least_80_rate": statistics.fmean(value >= 0.80 for value in values),
            "at_least_90_rate": statistics.fmean(value >= 0.90 for value in values),
        }
    return {
        "trials": len(rows),
        "oracle_schedule": stats("oracle_schedule_accuracy"),
        "oracle_base": stats("oracle_base_accuracy"),
        "joint": stats("joint_accuracy"),
        "oracle_base_period_accuracy": statistics.fmean(row["oracle_base_period_correct"] for row in rows),
        "joint_period_accuracy": statistics.fmean(row["joint_period_correct"] for row in rows),
        "joint_exact_rate": statistics.fmean(row["joint_exact"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iso", default="en")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--length", type=int, default=384)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--shift-iterations", type=int, default=100000)
    parser.add_argument("--shift-restarts", type=int, default=12)
    parser.add_argument("--joint-iterations", type=int, default=250000)
    parser.add_argument("--joint-restarts", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    manifest = root / "corpus_manifest_v050.json"
    languages = core.load_languages(manifest, args.repo / ".cache" / "v060-family-p")
    if args.iso not in languages:
        raise KeyError(args.iso)
    language = languages[args.iso]
    model = mono.build_language_model(language)
    trials = [
        make_trial(language, args.split, args.length, mode, replicate)
        for mode in args.modes
        for replicate in range(args.replicates)
    ]

    def run_one(trial: WheelTrial) -> dict[str, Any]:
        row = solve_trial(
            trial,
            language,
            model,
            args.shift_iterations,
            args.shift_restarts,
            args.joint_iterations,
            args.joint_restarts,
        )
        print("V060_P_TRIAL", json.dumps(row, sort_keys=True), flush=True)
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(run_one, trials))
    summary = summarize(rows)
    payload = {
        "config": vars(args) | {"repo": str(args.repo), "output": str(args.output)},
        "rows": rows,
        "summary": summary,
    }
    scientific = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(scientific).hexdigest()
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V060_P_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V060_P_SHA256", payload["sha256"], flush=True)


if __name__ == "__main__":
    main()
