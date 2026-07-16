#!/usr/bin/env python3
"""Final permitted Family T development solver.

The solver preserves both halves of the joint state (substitution key and
column permutation), evaluates every declared mode/width structure, and uses
independent deterministic starts before full refinement.
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
import v060_family_t_stage_a as t


@njit(cache=True, nogil=True)
def anneal_permutation_seeded(
    cipher: np.ndarray,
    width: int,
    mode_flag: int,
    line_starts: np.ndarray,
    trigram: np.ndarray,
    unigram: np.ndarray,
    initial: np.ndarray,
    iterations: int,
    restarts: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Anneal a column order while retaining the supplied state as restart zero."""
    state = np.uint64(seed if seed > 0 else 1)
    best = initial.copy()
    best_score = t.score_permutation(
        cipher, width, best, mode_flag, line_starts, trigram, unigram
    )
    for restart in range(restarts):
        permutation = initial.copy()
        if restart > 0:
            for _ in range(width + 2 * restart):
                state, first = mono._rng_int(state, width)
                state, second = mono._rng_int(state, width)
                if first != second:
                    temporary = permutation[first]
                    permutation[first] = permutation[second]
                    permutation[second] = temporary
        current = t.score_permutation(
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
            candidate = t.score_permutation(
                cipher, width, permutation, mode_flag, line_starts,
                trigram, unigram,
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


def initial_key_and_permutation(
    trial: t.ColumnarTrial,
    language: core.LanguageData,
    width: int,
    mode: str,
    start_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    key = mono.frequency_key(trial.cipher, language)
    rng = random.Random(
        core.stable_seed("v060-t-final-initial", trial.seed, mode, width, start_index)
    )
    if start_index > 0:
        for _ in range(2 + 2 * start_index):
            first = rng.randrange(len(key))
            second = rng.randrange(len(key))
            key[first], key[second] = key[second], key[first]
    permutation = list(range(width))
    if start_index > 0:
        rng.shuffle(permutation)
    return key, np.asarray(permutation, dtype=np.int32)


def coordinate_run(
    trial: t.ColumnarTrial,
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    mode: str,
    width: int,
    label: str,
    cycles: int,
    perm_iterations: int,
    perm_restarts: int,
    mono_iterations: int,
    mono_restarts: int,
    start_index: int = 0,
    initial_key: np.ndarray | None = None,
    initial_permutation: np.ndarray | None = None,
) -> dict[str, Any]:
    trigram, unigram = model
    cipher = np.asarray(trial.cipher, dtype=np.int32)
    line_starts = np.asarray(trial.line_starts + [len(trial.plain)], dtype=np.int32)
    mode_flag = 0 if mode == "global" else 1
    if initial_key is None or initial_permutation is None:
        key, permutation = initial_key_and_permutation(
            trial, language, width, mode, start_index
        )
    else:
        key = initial_key.copy()
        permutation = initial_permutation.copy()

    trajectory: list[dict[str, Any]] = []
    for cycle in range(cycles):
        mapped = key[cipher]
        permutation, permutation_score = anneal_permutation_seeded(
            mapped,
            width,
            mode_flag,
            line_starts,
            trigram,
            unigram,
            permutation,
            perm_iterations,
            perm_restarts,
            int(core.stable_seed(
                "v060-t-final-perm", trial.seed, mode, width,
                label, start_index, cycle,
            ) & 0x7FFFFFFFFFFFFFFF),
        )
        detransposed = t.decrypt_columnar_array(
            cipher, width, permutation, mode_flag, line_starts
        )
        key, mono_score = mono.anneal_mono(
            detransposed,
            key,
            trigram,
            unigram,
            mono_iterations,
            mono_restarts,
            int(core.stable_seed(
                "v060-t-final-mono", trial.seed, mode, width,
                label, start_index, cycle,
            ) & 0x7FFFFFFFFFFFFFFF),
        )
        prediction = key[detransposed].tolist()
        trajectory.append({
            "cycle": cycle + 1,
            "permutation_score": float(permutation_score),
            "mono_score": float(mono_score),
            "accuracy": mono.fast_accuracy(trial.plain, prediction),
        })

    detransposed = t.decrypt_columnar_array(
        cipher, width, permutation, mode_flag, line_starts
    )
    raw_score = float(mono.score_key(detransposed, key, trigram, unigram))
    prediction = key[detransposed].tolist()
    return {
        "mode": mode,
        "width": width,
        "key": key,
        "permutation": permutation,
        "score": t.mdl_score(raw_score, width, len(trial.plain)),
        "raw_score": raw_score,
        "accuracy": mono.fast_accuracy(trial.plain, prediction),
        "prediction": prediction,
        "trajectory": trajectory,
        "start_index": start_index,
    }


def full_finish(
    trial: t.ColumnarTrial,
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    candidate: dict[str, Any],
    final_mono_iterations: int,
    final_mono_restarts: int,
    final_perm_iterations: int,
    final_perm_restarts: int,
    candidate_index: int,
) -> dict[str, Any]:
    trigram, unigram = model
    cipher = np.asarray(trial.cipher, dtype=np.int32)
    line_starts = np.asarray(trial.line_starts + [len(trial.plain)], dtype=np.int32)
    mode = str(candidate["mode"])
    width = int(candidate["width"])
    mode_flag = 0 if mode == "global" else 1
    key = candidate["key"].copy()
    permutation = candidate["permutation"].copy()

    detransposed = t.decrypt_columnar_array(
        cipher, width, permutation, mode_flag, line_starts
    )
    key, _ = mono.anneal_mono(
        detransposed,
        key,
        trigram,
        unigram,
        final_mono_iterations,
        final_mono_restarts,
        int(core.stable_seed(
            "v060-t-final-full-mono-1", trial.seed, candidate_index
        ) & 0x7FFFFFFFFFFFFFFF),
    )
    first_prediction = key[detransposed].tolist()
    first_raw = float(mono.score_key(detransposed, key, trigram, unigram))
    first = {
        "mode": mode,
        "width": width,
        "key": key.copy(),
        "permutation": permutation.copy(),
        "score": t.mdl_score(first_raw, width, len(trial.plain)),
        "raw_score": first_raw,
        "accuracy": mono.fast_accuracy(trial.plain, first_prediction),
        "prediction": first_prediction,
    }

    mapped = key[cipher]
    updated_permutation, _ = anneal_permutation_seeded(
        mapped,
        width,
        mode_flag,
        line_starts,
        trigram,
        unigram,
        permutation,
        final_perm_iterations,
        final_perm_restarts,
        int(core.stable_seed(
            "v060-t-final-full-perm", trial.seed, candidate_index
        ) & 0x7FFFFFFFFFFFFFFF),
    )
    updated_detransposed = t.decrypt_columnar_array(
        cipher, width, updated_permutation, mode_flag, line_starts
    )
    updated_key, _ = mono.anneal_mono(
        updated_detransposed,
        key,
        trigram,
        unigram,
        final_mono_iterations,
        final_mono_restarts,
        int(core.stable_seed(
            "v060-t-final-full-mono-2", trial.seed, candidate_index
        ) & 0x7FFFFFFFFFFFFFFF),
    )
    second_prediction = updated_key[updated_detransposed].tolist()
    second_raw = float(mono.score_key(
        updated_detransposed, updated_key, trigram, unigram
    ))
    second = {
        "mode": mode,
        "width": width,
        "key": updated_key,
        "permutation": updated_permutation,
        "score": t.mdl_score(second_raw, width, len(trial.plain)),
        "raw_score": second_raw,
        "accuracy": mono.fast_accuracy(trial.plain, second_prediction),
        "prediction": second_prediction,
    }
    return max((first, second), key=lambda row: row["score"])


def solve_trial(
    trial: t.ColumnarTrial,
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    widths = t.TEST_WIDTHS if trial.split == "test" else t.DEV_WIDTHS
    screen_specs = [
        (mode, width, start_index)
        for mode in t.MODES
        for width in widths
        for start_index in range(args.screen_starts)
    ]

    def screen_one(spec: tuple[str, int, int]) -> dict[str, Any]:
        mode, width, start_index = spec
        return coordinate_run(
            trial, language, model, mode, width, "screen",
            args.screen_cycles,
            args.screen_perm_iterations,
            args.screen_perm_restarts,
            args.screen_mono_iterations,
            args.screen_mono_restarts,
            start_index=start_index,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.candidate_workers
    ) as executor:
        screen_rows = list(executor.map(screen_one, screen_specs))

    best_by_structure: dict[tuple[str, int], dict[str, Any]] = {}
    for row in screen_rows:
        structure = (str(row["mode"]), int(row["width"]))
        incumbent = best_by_structure.get(structure)
        if incumbent is None or row["score"] > incumbent["score"]:
            best_by_structure[structure] = row
    screen_structures = sorted(
        best_by_structure.values(), key=lambda row: row["score"], reverse=True
    )

    def refine_one(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        index, candidate = item
        return coordinate_run(
            trial,
            language,
            model,
            str(candidate["mode"]),
            int(candidate["width"]),
            f"refine-{index}",
            args.refine_cycles,
            args.refine_perm_iterations,
            args.refine_perm_restarts,
            args.refine_mono_iterations,
            args.refine_mono_restarts,
            start_index=int(candidate["start_index"]),
            initial_key=candidate["key"],
            initial_permutation=candidate["permutation"],
        )

    # Every structure receives state-preserving refinement; top_refine controls
    # only the expensive final validated finish.
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.candidate_workers
    ) as executor:
        refined = list(executor.map(refine_one, enumerate(screen_structures)))
    refined.sort(key=lambda row: row["score"], reverse=True)

    final_inputs = refined[: args.top_refine]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.candidate_workers, len(final_inputs))
    ) as executor:
        finished = list(executor.map(
            lambda item: full_finish(
                trial,
                language,
                model,
                item[1],
                args.final_mono_iterations,
                args.final_mono_restarts,
                args.final_perm_iterations,
                args.final_perm_restarts,
                item[0],
            ),
            enumerate(final_inputs),
        ))
    selected = max(finished, key=lambda row: row["score"])
    permutation_correct = (
        selected["mode"] == trial.mode
        and int(selected["width"]) == trial.width
        and selected["permutation"].tolist() == trial.permutation
    )
    return {
        "iso": trial.iso,
        "split": trial.split,
        "replicate": trial.replicate,
        "true_mode": trial.mode,
        "true_width": trial.width,
        "selected_mode": selected["mode"],
        "selected_width": int(selected["width"]),
        "mode_correct": selected["mode"] == trial.mode,
        "width_correct": int(selected["width"]) == trial.width,
        "permutation_correct": permutation_correct,
        "accuracy": float(selected["accuracy"]),
        "exact": selected["prediction"] == trial.plain,
        "screen_best_accuracy": float(screen_structures[0]["accuracy"]),
        "screen_best_mode": screen_structures[0]["mode"],
        "screen_best_width": int(screen_structures[0]["width"]),
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
        "width_accuracy": statistics.fmean(row["width_correct"] for row in rows),
        "permutation_accuracy": statistics.fmean(
            row["permutation_correct"] for row in rows
        ),
        "gate": {
            "pass": (
                statistics.fmean(values) >= 0.80
                and statistics.median(values) >= 0.90
                and sum(value >= 0.80 for value in values) >= math.ceil(0.875 * len(values))
                and sum(row["mode_correct"] for row in rows) >= math.ceil(0.875 * len(rows))
                and sum(row["width_correct"] for row in rows) >= math.ceil(0.8125 * len(rows))
                and min(values) >= 0.40
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
    parser.add_argument("--replicate-start", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--screen-starts", type=int, default=3)
    parser.add_argument("--screen-cycles", type=int, default=3)
    parser.add_argument("--screen-perm-iterations", type=int, default=75000)
    parser.add_argument("--screen-perm-restarts", type=int, default=10)
    parser.add_argument("--screen-mono-iterations", type=int, default=150000)
    parser.add_argument("--screen-mono-restarts", type=int, default=12)
    parser.add_argument("--refine-cycles", type=int, default=2)
    parser.add_argument("--refine-perm-iterations", type=int, default=200000)
    parser.add_argument("--refine-perm-restarts", type=int, default=24)
    parser.add_argument("--refine-mono-iterations", type=int, default=300000)
    parser.add_argument("--refine-mono-restarts", type=int, default=20)
    parser.add_argument("--top-refine", type=int, default=6)
    parser.add_argument("--final-mono-iterations", type=int, default=700000)
    parser.add_argument("--final-mono-restarts", type=int, default=50)
    parser.add_argument("--final-perm-iterations", type=int, default=300000)
    parser.add_argument("--final-perm-restarts", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--candidate-workers", type=int, default=4)
    args = parser.parse_args()

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        root / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v060-family-t-final",
    )
    language = languages[args.iso]
    model = mono.build_language_model(language)
    trial_specs = [
        (mode, replicate)
        for mode in t.MODES
        for replicate in range(
            args.replicate_start, args.replicate_start + args.replicates
        )
    ]
    trials = [
        t.make_trial(language, args.split, args.length, mode, replicate)
        for mode, replicate in trial_specs
    ]

    def run_one(trial: t.ColumnarTrial) -> dict[str, Any]:
        row = solve_trial(trial, language, model, args)
        print("V060_T_FINAL_TRIAL", json.dumps(row, sort_keys=True), flush=True)
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
    print("V060_T_FINAL_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V060_T_FINAL_SHA256", payload["sha256"], flush=True)


if __name__ == "__main__":
    main()
