#!/usr/bin/env python3
"""Oracle-component gates for v0.6 Family P wheel ciphers."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
V05 = HERE.parent / "recoverability_frontier_v0_5"
if str(V05) not in sys.path:
    sys.path.insert(0, str(V05))

import recoverability_v050 as core
import mono_solver_v051 as mono
import v060_family_p_stage_a as wheel


def solve_oracles(
    trial: wheel.WheelTrial,
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    mono_iterations: int,
    mono_restarts: int,
    shift_iterations: int,
    shift_restarts: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    trigram, unigram = model
    a = len(language.alphabet)
    true_phase = wheel.phase_array(trial.length, trial.period, trial.mode, trial.line_starts)
    detrended = np.asarray(
        [
            (trial.cipher[i] - trial.shifts[int(true_phase[i])]) % a
            for i in range(trial.length)
        ],
        dtype=np.int32,
    )
    initial = mono.frequency_key(detrended.tolist(), language)
    solved_inverse, mono_score = mono.anneal_mono(
        detrended,
        initial,
        trigram,
        unigram,
        mono_iterations,
        mono_restarts,
        int(core.stable_seed("v060-p-oracle-schedule", trial.seed) & 0x7FFFFFFFFFFFFFFF),
    )
    schedule_prediction = solved_inverse[detrended].tolist()

    cipher = np.asarray(trial.cipher, dtype=np.int32)
    true_inverse = np.asarray(trial.base_inverse, dtype=np.int32)
    period_rows = []
    for period in wheel.CANDIDATE_PERIODS:
        phase = wheel.phase_array(trial.length, period, trial.mode, trial.line_starts)
        solved_shifts, raw_score = wheel.anneal_shifts(
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
        prediction = wheel.decode(trial.cipher, phase, true_inverse, solved_shifts)
        period_rows.append(
            {
                "period": period,
                "score": wheel.mdl_score(float(raw_score), period, trial.length, a),
                "accuracy": mono.fast_accuracy(trial.plain, prediction),
            }
        )
    selected = max(period_rows, key=lambda row: row["score"])
    return {
        "iso": trial.iso,
        "split": trial.split,
        "length": trial.length,
        "mode": trial.mode,
        "replicate": trial.replicate,
        "true_period": trial.period,
        "oracle_schedule_accuracy": mono.fast_accuracy(trial.plain, schedule_prediction),
        "oracle_schedule_exact": schedule_prediction == trial.plain,
        "oracle_schedule_score": float(mono_score),
        "oracle_base_accuracy": selected["accuracy"],
        "oracle_base_selected_period": selected["period"],
        "oracle_base_period_correct": selected["period"] == trial.period,
        "elapsed_seconds": time.perf_counter() - started,
    }


def metric(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    values = [float(row[field]) for row in rows]
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "at_least_80_rate": statistics.fmean(value >= 0.80 for value in values),
        "at_least_90_rate": statistics.fmean(value >= 0.90 for value in values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iso", default="en")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--length", type=int, default=384)
    parser.add_argument("--modes", nargs="+", choices=wheel.MODES, default=list(wheel.MODES))
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--mono-iterations", type=int, default=700000)
    parser.add_argument("--mono-restarts", type=int, default=50)
    parser.add_argument("--shift-iterations", type=int, default=50000)
    parser.add_argument("--shift-restarts", type=int, default=12)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        root / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v060-family-p-oracle",
    )
    language = languages[args.iso]
    model = mono.build_language_model(language)
    trials = [
        wheel.make_trial(language, args.split, args.length, mode, replicate)
        for mode in args.modes
        for replicate in range(args.replicates)
    ]

    def run_one(trial: wheel.WheelTrial) -> dict[str, Any]:
        row = solve_oracles(
            trial,
            language,
            model,
            args.mono_iterations,
            args.mono_restarts,
            args.shift_iterations,
            args.shift_restarts,
        )
        print("V060_P_ORACLE_TRIAL", json.dumps(row, sort_keys=True), flush=True)
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(run_one, trials))
    summary = {
        "trials": len(rows),
        "oracle_schedule": metric(rows, "oracle_schedule_accuracy"),
        "oracle_base": metric(rows, "oracle_base_accuracy"),
        "oracle_schedule_exact_rate": statistics.fmean(row["oracle_schedule_exact"] for row in rows),
        "oracle_base_period_accuracy": statistics.fmean(row["oracle_base_period_correct"] for row in rows),
    }
    payload = {
        "config": vars(args) | {"repo": str(args.repo), "output": str(args.output)},
        "rows": rows,
        "summary": summary,
    }
    scientific = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(scientific).hexdigest()
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V060_P_ORACLE_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V060_P_ORACLE_SHA256", payload["sha256"], flush=True)


if __name__ == "__main__":
    main()
