#!/usr/bin/env python3
"""v0.6 Family T3: fully blind substitution-plus-columnar coordinate solver."""
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
import v060_family_t_stage_a as t


def coordinate_candidate(
    trial: t.ColumnarTrial,
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    mode: str,
    width: int,
    cycles: int,
    perm_iterations: int,
    perm_restarts: int,
    mono_iterations: int,
    mono_restarts: int,
    label: str,
    initial_key: np.ndarray | None = None,
) -> dict[str, Any]:
    trigram, unigram = model
    cipher = np.asarray(trial.cipher, dtype=np.int32)
    line_starts = np.asarray(trial.line_starts + [len(trial.plain)], dtype=np.int32)
    mode_flag = 0 if mode == "global" else 1
    key = initial_key.copy() if initial_key is not None else mono.frequency_key(
        trial.cipher, language
    )
    permutation = np.arange(width, dtype=np.int32)
    trajectory = []
    for cycle in range(cycles):
        mapped = key[cipher]
        permutation, perm_raw = t.anneal_permutation(
            mapped,
            width,
            mode_flag,
            line_starts,
            trigram,
            unigram,
            perm_iterations,
            perm_restarts,
            int(core.stable_seed(
                "v060-t3-perm", trial.seed, mode, width, label, cycle
            ) & 0x7FFFFFFFFFFFFFFF),
        )
        detransposed = t.decrypt_columnar_array(
            cipher, width, permutation, mode_flag, line_starts
        )
        key, mono_raw = mono.anneal_mono(
            detransposed,
            key,
            trigram,
            unigram,
            mono_iterations,
            mono_restarts,
            int(core.stable_seed(
                "v060-t3-mono", trial.seed, mode, width, label, cycle
            ) & 0x7FFFFFFFFFFFFFFF),
        )
        prediction = key[detransposed].tolist()
        trajectory.append({
            "cycle": cycle + 1,
            "perm_score": float(perm_raw),
            "mono_score": float(mono_raw),
            "accuracy": mono.fast_accuracy(trial.plain, prediction),
        })
    detransposed = t.decrypt_columnar_array(
        cipher, width, permutation, mode_flag, line_starts
    )
    raw_score = mono.score_key(detransposed, key, trigram, unigram)
    prediction = key[detransposed].tolist()
    return {
        "mode": mode,
        "width": width,
        "permutation": permutation,
        "key": key,
        "score": t.mdl_score(float(raw_score), width, len(trial.plain)),
        "raw_score": float(raw_score),
        "accuracy": mono.fast_accuracy(trial.plain, prediction),
        "prediction": prediction,
        "trajectory": trajectory,
    }


def solve_trial(
    trial: t.ColumnarTrial,
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    screen_cycles: int,
    screen_perm_iterations: int,
    screen_perm_restarts: int,
    screen_mono_iterations: int,
    screen_mono_restarts: int,
    top_refine: int,
    refine_cycles: int,
    refine_perm_iterations: int,
    refine_perm_restarts: int,
    refine_mono_iterations: int,
    refine_mono_restarts: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    widths = t.TEST_WIDTHS if trial.split == "test" else t.DEV_WIDTHS
    screen = []
    for mode in t.MODES:
        for width in widths:
            screen.append(coordinate_candidate(
                trial, language, model, mode, width,
                screen_cycles,
                screen_perm_iterations,
                screen_perm_restarts,
                screen_mono_iterations,
                screen_mono_restarts,
                "screen",
            ))
    screen.sort(key=lambda row: row["score"], reverse=True)
    refined = []
    for candidate in screen[:top_refine]:
        refined.append(coordinate_candidate(
            trial,
            language,
            model,
            candidate["mode"],
            candidate["width"],
            refine_cycles,
            refine_perm_iterations,
            refine_perm_restarts,
            refine_mono_iterations,
            refine_mono_restarts,
            "refine",
            initial_key=candidate["key"],
        ))
    selected = max(refined, key=lambda row: row["score"])
    permutation_correct = (
        selected["mode"] == trial.mode
        and selected["width"] == trial.width
        and selected["permutation"].tolist() == trial.permutation
    )
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
        "permutation_correct": permutation_correct,
        "accuracy": selected["accuracy"],
        "exact": selected["prediction"] == trial.plain,
        "screen_accuracy": screen[0]["accuracy"],
        "screen_mode": screen[0]["mode"],
        "screen_width": screen[0]["width"],
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
                and sum(value >= 0.80 for value in values) >= 14
                and sum(row["mode_correct"] for row in rows) >= 14
                and sum(row["width_correct"] for row in rows) >= 13
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
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--screen-cycles", type=int, default=2)
    parser.add_argument("--screen-perm-iterations", type=int, default=50000)
    parser.add_argument("--screen-perm-restarts", type=int, default=8)
    parser.add_argument("--screen-mono-iterations", type=int, default=100000)
    parser.add_argument("--screen-mono-restarts", type=int, default=8)
    parser.add_argument("--top-refine", type=int, default=4)
    parser.add_argument("--refine-cycles", type=int, default=2)
    parser.add_argument("--refine-perm-iterations", type=int, default=200000)
    parser.add_argument("--refine-perm-restarts", type=int, default=32)
    parser.add_argument("--refine-mono-iterations", type=int, default=700000)
    parser.add_argument("--refine-mono-restarts", type=int, default=50)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        root / "corpus_manifest_v050.json", args.repo / ".cache" / "v060-family-t3"
    )
    language = languages[args.iso]
    model = mono.build_language_model(language)
    trials = [
        t.make_trial(language, args.split, args.length, mode, replicate)
        for mode in t.MODES
        for replicate in range(args.replicates)
    ]

    def run_one(trial: t.ColumnarTrial) -> dict[str, Any]:
        row = solve_trial(
            trial, language, model,
            args.screen_cycles,
            args.screen_perm_iterations,
            args.screen_perm_restarts,
            args.screen_mono_iterations,
            args.screen_mono_restarts,
            args.top_refine,
            args.refine_cycles,
            args.refine_perm_iterations,
            args.refine_perm_restarts,
            args.refine_mono_iterations,
            args.refine_mono_restarts,
        )
        print("V060_T3_TRIAL", json.dumps(row, sort_keys=True), flush=True)
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
    print("V060_T3_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V060_T3_SHA256", payload["sha256"], flush=True)


if __name__ == "__main__":
    main()
