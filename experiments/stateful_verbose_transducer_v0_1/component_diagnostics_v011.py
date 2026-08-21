#!/usr/bin/env python3
"""Non-binding SVT v0.1.1 component diagnostics.

These diagnostics expose synthetic truth only to identify which instrument
component is failing. They do not alter any binding gate and contain no Voynich
loader.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

import svt_v011 as svt
import svt_v01 as v0


def invert_forward_maps(forward_maps: list[list[int]]) -> np.ndarray:
    period = len(forward_maps)
    a = len(forward_maps[0])
    inv = np.empty((period, a), dtype=np.int32)
    for state, forward in enumerate(forward_maps):
        for plain, cipher in enumerate(forward):
            inv[state, int(cipher)] = int(plain)
    return inv


def lm_raw(decoded: list[int], model: tuple[np.ndarray, np.ndarray]) -> float:
    trigram, unigram = model
    if not decoded:
        return -1e300
    score = 0.15 * float(unigram[int(decoded[0])])
    if len(decoded) > 1:
        score += 0.15 * float(unigram[int(decoded[1])])
    for i in range(2, len(decoded)):
        x, y, z = int(decoded[i-2]), int(decoded[i-1]), int(decoded[i])
        score += float(trigram[x, y, z]) + 0.15 * float(unigram[z])
    return score


def true_structure_key_search(trial: v0.SVTTrial, language, model, iterations: int, restarts: int) -> dict:
    heads = trial.head.cipher
    cipher = np.asarray(heads, dtype=np.int32)
    trigram, unigram = model
    freq = svt.mono.frequency_key(heads, language)
    base, _ = svt.mono.anneal_mono(
        cipher,
        np.asarray(freq, dtype=np.int32),
        trigram,
        unigram,
        max(5000, iterations // 2),
        max(2, restarts // 2),
        int(svt.core.stable_seed("svt-v011-diag-base", trial.head.seed) & 0x7FFFFFFFFFFFFFFF),
    )
    phase = v0._phase(
        len(heads), trial.head.period, trial.head.mode, trial.head.line_starts or [0]
    )
    _, inv, penalised, raw, swaps = svt.anneal_factorized(
        cipher,
        phase,
        np.asarray(base, dtype=np.int32),
        trial.head.period,
        trigram,
        unigram,
        iterations,
        restarts,
        int(svt.core.stable_seed("svt-v011-diag-true-structure", trial.head.seed) & 0x7FFFFFFFFFFFFFFF),
    )
    prediction = v0.decode_stateful(heads, phase, inv)
    return {
        "recovery": float(svt.mono.fast_accuracy(trial.head.plain, prediction)),
        "penalised_score": float(penalised),
        "raw_score": float(raw),
        "estimated_local_swaps": float(swaps),
    }


def truth_map_checks(trial: v0.SVTTrial, model) -> dict:
    true_inv = invert_forward_maps(trial.head.forward_maps)
    true_phase = v0._phase(
        len(trial.head.cipher), trial.head.period, trial.head.mode, trial.head.line_starts or [0]
    )
    exact = v0.decode_stateful(trial.head.cipher, true_phase, true_inv)
    exact_recovery = float(svt.mono.fast_accuracy(trial.head.plain, exact))

    # Holding the true period and true state maps fixed, ask whether the LM alone
    # selects periodic versus line-reset phase assignment.
    mode_rows = []
    for mode in svt.MODES:
        phase = v0._phase(
            len(trial.head.cipher), trial.head.period, mode, trial.head.line_starts or [0]
        )
        decoded = v0.decode_stateful(trial.head.cipher, phase, true_inv)
        mode_rows.append({
            "mode": mode,
            "score": lm_raw(decoded, model),
            "recovery": float(svt.mono.fast_accuracy(trial.head.plain, decoded)),
        })
    selected = max(mode_rows, key=lambda row: row["score"])
    return {
        "truth_map_exact_recovery": exact_recovery,
        "mode_selected_with_truth_maps": selected["mode"],
        "mode_correct_with_truth_maps": selected["mode"] == trial.head.mode,
        "mode_rows": mode_rows,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iso", default="de")
    p.add_argument("--length", type=int, default=192)
    p.add_argument("--replicates", type=int, default=2)
    p.add_argument("--iterations", type=int, default=40000)
    p.add_argument("--restarts", type=int, default=6)
    args = p.parse_args()

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = svt.core.load_languages(
        root / "corpus_manifest_v050.json", args.repo / ".cache" / "svt-v011-diag"
    )
    language = languages[args.iso]
    model = svt.mono.build_language_model(language)
    trials = [
        svt.make_svt_trial(language, "dev", args.length, mode, replicate)
        for mode in svt.MODES
        for replicate in range(args.replicates)
    ]

    rows = []
    for trial in trials:
        row = {
            "true_mode": trial.head.mode,
            "true_period": trial.head.period,
            "replicate": trial.head.replicate,
        }
        row.update(truth_map_checks(trial, model))
        row["true_structure_key_search"] = true_structure_key_search(
            trial, language, model, args.iterations, args.restarts
        )
        rows.append(row)

    key_rec = [row["true_structure_key_search"]["recovery"] for row in rows]
    payload = {
        "programme": "SVT-v0.1.1-component-diagnostics",
        "binding": False,
        "iso": args.iso,
        "length": args.length,
        "trials": len(rows),
        "rows": rows,
        "summary": {
            "truth_map_exact_mean_recovery": statistics.fmean(
                row["truth_map_exact_recovery"] for row in rows
            ),
            "truth_map_mode_accuracy": statistics.fmean(
                row["mode_correct_with_truth_maps"] for row in rows
            ),
            "true_structure_key_mean_recovery": statistics.fmean(key_rec),
            "true_structure_key_median_recovery": statistics.median(key_rec),
            "true_structure_key_min_recovery": min(key_rec),
            "true_structure_trials_ge_085": sum(x >= 0.85 for x in key_rec),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
