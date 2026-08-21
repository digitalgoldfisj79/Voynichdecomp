#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import svt_v02 as svt

ISO = "de"
LENGTH = 1536
MODE = "periodic"
REPLICATE = 5002
N_STARTS = 24
SUCCESS = 0.90
PATHOLOGY_MIN_SUCCESSES = 18
PATHOLOGY_MEDIAN = 0.95


def invert_forward_maps(forward_maps: list[list[int]]) -> np.ndarray:
    period = len(forward_maps)
    a = len(forward_maps[0])
    inv = np.empty((period, a), dtype=np.int32)
    for s, fwd in enumerate(forward_maps):
        for plain, cipher in enumerate(fwd):
            inv[s, int(cipher)] = int(plain)
    return inv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = svt.core.load_languages(
        root / "corpus_manifest_v050.json", args.repo / ".cache" / "svt-v03-outlier"
    )
    language = languages[ISO]
    model = svt.mono.build_language_model(language)
    trigram, unigram = model
    trial = svt.make_svt_trial(language, "dev", LENGTH, MODE, REPLICATE)
    assert trial.head.period == 4, trial.head.period

    heads = trial.head.cipher
    cipher = np.asarray(heads, dtype=np.int32)
    phase = svt.v0._phase(
        len(heads), trial.head.period, trial.head.mode, trial.head.line_starts or [0]
    )
    local_cap = svt.local_cap_for_alphabet(len(language.alphabet))

    truth_inv = invert_forward_maps(trial.head.forward_maps)
    truth_raw = float(svt._raw_score(cipher, phase, truth_inv, trigram, unigram))
    truth_prediction = svt.v0.decode_stateful(heads, phase, truth_inv)
    truth_recovery = float(svt.mono.fast_accuracy(trial.head.plain, truth_prediction))

    rows = []
    for k in range(N_STARTS):
        seed = int(svt.core.stable_seed("svt-v03-outlier-diagnostic", trial.head.seed, k))
        initial = svt.initial_shared_inverse(heads, language, model, seed)
        inv, raw, moves = svt.coordinate_refine(
            cipher, phase, initial, trigram, unigram, local_cap
        )
        prediction = svt.v0.decode_stateful(heads, phase, inv)
        recovery = float(svt.mono.fast_accuracy(trial.head.plain, prediction))
        rows.append({
            "start": k,
            "seed": seed,
            "recovery": recovery,
            "raw_score": float(raw),
            "raw_gap_to_truth": float(truth_raw - raw),
            "local_moves": [int(x) for x in moves],
        })

    rec = [r["recovery"] for r in rows]
    successes = sum(x >= SUCCESS for x in rec)
    classification = (
        "SEARCH_PATHOLOGY"
        if successes >= PATHOLOGY_MIN_SUCCESSES and statistics.median(rec) >= PATHOLOGY_MEDIAN
        else "SOLVER_INSTABILITY"
    )
    payload = {
        "programme": "SVT-v0.3",
        "diagnostic": "isolated_A3_outlier_multistart",
        "binding": False,
        "voynich_opened": False,
        "case": {"iso": ISO, "length": LENGTH, "mode": MODE, "replicate": REPLICATE, "period": trial.head.period},
        "preregistered_rule": {
            "starts": N_STARTS,
            "success_recovery": SUCCESS,
            "search_pathology_if_successes_at_least": PATHOLOGY_MIN_SUCCESSES,
            "and_median_at_least": PATHOLOGY_MEDIAN,
        },
        "truth": {"recovery": truth_recovery, "raw_score": truth_raw},
        "rows": rows,
        "summary": {
            "successes_ge_090": successes,
            "mean_recovery": statistics.fmean(rec),
            "median_recovery": statistics.median(rec),
            "minimum_recovery": min(rec),
            "maximum_recovery": max(rec),
            "classification": classification,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
