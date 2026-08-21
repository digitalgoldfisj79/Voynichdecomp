#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import svt_v02 as svt

ISO = "de"
LENGTH = 1536
REPLICATE_OFFSET = 11000
N_STARTS = 12
SHORTLIST_K = 6


def screen_score(heads, line_starts, language, model, mode, period):
    cipher = np.asarray(heads, dtype=np.int32)
    trigram, unigram = model
    phase = svt.v0._phase(len(heads), period, mode, line_starts or [0])
    inv = svt.v0._initial_inverses(heads, phase, period, language)
    raw = float(svt.v0.score_stateful(cipher, phase, inv, trigram, unigram))
    return float(raw - svt.SCHEDULE_BIC_WEIGHT * max(0, period - 1) * np.log(max(2, len(heads))))


def refine_candidate(trial, language, model, mode, period):
    heads = trial.head.cipher
    cipher = np.asarray(heads, dtype=np.int32)
    trigram, unigram = model
    phase = svt.v0._phase(len(heads), period, mode, trial.head.line_starts or [0])
    local_cap = svt.local_cap_for_alphabet(len(language.alphabet))
    starts = []
    for k in range(N_STARTS):
        seed = int(svt.core.stable_seed("svt-v033-binding", trial.head.seed, mode, period, k))
        initial = svt.initial_shared_inverse(heads, language, model, seed)
        inv, raw, moves = svt.coordinate_refine(cipher, phase, initial, trigram, unigram, local_cap)
        prediction = svt.v0.decode_stateful(heads, phase, inv)
        recovery = float(svt.mono.fast_accuracy(trial.head.plain, prediction))
        score = float(svt.candidate_score(raw, moves, period, len(heads)))
        starts.append({
            "start": k,
            "seed": seed,
            "recovery": recovery,
            "raw_score": float(raw),
            "score": score,
            "local_moves": [int(x) for x in moves],
        })
    selected = max(starts, key=lambda x: x["score"])
    return {
        "mode": mode,
        "period": int(period),
        "starts": starts,
        "selected": selected,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--true-mode", choices=list(svt.MODES), required=True)
    ap.add_argument("--replicate", type=int, required=True)
    args = ap.parse_args()

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    cache = args.repo / ".cache" / f"svt-v033-{args.true_mode}-{args.replicate}"
    languages = svt.core.load_languages(root / "corpus_manifest_v050.json", cache)
    language = languages[ISO]
    model = svt.mono.build_language_model(language)

    replicate = REPLICATE_OFFSET + args.replicate
    trial = svt.make_svt_trial(language, "dev", LENGTH, args.true_mode, replicate)

    screen = []
    for mode in svt.MODES:
        for period in svt.CANDIDATE_PERIODS:
            screen.append({
                "mode": mode,
                "period": int(period),
                "screen_score": screen_score(
                    trial.head.cipher,
                    trial.head.line_starts,
                    language,
                    model,
                    mode,
                    period,
                ),
            })
    screen.sort(key=lambda x: x["screen_score"], reverse=True)
    shortlist = screen[:SHORTLIST_K]
    truth_rank = next(
        i + 1 for i, row in enumerate(screen)
        if row["mode"] == trial.head.mode and row["period"] == trial.head.period
    )

    refined = [
        refine_candidate(trial, language, model, row["mode"], row["period"])
        for row in shortlist
    ]
    selected = max(refined, key=lambda x: x["selected"]["score"])

    payload = {
        "programme": "SVT-v0.3.3",
        "stage": "A3.3_screen6_then_multistart12",
        "binding": True,
        "voynich_opened": False,
        "iso": ISO,
        "length": LENGTH,
        "true_mode": trial.head.mode,
        "true_period": int(trial.head.period),
        "replicate": replicate,
        "screen_truth_rank": int(truth_rank),
        "screen_top6": shortlist,
        "refined": refined,
        "selected_mode": selected["mode"],
        "selected_period": int(selected["period"]),
        "selected_recovery": float(selected["selected"]["recovery"]),
        "selected_score": float(selected["selected"]["score"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "true_mode": payload["true_mode"],
        "true_period": payload["true_period"],
        "replicate": replicate,
        "screen_truth_rank": payload["screen_truth_rank"],
        "selected_mode": payload["selected_mode"],
        "selected_period": payload["selected_period"],
        "selected_recovery": payload["selected_recovery"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
