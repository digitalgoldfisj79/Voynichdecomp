#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import svt_v02 as svt

ISO = "de"
LENGTH = 1536
REPLICATE_OFFSET = 9000
N_STARTS = 12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--true-mode", choices=list(svt.MODES), required=True)
    ap.add_argument("--replicate", type=int, required=True)
    ap.add_argument("--candidate-mode", choices=list(svt.MODES), required=True)
    ap.add_argument("--candidate-period", type=int, choices=list(svt.CANDIDATE_PERIODS), required=True)
    args = ap.parse_args()

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    cache = args.repo / ".cache" / (
        f"svt-v032-{args.true_mode}-{args.replicate}-{args.candidate_mode}-{args.candidate_period}"
    )
    languages = svt.core.load_languages(root / "corpus_manifest_v050.json", cache)
    language = languages[ISO]
    model = svt.mono.build_language_model(language)
    trigram, unigram = model

    replicate = REPLICATE_OFFSET + args.replicate
    trial = svt.make_svt_trial(language, "dev", LENGTH, args.true_mode, replicate)
    heads = trial.head.cipher
    cipher = np.asarray(heads, dtype=np.int32)
    phase = svt.v0._phase(
        len(heads), args.candidate_period, args.candidate_mode, trial.head.line_starts or [0]
    )
    local_cap = svt.local_cap_for_alphabet(len(language.alphabet))

    starts = []
    for k in range(N_STARTS):
        seed = int(
            svt.core.stable_seed(
                "svt-v032-blind-multistart",
                trial.head.seed,
                args.candidate_mode,
                args.candidate_period,
                k,
            )
        )
        initial = svt.initial_shared_inverse(heads, language, model, seed)
        inv, raw, moves = svt.coordinate_refine(
            cipher, phase, initial, trigram, unigram, local_cap
        )
        prediction = svt.v0.decode_stateful(heads, phase, inv)
        recovery = float(svt.mono.fast_accuracy(trial.head.plain, prediction))
        score = float(svt.candidate_score(raw, moves, args.candidate_period, len(heads)))
        starts.append({
            "start": k,
            "seed": seed,
            "recovery": recovery,
            "raw_score": float(raw),
            "score": score,
            "local_moves": [int(x) for x in moves],
        })

    selected = max(starts, key=lambda r: r["score"])
    payload = {
        "programme": "SVT-v0.3.2",
        "stage": "A3.2_blind_structure_candidate",
        "binding": True,
        "voynich_opened": False,
        "selection_rule": "maximum frozen penalised score across exactly 12 starts",
        "iso": ISO,
        "length": LENGTH,
        "true_mode": args.true_mode,
        "true_period": int(trial.head.period),
        "replicate": replicate,
        "candidate_mode": args.candidate_mode,
        "candidate_period": int(args.candidate_period),
        "starts": starts,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "true_mode": args.true_mode,
        "true_period": int(trial.head.period),
        "replicate": replicate,
        "candidate_mode": args.candidate_mode,
        "candidate_period": int(args.candidate_period),
        "selected_recovery": selected["recovery"],
        "selected_score": selected["score"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
