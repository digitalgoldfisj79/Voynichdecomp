#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import svt_v02 as svt

ISO = "de"
LENGTH = 1536
REPLICATE_OFFSET = 7000
N_STARTS = 12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mode", choices=list(svt.MODES), required=True)
    ap.add_argument("--replicate", type=int, required=True)
    args = ap.parse_args()

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = svt.core.load_languages(
        root / "corpus_manifest_v050.json", args.repo / ".cache" / f"svt-v031-{args.mode}-{args.replicate}"
    )
    language = languages[ISO]
    model = svt.mono.build_language_model(language)
    trigram, unigram = model
    replicate = REPLICATE_OFFSET + args.replicate
    trial = svt.make_svt_trial(language, "dev", LENGTH, args.mode, replicate)

    heads = trial.head.cipher
    cipher = np.asarray(heads, dtype=np.int32)
    phase = svt.v0._phase(
        len(heads), trial.head.period, trial.head.mode, trial.head.line_starts or [0]
    )
    local_cap = svt.local_cap_for_alphabet(len(language.alphabet))

    rows = []
    for k in range(N_STARTS):
        seed = int(svt.core.stable_seed("svt-v031-multistart", trial.head.seed, k))
        initial = svt.initial_shared_inverse(heads, language, model, seed)
        inv, raw, moves = svt.coordinate_refine(
            cipher, phase, initial, trigram, unigram, local_cap
        )
        prediction = svt.v0.decode_stateful(heads, phase, inv)
        recovery = float(svt.mono.fast_accuracy(trial.head.plain, prediction))
        score = svt.candidate_score(raw, moves, trial.head.period, len(heads))
        rows.append({
            "start": k,
            "seed": seed,
            "recovery": recovery,
            "raw_score": float(raw),
            "score": float(score),
            "local_moves": [int(x) for x in moves],
        })

    selected = max(rows, key=lambda r: r["score"])
    payload = {
        "programme": "SVT-v0.3.1",
        "stage": "A3.1_true_structure_multistart",
        "binding": True,
        "voynich_opened": False,
        "selection_rule": "maximum penalised model score across exactly 12 starts",
        "iso": ISO,
        "length": LENGTH,
        "mode": args.mode,
        "replicate": replicate,
        "true_period": trial.head.period,
        "starts": rows,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "mode": args.mode,
        "replicate": replicate,
        "period": trial.head.period,
        "selected_recovery": selected["recovery"],
        "selected_score": selected["score"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
