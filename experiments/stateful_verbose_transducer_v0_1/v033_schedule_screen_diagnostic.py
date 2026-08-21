#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import svt_v02 as svt

ISO = "de"
LENGTH = 1536
OFFSETS = (5000, 7000)
MODES = tuple(svt.MODES)
PERIODS = tuple(svt.CANDIDATE_PERIODS)
TOPKS = (1, 2, 4, 6, 8)


def screen_candidate(heads, line_starts, language, model, mode, period):
    cipher = np.asarray(heads, dtype=np.int32)
    trigram, unigram = model
    phase = svt.v0._phase(len(heads), period, mode, line_starts or [0])
    # Cheap, truth-free candidate initialization: independent statewise frequency keys.
    inv = svt.v0._initial_inverses(heads, phase, period, language)
    raw = float(svt.v0.score_stateful(cipher, phase, inv, trigram, unigram))
    # Ranking-only score. Period penalty is the same sparse-schedule charge used by v0.2.
    score = float(raw - svt.SCHEDULE_BIC_WEIGHT * max(0, period - 1) * np.log(max(2, len(heads))))
    return score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = svt.core.load_languages(
        root / "corpus_manifest_v050.json", args.repo / ".cache" / "svt-v033-screen"
    )
    language = languages[ISO]
    model = svt.mono.build_language_model(language)

    rows = []
    for offset in OFFSETS:
        for true_mode in MODES:
            for r in range(4):
                replicate = offset + r
                trial = svt.make_svt_trial(language, "dev", LENGTH, true_mode, replicate)
                candidates = []
                for mode in MODES:
                    for period in PERIODS:
                        score = screen_candidate(
                            trial.head.cipher,
                            trial.head.line_starts,
                            language,
                            model,
                            mode,
                            period,
                        )
                        candidates.append({"mode": mode, "period": int(period), "score": score})
                candidates.sort(key=lambda x: x["score"], reverse=True)
                truth_rank = next(
                    i + 1
                    for i, c in enumerate(candidates)
                    if c["mode"] == trial.head.mode and c["period"] == trial.head.period
                )
                rows.append({
                    "offset": offset,
                    "true_mode": trial.head.mode,
                    "true_period": int(trial.head.period),
                    "replicate": replicate,
                    "truth_rank": int(truth_rank),
                    "top8": candidates[:8],
                })

    summary = {
        "trials": len(rows),
        "mean_truth_rank": float(np.mean([r["truth_rank"] for r in rows])),
        "median_truth_rank": float(np.median([r["truth_rank"] for r in rows])),
        "maximum_truth_rank": int(max(r["truth_rank"] for r in rows)),
        "topk_recall": {
            str(k): int(sum(r["truth_rank"] <= k for r in rows)) for k in TOPKS
        },
    }
    payload = {
        "programme": "SVT-v0.3.3-screen-diagnostic",
        "binding": False,
        "voynich_opened": False,
        "used_offsets": list(OFFSETS),
        "rows": rows,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
