#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rapidfuzz.distance import Levenshtein

import svt_v02 as svt
import v04_semimarkov_segmenter as seg
import joint_semimarkov_v042 as joint

ISO = "de"
LENGTH = 1536
OFFSET = 23000


def load_language(repo: Path, cache_name: str):
    root = repo / "experiments" / "recoverability_frontier_v0_5"
    cache = repo / ".cache" / cache_name
    languages = svt.core.load_languages(root / "corpus_manifest_v050.json", cache)
    language = languages[ISO]
    model = svt.mono.build_language_model(language)
    return language, model


def sequence_recovery(truth, pred) -> float:
    denom = max(1, len(truth), len(pred))
    return float(1.0 - Levenshtein.distance(list(truth), list(pred)) / denom)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mode", choices=list(svt.MODES), required=True)
    ap.add_argument("--replicate", type=int, required=True)
    args = ap.parse_args()

    rep = OFFSET + args.replicate
    language, model = load_language(args.repo, f"svt-v042-{args.mode}-{args.replicate}")
    trial = svt.make_svt_trial(language, "dev", LENGTH, args.mode, rep)

    # Binding solve: only surface, observed line starts, language/model and trial seed enter.
    solved = joint.solve(trial.surface, trial.surface_line_starts, language, model, trial.head.seed)
    canonical = solved["canonical"]

    # Truth is used only below this line for evaluation.
    starts = canonical["starts"]
    boundary_f1 = seg.boundary_f1(starts, trial.head_positions)
    count_error = abs(len(starts) - len(trial.head_positions)) / max(1, len(trial.head_positions))
    recovery = sequence_recovery(trial.head.plain, canonical["prediction"])
    exact = canonical["mode"] == trial.head.mode and int(canonical["period"]) == int(trial.head.period)
    truth_rank = next(
        i + 1 for i, r in enumerate(solved["cheap_ranking"])
        if r["mode"] == trial.head.mode and int(r["period"]) == int(trial.head.period)
    )

    payload = {
        "programme": "SVT-v0.4.2",
        "stage": "joint_semimarkov_segmentation_state_key",
        "binding": True,
        "voynich_opened": False,
        "iso": ISO,
        "replicate": int(rep),
        "true_mode": trial.head.mode,
        "true_period": int(trial.head.period),
        "cheap_truth_rank": int(truth_rank),
        "shortlist": solved["shortlist"],
        "selected_mode_precanonical": solved["selected_precanonical"]["mode"],
        "selected_period_precanonical": int(solved["selected_precanonical"]["period"]),
        "canonical_mode": canonical["mode"],
        "canonical_period": int(canonical["period"]),
        "canonical_exact": bool(exact),
        "boundary_f1": float(boundary_f1),
        "true_units": int(len(trial.head_positions)),
        "predicted_units": int(len(starts)),
        "count_relative_error": float(count_error),
        "sequence_recovery": float(recovery),
        "canonical_score": float(canonical["score"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
