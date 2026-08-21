#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import svt_v02 as svt
import v04_semimarkov_segmenter as seg

ISO = "de"
LENGTH = 1536
OFFSET = 17000


def load_language(repo: Path, cache_name: str):
    root = repo / "experiments" / "recoverability_frontier_v0_5"
    cache = repo / ".cache" / cache_name
    languages = svt.core.load_languages(root / "corpus_manifest_v050.json", cache)
    return languages[ISO]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mode", choices=list(svt.MODES), required=True)
    ap.add_argument("--replicate", type=int, required=True)
    args = ap.parse_args()

    rep = OFFSET + args.replicate
    language = load_language(args.repo, f"svt-v04-seg-{args.mode}-{args.replicate}")
    trial = svt.make_svt_trial(language, "dev", LENGTH, args.mode, rep)

    fitted = seg.fit(
        trial.surface,
        trial.surface_line_starts,
        len(language.alphabet),
        int(svt.core.stable_seed("svt-v04-binding", trial.head.seed)),
    )
    f1 = seg.boundary_f1(fitted.starts, trial.head_positions)
    count_rel_error = abs(len(fitted.starts) - len(trial.head_positions)) / max(1, len(trial.head_positions))

    legacy = svt.v0.top_segmentations(
        trial.surface,
        trial.surface_line_starts,
        len(language.alphabet),
        beam=1,
    )
    legacy_f1 = seg.boundary_f1(legacy[0].starts, trial.head_positions) if legacy else 0.0

    payload = {
        "programme": "SVT-v0.4",
        "stage": "S0_hidden_segmentation_component",
        "binding": True,
        "voynich_opened": False,
        "iso": ISO,
        "mode_generator_only": args.mode,
        "replicate": int(rep),
        "head_length": int(len(trial.head_positions)),
        "surface_length": int(len(trial.surface)),
        "selected_restart": int(fitted.restart),
        "em_iterations": int(fitted.iterations),
        "model_score": float(fitted.score),
        "predicted_units": int(len(fitted.starts)),
        "true_units": int(len(trial.head_positions)),
        "count_relative_error": float(count_rel_error),
        "boundary_f1": float(f1),
        "legacy_surprisal_f1": float(legacy_f1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
