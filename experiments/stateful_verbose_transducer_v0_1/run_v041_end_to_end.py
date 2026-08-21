#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from rapidfuzz.distance import Levenshtein

import svt_v02 as svt
import v04_semimarkov_segmenter as seg

ISO = "de"
LENGTH = 1536
OFFSET = 19000
N_STARTS = 12
SHORTLIST_K = 6


def load_language(repo: Path, cache_name: str):
    root = repo / "experiments" / "recoverability_frontier_v0_5"
    cache = repo / ".cache" / cache_name
    languages = svt.core.load_languages(root / "corpus_manifest_v050.json", cache)
    language = languages[ISO]
    model = svt.mono.build_language_model(language)
    return language, model


def inferred_head(surface, starts, surface_line_starts, seed):
    starts = [int(x) for x in starts]
    pos_to_idx = {p: i for i, p in enumerate(starts)}
    line_starts = []
    for s in surface_line_starts or [0]:
        s = int(s)
        if s not in pos_to_idx:
            raise RuntimeError(f"predicted segmentation missing observed line start {s}")
        line_starts.append(int(pos_to_idx[s]))
    return SimpleNamespace(
        cipher=[int(surface[s]) for s in starts],
        line_starts=line_starts,
        seed=int(seed),
    )


def screen_score(head, language, model, mode: str, period: int) -> float:
    heads = head.cipher
    cipher = np.asarray(heads, dtype=np.int32)
    trigram, unigram = model
    phase = svt.v0._phase(len(heads), period, mode, head.line_starts or [0])
    inv = svt.v0._initial_inverses(heads, phase, period, language)
    raw = float(svt.v0.score_stateful(cipher, phase, inv, trigram, unigram))
    return float(raw - svt.SCHEDULE_BIC_WEIGHT * max(0, period - 1) * np.log(max(2, len(heads))))


def fit_structure(head, language, model, mode: str, period: int, seed_tag: str) -> dict:
    heads = head.cipher
    cipher = np.asarray(heads, dtype=np.int32)
    trigram, unigram = model
    phase = svt.v0._phase(len(heads), period, mode, head.line_starts or [0])
    local_cap = svt.local_cap_for_alphabet(len(language.alphabet))
    rows = []
    for k in range(N_STARTS):
        seed = int(svt.core.stable_seed(seed_tag, head.seed, mode, period, k))
        initial = svt.initial_shared_inverse(heads, language, model, seed)
        inv, raw, moves = svt.coordinate_refine(cipher, phase, initial, trigram, unigram, local_cap)
        prediction = [int(x) for x in svt.v0.decode_stateful(heads, phase, inv)]
        score = float(svt.candidate_score(raw, moves, period, len(heads)))
        rows.append({
            "start": int(k),
            "seed": int(seed),
            "raw_score": float(raw),
            "score": score,
            "local_moves": [int(x) for x in moves],
            "prediction": prediction,
        })
    selected = max(rows, key=lambda x: x["score"])
    return {"mode": mode, "period": int(period), "starts": rows, "selected": selected}


def divisors_ge2(period: int) -> list[int]:
    return [d for d in range(2, period + 1) if period % d == 0]


def canonicalise(head, language, model, selected_fit: dict) -> dict:
    mode = selected_fit["mode"]
    period = int(selected_fit["period"])
    fits = {period: selected_fit}
    for d in divisors_ge2(period):
        if d == period:
            continue
        fits[d] = fit_structure(head, language, model, mode, d, "svt-v041-canonical")
    best = max(fits.values(), key=lambda row: row["selected"]["score"])
    return {
        "canonical_mode": best["mode"],
        "canonical_period": int(best["period"]),
        "canonical_score": float(best["selected"]["score"]),
        "prediction": best["selected"]["prediction"],
        "candidate_divisors": sorted(int(x) for x in fits),
    }


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
    language, model = load_language(args.repo, f"svt-v041-{args.mode}-{args.replicate}")
    trial = svt.make_svt_trial(language, "dev", LENGTH, args.mode, rep)

    fitted_seg = seg.fit(
        trial.surface,
        trial.surface_line_starts,
        len(language.alphabet),
        int(svt.core.stable_seed("svt-v041-seg", trial.head.seed)),
    )
    boundary_f1 = seg.boundary_f1(fitted_seg.starts, trial.head_positions)
    count_error = abs(len(fitted_seg.starts) - len(trial.head_positions)) / max(1, len(trial.head_positions))
    head = inferred_head(trial.surface, fitted_seg.starts, trial.surface_line_starts, trial.head.seed)

    screen = []
    for mode in svt.MODES:
        for period in svt.CANDIDATE_PERIODS:
            screen.append({"mode": mode, "period": int(period), "screen_score": screen_score(head, language, model, mode, period)})
    screen.sort(key=lambda x: x["screen_score"], reverse=True)
    shortlist = screen[:SHORTLIST_K]

    refined = [fit_structure(head, language, model, row["mode"], int(row["period"]), "svt-v041-blind") for row in shortlist]
    selected = max(refined, key=lambda row: row["selected"]["score"])
    canonical = canonicalise(head, language, model, selected)
    recovery = sequence_recovery(trial.head.plain, canonical["prediction"])

    payload = {
        "programme": "SVT-v0.4.1",
        "stage": "end_to_end_hidden_segmentation_blind_state_key",
        "binding": True,
        "voynich_opened": False,
        "iso": ISO,
        "replicate": int(rep),
        "generator_mode": args.mode,
        "true_mode": trial.head.mode,
        "true_period": int(trial.head.period),
        "surface_length": int(len(trial.surface)),
        "true_units": int(len(trial.head_positions)),
        "predicted_units": int(len(fitted_seg.starts)),
        "boundary_f1": float(boundary_f1),
        "count_relative_error": float(count_error),
        "screen_top6": shortlist,
        "selected_mode_precanonical": selected["mode"],
        "selected_period_precanonical": int(selected["period"]),
        "canonical_mode": canonical["canonical_mode"],
        "canonical_period": int(canonical["canonical_period"]),
        "canonical_exact": bool(canonical["canonical_mode"] == trial.head.mode and canonical["canonical_period"] == int(trial.head.period)),
        "sequence_recovery": float(recovery),
        "decoded_length": int(len(canonical["prediction"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
