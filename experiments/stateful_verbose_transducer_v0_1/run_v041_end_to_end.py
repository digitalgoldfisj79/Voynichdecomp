#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

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


def edit_recovery(truth: list[int], prediction: list[int]) -> float:
    denom = max(1, len(truth), len(prediction))
    return float(1.0 - Levenshtein.distance(truth, prediction) / denom)


def seq_sha256(values: list[int]) -> str:
    return hashlib.sha256(",".join(str(int(x)) for x in values).encode("utf-8")).hexdigest()


def screen_score(
    heads: list[int],
    head_line_starts: list[int],
    language,
    model,
    mode: str,
    period: int,
) -> float:
    cipher = np.asarray(heads, dtype=np.int32)
    trigram, unigram = model
    phase = svt.v0._phase(len(heads), period, mode, head_line_starts or [0])
    inv = svt.v0._initial_inverses(heads, phase, period, language)
    raw = float(svt.v0.score_stateful(cipher, phase, inv, trigram, unigram))
    return float(
        raw
        - svt.SCHEDULE_BIC_WEIGHT
        * max(0, period - 1)
        * np.log(max(2, len(heads)))
    )


def fit_structure(
    heads: list[int],
    head_line_starts: list[int],
    seed_base: int,
    language,
    model,
    mode: str,
    period: int,
    seed_tag: str,
    truth_plain: list[int],
) -> dict[str, Any]:
    cipher = np.asarray(heads, dtype=np.int32)
    trigram, unigram = model
    phase = svt.v0._phase(len(heads), period, mode, head_line_starts or [0])
    local_cap = svt.local_cap_for_alphabet(len(language.alphabet))
    rows = []
    internals = []
    for k in range(N_STARTS):
        seed = int(svt.core.stable_seed(seed_tag, seed_base, mode, period, k))
        initial = svt.initial_shared_inverse(heads, language, model, seed)
        inv, raw, moves = svt.coordinate_refine(
            cipher, phase, initial, trigram, unigram, local_cap
        )
        prediction = [int(x) for x in svt.v0.decode_stateful(heads, phase, inv)]
        score = float(svt.candidate_score(raw, moves, period, len(heads)))
        row = {
            "start": int(k),
            "seed": int(seed),
            "raw_score": float(raw),
            "score": score,
            "local_moves": [int(x) for x in moves],
            "decoded_length": int(len(prediction)),
            "decoded_sha256": seq_sha256(prediction),
        }
        rows.append(row)
        internals.append((row, prediction))

    # Selection is exclusively by model score; plaintext truth is touched only now.
    selected_row, selected_prediction = max(internals, key=lambda x: x[0]["score"])
    public_selected = dict(selected_row)
    public_selected["edit_recovery_posthoc"] = edit_recovery(
        [int(x) for x in truth_plain], selected_prediction
    )
    public_selected["decoded_sequence"] = selected_prediction
    return {
        "mode": mode,
        "period": int(period),
        "starts": rows,
        "selected": public_selected,
    }


def divisors_ge2(period: int) -> list[int]:
    return [d for d in range(2, period + 1) if period % d == 0]


def canonicalise(
    heads: list[int],
    head_line_starts: list[int],
    seed_base: int,
    language,
    model,
    selected_fit: dict[str, Any],
    truth_plain: list[int],
) -> dict[str, Any]:
    mode = str(selected_fit["mode"])
    period = int(selected_fit["period"])
    fits: dict[int, dict[str, Any]] = {period: selected_fit}
    for d in divisors_ge2(period):
        if d == period:
            continue
        fits[d] = fit_structure(
            heads,
            head_line_starts,
            seed_base,
            language,
            model,
            mode,
            d,
            "svt-v041-canonical",
            truth_plain,
        )
    best = max(fits.values(), key=lambda x: x["selected"]["score"])
    return {
        "input_mode": mode,
        "input_period": period,
        "candidate_divisors": sorted(int(x) for x in fits),
        "fits": [fits[d] for d in sorted(fits)],
        "canonical_mode": str(best["mode"]),
        "canonical_period": int(best["period"]),
        "canonical_score": float(best["selected"]["score"]),
        "canonical_recovery": float(best["selected"]["edit_recovery_posthoc"]),
        "canonical_decoded_length": int(best["selected"]["decoded_length"]),
        "canonical_decoded_sha256": str(best["selected"]["decoded_sha256"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mode", choices=list(svt.MODES), required=True)
    ap.add_argument("--replicate", type=int, required=True)
    args = ap.parse_args()

    replicate = OFFSET + args.replicate
    language, model = load_language(
        args.repo, f"svt-v041-{args.mode}-{args.replicate}"
    )
    trial = svt.make_svt_trial(language, "dev", LENGTH, args.mode, replicate)

    # Stage 1: ciphertext-only segmentation. No language model, key, period or truth.
    fitted = seg.fit(
        trial.surface,
        trial.surface_line_starts,
        len(language.alphabet),
        int(svt.core.stable_seed("svt-v041-seg", trial.head.seed)),
    )
    predicted_starts = [int(x) for x in fitted.starts]
    start_to_unit = {s: i for i, s in enumerate(predicted_starts)}
    missing_line_starts = [
        int(s) for s in trial.surface_line_starts if int(s) not in start_to_unit
    ]
    if missing_line_starts:
        raise RuntimeError(
            "semi-Markov segmentation violated line-start invariant: "
            + repr(missing_line_starts[:10])
        )
    head_line_starts = [start_to_unit[int(s)] for s in trial.surface_line_starts]
    heads = [int(trial.surface[s]) for s in predicted_starts]

    # Stage 2a: blind structure screening on the inferred head stream.
    screen = []
    for mode in svt.MODES:
        for period in svt.CANDIDATE_PERIODS:
            screen.append({
                "mode": mode,
                "period": int(period),
                "screen_score": screen_score(
                    heads,
                    head_line_starts,
                    language,
                    model,
                    mode,
                    period,
                ),
            })
    screen.sort(key=lambda x: x["screen_score"], reverse=True)
    shortlist = screen[:SHORTLIST_K]

    # Truth rank is calculated only after the screen is frozen; it does not alter shortlist.
    truth_rank = next(
        i + 1
        for i, row in enumerate(screen)
        if row["mode"] == trial.head.mode and row["period"] == trial.head.period
    )

    # Stage 2b: qualified 12-start state/key refinement for shortlisted structures.
    refined = [
        fit_structure(
            heads,
            head_line_starts,
            int(trial.head.seed),
            language,
            model,
            str(row["mode"]),
            int(row["period"]),
            "svt-v041-refine",
            [int(x) for x in trial.head.plain],
        )
        for row in shortlist
    ]
    selected = max(refined, key=lambda x: x["selected"]["score"])

    # Stage 2c: autonomous primitive-period divisor canonicalisation, inherited from v0.3.4.
    canonical = canonicalise(
        heads,
        head_line_starts,
        int(trial.head.seed),
        language,
        model,
        selected,
        [int(x) for x in trial.head.plain],
    )

    # Stage 3: truth is now used only for metrics.
    boundary_f1 = float(seg.boundary_f1(predicted_starts, trial.head_positions))
    count_error = float(
        abs(len(predicted_starts) - len(trial.head_positions))
        / max(1, len(trial.head_positions))
    )
    canonical_exact = bool(
        canonical["canonical_mode"] == trial.head.mode
        and canonical["canonical_period"] == int(trial.head.period)
    )

    payload = {
        "programme": "SVT-v0.4.1",
        "stage": "end_to_end_hidden_segmentation_blind_state_key",
        "binding": True,
        "voynich_opened": False,
        "iso": ISO,
        "latent_length": LENGTH,
        "replicate": int(replicate),
        "true_mode": str(trial.head.mode),
        "true_period": int(trial.head.period),
        "surface_length": int(len(trial.surface)),
        "predicted_units": int(len(predicted_starts)),
        "true_units": int(len(trial.head_positions)),
        "boundary_f1": boundary_f1,
        "count_relative_error": count_error,
        "segmentation_restart": int(fitted.restart),
        "segmentation_iterations": int(fitted.iterations),
        "segmentation_score": float(fitted.score),
        "screen_truth_rank_posthoc": int(truth_rank),
        "screen_top6": shortlist,
        "selected_mode_precanonical": str(selected["mode"]),
        "selected_period_precanonical": int(selected["period"]),
        "selected_score_precanonical": float(selected["selected"]["score"]),
        "refined": refined,
        "canonical": canonical,
        "canonical_exact_posthoc": canonical_exact,
        "end_to_end_edit_recovery_posthoc": float(canonical["canonical_recovery"]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "replicate": payload["replicate"],
        "truth": [payload["true_mode"], payload["true_period"]],
        "boundary_f1": payload["boundary_f1"],
        "count_relative_error": payload["count_relative_error"],
        "screen_truth_rank": payload["screen_truth_rank_posthoc"],
        "canonical": [canonical["canonical_mode"], canonical["canonical_period"]],
        "canonical_exact": canonical_exact,
        "edit_recovery": payload["end_to_end_edit_recovery_posthoc"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
