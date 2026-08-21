#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import svt_v02 as svt

ISO = "de"
LENGTH = 1536
ORDINARY_OFFSET = 13000
HARMONIC_OFFSET = 15000
N_STARTS = 12
SHORTLIST_K = 6


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
    starts = []
    for k in range(N_STARTS):
        seed = int(svt.core.stable_seed(seed_tag, head.seed, mode, period, k))
        initial = svt.initial_shared_inverse(heads, language, model, seed)
        inv, raw, moves = svt.coordinate_refine(cipher, phase, initial, trigram, unigram, local_cap)
        prediction = svt.v0.decode_stateful(heads, phase, inv)
        recovery = float(svt.mono.fast_accuracy(head.plain, prediction))
        score = float(svt.candidate_score(raw, moves, period, len(heads)))
        starts.append({
            "start": int(k),
            "seed": int(seed),
            "raw_score": float(raw),
            "score": score,
            "recovery": recovery,
            "local_moves": [int(x) for x in moves],
        })
    selected = max(starts, key=lambda x: x["score"])
    return {
        "mode": mode,
        "period": int(period),
        "starts": starts,
        "selected": selected,
    }


def divisors_ge2(period: int) -> list[int]:
    return [d for d in range(2, period + 1) if period % d == 0]


def canonicalise(head, language, model, selected_fit: dict, seed_tag: str) -> dict:
    mode = selected_fit["mode"]
    period = int(selected_fit["period"])
    fits = {period: selected_fit}
    for d in divisors_ge2(period):
        if d == period:
            continue
        fits[d] = fit_structure(head, language, model, mode, d, seed_tag)
    best = max(fits.values(), key=lambda row: row["selected"]["score"])
    return {
        "input_mode": mode,
        "input_period": period,
        "candidate_divisors": sorted(int(x) for x in fits),
        "fits": [fits[d] for d in sorted(fits)],
        "canonical_mode": best["mode"],
        "canonical_period": int(best["period"]),
        "canonical_score": float(best["selected"]["score"]),
        "canonical_recovery": float(best["selected"]["recovery"]),
    }


def load_language(repo: Path, cache_name: str):
    root = repo / "experiments" / "recoverability_frontier_v0_5"
    cache = repo / ".cache" / cache_name
    languages = svt.core.load_languages(root / "corpus_manifest_v050.json", cache)
    language = languages[ISO]
    model = svt.mono.build_language_model(language)
    return language, model


def run_ordinary(repo: Path, mode: str, rep: int) -> dict:
    language, model = load_language(repo, f"svt-v034-ordinary-{mode}-{rep}")
    replicate = ORDINARY_OFFSET + rep
    trial = svt.make_svt_trial(language, "dev", LENGTH, mode, replicate)
    head = trial.head

    screen = []
    for candidate_mode in svt.MODES:
        for period in svt.CANDIDATE_PERIODS:
            screen.append({
                "mode": candidate_mode,
                "period": int(period),
                "screen_score": screen_score(head, language, model, candidate_mode, period),
            })
    screen.sort(key=lambda x: x["screen_score"], reverse=True)
    shortlist = screen[:SHORTLIST_K]
    truth_rank = next(
        i + 1 for i, row in enumerate(screen)
        if row["mode"] == head.mode and row["period"] == head.period
    )

    refined = [
        fit_structure(head, language, model, row["mode"], int(row["period"]), "svt-v034-blind")
        for row in shortlist
    ]
    selected = max(refined, key=lambda row: row["selected"]["score"])
    canonical = canonicalise(head, language, model, selected, "svt-v034-canonical")

    return {
        "programme": "SVT-v0.3.4",
        "arm": "ordinary",
        "binding": True,
        "voynich_opened": False,
        "iso": ISO,
        "length": LENGTH,
        "replicate": int(replicate),
        "true_mode": head.mode,
        "true_period": int(head.period),
        "screen_truth_rank": int(truth_rank),
        "screen_top6": shortlist,
        "selected_mode_precanonical": selected["mode"],
        "selected_period_precanonical": int(selected["period"]),
        "selected_recovery_precanonical": float(selected["selected"]["recovery"]),
        "canonical": canonical,
        "canonical_exact": bool(
            canonical["canonical_mode"] == head.mode
            and canonical["canonical_period"] == int(head.period)
        ),
    }


def make_targeted_head(language, mode: str, primitive: int, challenge_index: int):
    replicate = HARMONIC_OFFSET + challenge_index
    chunks = svt.core.source_chunks(language, "dev", LENGTH)
    if not chunks:
        raise RuntimeError("no German dev chunks for harmonic challenge")
    plain = list(chunks[replicate % len(chunks)])
    seed = int(svt.core.stable_seed("svt-v034-harmonic-head", ISO, mode, primitive, replicate))
    rng = random.Random(seed)
    line_rng = random.Random(svt.core.stable_seed("svt-v034-harmonic-lines", seed))
    line_starts = svt.v0.pbase.make_line_starts(line_rng, LENGTH)
    phase = svt.v0._phase(LENGTH, primitive, mode, line_starts)
    maps = svt.v0._fresh_maps(rng, len(language.alphabet), primitive)
    cipher = [maps[int(phase[i])][int(x)] for i, x in enumerate(plain)]
    return SimpleNamespace(
        iso=ISO,
        split="dev",
        length=LENGTH,
        mode=mode,
        replicate=replicate,
        seed=seed,
        plain=plain,
        cipher=cipher,
        period=int(primitive),
        line_starts=line_starts,
        forward_maps=maps,
    )


def run_harmonic(repo: Path, mode: str, primitive: int, superperiod: int, challenge_index: int) -> dict:
    if superperiod <= primitive or superperiod % primitive != 0:
        raise ValueError("superperiod must be a strict integer multiple of primitive")
    language, model = load_language(repo, f"svt-v034-harmonic-{challenge_index}")
    head = make_targeted_head(language, mode, primitive, challenge_index)
    super_fit = fit_structure(head, language, model, mode, superperiod, "svt-v034-harmonic-super")
    canonical = canonicalise(head, language, model, super_fit, "svt-v034-harmonic-canonical")
    return {
        "programme": "SVT-v0.3.4",
        "arm": "harmonic",
        "binding": True,
        "voynich_opened": False,
        "iso": ISO,
        "length": LENGTH,
        "challenge_index": int(challenge_index),
        "replicate": int(head.replicate),
        "true_mode": mode,
        "true_primitive_period": int(primitive),
        "forced_superperiod": int(superperiod),
        "superperiod_recovery": float(super_fit["selected"]["recovery"]),
        "canonical": canonical,
        "canonical_exact": bool(
            canonical["canonical_mode"] == mode
            and canonical["canonical_period"] == int(primitive)
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--arm", choices=("ordinary", "harmonic"), required=True)
    ap.add_argument("--mode", choices=list(svt.MODES), required=True)
    ap.add_argument("--replicate", type=int)
    ap.add_argument("--primitive", type=int)
    ap.add_argument("--superperiod", type=int)
    ap.add_argument("--challenge-index", type=int)
    args = ap.parse_args()

    if args.arm == "ordinary":
        if args.replicate is None:
            raise SystemExit("ordinary arm requires --replicate")
        payload = run_ordinary(args.repo, args.mode, args.replicate)
    else:
        if args.primitive is None or args.superperiod is None or args.challenge_index is None:
            raise SystemExit("harmonic arm requires --primitive --superperiod --challenge-index")
        payload = run_harmonic(args.repo, args.mode, args.primitive, args.superperiod, args.challenge_index)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.arm == "ordinary":
        summary = {
            "arm": "ordinary",
            "replicate": payload["replicate"],
            "truth": [payload["true_mode"], payload["true_period"]],
            "screen_truth_rank": payload["screen_truth_rank"],
            "precanonical": [payload["selected_mode_precanonical"], payload["selected_period_precanonical"]],
            "canonical": [payload["canonical"]["canonical_mode"], payload["canonical"]["canonical_period"]],
            "recovery": payload["canonical"]["canonical_recovery"],
            "exact": payload["canonical_exact"],
        }
    else:
        summary = {
            "arm": "harmonic",
            "challenge_index": payload["challenge_index"],
            "truth": [payload["true_mode"], payload["true_primitive_period"]],
            "forced_superperiod": payload["forced_superperiod"],
            "canonical_period": payload["canonical"]["canonical_period"],
            "recovery": payload["canonical"]["canonical_recovery"],
            "exact": payload["canonical_exact"],
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
