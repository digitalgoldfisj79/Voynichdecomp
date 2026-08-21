#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import svt_v02 as svt
import v04_semimarkov_segmenter as seg
import joint_semimarkov_v042 as v042

CHEAP_BEAM = 160
FULL_BEAM = 320
CHEAP_KEY_STARTS = 1
FULL_KEY_STARTS = 12
FULL_ALTERNATIONS = 3
SHORTLIST_K = 6


@dataclass
class JointPath:
    starts: list[int]
    plaintext: list[int]
    comparable_score: float
    surface_score: float
    language_gain: float


def contextual_language_gain(prediction: list[int], language_model) -> float:
    """Log BF of trigram context against the same-length unigram baseline."""
    trigram, unigram = language_model
    total = 0.0
    for i in range(2, len(prediction)):
        a, b, x = int(prediction[i - 2]), int(prediction[i - 1]), int(prediction[i])
        total += float(trigram[a, b, x] - unigram[x])
    return float(total)


def complexity_penalty(local_moves, period: int, n: int) -> float:
    return float(
        svt.LOCAL_BIC_WEIGHT * int(np.sum(local_moves)) * math.log(max(2, n))
        + svt.SCHEDULE_BIC_WEIGHT * max(0, int(period) - 1) * math.log(max(2, n))
    )


def comparable_selection_score(prediction, language_model, surface_score: float, local_moves, period: int) -> float:
    n = max(1, len(prediction))
    return float(
        contextual_language_gain(list(prediction), language_model)
        + float(surface_score)
        - complexity_penalty(local_moves, int(period), n)
    )


def joint_viterbi(
    surface: list[int],
    surface_line_starts: list[int],
    a: int,
    language_model,
    surface_model,
    inv: np.ndarray,
    mode: str,
    period: int,
    beam: int,
) -> JointPath:
    """Joint segmentation/state search using contextual log-BF, not raw LM LL."""
    n = len(surface)
    trigram, unigram = language_model
    bounds = v042._line_bounds(surface_line_starts, n)
    line_end = np.empty(n, dtype=np.int32)
    line_start_set = set(int(x) for x in (surface_line_starts or [0]))
    for left, right in bounds:
        line_end[left:right] = right

    scores = [0.0]
    surface_scores = [0.0]
    language_gains = [0.0]
    phases = [0]
    prev2s = [-1]
    prev1s = [-1]
    parents = [-1]
    starts_used = [-1]
    plains = [-1]
    buckets: list[dict[tuple[int, int, int], int]] = [dict() for _ in range(n + 1)]
    buckets[0][(0, -1, -1)] = 0

    for pos in range(n):
        bucket = buckets[pos]
        if not bucket:
            continue
        if len(bucket) > beam:
            keep = sorted(bucket.values(), key=lambda i: scores[i], reverse=True)[:beam]
            bucket = {(phases[i], prev2s[i], prev1s[i]): i for i in keep}
            buckets[pos] = bucket
        max_end = int(line_end[pos])
        for idx in list(bucket.values()):
            phase = int(phases[idx])
            pp = int(prev2s[idx])
            p1 = int(prev1s[idx])
            if mode == "line_reset" and pos in line_start_set:
                phase = 0
            for L in (1, 2, 3):
                end = pos + L
                if end > max_end:
                    continue
                x = int(inv[phase, int(surface[pos])])
                if p1 < 0:
                    gain = 0.0
                    npp, np1 = -1, x
                elif pp < 0:
                    gain = 0.0
                    npp, np1 = p1, x
                else:
                    gain = float(trigram[pp, p1, x] - unigram[x])
                    npp, np1 = p1, x
                se = float(v042._surface_emit(surface, pos, L, a, surface_model))
                cand = float(scores[idx] + gain + se)
                next_phase = (phase + 1) % period
                if mode == "line_reset" and end in line_start_set:
                    next_phase = 0
                key = (int(next_phase), int(npp), int(np1))
                old = buckets[end].get(key)
                if old is None or cand > scores[old]:
                    nid = len(scores)
                    scores.append(cand)
                    surface_scores.append(float(surface_scores[idx] + se))
                    language_gains.append(float(language_gains[idx] + gain))
                    phases.append(int(next_phase))
                    prev2s.append(int(npp))
                    prev1s.append(int(np1))
                    parents.append(int(idx))
                    starts_used.append(int(pos))
                    plains.append(int(x))
                    buckets[end][key] = nid

    if not buckets[n]:
        raise RuntimeError("joint semi-Markov search found no complete path")
    best = max(buckets[n].values(), key=lambda i: scores[i])
    starts: list[int] = []
    plaintext: list[int] = []
    cur = best
    while parents[cur] >= 0:
        starts.append(int(starts_used[cur]))
        plaintext.append(int(plains[cur]))
        cur = int(parents[cur])
    starts.reverse()
    plaintext.reverse()
    ss = v042.surface_path_score(surface, starts, surface_line_starts, a, surface_model)
    gain = contextual_language_gain(plaintext, language_model)
    return JointPath(starts, plaintext, float(ss + gain), float(ss), float(gain))


def fit_candidate(
    surface,
    surface_line_starts,
    initial_starts,
    language,
    language_model,
    surface_model,
    seed: int,
    mode: str,
    period: int,
    full: bool,
) -> dict:
    a = len(language.alphabet)
    starts = list(initial_starts)
    n_key_starts = FULL_KEY_STARTS if full else CHEAP_KEY_STARTS
    alternations = FULL_ALTERNATIONS if full else 1
    beam = FULL_BEAM if full else CHEAP_BEAM

    head = v042.head_from_starts(surface, starts, surface_line_starts, seed)
    keyfit = v042.fit_key_multistart(head, language, language_model, mode, period, "svt-v043-key-init", n_key_starts)

    for alt in range(alternations):
        path = joint_viterbi(surface, surface_line_starts, a, language_model, surface_model, keyfit["inv"], mode, period, beam)
        starts = path.starts
        head = v042.head_from_starts(surface, starts, surface_line_starts, seed)
        keyfit = v042.fit_key_multistart(head, language, language_model, mode, period, f"svt-v043-key-alt-{alt}", 1)

    phase = svt.v0._phase(len(head.cipher), period, mode, head.line_starts or [0])
    prediction = [int(x) for x in svt.v0.decode_stateful(head.cipher, phase, keyfit["inv"])]
    ss = v042.surface_path_score(surface, starts, surface_line_starts, a, surface_model)
    gain = contextual_language_gain(prediction, language_model)
    score = comparable_selection_score(prediction, language_model, ss, keyfit["moves"], period)
    return {
        "mode": mode,
        "period": int(period),
        "score": float(score),
        "key_raw_score": float(keyfit["raw"]),
        "language_gain": float(gain),
        "surface_score": float(ss),
        "complexity_penalty": float(complexity_penalty(keyfit["moves"], period, len(prediction))),
        "local_moves": [int(x) for x in keyfit["moves"]],
        "starts": [int(x) for x in starts],
        "prediction": prediction,
    }


def solve(surface, surface_line_starts, language, language_model, seed: int) -> dict:
    a = len(language.alphabet)
    s0 = seg.fit(surface, surface_line_starts, a, int(svt.core.stable_seed("svt-v043-s0", seed)))
    initial_starts = [int(x) for x in s0.starts]
    smodel = v042._surface_model(surface, initial_starts, surface_line_starts, a)

    cheap = []
    for mode in svt.MODES:
        for period in svt.CANDIDATE_PERIODS:
            cheap.append(fit_candidate(surface, surface_line_starts, initial_starts, language, language_model, smodel, seed, mode, int(period), False))
    cheap.sort(key=lambda r: r["score"], reverse=True)
    shortlist_spec = [(r["mode"], int(r["period"])) for r in cheap[:SHORTLIST_K]]

    full = [fit_candidate(surface, surface_line_starts, initial_starts, language, language_model, smodel, seed, m, p, True) for m, p in shortlist_spec]
    selected = max(full, key=lambda r: r["score"])

    canonical_fits = [selected]
    for d in range(2, int(selected["period"])):
        if int(selected["period"]) % d == 0:
            canonical_fits.append(fit_candidate(surface, surface_line_starts, selected["starts"], language, language_model, smodel, seed, selected["mode"], d, True))
    canonical = max(canonical_fits, key=lambda r: r["score"])

    return {
        "initial_starts": initial_starts,
        "initial_units": int(len(initial_starts)),
        "cheap_ranking": [{"mode": r["mode"], "period": r["period"], "score": r["score"], "units": len(r["starts"])} for r in cheap],
        "shortlist": [{"mode": m, "period": p} for m, p in shortlist_spec],
        "selected_precanonical": selected,
        "canonical": canonical,
    }
