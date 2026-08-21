#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

import svt_v02 as svt
import v04_semimarkov_segmenter as seg

SURFACE_WEIGHT = 0.35
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
    combined_score: float
    surface_score: float


def _line_bounds(line_starts: list[int], n: int) -> list[tuple[int, int]]:
    starts = list(line_starts or [0])
    if not starts or starts[0] != 0:
        starts = [0] + starts
    starts = sorted(set(int(x) for x in starts if 0 <= int(x) < n))
    return list(zip(starts, starts[1:] + [n]))


def _surface_model(surface: list[int], starts: list[int], line_starts: list[int], a: int):
    return seg._estimate(surface, starts, line_starts, a)


def _surface_emit(surface: list[int], pos: int, L: int, a: int, model) -> float:
    log_head, log_d1, log_d2 = model
    score = math.log(seg.LENGTH_PRIOR[L - 1]) + float(log_head[int(surface[pos])])
    if L >= 2:
        score += float(log_d1[(int(surface[pos + 1]) - int(surface[pos])) % a])
    if L >= 3:
        score += float(log_d2[(int(surface[pos + 2]) - int(surface[pos + 1])) % a])
    return score


def surface_path_score(surface: list[int], starts: list[int], line_starts: list[int], a: int, model) -> float:
    sset = set(int(x) for x in starts)
    total = 0.0
    for left, right in _line_bounds(line_starts, len(surface)):
        local = sorted(x for x in sset if left <= x < right)
        if not local or local[0] != left:
            return -1e300
        for i, s in enumerate(local):
            e = local[i + 1] if i + 1 < len(local) else right
            L = e - s
            if L not in (1, 2, 3):
                return -1e300
            total += _surface_emit(surface, s, L, a, model)
    return float(total)


def head_from_starts(surface: list[int], starts: list[int], surface_line_starts: list[int], seed: int):
    starts = [int(x) for x in starts]
    pos_to_idx = {p: i for i, p in enumerate(starts)}
    head_lines = []
    for s in surface_line_starts or [0]:
        if int(s) not in pos_to_idx:
            raise RuntimeError(f"joint path omitted observed line start {s}")
        head_lines.append(int(pos_to_idx[int(s)]))
    return SimpleNamespace(cipher=[int(surface[s]) for s in starts], line_starts=head_lines, seed=int(seed))


def fit_key_multistart(head, language, model, mode: str, period: int, seed_tag: str, n_starts: int) -> dict:
    heads = list(head.cipher)
    cipher = np.asarray(heads, dtype=np.int32)
    trigram, unigram = model
    phase = svt.v0._phase(len(heads), period, mode, head.line_starts or [0])
    local_cap = svt.local_cap_for_alphabet(len(language.alphabet))
    rows = []
    for k in range(n_starts):
        seed = int(svt.core.stable_seed(seed_tag, head.seed, mode, period, k))
        initial = svt.initial_shared_inverse(heads, language, model, seed)
        inv, raw, moves = svt.coordinate_refine(cipher, phase, initial, trigram, unigram, local_cap)
        score = float(svt.candidate_score(raw, moves, period, len(heads)))
        rows.append({"seed": seed, "score": score, "raw": float(raw), "moves": moves, "inv": inv})
    return max(rows, key=lambda r: r["score"])


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
    """Jointly choose 1--3 glyph units and the state clock with fixed inverse maps.

    The dynamic state is (next cipher phase, previous two plaintext symbols).
    Observed line starts are hard unit boundaries. line_reset resets phase to zero;
    periodic carries phase through the line break.
    """
    n = len(surface)
    trigram, unigram = language_model
    bounds = _line_bounds(surface_line_starts, n)
    line_end = np.empty(n, dtype=np.int32)
    line_start_set = set(int(x) for x in (surface_line_starts or [0]))
    for left, right in bounds:
        line_end[left:right] = right

    # Node arrays. Root node 0 ends at position 0 and has no plaintext context.
    scores = [0.0]
    phases = [0]
    prev2s = [-1]
    prev1s = [-1]
    parents = [-1]
    starts_used = [-1]
    lens_used = [0]
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
                    lm = 0.15 * float(unigram[x])
                    npp, np1 = -1, x
                elif pp < 0:
                    lm = 0.15 * float(unigram[x])
                    npp, np1 = p1, x
                else:
                    lm = float(trigram[pp, p1, x]) + 0.15 * float(unigram[x])
                    npp, np1 = p1, x
                se = _surface_emit(surface, pos, L, a, surface_model)
                cand = float(scores[idx] + lm + SURFACE_WEIGHT * se)
                next_phase = (phase + 1) % period
                if mode == "line_reset" and end in line_start_set:
                    next_phase = 0
                key = (int(next_phase), int(npp), int(np1))
                old = buckets[end].get(key)
                if old is None or cand > scores[old]:
                    nid = len(scores)
                    scores.append(cand)
                    phases.append(int(next_phase))
                    prev2s.append(int(npp))
                    prev1s.append(int(np1))
                    parents.append(int(idx))
                    starts_used.append(int(pos))
                    lens_used.append(int(L))
                    plains.append(int(x))
                    buckets[end][key] = nid

    if not buckets[n]:
        raise RuntimeError("joint semi-Markov search found no complete path")
    best = max(buckets[n].values(), key=lambda i: scores[i])
    out_starts = []
    out_plain = []
    cur = best
    while parents[cur] >= 0:
        out_starts.append(int(starts_used[cur]))
        out_plain.append(int(plains[cur]))
        cur = int(parents[cur])
    out_starts.reverse()
    out_plain.reverse()
    ss = surface_path_score(surface, out_starts, surface_line_starts, a, surface_model)
    return JointPath(out_starts, out_plain, float(scores[best]), float(ss))


def fit_candidate(
    surface: list[int],
    surface_line_starts: list[int],
    initial_starts: list[int],
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

    head = head_from_starts(surface, starts, surface_line_starts, seed)
    keyfit = fit_key_multistart(head, language, language_model, mode, period, "svt-v042-key-init", n_key_starts)

    for alt in range(alternations):
        path = joint_viterbi(surface, surface_line_starts, a, language_model, surface_model, keyfit["inv"], mode, period, beam)
        starts = path.starts
        head = head_from_starts(surface, starts, surface_line_starts, seed)
        # Truth-free warm re-fit after each boundary update. Full multistart is paid once
        # at entry; subsequent alternations use one fresh deterministic start.
        keyfit = fit_key_multistart(head, language, language_model, mode, period, f"svt-v042-key-alt-{alt}", 1)

    phase = svt.v0._phase(len(head.cipher), period, mode, head.line_starts or [0])
    prediction = [int(x) for x in svt.v0.decode_stateful(head.cipher, phase, keyfit["inv"])]
    ss = surface_path_score(surface, starts, surface_line_starts, a, surface_model)
    final_score = float(keyfit["score"] + SURFACE_WEIGHT * ss)
    return {
        "mode": mode,
        "period": int(period),
        "score": final_score,
        "key_score": float(keyfit["score"]),
        "raw": float(keyfit["raw"]),
        "local_moves": [int(x) for x in keyfit["moves"]],
        "surface_score": float(ss),
        "starts": [int(x) for x in starts],
        "prediction": prediction,
    }


def solve(surface, surface_line_starts, language, language_model, seed: int) -> dict:
    a = len(language.alphabet)
    s0 = seg.fit(surface, surface_line_starts, a, int(svt.core.stable_seed("svt-v042-s0", seed)))
    initial_starts = [int(x) for x in s0.starts]
    smodel = _surface_model(surface, initial_starts, surface_line_starts, a)

    cheap = []
    for mode in svt.MODES:
        for period in svt.CANDIDATE_PERIODS:
            cheap.append(fit_candidate(surface, surface_line_starts, initial_starts, language, language_model, smodel, seed, mode, int(period), False))
    cheap.sort(key=lambda r: r["score"], reverse=True)
    shortlist_spec = [(r["mode"], int(r["period"])) for r in cheap[:SHORTLIST_K]]

    full = [fit_candidate(surface, surface_line_starts, initial_starts, language, language_model, smodel, seed, m, p, True) for m, p in shortlist_spec]
    selected = max(full, key=lambda r: r["score"])

    # Primitive-period canonicalisation under the same joint model.
    canonical_fits = [selected]
    for d in range(2, int(selected["period"])):
        if int(selected["period"]) % d == 0:
            canonical_fits.append(fit_candidate(surface, surface_line_starts, selected["starts"], language, language_model, smodel, seed, selected["mode"], d, True))
    canonical = max(canonical_fits, key=lambda r: r["score"])

    return {
        "initial_starts": initial_starts,
        "cheap_ranking": [{"mode": r["mode"], "period": r["period"], "score": r["score"]} for r in cheap],
        "shortlist": [{"mode": m, "period": p} for m, p in shortlist_spec],
        "selected_precanonical": selected,
        "canonical": canonical,
    }
