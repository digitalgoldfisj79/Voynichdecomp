#!/usr/bin/env python3
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import svt_v02 as svt

ALPHA = 0.5
LENGTH_PRIOR = (0.30, 0.45, 0.25)
EM_ITERS = 12
N_RESTARTS = 6


@dataclass
class SegmentationFit:
    starts: list[int]
    score: float
    restart: int
    iterations: int


def _line_bounds(line_starts: list[int], n: int) -> list[tuple[int, int]]:
    starts = list(line_starts or [0])
    if not starts or starts[0] != 0:
        starts = [0] + starts
    starts = sorted(set(int(x) for x in starts if 0 <= int(x) < n))
    return list(zip(starts, starts[1:] + [n]))


def _segments_from_starts(starts: list[int], line_starts: list[int], n: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    sset = set(int(x) for x in starts)
    for left, right in _line_bounds(line_starts, n):
        local = sorted(x for x in sset if left <= x < right)
        if not local or local[0] != left:
            local = [left] + local
        for i, s in enumerate(local):
            e = local[i + 1] if i + 1 < len(local) else right
            L = e - s
            if L not in (1, 2, 3):
                raise ValueError(f"invalid unit length {L} at {s}")
            out.append((s, L))
    return out


def _estimate(surface: list[int], starts: list[int], line_starts: list[int], a: int):
    head = np.full(a, ALPHA, dtype=np.float64)
    d1 = np.full(a, ALPHA, dtype=np.float64)
    d2 = np.full(a, ALPHA, dtype=np.float64)
    for s, L in _segments_from_starts(starts, line_starts, len(surface)):
        head[int(surface[s])] += 1.0
        if L >= 2:
            d1[(int(surface[s + 1]) - int(surface[s])) % a] += 1.0
        if L >= 3:
            d2[(int(surface[s + 2]) - int(surface[s + 1])) % a] += 1.0
    return np.log(head / head.sum()), np.log(d1 / d1.sum()), np.log(d2 / d2.sum())


def _segment_score(line: list[int], pos: int, L: int, a: int, log_head, log_d1, log_d2) -> float:
    if pos + L > len(line):
        return -1e300
    score = math.log(LENGTH_PRIOR[L - 1]) + float(log_head[int(line[pos])])
    if L >= 2:
        score += float(log_d1[(int(line[pos + 1]) - int(line[pos])) % a])
    if L >= 3:
        score += float(log_d2[(int(line[pos + 2]) - int(line[pos + 1])) % a])
    return score


def _viterbi_line(line: list[int], a: int, log_head, log_d1, log_d2) -> tuple[float, list[int]]:
    m = len(line)
    dp = np.full(m + 1, -1e300, dtype=np.float64)
    back = np.full(m + 1, -1, dtype=np.int32)
    dp[0] = 0.0
    for pos in range(m):
        if dp[pos] < -1e250:
            continue
        for L in (1, 2, 3):
            end = pos + L
            if end > m:
                continue
            cand = float(dp[pos]) + _segment_score(line, pos, L, a, log_head, log_d1, log_d2)
            if cand > dp[end]:
                dp[end] = cand
                back[end] = L
    if back[m] < 0:
        raise RuntimeError("semi-Markov DP found no complete line path")
    lens: list[int] = []
    pos = m
    while pos > 0:
        L = int(back[pos])
        lens.append(L)
        pos -= L
    lens.reverse()
    starts = []
    pos = 0
    for L in lens:
        starts.append(pos)
        pos += L
    return float(dp[m]), starts


def _viterbi(surface: list[int], line_starts: list[int], a: int, model) -> tuple[float, list[int]]:
    log_head, log_d1, log_d2 = model
    total = 0.0
    starts: list[int] = []
    for left, right in _line_bounds(line_starts, len(surface)):
        score, local = _viterbi_line(surface[left:right], a, log_head, log_d1, log_d2)
        total += score
        starts.extend(left + x for x in local)
    return total, starts


def _tiling_two(line_starts: list[int], n: int) -> list[int]:
    out: list[int] = []
    for left, right in _line_bounds(line_starts, n):
        pos = left
        while pos < right:
            out.append(pos)
            rem = right - pos
            L = 2 if rem != 1 else 1
            pos += L
    return out


def _random_tiling(line_starts: list[int], n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    out: list[int] = []
    for left, right in _line_bounds(line_starts, n):
        pos = left
        while pos < right:
            out.append(pos)
            rem = right - pos
            valid = [L for L in (1, 2, 3) if L <= rem]
            weights = [LENGTH_PRIOR[L - 1] for L in valid]
            u = rng.random() * sum(weights)
            acc = 0.0
            chosen = valid[-1]
            for L, w in zip(valid, weights):
                acc += w
                if u <= acc:
                    chosen = L
                    break
            pos += chosen
    return out


def _legacy_initial(surface: list[int], line_starts: list[int], a: int) -> list[int]:
    paths = svt.v0.top_segmentations(surface, line_starts, a, beam=1)
    if not paths:
        return _tiling_two(line_starts, len(surface))
    return [int(x) for x in paths[0].starts]


def fit(surface: list[int], line_starts: list[int], a: int, seed: int) -> SegmentationFit:
    inits: list[list[int]] = [
        _legacy_initial(surface, line_starts, a),
        _tiling_two(line_starts, len(surface)),
    ]
    for r in range(2, N_RESTARTS):
        inits.append(_random_tiling(line_starts, len(surface), int(svt.core.stable_seed("svt-v04-seg-init", seed, r))))

    fits: list[SegmentationFit] = []
    for r, starts0 in enumerate(inits):
        starts = starts0
        used = 0
        score = -1e300
        for it in range(EM_ITERS):
            used = it + 1
            model = _estimate(surface, starts, line_starts, a)
            score, new_starts = _viterbi(surface, line_starts, a, model)
            if new_starts == starts:
                starts = new_starts
                break
            starts = new_starts
        # Re-estimate once on the selected path and score that same path under its fitted model.
        model = _estimate(surface, starts, line_starts, a)
        score, starts = _viterbi(surface, line_starts, a, model)
        fits.append(SegmentationFit([int(x) for x in starts], float(score), int(r), int(used)))
    return max(fits, key=lambda x: x.score)


def boundary_f1(pred: Iterable[int], truth: Iterable[int]) -> float:
    return float(svt.v0.boundary_f1(pred, truth))
