#!/usr/bin/env python3
from __future__ import annotations

import bisect
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import svt_v02 as svt
import run_v034_primitive_period as v034
import v04_semimarkov_segmenter as seg
import latin_proiel_portability as lat

LATTICE_K = 8
N_NULLS = 8
Z_THRESHOLD = 2.0
SHORTLIST_K = 6
N_STARTS = 12

@dataclass
class LatticePath:
    starts: list[int]
    head_line_starts: list[int]
    surface_score: float
    surface_rank: int = -1

def load_language(repo: Path, iso: str, cache_name: str):
    if iso == "de":
        return v034.load_language(repo, cache_name)
    if iso == "la":
        return lat.load_latin(repo / ".cache" / cache_name)
    raise ValueError(f"unsupported iso {iso}")

def _line_kbest_fixed(line, a, model, k):
    log_head, log_d1, log_d2 = model
    m = len(line)
    dp = [[] for _ in range(m + 1)]
    dp[0] = [(0.0, [])]
    for pos in range(m):
        if not dp[pos]:
            continue
        for score, starts in dp[pos]:
            for L in (1, 2, 3):
                end = pos + L
                if end > m:
                    continue
                cand = score + seg._segment_score(line, pos, L, a, log_head, log_d1, log_d2)
                dp[end].append((float(cand), starts + [pos]))
        for end in range(pos + 1, min(m, pos + 3) + 1):
            if len(dp[end]) > k:
                dp[end].sort(key=lambda z: z[0], reverse=True)
                del dp[end][k:]
    out = dp[m]
    out.sort(key=lambda z: z[0], reverse=True)
    return out[:k]

def surface_lattice(surface, line_starts, a, seed, k=LATTICE_K):
    fitted = seg.fit(surface, line_starts, a, seed)
    model = seg._estimate(surface, fitted.starts, line_starts, a)
    combined = [(0.0, [], [])]
    for left, right in seg._line_bounds(line_starts, len(surface)):
        local = _line_kbest_fixed(surface[left:right], a, model, k)
        new = []
        for total, gstarts, hlines in combined:
            for ls, lstarts in local:
                new.append((total + ls,
                            gstarts + [left + x for x in lstarts],
                            hlines + [len(gstarts)]))
        new.sort(key=lambda z: z[0], reverse=True)
        combined = new[:k]
    paths = [LatticePath([int(x) for x in s], [int(x) for x in hl], float(sc), i)
             for i, (sc, s, hl) in enumerate(combined)]
    if not paths:
        raise RuntimeError("surface lattice empty")
    return fitted, paths

def _screen_rows(heads, head_line_starts, seed, language, model):
    head = SimpleNamespace(cipher=list(heads), line_starts=list(head_line_starts), seed=int(seed))
    rows = []
    for mode in svt.MODES:
        for period in svt.CANDIDATE_PERIODS:
            rows.append({
                "mode": mode,
                "period": int(period),
                "score": float(v034.screen_score(head, language, model, mode, int(period))),
            })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows

def _shuffle_within_lines(heads, head_line_starts, seed):
    rng = random.Random(int(seed))
    starts = list(head_line_starts or [0])
    if not starts or starts[0] != 0:
        starts = [0] + starts
    starts = sorted(set(int(x) for x in starts if 0 <= int(x) < len(heads)))
    ends = starts[1:] + [len(heads)]
    out = []
    for left, right in zip(starts, ends):
        block = list(heads[left:right])
        rng.shuffle(block)
        out.extend(block)
    return out

def matched_null_evidence(heads, head_line_starts, seed, language, model):
    actual_rows = _screen_rows(heads, head_line_starts, seed, language, model)
    actual = float(actual_rows[0]["score"])
    null_scores = []
    for j in range(N_NULLS):
        shuffled = _shuffle_within_lines(
            heads, head_line_starts,
            svt.core.stable_seed("svt-v051-null", seed, j)
        )
        rows = _screen_rows(shuffled, head_line_starts,
                            svt.core.stable_seed("svt-v051-null-screen", seed, j),
                            language, model)
        null_scores.append(float(rows[0]["score"]))
    mu = float(statistics.mean(null_scores))
    sd = float(statistics.stdev(null_scores)) if len(null_scores) > 1 else 0.0
    z = float((actual - mu) / max(sd, 1e-9))
    exceed = sum(x >= actual for x in null_scores)
    p = float((1 + exceed) / (1 + len(null_scores)))
    return {
        "actual_best_screen": actual,
        "actual_best_mode": actual_rows[0]["mode"],
        "actual_best_period": int(actual_rows[0]["period"]),
        "null_mean": mu,
        "null_sd": sd,
        "null_scores": null_scores,
        "z": z,
        "randomization_p": p,
    }

def choose_lattice_path(surface, surface_line_starts, a, seed, language, model):
    fitted, paths = surface_lattice(surface, surface_line_starts, a, seed, LATTICE_K)
    rows = []
    for p in paths:
        heads = [int(surface[s]) for s in p.starts]
        ev = matched_null_evidence(
            heads, p.head_line_starts,
            svt.core.stable_seed("svt-v051-evidence", seed, p.surface_rank),
            language, model
        )
        rows.append({
            "path": p,
            "heads": heads,
            "evidence": ev,
        })
    best_z = max(rows, key=lambda r: r["evidence"]["z"])
    selected = best_z if best_z["evidence"]["z"] >= Z_THRESHOLD else rows[0]
    return fitted, rows, selected

def _fit_structure_blind(heads, head_line_starts, seed, language, model, mode, period, tag):
    cipher = np.asarray(heads, dtype=np.int32)
    trigram, unigram = model
    phase = svt.v0._phase(len(heads), period, mode, head_line_starts or [0])
    local_cap = svt.local_cap_for_alphabet(len(language.alphabet))
    starts = []
    for k in range(N_STARTS):
        s = int(svt.core.stable_seed(tag, seed, mode, period, k))
        initial = svt.initial_shared_inverse(heads, language, model, s)
        inv, raw, moves = svt.coordinate_refine(cipher, phase, initial, trigram, unigram, local_cap)
        prediction = svt.v0.decode_stateful(heads, phase, inv)
        score = float(svt.candidate_score(raw, moves, period, len(heads)))
        starts.append({
            "start": int(k), "seed": s, "raw_score": float(raw), "score": score,
            "local_moves": [int(x) for x in moves],
            "prediction": [int(x) for x in prediction],
        })
    selected = max(starts, key=lambda x: x["score"])
    return {"mode": mode, "period": int(period), "starts": starts, "selected": selected}

def solve_selected_path(heads, head_line_starts, seed, language, model):
    screen = _screen_rows(heads, head_line_starts, seed, language, model)
    shortlist = screen[:SHORTLIST_K]
    fits = [_fit_structure_blind(
        heads, head_line_starts, seed, language, model,
        row["mode"], int(row["period"]), "svt-v051-refine"
    ) for row in shortlist]
    selected = max(fits, key=lambda r: r["selected"]["score"])
    mode, period = selected["mode"], int(selected["period"])
    candidates = {period: selected}
    for d in v034.divisors_ge2(period):
        if d == period:
            continue
        candidates[d] = _fit_structure_blind(
            heads, head_line_starts, seed, language, model,
            mode, int(d), "svt-v051-canonical"
        )
    canonical = max(candidates.values(), key=lambda r: r["selected"]["score"])
    return {
        "screen_top6": shortlist,
        "mode": canonical["mode"],
        "period": int(canonical["period"]),
        "score": float(canonical["selected"]["score"]),
        "prediction": canonical["selected"]["prediction"],
        "candidate_divisors": sorted(int(x) for x in candidates),
    }

def sequence_recovery(truth, pred):
    dist = int(svt.v0.levenshtein_distance(list(truth), list(pred)))
    return float(1.0 - dist / max(1, len(truth), len(pred)))

def run_trial(repo: Path, iso: str, split: str, mode: str, replicate: int):
    language, model = load_language(repo, iso, f"svt-v051-{split}-{iso}-{mode}-{replicate}")
    trial = svt.make_svt_trial(language, split, 1536, mode, replicate)
    seed = int(svt.core.stable_seed("svt-v051-lattice", iso, split, mode, replicate))
    fitted, rows, chosen = choose_lattice_path(
        trial.surface, trial.surface_line_starts, len(language.alphabet), seed, language, model
    )
    path = chosen["path"]
    heads = chosen["heads"]
    solved = solve_selected_path(
        heads, path.head_line_starts,
        int(svt.core.stable_seed("svt-v051-full", seed, path.surface_rank)),
        language, model
    )

    truth_starts = trial.head_positions
    map_path = rows[0]["path"]
    selected_f1 = float(seg.boundary_f1(path.starts, truth_starts))
    map_f1 = float(seg.boundary_f1(map_path.starts, truth_starts))
    oracle_f1 = max(float(seg.boundary_f1(r["path"].starts, truth_starts)) for r in rows)
    true_n = len(truth_starts)
    selected_n = len(path.starts)
    map_n = len(map_path.starts)
    count_signed = float((selected_n - true_n) / max(1, true_n))
    map_shift = float((selected_n - map_n) / max(1, map_n))
    recovery = sequence_recovery(trial.head.plain, solved["prediction"])

    screen_truth_rank = next(
        (i + 1 for i, row in enumerate(solved["screen_top6"])
         if row["mode"] == trial.head.mode and int(row["period"]) == int(trial.head.period)),
        None
    )
    lattice_rows = []
    for r in rows:
        p = r["path"]
        lattice_rows.append({
            "surface_rank": int(p.surface_rank),
            "surface_score": float(p.surface_score),
            "units": int(len(p.starts)),
            "boundary_f1_eval_only": float(seg.boundary_f1(p.starts, truth_starts)),
            "cipher_z": float(r["evidence"]["z"]),
            "cipher_p": float(r["evidence"]["randomization_p"]),
            "screen_mode": r["evidence"]["actual_best_mode"],
            "screen_period": int(r["evidence"]["actual_best_period"]),
        })
    return {
        "programme": "SVT-v0.5.1-lattice-coupling",
        "voynich_opened": False,
        "iso": iso, "split": split, "generator_mode": mode, "replicate": int(replicate),
        "true_mode_eval_only": trial.head.mode, "true_period_eval_only": int(trial.head.period),
        "surface_map_units": int(map_n), "surface_map_boundary_f1_eval_only": map_f1,
        "lattice_oracle_boundary_f1_eval_only": oracle_f1,
        "selected_surface_rank": int(path.surface_rank),
        "selected_cipher_z": float(chosen["evidence"]["z"]),
        "selected_cipher_p": float(chosen["evidence"]["randomization_p"]),
        "selected_units": int(selected_n),
        "true_units_eval_only": int(true_n),
        "count_signed_error_eval_only": count_signed,
        "count_abs_error_eval_only": abs(count_signed),
        "selected_vs_surface_map_count_shift": map_shift,
        "boundary_f1_eval_only": selected_f1,
        "solved_mode": solved["mode"], "solved_period": int(solved["period"]),
        "exact_structure_eval_only": bool(
            solved["mode"] == trial.head.mode and int(solved["period"]) == int(trial.head.period)
        ),
        "sequence_recovery_eval_only": float(recovery),
        "screen_truth_rank_within_top6_eval_only": screen_truth_rank,
        "lattice": lattice_rows,
    }
