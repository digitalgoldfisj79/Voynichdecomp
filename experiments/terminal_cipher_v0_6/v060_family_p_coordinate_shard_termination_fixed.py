#!/usr/bin/env python3
"""Execution-only termination-safe shard runner for frozen v0.6 Family P.

The scientific search is unchanged. The only correction is finite handling of
an impossible uniqueness request in period-2 histogram seed generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V05 = HERE.parent / "recoverability_frontier_v0_5"
if str(V05) not in sys.path:
    sys.path.insert(0, str(V05))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import recoverability_v050 as core
import mono_solver_v051 as mono
import v060_family_p_mode_blind as blind
import v060_family_p_coordinate_final as frozen
from v060_family_p_coordinate_shard import FROZEN_CONFIG


def finite_phase_histogram_seeds(
    cipher: list[int],
    phase: np.ndarray,
    period: int,
    alphabet_size: int,
    count: int,
    seed: int,
) -> list[np.ndarray]:
    histograms = np.zeros((period, alphabet_size), dtype=np.float64)
    for index, symbol in enumerate(cipher):
        histograms[int(phase[index]), int(symbol)] += 1.0
    for row in histograms:
        total = row.sum()
        if total > 0:
            row /= total
    reference = histograms[0]
    ranked: list[list[int]] = [[0]]
    for slot in range(1, period):
        scores = []
        for shift in range(alphabet_size):
            aligned = np.roll(histograms[slot], -shift)
            scores.append((float(np.dot(reference, aligned)), shift))
        scores.sort(reverse=True)
        ranked.append([shift for _score, shift in scores[: min(5, alphabet_size)]])

    seeds: list[np.ndarray] = []
    best = np.zeros(period, dtype=np.int32)
    for slot in range(1, period):
        best[slot] = ranked[slot][0]
    seeds.append(best.copy())
    for slot in range(1, period):
        for alternative in ranked[slot][1:3]:
            candidate = best.copy()
            candidate[slot] = alternative
            if not any(np.array_equal(candidate, existing) for existing in seeds):
                seeds.append(candidate)
            if len(seeds) >= count:
                return seeds

    distinct_capacity = math.prod(len(ranked[slot]) for slot in range(1, period))
    unique_target = min(count, distinct_capacity)
    rng = random.Random(seed)
    while len(seeds) < unique_target:
        candidate = np.zeros(period, dtype=np.int32)
        for slot in range(1, period):
            candidate[slot] = rng.choice(ranked[slot])
        if not any(np.array_equal(candidate, existing) for existing in seeds):
            seeds.append(candidate)

    # If the finite unique space is smaller than the registered start count,
    # sample with replacement. Downstream seed_index keeps trajectories distinct.
    while len(seeds) < count:
        candidate = np.zeros(period, dtype=np.int32)
        for slot in range(1, period):
            candidate[slot] = rng.choice(ranked[slot])
        seeds.append(candidate)
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iso", default="en")
    parser.add_argument("--split", choices=("dev",), default="dev")
    parser.add_argument("--length", type=int, default=384)
    parser.add_argument("--mode", choices=tuple(blind.MODES), required=True)
    parser.add_argument("--replicate", type=int, choices=range(8), required=True)
    args = parser.parse_args()

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        root / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v060-family-p2",
    )
    language = languages[args.iso]
    model = mono.build_language_model(language)
    trial = blind.make_trial(language, args.split, args.length, args.mode, args.replicate)

    # Preflight every frozen structure before starting expensive work.
    preflight = []
    for candidate_mode in blind.MODES:
        for period in blind.CANDIDATE_PERIODS:
            phase = frozen.base.phase_array(
                trial.length, period, candidate_mode, trial.line_starts
            )
            seeds = finite_phase_histogram_seeds(
                trial.cipher,
                phase,
                period,
                len(language.alphabet),
                FROZEN_CONFIG["seed_count"],
                core.stable_seed(
                    "v060-p2-seeds", trial.seed, candidate_mode, period
                ),
            )
            if len(seeds) != FROZEN_CONFIG["seed_count"]:
                raise RuntimeError("seed preflight did not satisfy frozen count")
            unique = len({tuple(int(x) for x in item) for item in seeds})
            preflight.append({
                "mode": candidate_mode,
                "period": period,
                "starts": len(seeds),
                "unique_starts": unique,
            })
    print("V060_P2_SEED_PREFLIGHT", json.dumps(preflight, sort_keys=True), flush=True)

    frozen.phase_histogram_seeds = finite_phase_histogram_seeds
    original_coordinate_run = frozen.coordinate_run
    counters = {"screen": 0, "refine": 0}

    def instrumented_coordinate_run(*positional, **keywords):
        result = original_coordinate_run(*positional, **keywords)
        label = str(keywords.get("label", positional[12] if len(positional) > 12 else "unknown"))
        if label in counters:
            counters[label] += 1
            if label == "refine" or counters[label] % FROZEN_CONFIG["seed_count"] == 0:
                print(
                    "V060_P2_PROGRESS",
                    json.dumps({
                        "label": label,
                        "completed": counters[label],
                        "mode": result["mode"],
                        "period": result["period"],
                    }, sort_keys=True),
                    flush=True,
                )
        return result

    frozen.coordinate_run = instrumented_coordinate_run
    started = time.perf_counter()
    row = frozen.solve_trial(
        trial,
        language,
        model,
        FROZEN_CONFIG["seed_count"],
        FROZEN_CONFIG["screen_cycles"],
        FROZEN_CONFIG["screen_mono_iterations"],
        FROZEN_CONFIG["screen_mono_restarts"],
        FROZEN_CONFIG["screen_shift_iterations"],
        FROZEN_CONFIG["screen_shift_restarts"],
        FROZEN_CONFIG["top_refine"],
        FROZEN_CONFIG["refine_cycles"],
        FROZEN_CONFIG["refine_mono_iterations"],
        FROZEN_CONFIG["refine_mono_restarts"],
        FROZEN_CONFIG["refine_shift_iterations"],
        FROZEN_CONFIG["refine_shift_restarts"],
        FROZEN_CONFIG["final_mono_iterations"],
        FROZEN_CONFIG["final_mono_restarts"],
        FROZEN_CONFIG["final_shift_iterations"],
        FROZEN_CONFIG["final_shift_restarts"],
    )
    row["wrapper_elapsed_seconds"] = time.perf_counter() - started
    print("V060_P2_SHARD_TRIAL", json.dumps(row, sort_keys=True), flush=True)

    payload = {
        "config": {
            "iso": args.iso,
            "split": args.split,
            "length": args.length,
            "mode": args.mode,
            "replicate": args.replicate,
            "frozen_algorithm": FROZEN_CONFIG,
            "termination_correction": "finite_unique_then_seeded_replacement",
        },
        "row": row,
    }
    scientific = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(scientific).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V060_P2_SHARD_SHA256", payload["sha256"], flush=True)
    print("V060_P2_SHARD_OUTPUT", str(args.output), flush=True)


if __name__ == "__main__":
    main()
