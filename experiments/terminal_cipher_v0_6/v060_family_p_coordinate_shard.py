#!/usr/bin/env python3
"""Execution-only single-trial shard runner for the frozen Family P solver.

This wrapper does not alter the scientific algorithm or any search budget. It
selects one deterministic development trial, executes
``v060_family_p_coordinate_final.solve_trial`` with the frozen constants from
that script, emits the row immediately, and persists a self-hashed JSON result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

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


FROZEN_CONFIG = {
    "seed_count": 8,
    "screen_cycles": 2,
    "screen_mono_iterations": 50_000,
    "screen_mono_restarts": 5,
    "screen_shift_iterations": 25_000,
    "screen_shift_restarts": 6,
    "top_refine": 4,
    "refine_cycles": 2,
    "refine_mono_iterations": 250_000,
    "refine_mono_restarts": 24,
    "refine_shift_iterations": 50_000,
    "refine_shift_restarts": 12,
    "final_mono_iterations": 700_000,
    "final_mono_restarts": 50,
    "final_shift_iterations": 100_000,
    "final_shift_restarts": 24,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iso", default="en")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--length", type=int, default=384)
    parser.add_argument("--mode", choices=tuple(blind.MODES), required=True)
    parser.add_argument("--replicate", type=int, choices=range(8), required=True)
    args = parser.parse_args()

    if args.split != "dev":
        raise RuntimeError("Family P locked test remains sealed; only --split dev is permitted")

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        root / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v060-family-p2",
    )
    language = languages[args.iso]
    model = mono.build_language_model(language)
    trial = blind.make_trial(
        language, args.split, args.length, args.mode, args.replicate
    )

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
    print("V060_P2_SHARD_TRIAL", json.dumps(row, sort_keys=True), flush=True)

    payload = {
        "config": {
            "iso": args.iso,
            "split": args.split,
            "length": args.length,
            "mode": args.mode,
            "replicate": args.replicate,
            "frozen_algorithm": FROZEN_CONFIG,
        },
        "row": row,
    }
    scientific = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload["sha256"] = hashlib.sha256(scientific).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("V060_P2_SHARD_SHA256", payload["sha256"], flush=True)
    print("V060_P2_SHARD_OUTPUT", str(args.output), flush=True)


if __name__ == "__main__":
    main()
