#!/usr/bin/env python3
"""Language-sharded full v0.5.2 homophonic generalisation runner."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import recoverability_v050 as core
import homophonic_solver_v052 as fixed
import mono_solver_v051 as mono
from homophonic_confirm_v052_quadgram import (
    build_quadgram_model,
    load_flexible_namespace,
    quadgram_score_key,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iso", required=True)
    parser.add_argument("--offset", type=int, default=64)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    experiment = args.repo / "experiments" / "recoverability_frontier_v0_5"
    namespace, patched_sha = load_flexible_namespace(
        experiment / "homophonic_solver_v052_flexible.py"
    )
    flexible_search = namespace["flexible_homophonic_search"]
    flexible_solve = namespace["flexible_solve_trial"]
    summarize = namespace["summarize"]
    family_arrays = namespace["family_arrays"]

    mono.score_key = quadgram_score_key
    mono.build_language_model = build_quadgram_model
    fixed.solve_trial = flexible_solve
    fixed.summarize = summarize
    original_make_trial = fixed.make_trial

    def offset_make_trial(language, split, length, replicate):
        return original_make_trial(language, split, length, replicate + args.offset)

    fixed.make_trial = offset_make_trial
    languages = core.load_languages(
        experiment / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v050_corpora",
    )
    if args.iso not in languages:
        raise RuntimeError(f"unknown language {args.iso}")
    languages = {args.iso: languages[args.iso]}
    models = {args.iso: build_quadgram_model(languages[args.iso])}

    pool, caps, cdf = family_arrays(languages[args.iso])
    flexible_search(
        np.asarray([0, 1, 2, 3, 0, 2], dtype=np.int32),
        np.asarray([0, 0, 1, 1], dtype=np.int32),
        models[args.iso][0], models[args.iso][1], pool, caps, cdf, 2, 1, 1,
    )

    rows = fixed.run_grid(
        languages, models, "test", args.replicates, (96, 192, 384),
        700000, 50, args.workers,
    )
    summary = summarize(rows)
    gate = {
        "language_pass": summary["mean_accuracy"] >= 0.50,
        "short_text_pass": summary["by_length"]["96"]["mean_accuracy"] >= 0.60,
        "long_text_noncollapse": (
            summary["by_length"]["192"]["mean_accuracy"] >= 0.60
            and summary["by_length"]["384"]["mean_accuracy"] >= 0.60
        ),
    }
    gate["pass"] = all(gate.values())
    payload = {
        "programme": "v0.5.2-homophonic-full-generalisation",
        "iso": args.iso,
        "offset": args.offset,
        "replicates_per_length": args.replicates,
        "lengths": [96, 192, 384],
        "schedule": {"iterations": 700000, "restarts": 50},
        "patched_solver_sha256": patched_sha,
        "summary": summary,
        "gate": gate,
        "rows": rows,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["scientific_sha256"] = hashlib.sha256(blob).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V052_FULL_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V052_FULL_GATE", json.dumps(gate, sort_keys=True), flush=True)
    print("V052_FULL_SHA256", payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
