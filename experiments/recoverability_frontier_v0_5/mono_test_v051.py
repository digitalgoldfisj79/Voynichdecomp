#!/usr/bin/env python3
"""One-shot test-only runner for a development-selected v0.5.1 mono schedule."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import recoverability_v050 as core
import mono_solver_v051 as base
import mono_solver_v051_search2 as search2


def canonical_sha(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--restarts", type=int, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--iso", action="append", default=[])
    args = parser.parse_args()

    base.anneal_mono = search2.anneal_mono_search2
    experiment = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        experiment / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v050_corpora",
    )
    if args.iso:
        requested = tuple(dict.fromkeys(args.iso))
        missing = [iso for iso in requested if iso not in languages]
        if missing:
            raise RuntimeError(f"unknown language codes: {missing}")
        languages = {iso: languages[iso] for iso in requested}
    models = {iso: base.build_language_model(language) for iso, language in languages.items()}
    first_iso = sorted(languages)[0]
    search2.anneal_mono_search2(
        np.asarray([0, 1, 0, 1], dtype=np.int32),
        np.arange(len(languages[first_iso].alphabet), dtype=np.int32),
        models[first_iso][0],
        models[first_iso][1],
        2,
        1,
        1,
    )

    rows = base.run_grid(
        languages,
        models,
        "test",
        args.replicates,
        (96, 192, 384),
        args.iterations,
        args.restarts,
        args.workers,
    )
    summary = base.summarize(rows)
    language_floor = min(value["mean_accuracy"] for value in summary["by_language"].values())
    gate = {
        "mean_accuracy_pass": summary["mean_accuracy"] >= 0.70,
        "language_floor_pass": language_floor >= 0.50,
    }
    gate["pass"] = all(gate.values())
    payload = {
        "programme": "recoverability-frontier-v0.5.1-mono-one-shot-test",
        "language_shard": sorted(languages),
        "selected_schedule": {
            "iterations": args.iterations,
            "restarts": args.restarts,
        },
        "test_summary": summary,
        "test_rows": rows,
        "gate": gate,
    }
    payload["scientific_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V051_MONO_TEST_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V051_MONO_TEST_GATE", json.dumps(gate, sort_keys=True), flush=True)
    print("V051_MONO_TEST_SHA256", payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
