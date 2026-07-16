#!/usr/bin/env python3
"""v0.5.3 strict CrypTool-style restart-prefix calibration.

Runs independent fixed-inventory trajectories once and reports prefix-best
recovery after 12, 24, 48, 96 and 192 restarts.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

import recoverability_v050 as core
import homophonic_solver_v052 as homophonic
import mono_solver_v051 as mono
from homophonic_confirm_v052_quadgram import build_quadgram_model

PREFIXES = (12, 24, 48, 96, 192)


def load_cryptool_namespace(path: Path) -> tuple[dict[str, Any], str, str]:
    source = path.read_text(encoding="utf-8")
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    cast_needle = (
        "        if selected < 0:\n"
        "            state, selected = rng_int(state, alphabet_size)\n"
        "        key[key_index] = selected\n"
        "        distribution[selected] += 1\n"
    )
    cast_replacement = (
        "        if selected < 0:\n"
        "            state, selected = rng_int(state, alphabet_size)\n"
        "        selected = int(selected)\n"
        "        key[key_index] = selected\n"
        "        distribution[selected] += 1\n"
    )
    if source.count(cast_needle) != 1:
        raise RuntimeError("CrypTool port cast site mismatch")
    patched = source.replace(cast_needle, cast_replacement)
    namespace: dict[str, Any] = {
        "__name__": "v053_cryptool_library",
        "__file__": str(path),
    }
    exec(compile(patched, str(path), "exec"), namespace)
    return namespace, source_sha, hashlib.sha256(patched.encode("utf-8")).hexdigest()


def shuffled_key(initial_key: np.ndarray, seed: int) -> np.ndarray:
    output = initial_key.copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(output)
    return output


def one_restart(
    restart_index: int,
    trial: dict[str, Any],
    model: tuple[np.ndarray, np.ndarray],
    namespace: dict[str, Any],
    steps: int,
    target_acceptance: float,
) -> dict[str, Any]:
    cipher_values = list(map(int, trial["cipher"]))
    cipher = np.asarray(cipher_values, dtype=np.int32)
    initial_key = homophonic.frequency_slot_key(
        cipher_values,
        trial["inferred_labels"],
        trial["expected_slot_probabilities"],
    )
    if restart_index == 0:
        start_key = initial_key.copy()
    else:
        start_key = shuffled_key(
            initial_key,
            core.stable_seed("v053-classical-shuffle", trial["seed"], restart_index),
        )
    positions, offsets, rare_order = namespace["build_positions"](cipher_values)
    _min_counts, max_counts, proposal_cdf = namespace["distribution_arrays"](
        CURRENT_LANGUAGE
    )
    key, score, start_temperature, mutation_events = namespace[
        "cryptool_style_single_run"
    ](
        cipher,
        start_key,
        model[0],
        model[1],
        positions,
        offsets,
        rare_order,
        max_counts,
        proposal_cdf,
        steps,
        target_acceptance,
        50,
        0,
        int(
            core.stable_seed(
                "v053-classical-run", trial["seed"], restart_index
            )
            & 0x7FFFFFFFFFFFFFFF
        ),
    )
    prediction = key[cipher].tolist()
    return {
        "replicate": int(trial["replicate"]),
        "restart": restart_index + 1,
        "score": float(score),
        "accuracy": mono.fast_accuracy(trial["plain"], prediction),
        "exact": prediction == trial["plain"],
        "start_temperature": float(start_temperature),
        "mutation_events": int(mutation_events),
    }


def summarize_prefix(rows: list[dict[str, Any]], prefix: int) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for replicate in sorted({int(row["replicate"]) for row in rows}):
        candidates = [
            row
            for row in rows
            if int(row["replicate"]) == replicate and int(row["restart"]) <= prefix
        ]
        selected.append(max(candidates, key=lambda row: float(row["score"])))
    accuracies = [float(row["accuracy"]) for row in selected]
    return {
        "prefix_restarts": prefix,
        "trials": len(selected),
        "mean_accuracy": statistics.fmean(accuracies),
        "median_accuracy": statistics.median(accuracies),
        "exact_rate": statistics.fmean(float(row["exact"]) for row in selected),
        "at_least_70_rate": statistics.fmean(value >= 0.70 for value in accuracies),
        "at_least_90_rate": statistics.fmean(value >= 0.90 for value in accuracies),
        "at_least_95_rate": statistics.fmean(value >= 0.95 for value in accuracies),
        "selected_rows": selected,
    }


def canonical_sha(payload: dict[str, Any]) -> str:
    blob = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


CURRENT_LANGUAGE: core.LanguageData


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iso", default="en")
    parser.add_argument("--split", default="dev", choices=("dev", "test"))
    parser.add_argument("--length", type=int, default=384)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--max-restarts", type=int, default=192)
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--target-acceptance", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    if args.max_restarts < max(PREFIXES):
        raise RuntimeError("max-restarts must cover all frozen prefixes")

    experiment = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        experiment / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v050_corpora",
    )
    global CURRENT_LANGUAGE
    CURRENT_LANGUAGE = languages[args.iso]
    model = build_quadgram_model(CURRENT_LANGUAGE)
    namespace, source_sha, patched_sha = load_cryptool_namespace(
        experiment / "cryptool_homophonic_port_v052.py"
    )
    trials = [
        homophonic.make_trial(
            CURRENT_LANGUAGE, args.split, args.length, args.offset + replicate
        )
        for replicate in range(args.replicates)
    ]

    # Compile once before starting concurrent trajectories.
    compile_trial = trials[0]
    cipher_values = list(map(int, compile_trial["cipher"]))
    initial_key = homophonic.frequency_slot_key(
        cipher_values,
        compile_trial["inferred_labels"],
        compile_trial["expected_slot_probabilities"],
    )
    positions, offsets, rare_order = namespace["build_positions"](cipher_values)
    _min_counts, max_counts, proposal_cdf = namespace["distribution_arrays"](
        CURRENT_LANGUAGE
    )
    namespace["cryptool_style_single_run"](
        np.asarray(cipher_values, dtype=np.int32),
        initial_key,
        model[0],
        model[1],
        positions,
        offsets,
        rare_order,
        max_counts,
        proposal_cdf,
        10,
        args.target_acceptance,
        2,
        0,
        1,
    )

    jobs = [
        (restart, trial)
        for trial in trials
        for restart in range(args.max_restarts)
    ]
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                one_restart,
                restart,
                trial,
                model,
                namespace,
                args.steps,
                args.target_acceptance,
            )
            for restart, trial in jobs
        ]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            rows.append(future.result())
            if completed % 96 == 0 or completed == len(futures):
                print(
                    f"V053_CLASSICAL_PROGRESS {completed}/{len(futures)}",
                    flush=True,
                )
    rows.sort(key=lambda row: (int(row["replicate"]), int(row["restart"])))

    prefix_results = [summarize_prefix(rows, prefix) for prefix in PREFIXES]
    eligible = [
        result
        for result in prefix_results
        if result["mean_accuracy"] >= 0.70
        and result["median_accuracy"] >= 0.90
        and result["at_least_70_rate"] >= 0.875
    ]
    selected = min(eligible, key=lambda item: item["prefix_restarts"]) if eligible else None
    payload: dict[str, Any] = {
        "programme": "recoverability-frontier-v0.5.3-classical-restart-curve",
        "iso": args.iso,
        "split": args.split,
        "length": args.length,
        "offset": args.offset,
        "replicates": args.replicates,
        "steps_per_restart": args.steps,
        "maximum_restarts": args.max_restarts,
        "target_initial_acceptance": args.target_acceptance,
        "inventory": "fixed inferred multiset",
        "cryptool_source_sha256": source_sha,
        "cryptool_patched_sha256": patched_sha,
        "prefix_results": prefix_results,
        "selected_prefix": None if selected is None else selected["prefix_restarts"],
        "development_gate_pass": selected is not None,
        "all_restart_rows": rows,
    }
    payload["scientific_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    for result in prefix_results:
        compact = {key: value for key, value in result.items() if key != "selected_rows"}
        print("V053_CLASSICAL_PREFIX", json.dumps(compact, sort_keys=True), flush=True)
    print("V053_CLASSICAL_SELECTED", payload["selected_prefix"], flush=True)
    print("V053_CLASSICAL_SHA256", payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
