#!/usr/bin/env python3
"""Deeper global-search launcher for the v0.5.1 mono solver."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from numba import njit

import recoverability_v050 as core
import mono_solver_v051 as base


@njit(cache=True, nogil=True)
def anneal_mono_search2(
    cipher: np.ndarray,
    initial_key: np.ndarray,
    trigram_logp: np.ndarray,
    unigram_logp: np.ndarray,
    iterations: int,
    restarts: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    alphabet_size = initial_key.shape[0]
    best_key = initial_key.copy()
    best_score = base.score_key(cipher, best_key, trigram_logp, unigram_logp)
    state = np.uint64(seed if seed > 0 else 1)

    for restart in range(restarts):
        if restart == 0:
            key = initial_key.copy()
        elif restart % 5 == 0:
            key = initial_key.copy()
            # Full Fisher-Yates shuffle supplies genuinely global restarts.
            for index in range(alphabet_size - 1, 0, -1):
                state, other = base._rng_int(state, index + 1)
                temporary = key[index]
                key[index] = key[other]
                key[other] = temporary
        elif restart % 2 == 0:
            key = best_key.copy()
            perturbations = 8 + (restart % 11) * 4
            for _ in range(perturbations):
                state, first = base._rng_int(state, alphabet_size)
                state, second = base._rng_int(state, alphabet_size)
                if first != second:
                    temporary = key[first]
                    key[first] = key[second]
                    key[second] = temporary
        else:
            key = initial_key.copy()
            perturbations = 15 + (restart % 13) * 5
            for _ in range(perturbations):
                state, first = base._rng_int(state, alphabet_size)
                state, second = base._rng_int(state, alphabet_size)
                if first != second:
                    temporary = key[first]
                    key[first] = key[second]
                    key[second] = temporary

        current_score = base.score_key(cipher, key, trigram_logp, unigram_logp)
        if current_score > best_score:
            best_score = current_score
            best_key = key.copy()

        temperature = 35.0
        cooling = math.exp(math.log(0.025 / 35.0) / max(1, iterations))
        stagnant = 0

        for _ in range(iterations):
            state, first = base._rng_int(state, alphabet_size)
            state, second = base._rng_int(state, alphabet_size)
            if first == second:
                continue
            temporary = key[first]
            key[first] = key[second]
            key[second] = temporary
            candidate_score = base.score_key(cipher, key, trigram_logp, unigram_logp)
            delta = candidate_score - current_score
            accept = delta >= 0.0
            if not accept:
                state, uniform = base._rng_float(state)
                accept = uniform < math.exp(delta / max(temperature, 1e-12))
            if accept:
                current_score = candidate_score
                if current_score > best_score:
                    best_score = current_score
                    best_key = key.copy()
                    stagnant = 0
                else:
                    stagnant += 1
            else:
                temporary = key[first]
                key[first] = key[second]
                key[second] = temporary
                stagnant += 1
            temperature *= cooling
            if stagnant >= 20000:
                temperature = max(temperature, 3.0)
                stagnant = 0

        # Greedy polishing after each annealing trajectory.
        polish_iterations = max(5000, iterations // 5)
        for _ in range(polish_iterations):
            state, first = base._rng_int(state, alphabet_size)
            state, second = base._rng_int(state, alphabet_size)
            if first == second:
                continue
            temporary = key[first]
            key[first] = key[second]
            key[second] = temporary
            candidate_score = base.score_key(cipher, key, trigram_logp, unigram_logp)
            if candidate_score >= current_score:
                current_score = candidate_score
                if current_score > best_score:
                    best_score = current_score
                    best_key = key.copy()
            else:
                temporary = key[first]
                key[first] = key[second]
                key[second] = temporary

    return best_key, best_score


def canonical_sha(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dev-reps", type=int, default=8)
    parser.add_argument("--test-reps", type=int, default=20)
    args = parser.parse_args()

    base.anneal_mono = anneal_mono_search2
    experiment = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        experiment / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v050_corpora",
    )
    if args.smoke:
        languages = {iso: languages[iso] for iso in ("en", "tr")}
        lengths = (96,)
        args.dev_reps = min(args.dev_reps, 4)
        args.test_reps = min(args.test_reps, 6)
        schedule_grid = ((30000, 10), (100000, 20), (250000, 30))
    else:
        lengths = (96, 192, 384)
        schedule_grid = ((100000, 20), (300000, 35), (700000, 50))

    models = {iso: base.build_language_model(language) for iso, language in languages.items()}
    first_iso = sorted(languages)[0]
    first_language = languages[first_iso]
    first_model = models[first_iso]
    anneal_mono_search2(
        np.asarray([0, 1, 0, 1], dtype=np.int32),
        np.arange(len(first_language.alphabet), dtype=np.int32),
        first_model[0],
        first_model[1],
        2,
        1,
        1,
    )

    candidates = []
    selected = None
    selected_score = -1.0
    for iterations, restarts in schedule_grid:
        rows = base.run_grid(
            languages,
            models,
            "dev",
            args.dev_reps,
            lengths,
            iterations,
            restarts,
            args.workers,
        )
        summary = base.summarize(rows)
        candidate = {"iterations": iterations, "restarts": restarts, "summary": summary}
        candidates.append(candidate)
        print("V051_SEARCH2_DEV", json.dumps(candidate, sort_keys=True), flush=True)
        if summary["mean_accuracy"] > selected_score:
            selected_score = summary["mean_accuracy"]
            selected = (iterations, restarts)

    assert selected is not None
    test_rows = base.run_grid(
        languages,
        models,
        "test",
        args.test_reps,
        lengths,
        selected[0],
        selected[1],
        args.workers,
    )
    test_summary = base.summarize(test_rows)
    language_floor = min(value["mean_accuracy"] for value in test_summary["by_language"].values())
    gate = {
        "mean_accuracy_pass": test_summary["mean_accuracy"] >= 0.70,
        "language_floor_pass": language_floor >= 0.50,
    }
    gate["pass"] = all(gate.values())
    payload = {
        "programme": "recoverability-frontier-v0.5.1-mono-search2",
        "development_candidates": candidates,
        "selected_schedule": {"iterations": selected[0], "restarts": selected[1]},
        "test_summary": test_summary,
        "test_rows": test_rows,
        "gate": gate,
    }
    payload["scientific_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V051_SEARCH2_SELECTED", json.dumps(payload["selected_schedule"], sort_keys=True), flush=True)
    print("V051_SEARCH2_TEST", json.dumps(test_summary, sort_keys=True), flush=True)
    print("V051_SEARCH2_GATE", json.dumps(gate, sort_keys=True), flush=True)
    print("V051_SEARCH2_SHA256", payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
