#!/usr/bin/env python3
"""Test-only confirmation runner for the frozen v0.5.2 quadgram solver."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numba import njit

import recoverability_v050 as core
import homophonic_solver_v052 as fixed
import mono_solver_v051 as mono


@njit(cache=True, nogil=True)
def quadgram_score_key(cipher, key, quadgram_logp, unigram_logp):
    length = cipher.shape[0]
    if length == 0:
        return -1e300
    score = 0.0
    prefix = 3 if length >= 3 else length
    for index in range(prefix):
        score += 0.12 * unigram_logp[key[cipher[index]]]
    for index in range(3, length):
        a = key[cipher[index - 3]]
        b = key[cipher[index - 2]]
        c = key[cipher[index - 1]]
        d = key[cipher[index]]
        score += quadgram_logp[a, b, c, d]
        score += 0.12 * unigram_logp[d]
    return score


def build_quadgram_model(language, alpha: float = 0.05):
    size = len(language.alphabet)
    counts = np.full((size, size, size, size), alpha, dtype=np.float32)
    contexts = np.full((size, size, size), alpha * size, dtype=np.float32)
    stream = language.train_stream
    for a, b, c, d in zip(stream, stream[1:], stream[2:], stream[3:]):
        counts[a, b, c, d] += 1.0
        contexts[a, b, c] += 1.0
    counts /= contexts[:, :, :, None]
    np.log(counts, out=counts)
    return counts, np.log(np.asarray(language.probabilities, dtype=np.float64))


def load_flexible_namespace(path: Path):
    source = path.read_text(encoding="utf-8")
    source = source.replace(
        "                old_label = int(key[first])\n                if new_label != old_label",
        "                first = int(first)\n                old_label = int(key[first])\n                new_label = int(new_label)\n                if new_label != old_label",
    )
    source = source.replace(
        "            if not changed:\n                temperature *= cooling\n                continue\n\n            candidate_score",
        "            if not changed:\n                temperature *= cooling\n                continue\n\n            first = int(first)\n            second = int(second)\n            old_label = int(old_label)\n            new_label = int(new_label)\n            candidate_score",
    )
    source = source.replace(
        "            if not changed:\n                continue\n            candidate_score",
        "            if not changed:\n                continue\n            first = int(first)\n            second = int(second)\n            old_label = int(old_label)\n            new_label = int(new_label)\n            candidate_score",
    )
    namespace = {"__name__": "v052_flexible_library", "__file__": str(path)}
    exec(compile(source, str(path), "exec"), namespace)
    return namespace, hashlib.sha256(source.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iso", required=True)
    parser.add_argument("--offset", type=int, default=32)
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

    family_arrays = namespace["family_arrays"]
    pool, caps, cdf = family_arrays(languages[args.iso])
    flexible_search(
        np.asarray([0, 1, 2, 3, 0, 2], dtype=np.int32),
        np.asarray([0, 0, 1, 1], dtype=np.int32),
        models[args.iso][0], models[args.iso][1], pool, caps, cdf, 2, 1, 1,
    )

    rows = fixed.run_grid(
        languages, models, "test", args.replicates, (96,),
        700000, 50, args.workers,
    )
    summary = summarize(rows)
    gate = {
        "language_pass": summary["mean_accuracy"] >= 0.60,
        "improves_original_fixed_smoke": summary["mean_accuracy"] > 0.572048611111111,
    }
    gate["pass"] = all(gate.values())
    payload = {
        "programme": "v0.5.2-quadgram-homophonic-confirmation",
        "iso": args.iso,
        "offset": args.offset,
        "replicates": args.replicates,
        "schedule": {"iterations": 700000, "restarts": 50},
        "patched_solver_sha256": patched_sha,
        "summary": summary,
        "gate": gate,
        "rows": rows,
    }
    scientific_blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["scientific_sha256"] = hashlib.sha256(scientific_blob).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V052_CONFIRM_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V052_CONFIRM_GATE", json.dumps(gate, sort_keys=True), flush=True)
    print("V052_CONFIRM_SHA256", payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
