#!/usr/bin/env python3
"""Length-normalized quadgram homophonic search for v0.5.2 diagnostics."""
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


def load_scaled_namespace(path: Path):
    source = path.read_text(encoding="utf-8")
    original_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()

    reassignment = "                old_label = int(key[first])\n                if new_label != old_label"
    reassignment_fixed = (
        "                first = int(first)\n"
        "                old_label = int(key[first])\n"
        "                new_label = int(new_label)\n"
        "                if new_label != old_label"
    )
    if source.count(reassignment) != 3:
        raise RuntimeError("reassignment site mismatch")
    source = source.replace(reassignment, reassignment_fixed)

    anneal_site = (
        "            if not changed:\n"
        "                temperature *= cooling\n"
        "                continue\n\n"
        "            candidate_score"
    )
    anneal_fixed = (
        "            if not changed:\n"
        "                temperature *= cooling\n"
        "                continue\n\n"
        "            first = int(first)\n"
        "            second = int(second)\n"
        "            old_label = int(old_label)\n"
        "            new_label = int(new_label)\n"
        "            candidate_score"
    )
    if source.count(anneal_site) != 1:
        raise RuntimeError("annealing scoring site mismatch")
    source = source.replace(anneal_site, anneal_fixed)

    polish_site = (
        "            if not changed:\n"
        "                continue\n"
        "            candidate_score"
    )
    polish_fixed = (
        "            if not changed:\n"
        "                continue\n"
        "            first = int(first)\n"
        "            second = int(second)\n"
        "            old_label = int(old_label)\n"
        "            new_label = int(new_label)\n"
        "            candidate_score"
    )
    if source.count(polish_site) != 1:
        raise RuntimeError("polishing scoring site mismatch")
    source = source.replace(polish_site, polish_fixed)

    temperature_site = "        temperature = 35.0\n"
    temperature_fixed = (
        "        length_scale = max(1.0, cipher.shape[0] / 96.0)\n"
        "        temperature = 35.0 * length_scale\n"
    )
    if source.count(temperature_site) != 1:
        raise RuntimeError("temperature site mismatch")
    source = source.replace(temperature_site, temperature_fixed)

    reheat_site = "                temperature = max(temperature, 3.0)\n"
    reheat_fixed = "                temperature = max(temperature, 3.0 * length_scale)\n"
    if source.count(reheat_site) != 1:
        raise RuntimeError("reheat site mismatch")
    source = source.replace(reheat_site, reheat_fixed)

    namespace = {"__name__": "v052_temperature_scaled_library", "__file__": str(path)}
    exec(compile(source, str(path), "exec"), namespace)
    return namespace, original_sha, hashlib.sha256(source.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iso", required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--lengths", default="96,192,384")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=700000)
    parser.add_argument("--restarts", type=int, default=50)
    args = parser.parse_args()

    lengths = tuple(int(value) for value in args.lengths.split(",") if value)
    experiment = args.repo / "experiments" / "recoverability_frontier_v0_5"
    namespace, source_sha, scaled_sha = load_scaled_namespace(
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
        languages,
        models,
        args.split,
        args.replicates,
        lengths,
        args.iterations,
        args.restarts,
        args.workers,
    )
    summary = summarize(rows)
    payload = {
        "programme": "v0.5.2-length-normalized-homophonic-search",
        "iso": args.iso,
        "split": args.split,
        "offset": args.offset,
        "replicates_per_length": args.replicates,
        "lengths": list(lengths),
        "schedule": {"iterations": args.iterations, "restarts": args.restarts},
        "temperature_reference_length": 96,
        "source_solver_sha256": source_sha,
        "scaled_solver_sha256": scaled_sha,
        "summary": summary,
        "rows": rows,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["scientific_sha256"] = hashlib.sha256(blob).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V052_TEMP_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V052_TEMP_SHA256", payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
