#!/usr/bin/env python3
"""Diagnose whether the v0.5.1 trigram objective ranks the true mono key."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

import recoverability_v050 as core
import mono_solver_v051 as mono


def true_key_from_alignment(cipher: list[int], plain: list[int], alphabet_size: int) -> np.ndarray:
    key = np.full(alphabet_size, -1, dtype=np.int32)
    used_plain: set[int] = set()
    for symbol, value in zip(cipher, plain):
        if key[symbol] >= 0 and int(key[symbol]) != value:
            raise RuntimeError("inconsistent mono alignment")
        key[symbol] = value
        used_plain.add(value)
    remaining_plain = [value for value in range(alphabet_size) if value not in used_plain]
    cursor = 0
    for symbol in range(alphabet_size):
        if key[symbol] < 0:
            key[symbol] = remaining_plain[cursor]
            cursor += 1
    return key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=12)
    args = parser.parse_args()

    experiment = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        experiment / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v050_corpora",
    )
    languages = {iso: languages[iso] for iso in ("en", "tr")}
    rows = []

    # Compile before timing or interpretation.
    first = languages["en"]
    model = mono.build_language_model(first)
    mono.anneal_mono(
        np.asarray([0, 1, 0, 1], dtype=np.int32),
        np.arange(len(first.alphabet), dtype=np.int32),
        model[0],
        model[1],
        2,
        1,
        1,
    )

    for iso, language in languages.items():
        trigram, unigram = mono.build_language_model(language)
        chunks = core.source_chunks(language, "dev", 96)
        for replicate in range(args.replicates):
            plain = list(chunks[replicate % len(chunks)])
            seed = core.stable_seed("v051-objective", iso, replicate)
            packet = core.encrypt_sequence(
                plain,
                "mono",
                language,
                random.Random(seed),
                parameter_mode="dev",
            )
            cipher = mono.canonicalize(packet.cipher)
            cipher_array = np.asarray(cipher, dtype=np.int32)
            initial = mono.frequency_key(cipher, language)
            true_key = true_key_from_alignment(cipher, plain, len(language.alphabet))
            recovered, recovered_score = mono.anneal_mono(
                cipher_array,
                initial,
                trigram,
                unigram,
                12000,
                6,
                int(seed & 0x7FFFFFFFFFFFFFFF),
            )
            true_score = mono.score_key(cipher_array, true_key, trigram, unigram)
            baseline_score = mono.score_key(cipher_array, initial, trigram, unigram)
            recovered_plain = recovered[cipher_array].tolist()
            rows.append({
                "iso": iso,
                "replicate": replicate,
                "true_score": float(true_score),
                "baseline_score": float(baseline_score),
                "recovered_score": float(recovered_score),
                "true_beats_recovered": bool(true_score >= recovered_score),
                "accuracy": mono.fast_accuracy(plain, recovered_plain),
                "baseline_accuracy": mono.fast_accuracy(plain, initial[cipher_array].tolist()),
            })

    summary = {
        "trials": len(rows),
        "true_beats_recovered_rate": sum(row["true_beats_recovered"] for row in rows) / len(rows),
        "mean_accuracy": sum(row["accuracy"] for row in rows) / len(rows),
        "mean_baseline_accuracy": sum(row["baseline_accuracy"] for row in rows) / len(rows),
        "mean_recovered_minus_true_score": sum(
            row["recovered_score"] - row["true_score"] for row in rows
        ) / len(rows),
        "rows": rows,
    }
    print("V051_OBJECTIVE_DIAGNOSTIC", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
