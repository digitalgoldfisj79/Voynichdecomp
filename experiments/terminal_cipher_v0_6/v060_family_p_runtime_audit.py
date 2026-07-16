#!/usr/bin/env python3
"""Execution-only runtime audit for the frozen v0.6 Family P solver.

This script changes no search objective, data, seed, threshold, or scientific
budget. It times the exact Numba kernels on a deterministic development trial
and compares independent coordinate starts serially and in parallel.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
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
import v060_family_p_stage_a as base


def timed(label, fn):
    started = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - started
    print("V060_P_AUDIT", json.dumps({"label": label, "elapsed_seconds": elapsed}, sort_keys=True), flush=True)
    return value, elapsed


def main() -> None:
    repo = Path(sys.argv[1])
    root = repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(root / "corpus_manifest_v050.json", repo / ".cache" / "v060-family-p-audit")
    language = languages["en"]
    model = mono.build_language_model(language)
    trial = blind.make_trial(language, "dev", 384, "periodic", 0)
    trigram, unigram = model
    a = len(language.alphabet)
    phase = base.phase_array(trial.length, 2, "periodic", trial.line_starts)
    cipher = np.asarray(trial.cipher, dtype=np.int32)
    shifts = np.zeros(2, dtype=np.int32)
    inverse = mono.frequency_key(trial.cipher, language)

    print("V060_P_AUDIT_ENV", json.dumps({
        "logical_cpus": os.cpu_count(),
        "alphabet_size": a,
        "length": trial.length,
        "true_period": trial.period,
    }, sort_keys=True), flush=True)

    # Compile exact kernels before timing.
    mono.anneal_mono(cipher, inverse, trigram, unigram, 1, 1, 101)
    frozen.anneal_shifts_seeded(cipher, phase, inverse, shifts, trigram, unigram, 1, 1, 102)

    iterations = 100_000
    _, mono_elapsed = timed("mono_100k_x1", lambda: mono.anneal_mono(
        cipher, inverse, trigram, unigram, iterations, 1, 201
    ))
    _, shift_elapsed = timed("shift_100k_x1", lambda: frozen.anneal_shifts_seeded(
        cipher, phase, inverse, shifts, trigram, unigram, iterations, 1, 202
    ))

    # Identical independent starts; executor.map preserves deterministic input order.
    task_count = 32
    task_iterations = 20_000
    def one_task(index: int):
        return mono.anneal_mono(
            cipher, inverse, trigram, unigram, task_iterations, 1, 1000 + index
        )[1]

    _, serial_elapsed = timed("32x_mono20k_serial", lambda: [one_task(i) for i in range(task_count)])
    _, parallel_elapsed = timed("32x_mono20k_parallel32", lambda: list(
        concurrent.futures.ThreadPoolExecutor(max_workers=32).map(one_task, range(task_count))
    ))

    mono_rate = iterations / mono_elapsed
    shift_rate = iterations / shift_elapsed
    screen_proposals = 2 * 11 * 8 * 2 * (50_000 * 5 + 25_000 * 6)
    refine_proposals = 4 * 2 * (250_000 * 24 + 50_000 * 12)
    final_proposals = 2 * 700_000 * 50 + 100_000 * 24
    total_proposals = screen_proposals + refine_proposals + final_proposals
    estimated_serial_seconds = (
        (2 * 11 * 8 * 2 * 50_000 * 5 + 4 * 2 * 250_000 * 24 + 2 * 700_000 * 50) / mono_rate
        + (2 * 11 * 8 * 2 * 25_000 * 6 + 4 * 2 * 50_000 * 12 + 100_000 * 24) / shift_rate
    )
    print("V060_P_AUDIT_SUMMARY", json.dumps({
        "mono_proposals_per_second": mono_rate,
        "shift_proposals_per_second": shift_rate,
        "parallel_speedup_32": serial_elapsed / parallel_elapsed,
        "screen_proposals": screen_proposals,
        "refine_proposals": refine_proposals,
        "final_proposals": final_proposals,
        "total_full_rescore_proposals": total_proposals,
        "estimated_serial_seconds": estimated_serial_seconds,
        "estimated_serial_hours": estimated_serial_seconds / 3600,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
