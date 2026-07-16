#!/usr/bin/env python3
"""Run the full frozen Family P development grid with the termination fix.

The original frozen main already uses 16 concurrent trial workers. This wrapper
only supplies the pre-registered seed termination correction, bounded progress
instrumentation, and a complete final payload log.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
V05 = HERE.parent / "recoverability_frontier_v0_5"
if str(V05) not in sys.path:
    sys.path.insert(0, str(V05))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import recoverability_v050 as core
import v060_family_p_coordinate_final as frozen
import v060_family_p_mode_blind as blind
from v060_family_p_coordinate_shard import FROZEN_CONFIG
from v060_family_p_coordinate_shard_termination_fixed import finite_phase_histogram_seeds


def main() -> None:
    repo = Path(sys.argv[1])
    output = Path(sys.argv[2])

    root = repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        root / "corpus_manifest_v050.json",
        repo / ".cache" / "v060-family-p2-preflight",
    )
    language = languages["en"]
    trial = blind.make_trial(language, "dev", 384, "periodic", 0)
    preflight = []
    for mode in blind.MODES:
        for period in blind.CANDIDATE_PERIODS:
            phase = frozen.base.phase_array(trial.length, period, mode, trial.line_starts)
            seeds = finite_phase_histogram_seeds(
                trial.cipher,
                phase,
                period,
                len(language.alphabet),
                FROZEN_CONFIG["seed_count"],
                core.stable_seed("v060-p2-seeds", trial.seed, mode, period),
            )
            unique = len({tuple(int(x) for x in seed) for seed in seeds})
            preflight.append({
                "mode": mode,
                "period": period,
                "starts": len(seeds),
                "unique_starts": unique,
            })
            if len(seeds) != FROZEN_CONFIG["seed_count"]:
                raise RuntimeError("seed preflight failed")
    print("V060_P2_FULL_SEED_PREFLIGHT", json.dumps(preflight, sort_keys=True), flush=True)

    frozen.phase_histogram_seeds = finite_phase_histogram_seeds
    original_coordinate_run = frozen.coordinate_run
    lock = threading.Lock()
    counts = {"screen": 0, "refine": 0}

    def instrumented_coordinate_run(*positional, **keywords):
        result = original_coordinate_run(*positional, **keywords)
        label = str(keywords.get("label", positional[12] if len(positional) > 12 else "unknown"))
        if label in counts:
            with lock:
                counts[label] += 1
                count = counts[label]
            interval = 128 if label == "screen" else 8
            if count % interval == 0:
                print(
                    "V060_P2_FULL_PROGRESS",
                    json.dumps({"label": label, "completed": count}, sort_keys=True),
                    flush=True,
                )
        return result

    frozen.coordinate_run = instrumented_coordinate_run
    sys.argv = [
        sys.argv[0],
        "--repo", str(repo),
        "--output", str(output),
        "--iso", "en",
        "--split", "dev",
        "--length", "384",
        "--replicates", "8",
        "--workers", "16",
    ]
    frozen.main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["execution_clarification"] = {
        "termination_fix": "finite_unique_then_seeded_replacement",
        "clarification_commit": "b4f7f556766e29e8aeadf85b3f596e7f81bfac7b",
    }
    print("V060_P2_FULL_PAYLOAD", json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
