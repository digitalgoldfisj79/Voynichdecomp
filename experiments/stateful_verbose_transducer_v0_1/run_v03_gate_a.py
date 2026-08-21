#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import svt_v02 as svt

DEV_OFFSET = 5000
LENGTH = 1536


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iso", default="de")
    p.add_argument("--replicates", type=int, default=4, help="per mode")
    args = p.parse_args()

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = svt.core.load_languages(
        root / "corpus_manifest_v050.json", args.repo / ".cache" / "svt-v03-gate-a"
    )
    language = languages[args.iso]
    model = svt.mono.build_language_model(language)
    trials = [
        svt.make_svt_trial(language, "dev", LENGTH, mode, DEV_OFFSET + replicate)
        for mode in svt.MODES
        for replicate in range(args.replicates)
    ]
    rows = [svt.solve_true_structure(trial, language, model) for trial in trials]
    rec = [float(row["recovery"]) for row in rows]
    summary = {
        "trials": len(rows),
        "mean_recovery": statistics.fmean(rec),
        "median_recovery": statistics.median(rec),
        "minimum_recovery": min(rec),
        "maximum_recovery": max(rec),
        "trials_ge_090": sum(value >= 0.90 for value in rec),
    }
    gate = (
        len(rows) == 8
        and summary["mean_recovery"] >= 0.95
        and summary["median_recovery"] >= 0.97
        and summary["minimum_recovery"] >= 0.85
        and summary["trials_ge_090"] >= 7
    )
    payload = {
        "programme": "SVT-v0.3",
        "stage": "A3_true_structure_key",
        "binding_development_gate": True,
        "voynich_opened": False,
        "replicate_offset": DEV_OFFSET,
        "iso": args.iso,
        "length": LENGTH,
        "rows": rows,
        "summary": summary,
        "gate_pass": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate_pass": gate, **summary}, indent=2, sort_keys=True))
    if not gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
