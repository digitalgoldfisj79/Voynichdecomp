#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import svt_v02 as svt

BASE_OFFSET = 3000
LENGTHS = (384, 768, 1536)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iso", default="de")
    p.add_argument("--replicates", type=int, default=2, help="per mode and length")
    args = p.parse_args()

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = svt.core.load_languages(
        root / "corpus_manifest_v050.json", args.repo / ".cache" / "svt-v02-scale"
    )
    language = languages[args.iso]
    model = svt.mono.build_language_model(language)
    all_rows = []
    by_length = {}
    for length in LENGTHS:
        rows = []
        for mode in svt.MODES:
            for replicate in range(args.replicates):
                rid = BASE_OFFSET + length * 10 + replicate
                trial = svt.make_svt_trial(language, "dev", length, mode, rid)
                row = svt.solve_true_structure(trial, language, model)
                row["length"] = length
                rows.append(row)
                all_rows.append(row)
        rec = [float(row["recovery"]) for row in rows]
        by_length[str(length)] = {
            "trials": len(rows),
            "mean_recovery": statistics.fmean(rec),
            "median_recovery": statistics.median(rec),
            "minimum_recovery": min(rec),
            "maximum_recovery": max(rec),
            "trials_ge_085": sum(x >= 0.85 for x in rec),
        }
    payload = {
        "programme": "SVT-v0.2-post-failure-scale-diagnostic",
        "binding": False,
        "voynich_opened": False,
        "base_offset": BASE_OFFSET,
        "rows": all_rows,
        "by_length": by_length,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(by_length, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
