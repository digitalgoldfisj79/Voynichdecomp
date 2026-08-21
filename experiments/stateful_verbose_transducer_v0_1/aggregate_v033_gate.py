#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    files = sorted(args.input_dir.rglob("svt_v033_*.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    if len(rows) != 8:
        raise RuntimeError(f"got {len(rows)} trials, expected 8")

    rec = [float(r["selected_recovery"]) for r in rows]
    screen_ok = sum(int(r["screen_truth_rank"]) <= 6 for r in rows)
    structure_ok = sum(
        r["selected_mode"] == r["true_mode"]
        and int(r["selected_period"]) == int(r["true_period"])
        for r in rows
    )
    summary = {
        "trials": 8,
        "screen_truth_in_top6": int(screen_ok),
        "exact_structure_correct": int(structure_ok),
        "mean_recovery": statistics.fmean(rec),
        "median_recovery": statistics.median(rec),
        "minimum_recovery": min(rec),
        "maximum_recovery": max(rec),
        "trials_ge_090": sum(v >= 0.90 for v in rec),
    }
    gate = (
        screen_ok == 8
        and structure_ok == 8
        and summary["mean_recovery"] >= 0.95
        and summary["median_recovery"] >= 0.97
        and summary["minimum_recovery"] >= 0.85
        and summary["trials_ge_090"] == 8
    )
    payload = {
        "programme": "SVT-v0.3.3",
        "stage": "A3.3_blind_structure_shortlist",
        "binding": True,
        "voynich_opened": False,
        "selection_rule": "truth-free top6 screen over 22 structures, then frozen 12-start penalised-score decoder on each shortlisted structure",
        "trials": rows,
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
