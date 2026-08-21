#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    files = sorted(args.input_dir.rglob("svt_v032_*.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    groups = defaultdict(list)
    for row in rows:
        key = (row["true_mode"], int(row["replicate"]))
        groups[key].append(row)

    selected_trials = []
    for key, candidates in sorted(groups.items()):
        if len(candidates) != 22:
            raise RuntimeError(f"trial {key} has {len(candidates)} candidates, expected 22")
        selected = max(candidates, key=lambda r: float(r["selected"]["score"]))
        ordered = sorted(candidates, key=lambda r: float(r["selected"]["score"]), reverse=True)
        second = ordered[1]
        selected_trials.append({
            "true_mode": selected["true_mode"],
            "true_period": int(selected["true_period"]),
            "replicate": int(selected["replicate"]),
            "selected_mode": selected["candidate_mode"],
            "selected_period": int(selected["candidate_period"]),
            "structure_correct": (
                selected["candidate_mode"] == selected["true_mode"]
                and int(selected["candidate_period"]) == int(selected["true_period"])
            ),
            "recovery": float(selected["selected"]["recovery"]),
            "score": float(selected["selected"]["score"]),
            "runner_up_mode": second["candidate_mode"],
            "runner_up_period": int(second["candidate_period"]),
            "runner_up_score": float(second["selected"]["score"]),
            "score_margin": float(selected["selected"]["score"]) - float(second["selected"]["score"]),
        })

    if len(selected_trials) != 8:
        raise RuntimeError(f"got {len(selected_trials)} trials, expected 8")

    rec = [row["recovery"] for row in selected_trials]
    structure_correct = sum(bool(row["structure_correct"]) for row in selected_trials)
    summary = {
        "trials": len(selected_trials),
        "exact_structure_correct": structure_correct,
        "mean_recovery": statistics.fmean(rec),
        "median_recovery": statistics.median(rec),
        "minimum_recovery": min(rec),
        "maximum_recovery": max(rec),
        "trials_ge_090": sum(value >= 0.90 for value in rec),
    }
    gate = (
        structure_correct == 8
        and summary["mean_recovery"] >= 0.95
        and summary["median_recovery"] >= 0.97
        and summary["minimum_recovery"] >= 0.85
        and summary["trials_ge_090"] == 8
    )
    payload = {
        "programme": "SVT-v0.3.2",
        "stage": "A3.2_blind_mode_period_key",
        "binding": True,
        "voynich_opened": False,
        "selection_rule": "for each trial select maximum frozen penalised score over 22 structures; each structure itself selected over exactly 12 starts",
        "trials": selected_trials,
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
