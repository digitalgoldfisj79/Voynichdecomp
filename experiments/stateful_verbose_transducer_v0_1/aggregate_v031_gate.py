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

    files = sorted(args.input_dir.rglob("svt_v031_*.json"))
    rows = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        sel = payload["selected"]
        rows.append({
            "mode": payload["mode"],
            "replicate": payload["replicate"],
            "period": payload["true_period"],
            "recovery": float(sel["recovery"]),
            "score": float(sel["score"]),
            "selected_start": int(sel["start"]),
        })

    if len(rows) != 8:
        raise SystemExit(f"expected 8 trial files, found {len(rows)}")
    rec = [r["recovery"] for r in rows]
    summary = {
        "trials": len(rows),
        "mean_recovery": statistics.fmean(rec),
        "median_recovery": statistics.median(rec),
        "minimum_recovery": min(rec),
        "maximum_recovery": max(rec),
        "trials_ge_090": sum(v >= 0.90 for v in rec),
    }
    gate = (
        summary["mean_recovery"] >= 0.95
        and summary["median_recovery"] >= 0.97
        and summary["minimum_recovery"] >= 0.85
        and summary["trials_ge_090"] >= 7
    )
    payload = {
        "programme": "SVT-v0.3.1",
        "stage": "A3.1_true_structure_multistart",
        "binding": True,
        "voynich_opened": False,
        "fresh_replicate_offset": 7000,
        "starts_per_trial": 12,
        "selection_rule": "maximum penalised model score; plaintext truth not used for selection",
        "gate_rule": {
            "mean_recovery_ge": 0.95,
            "median_recovery_ge": 0.97,
            "minimum_recovery_ge": 0.85,
            "trials_ge_090_at_least": 7,
        },
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
