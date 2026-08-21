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

    rows = []
    for path in sorted(args.input_dir.rglob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if row.get("programme") == "SVT-v0.3.4":
            rows.append(row)

    ordinary = [r for r in rows if r.get("arm") == "ordinary"]
    harmonic = [r for r in rows if r.get("arm") == "harmonic"]

    ordinary_rec = [float(r["canonical"]["canonical_recovery"]) for r in ordinary]
    harmonic_rec = [float(r["canonical"]["canonical_recovery"]) for r in harmonic]

    arm_a = {
        "trials": len(ordinary),
        "exact_structure_correct": sum(bool(r.get("canonical_exact")) for r in ordinary),
        "mean_recovery": statistics.mean(ordinary_rec) if ordinary_rec else 0.0,
        "median_recovery": statistics.median(ordinary_rec) if ordinary_rec else 0.0,
        "minimum_recovery": min(ordinary_rec) if ordinary_rec else 0.0,
        "maximum_recovery": max(ordinary_rec) if ordinary_rec else 0.0,
        "trials_ge_090": sum(x >= 0.90 for x in ordinary_rec),
        "screen_truth_in_top6": sum(int(r.get("screen_truth_rank", 999)) <= 6 for r in ordinary),
    }
    arm_a["pass"] = bool(
        arm_a["trials"] == 8
        and arm_a["exact_structure_correct"] == 8
        and arm_a["mean_recovery"] >= 0.95
        and arm_a["median_recovery"] >= 0.97
        and arm_a["minimum_recovery"] >= 0.85
        and arm_a["trials_ge_090"] == 8
    )

    arm_b = {
        "trials": len(harmonic),
        "exact_primitive_correct": sum(bool(r.get("canonical_exact")) for r in harmonic),
        "mean_recovery": statistics.mean(harmonic_rec) if harmonic_rec else 0.0,
        "minimum_recovery": min(harmonic_rec) if harmonic_rec else 0.0,
        "maximum_recovery": max(harmonic_rec) if harmonic_rec else 0.0,
    }
    arm_b["pass"] = bool(
        arm_b["trials"] == 12
        and arm_b["exact_primitive_correct"] == 12
        and arm_b["mean_recovery"] >= 0.95
        and arm_b["minimum_recovery"] >= 0.90
    )

    payload = {
        "programme": "SVT-v0.3.4",
        "binding": True,
        "voynich_opened": False,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "gate_pass": bool(arm_a["pass"] and arm_b["pass"]),
        "ordinary_trials": ordinary,
        "harmonic_trials": harmonic,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "arm_a": arm_a,
        "arm_b": arm_b,
        "gate_pass": payload["gate_pass"],
    }, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["gate_pass"] else 2)


if __name__ == "__main__":
    main()
