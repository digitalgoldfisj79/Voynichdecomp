#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(str(args.input_root / "**" / "*.json"), recursive=True)):
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("programme") == "SVT-v0.4.1" and d.get("binding") is True:
            rows.append(d)

    if len(rows) != 8:
        raise SystemExit(f"expected exactly 8 binding v0.4.1 rows, found {len(rows)}")

    f1 = [float(r["boundary_f1"]) for r in rows]
    count_err = [float(r["count_relative_error"]) for r in rows]
    recovery = [float(r["end_to_end_edit_recovery_posthoc"]) for r in rows]
    exact = [bool(r["canonical_exact_posthoc"]) for r in rows]

    segmentation_gate = bool(
        statistics.fmean(f1) >= 0.90
        and statistics.median(f1) >= 0.90
        and min(f1) >= 0.85
        and sum(x >= 0.85 for x in f1) == 8
        and statistics.fmean(count_err) <= 0.05
    )
    plaintext_gate = bool(
        statistics.fmean(recovery) >= 0.75
        and statistics.median(recovery) >= 0.85
        and sum(x >= 0.70 for x in recovery) >= 6
    )
    structure_gate = bool(sum(exact) >= 6)
    passed = bool(segmentation_gate and plaintext_gate and structure_gate)

    result = {
        "programme": "SVT-v0.4.1",
        "stage": "aggregate_binding_gate",
        "binding": True,
        "voynich_opened": False,
        "trials": len(rows),
        "segmentation": {
            "mean_boundary_f1": statistics.fmean(f1),
            "median_boundary_f1": statistics.median(f1),
            "minimum_boundary_f1": min(f1),
            "at_least_085": sum(x >= 0.85 for x in f1),
            "mean_count_relative_error": statistics.fmean(count_err),
            "gate_pass": segmentation_gate,
        },
        "plaintext": {
            "mean_edit_recovery": statistics.fmean(recovery),
            "median_edit_recovery": statistics.median(recovery),
            "minimum_edit_recovery": min(recovery),
            "at_least_070": sum(x >= 0.70 for x in recovery),
            "gate_pass": plaintext_gate,
        },
        "structure": {
            "canonical_exact": sum(exact),
            "accuracy": statistics.fmean(exact),
            "gate_pass": structure_gate,
        },
        "pass": passed,
        "verdict": "PASS" if passed else "FAIL",
        "rows": [{
            "replicate": int(r["replicate"]),
            "true_mode": r["true_mode"],
            "true_period": int(r["true_period"]),
            "boundary_f1": float(r["boundary_f1"]),
            "count_relative_error": float(r["count_relative_error"]),
            "screen_truth_rank": int(r["screen_truth_rank_posthoc"]),
            "canonical_mode": r["canonical"]["canonical_mode"],
            "canonical_period": int(r["canonical"]["canonical_period"]),
            "canonical_exact": bool(r["canonical_exact_posthoc"]),
            "edit_recovery": float(r["end_to_end_edit_recovery_posthoc"]),
        } for r in sorted(rows, key=lambda x: (x["true_mode"], x["replicate"]))],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))

    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
