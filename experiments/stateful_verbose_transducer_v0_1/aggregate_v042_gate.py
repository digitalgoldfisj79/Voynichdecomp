#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for p in sorted(args.input_dir.rglob("*.json")):
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if row.get("programme") == "SVT-v0.4.2" and row.get("binding") is True:
            rows.append(row)
    if len(rows) != 8:
        raise SystemExit(f"expected 8 binding rows, found {len(rows)}")

    rec = np.asarray([float(r["sequence_recovery"]) for r in rows], dtype=float)
    f1 = np.asarray([float(r["boundary_f1"]) for r in rows], dtype=float)
    cerr = np.asarray([float(r["count_relative_error"]) for r in rows], dtype=float)
    exact = int(sum(bool(r["canonical_exact"]) for r in rows))

    payload = {
        "programme": "SVT-v0.4.2",
        "stage": "joint_semimarkov_segmentation_state_key",
        "binding": True,
        "voynich_opened": False,
        "trials": 8,
        "canonical_exact_count": exact,
        "mean_sequence_recovery": float(rec.mean()),
        "median_sequence_recovery": float(np.median(rec)),
        "minimum_sequence_recovery": float(rec.min()),
        "mean_boundary_f1": float(f1.mean()),
        "minimum_boundary_f1": float(f1.min()),
        "mean_abs_count_relative_error": float(cerr.mean()),
        "per_trial": rows,
    }
    payload["gate_pass"] = bool(
        exact == 8
        and payload["mean_sequence_recovery"] >= 0.90
        and payload["median_sequence_recovery"] >= 0.90
        and payload["minimum_sequence_recovery"] >= 0.85
        and payload["mean_boundary_f1"] >= 0.90
        and payload["minimum_boundary_f1"] >= 0.85
        and payload["mean_abs_count_relative_error"] <= 0.05
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "per_trial"}, indent=2, sort_keys=True))
    if not payload["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
