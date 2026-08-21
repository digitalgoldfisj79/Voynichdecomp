#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

def load_rows(root: Path, stage: str):
    rows = []
    for p in root.rglob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("programme") == "SVT-v0.5.1-lattice-coupling" and d.get("stage") == stage:
            rows.append(d)
    return rows

def mean(xs):
    return float(statistics.mean(xs)) if xs else 0.0

def median(xs):
    return float(statistics.median(xs)) if xs else 0.0

def summarise(rows):
    return {
        "n": len(rows),
        "exact_structure": sum(bool(r["exact_structure_eval_only"]) for r in rows),
        "mean_boundary_f1": mean([r["boundary_f1_eval_only"] for r in rows]),
        "median_boundary_f1": median([r["boundary_f1_eval_only"] for r in rows]),
        "min_boundary_f1": min([r["boundary_f1_eval_only"] for r in rows], default=0.0),
        "mean_map_boundary_f1": mean([r["surface_map_boundary_f1_eval_only"] for r in rows]),
        "mean_oracle_boundary_f1": mean([r["lattice_oracle_boundary_f1_eval_only"] for r in rows]),
        "mean_abs_count_error": mean([r["count_abs_error_eval_only"] for r in rows]),
        "mean_signed_count_error": mean([r["count_signed_error_eval_only"] for r in rows]),
        "mean_abs_map_count_shift": mean([abs(r["selected_vs_surface_map_count_shift"]) for r in rows]),
        "mean_sequence_recovery": mean([r["sequence_recovery_eval_only"] for r in rows]),
        "median_sequence_recovery": median([r["sequence_recovery_eval_only"] for r in rows]),
        "min_sequence_recovery": min([r["sequence_recovery_eval_only"] for r in rows], default=0.0),
        "mean_selected_cipher_z": mean([r["selected_cipher_z"] for r in rows]),
        "reranked_from_map": sum(int(r["selected_surface_rank"]) != 0 for r in rows),
    }

def gate_dev(rows, s):
    return {
        "n_eq_8": len(rows) == 8,
        "mean_boundary_f1_ge_0_90": s["mean_boundary_f1"] >= 0.90,
        "min_boundary_f1_ge_0_85": s["min_boundary_f1"] >= 0.85,
        "mean_abs_count_error_le_0_05": s["mean_abs_count_error"] <= 0.05,
        "abs_mean_signed_count_error_le_0_03": abs(s["mean_signed_count_error"]) <= 0.03,
        "mean_abs_map_count_shift_le_0_03": s["mean_abs_map_count_shift"] <= 0.03,
        "exact_structure_ge_6_of_8": s["exact_structure"] >= 6,
        "mean_sequence_recovery_ge_0_85": s["mean_sequence_recovery"] >= 0.85,
        "min_sequence_recovery_ge_0_70": s["min_sequence_recovery"] >= 0.70,
    }

def gate_binding(rows, overall, by_iso):
    checks = {
        "n_eq_16": len(rows) == 16,
        "exact_structure_16_of_16": overall["exact_structure"] == 16,
        "all_boundary_f1_ge_0_85": all(r["boundary_f1_eval_only"] >= 0.85 for r in rows),
        "all_sequence_recovery_ge_0_85": all(r["sequence_recovery_eval_only"] >= 0.85 for r in rows),
    }
    for iso in ("de", "la"):
        s = by_iso.get(iso, {})
        checks.update({
            f"{iso}_n_eq_8": s.get("n") == 8,
            f"{iso}_exact_8_of_8": s.get("exact_structure") == 8,
            f"{iso}_mean_boundary_f1_ge_0_90": s.get("mean_boundary_f1", 0.0) >= 0.90,
            f"{iso}_mean_abs_count_error_le_0_05": s.get("mean_abs_count_error", 1.0) <= 0.05,
            f"{iso}_abs_mean_signed_count_error_le_0_03": abs(s.get("mean_signed_count_error", 1.0)) <= 0.03,
            f"{iso}_mean_abs_map_count_shift_le_0_03": s.get("mean_abs_map_count_shift", 1.0) <= 0.03,
            f"{iso}_mean_sequence_recovery_ge_0_90": s.get("mean_sequence_recovery", 0.0) >= 0.90,
            f"{iso}_median_sequence_recovery_ge_0_90": s.get("median_sequence_recovery", 0.0) >= 0.90,
        })
    return checks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--stage", choices=("dev", "binding"), required=True)
    args = ap.parse_args()

    rows = load_rows(args.input_dir, args.stage)
    rows.sort(key=lambda r: (r["iso"], r["generator_mode"], r["replicate"]))
    overall = summarise(rows)
    by_iso = {iso: summarise([r for r in rows if r["iso"] == iso]) for iso in ("de", "la")}
    checks = gate_dev(rows, overall) if args.stage == "dev" else gate_binding(rows, overall, by_iso)
    gate_pass = bool(checks) and all(checks.values())
    payload = {
        "programme": "SVT-v0.5.1-lattice-coupling",
        "stage": args.stage,
        "voynich_opened": False,
        "gate_pass": gate_pass,
        "checks": checks,
        "overall": overall,
        "by_iso": by_iso,
        "trials": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"stage": args.stage, "gate_pass": gate_pass, "checks": checks, "overall": overall, "by_iso": by_iso}, indent=2, sort_keys=True))
    if not gate_pass:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
