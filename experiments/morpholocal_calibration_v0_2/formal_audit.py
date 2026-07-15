#!/usr/bin/env python3
"""Independent standard-library audit of a formal calibration JSON result."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED = {
    "formal_seed": 8675309,
    "positives": 96,
    "controls": 320,
    "anneal_steps": 400,
    "anneal_restarts": 2,
}
THRESHOLDS = {
    "positive_lower": 0.70,
    "positive_stratum_lower": 0.50,
    "false_positive_upper": 0.05,
    "control_family_upper": 0.10,
    "mapping": 0.60,
    "null_f1": 0.75,
    "selector": 0.80,
    "structure": 0.65,
}


def wilson90(successes: int, trials: int) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 1.0
    z = 1.6448536269514722
    p = successes / trials
    d = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / d
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / d
    return max(0.0, centre - radius), min(1.0, centre + radius)


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result", type=Path)
    args = ap.parse_args()
    raw = args.result.read_bytes()
    payload = json.loads(raw)
    errors: list[str] = []

    params = payload.get("parameters", {})
    for key, expected in EXPECTED.items():
        observed = payload.get("formal_seed") if key == "formal_seed" else params.get(key)
        if observed != expected:
            errors.append(f"parameter {key}: expected {expected!r}, got {observed!r}")

    rows = payload.get("results", [])
    positives = [r for r in rows if r.get("trial_type") == "positive"]
    controls = [r for r in rows if r.get("trial_type") == "control"]
    if len(rows) != 416 or len(positives) != 96 or len(controls) != 320:
        errors.append(f"trial counts: total={len(rows)} positives={len(positives)} controls={len(controls)}")
    for label, subset, expected_n in (("positive", positives, 96), ("control", controls, 320)):
        indices = [int(r["trial_index"]) for r in subset]
        if sorted(indices) != list(range(expected_n)):
            errors.append(f"{label} indices are incomplete or duplicated")

    def finite_row(row: dict) -> bool:
        for value in row.values():
            if isinstance(value, float) and not math.isfinite(value):
                return False
        return True
    if not all(finite_row(r) for r in rows):
        errors.append("non-finite value found in result rows")

    successes = sum(bool(r.get("positive_success")) for r in positives)
    false_positives = sum(bool(r.get("false_positive")) for r in controls)
    mapping = statistics.median(float(r.get("mapping_accuracy", 0.0)) for r in positives)
    null_f1 = statistics.median(float(r.get("null_f1", 0.0)) for r in positives)
    selector = sum(bool(r.get("selector_correct")) for r in positives) / 96
    structure = sum(bool(r.get("structure_correct")) for r in positives) / 96
    pos_ci = wilson90(successes, 96)
    ctrl_ci = wilson90(false_positives, 320)

    summary = payload.get("summary", {})
    sp = summary.get("positive", {})
    sc = summary.get("control", {})
    checks = [
        (successes == sp.get("successes"), "positive success count"),
        (false_positives == sc.get("false_positives"), "false-positive count"),
        (close(mapping, sp.get("median_mapping_accuracy", float("nan"))), "median mapping"),
        (close(null_f1, sp.get("median_null_f1", float("nan"))), "median null F1"),
        (close(selector, sp.get("selector_recovery", float("nan"))), "selector recovery"),
        (close(structure, sp.get("structure_recovery", float("nan"))), "structure recovery"),
        (all(close(a, b) for a, b in zip(pos_ci, sp.get("wilson90", []))), "positive Wilson interval"),
        (all(close(a, b) for a, b in zip(ctrl_ci, sc.get("wilson90", []))), "control Wilson interval"),
    ]
    for ok, label in checks:
        if not ok:
            errors.append(f"summary mismatch: {label}")

    family_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in controls:
        family = str(row.get("control_family"))
        family_counts[family][1] += 1
        family_counts[family][0] += int(bool(row.get("false_positive")))
    expected_families = {"cell_markov", "context_iid", "copy_mutate", "permuted_cipher"}
    if set(family_counts) != expected_families:
        errors.append(f"control families: {sorted(family_counts)}")
    family_upper_ok = True
    for family, (fp, n) in family_counts.items():
        if n != 80:
            errors.append(f"control family {family} has {n} trials")
        family_upper_ok &= wilson90(fp, n)[1] <= THRESHOLDS["control_family_upper"]

    criteria = {
        "overall_positive_lower_ge_0_70": pos_ci[0] >= THRESHOLDS["positive_lower"],
        "overall_false_positive_upper_le_0_05": ctrl_ci[1] <= THRESHOLDS["false_positive_upper"],
        "all_control_families_upper_le_0_10": family_upper_ok,
        "median_mapping_accuracy_ge_0_60": mapping >= THRESHOLDS["mapping"],
        "median_null_f1_ge_0_75": null_f1 >= THRESHOLDS["null_f1"],
        "selector_recovery_ge_0_80": selector >= THRESHOLDS["selector"],
        "structure_recovery_ge_0_65": structure >= THRESHOLDS["structure"],
    }
    for key, value in criteria.items():
        if summary.get("criteria", {}).get(key) is not value:
            errors.append(f"criterion mismatch: {key}")

    if summary.get("gate_verdict") != "FAIL_MORPHOLOCAL_CLASS_CALIBRATION":
        errors.append(f"unexpected verdict: {summary.get('gate_verdict')!r}")

    report = {
        "audit": "PASS" if not errors else "FAIL",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "trials": len(rows),
        "positive_successes": successes,
        "false_positives": false_positives,
        "median_mapping_accuracy": mapping,
        "median_null_f1": null_f1,
        "selector_recovery": selector,
        "structure_recovery": structure,
        "recomputed_positive_wilson90": pos_ci,
        "recomputed_control_wilson90": ctrl_ci,
        "errors": errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
