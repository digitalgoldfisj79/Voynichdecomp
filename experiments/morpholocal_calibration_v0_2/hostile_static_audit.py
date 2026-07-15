#!/usr/bin/env python3
"""Hostile post-run audit for morpholocal calibration v0.2.

This audit is deliberately separate from the calibration implementation. It
recomputes every declared gate from per-trial records, validates frozen bundle
hashes and seed schedules, and records provenance weaknesses without changing
the frozen scientific result.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

EXPECTED_HASH = "c12c48d5585dd4efc5935d29ca2eae46df3c1dabd6475ed89ae6eb7a3c0b1705"
EXPECTED = {"formal_seed": 8675309, "positives": 96, "controls": 320, "anneal_steps": 400, "anneal_restarts": 2}
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
POSITIVE_DIMS = ("external_profile", "key_scheme", "null_count", "selection_policy", "selector", "size_profile")
CONTROL_FAMILIES = {"cell_markov", "context_iid", "copy_mutate", "permuted_cipher"}


def wilson90(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    z = 1.6448536269514722
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    r = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return max(0.0, c - r), min(1.0, c + r)


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def add(findings: list[dict], severity: str, code: str, message: str) -> None:
    findings.append({"severity": severity, "code": code, "message": message})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path)
    ap.add_argument("result", type=Path)
    args = ap.parse_args()
    root = args.repo.resolve()
    exp = root / "experiments/morpholocal_calibration_v0_2"
    findings: list[dict] = []
    errors: list[str] = []

    raw = args.result.read_bytes()
    result_hash = hashlib.sha256(raw).hexdigest()
    if result_hash != EXPECTED_HASH:
        errors.append(f"result SHA-256 mismatch: {result_hash}")
    payload = json.loads(raw)
    summary = payload.get("summary", {})
    params = payload.get("parameters", {})

    for key, expected in EXPECTED.items():
        observed = payload.get("formal_seed") if key == "formal_seed" else params.get(key)
        if observed != expected:
            errors.append(f"parameter {key}: expected {expected!r}, got {observed!r}")

    rows = payload.get("results", [])
    positives = [r for r in rows if r.get("trial_type") == "positive"]
    controls = [r for r in rows if r.get("trial_type") == "control"]
    if (len(rows), len(positives), len(controls)) != (416, 96, 320):
        errors.append(f"trial counts {(len(rows), len(positives), len(controls))}")
    for label, subset, n in (("positive", positives, 96), ("control", controls, 320)):
        indices = sorted(int(r.get("trial_index", -1)) for r in subset)
        if indices != list(range(n)):
            errors.append(f"{label} indices incomplete or duplicated")

    for r in positives:
        i = int(r["trial_index"])
        expected_seed = EXPECTED["formal_seed"] + 100000 + i * 7919
        if int(r.get("seed", -1)) != expected_seed:
            errors.append(f"positive seed mismatch at {i}")
    for r in controls:
        i = int(r["trial_index"])
        expected_seed = EXPECTED["formal_seed"] + 900000 + i * 104729
        if int(r.get("seed", -1)) != expected_seed:
            errors.append(f"control seed mismatch at {i}")

    successes = sum(bool(r.get("positive_success")) for r in positives)
    fps = sum(bool(r.get("false_positive")) for r in controls)
    mapping = statistics.median(float(r.get("mapping_accuracy", 0.0)) for r in positives)
    null_f1 = statistics.median(float(r.get("null_f1", 0.0)) for r in positives)
    selector = sum(bool(r.get("selector_correct")) for r in positives) / 96
    structure = sum(bool(r.get("structure_correct")) for r in positives) / 96
    pos_ci = wilson90(successes, 96)
    ctrl_ci = wilson90(fps, 320)

    strata: dict[str, dict[str, tuple[int, int, tuple[float, float]]]] = {}
    all_positive_strata_ok = True
    for dim in POSITIVE_DIMS:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for r in positives:
            buckets[str(r.get(dim))].append(r)
        strata[dim] = {}
        for value, subset in sorted(buckets.items()):
            k = sum(bool(r.get("positive_success")) for r in subset)
            ci = wilson90(k, len(subset))
            strata[dim][value] = (k, len(subset), ci)
            all_positive_strata_ok &= ci[0] >= THRESHOLDS["positive_stratum_lower"]
            recorded = summary.get("positive_strata", {}).get(dim, {}).get(value, {})
            if recorded.get("successes") != k or recorded.get("trials") != len(subset):
                errors.append(f"positive stratum count mismatch {dim}={value}")
            rec_ci = recorded.get("wilson90", [])
            if len(rec_ci) != 2 or not all(close(a, b) for a, b in zip(ci, rec_ci)):
                errors.append(f"positive stratum CI mismatch {dim}={value}")

    family_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in controls:
        f = str(r.get("control_family"))
        family_counts[f][1] += 1
        family_counts[f][0] += int(bool(r.get("false_positive")))
    if set(family_counts) != CONTROL_FAMILIES:
        errors.append(f"control families mismatch: {sorted(family_counts)}")
    all_control_families_ok = True
    for f, (k, n) in family_counts.items():
        all_control_families_ok &= n == 80 and wilson90(k, n)[1] <= THRESHOLDS["control_family_upper"]

    recomputed = {
        "overall_positive_lower_ge_0_70": pos_ci[0] >= THRESHOLDS["positive_lower"],
        "all_positive_strata_lower_ge_0_50": all_positive_strata_ok,
        "overall_false_positive_upper_le_0_05": ctrl_ci[1] <= THRESHOLDS["false_positive_upper"],
        "all_control_families_upper_le_0_10": all_control_families_ok,
        "median_mapping_accuracy_ge_0_60": mapping >= THRESHOLDS["mapping"],
        "median_null_f1_ge_0_75": null_f1 >= THRESHOLDS["null_f1"],
        "selector_recovery_ge_0_80": selector >= THRESHOLDS["selector"],
        "structure_recovery_ge_0_65": structure >= THRESHOLDS["structure"],
    }
    if summary.get("criteria") != recomputed:
        errors.append(f"criteria mismatch: recorded={summary.get('criteria')} recomputed={recomputed}")
    expected_verdict = "PASS_MORPHOLOCAL_CLASS_CALIBRATION" if all(recomputed.values()) else "FAIL_MORPHOLOCAL_CLASS_CALIBRATION"
    if summary.get("gate_verdict") != expected_verdict:
        errors.append(f"verdict mismatch: {summary.get('gate_verdict')} vs {expected_verdict}")

    freeze = json.loads((exp / "FREEZE_RECORD.json").read_text())
    manifest = json.loads((exp / "SOURCE_BUNDLE_MANIFEST.json").read_text())
    if freeze.get("source_raw_sha256") != manifest.get("raw_sha256"):
        errors.append("freeze raw-source hash does not match source manifest")
    if freeze.get("source_bundle_sha256") != manifest.get("gzip_base64_sha256"):
        errors.append("freeze bundle hash does not match source manifest")
    encoded_parts = []
    for part in manifest.get("parts", []):
        text = (exp / part["path"]).read_text(encoding="ascii").strip()
        if hashlib.sha256(text.encode("ascii")).hexdigest() != part["sha256"]:
            errors.append(f"source part hash mismatch: {part['path']}")
        encoded_parts.append(text)
    encoded = "".join(encoded_parts)
    if hashlib.sha256(encoded.encode("ascii")).hexdigest() != manifest.get("gzip_base64_sha256"):
        errors.append("reconstructed encoded bundle hash mismatch")
    source_raw = gzip.decompress(base64.b64decode(encoded))
    if hashlib.sha256(source_raw).hexdigest() != manifest.get("raw_sha256"):
        errors.append("reconstructed raw source hash mismatch")

    # Hostile provenance checks.
    frozen_names = {"apply_development_patch.py", "gpu_runner.py", "cpu_batched_runner.py"}
    freeze_text = json.dumps(freeze, sort_keys=True)
    missing_effective_hashes = [name for name in sorted(frozen_names) if name not in freeze_text]
    if missing_effective_hashes:
        add(findings, "HIGH", "INCOMPLETE_EFFECTIVE_SOURCE_FREEZE",
            "FREEZE_RECORD hashes the unpatched source bundle but not the patcher or accelerated runners that define the effective formal implementation: " + ", ".join(missing_effective_hashes))

    audit_text = (exp / "formal_audit.py").read_text(encoding="utf-8")
    if "all_positive_strata_lower_ge_0_50" not in audit_text:
        add(findings, "MEDIUM", "FIRST_AUDIT_OMITS_POSITIVE_STRATA_GATE",
            "formal_audit.py does not independently recompute the declared all-positive-strata gate.")
    if '!= "FAIL_MORPHOLOCAL_CLASS_CALIBRATION"' in audit_text:
        add(findings, "MEDIUM", "FIRST_AUDIT_HARDCODES_OUTCOME",
            "formal_audit.py hard-codes the expected failure verdict instead of deriving pass/fail generically from all gates.")

    if payload.get("development_accelerator"):
        add(findings, "LOW", "FORMAL_ARTIFACT_DEVELOPMENT_LABEL",
            "The formal result is labelled with a development_accelerator field; scientifically harmless, but provenance naming is misleading.")

    py_files = list(exp.glob("*.py"))
    executable_manuscript_refs = []
    for path in py_files:
        if path.name == "hostile_static_audit.py":
            continue
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            code = line.strip().lower()
            if code.startswith("#"):
                continue
            if "manuscript" in code and not code.startswith(('"""', "'''")):
                executable_manuscript_refs.append(f"{path.name}:{lineno}")
    if executable_manuscript_refs:
        add(findings, "MEDIUM", "MANUSCRIPT_EXECUTION_REFERENCES_PRESENT",
            "Executable manuscript references found: " + ", ".join(executable_manuscript_refs[:20]))
    else:
        add(findings, "INFO", "NO_MANUSCRIPT_APPLICATION_ENTRYPOINT",
            "No executable manuscript-application entry point was found in the calibration directory; the failed gate was not followed by manuscript inference.")

    spec = (exp / "SPEC.md").read_text(encoding="utf-8")
    if "prohibits manuscript analysis" not in spec.lower():
        errors.append("SPEC does not state manuscript prohibition")
    add(findings, "LOW", "PROCEDURAL_NOT_EXECUTABLE_INTERLOCK",
        "The manuscript prohibition is documented but not enforced by a cryptographic or executable interlock. No manuscript entry point exists here, so this did not affect the run.")

    status = "FAIL" if errors else ("PASS_WITH_FINDINGS" if any(f["severity"] != "INFO" for f in findings) else "PASS")
    report = {
        "audit_status": status,
        "scientific_result_validated": not errors,
        "result_sha256": result_hash,
        "recomputed_verdict": expected_verdict,
        "recomputed_criteria": recomputed,
        "positive_successes": successes,
        "false_positives": fps,
        "positive_wilson90": pos_ci,
        "control_wilson90": ctrl_ci,
        "median_mapping_accuracy": mapping,
        "median_null_f1": null_f1,
        "selector_recovery": selector,
        "structure_recovery": structure,
        "errors": errors,
        "findings": findings,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
