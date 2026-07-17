#!/usr/bin/env python3
"""Stream the inherited Voynich visual assets without loading hand labels.

This is a data/feasibility audit only. It does not fit or select a hand model.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

FORBIDDEN = re.compile(r"davis|scribe|hand", re.I)
REG_THRESHOLDS = {"inliers": 50, "inlier_ratio": 0.55, "median_reproj_px": 3.0}


def file_sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def neutral_id(kind: str, value: str) -> str:
    digest = hashlib.sha256(f"blind-pal-v1|{kind}|{value}".encode()).hexdigest()[:16]
    return f"{kind}_{digest}"


def stream_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_no}: {exc}") from exc


def audit_registration(path: Path) -> dict[str, Any]:
    total = passed = threshold_pass = 0
    missing = []
    failures = []
    folios = []
    forbidden = set()
    metrics = collections.defaultdict(list)
    for line_no, row in stream_jsonl(path):
        total += 1
        forbidden.update(k for k in row if FORBIDDEN.search(k))
        folio = str(row.get("folio", ""))
        folios.append(folio)
        selected = row.get("selected")
        if not isinstance(selected, dict):
            missing.append({"line": line_no, "folio": neutral_id("folio", folio)})
            continue
        forbidden.update(k for k in selected if FORBIDDEN.search(k))
        passed += int(bool(selected.get("passed")))
        inliers = int(selected.get("inliers", 0) or 0)
        ratio = float(selected.get("inlier_ratio", 0.0) or 0.0)
        median = float(selected.get("median_reproj_px", float("inf")) or float("inf"))
        metrics["inliers"].append(inliers)
        metrics["inlier_ratio"].append(ratio)
        metrics["median_reproj_px"].append(median)
        ok = (
            bool(selected.get("passed"))
            and inliers >= REG_THRESHOLDS["inliers"]
            and ratio >= REG_THRESHOLDS["inlier_ratio"]
            and median <= REG_THRESHOLDS["median_reproj_px"]
        )
        threshold_pass += int(ok)
        if not ok:
            failures.append(
                {
                    "folio": neutral_id("folio", folio),
                    "inliers": inliers,
                    "inlier_ratio": ratio,
                    "median_reproj_px": median,
                    "reported_passed": bool(selected.get("passed")),
                    "reason": selected.get("reason", ""),
                }
            )
    def summary(values):
        if not values:
            return None
        ordered = sorted(values)
        def q(p):
            return ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))]
        return {"min": ordered[0], "p05": q(0.05), "median": q(0.5), "p95": q(0.95), "max": ordered[-1]}
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "rows": total,
        "unique_folios": len(set(folios)),
        "reported_passed": passed,
        "frozen_threshold_passed": threshold_pass,
        "missing_selected_count": len(missing),
        "missing_selected_sample": missing[:25],
        "threshold_failure_count": len(failures),
        "threshold_failure_sample": failures[:100],
        "metric_summary": {k: summary(v) for k, v in metrics.items()},
        "forbidden_field_names": sorted(forbidden),
    }


def audit_manifest(path: Path, sample_limit: int = 100) -> dict[str, Any]:
    total = low_conf = 0
    kinds = collections.Counter()
    views = collections.Counter()
    folios = collections.Counter()
    aligned = collections.Counter()
    word_types = collections.Counter()
    word_lengths = collections.Counter()
    fields = collections.Counter()
    forbidden = collections.Counter()
    blank_ids = 0
    duplicated_key = collections.Counter()
    samples = []
    reserved_count = 0
    for line_no, row in stream_jsonl(path):
        total += 1
        for key in row:
            fields[key] += 1
            if FORBIDDEN.search(key):
                forbidden[key] += 1
        folio = str(row.get("folio", ""))
        kind = str(row.get("kind", ""))
        view = str(row.get("view", ""))
        identifier = str(row.get("id", ""))
        kinds[kind] += 1
        views[view] += 1
        folios[folio] += 1
        low_conf += int(bool(row.get("low_conf")))
        blank_ids += int(not identifier)
        if folio == "f115r":
            reserved_count += 1
        if "eva_aligned" in row:
            aligned[str(row.get("eva_aligned"))] += 1
        if "word" in row:
            word = str(row.get("word", ""))
            word_types[word] += 1
            word_lengths[len(word)] += 1
        duplicated_key[(identifier, view, kind)] += 1
        if len(samples) < sample_limit:
            neutral = {
                "folio_id": neutral_id("folio", folio),
                "unit_id": neutral_id("unit", identifier),
                "kind": kind,
                "view": view,
                "low_conf": bool(row.get("low_conf")),
                "has_alignment": "eva_aligned" in row,
                "word_length": len(str(row.get("word", ""))) if "word" in row else None,
            }
            samples.append(neutral)
    dup = sum(v - 1 for v in duplicated_key.values() if v > 1)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "rows": total,
        "unique_folios": len(folios),
        "folio_row_min": min(folios.values()) if folios else 0,
        "folio_row_median": sorted(folios.values())[len(folios) // 2] if folios else 0,
        "folio_row_max": max(folios.values()) if folios else 0,
        "kinds": dict(kinds.most_common()),
        "views": dict(views.most_common()),
        "low_conf_count": low_conf,
        "low_conf_fraction": low_conf / max(total, 1),
        "aligned_family_count": len(aligned),
        "aligned_rows": sum(aligned.values()),
        "aligned_top20": aligned.most_common(20),
        "word_type_count": len(word_types),
        "word_length_distribution": dict(sorted(word_lengths.items())),
        "blank_id_count": blank_ids,
        "duplicate_id_view_kind_excess": dup,
        "reserved_page_unit_count": reserved_count,
        "field_coverage": dict(fields.most_common()),
        "forbidden_fields": dict(forbidden),
        "neutral_samples": samples,
    }


def audit_crop_paths(path: Path, root: Path) -> dict[str, Any]:
    total = exists = 0
    missing = []
    fields = collections.Counter()
    forbidden = collections.Counter()
    for line_no, row in stream_jsonl(path):
        total += 1
        for key in row:
            fields[key] += 1
            if FORBIDDEN.search(key):
                forbidden[key] += 1
        rel = row.get("path")
        candidate = path.parent / str(rel) if rel else None
        ok = bool(candidate and candidate.is_file())
        exists += int(ok)
        if not ok and len(missing) < 100:
            missing.append(
                {
                    "line": line_no,
                    "unit_id": neutral_id("unit", str(row.get("id", ""))),
                    "relative_path": str(rel),
                }
            )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "rows": total,
        "existing_paths": exists,
        "existing_fraction": exists / max(total, 1),
        "missing_count": total - exists,
        "missing_sample": missing,
        "field_coverage": dict(fields.most_common()),
        "forbidden_fields": dict(forbidden),
    }


def npz_header(path: Path) -> dict[str, Any]:
    import zipfile
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        entries = [
            {"name": i.filename, "compressed_size": i.compress_size, "size": i.file_size}
            for i in z.infolist()
        ]
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "zip_entries": entries,
        "entry_names": names,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("VDINO_ROOT", "/vdino3"))
    ap.add_argument("--output", default="/tmp/voynich_data_audit.json")
    args = ap.parse_args()
    root = Path(args.root)
    reg = root / "register" / "reg_full_225.jsonl"
    manifest = root / "results" / "corpus_crop_manifest.jsonl"
    crop_manifest = root / "crops" / "crop_shard_000" / "crop_manifest.jsonl"
    required = [reg, manifest, crop_manifest]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise SystemExit(f"missing required data assets: {missing}")
    report = {
        "schema": "blind-palaeography-voynich-data-audit-v1",
        "davis_labels_loaded": False,
        "phase1_model_fitted": False,
        "root": str(root),
        "registration": audit_registration(reg),
        "manifest": audit_manifest(manifest),
        "crop_paths": audit_crop_paths(crop_manifest, root),
        "embedding_archives": {},
    }
    for name in ["corpus_embeddings_full.npz", "corpus_embeddings_full_dense.npz", "embeddings.npz"]:
        p = root / "results" / name
        if p.is_file():
            report["embedding_archives"][name] = npz_header(p)
    report["blinding_pass"] = not (
        report["registration"]["forbidden_field_names"]
        or report["manifest"]["forbidden_fields"]
        or report["crop_paths"]["forbidden_fields"]
    )
    report["scientific_blockers"] = []
    if report["registration"]["frozen_threshold_passed"] < report["registration"]["rows"]:
        report["scientific_blockers"].append("registration failures require exclusion/restriction registry")
    if report["crop_paths"]["existing_fraction"] < 0.999:
        report["scientific_blockers"].append("crop-path coverage below 99.9%")
    report["scientific_blockers"].append("physical-bifolium and quire registry not present in inherited visual manifest")
    out = Path(args.output)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("VOYNICH_DATA_AUDIT " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
