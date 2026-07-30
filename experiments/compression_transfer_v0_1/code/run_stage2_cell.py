#!/usr/bin/env python3
"""Evaluate one frozen Stage-2 compressor x representation cell.

The implementation is arithmetic-equivalent to run_benchmark.py but caches
C(reference || boundary), which avoids repeated baseline compression. A fixed
sample is recomputed through conditional_bits_per_byte as an independent
arithmetic check.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from compression_metrics import (
    BOUNDARY,
    compressed_size,
    conditional_bits_per_byte,
    normalized_compression_distance,
)
from manifest import DocumentRecord, load_manifest
from representations import chunk_text, encode_representation
from run_benchmark import build_reference, read_text

GENERATOR_DISJOINT_FAMILIES = {
    "family_p",
    "polygraphic_fractionating",
    "structured_generator",
    "matched_null",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--acquisition-freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-payload", required=True)
    parser.add_argument("--compressor", required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-units", type=int, default=4096)
    parser.add_argument("--probe-stride", type=int, default=4096)
    parser.add_argument("--reference-units", type=int, default=131072)
    parser.add_argument("--max-probes-per-document", type=int, default=8)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    acquisition = json.loads(args.acquisition_freeze.read_text(encoding="utf-8"))
    if acquisition["freeze_payload_sha256"] != args.expected_freeze_payload:
        raise ValueError("Stage-2 acquisition freeze mismatch")
    if acquisition["qualification"]["status"] != "PASS":
        raise ValueError("Stage-2 acquisition did not qualify")
    if acquisition["manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError("Stage-2 manifest hash mismatch")

    rows = load_manifest(manifest_path, verify_hashes=True)
    train: dict[str, list[DocumentRecord]] = collections.defaultdict(list)
    test: list[DocumentRecord] = []
    for row in rows:
        if row.split == "train":
            train[row.corpus_id].append(row)
        elif row.split == "test":
            test.append(row)
    families = sorted(train)
    if set(families) != set(acquisition["qualification"]["documents_per_family"]):
        raise ValueError("family inventory mismatch")

    refs = {
        family: build_reference(train[family], args.representation, args.reference_units)[0]
        for family in families
    }
    baseline = {
        family: compressed_size(refs[family] + BOUNDARY, args.compressor)
        for family in families
    }

    predictions: list[dict[str, Any]] = []
    arithmetic_checks: list[bool] = []
    for doc in sorted(test, key=lambda r: (r.corpus_id, r.document_id)):
        chunks = chunk_text(
            read_text(doc), args.representation,
            args.probe_units, args.probe_stride,
        )[:args.max_probes_per_document]
        if not chunks:
            raise ValueError(f"no formal probe for {doc.document_id}/{args.representation}")
        for probe_index, chunk in enumerate(chunks):
            probe = encode_representation(chunk, args.representation)
            costs = {
                family: 8.0 * (
                    compressed_size(refs[family] + BOUNDARY + probe, args.compressor)
                    - baseline[family]
                ) / len(probe)
                for family in families
            }
            ordered = sorted(families, key=lambda family: (costs[family], family))
            winner = ordered[0]
            margin = costs[ordered[1]] - costs[ordered[0]]
            predictions.append({
                "compressor": args.compressor,
                "representation": args.representation,
                "target_family": doc.corpus_id,
                "target_document": doc.document_id,
                "probe_index": probe_index,
                "probe_sha256": hashlib.sha256(probe).hexdigest(),
                "probe_bytes": len(probe),
                "predicted_family": winner,
                "correct": winner == doc.corpus_id,
                "own_rank": ordered.index(doc.corpus_id) + 1,
                "winner_margin_bits_per_byte": margin,
                "winner_cost_bits_per_byte": costs[winner],
                "own_cost_bits_per_byte": costs[doc.corpus_id],
            })
            if len(arithmetic_checks) < 64:
                family = families[len(arithmetic_checks) % len(families)]
                exact = conditional_bits_per_byte(refs[family], probe, args.compressor)
                arithmetic_checks.append(abs(exact - costs[family]) <= 1e-12)

    by_family: dict[str, list[bool]] = collections.defaultdict(list)
    ranks: list[int] = []
    for row in predictions:
        by_family[row["target_family"]].append(bool(row["correct"]))
        ranks.append(int(row["own_rank"]))
    recalls = {
        family: sum(values) / len(values)
        for family, values in sorted(by_family.items())
    }
    macro = statistics.mean(recalls.values())
    generator_rows = [
        row for row in predictions
        if row["target_family"] in GENERATOR_DISJOINT_FAMILIES
    ]
    null_rows = [row for row in predictions if row["target_family"] == "matched_null"]

    ncd_gaps: list[float] = []
    ncd_symmetric: list[dict[str, Any]] = []
    for i, left in enumerate(families):
        for right in families[i:]:
            forward, reverse, symmetric = normalized_compression_distance(
                refs[left], refs[right], args.compressor,
            )
            gap = abs(forward - reverse)
            ncd_gaps.append(gap)
            ncd_symmetric.append({
                "left": left,
                "right": right,
                "forward": forward,
                "reverse": reverse,
                "symmetric": symmetric,
                "order_gap": gap,
            })

    metrics = {
        "n_probes": len(predictions),
        "top1_accuracy": sum(bool(r["correct"]) for r in predictions) / len(predictions),
        "macro_accuracy": macro,
        "worst_class_recall": min(recalls.values()),
        "per_family_recall": recalls,
        "median_own_rank": statistics.median(ranks),
        "mean_own_rank": statistics.mean(ranks),
        "generator_disjoint_accuracy": sum(bool(r["correct"]) for r in generator_rows) / len(generator_rows),
        "matched_null_false_positive_rate": sum(r["predicted_family"] != "matched_null" for r in null_rows) / len(null_rows),
        "median_ncd_order_gap": statistics.median(ncd_gaps),
        "max_ncd_order_gap": max(ncd_gaps),
    }
    support_gate = {
        "macro_accuracy": metrics["macro_accuracy"] >= 0.80,
        "worst_class_recall": metrics["worst_class_recall"] >= 0.60,
        "generator_disjoint_accuracy": metrics["generator_disjoint_accuracy"] >= 0.75,
        "matched_null_false_positive_rate": metrics["matched_null_false_positive_rate"] <= 0.05,
        "median_own_rank": metrics["median_own_rank"] <= 1,
        "median_ncd_order_gap": metrics["median_ncd_order_gap"] <= 0.05,
    }
    result = {
        "programme": "compression-transfer-v0.1-stage2-surface-cell",
        "acquisition_freeze_payload_sha256": args.expected_freeze_payload,
        "manifest_sha256": sha256_file(manifest_path),
        "compressor": args.compressor,
        "representation": args.representation,
        "probe_units": args.probe_units,
        "probe_stride": args.probe_stride,
        "reference_units": args.reference_units,
        "max_probes_per_document": args.max_probes_per_document,
        "arithmetic_crosschecks": {
            "passed": sum(arithmetic_checks),
            "total": len(arithmetic_checks),
        },
        "metrics": metrics,
        "cell_support_gate": support_gate,
        "cell_support_pass": all(support_gate.values()),
        "predictions": predictions,
        "ncd_pairs": ncd_symmetric,
    }
    result["scientific_payload_sha256"] = hashlib.sha256(canonical_json(result)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "compressor": args.compressor,
        "representation": args.representation,
        "metrics": metrics,
        "cell_support_gate": support_gate,
        "cell_support_pass": result["cell_support_pass"],
        "arithmetic_crosschecks": result["arithmetic_crosschecks"],
        "scientific_payload_sha256": result["scientific_payload_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
