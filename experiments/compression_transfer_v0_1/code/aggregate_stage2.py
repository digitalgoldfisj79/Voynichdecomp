#!/usr/bin/env python3
"""Aggregate all frozen Stage-2 mandatory compressor/representation cells."""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import statistics
from pathlib import Path
from typing import Any

COMPRESSORS = ("zlib9", "bz2_9", "lzma9e")
REPRESENTATIONS = (
    "codepoint_u32_ws",
    "surface_utf8",
    "codepoint_u32_nospace",
    "token_recurrence_u32",
    "char_recurrence_u32",
    "token_length_u32",
)
PRIMARY = "codepoint_u32_ws"
RECURRENCE = {"token_recurrence_u32", "char_recurrence_u32"}
GENERATOR_DISJOINT = {
    "family_p",
    "polygraphic_fractionating",
    "structured_generator",
    "matched_null",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def powerset(values: tuple[str, ...], minimum: int):
    for n in range(minimum, len(values) + 1):
        yield from itertools.combinations(values, n)


def select_support(cells: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any] | None:
    passing = {key for key, value in cells.items() if value["cell_support_pass"]}
    candidates = []
    for compressors in powerset(COMPRESSORS, 2):
        for representations in powerset(REPRESENTATIONS, 4):
            if PRIMARY not in representations or not (set(representations) & RECURRENCE):
                continue
            edges = {
                (compressor, representation)
                for compressor in compressors
                for representation in representations
                if (compressor, representation) in passing
            }
            rep_degrees = {
                representation: sum((compressor, representation) in edges for compressor in compressors)
                for representation in representations
            }
            comp_degrees = {
                compressor: sum((compressor, representation) in edges for representation in representations)
                for compressor in compressors
            }
            if min(rep_degrees.values(), default=0) < 2:
                continue
            if min(comp_degrees.values(), default=0) < 4:
                continue
            candidates.append({
                "compressors": list(compressors),
                "representations": list(representations),
                "edges": sorted([list(edge) for edge in edges]),
                "rep_degrees": rep_degrees,
                "compressor_degrees": comp_degrees,
            })
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            -len(row["edges"]),
            -len(row["representations"]),
            -len(row["compressors"]),
            row["compressors"],
            row["representations"],
        )
    )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cell_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-acquisition-freeze", required=True)
    args = parser.parse_args()

    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(args.cell_dir.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("programme") != "compression-transfer-v0.1-stage2-surface-cell":
            continue
        if value["acquisition_freeze_payload_sha256"] != args.expected_acquisition_freeze:
            raise ValueError(f"acquisition mismatch: {path}")
        key = (value["compressor"], value["representation"])
        if key in cells:
            raise ValueError(f"duplicate cell: {key}")
        cells[key] = value
    expected = {(c, r) for c in COMPRESSORS for r in REPRESENTATIONS}
    if set(cells) != expected:
        raise ValueError(f"cell inventory mismatch; missing={sorted(expected-set(cells))}; extra={sorted(set(cells)-expected)}")
    for key, cell in cells.items():
        checks = cell["arithmetic_crosschecks"]
        if checks["passed"] != checks["total"] or checks["total"] == 0:
            raise ValueError(f"arithmetic crosscheck failed: {key}")

    support = select_support(cells)
    all_cell_rows = []
    for (compressor, representation), cell in sorted(cells.items()):
        all_cell_rows.append({
            "compressor": compressor,
            "representation": representation,
            "cell_support_pass": cell["cell_support_pass"],
            "cell_support_gate": cell["cell_support_gate"],
            "metrics": cell["metrics"],
            "scientific_payload_sha256": cell["scientific_payload_sha256"],
        })

    if support is None:
        consensus = {
            "n_units": 0,
            "accepted": 0,
            "coverage": 0.0,
            "accuracy_conditional_on_acceptance": None,
            "macro_accuracy_with_abstention_as_error": 0.0,
            "worst_class_recall": 0.0,
            "generator_disjoint_accuracy": 0.0,
            "matched_null_false_positive_rate": 0.0,
            "per_family_recall": {},
            "rows": [],
        }
    else:
        support_edges = {tuple(edge) for edge in support["edges"]}
        by_unit: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
        targets: dict[tuple[str, int], str] = {}
        for edge in sorted(support_edges):
            cell = cells[edge]
            for row in cell["predictions"]:
                unit = (row["target_document"], int(row["probe_index"]))
                target = row["target_family"]
                if unit in targets and targets[unit] != target:
                    raise ValueError(f"target mismatch for {unit}")
                targets[unit] = target
                by_unit[unit].append(row)

        consensus_rows = []
        for unit in sorted(by_unit):
            voters = by_unit[unit]
            counts = collections.Counter(row["predicted_family"] for row in voters)
            ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
            unique = len(ordered) == 1 or ordered[0][1] > ordered[1][1]
            winner = ordered[0][0] if unique else None
            winner_rows = [row for row in voters if row["predicted_family"] == winner] if winner else []
            compressors = {row["compressor"] for row in winner_rows}
            representations = {row["representation"] for row in winner_rows}
            accepted = bool(
                winner
                and len(compressors) >= 2
                and len(representations) >= 4
                and PRIMARY in representations
                and bool(representations & RECURRENCE)
            )
            target = targets[unit]
            consensus_rows.append({
                "target_document": unit[0],
                "probe_index": unit[1],
                "target_family": target,
                "accepted": accepted,
                "predicted_family": winner if accepted else "ABSTAIN",
                "correct": bool(accepted and winner == target),
                "winning_votes": ordered[0][1] if winner else 0,
                "total_voters": len(voters),
                "supporting_compressors": sorted(compressors),
                "supporting_representations": sorted(representations),
            })

        by_family: dict[str, list[bool]] = collections.defaultdict(list)
        for row in consensus_rows:
            by_family[row["target_family"]].append(bool(row["correct"]))
        recalls = {
            family: sum(values) / len(values)
            for family, values in sorted(by_family.items())
        }
        accepted_rows = [row for row in consensus_rows if row["accepted"]]
        generator_rows = [row for row in consensus_rows if row["target_family"] in GENERATOR_DISJOINT]
        null_rows = [row for row in consensus_rows if row["target_family"] == "matched_null"]
        consensus = {
            "n_units": len(consensus_rows),
            "accepted": len(accepted_rows),
            "coverage": len(accepted_rows) / len(consensus_rows),
            "accuracy_conditional_on_acceptance": (
                sum(row["correct"] for row in accepted_rows) / len(accepted_rows)
                if accepted_rows else None
            ),
            "macro_accuracy_with_abstention_as_error": statistics.mean(recalls.values()),
            "worst_class_recall": min(recalls.values()),
            "generator_disjoint_accuracy": sum(row["correct"] for row in generator_rows) / len(generator_rows),
            "matched_null_false_positive_rate": sum(
                row["accepted"] and row["predicted_family"] != "matched_null"
                for row in null_rows
            ) / len(null_rows),
            "per_family_recall": recalls,
            "rows": consensus_rows,
        }

    formal_gates = {
        "cross_cell_support": support is not None,
        "consensus_macro_accuracy": consensus["macro_accuracy_with_abstention_as_error"] >= 0.80,
        "consensus_worst_class_recall": consensus["worst_class_recall"] >= 0.60,
        "consensus_coverage": consensus["coverage"] >= 0.75,
        "consensus_generator_disjoint_accuracy": consensus["generator_disjoint_accuracy"] >= 0.75,
        "consensus_matched_null_false_positive_rate": consensus["matched_null_false_positive_rate"] <= 0.05,
    }
    decision = "STAGE2_SURFACE_PASS" if all(formal_gates.values()) else "STAGE2_SURFACE_FAIL"
    result = {
        "programme": "compression-transfer-v0.1-stage2-surface",
        "acquisition_freeze_payload_sha256": args.expected_acquisition_freeze,
        "cell_inventory": all_cell_rows,
        "qualifying_cross_cell_support": support,
        "consensus": consensus,
        "formal_gates": formal_gates,
        "decision": decision,
        "voynich_loaded_or_scored": False,
        "source_language_use": "CLOSED_BY_STAGE1_FAIL",
    }
    result["scientific_payload_sha256"] = hashlib.sha256(canonical_json(result)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("STAGE2_FINAL_ONE_LINE=" + json.dumps({
        "decision": decision,
        "formal_gates": formal_gates,
        "qualifying_cross_cell_support": support,
        "consensus": {k: v for k, v in consensus.items() if k != "rows"},
        "cells": all_cell_rows,
        "scientific_payload_sha256": result["scientific_payload_sha256"],
    }, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
