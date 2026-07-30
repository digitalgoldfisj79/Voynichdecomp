#!/usr/bin/env python3
"""Run the frozen directional compression-transfer benchmark."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from compression_metrics import BOUNDARY, available_compressors, compressed_size, compressor_spec, conditional_bits_per_byte, directional_excess_bits_per_byte, normalized_compression_distance, sha256_bytes
from manifest import DocumentRecord, load_manifest, sha256_file
from representations import chunk_text, encode_representation


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def read_text(row: DocumentRecord) -> str:
    return row.path.read_text(encoding=row.encoding)


def truncate_units(text: str, representation: str, max_units: int) -> str:
    if max_units <= 0:
        return text
    if representation in {"token_recurrence_u32", "token_length_u32"}:
        tokens = text.split()
        return " ".join(tokens[-max(1, max_units // 2):])
    if representation == "codepoint_u32_nospace":
        compact = "".join(ch for ch in text if not ch.isspace())
        return compact[-max_units:]
    return text[-max_units:]


def build_reference(rows: list[DocumentRecord], representation: str, reference_units: int) -> tuple[bytes, list[str]]:
    ordered = sorted(rows, key=lambda r: (r.corpus_id, r.document_id))
    text = "\n\n<CTD_DOCUMENT_BOUNDARY>\n\n".join(read_text(row) for row in ordered)
    text = truncate_units(text, representation, reference_units)
    return encode_representation(text, representation), [r.document_id for r in ordered]


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = (config_path.parent / manifest_path).resolve()
    rows = load_manifest(manifest_path, verify_hashes=bool(config.get("verify_hashes", True)))

    selected_compressors = list(config["compressors"])
    available = available_compressors()
    missing = [name for name in selected_compressors if not available.get(name, False)]
    if missing and config.get("require_all_compressors", True):
        raise RuntimeError(f"required compressors unavailable: {missing}; availability={available}")
    compressors = [name for name in selected_compressors if available.get(name, False)]
    if not compressors:
        raise RuntimeError("no selected compressor is available")

    representations = list(config["representations"])
    split = config.get("evaluation_split", "test")
    probe_units = int(config["probe_units"])
    probe_stride = int(config.get("probe_stride", probe_units))
    reference_units = int(config.get("reference_units", 0))
    max_probes_per_document = int(config.get("max_probes_per_document", 0))
    label_field = config.get("label_field", "corpus_id")
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    by_corpus_train: dict[str, list[DocumentRecord]] = defaultdict(list)
    eval_rows: list[DocumentRecord] = []
    for row in rows:
        if row.split == "train":
            by_corpus_train[row.corpus_id].append(row)
        elif row.split == split:
            eval_rows.append(row)
    corpus_ids = sorted(by_corpus_train)
    if not eval_rows:
        raise ValueError(f"no rows in evaluation split {split!r}")

    observation_rows: list[dict[str, Any]] = []
    ncd_rows: list[dict[str, Any]] = []
    reference_cache: dict[tuple[str, str], tuple[bytes, list[str]]] = {}

    for representation in representations:
        for corpus_id in corpus_ids:
            reference_cache[(representation, corpus_id)] = build_reference(by_corpus_train[corpus_id], representation, reference_units)

        for compressor in compressors:
            for i, a in enumerate(corpus_ids):
                for b in corpus_ids[i:]:
                    a_ref, _ = reference_cache[(representation, a)]
                    b_ref, _ = reference_cache[(representation, b)]
                    ncd_f, ncd_r, ncd_s = normalized_compression_distance(a_ref, b_ref, compressor)
                    ncd_rows.append({
                        "representation": representation,
                        "compressor": compressor,
                        "corpus_a": a,
                        "corpus_b": b,
                        "ncd_forward": ncd_f,
                        "ncd_reverse": ncd_r,
                        "ncd_symmetric": ncd_s,
                        "concat_order_gap": abs(ncd_f - ncd_r),
                    })

        for doc in sorted(eval_rows, key=lambda r: (r.corpus_id, r.document_id)):
            if doc.corpus_id not in by_corpus_train:
                raise ValueError(f"evaluation corpus {doc.corpus_id} lacks train reference")
            chunks = chunk_text(read_text(doc), representation, probe_units, probe_stride)
            if max_probes_per_document > 0:
                chunks = chunks[:max_probes_per_document]
            for probe_index, chunk in enumerate(chunks):
                probe = encode_representation(chunk, representation)
                if not probe:
                    continue
                own_ref, own_docs = reference_cache[(representation, doc.corpus_id)]
                for compressor in compressors:
                    own_cost = conditional_bits_per_byte(own_ref, probe, compressor)
                    for candidate_id in corpus_ids:
                        candidate_ref, candidate_docs = reference_cache[(representation, candidate_id)]
                        cost = conditional_bits_per_byte(candidate_ref, probe, compressor)
                        excess = directional_excess_bits_per_byte(candidate_ref, own_ref, probe, compressor)
                        observation_rows.append({
                            "representation": representation,
                            "compressor": compressor,
                            "target_corpus": doc.corpus_id,
                            "target_class": getattr(doc, label_field),
                            "target_document": doc.document_id,
                            "probe_index": probe_index,
                            "probe_sha256": sha256_bytes(probe),
                            "probe_bytes": len(probe),
                            "candidate_corpus": candidate_id,
                            "candidate_reference_documents": "|".join(candidate_docs),
                            "own_reference_documents": "|".join(own_docs),
                            "candidate_conditional_bits_per_byte": cost,
                            "own_conditional_bits_per_byte": own_cost,
                            "directional_excess_bits_per_byte": excess,
                            "candidate_reference_bytes": len(candidate_ref),
                            "own_reference_bytes": len(own_ref),
                            "c_candidate_reference": compressed_size(candidate_ref, compressor),
                            "c_probe": compressed_size(probe, compressor),
                        })

    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in observation_rows:
        key = (row["representation"], row["compressor"], row["target_document"], row["probe_index"])
        grouped[key].append(row)
    for group in grouped.values():
        ordered = sorted(group, key=lambda r: (r["candidate_conditional_bits_per_byte"], r["candidate_corpus"]))
        own_rank = next(i for i, candidate in enumerate(ordered, start=1) if candidate["candidate_corpus"] == candidate["target_corpus"])
        margin = ordered[1]["candidate_conditional_bits_per_byte"] - ordered[0]["candidate_conditional_bits_per_byte"] if len(ordered) > 1 else float("nan")
        for rank, row in enumerate(ordered, start=1):
            row["candidate_rank"] = rank
            row["predicted_corpus"] = ordered[0]["candidate_corpus"]
            row["correct_top1"] = int(ordered[0]["candidate_corpus"] == row["target_corpus"])
            row["own_source_rank"] = own_rank
            row["winner_margin_bits_per_byte"] = margin

    obs_path = output_dir / "directional_observations.csv"
    if observation_rows:
        with obs_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(observation_rows[0].keys()))
            writer.writeheader()
            writer.writerows(observation_rows)

    ncd_path = output_dir / "ncd_pairs.csv"
    if ncd_rows:
        with ncd_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(ncd_rows[0].keys()))
            writer.writeheader()
            writer.writerows(ncd_rows)

    probe_rows = [min(group, key=lambda r: (r["candidate_conditional_bits_per_byte"], r["candidate_corpus"])) for group in grouped.values()]
    summary_cells = []
    for representation in representations:
        for compressor in compressors:
            subset = [r for r in probe_rows if r["representation"] == representation and r["compressor"] == compressor]
            if not subset:
                continue
            correct = [r["predicted_corpus"] == r["target_corpus"] for r in subset]
            ranks = [int(r["own_source_rank"]) for r in subset]
            margins = [float(r["winner_margin_bits_per_byte"]) for r in subset if math.isfinite(float(r["winner_margin_bits_per_byte"]))]
            summary_cells.append({
                "representation": representation,
                "compressor": compressor,
                "n_probes": len(subset),
                "top1_accuracy": sum(correct) / len(correct),
                "median_own_source_rank": statistics.median(ranks),
                "mean_own_source_rank": statistics.mean(ranks),
                "median_winner_margin_bits_per_byte": statistics.median(margins) if margins else float("nan"),
            })

    summary = {
        "programme": config.get("programme", "compression-transfer-v0.1"),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "manifest_sha256": sha256_file(manifest_path),
        "boundary_hex": BOUNDARY.hex(),
        "compressor_availability": available,
        "compressor_specs": [asdict(compressor_spec(name)) for name in compressors],
        "representations": representations,
        "probe_units": probe_units,
        "probe_stride": probe_stride,
        "reference_units": reference_units,
        "evaluation_split": split,
        "n_manifest_documents": len(rows),
        "n_observation_rows": len(observation_rows),
        "n_probe_compressor_cells": len(probe_rows),
        "summary_cells": summary_cells,
        "outputs": {
            "directional_observations.csv": sha256_file(obs_path) if obs_path.exists() else None,
            "ncd_pairs.csv": sha256_file(ncd_path) if ncd_path.exists() else None,
        },
    }
    summary["scientific_payload_sha256"] = hashlib.sha256(canonical_json_bytes(summary)).hexdigest()
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
