#!/usr/bin/env python3
"""Post-hoc finite-sample audit of the historical five-axis PGCS information budget.

This reproduces the historical plug-in mutual-information calculation and estimates
its high-cardinality finite-sample component by globally permuting the four-slot
outcomes while keeping the observed context cells fixed.

The chance-subtracted difference is diagnostic only. It is not a cross-validated
estimate of explained entropy and is not used as a replacement headline.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Hashable, Iterable

import numpy as np


def position_category(record: dict) -> str:
    if bool(record["is_first_word"]):
        return "FIRST"
    if bool(record["is_last_word"]):
        return "LAST"
    return "MID"


def folio_to_quire(folio: str) -> str:
    raw = folio.replace("f", "").replace("r", "").replace("v", "")
    try:
        number = int(raw.split(".")[0])
    except ValueError:
        return "UNK"
    cutoffs = [
        (8, "Q1"), (16, "Q2"), (22, "Q3"), (32, "Q4"),
        (38, "Q5"), (42, "Q6"), (50, "Q7"), (58, "Q8"),
        (66, "Q9-12"), (73, "Q13"), (84, "Q14"), (86, "Q15"),
        (90, "Q16-17"), (96, "Q18"), (103, "Q19"), (116, "Q20"),
    ]
    for cutoff, label in cutoffs:
        if number <= cutoff:
            return label
    return "UNK"


def integer_codes(values: Iterable[Hashable]) -> tuple[np.ndarray, int]:
    mapping: dict[Hashable, int] = {}
    values = list(values)
    output = np.empty(len(values), dtype=np.int64)
    for index, value in enumerate(values):
        if value not in mapping:
            mapping[value] = len(mapping)
        output[index] = mapping[value]
    return output, len(mapping)


def entropy(codes: np.ndarray, categories: int) -> float:
    counts = np.bincount(codes, minlength=categories)
    probabilities = counts[counts > 0] / len(codes)
    return float(-(probabilities * np.log2(probabilities)).sum())


def mutual_information(x: np.ndarray, x_categories: int, y: np.ndarray, y_categories: int) -> float:
    joint = x * y_categories + y
    return entropy(x, x_categories) + entropy(y, y_categories) - entropy(joint, x_categories * y_categories)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path, help="Committed enriched_records.pkl")
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.records.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a non-empty list of token records")

    lines: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        lines[(record["folio"], record["line_no"])].append(record)
    for tokens in lines.values():
        tokens.sort(key=lambda record: int(record["pos"]))
        for index, record in enumerate(tokens):
            record["_prev_sfx"] = "LINE_START" if index == 0 else tokens[index - 1]["sfx_fam"]
            record["_para_flag"] = bool(record["is_first_line"] and index == 0)

    quad, n_quad = integer_codes((r["prefix"], r["gallows"], r["m_core"], r["sfx_fam"]) for r in records)
    context, n_context = integer_codes((r["section"], position_category(r), r["_prev_sfx"], r["_para_flag"], folio_to_quire(r["folio"])) for r in records)

    h_quad = entropy(quad, n_quad)
    observed = mutual_information(quad, n_quad, context, n_context)
    rng = np.random.default_rng(args.seed)
    null = np.array([mutual_information(rng.permutation(quad), n_quad, context, n_context) for _ in range(args.permutations)], dtype=float)
    result = {
        "status": "historical arithmetic reproduced; interpretation withdrawn",
        "n_tokens": len(records),
        "quad_categories": n_quad,
        "context_cells": n_context,
        "h_quad_bits": h_quad,
        "observed_plugin_mi_bits": observed,
        "observed_percent": 100 * observed / h_quad,
        "permutation_count": args.permutations,
        "permutation_seed": args.seed,
        "null_mean_bits": float(null.mean()),
        "null_95th_percentile_bits": float(np.quantile(null, 0.95)),
        "observed_minus_null_mean_bits": float(observed - null.mean()),
        "observed_minus_null_percent_hquad": float(100 * (observed - null.mean()) / h_quad),
        "interpretation": "The plug-in estimate is strongly inflated by high-cardinality finite-sample bias. The chance-subtracted value is a post-hoc diagnostic, not a cross-validated replacement estimate."
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
