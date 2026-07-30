#!/usr/bin/env python3
"""Independent arithmetic validation of compression-transfer outputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((result_dir / "directional_observations.csv").open(newline="", encoding="utf-8")))
    groups = defaultdict(list)
    checks: list[bool] = []
    for row in rows:
        key = (row["representation"], row["compressor"], row["target_document"], int(row["probe_index"]))
        groups[key].append(row)
        expected = float(row["candidate_conditional_bits_per_byte"]) - float(row["own_conditional_bits_per_byte"])
        checks.append(abs(expected - float(row["directional_excess_bits_per_byte"])) < 1e-9)
    probe_rows = []
    for group in groups.values():
        ordered = sorted(group, key=lambda r: (float(r["candidate_conditional_bits_per_byte"]), r["candidate_corpus"]))
        winner = ordered[0]
        probe_rows.append(winner)
        checks.append(winner["predicted_corpus"] == winner["candidate_corpus"])
        own_rank = next(i for i, row in enumerate(ordered, start=1) if row["candidate_corpus"] == row["target_corpus"])
        checks.append(int(winner["own_source_rank"]) == own_rank)
    checks.append(summary["outputs"]["directional_observations.csv"] == sha256_file(result_dir / "directional_observations.csv"))
    checks.append(summary["outputs"]["ncd_pairs.csv"] == sha256_file(result_dir / "ncd_pairs.csv"))
    verdict = {
        "checks": len(checks),
        "passed": sum(checks),
        "failed": len(checks) - sum(checks),
        "n_observation_rows": len(rows),
        "n_probe_cells": len(probe_rows),
        "verdict": "PASS" if all(checks) else "FAIL",
    }
    (result_dir / "validation.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    if not all(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
