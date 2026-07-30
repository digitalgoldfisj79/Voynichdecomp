#!/usr/bin/env python3
"""Assign one benchmark corpus_id per source class and verify split coverage."""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


def slug(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z]+", "_", value.strip()).strip("_").lower()
    if not value:
        raise ValueError("empty class label")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--expected-classes", nargs="*", default=[])
    args = parser.parse_args()

    path = Path(args.manifest)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    required = {"corpus_id", "class_label", "split"}
    if not required.issubset(fields):
        raise ValueError(f"manifest missing fields: {sorted(required - set(fields))}")

    counts: Counter[str] = Counter()
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        label = slug(row["class_label"])
        row["corpus_id"] = f"{slug(args.prefix)}_{label}"
        counts[label] += 1
        split_counts[label][row["split"]] += 1

    expected = {slug(x) for x in args.expected_classes}
    missing = sorted(expected - set(counts))
    unexpected = sorted(set(counts) - expected) if expected else []
    if missing:
        raise ValueError(f"missing classes: {missing}")
    if unexpected:
        raise ValueError(f"unexpected classes: {unexpected}")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"NAMESPACED {len(rows)} rows across {len(counts)} classes")
    for label in sorted(counts):
        print(label, counts[label], dict(sorted(split_counts[label].items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
