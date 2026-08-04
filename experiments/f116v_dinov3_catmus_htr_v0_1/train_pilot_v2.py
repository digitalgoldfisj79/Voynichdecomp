#!/usr/bin/env python3
"""Corrected deterministic manuscript-disjoint sampler for train_pilot.py.

The original SHA bucket rule was valid in principle but the local streaming
window contained no eligible shelfmark in its 10% test bucket. This wrapper
uses deterministic first-encounter assignment in a fixed train/dev/train/test
cycle. A shelfmark is permanently assigned to one split, so leakage remains
impossible while all pilot partitions can be populated from locally clustered
streaming shards.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import train_pilot as base


def collect_samples_balanced(
    train_n: int,
    dev_n: int,
    test_n: int,
    seed: int,
    max_chars: int,
    max_scan: int,
):
    from datasets import load_dataset

    quotas = {"train": train_n, "dev": dev_n, "test": test_n}
    # A pilot may use one held-out manuscript if the local stream is highly
    # clustered, but no manuscript can occur in two splits.
    per_shelf_cap = {"train": 64, "dev": 64, "test": 64}
    samples = {k: [] for k in quotas}
    shelf_counts = {k: Counter() for k in quotas}
    rejected = Counter()
    assignment: dict[str, str] = {}
    cycle = ("train", "dev", "train", "test")

    stream = load_dataset(base.DATA_REPO, split="train", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=500)

    scanned = 0
    for row in stream:
        scanned += 1
        if scanned > max_scan:
            break
        if all(len(samples[k]) >= quotas[k] for k in quotas):
            break
        text = base.normalize_text(row.get("text", ""))
        if row.get("line_type") != "DefaultLine":
            rejected["line_type"] += 1
            continue
        if row.get("century") not in (14, 15, 16):
            rejected["century"] += 1
            continue
        if not (8 <= len(text) <= max_chars):
            rejected["length"] += 1
            continue
        shelfmark = str(row.get("shelfmark") or "UNKNOWN")
        if shelfmark not in assignment:
            preferred = cycle[len(assignment) % len(cycle)]
            if len(samples[preferred]) >= quotas[preferred]:
                preferred = max(
                    quotas,
                    key=lambda k: (quotas[k] - len(samples[k])) / max(1, quotas[k]),
                )
            assignment[shelfmark] = preferred
        split = assignment[shelfmark]
        if len(samples[split]) >= quotas[split]:
            rejected["quota"] += 1
            continue
        if shelf_counts[split][shelfmark] >= per_shelf_cap[split]:
            rejected["shelf_cap"] += 1
            continue
        try:
            image, valid_steps = base.prepare_line(row["im"])
        except Exception:
            rejected["decode"] += 1
            continue
        if base.ctc_required_steps(text) > valid_steps:
            rejected["ctc_infeasible"] += 1
            continue
        samples[split].append(
            base.Sample(
                split=split,
                shelfmark=shelfmark,
                text=text,
                image=image,
                valid_steps=valid_steps,
                century=int(row["century"]),
                script_type=str(row.get("script_type") or ""),
                language=str(row.get("language") or ""),
            )
        )
        shelf_counts[split][shelfmark] += 1
        if sum(len(v) for v in samples.values()) % 64 == 0:
            print(
                "ACQUIRE",
                scanned,
                {k: len(v) for k, v in samples.items()},
                {k: len(shelf_counts[k]) for k in shelf_counts},
                flush=True,
            )

    short = {k: quotas[k] - len(samples[k]) for k in quotas if len(samples[k]) < quotas[k]}
    if short:
        raise RuntimeError(f"Could not fill corrected shelfmark-disjoint quotas after {scanned} rows: {short}")

    shelves = {k: {s.shelfmark for s in v} for k, v in samples.items()}
    assert shelves["train"].isdisjoint(shelves["dev"])
    assert shelves["train"].isdisjoint(shelves["test"])
    assert shelves["dev"].isdisjoint(shelves["test"])

    manifest = {
        "split_rule": "deterministic first-encounter train/dev/train/test cycle",
        "scanned_rows": scanned,
        "counts": {k: len(v) for k, v in samples.items()},
        "shelfmark_counts": {k: len(shelves[k]) for k in shelves},
        "shelfmarks": {k: sorted(shelves[k]) for k in shelves},
        "rejected": dict(rejected),
        "centuries": {k: dict(Counter(s.century for s in v)) for k, v in samples.items()},
        "scripts": {k: dict(Counter(s.script_type for s in v)) for k, v in samples.items()},
        "languages": {k: dict(Counter(s.language for s in v)) for k, v in samples.items()},
    }
    return samples, manifest


base.collect_samples = collect_samples_balanced

if __name__ == "__main__":
    base.main()
