#!/usr/bin/env python3
"""Build deterministic UPGMA trees from a symmetric distance matrix."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_pairs(path: Path, representation: str, compressor: str) -> tuple[list[str], dict[tuple[str, str], float]]:
    labels: set[str] = set()
    distances: dict[tuple[str, str], float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["representation"] != representation or row["compressor"] != compressor:
                continue
            a, b = row["corpus_a"], row["corpus_b"]
            labels.update((a, b))
            value = float(row["ncd_symmetric"])
            distances[(a, b)] = value
            distances[(b, a)] = value
    for label in labels:
        distances[(label, label)] = 0.0
    return sorted(labels), distances


def upgma(labels: list[str], distances: dict[tuple[str, str], float]) -> str:
    clusters: dict[str, dict[str, object]] = {
        label: {"members": [label], "height": 0.0, "newick": label.replace(" ", "_")} for label in labels
    }
    while len(clusters) > 1:
        keys = sorted(clusters)
        best = None
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                ma = clusters[a]["members"]
                mb = clusters[b]["members"]
                assert isinstance(ma, list) and isinstance(mb, list)
                vals = [distances[(x, y)] for x in ma for y in mb]
                candidate = (sum(vals) / len(vals), a, b)
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        distance, a, b = best
        ca, cb = clusters.pop(a), clusters.pop(b)
        height = distance / 2.0
        ha, hb = float(ca["height"]), float(cb["height"])
        na, nb = str(ca["newick"]), str(cb["newick"])
        newick = f"({na}:{max(0.0, height-ha):.8f},{nb}:{max(0.0, height-hb):.8f})"
        members = sorted(list(ca["members"]) + list(cb["members"]))  # type: ignore[arg-type]
        clusters["+".join(members)] = {"members": members, "height": height, "newick": newick}
    return str(next(iter(clusters.values()))["newick"]) + ";"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", type=Path)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--compressor", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels, distances = load_pairs(args.pairs, args.representation, args.compressor)
    if len(labels) < 2:
        raise SystemExit("fewer than two labels after filtering")
    tree = upgma(labels, distances)
    args.output.write_text(tree + "\n", encoding="utf-8")
    print(json.dumps({"labels": labels, "newick": tree}, indent=2))


if __name__ == "__main__":
    main()
