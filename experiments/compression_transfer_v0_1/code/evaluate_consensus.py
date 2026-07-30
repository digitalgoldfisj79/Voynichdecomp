#!/usr/bin/env python3
"""Evaluate preregistered multi-compressor/representation consensus."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.6448536269514722) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("observations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-votes", type=int, default=2)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.observations.open(newline="", encoding="utf-8")))
    groups = defaultdict(list)
    for row in rows:
        key = (row["representation"], row["compressor"], row["target_document"], row["probe_index"])
        groups[key].append(row)

    votes_by_probe = defaultdict(list)
    target_by_probe = {}
    for key, group in groups.items():
        ordered = sorted(group, key=lambda r: (float(r["candidate_conditional_bits_per_byte"]), r["candidate_corpus"]))
        winner = ordered[0]["candidate_corpus"]
        probe_key = (key[2], key[3])
        votes_by_probe[probe_key].append((key[0], key[1], winner))
        target_by_probe[probe_key] = ordered[0]["target_corpus"]

    consensus_rows = []
    for probe_key, votes in sorted(votes_by_probe.items()):
        counts = Counter(v[2] for v in votes)
        winner, n_votes = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        target = target_by_probe[probe_key]
        tied = sum(count == n_votes for count in counts.values()) > 1
        accepted = (n_votes >= args.required_votes) and not tied
        consensus_rows.append({
            "target_document": probe_key[0],
            "probe_index": probe_key[1],
            "target_corpus": target,
            "consensus_winner": winner,
            "winner_votes": n_votes,
            "total_votes": len(votes),
            "accepted": accepted,
            "correct_if_accepted": accepted and winner == target,
            "vote_detail": votes,
        })

    accepted = [row for row in consensus_rows if row["accepted"]]
    correct = [row for row in accepted if row["correct_if_accepted"]]
    coverage = len(accepted) / len(consensus_rows) if consensus_rows else float("nan")
    accuracy = len(correct) / len(accepted) if accepted else float("nan")
    recalls = {}
    for target in sorted(set(target_by_probe.values())):
        target_rows = [row for row in consensus_rows if row["target_corpus"] == target]
        target_correct = [row for row in target_rows if row["accepted"] and row["correct_if_accepted"]]
        recalls[target] = len(target_correct) / len(target_rows) if target_rows else float("nan")

    result = {
        "required_votes": args.required_votes,
        "n_probe_units": len(consensus_rows),
        "accepted": len(accepted),
        "correct_accepted": len(correct),
        "consensus_coverage": coverage,
        "consensus_coverage_wilson90": wilson(len(accepted), len(consensus_rows)) if consensus_rows else (float("nan"), float("nan")),
        "consensus_accuracy_conditional_on_acceptance": accuracy,
        "consensus_accuracy_wilson90": wilson(len(correct), len(accepted)) if accepted else (float("nan"), float("nan")),
        "worst_target_recall": min(recalls.values()) if recalls else float("nan"),
        "target_recalls": recalls,
        "rows": consensus_rows,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
